"""CPU-only contract tests for the direct-MVS Scene initialization path."""

import ast
from argparse import ArgumentParser
from collections import namedtuple
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from arguments import ModelParams, OptimizationParams
from utils.mvs_init import resolve_mvs_init_strategy


REPO_ROOT = Path(__file__).resolve().parents[1]
SceneInfo = namedtuple(
    "SceneInfo", "point_cloud train_cameras test_cameras nerf_normalization ply_path"
)


def _cloud(offset, count):
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + offset
    return SimpleNamespace(
        points=points,
        colors=np.full((count, 3), 0.25, dtype=np.float32),
        normals=np.zeros_like(points),
    )


class DirectMVSSceneTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="direct-mvs-scene-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "sparse").mkdir()
        self.output = self.root / "model"
        self.output.mkdir()
        self.base = _cloud(0.0, 4)
        self.mvs = _cloud(100.0, 7)
        self.train_cameras = [
            SimpleNamespace(image_name="train_a"),
            SimpleNamespace(image_name="train_b"),
            SimpleNamespace(image_name="train_c"),
        ]
        self.test_cameras = [SimpleNamespace(image_name="held_out")]
        self.scene_info = SceneInfo(
            self.base, self.train_cameras, self.test_cameras,
            {"radius": 2.0}, str(self.source / "baseline.ply"),
        )
        self.args = SimpleNamespace(
            source_path=str(self.source), model_path=str(self.output),
            images="images", eval=True, n_views=3, rand_pcd=False,
            white_background=False, dataset_name="llff",
            init_mode="mvs_anywhere", mvs_init_strategy="replace",
            mvs_init_path=str(self.root / "cache"), mvs_init_max_points=123,
        )
        self.gaussians = mock.Mock()
        self.load_mvs = mock.Mock(return_value=(
            self.mvs, self.root / "cache" / "mvs_input.ply",
            {"camera_provenance_verified": True,
             "selected_point_count": 7, "selected_sha256": "mvs-digest"},
        ))
        self.store_ply = mock.Mock()
        self.colmap_reader = mock.Mock(return_value=self.scene_info)
        self.blender_reader = mock.Mock(return_value=self.scene_info)
        # Execute the real Scene class, but substitute its IO/CUDA dependencies.
        # The body under test is not reimplemented or copied into this fixture.
        module = ast.parse((REPO_ROOT / "scene" / "__init__.py").read_text())
        scene_node = next(node for node in module.body
                          if isinstance(node, ast.ClassDef) and node.name == "Scene")
        namespace = {
            "os": os, "random": random, "json": json, "hashlib": hashlib,
            "np": np, "GaussianModel": object, "ModelParams": object,
            "searchForMaxIteration": mock.Mock(return_value=10000),
            "sceneLoadTypeCallbacks": {
                "Colmap": self.colmap_reader, "Blender": self.blender_reader,
            },
            "load_mvs_point_cloud": self.load_mvs,
            "resolve_mvs_init_strategy": resolve_mvs_init_strategy,
            "storePly": self.store_ply,
            "camera_to_JSON": lambda index, camera: {
                "id": index, "image_name": camera.image_name,
            },
            "cameraList_from_camInfos": lambda cameras, scale, args: list(cameras),
        }
        exec(compile(ast.Module(body=[scene_node], type_ignores=[]),
                     str(REPO_ROOT / "scene" / "__init__.py"), "exec"), namespace)
        self.Scene = namespace["Scene"]

    def make_scene(self, **kwargs):
        with mock.patch("builtins.print"):
            return self.Scene(self.args, self.gaussians, shuffle=False, **kwargs)

    def test_mvs_replaces_baseline_and_uses_ordinary_gaussian_initializer(self):
        scene = self.make_scene()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        self.assertEqual(self.store_ply.call_count, 1)
        path, points, colors = self.store_ply.call_args.args
        self.assertEqual(path, str(self.output / "input.ply"))
        np.testing.assert_array_equal(points, self.mvs.points)
        np.testing.assert_array_equal(colors, self.mvs.colors * 255.0)
        self.assertEqual(scene.initialization_metadata["mode"], "mvs_anywhere")
        self.assertEqual(scene.initialization_metadata["resolved_strategy"], "replace")
        self.assertEqual(scene.initialization_metadata["mvs_point_count"], 7)
        self.assertEqual(scene.initialization_metadata["fused_point_count"], 7)
        self.assertFalse(any("reliability" in name
                             for name in scene.initialization_metadata))

    def test_missing_legacy_namespace_fields_default_to_direct_mvs(self):
        del self.args.init_mode
        del self.args.mvs_init_strategy
        scene = self.make_scene()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        self.assertEqual(scene.initialization_metadata["mode"], "mvs_anywhere")
        self.assertEqual(scene.initialization_metadata["resolved_strategy"], "replace")

    def test_only_training_view_names_and_explicit_budget_reach_cache_loader(self):
        scene = self.make_scene()
        self.load_mvs.assert_called_once_with(
            self.args.source_path, n_views=3,
            train_camera_names=["train_a", "train_b", "train_c"],
            explicit_path=self.args.mvs_init_path, max_points=123,
        )
        self.assertEqual(scene.train_camera_names, ["train_a", "train_b", "train_c"])
        self.assertEqual(scene.test_camera_names, ["held_out"])
        self.assertEqual(scene.getTrainCameras(), self.train_cameras)
        self.assertEqual(scene.getTestCameras(), self.test_cameras)

    def test_missing_mvs_cache_fails_without_baseline_fallback(self):
        self.load_mvs.side_effect = FileNotFoundError("missing direct MVS cache")
        with self.assertRaisesRegex(FileNotFoundError, "missing direct MVS cache"):
            self.make_scene()
        self.gaussians.create_from_pcd.assert_not_called()
        self.store_ply.assert_not_called()
        self.assertFalse((self.output / "cameras.json").exists())

    def test_invalid_mvs_cache_fails_without_baseline_fallback(self):
        self.load_mvs.side_effect = ValueError("training split mismatch")
        with self.assertRaisesRegex(ValueError, "training split mismatch"):
            self.make_scene()
        self.gaussians.create_from_pcd.assert_not_called()
        self.store_ply.assert_not_called()

    def test_adaptive_strategy_is_rejected_before_cache_load(self):
        self.args.mvs_init_strategy = "adaptive"
        with self.assertRaises(ValueError):
            self.make_scene()
        self.load_mvs.assert_not_called()
        self.gaussians.create_from_pcd.assert_not_called()

    def test_baseline_initialization_remains_explicit_ablation(self):
        self.args.init_mode = "colmap_dense"
        self.args.rand_pcd = True
        scene = self.make_scene()
        self.load_mvs.assert_not_called()
        self.gaussians.create_from_pcd.assert_called_once_with(self.base, 2.0)
        self.assertEqual(scene.initialization_metadata["mode"], "colmap_dense")

    def test_blender_uses_the_same_direct_mvs_replacement(self):
        (self.source / "transforms_train.json").write_text("{}")
        self.args.dataset_name = "blender"
        self.args.rand_pcd = True
        self.args.white_background = True
        self.make_scene()
        self.blender_reader.assert_called_once_with(
            self.args.source_path, True, True, True, 3, "blender",
        )
        self.colmap_reader.assert_not_called()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        self.load_mvs.assert_called_once()

    def test_loading_trained_ply_skips_initialization_and_cache_access(self):
        self.load_mvs.side_effect = AssertionError("rendering must not need MVS cache")
        # A historical cfg_args strategy describes the old initialization;
        # rendering its trained PLY must not reactivate or require that module.
        self.args.mvs_init_strategy = "adaptive"
        scene = self.make_scene(load_iteration=10000)
        self.load_mvs.assert_not_called()
        self.store_ply.assert_not_called()
        self.gaussians.create_from_pcd.assert_not_called()
        self.gaussians.load_ply.assert_called_once_with(str(
            self.output / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        ))
        self.assertEqual(scene.loaded_iter, 10000)


class DirectMVSBranchDefaultsTest(unittest.TestCase):
    def test_cli_defaults_enable_direct_mvs_and_both_density_modules(self):
        parser = ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.init_mode, "mvs_anywhere")
        self.assertEqual(args.mvs_init_strategy, "replace")
        self.assertEqual(args.density_control, "vcd_vcp")
        for removed in ("mvs_init_total_max_points", "mvs_init_dense_base_threshold",
                        "mvs_init_dense_residual_points"):
            self.assertFalse(hasattr(args, removed))


if __name__ == "__main__":
    unittest.main()
