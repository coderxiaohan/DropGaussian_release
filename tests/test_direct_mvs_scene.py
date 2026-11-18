"""CPU-only contract tests for the direct-MVS Scene initialization path."""

import ast
from argparse import ArgumentParser
from collections import namedtuple
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from arguments import ModelParams, OptimizationParams
from utils.mvs_init import resolve_mvs_init_strategy


REPO_ROOT = Path(__file__).resolve().parents[1]
SceneInfo = namedtuple(
    "SceneInfo", "point_cloud train_cameras test_cameras nerf_normalization ply_path point_cloud_source",
    defaults=["unknown"],
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
        self.args.mvs_geometry_correction = "tps"
        self.args.mvs_surface_init = "oas"
        scene = self.make_scene(load_iteration=10000)
        self.load_mvs.assert_not_called()
        self.store_ply.assert_not_called()
        self.gaussians.create_from_pcd.assert_not_called()
        self.gaussians.load_ply.assert_called_once_with(str(
            self.output / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        ))
        self.assertEqual(scene.loaded_iter, 10000)

    def test_removed_initializers_reject_new_training_before_cache_access(self):
        for init_mode in ("mvs_anywhere", "colmap_dense"):
            for name, value in (
                ("mvs_geometry_correction", "similarity"),
                ("mvs_geometry_correction", "tps"),
                ("mvs_geometry_correction", "tps_consistent"),
                ("mvs_geometry_correction", "tps_local"),
                ("mvs_surface_init", "oas"),
            ):
                with self.subTest(init_mode=init_mode, option=name, value=value), \
                        mock.patch.object(self.args, "init_mode", init_mode), \
                        mock.patch.object(self.args, name, value, create=True):
                    with self.assertRaisesRegex(ValueError, f"{name} was removed"):
                        self.make_scene()
        self.load_mvs.assert_not_called()
        self.gaussians.create_from_pcd.assert_not_called()
        self.store_ply.assert_not_called()

    def test_historical_disabled_initializer_flags_keep_direct_behavior(self):
        self.args.mvs_geometry_correction = "none"
        self.args.mvs_surface_init = "none"
        self.make_scene()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        for artifact in ("geometry_correction.json", "geometry_controls.npz",
                         "surface_initialization.json", "surface_initialization.npz"):
            self.assertFalse((self.output / artifact).exists())

    def enable_complement(self, mode="coverage", source="colmap_dense"):
        self.args.mvs_complement_mode = mode
        self.args.mvs_complement_max_points = 5
        self.args.mvs_complement_candidate_points = 20
        self.args.mvs_complement_total_points = 0
        self.scene_info = self.scene_info._replace(point_cloud_source=source)
        self.colmap_reader.return_value = self.scene_info
        self.blender_reader.return_value = self.scene_info

    def test_complement_is_opt_in_and_default_skips_helper(self):
        with mock.patch("utils.mvs_complement.build_complement_cloud") as build:
            self.make_scene()
        build.assert_not_called()
        self.assertFalse((self.output / "complement_initialization.json").exists())

    def test_complement_receives_base_training_cameras_and_larger_candidate_pool(self):
        self.enable_complement()
        output = _cloud(80, 9)
        with mock.patch("utils.mvs_complement.build_complement_cloud", return_value=(
                output, {"selected_base_count": 4, "selected_mvs_count": 5},
                {"selected_base_indices": np.arange(4)})) as build:
            scene = self.make_scene()
        self.assertIs(build.call_args.args[0], self.base)
        self.assertIs(build.call_args.args[1], self.mvs)
        self.assertIs(build.call_args.args[2], self.train_cameras)
        self.assertEqual(build.call_args.kwargs["mvs_budget"], 5)
        self.assertEqual(build.call_args.kwargs["total_budget"], 0)
        self.assertTrue(build.call_args.kwargs["base_is_geometry"])
        self.assertIsNone(build.call_args.kwargs["direct_pcd"])
        self.assertEqual(self.load_mvs.call_args.kwargs["max_points"], 20)
        self.gaussians.create_from_pcd.assert_called_once_with(output, 2.0)
        info = json.loads((self.output / "complement_initialization.json").read_text())
        self.assertEqual(info["selection_camera_names"], ["train_a", "train_b", "train_c"])
        self.assertIn("not_train_view_verified", info["base_provenance"])
        self.assertEqual(info["base_point_cloud_source"], "colmap_dense")
        self.assertEqual(scene.initialization_metadata["fused_point_count"], 9)
        self.assertEqual(scene.initialization_metadata["mvs_point_count"], 5)
        self.assertEqual(scene.initialization_metadata["selection"], "base_plus_coverage_mvs")
        self.assertEqual(scene.initialization_metadata["candidate_selected_sha256"], "mvs-digest")
        with np.load(self.output / "complement_initialization.npz", allow_pickle=False) as values:
            np.testing.assert_array_equal(values["selected_base_indices"], np.arange(4))

    def test_complement_calibration_repair_does_not_change_training_cameras(self):
        self.enable_complement()
        folder = self.source / "sparse" / "0"
        folder.mkdir()
        (folder / "cameras.bin").write_bytes(b"test fixture, mocked reader")
        calibrated = [SimpleNamespace(image_name=c.image_name) for c in self.train_cameras]
        intrinsics = {1: object()}
        loader = SimpleNamespace(read_intrinsics_binary=mock.Mock(return_value=intrinsics),
                                 read_intrinsics_text=mock.Mock())
        # Resolve utility modules before sys.modules is patched so mock targets
        # and the imports inside Scene always refer to the same objects.
        with mock.patch("utils.train_view_controls.geometry_cameras_with_intrinsics",
                        return_value=(calibrated, {"source": "colmap"})) as calibration, \
                mock.patch("utils.mvs_complement.build_complement_cloud", return_value=(
                    self.mvs, {"selected_base_count": 0, "selected_mvs_count": 7}, {})) as build, \
                mock.patch.dict(sys.modules, {"scene.colmap_loader": loader}):
            scene = self.make_scene()
        calibration.assert_called_once_with(self.train_cameras, intrinsics)
        self.assertIs(build.call_args.args[2], calibrated)
        self.assertIs(self.scene_info.train_cameras, self.train_cameras)
        self.assertEqual(scene.getTrainCameras(), self.train_cameras)
        self.assertEqual(scene.initialization_metadata["complement_initialization"]["calibration"],
                         {"source": "colmap"})

    def test_complement_fallback_loads_exact_direct_budget_not_candidate_pool(self):
        for source in ("random", "external", "unknown"):
            with self.subTest(source=source):
                self.enable_complement(source=source)
                self.args.mvs_complement_total_points = 3
                self.gaussians.reset_mock()
                with mock.patch("utils.mvs_complement.build_complement_cloud", return_value=(
                        self.mvs, {"selected_base_count": 0, "selected_mvs_count": 7}, {})) as build:
                    self.make_scene()
                self.assertFalse(build.call_args.kwargs["base_is_geometry"])
                self.assertIs(build.call_args.kwargs["direct_pcd"], self.mvs)
                self.assertEqual(self.load_mvs.call_args.kwargs["max_points"], 3)
                self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)

    def test_complement_concat_uses_direct_sized_mvs_batch(self):
        self.enable_complement(mode="concat")
        with mock.patch("utils.mvs_complement.build_complement_cloud", return_value=(
                self.mvs, {"selected_base_count": 0, "selected_mvs_count": 7}, {})):
            self.make_scene()
        self.assertEqual(self.load_mvs.call_args.kwargs["max_points"], 5)

    def test_complement_rejects_mixed_modules_invalid_mode_and_budgets(self):
        self.enable_complement()
        for name, value in (("mvs_complement_mode", "bad"),
                            ("init_mode", "colmap_dense"),
                            ("mvs_surface_init", "oas"),
                            ("mvs_geometry_correction", "tps"),
                            ("mvs_complement_max_points", 0),
                            ("mvs_complement_total_points", -1),
                            ("mvs_complement_candidate_points", False)):
            with self.subTest(name=name), mock.patch.object(self.args, name, value, create=True):
                with self.assertRaises(ValueError):
                    self.make_scene()
        self.load_mvs.assert_not_called()

    def test_complement_rejects_unverified_mvs_provenance(self):
        self.enable_complement()
        self.load_mvs.return_value[2]["camera_provenance_verified"] = False
        with mock.patch("utils.mvs_complement.build_complement_cloud") as build:
            with self.assertRaisesRegex(ValueError, "verified MVS"):
                self.make_scene()
        build.assert_not_called()
        self.gaussians.create_from_pcd.assert_not_called()

    def test_non_geometry_fallback_preserves_direct_legacy_cache_acceptance(self):
        self.enable_complement(source="random")
        self.load_mvs.return_value[2]["camera_provenance_verified"] = False
        with mock.patch("utils.mvs_complement.build_complement_cloud", return_value=(
                self.mvs, {"selected_base_count": 0, "selected_mvs_count": 7,
                           "fallback": "no_reconstruction_geometry_direct_mvs_unchanged"}, {})):
            scene = self.make_scene()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        info = scene.initialization_metadata["complement_initialization"]
        self.assertFalse(info["mvs_camera_provenance_verified"])
        self.assertEqual(scene.initialization_metadata["selection"], "unchanged_direct_mvs")

    def test_render_saved_model_needs_neither_base_provenance_nor_complement_cache(self):
        self.enable_complement()
        with mock.patch("utils.mvs_complement.build_complement_cloud") as build:
            self.make_scene(load_iteration=10000)
        build.assert_not_called()
        self.load_mvs.assert_not_called()


class DirectMVSBranchDefaultsTest(unittest.TestCase):
    def test_cli_defaults_enable_direct_mvs_and_both_density_modules(self):
        parser = ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.init_mode, "mvs_anywhere")
        self.assertEqual(args.mvs_init_strategy, "replace")
        self.assertEqual(args.density_control, "vcd_vcp")
        self.assertEqual(args.mvs_complement_mode, "none")
        for removed in ("mvs_init_total_max_points", "mvs_init_dense_base_threshold",
                        "mvs_init_dense_residual_points", "mvs_geometry_correction",
                        "mvs_geometry_max_controls", "mvs_geometry_smoothing",
                        "mvs_geometry_max_image_size", "mvs_geometry_max_features",
                        "mvs_geometry_ratio", "mvs_geometry_reprojection_px",
                        "mvs_geometry_min_angle_deg", "mvs_geometry_match_radius_px",
                        "mvs_geometry_local_neighbors", "mvs_surface_init", "mvs_surface_neighbors"):
            self.assertFalse(hasattr(args, removed))

    def test_removed_experiment_options_are_not_accepted_by_cli(self):
        parser = ArgumentParser()
        ModelParams(parser)
        for option, value in (
            ("--mvs_geometry_correction", "tps"),
            ("--mvs_geometry_correction", "none"),
            ("--mvs_geometry_max_controls", "128"),
            ("--mvs_surface_init", "oas"),
            ("--mvs_surface_neighbors", "32"),
        ):
            with self.subTest(option=option, value=value), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as rejected:
                    parser.parse_args([option, value])
                self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
