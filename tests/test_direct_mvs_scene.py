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
from plyfile import PlyData, PlyElement

from arguments import ModelParams, OptimizationParams
from utils.dataset_splits import nerf_synthetic_split, nerf_synthetic_test_split
from utils.mvs_init import resolve_mvs_init_strategy
from utils.sh_utils import SH2RGB


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

    def synthetic_reader_with_real_random_cloud(self):
        # Execute the production reader and PLY conversion without importing
        # GaussianModel's CUDA extensions. Only camera/image IO is substituted.
        reader_path = REPO_ROOT / "scene" / "dataset_readers.py"
        module = ast.parse(reader_path.read_text())
        names = {"readNerfSyntheticInfo", "pointCloudFromPlyValues", "storePly", "fetchPly"}
        functions = [node for node in module.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        cameras = [SimpleNamespace(image_name=f"train_{index:03d}") for index in range(100)]
        held_out = [SimpleNamespace(image_name=f"test_{index:03d}") for index in range(16)]
        namespace = {
            "os": os, "np": np, "SceneInfo": SceneInfo,
            "BasicPointCloud": SimpleNamespace, "SH2RGB": SH2RGB,
            "PlyData": PlyData, "PlyElement": PlyElement,
            "nerf_synthetic_split": nerf_synthetic_split,
            "nerf_synthetic_test_split": nerf_synthetic_test_split,
            "getNerfppNorm": mock.Mock(return_value={"radius": 2.0}),
            "readCamerasFromTransforms": mock.Mock(side_effect=(
                lambda path, name, white, extension:
                list(cameras if name == "transforms_train.json" else held_out)
            )),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(reader_path), "exec"),
             namespace)
        return namespace

    def test_blender_complement_matches_original_random_baseline_and_preserves_cache(self):
        (self.source / "transforms_train.json").write_text("{}")
        rng_state = np.random.get_state()
        self.addCleanup(np.random.set_state, rng_state)
        reader = self.synthetic_reader_with_real_random_cloud()
        write_ply, read_ply = reader["storePly"], reader["fetchPly"]
        cached_path = self.source / "points3d.ply"
        write_ply(cached_path, self.base.points, self.base.colors * 255.0)
        cached_bytes = cached_path.read_bytes()
        reader["storePly"] = mock.Mock(side_effect=AssertionError("shared PLY must not be written"))
        reader["fetchPly"] = mock.Mock(side_effect=AssertionError("shared PLY must not be read"))
        self.blender_reader.side_effect = reader["readNerfSyntheticInfo"]
        self.store_ply.side_effect = write_ply
        self.load_mvs.side_effect = AssertionError("Blender main method must not access MVS cache")
        self.args.white_background = True

        for dataset_name in ("auto", "blender"):
            for seed in (0, 17):
                with self.subTest(dataset_name=dataset_name, seed=seed):
                    self.args.dataset_name = dataset_name
                    self.args.init_mode = "colmap_dense"
                    self.args.rand_pcd = True
                    self.args.mvs_complement_mode = "none"
                    self.gaussians.reset_mock()
                    np.random.seed(seed)
                    baseline = self.make_scene()
                    baseline_cloud = self.gaussians.create_from_pcd.call_args.args[0]
                    self.assertEqual(len(baseline_cloud.points), 10000)
                    np.testing.assert_array_equal(baseline_cloud.normals, 0)
                    self.assertTrue(np.all(np.abs(baseline_cloud.points) <= 1.0))

                    for mode in ("coverage", "concat", "balanced"):
                        with self.subTest(mode=mode):
                            self.args.init_mode = "mvs_anywhere"
                            self.args.rand_pcd = False
                            self.args.mvs_complement_mode = mode
                            self.blender_reader.reset_mock()
                            self.gaussians.reset_mock()
                            self.store_ply.reset_mock()
                            np.random.seed(seed)
                            with mock.patch("utils.mvs_complement.build_complement_cloud",
                                            side_effect=AssertionError("Blender must skip complement")) as build:
                                scene = self.make_scene()
                            cloud = self.gaussians.create_from_pcd.call_args.args[0]
                            self.gaussians.create_from_pcd.assert_called_once_with(cloud, 2.0)
                            self.blender_reader.assert_called_once_with(
                                self.args.source_path, True, True, True, 3, dataset_name,
                            )
                            self.colmap_reader.assert_not_called()
                            self.load_mvs.assert_not_called()
                            build.assert_not_called()
                            self.assertFalse(self.args.rand_pcd)
                            self.assertTrue(scene.effective_rand_pcd)
                            self.assertEqual(scene.base_initialization_sha256,
                                             baseline.base_initialization_sha256)
                            self.assertEqual(scene.train_camera_names, baseline.train_camera_names)
                            self.assertEqual(scene.test_camera_names, baseline.test_camera_names)
                            np.testing.assert_array_equal(cloud.points, baseline_cloud.points)
                            np.testing.assert_array_equal(cloud.colors, baseline_cloud.colors)
                            np.testing.assert_array_equal(cloud.normals, baseline_cloud.normals)
                            self.store_ply.assert_called_once()
                            self.assertEqual(self.store_ply.call_args.args[0],
                                             str(self.output / "input.ply"))
                            artifact_cloud = read_ply(self.output / "input.ply")
                            np.testing.assert_array_equal(artifact_cloud.points, baseline_cloud.points)
                            np.testing.assert_array_equal(artifact_cloud.colors, baseline_cloud.colors)
                            self.assertEqual(cached_path.read_bytes(), cached_bytes)
                            for artifact in ("complement_initialization.json",
                                             "complement_initialization.npz"):
                                self.assertFalse((self.output / artifact).exists())
                            metadata = scene.initialization_metadata
                            self.assertEqual(metadata["mode"], "colmap_dense")
                            self.assertEqual(metadata["requested_mode"], "mvs_anywhere")
                            self.assertEqual(metadata["requested_complement_mode"], mode)
                            self.assertEqual(metadata["effective_complement_mode"], "none")
                            self.assertEqual(metadata["fusion_policy"], "blender_random_dropgaussian")
                            self.assertEqual(metadata["base_point_cloud_source"], "random")
                            self.assertEqual(metadata["path"], str(self.output / "input.ply"))
                            self.assertTrue(metadata["random_cloud_generated_in_memory"])
                            self.assertTrue(metadata["blender_random_override"])
                            self.assertFalse(metadata["mvs_cache_used"])
                            self.assertEqual(metadata["mvs_point_count"], 0)
                            for field in ("base_point_count", "selected_base_point_count",
                                          "fused_point_count", "selected_point_count"):
                                self.assertEqual(metadata[field], 10000)
        reader["storePly"].assert_not_called()
        reader["fetchPly"].assert_not_called()

    def test_blender_explicit_direct_mvs_does_not_force_random_reader(self):
        (self.source / "transforms_train.json").write_text("{}")
        self.args.dataset_name = "auto"
        self.args.mvs_complement_mode = "none"
        self.args.rand_pcd = False
        with mock.patch("utils.mvs_complement.build_complement_cloud") as build:
            scene = self.make_scene()
        self.blender_reader.assert_called_once_with(
            self.args.source_path, False, True, False, 3, "auto",
        )
        self.load_mvs.assert_called_once()
        build.assert_not_called()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        self.assertFalse(scene.effective_rand_pcd)
        self.assertEqual(scene.initialization_metadata["mode"], "mvs_anywhere")
        self.assertEqual(scene.initialization_metadata["fusion_policy"], "direct_mvs_replace")

    def test_blender_dataset_label_without_transforms_does_not_trigger_random_override(self):
        self.args.dataset_name = "blender"
        self.enable_complement(source="random")
        with mock.patch("utils.mvs_complement.build_complement_cloud", return_value=(
                self.mvs, {"selected_base_count": 0, "selected_mvs_count": 7,
                           "fallback": "no_reconstruction_geometry_direct_mvs_unchanged"}, {})) as build:
            scene = self.make_scene()
        self.colmap_reader.assert_called_once()
        self.blender_reader.assert_not_called()
        self.load_mvs.assert_called_once()
        build.assert_called_once()
        self.gaussians.create_from_pcd.assert_called_once_with(self.mvs, 2.0)
        self.assertFalse(scene.effective_rand_pcd)
        self.assertEqual(scene.initialization_metadata["mode"], "mvs_anywhere")
        self.assertEqual(scene.initialization_metadata["fusion_policy"],
                         "direct_mvs_no_reconstructed_base")

    def test_blender_saved_model_skips_random_gaussian_initialization_and_mvs(self):
        (self.source / "transforms_train.json").write_text("{}")
        self.args.dataset_name = "blender"
        self.enable_complement(source="random")
        self.load_mvs.side_effect = AssertionError("rendering must not access MVS cache")
        with mock.patch("utils.mvs_complement.build_complement_cloud") as build:
            scene = self.make_scene(load_iteration=10000)
        self.gaussians.load_ply.assert_called_once_with(str(
            self.output / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        ))
        self.gaussians.create_from_pcd.assert_not_called()
        self.store_ply.assert_not_called()
        self.load_mvs.assert_not_called()
        build.assert_not_called()
        self.assertEqual(scene.loaded_iter, 10000)
        self.assertFalse((self.output / "input.ply").exists())

    def test_training_manifest_reports_effective_blender_initializer_and_density_mode(self):
        (self.source / "transforms_train.json").write_text("{}")
        self.args.dataset_name = "blender"
        self.args.resolution = 1
        self.enable_complement(source="random")
        scene = self.make_scene()
        train_path = REPO_ROOT / "train.py"
        module = ast.parse(train_path.read_text())
        training = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "training")
        # Run the real manifest-writing statements after training, substituting
        # the timer/GPU counters so the test never starts optimization or CUDA.
        start = next(index for index, node in enumerate(training.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "effective_init_mode"
                             for target in node.targets))
        namespace = {
            "scene": scene, "dataset": self.args, "opt": SimpleNamespace(iterations=10000),
            "seed": 17, "initial_point_count": 4,
            "gaussians": SimpleNamespace(get_xyz=np.zeros((5, 3))),
            "convergence_curve": [], "training_start": 0.0,
            "time": SimpleNamespace(perf_counter=lambda: 1.0),
            "torch": SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
            "git_commit": "fixture", "git_dirty": True,
            "density_config": SimpleNamespace(mode="vcd", use_vcd=True, as_dict=lambda: {"mode": "vcd"}),
            "pruning_config": SimpleNamespace(enabled=False, strategy="none", dpp_only=False),
            "protection_config": None,
            "density_events": [], "first_iter": 0, "checkpoint": None,
            "density_log_path": str(self.output / "density.jsonl"),
            "os": os, "sys": sys, "json": json,
        }
        namespace["periodic_config"] = namespace["density_config"]
        exec(compile(ast.Module(body=training.body[start:], type_ignores=[]),
                     str(train_path), "exec"), namespace)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["method"], "DropGaussian+VCD")
        self.assertEqual(manifest["init_mode"], "colmap_dense")
        self.assertEqual(manifest["requested_init_mode"], "mvs_anywhere")
        self.assertTrue(manifest["protocol"]["rand_pcd"])
        self.assertFalse(self.args.rand_pcd)
        self.assertEqual(manifest["initialization_source"]["mvs_point_count"], 0)
        self.assertEqual(manifest["initialization_source"]["fusion_policy"],
                         "blender_random_dropgaussian")
        self.assertEqual(manifest["density_control"]["config"]["mode"], "vcd")
        self.assertEqual(manifest["density_control"]["pruning_strategy"], "none")
        self.assertNotIn("fastgs_pruning", manifest)

        # Late pruning is separate from the periodic EGD diagnostics.
        namespace["density_config"] = SimpleNamespace(
            mode="vcd_vcp", use_vcd=True, as_dict=lambda: {"mode": "vcd_vcp"})
        namespace["periodic_config"] = SimpleNamespace(as_dict=lambda: {"mode": "vcd"})
        namespace["pruning_config"] = SimpleNamespace(enabled=True, strategy="fastgs", protect_egd=False, dpp_only=False,
                                                      as_dict=lambda: {"strategy": "fastgs"})
        namespace.update(backend_setup_seconds=2.0, fastgs_events=[],
                         fastgs_log_path=str(self.output / "fastgs_pruning.jsonl"))
        exec(compile(ast.Module(body=training.body[start:], type_ignores=[]),
                     str(train_path), "exec"), namespace)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["method"], "DropGaussian+VCD+FASTGS_VCP")
        self.assertEqual(manifest["density_control"]["config"]["mode"], "vcd")
        self.assertEqual(manifest["fastgs_pruning"]["backend_setup_seconds"], 2.0)
        self.assertEqual(manifest["training_time_including_backend_setup"], 3.0)
        self.assertEqual(manifest["initialization_source"]["mvs_point_count"], 0)

        namespace["pruning_config"].protect_egd = True
        exec(compile(ast.Module(body=training.body[start:], type_ignores=[]),
                     str(train_path), "exec"), namespace)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["method"], "DropGaussian+VCD+FASTGS_VCP_EGD_PROTECTED")
        self.assertIn("after_native_cleanup", manifest["fastgs_pruning"]["protection_scope"])

        namespace["pruning_config"].dpp_only = True
        namespace["density_config"] = SimpleNamespace(mode="vcp", use_vcd=False)
        namespace["periodic_config"] = SimpleNamespace(as_dict=lambda: {"mode": "none"})
        namespace["protection_config"] = SimpleNamespace(as_dict=lambda: {"mode": "vcd"})
        exec(compile(ast.Module(body=training.body[start:], type_ignores=[]),
                     str(train_path), "exec"), namespace)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["method"], "DropGaussian+DPP")
        self.assertFalse(manifest["density_control"]["densification_gate"])
        self.assertEqual(manifest["density_control"]["config"]["mode"], "none")
        self.assertEqual(manifest["density_control"]["protection_evidence_config"]["mode"], "vcd")
        self.assertIn("native_adc_lineage", manifest["fastgs_pruning"]["protection_scope"])


class DirectMVSBranchDefaultsTest(unittest.TestCase):
    def test_cli_defaults_enable_direct_mvs_and_both_density_modules(self):
        parser = ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.init_mode, "mvs_anywhere")
        self.assertEqual(args.mvs_init_strategy, "replace")
        self.assertEqual(args.density_control, "vcd_vcp")
        self.assertEqual(args.vcp_strategy, "auto")
        self.assertEqual(args.mvs_complement_mode, "none")
        for removed in ("mvs_init_total_max_points", "mvs_init_dense_base_threshold",
                        "mvs_init_dense_residual_points", "mvs_geometry_correction",
                        "mvs_geometry_max_controls", "mvs_geometry_smoothing",
                        "mvs_geometry_max_image_size", "mvs_geometry_max_features",
                        "mvs_geometry_ratio", "mvs_geometry_reprojection_px",
                        "mvs_geometry_min_angle_deg", "mvs_geometry_match_radius_px",
                        "mvs_geometry_local_neighbors", "mvs_surface_init", "mvs_surface_neighbors",
                        "vcp_max_trials", "vcp_contribution_quantile"):
            self.assertFalse(hasattr(args, removed))

    def test_removed_experiment_options_are_not_accepted_by_cli(self):
        parser = ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        for option, value in (
            ("--mvs_geometry_correction", "tps"),
            ("--mvs_geometry_correction", "none"),
            ("--mvs_geometry_max_controls", "128"),
            ("--mvs_surface_init", "oas"),
            ("--mvs_surface_neighbors", "32"),
            ("--vcp_max_trials", "2"),
            ("--vcp_contribution_quantile", "0.1"),
        ):
            with self.subTest(option=option, value=value), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as rejected:
                    parser.parse_args([option, value])
                self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
