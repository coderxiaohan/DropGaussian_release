import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

import utils.mvs_init as mvs_init
from utils.mvs_init import (
    load_mvs_point_cloud,
    resolve_mvs_init_strategy,
    resolve_mvs_init_path,
)


def _write_ply(path: Path, count: int, *, points=None):
    values = np.zeros(
        count,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    if points is None:
        values["x"] = np.arange(count)
    else:
        for axis, name in enumerate(("x", "y", "z")):
            values[name] = np.asarray(points)[:, axis]
    values["red"] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(values, "vertex")]).write(path)


class MVSInitTest(unittest.TestCase):
    def test_strategy_accepts_direct_replace_only(self):
        self.assertEqual(resolve_mvs_init_strategy("replace"), "replace")
        for strategy in ("auto", "adaptive", "augment", "", "unknown", None):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "only.*replace"):
                    resolve_mvs_init_strategy(strategy)

    def test_no_adaptive_or_residual_api_remains(self):
        for name in (
            "AdaptiveMvsPrior", "MvsReliabilityScore",
            "build_adaptive_mvs_prior", "score_mvs_reliability",
            "score_spatial_alignment", "score_surface_structure",
            "fuse_residual_mvs_point_cloud", "resolve_mvs_fusion_budget",
            "resolve_mvs_init_point_budget",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(mvs_init, name))

    def test_model_initializer_has_no_adaptive_prior_argument(self):
        # Parse without importing CUDA extensions; all runtime density tests
        # remain in test_view_consistent_model.py.
        source = Path(__file__).resolve().parents[1] / "scene/gaussian_model.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        model = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
        initializer = next(node for node in model.body
                           if isinstance(node, ast.FunctionDef) and node.name == "create_from_pcd")
        self.assertEqual(
            [argument.arg for argument in initializer.args.args],
            ["self", "pcd", "spatial_lr_scale"],
        )

    def test_resolver_prefers_verified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "MVSAnywhere_3_views_H_1440_W_1920_x_init" / "mvs_input.ply"
            verified = root / "MVSAnywhere_3_views_H_960_W_1280_x_init" / "mvs_input.ply"
            _write_ply(legacy, 2)
            _write_ply(verified, 2)
            (verified.parent / "mvs_cache_meta.json").write_text(
                json.dumps({"view_names": ["a", "b", "c"]}), encoding="utf-8"
            )
            path, _, provenance = resolve_mvs_init_path(
                root, n_views=3, train_camera_names=["a", "b", "c"]
            )
            self.assertEqual(path, verified)
            self.assertTrue(provenance)

    def test_resolver_prefers_requested_backend_over_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anywhere = root / "MVSAnywhere_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            former = root / "MVSFormer_3_views_H_8_W_8_x_init" / "mvs_input.ply"
            for path in (anywhere, former):
                _write_ply(path, 2)
                (path.parent / "mvs_cache_meta.json").write_text(
                    json.dumps({"view_names": ["a", "b", "c"]}), encoding="utf-8",
                )
            path, _, provenance = resolve_mvs_init_path(
                root, n_views=3, train_camera_names=["a", "b", "c"]
            )
            self.assertEqual(path, anywhere)
            self.assertTrue(provenance)

    def test_resolver_prefers_higher_resolution_within_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "MVSAnywhere_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            large = root / "MVSAnywhere_3_views_H_8_W_8_x_init" / "mvs_input.ply"
            for path in (small, large):
                _write_ply(path, 2)
            path, _, provenance = resolve_mvs_init_path(
                root, n_views=3, train_camera_names=["a", "b", "c"]
            )
            self.assertEqual(path, large)
            self.assertFalse(provenance)

    def test_resolver_rejects_known_split_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MVSAnywhere_3_views_H_960_W_1280_x_init" / "mvs_input.ply"
            _write_ply(path, 2)
            (path.parent / "mvs_cache_meta.json").write_text(
                json.dumps({"view_names": ["test_view"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "differ"):
                resolve_mvs_init_path(root, n_views=3, train_camera_names=["train"])
            with self.assertRaisesRegex(ValueError, "differ"):
                resolve_mvs_init_path(
                    root, n_views=3, train_camera_names=["train"], explicit_path=path
                )

    def test_resolver_reads_legacy_npz_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MVSFormer_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            _write_ply(path, 2)
            np.savez_compressed(
                path.parent / "mvs_depths_masks.npz",
                image_names=np.asarray(["a", "b", "c"]),
            )
            selected, _, provenance = resolve_mvs_init_path(
                root, n_views=3, train_camera_names=["a", "b", "c"]
            )
            self.assertEqual(selected, path)
            self.assertTrue(provenance)
            _, _, metadata = load_mvs_point_cloud(
                root, n_views=3, train_camera_names=["a", "b", "c"]
            )
            self.assertEqual(metadata["backend"], "MVSFormer")

    def test_explicit_directory_or_file_both_load_without_quality_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "arbitrary cloud directory" / "mvs_input.ply"
            points = np.asarray([[90, 0, -12], [-90, 42, 30]], dtype=np.float32)
            _write_ply(path, 2, points=points)
            for explicit_path in (path, path.parent):
                cloud, selected, metadata = load_mvs_point_cloud(
                    root, n_views=3, train_camera_names=["a", "b", "c"],
                    explicit_path=explicit_path,
                )
                np.testing.assert_array_equal(cloud.points, points)
                np.testing.assert_array_equal(cloud.colors, [[1, 0, 0], [1, 0, 0]])
                np.testing.assert_array_equal(cloud.normals, np.zeros_like(points))
                self.assertEqual(selected, path)
                self.assertFalse(metadata["camera_provenance_verified"])
                self.assertEqual(metadata["selection"], "all")

    def test_loader_caps_points_deterministically_without_global_rng_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MVSAnywhere_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            _write_ply(path, 10)
            rng_before = np.random.get_state()
            first, _, metadata = load_mvs_point_cloud(
                root, n_views=3, train_camera_names=[], max_points=4
            )
            second, _, repeated = load_mvs_point_cloud(
                root, n_views=3, train_camera_names=[], max_points=4
            )
            rng_after = np.random.get_state()
            self.assertEqual(metadata["source_point_count"], 10)
            self.assertEqual(metadata["selected_point_count"], 4)
            self.assertEqual(metadata["selection"], "seed0_uniform_without_replacement")
            self.assertEqual(metadata["selected_sha256"], repeated["selected_sha256"])
            np.testing.assert_array_equal(first.points, second.points)
            for before, after in zip(rng_before, rng_after):
                np.testing.assert_equal(before, after)

    def test_loader_filters_only_nonfinite_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MVSAnywhere_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            points = np.asarray([[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0], [-20, 5, -8]])
            _write_ply(path, 4, points=points)
            cloud, _, metadata = load_mvs_point_cloud(
                root, n_views=3, train_camera_names=[]
            )
            np.testing.assert_array_equal(cloud.points, points[[0, 3]].astype(np.float32))
            self.assertEqual(metadata["source_point_count"], 2)

    def test_loader_rejects_missing_cache_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "no MVS"):
                load_mvs_point_cloud(
                    directory, n_views=3, train_camera_names=["a", "b", "c"]
                )

    def test_loader_rejects_empty_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MVSAnywhere_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            _write_ply(path, 0)
            with self.assertRaisesRegex(ValueError, "empty"):
                load_mvs_point_cloud(root, n_views=3, train_camera_names=[])

    def test_loader_rejects_invalid_point_budget(self):
        for max_points in (0, -1, 1.5, True, "100", None):
            with self.subTest(max_points=max_points):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    load_mvs_point_cloud(
                        "unused", n_views=3, train_camera_names=[], max_points=max_points
                    )


if __name__ == "__main__":
    unittest.main()
