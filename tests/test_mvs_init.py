import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud
from utils.mvs_init import (
    fuse_residual_mvs_point_cloud,
    load_mvs_point_cloud,
    resolve_mvs_fusion_budget,
    resolve_mvs_init_point_budget,
    resolve_mvs_init_strategy,
    resolve_mvs_init_path,
    should_use_dtu_mvs_prior,
)


def _write_ply(path: Path, count: int):
    values = np.zeros(
        count,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    values["x"] = np.arange(count)
    values["red"] = 255
    path.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(values, "vertex")]).write(path)


class MVSInitTest(unittest.TestCase):
    def test_dtu_quality_gate_uses_only_input_point_statistics(self):
        common = {
            "requested_strategy": "auto",
            "dataset_name": "dtu",
            "n_views": 3,
        }
        self.assertTrue(
            should_use_dtu_mvs_prior(
                base_point_count=9_999,
                mvs_source_point_count=100_000,
                **common,
            )
        )
        self.assertFalse(
            should_use_dtu_mvs_prior(
                base_point_count=10_000,
                mvs_source_point_count=1_000_000,
                **common,
            )
        )
        self.assertFalse(
            should_use_dtu_mvs_prior(
                base_point_count=5_000,
                mvs_source_point_count=99_999,
                **common,
            )
        )
        self.assertFalse(
            should_use_dtu_mvs_prior(
                requested_strategy="auto",
                dataset_name="dtu",
                n_views=6,
                base_point_count=5_000,
                mvs_source_point_count=1_000_000,
            )
        )
        self.assertTrue(
            should_use_dtu_mvs_prior(
                requested_strategy="augment",
                dataset_name="dtu",
                n_views=9,
                base_point_count=50_000,
                mvs_source_point_count=1,
            )
        )

    def test_auto_strategy_uses_only_dataset_and_training_view_count(self):
        self.assertEqual(
            resolve_mvs_init_strategy("auto", n_views=3, is_synthetic=False),
            "replace",
        )
        self.assertEqual(
            resolve_mvs_init_strategy("auto", n_views=6, is_synthetic=False),
            "augment",
        )
        self.assertEqual(
            resolve_mvs_init_strategy(
                "auto", n_views=9, is_synthetic=False, dataset_name="dtu"
            ),
            "augment",
        )
        self.assertEqual(
            resolve_mvs_init_strategy("auto", n_views=8, is_synthetic=True),
            "passthrough",
        )
        self.assertEqual(
            resolve_mvs_init_strategy("replace", n_views=8, is_synthetic=True),
            "replace",
        )

    def test_auto_three_view_replacement_uses_lightweight_budget(self):
        self.assertEqual(
            resolve_mvs_init_point_budget("auto", "replace", 250_000),
            100_000,
        )
        self.assertEqual(
            resolve_mvs_init_point_budget(
                "auto", "augment", 250_000, dataset_name="dtu"
            ),
            100_000,
        )
        self.assertEqual(
            resolve_mvs_init_point_budget("auto", "augment", 250_000),
            250_000,
        )
        self.assertEqual(
            resolve_mvs_init_point_budget(
                "auto", "augment", 250_000, dense_base=True
            ),
            100_000,
        )
        self.assertEqual(
            resolve_mvs_init_point_budget("replace", "replace", 150_000),
            150_000,
        )

    def test_auto_fusion_budget_uses_fixed_training_input_rules(self):
        common = {
            "configured_total_points": 350_000,
            "dense_base_threshold": 200_000,
            "dense_residual_points": 100_000,
        }
        self.assertEqual(
            resolve_mvs_fusion_budget(
                "auto", base_point_count=30_000, dataset_name="dtu", **common
            ),
            100_000,
        )
        self.assertEqual(
            resolve_mvs_fusion_budget(
                "auto", base_point_count=300_000, **common
            ),
            400_000,
        )
        self.assertEqual(
            resolve_mvs_fusion_budget(
                "auto", base_point_count=100_000, **common
            ),
            350_000,
        )

    @staticmethod
    def _cloud(count: int, offset: float = 0.0) -> BasicPointCloud:
        points = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + offset
        return BasicPointCloud(
            points=points,
            colors=np.zeros_like(points),
            normals=np.zeros_like(points),
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

    def test_loader_caps_points_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MVSAnywhere_3_views_H_4_W_4_x_init" / "mvs_input.ply"
            _write_ply(path, 10)
            first, _, metadata = load_mvs_point_cloud(
                root, n_views=3, train_camera_names=[], max_points=4
            )
            second, _, _ = load_mvs_point_cloud(
                root, n_views=3, train_camera_names=[], max_points=4
            )
            self.assertEqual(metadata["source_point_count"], 10)
            self.assertEqual(metadata["selected_point_count"], 4)
            np.testing.assert_array_equal(first.points, second.points)

    def test_residual_fusion_preserves_base_and_fills_budget(self):
        base = self._cloud(4)
        mvs = self._cloud(10, offset=100.0)
        fused, metadata = fuse_residual_mvs_point_cloud(
            base, mvs, total_max_points=9
        )
        np.testing.assert_array_equal(fused.points[:4], base.points)
        self.assertEqual(fused.points[4:].shape, (5, 3))
        self.assertTrue(np.all(fused.points[4:] >= 100.0))
        self.assertEqual(metadata["mvs_point_count"], 5)
        self.assertEqual(metadata["fusion_policy"], "preserve_base_then_fill_with_mvs")

        repeated, repeated_metadata = fuse_residual_mvs_point_cloud(
            base, mvs, total_max_points=9
        )
        np.testing.assert_array_equal(fused.points, repeated.points)
        self.assertEqual(metadata["fused_sha256"], repeated_metadata["fused_sha256"])

    def test_residual_fusion_leaves_base_above_budget_unchanged(self):
        base = self._cloud(8)
        fused, metadata = fuse_residual_mvs_point_cloud(
            base, self._cloud(10, offset=100.0),
            total_max_points=8,
        )
        np.testing.assert_array_equal(fused.points, base.points)
        self.assertEqual(metadata["mvs_point_count"], 0)
        self.assertEqual(metadata["fusion_policy"], "base_budget_passthrough")


if __name__ == "__main__":
    unittest.main()
