import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud
from utils.mvs_init import (
    build_adaptive_mvs_prior,
    fuse_residual_mvs_point_cloud,
    load_mvs_point_cloud,
    resolve_mvs_fusion_budget,
    resolve_mvs_init_point_budget,
    resolve_mvs_init_strategy,
    resolve_mvs_init_path,
    score_spatial_alignment,
    score_surface_structure,
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
    def test_auto_strategy_is_always_scene_adaptive(self):
        self.assertEqual(resolve_mvs_init_strategy("auto"), "adaptive")
        self.assertEqual(resolve_mvs_init_strategy("adaptive"), "adaptive")
        self.assertEqual(resolve_mvs_init_strategy("replace"), "replace")
        self.assertEqual(resolve_mvs_init_strategy("augment"), "augment")

    def test_candidate_budget_has_no_dataset_routing(self):
        self.assertEqual(
            resolve_mvs_init_point_budget("auto", "adaptive", 250_000),
            250_000,
        )
        self.assertEqual(
            resolve_mvs_init_point_budget("replace", "replace", 150_000),
            150_000,
        )

    def test_explicit_fusion_budget_has_no_dataset_routing(self):
        common = {
            "configured_total_points": 350_000,
            "dense_base_threshold": 200_000,
            "dense_residual_points": 100_000,
        }
        self.assertEqual(
            resolve_mvs_fusion_budget(
                "auto", base_point_count=30_000, **common
            ),
            350_000,
        )
        self.assertEqual(
            resolve_mvs_fusion_budget(
                "auto", base_point_count=300_000, **common
            ),
            350_000,
        )
        self.assertEqual(
            resolve_mvs_fusion_budget(
                "auto", base_point_count=100_000, **common
            ),
            350_000,
        )

    @staticmethod
    def _gradient_cameras(count=3):
        width = height = 64
        yy, xx = np.mgrid[:height, :width]
        image = np.stack(
            (xx / (width - 1), yy / (height - 1), np.full_like(xx, 0.5)),
            axis=-1,
        )
        image = Image.fromarray(np.rint(image * 255).astype(np.uint8))
        cameras = []
        for index in range(count):
            cameras.append(
                SimpleNamespace(
                    width=width,
                    height=height,
                    image=image,
                    image_name=f"view_{index}",
                    R=np.eye(3),
                    T=np.array([0.05 * (index - 1), 0.0, 0.0]),
                    K=np.array(
                        [[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]]
                    ),
                )
            )
        return cameras

    def test_structure_score_separates_surface_from_volume(self):
        generator = np.random.default_rng(4)
        xy = generator.uniform(-1.0, 1.0, (4_000, 2))
        surface_points = np.column_stack((xy, 0.1 * xy[:, 0]))
        volume_points = generator.uniform(-1.0, 1.0, (4_000, 3))
        surface = BasicPointCloud(
            surface_points,
            np.zeros_like(surface_points),
            np.zeros_like(surface_points),
        )
        volume = BasicPointCloud(
            volume_points,
            np.zeros_like(volume_points),
            np.zeros_like(volume_points),
        )
        self.assertGreater(
            score_surface_structure(surface),
            score_surface_structure(volume) + 0.1,
        )

    def test_alignment_score_rejects_counterfactual_coordinates(self):
        generator = np.random.default_rng(17)
        xy = generator.uniform(-1.0, 1.0, (4_000, 2))
        points = np.column_stack((xy, 0.2 * xy[:, 0] - 0.1 * xy[:, 1]))
        dense_xy = generator.uniform(-1.0, 1.0, (12_000, 2))
        aligned_points = np.column_stack(
            (
                dense_xy,
                0.2 * dense_xy[:, 0] - 0.1 * dense_xy[:, 1],
            )
        )
        deranged_points = np.column_stack(
            [
                aligned_points[generator.permutation(12_000), axis]
                for axis in range(3)
            ]
        )
        base = BasicPointCloud(points, np.zeros_like(points), np.zeros_like(points))
        aligned = BasicPointCloud(
            aligned_points,
            np.zeros_like(aligned_points),
            np.zeros_like(aligned_points),
        )
        deranged = BasicPointCloud(
            deranged_points,
            np.zeros_like(deranged_points),
            np.zeros_like(deranged_points),
        )
        self.assertGreater(
            score_spatial_alignment(base, aligned),
            score_spatial_alignment(base, deranged) + 0.2,
        )

    def test_unstructured_base_degrades_to_no_residual_points(self):
        generator = np.random.default_rng(5)
        volume_points = generator.uniform(-1.0, 1.0, (4_000, 3))
        xy = generator.uniform(-1.0, 1.0, (4_000, 2))
        mvs_points = np.column_stack((xy, 3.0 + 0.1 * xy[:, 0]))
        base = BasicPointCloud(
            volume_points,
            np.full_like(volume_points, 0.5),
            np.zeros_like(volume_points),
        )
        mvs = BasicPointCloud(
            mvs_points,
            np.full_like(mvs_points, 0.5),
            np.zeros_like(mvs_points),
        )
        prior = build_adaptive_mvs_prior(
            base,
            mvs,
            self._gradient_cameras(),
            cache_path=Path("missing.ply"),
            max_residual_points=1_000,
        )
        self.assertEqual(prior.point_count, 0)
        self.assertEqual(
            prior.metadata["fusion_policy"], "adaptive_exact_baseline"
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

    def test_resolver_prefers_requested_backend_over_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anywhere = (
                root
                / "MVSAnywhere_3_views_H_4_W_4_x_init"
                / "mvs_input.ply"
            )
            former = (
                root
                / "MVSFormer_3_views_H_8_W_8_x_init"
                / "mvs_input.ply"
            )
            for path in (anywhere, former):
                _write_ply(path, 2)
                (path.parent / "mvs_cache_meta.json").write_text(
                    json.dumps({"view_names": ["a", "b", "c"]}),
                    encoding="utf-8",
                )
            path, _, provenance = resolve_mvs_init_path(
                root, n_views=3, train_camera_names=["a", "b", "c"]
            )
            self.assertEqual(path, anywhere)
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
