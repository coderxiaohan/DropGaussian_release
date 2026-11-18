"""CPU-only complementary point selection tests; no scene/CUDA dependencies."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from utils.graphics_utils import BasicPointCloud
from utils.mvs_complement import build_complement_cloud, _depth_geometry, _project_support


def cloud(points, *, dtype=np.float32):
    xyz = np.asarray(points, dtype=dtype).reshape(-1, 3)
    colors = np.arange(len(xyz) * 3, dtype=np.float64).reshape(-1, 3) / max(1, 3 * len(xyz))
    return BasicPointCloud(xyz, colors, np.zeros_like(xyz))


def camera(name, translation=0.0):
    return SimpleNamespace(
        image_name=name, image=np.zeros((32, 32, 3), dtype=np.uint8),
        width=32, height=32, K=np.array([[8., 0, 16], [0, 8., 16], [0, 0, 1]]),
        R=np.eye(3), T=np.array([translation, 0, 0.]),
    )


class ComplementCloudTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "mvs_depths_masks.npz"
        self.cameras = [camera("a"), camera("b", -0.2)]
        self.base = cloud([[-1, 0, 2], [-1, -1, 2]], dtype=np.float64)
        # Exact overlap; near overlap; real hole; occluded; one view only;
        # behind camera; completely outside both training images.
        self.candidates = cloud([[-1, 0, 2], [-0.99, 0, 2], [1, 0, 2], [1, 0, 4],
                                 [4, 0, 2], [0, 0, -2], [100, 0, 2]])
        self.write_cache()

    def write_cache(self, cameras=None, *, depths=None, masks=None, names=None):
        cameras = self.cameras if cameras is None else cameras
        if depths is None:
            depths = np.full((len(cameras), 32, 32), 2., dtype=np.float32)
        if masks is None:
            masks = np.ones_like(depths, dtype=bool)
        np.savez(self.path, depths=depths, masks=masks,
                 image_names=np.asarray(names if names is not None else [c.image_name for c in cameras]))

    def run_helper(self, **kwargs):
        return build_complement_cloud(self.base, self.candidates, self.cameras,
                                      depth_cache_path=self.path, **kwargs)

    def test_coverage_prefers_hole_rejects_overlap_occlusion_and_single_view(self):
        result, meta, arrays = self.run_helper(mvs_budget=1)
        self.assertEqual(arrays["selected_mvs_indices"].tolist(), [2])
        self.assertGreater(arrays["candidate_scores"][2], arrays["candidate_scores"][1])
        self.assertGreater(arrays["candidate_scores"][1], 0)
        self.assertEqual(arrays["candidate_scores"][0], 0)
        self.assertEqual(arrays["candidate_scores"][3:].tolist(), [0, 0, 0, 0])
        self.assertEqual(arrays["candidate_support_views"].tolist(), [2, 2, 2, 0, 1, 0, 0])
        np.testing.assert_array_equal(result.points[:2], self.base.points)
        np.testing.assert_array_equal(result.colors[:2], self.base.colors)
        np.testing.assert_array_equal(result.points[2], self.candidates.points[2])
        np.testing.assert_array_equal(result.colors[2], self.candidates.colors[2])
        self.assertEqual(result.points.dtype, np.float64)
        self.assertEqual(result.colors.dtype, np.float64)
        self.assertEqual(meta["selected_base_count"], 2)
        self.assertEqual(meta["selected_mvs_count"], 1)
        self.assertTrue(meta["complete_base_preserved"])
        self.assertEqual(meta["base_reconstruction_camera_provenance"], "not_verified_by_complement_helper")
        json.dumps(meta, allow_nan=False)

    def test_no_valid_candidate_keeps_base_without_forcing_count(self):
        self.write_cache(masks=np.zeros((2, 32, 32), dtype=bool))
        result, meta, arrays = self.run_helper()
        np.testing.assert_array_equal(result.points, self.base.points)
        self.assertEqual(len(arrays["selected_mvs_indices"]), 0)
        self.assertEqual(meta["geometry_filter_shortfall"], len(self.candidates.points))

    def test_tiny_budget_without_supported_candidates_fails_clearly(self):
        self.write_cache(masks=np.zeros((2, 32, 32), dtype=bool))
        with self.assertRaisesRegex(ValueError, "empty cloud"):
            self.run_helper(total_budget=1)

    def test_empty_candidate_pool_preserves_complete_base(self):
        result, meta, arrays = build_complement_cloud(
            self.base, cloud([]), self.cameras, depth_cache_path=self.path,
        )
        np.testing.assert_array_equal(result.points, self.base.points)
        self.assertTrue(meta["complete_base_preserved"])
        self.assertEqual(meta["selected_mvs_count"], 0)
        self.assertEqual(len(arrays["candidate_scores"]), 0)

    def test_equal_pixel_different_depth_is_not_duplicate_coverage(self):
        cameras = [camera("a"), camera("b")]
        self.write_cache(cameras, depths=np.full((2, 32, 32), 4., dtype=np.float32))
        result, meta, arrays = build_complement_cloud(
            cloud([[0, 0, 2]]), cloud([[0, 0, 4]]), cameras, depth_cache_path=self.path,
        )
        self.assertEqual(len(result.points), 2)
        self.assertEqual(arrays["candidate_scores"].tolist(), [1.0])
        self.assertTrue(all(record["supported_base_projection_count"] == 0 for record in meta["views"]))

    def test_balanced_and_coverage_use_same_pool_quotas_and_base(self):
        outputs = [self.run_helper(mode=mode, mvs_budget=1, total_budget=2)
                   for mode in ("coverage", "balanced")]
        for output in outputs:
            self.assertEqual(len(output[0].points), 2)
            self.assertEqual(output[1]["selected_base_count"], 1)
            self.assertFalse(output[1]["complete_base_preserved"])
        for name in ("eligible_mvs_count", "base_quota", "mvs_quota"):
            self.assertEqual(outputs[0][1][name], outputs[1][1][name])
        np.testing.assert_array_equal(outputs[0][2]["selected_base_indices"], outputs[1][2]["selected_base_indices"])
        np.testing.assert_array_equal(outputs[0][2]["candidate_scores"], outputs[1][2]["candidate_scores"])

    def test_concat_is_unfiltered_and_preserves_attributes(self):
        result, meta, arrays = build_complement_cloud(self.base, self.candidates, [], mode="concat", mvs_budget=99)
        np.testing.assert_array_equal(result.points, np.concatenate((self.base.points, self.candidates.points)))
        np.testing.assert_array_equal(result.colors, np.concatenate((self.base.colors, self.candidates.colors)))
        self.assertFalse(meta["depth_cache_checked"])
        self.assertEqual(meta["candidate_pool_policy"], "unfiltered")
        self.assertEqual(arrays["selected_mvs_indices"].tolist(), list(range(len(self.candidates.points))))

    def test_budget_cap_and_borrowing(self):
        cases = [(8, 8, 8, 6, (3, 3)), (1, 8, 8, 6, (1, 5)),
                 (8, 1, 8, 6, (5, 1)), (8, 8, 2, 6, (4, 2)),
                 (1, 1, 8, 6, (1, 1)), (8, 8, 8, 5, (2, 3))]
        for base_count, mvs_count, budget, cap, expected in cases:
            with self.subTest(base_count=base_count, mvs_count=mvs_count, budget=budget, cap=cap):
                base = cloud(np.repeat([[0, 0, 2]], base_count, axis=0))
                candidates = cloud(np.repeat([[1, 0, 2]], mvs_count, axis=0))
                result, meta, arrays = build_complement_cloud(base, candidates, [], mode="concat", mvs_budget=budget, total_budget=cap)
                self.assertEqual((meta["selected_base_count"], meta["selected_mvs_count"]), expected)
                self.assertLessEqual(len(result.points), cap)
                self.assertEqual(meta["budget_shortfall"], cap - len(result.points))
                self.assertEqual(meta["complete_base_preserved"], expected[0] == base_count)
                np.testing.assert_array_equal(result.colors[:expected[0]], base.colors[arrays["selected_base_indices"]])

    def test_determinism_input_immutability_and_global_rng_unchanged(self):
        copies = [array.copy() for pcd in (self.base, self.candidates) for array in pcd]
        state = np.random.get_state()
        outputs = [self.run_helper(mvs_budget=1) for _ in range(2)]
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        self.assertEqual(state[2:], after[2:])
        np.testing.assert_array_equal(state[1], after[1])
        for first, second in zip(outputs[0][0], outputs[1][0]):
            np.testing.assert_array_equal(first, second)
        for original, copy in zip([array for pcd in (self.base, self.candidates) for array in pcd], copies):
            np.testing.assert_array_equal(original, copy)

    def test_scores_are_view_count_normalized(self):
        original = self.run_helper()[2]
        cameras = [camera(str(index), 0.0 if index % 2 == 0 else -0.2) for index in range(6)]
        self.write_cache(cameras)
        more = build_complement_cloud(self.base, self.candidates, cameras, depth_cache_path=self.path)[2]
        # Compare already-two-view-supported points, not a repeated sole view.
        np.testing.assert_allclose(original["candidate_scores"][:4], more["candidate_scores"][:4])
        np.testing.assert_array_equal(original["candidate_support_views"][:3] * 3, more["candidate_support_views"][:3])

    def test_cache_names_must_match_exact_ordered_training_list(self):
        cases = [["b", "a"], ["a", "test"], ["a", "b", "test"], ["a", "a"], [["a", "b"]]]
        for names in cases:
            with self.subTest(names=names):
                self.write_cache(names=names)
                with self.assertRaisesRegex(ValueError, "image_names"):
                    self.run_helper()

    def test_duplicate_training_names_are_rejected(self):
        cameras = [camera("a"), camera("a")]
        self.write_cache(cameras)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_complement_cloud(self.base, self.candidates, cameras, depth_cache_path=self.path)

    def test_missing_cache_or_keys_rejected(self):
        with self.assertRaises(FileNotFoundError):
            build_complement_cloud(self.base, self.candidates, self.cameras)
        np.savez(self.path, image_names=np.asarray(["a", "b"]), depths=np.ones((2, 32, 32)))
        with self.assertRaisesRegex(ValueError, "requires"):
            self.run_helper()

    def test_invalid_depth_has_no_support(self):
        for bad in (np.nan, np.inf, 0., -1.):
            with self.subTest(depth=bad):
                self.write_cache(depths=np.full((2, 32, 32), bad))
                _, meta, arrays = self.run_helper()
                self.assertEqual(meta["selected_mvs_count"], 0)
                self.assertFalse(arrays["candidate_support_views"].any())

    def test_invalid_mask_has_no_support(self):
        for bad in (np.nan, -1., 0.):
            with self.subTest(mask=bad):
                self.write_cache(masks=np.full((2, 32, 32), bad))
                self.assertEqual(self.run_helper()[1]["selected_mvs_count"], 0)

    def test_one_training_view_cannot_support_completion(self):
        self.write_cache(self.cameras[:1])
        _, meta, _ = build_complement_cloud(self.base, self.candidates, self.cameras[:1], depth_cache_path=self.path)
        self.assertEqual(meta["selected_mvs_count"], 0)

    def test_no_geometry_returns_exact_direct_without_cache_access(self):
        for reason in ("random", "unknown", "absent"):
            with self.subTest(reason=reason):
                direct = cloud([[0, 0, 3], [1, 1, 3]])
                result, meta, arrays = build_complement_cloud(
                    None if reason == "absent" else self.base, self.candidates, self.cameras,
                    base_is_geometry=reason == "absent", direct_pcd=direct,
                    depth_cache_path="/does/not/exist.npz",
                )
                self.assertIs(result, direct)
                self.assertEqual(meta["effective_mode"], "direct_mvs")
                self.assertEqual(meta["selected_base_count"], 0)
                self.assertEqual(meta["selected_mvs_count"], 2)
                self.assertFalse(meta["depth_cache_checked"])
                self.assertEqual(arrays["selected_mvs_indices"].tolist(), [0, 1])

    def test_fallback_requires_exact_direct_and_enforces_cap(self):
        with self.assertRaisesRegex(ValueError, "direct_pcd"):
            self.run_helper(base_is_geometry=False)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.run_helper(base_is_geometry=False, direct_pcd=self.candidates, total_budget=2)

    def test_nonuniform_resolution_scaling_and_camera_z_geometry(self):
        cam = camera("a")
        cam.image = np.zeros((16, 8, 3), dtype=np.uint8)
        cam.K = np.array([[16., 0, 10.], [0, 12., 7.], [0, 0, 1.]])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        cam.R = rotation.T
        cam.T = np.array([0.5, -0.25, 1.])
        intrinsics, extrinsics = _depth_geometry(cam, width=64, height=8)
        np.testing.assert_array_equal(intrinsics, [[32., 0, 20.], [0, 3., 1.75], [0, 0, 1.]])
        point = np.array([[0.25, -1.5, 1.]])
        depth = np.full((8, 64), 2.)
        pixels, weight = _project_support(point, intrinsics, extrinsics, depth, np.ones_like(depth, dtype=bool), 0.02)
        np.testing.assert_allclose(pixels, [[52., 1.75]])
        self.assertEqual(weight.tolist(), [1.0])  # z=2; Euclidean distance >2.

    def test_invalid_parameters_fail_before_processing(self):
        cases = [{"mode": "auto"}, {"mvs_budget": 0}, {"mvs_budget": True}, {"mvs_budget": 1.5},
                 {"total_budget": -1}, {"total_budget": False}, {"coverage_pixels": 0},
                 {"coverage_pixels": np.inf}, {"depth_relative_scale": np.nan},
                 {"depth_relative_scale": -0.1}, {"base_is_geometry": "random"}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_helper(**kwargs)

    def test_invalid_cloud_rejected(self):
        bad = BasicPointCloud(np.full((2, 3), np.nan), np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_complement_cloud(self.base, bad, self.cameras, depth_cache_path=self.path)


if __name__ == "__main__":
    unittest.main()
