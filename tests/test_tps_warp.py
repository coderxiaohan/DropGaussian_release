import json
import unittest

import numpy as np

from utils.tps_warp import fit_similarity, fit_tps, select_control_indices


class TPSWarpTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.source = rng.uniform(-2, 2, size=(19, 3))
        self.query = rng.uniform(-3, 3, size=(31, 3))
        self.affine = np.asarray([[1.2, 0.1, -0.2], [0.3, 0.7, 0.15], [0.02, -0.1, 1.1]])
        self.offset = np.asarray([0.05, -0.03, 0.004])

    def test_identity_preserves_submillimeter_coordinates(self):
        warp = fit_tps(self.source, self.source)
        query = np.vstack((self.query, [[0.0000123, -0.0000456, 0.0000789]]))
        np.testing.assert_array_equal(warp.transform(query), query)
        self.assertEqual(warp.diagnostics["control_rmse_after"], 0.0)

    def test_affine_map_is_exact_with_smoothing(self):
        target = self.source @ self.affine.T + self.offset
        for smoothing in (0, 1e-3, 10):
            with self.subTest(smoothing=smoothing):
                warp = fit_tps(self.source, target, smoothing=smoothing)
                np.testing.assert_allclose(
                    warp.transform(self.query), self.query @ self.affine.T + self.offset,
                    rtol=1e-12, atol=1e-12,
                )

    def test_four_non_coplanar_controls_are_sufficient(self):
        source = np.asarray([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        target = source @ self.affine.T + self.offset
        warp = fit_tps(source, target)
        np.testing.assert_allclose(warp.transform(self.query), self.query @ self.affine.T + self.offset, atol=1e-12)

    def test_nonrigid_controls_interpolate_without_smoothing(self):
        target = self.source.copy()
        target[:, 2] += 0.1 * self.source[:, 0] ** 2
        warp = fit_tps(self.source, target, smoothing=0)
        np.testing.assert_allclose(warp.transform(self.source), target, atol=1e-12)
        self.assertLess(warp.diagnostics["linear_system_relative_residual"], 1e-12)
        self.assertLess(warp.diagnostics["affine_constraint_residual"], 1e-12)

    def test_positive_smoothing_relaxes_nonaffine_interpolation(self):
        target = self.source.copy()
        target[0, 2] += 0.7
        interpolated = fit_tps(self.source, target, smoothing=0)
        smooth = fit_tps(self.source, target, smoothing=0.5)
        self.assertGreater(smooth.diagnostics["control_rmse_after"], interpolated.diagnostics["control_rmse_after"])
        self.assertLess(np.linalg.norm(smooth.weights), np.linalg.norm(interpolated.weights))

    def test_uniform_scene_scale_and_translation_equivariance(self):
        target = self.source.copy()
        target[:, 2] += 0.1 * self.source[:, 0] ** 2
        expected = fit_tps(self.source, target, smoothing=0.2).transform(self.query)
        for scale in (1e-5, 3, 1e6):
            offset = np.asarray([4, -2, 9]) * scale
            warp = fit_tps(self.source * scale + offset, target * scale + offset, smoothing=0.2)
            actual = (warp.transform(self.query * scale + offset) - offset) / scale
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_rigid_coordinate_frame_equivariance(self):
        theta = 0.63
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        target = self.source.copy()
        target[:, 2] += 0.2 * np.sin(self.source[:, 0])
        expected = fit_tps(self.source, target).transform(self.query)
        warp = fit_tps(self.source @ rotation.T, target @ rotation.T)
        np.testing.assert_allclose(warp.transform(self.query @ rotation.T), expected @ rotation.T, atol=1e-12)

    def test_chunked_application_matches_whole_and_keeps_inputs(self):
        target = self.source @ self.affine.T + self.offset
        source_before, target_before, query_before = self.source.copy(), target.copy(), self.query.copy()
        warp = fit_tps(self.source, target)
        np.testing.assert_allclose(warp.transform(self.query, chunk_size=2), warp.transform(self.query, chunk_size=100), atol=1e-14)
        np.testing.assert_array_equal(self.source, source_before)
        np.testing.assert_array_equal(target, target_before)
        np.testing.assert_array_equal(self.query, query_before)
        self.assertEqual(warp.transform(np.empty((0, 3))).shape, (0, 3))
        self.assertEqual(warp.transform(self.query.astype(np.float32)).dtype, np.float64)
        self.source[:] = 100
        np.testing.assert_allclose(warp.transform(query_before), query_before @ self.affine.T + self.offset, atol=1e-12)

    def test_diagnostics_are_json_serializable(self):
        diagnostics = fit_tps(self.source, self.source + self.offset).diagnostics
        self.assertTrue(diagnostics["finite"])
        self.assertEqual(diagnostics["control_count"], len(self.source))
        self.assertEqual(diagnostics["affine_rank"], 4)
        json.dumps(diagnostics, allow_nan=False)

    def test_rejects_degenerate_controls(self):
        cases = [
            np.zeros((5, 3)),
            np.asarray([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
            np.vstack((self.source[:4], self.source[0])),
        ]
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "degenerate"):
                    fit_tps(source, source)

    def test_rejects_invalid_coordinates(self):
        for invalid in (np.ones((4, 2)), np.ones((3, 3)), np.full((4, 3), np.nan), np.full((4, 3), np.inf), np.ones((4, 3), dtype=complex)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    fit_tps(invalid, invalid)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            fit_tps(self.source, self.source[:-1])

    def test_rejects_invalid_smoothing_and_chunk_size(self):
        for smoothing in (-1, np.inf, np.nan, True, "0", None):
            with self.subTest(smoothing=smoothing):
                with self.assertRaisesRegex(ValueError, "smoothing"):
                    fit_tps(self.source, self.source, smoothing=smoothing)
        warp = fit_tps(self.source, self.source)
        for chunk_size in (0, -1, 1.5, True, None):
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                warp.transform(self.query, chunk_size=chunk_size)
        with self.assertRaisesRegex(ValueError, "finite"):
            warp.transform([[0, np.inf, 0]])


class SimilarityWarpTest(unittest.TestCase):
    def test_known_similarity_and_out_of_sample_transform(self):
        rng = np.random.default_rng(12)
        source = rng.normal(size=(20, 3))
        query = rng.normal(size=(14, 3))
        rotation = np.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        target = 1.8 * source @ rotation.T + [1.1, -0.3, 2.2]
        source_before, target_before = source.copy(), target.copy()
        warp = fit_similarity(source, target)
        np.testing.assert_allclose(warp.transform(query, chunk_size=3), 1.8 * query @ rotation.T + [1.1, -0.3, 2.2], atol=1e-12)
        self.assertAlmostEqual(warp.scale, 1.8)
        self.assertAlmostEqual(np.linalg.det(warp.rotation), 1)
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(target, target_before)
        self.assertEqual(warp.transform(np.empty((0, 3))).shape, (0, 3))
        json.dumps(warp.diagnostics, allow_nan=False)

    def test_non_collinear_planar_controls_work(self):
        source = np.asarray([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        target = 2 * source + [1, 2, 3]
        np.testing.assert_allclose(fit_similarity(source, target).transform(source), target, atol=1e-12)

    def test_reflection_is_not_a_valid_rotation(self):
        source = np.random.default_rng(2).normal(size=(40, 3))
        target = source * [-1, 1, 1]
        warp = fit_similarity(source, target)
        self.assertAlmostEqual(np.linalg.det(warp.rotation), 1)
        self.assertGreater(warp.diagnostics["control_rmse_after"], 0.1)

    def test_rejects_degeneracy_or_invalid_input(self):
        line = np.asarray([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        for source, target in ((line, line), (np.zeros((4, 3)), np.zeros((4, 3))), (line[:2], line[:2]), (line, line[:-1]), (line, line * np.nan)):
            with self.subTest(source=source, target=target):
                with self.assertRaises(ValueError):
                    fit_similarity(source, target)


class ControlSelectionTest(unittest.TestCase):
    def test_deterministic_spread_and_no_global_rng_changes(self):
        points = np.random.default_rng(3).normal(size=(500, 3))
        before = points.copy()
        state_before = np.random.get_state()
        first = select_control_indices(points, 16)
        second = select_control_indices(points, 16)
        state_after = np.random.get_state()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len(np.unique(first)), 16)
        np.testing.assert_array_equal(points, before)
        for left, right in zip(state_before, state_after):
            np.testing.assert_equal(left, right)
        np.testing.assert_array_equal(select_control_indices(points, 999), np.arange(len(points)))

    def test_ignores_exact_duplicate_locations(self):
        points = np.asarray([[0., 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(select_control_indices(points, 100), [0, 1, 3])
        np.testing.assert_array_equal(select_control_indices(np.zeros((4, 3)), 2), [0])

    def test_rejects_invalid_budget_and_points(self):
        for budget in (0, -1, 2.5, True, "4", None):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                select_control_indices(np.ones((3, 3)), budget)
        with self.assertRaises(ValueError):
            select_control_indices(np.empty((0, 3)), 5)


if __name__ == "__main__":
    unittest.main()
