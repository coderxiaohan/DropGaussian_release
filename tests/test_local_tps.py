import json
import unittest

import numpy as np

from utils.local_tps import localize_tps, sampled_warp_diagnostics
from utils.tps_warp import fit_tps


class AffineWarp:
    def __init__(self, matrix=None, offset=None):
        self.matrix = np.eye(3) if matrix is None else np.asarray(matrix)
        self.offset = np.zeros(3) if offset is None else np.asarray(offset)

    def transform(self, points):
        return np.asarray(points) @ self.matrix.T + self.offset


class LocalTPSWarpTest(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(23)
        self.cloud = generator.normal(size=(90, 3))
        self.controls = self.cloud[:12].copy()
        self.target = self.controls + np.column_stack((
            0.03 * np.sin(self.controls[:, 1]),
            0.02 * self.controls[:, 0] ** 2,
            np.full(len(self.controls), 0.015),
        ))
        self.base = fit_tps(self.controls, self.target)
        self.local = localize_tps(self.base, self.cloud, self.controls)

    def test_anchor_support_is_one_and_base_warp_is_preserved(self):
        np.testing.assert_array_equal(self.local.support(self.controls), np.ones(len(self.controls)))
        np.testing.assert_array_equal(self.local.transform(self.controls), self.base.transform(self.controls))

    def test_far_points_are_exact_identity_without_evaluating_base(self):
        class RejectEvaluation:
            def transform(self, points):
                raise AssertionError("outside support must not evaluate TPS")

        localized = localize_tps(RejectEvaluation(), self.cloud, self.controls)
        outside = np.asarray([[1000., -2000, 3000], [-4000., 5000, -6000]])
        np.testing.assert_array_equal(localized.support(outside), [0, 0])
        np.testing.assert_array_equal(localized.transform(outside), outside)

    def test_duplicate_cloud_locations_do_not_change_radius(self):
        duplicated = np.vstack((self.cloud, self.cloud[:40], self.cloud[:10]))
        localized = localize_tps(self.base, duplicated, self.controls)
        np.testing.assert_array_equal(localized.radii, self.local.radii)
        np.testing.assert_array_equal(localized.support(self.cloud), self.local.support(self.cloud))
        self.assertEqual(localized.diagnostics["unique_source_point_count"], len(self.cloud))

    def test_radius_is_kth_nonself_unique_neighbor_and_small_cloud_caps_k(self):
        cloud = np.asarray([[0., 0, 0], [1., 0, 0], [3., 0, 0], [7., 0, 0]])
        localized = localize_tps(AffineWarp(), cloud, cloud[:1], neighbors=2)
        np.testing.assert_allclose(localized.radii, [3])
        capped = localize_tps(AffineWarp(), cloud, cloud[:1], neighbors=64)
        np.testing.assert_allclose(capped.radii, [7])
        self.assertEqual(capped.diagnostics["neighbors_effective"], 3)

    def test_wendland_weight_and_boundary(self):
        cloud = np.asarray([[0., 0, 0], [1., 0, 0]])
        localized = localize_tps(AffineWarp(offset=[0, 1, 0]), cloud, cloud[:1], neighbors=1)
        query = np.asarray([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]])
        np.testing.assert_allclose(localized.support(query), [1, 0.1875, 0, 0], atol=1e-15)
        expected = query.copy()
        expected[:, 1] += [1, 0.1875, 0, 0]
        np.testing.assert_allclose(localized.transform(query), expected)

    def test_union_support_uses_product_not_nearest_anchor(self):
        cloud = np.asarray([[-1., 0, 0], [1., 0, 0], [0., 1, 0]])
        localized = localize_tps(AffineWarp(), cloud, cloud[:2], neighbors=2)
        np.testing.assert_allclose(localized.support([[0., 0, 0]]), [1 - (1 - 0.1875) ** 2])

    def test_identity_and_chunking_preserve_values_order_and_inputs(self):
        identity = localize_tps(fit_tps(self.controls, self.controls), self.cloud, self.controls)
        np.testing.assert_array_equal(identity.transform(self.cloud), self.cloud)
        before = self.cloud.copy()
        np.testing.assert_allclose(self.local.transform(self.cloud, 1), self.local.transform(self.cloud, 100), atol=1e-14)
        np.testing.assert_array_equal(self.local.support(self.cloud, 1), self.local.support(self.cloud, 100))
        np.testing.assert_array_equal(self.cloud, before)
        self.assertEqual(self.local.transform(np.empty((0, 3))).shape, (0, 3))
        self.assertEqual(self.local.support(np.empty((0, 3))).shape, (0,))
        self.assertEqual(self.local.transform(self.cloud.astype(np.float32)).dtype, np.float64)

    def test_controls_and_radius_storage_do_not_alias_inputs(self):
        saved = self.local.transform(self.cloud)
        self.controls[:] = 1000
        np.testing.assert_array_equal(self.local.transform(self.cloud), saved)
        with self.assertRaises(ValueError):
            self.local.radii[0] = 99

    def test_rigid_and_uniform_scale_equivariance(self):
        theta = 0.61
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        expected = self.local.transform(self.cloud)
        for scale in (1e-5, 3., 1e5):
            with self.subTest(scale=scale):
                offset = np.asarray([4, -7, 2]) * scale
                cloud = scale * (self.cloud @ rotation.T) + offset
                controls = scale * (self.controls @ rotation.T) + offset
                targets = scale * (self.target @ rotation.T) + offset
                transformed = localize_tps(fit_tps(controls, targets), cloud, controls)
                np.testing.assert_allclose(
                    (transformed.transform(cloud) - offset) / scale,
                    expected @ rotation.T, atol=1e-11, rtol=1e-11,
                )
                np.testing.assert_allclose(transformed.radii / scale, self.local.radii, rtol=1e-13)

    def test_diagnostics_serialize_strict_json_and_do_not_claim_injectivity(self):
        json.dumps(self.local.diagnostics, allow_nan=False)
        self.assertFalse(self.local.diagnostics["injectivity_guaranteed"])
        self.assertEqual(self.local.diagnostics["neighbors_effective"], 64)
        self.assertEqual(self.local.diagnostics["base_fit"], self.base.diagnostics)
        self.assertIsNot(self.local.diagnostics["base_fit"], self.base.diagnostics)
        support = self.local.support(self.cloud)
        self.assertEqual(self.local.diagnostics["support_nonzero_fraction"], float(np.mean(support > 0)))
        self.assertEqual(self.local.diagnostics["support_gt_half_fraction"], float(np.mean(support > 0.5)))
        self.assertEqual(self.local.diagnostics["support_quantiles"]["median"], float(np.median(support)))

    def test_rejects_invalid_inputs(self):
        for invalid in (np.ones((3, 2)), np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.ones((3, 3), dtype=complex)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    localize_tps(self.base, invalid, self.controls)
                with self.assertRaises(ValueError):
                    localize_tps(self.base, self.cloud, invalid)
                with self.assertRaises(ValueError):
                    self.local.transform(invalid)
        with self.assertRaisesRegex(ValueError, "distinct source cloud"):
            localize_tps(self.base, np.zeros((5, 3)), np.zeros((1, 3)))
        with self.assertRaisesRegex(ValueError, "controls must be distinct"):
            localize_tps(self.base, self.cloud, self.controls[[0, 0]])
        with self.assertRaisesRegex(ValueError, "exact source_cloud"):
            localize_tps(self.base, self.cloud, self.controls + 0.01)
        with self.assertRaisesRegex(ValueError, "callable"):
            localize_tps(None, self.cloud, self.controls)
        for value in (0, -1, 1.5, True, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    localize_tps(self.base, self.cloud, self.controls, neighbors=value)
                with self.assertRaises(ValueError):
                    self.local.transform(self.cloud, chunk_size=value)
                with self.assertRaises(ValueError):
                    self.local.support(self.cloud, chunk_size=value)

    def test_rejects_nonfinite_or_wrong_shape_base_results(self):
        class BadWarp:
            def __init__(self, output):
                self.output = output

            def transform(self, points):
                return np.full((len(points), self.output), np.nan if self.output == 3 else 0.)

        for dimensions in (2, 3):
            with self.assertRaises(ValueError):
                localize_tps(BadWarp(dimensions), self.cloud, self.controls).transform(self.controls)


class SampledDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.cloud = np.random.default_rng(19).normal(size=(180, 3))

    def test_detects_reflection_and_reports_singular_values(self):
        diagnostics = sampled_warp_diagnostics(AffineWarp(np.diag([-1., 2, 3])), self.cloud, max_samples=31)
        self.assertEqual(diagnostics["sample_count"], 31)
        self.assertEqual(diagnostics["determinant_nonpositive_fraction"], 1)
        self.assertAlmostEqual(diagnostics["determinant_median"], -6., places=9)
        self.assertAlmostEqual(diagnostics["smallest_singular_value_min"], 1., places=9)
        self.assertAlmostEqual(diagnostics["largest_singular_value_max"], 3., places=9)
        self.assertIn("not_a_global", diagnostics["scope"])
        json.dumps(diagnostics, allow_nan=False)

    def test_detects_singular_mapping(self):
        diagnostics = sampled_warp_diagnostics(AffineWarp(np.diag([0., 1, 1])), self.cloud)
        self.assertEqual(diagnostics["determinant_nonpositive_count"], len(self.cloud))
        self.assertEqual(diagnostics["smallest_singular_value_min"], 0)

    def test_identity_is_deterministic_and_displacement_is_zero(self):
        first = sampled_warp_diagnostics(AffineWarp(), self.cloud, max_samples=29)
        second = sampled_warp_diagnostics(AffineWarp(), self.cloud, max_samples=29)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_displacement_max"], 0.)
        self.assertEqual(first["determinant_nonpositive_fraction"], 0.)
        self.assertAlmostEqual(first["determinant_median"], 1., places=9)

    def test_tps_and_local_tps_skip_control_neighborhoods(self):
        controls = self.cloud[:7]
        base = fit_tps(controls, controls + [0.01, 0, 0])
        local = localize_tps(base, self.cloud, controls)
        for warp in (base, local):
            with self.subTest(warp=type(warp).__name__):
                diagnostics = sampled_warp_diagnostics(warp, self.cloud, max_samples=1000)
                self.assertEqual(diagnostics["excluded_near_control_count"], 7)
                self.assertEqual(diagnostics["sample_count"], 173)
                json.dumps(diagnostics, allow_nan=False)

    def test_localized_jacobian_includes_support_derivative(self):
        cloud = np.asarray([[0., 0, 0], [1., 0, 0]])
        local = localize_tps(AffineWarp(offset=[3, 0, 0]), cloud, cloud[:1], neighbors=1)
        query = np.asarray([[0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0]])
        diagnostics = sampled_warp_diagnostics(local, query)
        # The base translation has identity Jacobian.  Negative determinants
        # therefore show the full localization derivative was included.
        self.assertEqual(diagnostics["determinant_nonpositive_count"], 2)
        self.assertEqual(diagnostics["active_sample_count"], 3)
        self.assertEqual(diagnostics["nonpositive_fraction_among_active_support"], 2 / 3)
        self.assertLess(diagnostics["minimum_determinant_among_active_support"], 0)
        self.assertAlmostEqual(diagnostics["determinant_median"], -2.75, places=7)

    def test_outside_local_support_does_not_fabricate_active_fold_rate(self):
        local = localize_tps(AffineWarp(offset=[3, 0, 0]), self.cloud, self.cloud[:8])
        diagnostics = sampled_warp_diagnostics(local, self.cloud + 1000.)
        self.assertEqual(diagnostics["active_sample_count"], 0)
        self.assertEqual(diagnostics["active_sample_fraction"], 0.)
        self.assertIsNone(diagnostics["nonpositive_fraction_among_active_support"])
        self.assertIsNone(diagnostics["minimum_determinant_among_active_support"])
        self.assertEqual(diagnostics["determinant_nonpositive_fraction"], 0.)
        json.dumps(diagnostics, allow_nan=False)

    def test_diagnostic_step_scales_with_scene_units(self):
        first = sampled_warp_diagnostics(AffineWarp(), self.cloud)
        second = sampled_warp_diagnostics(AffineWarp(), 1e4 * self.cloud)
        self.assertAlmostEqual(second["step"] / first["step"], 1e4, places=8)
        self.assertAlmostEqual(second["determinant_median"], first["determinant_median"], places=9)

    def test_invalid_points_sample_counts_and_warps_fail(self):
        for invalid in (np.ones((3, 2)), np.ones((1, 3)), np.ones((5, 3)), np.full((5, 3), np.nan)):
            with self.assertRaises(ValueError):
                sampled_warp_diagnostics(AffineWarp(), invalid)
        for count in (0, -1, 1.5, True, None):
            with self.assertRaises(ValueError):
                sampled_warp_diagnostics(AffineWarp(), self.cloud, max_samples=count)
        with self.assertRaises(ValueError):
            sampled_warp_diagnostics(None, self.cloud)

    def test_only_controls_report_unavailable_without_fabricating_jacobians(self):
        controls = self.cloud[:8]
        diagnostics = sampled_warp_diagnostics(fit_tps(controls, controls), controls)
        self.assertEqual(diagnostics["status"], "unavailable_no_noncontrol_samples")
        self.assertEqual(diagnostics["sample_count"], 0)
        self.assertEqual(diagnostics["excluded_near_control_count"], len(controls))
        self.assertNotIn("determinant_median", diagnostics)
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
