import json
import random
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from utils.mvs_surface_init import estimate_surface_frames, initialize_surface_covariance


def covariance_from_frames(factors, quaternions):
    matrices = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    return np.einsum("nik,nk,njk->nij", matrices, factors ** 2, matrices)


def fixture(points):
    points = np.asarray(points, dtype=np.float32)
    count = len(points)
    parameter = lambda value: torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32).clone())
    base = np.linspace(-4., -1., count, dtype=np.float32)
    rotation = np.zeros((count, 4), dtype=np.float32)
    rotation[:, 0] = 1.
    return SimpleNamespace(
        optimizer=None,
        _xyz=parameter(points),
        _scaling=parameter(np.repeat(base[:, None], 3, axis=1)),
        _rotation=parameter(rotation),
        _opacity=parameter(np.full((count, 1), -2.1972246, dtype=np.float32)),
        _features_dc=parameter(np.linspace(-.1, .1, count * 3).reshape(count, 1, 3)),
        _features_rest=parameter(np.zeros((count, 15, 3))),
    )


class SurfaceFramesTest(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(20).normal(size=(117, 3)) * [1., .6, .02]

    def test_planar_normal_is_smallest_axis_and_spd(self):
        xy = np.random.default_rng(2).normal(size=(200, 2))
        points = np.column_stack((xy, np.zeros(len(xy))))
        rotation = Rotation.from_rotvec([.3, -.6, .2]).as_matrix()
        points = points @ rotation.T
        factors, quaternions, metadata = estimate_surface_frames(points)
        normal = rotation[:, 2]
        covariance = covariance_from_frames(factors, quaternions)
        residual = np.einsum("nij,j->ni", covariance, normal) - factors[:, :1] ** 2 * normal
        np.testing.assert_allclose(residual, 0, atol=2e-14)
        self.assertTrue(np.all(factors[:, 0] < factors[:, 1]))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        np.testing.assert_allclose(np.trace(covariance, axis1=1, axis2=2), 3., atol=3e-14)
        self.assertEqual(metadata["effective_neighbors"], 32)

    def test_original_low_dimension_oas_formula(self):
        # All neighbors used: direct reference separates exact p=3 Eq.23 from
        # the high-dimensional approximation in sklearn's OAS implementation.
        points = self.points[:10]
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / len(points)
        trace = np.trace(covariance)
        square_trace = np.trace(covariance @ covariance)
        rho = min(1., ((1. - 2. / 3.) * square_trace + trace ** 2) /
                  ((len(points) + 1. - 2. / 3.) * (square_trace - trace ** 2 / 3.)))
        expected = ((1. - rho) * covariance + rho * trace / 3. * np.eye(3)) * 3. / trace
        factors, quaternions, metadata = estimate_surface_frames(points, neighbors=10)
        actual = covariance_from_frames(factors, quaternions)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=3e-14)
        self.assertAlmostEqual(metadata["shrinkage_coefficient"]["median"], rho)

    def test_isotropic_cloud_keeps_identity(self):
        points = np.concatenate((np.eye(3), -np.eye(3)))
        factors, rotations, metadata = estimate_surface_frames(points, neighbors=6)
        np.testing.assert_array_equal(factors, np.ones_like(factors))
        np.testing.assert_array_equal(rotations, np.tile([1., 0, 0, 0], (len(points), 1)))
        self.assertEqual(metadata["isotropic_count"], len(points))

    def test_degenerate_repeated_cloud_preserves_all_points(self):
        for count in (1, 2, 41):
            with self.subTest(count=count):
                points = np.tile([3., -1., 7.], (count, 1))
                factors, rotations, metadata = estimate_surface_frames(points)
                np.testing.assert_array_equal(factors, np.ones((count, 3)))
                np.testing.assert_array_equal(rotations, np.tile([1., 0, 0, 0], (count, 1)))
                self.assertEqual(metadata["unique_support_point_count"], 1)
                self.assertEqual(metadata["zero_covariance_count"], count)
                self.assertEqual(metadata["effective_neighbors"], 1)

    def test_two_distinct_points_are_valid_and_finite(self):
        factors, rotations, metadata = estimate_surface_frames([[0., 0, 0], [1, 0, 0]])
        self.assertEqual(metadata["effective_neighbors"], 2)
        self.assertTrue(np.isfinite(factors).all())
        self.assertTrue(np.all(factors > 0))
        np.testing.assert_allclose(np.sum(factors ** 2, axis=1), 3.)

    def test_duplicate_positions_do_not_manufacture_support(self):
        repeated = np.concatenate((self.points, self.points[[3] * 40], self.points[[4] * 50]))
        expected_factors, expected_rotations, _ = estimate_surface_frames(self.points)
        actual_factors, actual_rotations, metadata = estimate_surface_frames(repeated)
        np.testing.assert_array_equal(actual_factors[:len(self.points)], expected_factors)
        np.testing.assert_array_equal(actual_rotations[:len(self.points)], expected_rotations)
        np.testing.assert_array_equal(actual_factors[len(self.points):len(self.points) + 40],
                                      np.tile(expected_factors[3], (40, 1)))
        self.assertEqual(metadata["duplicate_position_count"], 90)

    def test_rigid_coordinate_frame_equivariance(self):
        rotation = Rotation.from_rotvec([.3, -.6, .2]).as_matrix()
        factors, quaternions, _ = estimate_surface_frames(self.points)
        expected = np.einsum("ij,njk,lk->nil", rotation, covariance_from_frames(factors, quaternions), rotation)
        factors, quaternions, _ = estimate_surface_frames(self.points @ rotation.T + [2., -1., 3.])
        np.testing.assert_allclose(covariance_from_frames(factors, quaternions), expected, atol=1e-12)

    def test_scene_unit_and_translation_invariance(self):
        factors, quaternions, _ = estimate_surface_frames(self.points)
        expected = covariance_from_frames(factors, quaternions)
        for scale in (1e-100, 1e-5, 3., 1e100):
            with self.subTest(scale=scale):
                factors, quaternions, _ = estimate_surface_frames((self.points + [2., -1., 3.]) * scale)
                np.testing.assert_allclose(covariance_from_frames(factors, quaternions), expected, atol=1e-12)

    def test_chunking_and_order_preserve_covariances(self):
        factors, rotations, _ = estimate_surface_frames(self.points, chunk_size=5)
        factors2, rotations2, _ = estimate_surface_frames(self.points, chunk_size=1000)
        np.testing.assert_array_equal(factors, factors2)
        np.testing.assert_array_equal(rotations, rotations2)
        permutation = np.random.default_rng(1).permutation(len(self.points))
        permuted_factors, permuted_rotations, _ = estimate_surface_frames(self.points[permutation])
        np.testing.assert_array_equal(permuted_factors, factors[permutation])
        np.testing.assert_array_equal(permuted_rotations, rotations[permutation])

    def test_no_input_mutation_and_no_global_rng_draws(self):
        before = self.points.copy()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        first = estimate_surface_frames(self.points)
        second = estimate_surface_frames(self.points)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(self.points, before)
        for left, right in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(left, right)
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_wxyz_quaternion_matches_gaussian_rotation_convention(self):
        factors, q, _ = estimate_surface_frames(self.points)
        w, x, y, z = q.T
        rotation = np.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ), axis=-1).reshape(-1, 3, 3)
        expected = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(rotation, expected, atol=2e-15)
        np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=2e-15)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1., atol=1e-15)

    def test_metadata_json_safe(self):
        _, _, metadata = estimate_surface_frames(self.points)
        json.dumps(metadata, allow_nan=False)
        self.assertLess(metadata["max_dimensionless_trace_error"], 1e-14)
        self.assertEqual(metadata["rotation_convention"], "wxyz_columns_are_world_space_principal_axes")

    def test_invalid_inputs(self):
        for points in ([], np.empty((0, 3)), np.zeros((4, 2)), np.full((4, 3), np.nan),
                       np.full((4, 3), np.inf), np.ones((4, 3), dtype=complex), [["1", "2", "3"]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                estimate_surface_frames(points)
        for neighbors in (0, 1, 2, -1, True, 3.5, "32", None):
            with self.subTest(neighbors=neighbors), self.assertRaisesRegex(ValueError, "neighbors"):
                estimate_surface_frames(self.points, neighbors=neighbors)
        for chunk_size in (0, -1, True, 3.5, "32", None):
            with self.subTest(chunk_size=chunk_size), self.assertRaisesRegex(ValueError, "chunk_size"):
                estimate_surface_frames(self.points, chunk_size=chunk_size)


class SurfaceApplicationTest(unittest.TestCase):
    def setUp(self):
        self.points = (np.random.default_rng(3).normal(size=(81, 3)) * [1., 1., .02]).astype(np.float32)
        self.gaussians = fixture(self.points)

    def test_only_scale_rotation_values_change_parameter_objects_remain(self):
        names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
        original = {name: getattr(self.gaussians, name) for name in names}
        before = {name: parameter.detach().clone() for name, parameter in original.items()}
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        metadata, arrays = initialize_surface_covariance(self.gaussians, self.points)
        for name in names:
            self.assertIs(getattr(self.gaussians, name), original[name])
            self.assertTrue(getattr(self.gaussians, name).requires_grad)
        for name in names[:4]:
            self.assertTrue(torch.equal(getattr(self.gaussians, name), before[name]))
            self.assertEqual(metadata["attributes_before_sha256"][name], metadata["attributes_after_sha256"][name])
        self.assertFalse(torch.equal(self.gaussians._scaling, before["_scaling"]))
        self.assertFalse(torch.equal(self.gaussians._rotation, before["_rotation"]))
        self.assertIsNone(self.gaussians.optimizer)
        self.assertLess(metadata["max_relative_covariance_trace_error"], 1e-6)
        np.testing.assert_allclose(np.sum(arrays["initialized_scales"].astype(np.float64) ** 2, axis=1),
                                   np.sum(arrays["base_scales"].astype(np.float64) ** 2, axis=1), rtol=1e-6)
        for left, right in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(left, right)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        json.dumps(metadata, allow_nan=False)

    def test_uses_actual_float32_model_positions(self):
        high_precision = self.points.astype(np.float64) + 1e-13
        first = fixture(self.points)
        second = fixture(self.points)
        metadata1, arrays1 = initialize_surface_covariance(first, self.points)
        metadata2, arrays2 = initialize_surface_covariance(second, high_precision)
        np.testing.assert_array_equal(arrays1["scale_factors"], arrays2["scale_factors"])
        np.testing.assert_array_equal(arrays1["rotations"], arrays2["rotations"])

    def test_rejects_existing_optimizer_without_modification(self):
        self.gaussians.optimizer = object()
        before = self.gaussians._scaling.detach().clone()
        with self.assertRaisesRegex(ValueError, "optimizer=None"):
            initialize_surface_covariance(self.gaussians, self.points)
        self.assertTrue(torch.equal(before, self.gaussians._scaling))

    def test_rejects_mismatched_point_order_and_shape(self):
        for points in (self.points[::-1], self.points[:-1], self.points + 1):
            with self.subTest(shape=points.shape), self.assertRaisesRegex(ValueError, "point count|match"):
                initialize_surface_covariance(self.gaussians, points)

    def test_rejects_already_anisotropic_scale(self):
        with torch.no_grad():
            self.gaussians._scaling[0, 1] += .2
        with self.assertRaisesRegex(ValueError, "isotropic"):
            initialize_surface_covariance(self.gaussians, self.points)

    def test_rejects_uninitialized_or_nonfinite_attributes(self):
        for name, value in (
            ("_scaling", torch.zeros((len(self.points), 3))),
            ("_rotation", torch.nn.Parameter(torch.zeros((len(self.points), 3)))),
            ("_opacity", torch.nn.Parameter(torch.full((len(self.points), 1), float("nan")))),
        ):
            with self.subTest(name=name):
                gaussians = fixture(self.points)
                setattr(gaussians, name, value)
                with self.assertRaises(ValueError):
                    initialize_surface_covariance(gaussians, self.points)

    def test_rejects_overflowed_or_underflowed_scales_before_writing(self):
        for value in (1000., -1000.):
            with self.subTest(value=value):
                gaussians = fixture(self.points)
                with torch.no_grad():
                    gaussians._scaling.fill_(value)
                before = gaussians._rotation.detach().clone()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    initialize_surface_covariance(gaussians, self.points)
                self.assertTrue(torch.equal(before, gaussians._rotation))


if __name__ == "__main__":
    unittest.main()
