import copy
import json
import random
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from utils.mvs_control_consistency import associate_consistent_control_points
from utils.mvs_geometry import associate_control_points
from utils.train_view_controls import camera_geometry


def camera(translation=(0, 0, 0), size=(100, 80), calibration_size=None):
    width, height = calibration_size or size
    return SimpleNamespace(
        image=Image.new("RGB", size), image_name="training", image_path=None,
        K=np.array([[width / 2, 0, width / 2], [0, height * .625, height / 2], [0, 0, 1.]]),
        R=np.eye(3), T=np.asarray(translation, dtype=np.float64),
        width=width, height=height,
    )


def track(target, cameras):
    pixels = []
    for view in cameras:
        k, rt = camera_geometry(view)
        point = rt[:, :3] @ np.asarray(target) + rt[:, 3]
        homogeneous = k @ point
        pixels.append((homogeneous[:2] / homogeneous[2]).tolist())
    return dict(point=list(target), camera_indices=list(range(len(cameras))),
                pixels=pixels, reprojection_error_px=0.)


class ConsistentControlTests(unittest.TestCase):
    def setUp(self):
        self.cameras = [camera(), camera((-1, 0, 0))]

    def test_one_view_same_ray_trap_is_rejected(self):
        points = np.array([[0., 0, 2]])
        tracks = [track([0, 0, 5], self.cameras)]
        self.assertEqual(len(associate_control_points(points, self.cameras, tracks)[2]), 1)
        source, target, ids, metadata = associate_consistent_control_points(points, self.cameras, tracks)
        self.assertEqual(source.shape, (0, 3))
        self.assertEqual(target.shape, (0, 3))
        self.assertEqual(ids.shape, (0,))
        self.assertGreater(metadata["proposed_candidate_edges"], 0)
        self.assertEqual(metadata["accepted_multiview_candidate_edges"], 0)

    def test_valid_two_view_identity_and_order(self):
        points = np.array([[.8, .6, 4.], [-.8, -.4, 5], [.3, -.6, 5.5]])
        tracks = [track(point, self.cameras) for point in points[::-1]]
        source, target, ids, metadata = associate_consistent_control_points(points, self.cameras, tracks)
        np.testing.assert_array_equal(ids, [0, 1, 2])
        np.testing.assert_array_equal(source, points)
        np.testing.assert_array_equal(target, points)
        self.assertEqual(metadata["chosen_track_indices"], [2, 1, 0])
        self.assertEqual(metadata["chosen_max_reprojection_matching_px"], [0., 0., 0.])
        json.dumps(metadata, allow_nan=False)

    def test_second_nearest_candidate_can_replace_one_view_impostor(self):
        points = np.array([[0., 0, 2], [.08, 0, 5]])
        tracks = [track([0, 0, 5], self.cameras)]
        source, _, ids, _ = associate_consistent_control_points(points, self.cameras, tracks)
        np.testing.assert_array_equal(ids, [1])
        np.testing.assert_array_equal(source, points[1:])

    def test_alternative_edge_survives_preferred_source_collision(self):
        points = np.array([[0., 0, 5], [.12, 0, 5]])
        tracks = [track([0, 0, 5], self.cameras), track([.03, 0, 5], self.cameras)]
        _, targets, ids, metadata = associate_consistent_control_points(points, self.cameras, tracks)
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(targets, [[0, 0, 5], [.03, 0, 5]])
        self.assertEqual(metadata["chosen_track_indices"], [0, 1])
        self.assertEqual(metadata["accepted_pairs"], 2)

    def test_one_track_does_not_take_multiple_sources(self):
        points = np.array([[0., 0, 5], [.12, 0, 5]])
        _, _, ids, metadata = associate_consistent_control_points(points, self.cameras, [track([0, 0, 5], self.cameras)])
        np.testing.assert_array_equal(ids, [0])
        self.assertEqual(metadata["accepted_multiview_candidate_edges"], 2)

    def test_exact_target_duplicates_are_unique_even_across_tracks(self):
        points = np.array([[0., 0, 5], [.12, 0, 5]])
        tracks = [track([0, 0, 5], self.cameras)] * 2
        _, targets, ids, _ = associate_consistent_control_points(points, self.cameras, tracks)
        self.assertEqual(len(ids), 1)
        np.testing.assert_array_equal(targets, [[0, 0, 5]])

    def test_exact_source_duplicates_are_unique(self):
        points = np.array([[0., 0, 5], [0., 0, 5]])
        tracks = [track([0, 0, 5], self.cameras), track([.03, 0, 5], self.cameras)]
        _, _, ids, _ = associate_consistent_control_points(points, self.cameras, tracks)
        np.testing.assert_array_equal(ids, [0])

    def test_three_view_track_requires_all_three_not_just_first_pair(self):
        cameras = self.cameras + [camera((-2, 0, 0))]
        value = track([0, 0, 5], cameras)
        value["pixels"][2][0] += 10
        _, _, ids, _ = associate_consistent_control_points(np.array([[0., 0, 5]]), cameras, [value])
        self.assertEqual(len(ids), 0)

    def test_positive_depth_is_required_in_other_view(self):
        cameras = [camera(), camera((0, 0, -10))]
        value = track([0, 0, 5], cameras)
        _, _, ids, _ = associate_consistent_control_points(np.array([[0., 0, 5]]), cameras, [value])
        self.assertEqual(len(ids), 0)

    def test_image_bounds_are_required_in_other_view(self):
        cameras = [camera(), camera((-10, 0, 0))]
        value = track([0, 0, 5], cameras)
        _, _, ids, _ = associate_consistent_control_points(np.array([[0., 0, 5]]), cameras, [value])
        self.assertEqual(len(ids), 0)

    def test_radius_uses_matching_pixels_with_calibration_and_image_resizing(self):
        cameras = [camera(calibration_size=(200, 160)), camera((-1, 0, 0), calibration_size=(200, 160))]
        for offset, expected in ((5, 1), (7, 0)):
            value = track([0, 0, 5], cameras)
            for pixel in value["pixels"]:
                pixel[0] += offset
            _, _, ids, metadata = associate_consistent_control_points(
                np.array([[0., 0, 5]]), cameras, [value], max_image_size=50, match_radius_px=3)
            self.assertEqual(len(ids), expected)
            if expected:
                self.assertEqual(metadata["chosen_max_reprojection_matching_px"], [2.5])

    def test_empty_cloud_and_empty_tracks_produce_empty_pairs(self):
        for points, tracks in ((np.empty((0, 3)), [track([0, 0, 5], self.cameras)]),
                               (np.array([[0., 0, 5]]), [])):
            source, targets, ids, metadata = associate_consistent_control_points(points, self.cameras, tracks)
            self.assertEqual(source.shape, (0, 3))
            self.assertEqual(targets.shape, (0, 3))
            self.assertEqual(ids.shape, (0,))
            json.dumps(metadata, allow_nan=False)

    def test_reprojection_error_breaks_equal_geometry_ties(self):
        first = track([.01, 0, 5], self.cameras)
        second = copy.deepcopy(first)
        # Targets may differ despite coincident feature observations; lower
        # triangulation residual should win when association errors tie.
        second["point"][2] += .02
        first["reprojection_error_px"] = 1.
        _, targets, ids, metadata = associate_consistent_control_points(np.array([[0., 0, 5]]), self.cameras, [first, second])
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_array_equal(targets, [second["point"]])
        self.assertEqual(metadata["chosen_track_indices"], [1])

    def test_deterministic_and_does_not_modify_inputs_or_global_rng(self):
        points = np.array([[0., 0, 5], [.12, 0, 5]])
        tracks = [track([0, 0, 5], self.cameras), track([.03, 0, 5], self.cameras)]
        point_copy, track_copy = points.copy(), copy.deepcopy(tracks)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        first = associate_consistent_control_points(points, self.cameras, tracks)
        second = associate_consistent_control_points(points, self.cameras, tracks)
        for index in range(3):
            np.testing.assert_array_equal(first[index], second[index])
        self.assertEqual(first[3], second[3])
        np.testing.assert_array_equal(points, point_copy)
        self.assertEqual(tracks, track_copy)
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(before, after)
        self.assertEqual(python_state, random.getstate())

    def test_invalid_cloud_and_parameters_fail(self):
        for points in (np.ones((3, 2)), np.ones(3), np.full((2, 3), np.nan),
                       np.full((2, 3), np.inf), np.ones((2, 3), dtype=np.complex128)):
            with self.subTest(points=points), self.assertRaises(ValueError):
                associate_consistent_control_points(points, self.cameras, [])
        for key, values in (("max_image_size", (0, -1, 2.5, "4", True)),
                            ("match_radius_px", (0, -1, np.nan, np.inf, "4", True))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    associate_consistent_control_points(np.array([[0., 0, 5]]), self.cameras, [], **{key: value})

    def test_malformed_tracks_fail_before_association(self):
        template = track([0, 0, 5], self.cameras)
        bad_values = (
            ("point", [0, 0, np.nan]), ("point", [0, 5]), ("point", [0, 0, 1j]),
            ("camera_indices", [0, 0]), ("camera_indices", [0, 2]),
            ("camera_indices", [-1, 1]), ("camera_indices", [False, 1]),
            ("camera_indices", [0., 1]), ("camera_indices", [0]),
            ("pixels", [[50, 40]]), ("pixels", [[50, np.nan], [40, 40]]),
            ("reprojection_error_px", -1), ("reprojection_error_px", np.nan),
            ("reprojection_error_px", np.inf), ("reprojection_error_px", "1"),
        )
        for key, value in bad_values:
            malformed = copy.deepcopy(template)
            malformed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                associate_consistent_control_points(np.array([[0., 0, 5]]), self.cameras, [malformed])
        for malformed in ({}, None, {"point": [0, 0, 5]}, dict(template, camera_indices=None)):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                associate_consistent_control_points(np.array([[0., 0, 5]]), self.cameras, [malformed])


if __name__ == "__main__":
    unittest.main()
