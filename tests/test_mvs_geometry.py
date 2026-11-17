import copy
import json
import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import torch

from utils.graphics_utils import BasicPointCloud
from utils.mvs_geometry import (
    associate_control_points,
    cloud_digest,
    correct_mvs_geometry,
    projected_cloud_index,
)
from utils.train_view_controls import camera_geometry


def camera(name="train_a", translation=(0, 0, 0), image_size=(100, 80), calibration_size=None):
    width, height = calibration_size or image_size
    return SimpleNamespace(
        image_name=name,
        image=Image.new("RGB", image_size, (21, 34, 55)),
        image_path=None,
        width=width,
        height=height,
        K=np.asarray([[width / 2, 0., width / 2], [0., height * 0.625, height / 2], [0., 0., 1.]]),
        R=np.eye(3),
        T=np.asarray(translation, dtype=np.float64),
    )


def project(points, view):
    intrinsics, extrinsic = camera_geometry(view)
    coordinates = np.asarray(points) @ extrinsic[:, :3].T + extrinsic[:, 3]
    homogeneous = coordinates @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def tracks_for_targets(targets, cameras):
    pixels = [project(targets, view) for view in cameras]
    return [
        {
            "point": target.tolist(),
            "camera_indices": list(range(len(cameras))),
            "pixels": [image_pixels[index].tolist() for image_pixels in pixels],
            "reprojection_error_px": 0.0,
            "triangulation_angle_deg": 5.0,
        }
        for index, target in enumerate(targets)
    ]


class ProjectedCloudIndexTest(unittest.TestCase):
    def test_per_pixel_frontmost_and_tie_breaking(self):
        points = np.asarray([
            [0., 0, 2], [0., 0, 1], [0.001, 0, 1],
            [0., 0, -1], [20., 0, 1], [0., 0.2, 1], [0., 0, 0],
        ])
        pixels, indices, scale = projected_cloud_index(points, camera())
        np.testing.assert_array_equal(indices, [1, 5])
        np.testing.assert_allclose(pixels, [[50, 40], [50, 50]])
        self.assertEqual(scale, 1.0)

    def test_camera_rotation_translation_and_calibration_resize(self):
        view = camera(image_size=(100, 40), calibration_size=(200, 160), translation=(-0.2, 0.1, 0.5))
        world_to_camera = np.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        view.R = world_to_camera.T
        camera_point = np.asarray([[0.4, 0.2, 2.]])
        points = (camera_point - view.T) @ world_to_camera
        pixels, indices, scale = projected_cloud_index(points, view, max_image_size=50)
        self.assertEqual(scale, 0.5)
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_allclose(pixels, [[30, 11.25]])

    def test_numpy_images_have_same_projection_as_pil(self):
        view = camera()
        points = np.asarray([[0., 0, 2], [0.2, 0.1, 2]])
        expected = projected_cloud_index(points, view)
        view.image = np.asarray(view.image)
        actual = projected_cloud_index(points, view)
        for first, second in zip(actual, expected):
            np.testing.assert_equal(first, second)

    def test_empty_or_all_invisible_cloud(self):
        for points in (np.empty((0, 3)), np.asarray([[0., 0, -1], [100, 0, 1]])):
            pixels, indices, _ = projected_cloud_index(points, camera())
            self.assertEqual(pixels.shape, (0, 2))
            self.assertEqual(indices.shape, (0,))

    def test_rejects_invalid_cloud_and_image_resolution(self):
        for points in (np.ones((3, 2)), np.full((3, 3), np.nan), np.full((3, 3), np.inf)):
            with self.assertRaises(ValueError):
                projected_cloud_index(points, camera())
        for resolution in (0, -1, 1.5, "10"):
            with self.assertRaises(ValueError):
                projected_cloud_index(np.ones((3, 3)), camera(), max_image_size=resolution)


class AssociationTest(unittest.TestCase):
    def setUp(self):
        self.cameras = [camera(), camera("train_b", translation=(-0.3, 0, 0))]
        self.points = np.asarray([[-0.8, -0.6, 4], [0.6, -0.4, 4.3], [-0.4, 0.6, 5], [0.8, 0.7, 5.7]])

    def test_uses_training_indices_and_preserves_correspondence(self):
        tracks = tracks_for_targets(self.points, self.cameras)
        source, target, indices, metadata = associate_control_points(self.points, self.cameras, tracks)
        np.testing.assert_array_equal(indices, np.arange(4))
        np.testing.assert_array_equal(source, self.points)
        np.testing.assert_array_equal(target, self.points)
        self.assertEqual(metadata["accepted_observations"], 8)
        self.assertEqual(metadata["visible_mvs_points_per_camera"], [4, 4])
        json.dumps(metadata, allow_nan=False)

    def test_duplicate_source_keeps_best_reprojection_association(self):
        tracks = tracks_for_targets(self.points[:1], self.cameras)
        worse = copy.deepcopy(tracks[0])
        worse["point"][2] += 0.2
        worse["reprojection_error_px"] = 0.5
        source, target, indices, metadata = associate_control_points(self.points, self.cameras, [worse, tracks[0]])
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_array_equal(source, self.points[:1])
        np.testing.assert_array_equal(target, self.points[:1])
        self.assertEqual(metadata["accepted_observations"], 4)

    def test_association_radius_is_in_matching_resolution_pixels(self):
        for offset, expected_count in ((5, 1), (7, 0)):
            tracks = tracks_for_targets(self.points[:1], self.cameras)
            for pixel in tracks[0]["pixels"]:
                pixel[0] += offset
            _, _, indices, _ = associate_control_points(
                self.points[:1], self.cameras, tracks,
                max_image_size=50, match_radius_px=3,
            )
            self.assertEqual(len(indices), expected_count)

    def test_invalid_targets_pixels_lengths_and_non_training_views_fail(self):
        template = tracks_for_targets(self.points[:1], self.cameras)[0]
        cases = []
        for key, value in (
            ("point", [0, 1, np.nan]),
            ("pixels", [[1, np.nan], [1, 2]]),
            ("pixels", [[1, 2]]),
            ("camera_indices", [0, 2]),
            ("camera_indices", [-1, 1]),
        ):
            track = copy.deepcopy(template)
            track[key] = value
            cases.append(track)
        for track in cases:
            with self.subTest(track=track):
                with self.assertRaises(ValueError):
                    associate_control_points(self.points, self.cameras, [track])

    def test_malformed_multiview_tracks_are_not_accepted(self):
        template = tracks_for_targets(self.points[:1], self.cameras)[0]
        cases = []
        for indices, pixels in (([0], template["pixels"][:1]), ([0, 0], template["pixels"])):
            track = copy.deepcopy(template)
            track["camera_indices"], track["pixels"] = indices, pixels
            cases.append(track)
        for indices in ([0.5, 1], [False, 1]):
            track = copy.deepcopy(template)
            track["camera_indices"] = indices
            cases.append(track)
        for error in (np.nan, np.inf, -1):
            track = copy.deepcopy(template)
            track["reprojection_error_px"] = error
            cases.append(track)
        for track in cases:
            with self.subTest(track=track):
                with self.assertRaises(ValueError):
                    associate_control_points(self.points, self.cameras, [track])

    def test_no_nearby_observations_returns_empty_controls_not_synthetic_matches(self):
        source, target, indices, metadata = associate_control_points(self.points, self.cameras, [])
        self.assertEqual(source.shape, (0, 3))
        self.assertEqual(target.shape, (0, 3))
        self.assertEqual(indices.shape, (0,))
        self.assertEqual(metadata["accepted_observations"], 0)


class CorrectionTest(unittest.TestCase):
    def setUp(self):
        self.cameras = [camera(), camera("train_b", translation=(-0.3, 0, 0))]
        self.controls = np.asarray([
            [-0.8, -0.6, 4], [0.6, -0.4, 4.3], [-0.4, 0.6, 5],
            [0.8, 0.7, 5.7], [0.1, 0.1, 3.1], [-0.3, 0.2, 6.4],
        ], dtype=np.float32)
        points = np.vstack((self.controls, [[-50., 0, 5], [0, 0, -3]])).astype(np.float32)
        colors = np.arange(len(points) * 3, dtype=np.float32).reshape(-1, 3) / 24
        self.pcd = BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))

    def test_identity_preserves_count_order_colors_and_float32_hash(self):
        tracks = tracks_for_targets(self.controls, self.cameras)
        before = self.pcd.points.copy()
        with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {"test": True})) as matching:
            corrected, metadata, arrays = correct_mvs_geometry(self.pcd, self.cameras)
        self.assertIs(matching.call_args.args[0], self.cameras)
        self.assertIs(corrected.colors, self.pcd.colors)
        self.assertIs(corrected.normals, self.pcd.normals)
        np.testing.assert_array_equal(corrected.points, before)
        np.testing.assert_array_equal(self.pcd.points, before)
        np.testing.assert_array_equal(arrays["source_indices"], np.arange(6))
        np.testing.assert_array_equal(arrays["source"], self.pcd.points[arrays["source_indices"]])
        self.assertEqual(metadata["input_sha256"], metadata["output_sha256"])
        self.assertEqual(metadata["input_sha256"], cloud_digest(self.pcd.points, self.pcd.colors))
        self.assertEqual(metadata["point_count"], len(self.pcd.points))
        self.assertEqual(metadata["train_camera_names"], ["train_a", "train_b"])
        self.assertEqual(metadata["displacement_max"], 0)
        json.dumps(metadata, allow_nan=False)

    def test_known_similarity_corrects_all_points_not_only_controls(self):
        scale, offset = 1.03, np.asarray([0.01, -0.005, 0.02])
        targets = scale * self.controls.astype(np.float64) + offset
        tracks = tracks_for_targets(targets, self.cameras)
        for mode in ("similarity", "tps"):
            with self.subTest(mode=mode):
                with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
                    corrected, metadata, arrays = correct_mvs_geometry(self.pcd, self.cameras, mode=mode)
                expected = (scale * self.pcd.points.astype(np.float64) + offset).astype(np.float32)
                np.testing.assert_allclose(corrected.points, expected, atol=2e-6, rtol=1e-6)
                np.testing.assert_array_equal(corrected.colors, self.pcd.colors)
                np.testing.assert_allclose(arrays["transformed_controls"], arrays["target"], atol=1e-11)
                self.assertNotEqual(metadata["input_sha256"], metadata["output_sha256"])

    def test_known_nonrigid_controls_interpolate_without_changing_point_budget(self):
        targets = self.controls.astype(np.float64).copy()
        targets[:, 2] += 0.1 * targets[:, 0] ** 2
        tracks = tracks_for_targets(targets, self.cameras)
        with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
            corrected, metadata, arrays = correct_mvs_geometry(self.pcd, self.cameras, smoothing=0)
        self.assertEqual(corrected.points.shape, self.pcd.points.shape)
        np.testing.assert_allclose(arrays["transformed_controls"], arrays["target"], atol=1e-12)
        np.testing.assert_array_equal(corrected.colors, self.pcd.colors)
        self.assertEqual(metadata["selected_controls"], 6)

    def test_control_cap_and_source_indices_are_consistent(self):
        tracks = tracks_for_targets(self.controls, self.cameras)
        with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
            corrected, metadata, arrays = correct_mvs_geometry(self.pcd, self.cameras, max_controls=4)
        self.assertEqual(metadata["selected_controls"], 4)
        np.testing.assert_array_equal(arrays["source"], self.pcd.points[arrays["source_indices"]])
        np.testing.assert_array_equal(corrected.points, self.pcd.points)
        diagnostic = metadata["unused_control_diagnostic"]
        self.assertEqual(diagnostic["count"], 2)
        self.assertEqual(diagnostic["rmse_before"], 0)
        self.assertEqual(diagnostic["rmse_after"], 0)
        self.assertIn("not ground truth", diagnostic["note"])

    def test_correction_does_not_change_python_numpy_or_torch_rng(self):
        tracks = tracks_for_targets(self.controls, self.cameras)
        numpy_before, python_before, torch_before = np.random.get_state(), random.getstate(), torch.random.get_rng_state().clone()
        with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
            correct_mvs_geometry(self.pcd, self.cameras)
        for before, after in zip(numpy_before, np.random.get_state()):
            np.testing.assert_equal(before, after)
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_matching_failure_propagates_without_uncorrected_fallback(self):
        with mock.patch("utils.mvs_geometry.build_training_tracks", side_effect=RuntimeError("no usable training overlap")):
            with self.assertRaisesRegex(RuntimeError, "no usable training overlap"):
                correct_mvs_geometry(self.pcd, self.cameras)

    def test_consistent_modes_preserve_budget_rgb_rng_and_record_track_observations(self):
        targets = self.controls.astype(np.float64) + [0.01, -0.005, 0.02]
        tracks = tracks_for_targets(targets, self.cameras)
        for mode in ("tps_consistent", "tps_local"):
            with self.subTest(mode=mode):
                numpy_before = np.random.get_state()
                python_before = random.getstate()
                torch_before = torch.random.get_rng_state().clone()
                with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
                    corrected, metadata, arrays = correct_mvs_geometry(
                        self.pcd, self.cameras, mode=mode, local_neighbors=2)
                self.assertEqual(corrected.points.shape, self.pcd.points.shape)
                self.assertIs(corrected.colors, self.pcd.colors)
                self.assertIs(corrected.normals, self.pcd.normals)
                self.assertEqual(metadata["implementation"], "twings_inspired_fused_mvs_sift_v2")
                self.assertIn("warp_diagnostic", metadata)
                json.dumps(metadata, allow_nan=False)
                for array_index, track_index in enumerate(arrays["track_indices"]):
                    np.testing.assert_array_equal(arrays["target"][array_index], tracks[track_index]["point"])
                    np.testing.assert_array_equal(arrays["track_pixels"][array_index], tracks[track_index]["pixels"])
                np.testing.assert_array_equal(arrays["source"], self.pcd.points[arrays["source_indices"]])
                if mode == "tps_local":
                    # Far-away sample is retained unchanged, not pruned or globally translated.
                    np.testing.assert_array_equal(corrected.points[-2], self.pcd.points[-2])
                else:
                    self.assertFalse(np.array_equal(corrected.points[-2], self.pcd.points[-2]))
                for before, after in zip(numpy_before, np.random.get_state()):
                    np.testing.assert_equal(before, after)
                self.assertEqual(python_before, random.getstate())
                self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_insufficient_or_degenerate_controls_fail(self):
        tracks = tracks_for_targets(self.controls[:3], self.cameras)
        with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
            with self.assertRaisesRegex(ValueError, "No uncorrected fallback"):
                correct_mvs_geometry(self.pcd, self.cameras)
        source = np.asarray([[-1., -1, 5], [1, -1, 5], [-1, 1, 5], [1, 1, 5]])
        plane = BasicPointCloud(points=source, colors=np.ones_like(source), normals=np.zeros_like(source))
        tracks = tracks_for_targets(source, self.cameras)
        with mock.patch("utils.mvs_geometry.build_training_tracks", return_value=(tracks, {})):
            with self.assertRaisesRegex(ValueError, "non-coplanar"):
                correct_mvs_geometry(plane, self.cameras)

    def test_invalid_settings_and_duplicate_training_names_fail_before_matching(self):
        for kwargs in ({"mode": "none"}, {"mode": "adaptive"}, {"max_controls": 3}, {"max_controls": True}, {"smoothing": -1}, {"smoothing": np.nan}):
            with self.subTest(kwargs=kwargs):
                with mock.patch("utils.mvs_geometry.build_training_tracks") as matching:
                    with self.assertRaises(ValueError):
                        correct_mvs_geometry(self.pcd, self.cameras, **kwargs)
                    matching.assert_not_called()
        for cameras in (self.cameras[:1], [self.cameras[0], self.cameras[0]]):
            with self.assertRaisesRegex(ValueError, "distinct training cameras"):
                correct_mvs_geometry(self.pcd, cameras)


if __name__ == "__main__":
    unittest.main()
