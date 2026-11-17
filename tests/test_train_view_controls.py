import random
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from utils.train_view_controls import (
    _extract_features,
    _image_and_mask,
    _mutual_ratio_matches,
    build_training_tracks,
    camera_geometry,
    geometry_cameras_with_intrinsics,
    triangulate_correspondences,
)


def camera(center=(0, 0, 0), image=None, **kwargs):
    if image is None:
        image = Image.new("RGB", (128, 128))
    result = SimpleNamespace(
        K=np.array([[120.0, 0, 64], [0, 120.0, 64], [0, 0, 1]]),
        R=np.eye(3), T=-np.asarray(center, dtype=np.float64),
        image=image, width=128, height=128, mask=None,
        image_path=None, image_name="training",
    )
    for name, value in kwargs.items():
        setattr(result, name, value)
    return result


def project(cam, points):
    k, rt = camera_geometry(cam)
    points = np.asarray(points, dtype=np.float64)
    xyz = points @ rt[:, :3].T + rt[:, 3]
    projected = xyz @ k.T
    return projected[:, :2] / projected[:, 2, None]


class TrainViewControlTests(unittest.TestCase):
    def test_colmap_simple_models_are_decoded_without_mutating_training_camera(self):
        for model, parameters in (
            ("SIMPLE_PINHOLE", [120, 61, 59]),
            ("SIMPLE_RADIAL", [120, 61, 59, 0.02]),
        ):
            with self.subTest(model=model):
                original = camera(uid=4, K=np.array([[120., 0, 59], [0, 61, .02], [0, 0, 1]]))
                camera_tuple = namedtuple("CameraFixture", vars(original))(**vars(original))
                before = camera_tuple.K.copy()
                record = SimpleNamespace(model=model, params=parameters, width=128, height=128)
                result, metadata = geometry_cameras_with_intrinsics([camera_tuple], {4: record})
                corrected = result[0]
                np.testing.assert_array_equal(corrected.K, [[120, 0, 61], [0, 120, 59], [0, 0, 1]])
                np.testing.assert_array_equal(camera_tuple.K, before)
                self.assertIsNot(corrected, camera_tuple)
                self.assertIs(corrected.image, camera_tuple.image)
                self.assertIs(corrected.R, camera_tuple.R)
                self.assertIs(corrected.T, camera_tuple.T)
                self.assertEqual(metadata["changed_camera_count"], 1)
                self.assertFalse(metadata["training_camera_objects_modified"])
                self.assertEqual(metadata["cameras"][0]["ignored_radial_coefficient"], .02 if model == "SIMPLE_RADIAL" else None)

    def test_colmap_pinhole_offcenter_intrinsics_and_fov_are_preserved(self):
        original = camera(uid=9, FovX=0.5, FovY=0.75, image=Image.new("RGB", (64, 32)))
        record = SimpleNamespace(model="PINHOLE", params=[121, 119, 58, 60], width=128, height=128)
        result, metadata = geometry_cameras_with_intrinsics([original], {9: record})
        corrected = result[0]
        np.testing.assert_array_equal(corrected.K, [[121, 0, 58], [0, 119, 60], [0, 0, 1]])
        actual_k, _ = camera_geometry(corrected)
        np.testing.assert_array_equal(actual_k, [[60.5, 0, 29], [0, 29.75, 15], [0, 0, 1]])
        self.assertEqual(corrected.FovX, original.FovX)
        self.assertEqual(corrected.FovY, original.FovY)
        self.assertEqual(metadata["cameras"][0]["actual_image_K"], actual_k.tolist())
        np.testing.assert_array_equal(original.K, [[120, 0, 64], [0, 120, 64], [0, 0, 1]])

    def test_colmap_intrinsic_id_not_image_name_selects_calibration(self):
        original = camera(uid=7, image_name="image_999")
        record = SimpleNamespace(model="PINHOLE", params=[120, 120, 64, 64], width=128, height=128)
        result, metadata = geometry_cameras_with_intrinsics([original], {7: record})
        self.assertEqual(metadata["changed_camera_count"], 0)
        self.assertIsNot(result[0], original)
        with self.assertRaisesRegex(ValueError, "absent"):
            geometry_cameras_with_intrinsics([original], {999: record})

    def test_invalid_colmap_calibration_fails(self):
        original = camera(uid=1)
        for settings in (
            {"model": "OPENCV"}, {"params": [0, 120, 64, 64]},
            {"params": [120, np.nan, 64, 64]}, {"params": [120, 120, 64]},
            {"width": 0}, {"height": 128.5},
        ):
            values = dict(model="PINHOLE", params=[120, 120, 64, 64], width=128, height=128)
            values.update(settings)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                geometry_cameras_with_intrinsics([original], {1: SimpleNamespace(**values)})

    def test_intrinsics_are_scaled_to_loaded_image_without_mutation(self):
        cam = camera(image=Image.new("RGB", (64, 32)))
        before = cam.K.copy()
        k, rt = camera_geometry(cam)
        np.testing.assert_allclose(k, [[60, 0, 32], [0, 30, 16], [0, 0, 1]])
        np.testing.assert_array_equal(cam.K, before)
        np.testing.assert_array_equal(rt[:, :3], np.eye(3))

    def test_stored_rotation_is_transposed(self):
        theta = 0.3
        rotation = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)],
        ])
        cam = camera(R=rotation.T, T=np.array([1, 2, 3]))
        _, rt = camera_geometry(cam)
        np.testing.assert_array_equal(rt[:, :3], rotation)
        np.testing.assert_array_equal(rt[:, 3], [1, 2, 3])

    def test_invalid_calibration_rejected(self):
        for settings in ({"width": 0}, {"K": np.zeros((3, 3))}, {"R": np.eye(3) * 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                camera_geometry(camera(**settings))

    def test_known_world_points_recovered(self):
        first, second = camera(), camera((1, 0, 0))
        expected = np.array([[0, 0, 4], [0.5, 0.2, 3], [-0.25, -0.4, 5]])
        points, valid, errors, angles = triangulate_correspondences(
            camera_geometry(first), camera_geometry(second),
            project(first, expected), project(second, expected),
        )
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, expected, atol=1e-12)
        self.assertTrue((errors < 1e-10).all())
        self.assertTrue((angles > 1).all())

    def test_cheirality_and_low_parallax_rejected(self):
        first, second = camera(), camera((1, 0, 0))
        expected = np.array([[0, 0, -4], [0, 0, 1e6], [0.5, 0, 4]])
        _, valid, _, _ = triangulate_correspondences(
            camera_geometry(first), camera_geometry(second),
            project(first, expected), project(second, expected),
        )
        np.testing.assert_array_equal(valid, [False, False, True])

    def test_bad_reprojection_and_nonfinite_match_rejected(self):
        first, second = camera(), camera((1, 0, 0))
        expected = np.array([[0, 0, 4], [0.5, 0.2, 3], [-0.25, -0.4, 5]])
        a, b = project(first, expected), project(second, expected)
        b[1, 1] += 20
        a[2] = np.nan
        _, valid, errors, _ = triangulate_correspondences(
            camera_geometry(first), camera_geometry(second), a, b,
        )
        np.testing.assert_array_equal(valid, [True, False, False])
        self.assertGreater(errors[1], 2)

    def test_empty_correspondences(self):
        geometry = camera_geometry(camera())
        points, valid, _, _ = triangulate_correspondences(geometry, geometry, np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(valid.shape, (0,))

    def test_alpha_and_explicit_mask_intersection(self):
        rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
        rgba[0, :, 3] = 0
        explicit = np.ones((4, 4), dtype=np.uint8)
        explicit[:, 0] = 0
        rgb, mask = _image_and_mask(camera(image=Image.fromarray(rgba), mask=explicit))
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertFalse(mask[0].any())
        self.assertFalse(mask[:, 0].any())
        self.assertTrue((mask[1:, 1:] == 255).all())

    def test_composited_rgb_can_recover_only_its_source_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.png"
            rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
            rgba[:, :2, 3] = 0
            Image.fromarray(rgba).save(path)
            _, mask = _image_and_mask(camera(image=Image.new("RGB", (2, 2)), image_path=str(path)))
            np.testing.assert_array_equal(mask, [[0, 255], [0, 255]])

    def test_resize_keypoints_map_back_to_original_pixels(self):
        class Sift:
            def detectAndCompute(self, gray, mask):
                self_shape = gray.shape
                if self_shape != (50, 100):
                    raise AssertionError(self_shape)
                return [SimpleNamespace(pt=(10.0, 20.0))], np.ones((1, 128), np.float32)

        import cv2
        with patch.object(cv2, "SIFT_create", return_value=Sift()):
            pixels, _, metadata = _extract_features(
                camera(image=Image.new("RGB", (200, 100))), cv2, 100, 100,
            )
        np.testing.assert_allclose(pixels, [[20.5, 40.5]])
        self.assertEqual(metadata["matching_size"], [100, 50])

    def test_descriptor_matches_require_mutual_ratio(self):
        import cv2
        first = np.eye(4, dtype=np.float32)
        second = first[[2, 0, 3, 1]]
        self.assertEqual(_mutual_ratio_matches(first, second, cv2, 0.75), [(0, 1), (1, 3), (2, 0), (3, 2)])
        ambiguous = np.ones((4, 4), dtype=np.float32)
        self.assertEqual(_mutual_ratio_matches(ambiguous, ambiguous, cv2, 0.75), [])
        self.assertEqual(_mutual_ratio_matches(first[:1], second, cv2, 0.75), [])

    def test_pairwise_tracks_only_use_supplied_camera_list_and_preserve_rng(self):
        cams = [camera(image_name="train_a"), camera((1, 0, 0), image_name="train_b")]
        expected = np.array([[0, 0, 4], [0.5, 0.2, 3]])
        descriptors = np.ones((2, 128), np.float32)
        features = [
            (project(cam, expected), descriptors, {"image_name": cam.image_name, "features": 2})
            for cam in cams
        ]
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with patch("utils.train_view_controls._extract_features", side_effect=features) as extract:
            with patch("utils.train_view_controls._mutual_ratio_matches", return_value=[(0, 0), (1, 1)]):
                tracks, metadata = build_training_tracks(cams)
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["camera_indices"], [0, 1])
        np.testing.assert_allclose([track["point"] for track in tracks], expected, atol=1e-12)
        self.assertEqual([item["image_name"] for item in metadata["cameras"]], ["train_a", "train_b"])
        self.assertFalse(metadata["matcher_is_twings_mast3r"])
        current_state = np.random.get_state()
        self.assertEqual(numpy_state[0], current_state[0])
        np.testing.assert_array_equal(numpy_state[1], current_state[1])
        self.assertEqual(numpy_state[2:], current_state[2:])
        self.assertEqual(python_state, random.getstate())

    def test_real_sift_on_translated_texture(self):
        rng = np.random.default_rng(19)
        texture = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        shifted = np.zeros_like(texture)
        shifted[:, :-15] = texture[:, 15:]
        first = camera(image=Image.fromarray(texture).convert("RGB"))
        second = camera((0.25, 0, 0), image=Image.fromarray(shifted).convert("RGB"))
        numpy_state, python_state = np.random.get_state(), random.getstate()
        tracks, metadata = build_training_tracks([first, second], max_features=500)
        self.assertGreater(len(tracks), 3)
        depths = np.array([track["point"][2] for track in tracks])
        self.assertLess(abs(np.median(depths) - 2), 0.03)
        self.assertEqual(metadata["pairs"][0]["accepted_tracks"], len(tracks))
        repeated, _ = build_training_tracks([first, second], max_features=500)
        self.assertEqual(tracks, repeated)
        current_state = np.random.get_state()
        np.testing.assert_array_equal(numpy_state[1], current_state[1])
        self.assertEqual(numpy_state[2:], current_state[2:])
        self.assertEqual(python_state, random.getstate())

    def test_empty_images_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "No valid train-view"):
            build_training_tracks([camera(), camera((1, 0, 0))])

    def test_invalid_parameters_fail_before_matching(self):
        for kwargs in ({"ratio": 1}, {"max_features": 0}, {"reprojection_px": -1},
                       {"min_triangulation_angle_deg": 0}, {"max_image_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_training_tracks([camera(), camera((1, 0, 0))], **kwargs)
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_training_tracks([camera()])


if __name__ == "__main__":
    unittest.main()
