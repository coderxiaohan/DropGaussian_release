"""CPU tests for the camera geometry used by MVS complement selection."""

import unittest
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from PIL import Image

from utils.train_view_controls import camera_geometry, geometry_cameras_with_intrinsics


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


if __name__ == "__main__":
    unittest.main()
