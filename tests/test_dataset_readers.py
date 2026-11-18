import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scene.dataset_readers import (
    fetchPly,
    pointCloudFromPlyValues,
    storePly,
    SceneInfo,
    readNerfSyntheticInfo,
)


class DatasetReadersTest(unittest.TestCase):
    def test_legacy_scene_info_has_unknown_point_source(self):
        info = SceneInfo(None, [], [], {}, "legacy.ply")
        self.assertEqual(info.point_cloud_source, "unknown")

    def test_synthetic_random_and_existing_external_clouds_are_not_reconstruction(self):
        # Restore RNG state: the legacy random-cloud creation itself is
        # intentionally unchanged by the new provenance label.
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        camera = SimpleNamespace(image_name="train_only")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("scene.dataset_readers.readCamerasFromTransforms", return_value=[camera]), \
                mock.patch("scene.dataset_readers.getNerfppNorm", return_value={"radius": 1.0}):
            random_info = readNerfSyntheticInfo(directory, False, False, True)
            self.assertEqual(random_info.point_cloud_source, "random")
            self.assertFalse((Path(directory) / "points3d.ply").exists())
            generated_info = readNerfSyntheticInfo(directory, False, False, False)
            self.assertEqual(generated_info.point_cloud_source, "random")
            external_info = readNerfSyntheticInfo(directory, False, False, False)
            self.assertEqual(external_info.point_cloud_source, "external")
            np.testing.assert_array_equal(generated_info.point_cloud.points, external_info.point_cloud.points)

    def test_in_memory_point_cloud_matches_legacy_ply_round_trip(self):
        xyz = np.array(
            [
                [0.123456789, -0.25, 0.99999999],
                [-0.987654321, 0.75, -0.00000001],
            ],
            dtype=np.float64,
        )
        rgb = np.array(
            [[127.9, 128.1, 254.99], [0.99, 42.75, 200.5]],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            storePly(path, xyz, rgb)
            from_disk = fetchPly(path)

        in_memory = pointCloudFromPlyValues(xyz, rgb)
        np.testing.assert_array_equal(in_memory.points, from_disk.points)
        np.testing.assert_array_equal(in_memory.colors, from_disk.colors)
        np.testing.assert_array_equal(in_memory.normals, from_disk.normals)


if __name__ == "__main__":
    unittest.main()
