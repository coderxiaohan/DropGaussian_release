import tempfile
import unittest
from pathlib import Path

import numpy as np

from scene.dataset_readers import (
    fetchPly,
    pointCloudFromPlyValues,
    storePly,
)


class DatasetReadersTest(unittest.TestCase):
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
