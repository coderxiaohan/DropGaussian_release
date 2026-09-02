import unittest

from utils.dataset_splits import (
    DTU_TEST_INDICES,
    dtu_split,
    nerf_synthetic_split,
    nerf_synthetic_test_split,
)


class DatasetSplitsTest(unittest.TestCase):
    def test_dtu_matches_stereogs_protocol(self):
        train, test = dtu_split(list(range(49)), 3)
        self.assertEqual(train, [25, 22, 28])
        self.assertEqual(test, list(DTU_TEST_INDICES))

    def test_nerf_synthetic_matches_stereogs_protocol(self):
        self.assertEqual(
            nerf_synthetic_split(list(range(100)), 8),
            [2, 16, 26, 55, 73, 76, 86, 93],
        )
        self.assertEqual(
            nerf_synthetic_test_split(list(range(100))), list(range(0, 100, 8))
        )

    def test_incomplete_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires index"):
            dtu_split(list(range(20)), 3)


if __name__ == "__main__":
    unittest.main()
