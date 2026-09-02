import unittest

import torch

from train import prune_to_opacity_budget


class _FakeGaussians:
    def __init__(self):
        self._opacity = torch.tensor([[0.1], [0.9], [0.3], [0.8]])
        self._xyz = torch.arange(12).reshape(4, 3)

    @property
    def get_xyz(self):
        return self._xyz

    def prune_points(self, mask):
        self._opacity = self._opacity[~mask]
        self._xyz = self._xyz[~mask]


class MVSFinalPruningTest(unittest.TestCase):
    def test_keeps_highest_opacity_gaussians(self):
        gaussians = _FakeGaussians()
        metadata = prune_to_opacity_budget(gaussians, 2)
        self.assertEqual(metadata["before"], 4)
        self.assertEqual(metadata["after"], 2)
        kept = sorted(gaussians._opacity.squeeze().tolist())
        for actual, expected in zip(kept, [0.8, 0.9]):
            self.assertAlmostEqual(actual, expected)

    def test_noop_below_budget(self):
        gaussians = _FakeGaussians()
        metadata = prune_to_opacity_budget(gaussians, 8)
        self.assertEqual(metadata["after"], 4)
        self.assertEqual(len(gaussians.get_xyz), 4)


if __name__ == "__main__":
    unittest.main()
