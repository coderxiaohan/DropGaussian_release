import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MVSFinalPruningTest(unittest.TestCase):
    def test_adaptive_initialization_has_no_metric_specific_final_pruning(self):
        train_source = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
        argument_source = (REPO_ROOT / "arguments/__init__.py").read_text(
            encoding="utf-8"
        )
        policy_source = train_source + argument_source
        self.assertNotIn("prune_to_opacity_budget", policy_source)
        self.assertNotIn("mvs_final_max_gaussians", policy_source)


if __name__ == "__main__":
    unittest.main()
