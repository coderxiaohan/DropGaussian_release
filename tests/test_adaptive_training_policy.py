import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AdaptiveTrainingPolicyTest(unittest.TestCase):
    def test_training_has_no_dataset_specific_mvs_policy(self):
        training_source = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
        scene_source = (REPO_ROOT / "scene/__init__.py").read_text(
            encoding="utf-8"
        )
        mvs_source = (REPO_ROOT / "utils/mvs_init.py").read_text(
            encoding="utf-8"
        )
        policy_source = training_source + scene_source + mvs_source
        forbidden = (
            "should_use_dtu_mvs_prior",
            "dtu_quality_gate",
            "synthetic_base_passthrough",
            "mvs_init_dtu",
        )
        for marker in forbidden:
            self.assertNotIn(marker, policy_source)


if __name__ == "__main__":
    unittest.main()
