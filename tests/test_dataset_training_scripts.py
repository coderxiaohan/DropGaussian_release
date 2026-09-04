import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
DATASETS = ("llff", "mipnerf360", "dtu", "blender")


class DatasetTrainingScriptsTest(unittest.TestCase):
    def test_all_dataset_scripts_have_valid_shell_syntax(self):
        paths = [SCRIPT_ROOT / f"train_{dataset}.sh" for dataset in DATASETS]
        paths += [
            SCRIPT_ROOT / f"train_mvsinit_{dataset}.sh"
            for dataset in DATASETS
        ]
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_baseline_scripts_force_original_initializer(self):
        for dataset in DATASETS:
            path = SCRIPT_ROOT / f"train_{dataset}.sh"
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("--init_mode colmap_dense", source)
                self.assertNotIn("--init_mode mvs_anywhere", source)

    def test_mvsinit_scripts_force_scene_adaptive_initializer(self):
        for dataset in DATASETS:
            path = SCRIPT_ROOT / f"train_mvsinit_{dataset}.sh"
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("MODE=mvsdrop", source)
                self.assertIn(f"DATASET={dataset}", source)
                self.assertIn("train_mvsdrop_benchmark.sh", source)
                self.assertIn("--init_mode mvs_anywhere", source)
                self.assertIn("--mvs_init_strategy auto", source)


if __name__ == "__main__":
    unittest.main()
