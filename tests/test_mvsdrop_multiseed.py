import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mvsdrop_multiseed import SCENES, collect


class MVSDropMultiSeedTest(unittest.TestCase):
    @staticmethod
    def _write_setting(root: Path, seed: int, mode: str) -> None:
        for scene in SCENES["llff"]:
            scene_root = root / f"seed_{seed}" / mode / "llff3" / scene
            scene_root.mkdir(parents=True)
            method = mode == "mvsdrop"
            scene_root.joinpath("metrics_10000.txt").write_text(
                "PSNR : {psnr}\nSSIM : {ssim}\nLPIPS : {lpips}\n".format(
                    psnr=21 if method else 20,
                    ssim=0.8 if method else 0.7,
                    lpips=0.1 if method else 0.2,
                ),
                encoding="utf-8",
            )
            scene_root.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "seed": seed,
                        "mask_seed": seed,
                        "n_views": 3,
                        "iterations": 10000,
                        "metricboost_mode": "dropgaussian",
                        "init_mode": "mvs_anywhere" if method else "colmap_dense",
                        "training_time": 11 if method else 10,
                        "peak_vram": 120 if method else 100,
                        "gaussian_count": 130 if method else 100,
                    }
                ),
                encoding="utf-8",
            )

    def test_complete_paired_seeds_report_metrics_and_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1, 2):
                self._write_setting(root, seed, "dropgaussian")
                self._write_setting(root, seed, "mvsdrop")
            rows, audit = collect(root, (1, 2))
            llff3 = rows[0]
            self.assertTrue(llff3["complete"])
            self.assertEqual(llff3["baseline_psnr"], 20.0)
            self.assertEqual(llff3["method_psnr"], 21.0)
            self.assertEqual(llff3["delta_psnr"], 1.0)
            self.assertEqual(llff3["delta_psnr_std"], 0.0)
            self.assertAlmostEqual(llff3["delta_lpips"], -0.1)
            self.assertTrue(llff3["setting_pass"])
            self.assertTrue(llff3["resource_pass"])
            self.assertFalse(audit["all_settings_complete"])
            self.assertIsNone(audit["goal_pass"])

    def test_incomplete_seed_is_not_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_setting(root, 1, "dropgaussian")
            self._write_setting(root, 1, "mvsdrop")
            missing = root / "seed_1/mvsdrop/llff3/fern/metrics_10000.txt"
            missing.unlink()
            rows, _ = collect(root, (1,))
            self.assertFalse(rows[0]["complete"])
            self.assertIsNone(rows[0]["delta_psnr"])


if __name__ == "__main__":
    unittest.main()
