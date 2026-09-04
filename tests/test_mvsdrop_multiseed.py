import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mvsdrop_multiseed import (
    SCENES,
    collect,
)


class MVSDropMultiSeedTest(unittest.TestCase):
    @staticmethod
    def _protocol(scene: str, seed: int, *, tag: str, views: int) -> dict:
        return {
            "dataset_name": tag,
            "source_path": f"/data/{tag}/{scene}",
            "resolution": 4 if tag == "dtu" else 8,
            "white_background": tag == "blender",
            "rand_pcd": tag == "blender",
            "n_views": views,
            "iterations": 10000,
            "seed": seed,
            "train_camera_names": [f"train_{index}" for index in range(views)],
            "test_camera_names": ["test_0"],
            "base_initialization_sha256": f"base-{tag}-{scene}-seed-{seed}",
        }

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
                        "git_commit": "frozen-test-commit",
                        "git_dirty": False,
                        "protocol": MVSDropMultiSeedTest._protocol(
                            scene, seed, tag="llff", views=3
                        ),
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

    def test_adaptive_setting_requires_its_own_measured_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in SCENES["dtu"]:
                scene_root = root / "seed_1/dropgaussian/dtu6" / scene
                scene_root.mkdir(parents=True)
                scene_root.joinpath("metrics_10000.txt").write_text(
                    "PSNR : 20\nSSIM : 0.8\nLPIPS : 0.1\n",
                    encoding="utf-8",
                )
                scene_root.joinpath("manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "seed": 1,
                            "mask_seed": 1,
                            "n_views": 6,
                            "iterations": 10000,
                            "metricboost_mode": "dropgaussian",
                            "init_mode": "colmap_dense",
                            "training_time": 10,
                            "peak_vram": 100,
                            "gaussian_count": 100,
                            "git_commit": "frozen-test-commit",
                            "git_dirty": False,
                            "protocol": self._protocol(
                                scene, 1, tag="dtu", views=6
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
            rows, _ = collect(root, (1,))
            dtu6 = next(
                row
                for row in rows
                if row["tag"] == "dtu" and row["views"] == 6
            )
            self.assertFalse(dtu6["complete"])
            self.assertEqual(dtu6["evaluation_route"], "trained_pair")
            self.assertIsNone(dtu6["delta_psnr"])

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

    def test_pair_rejects_different_base_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_setting(root, 1, "dropgaussian")
            self._write_setting(root, 1, "mvsdrop")
            manifest_path = root / "seed_1/mvsdrop/llff3/fern/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol"]["base_initialization_sha256"] = "different"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paired protocol mismatch"):
                collect(root, (1,))

    def test_dirty_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_setting(root, 1, "dropgaussian")
            manifest_path = root / "seed_1/dropgaussian/llff3/fern/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["git_dirty"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                collect(root, (1,))


if __name__ == "__main__":
    unittest.main()
