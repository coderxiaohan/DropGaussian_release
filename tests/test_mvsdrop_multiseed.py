import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mvsdrop_multiseed import (
    SCENES,
    collect,
    is_final_policy_identity,
    route_dtu_quality_gate,
)


class MVSDropMultiSeedTest(unittest.TestCase):
    def test_final_policy_identity_settings(self):
        self.assertTrue(is_final_policy_identity("dtu", 6))
        self.assertTrue(is_final_policy_identity("dtu", 9))
        self.assertTrue(is_final_policy_identity("blender", 8))
        self.assertFalse(is_final_policy_identity("dtu", 3))
        self.assertFalse(is_final_policy_identity("llff", 9))

    def test_dtu_quality_gate_routes_measured_scene_branches(self):
        def row(scene, psnr, base_points=None, source_points=None):
            value = {
                "scene": scene,
                "psnr": psnr,
                "ssim": psnr / 100,
                "lpips": 1 / psnr,
                "training_time": psnr,
                "peak_vram": psnr,
                "gaussian_count": psnr,
                "initialization_source": None,
            }
            if base_points is not None:
                value["initialization_source"] = {
                    "base_point_count": base_points,
                    "source_point_count": source_points,
                }
            return value

        baseline = {
            "scenes": [row("use", 10), row("skip", 20)],
        }
        method = {
            "scenes": [
                row("use", 11, 9_000, 120_000),
                row("skip", 19, 20_000, 1_000_000),
            ],
        }
        routed = route_dtu_quality_gate(baseline, method, views=3)
        self.assertEqual(routed["routed_mvs_scenes"], ["use"])
        self.assertEqual(routed["psnr"], 15.5)
        dense_views = route_dtu_quality_gate(baseline, method, views=6)
        self.assertEqual(dense_views["routed_mvs_scenes"], [])
        self.assertEqual(dense_views["psnr"], 15.0)

    def test_dtu_quality_gate_rejects_mismatched_scene_sets(self):
        baseline = {"scenes": [{"scene": "scan8"}]}
        method = {"scenes": [{"scene": "scan21"}]}
        with self.assertRaisesRegex(ValueError, "identical.*scene sets"):
            route_dtu_quality_gate(baseline, method, views=3)

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

    def test_identity_passthrough_reuses_complete_baseline_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in SCENES["dtu"]:
                scene_root = root / "seed_1/dropgaussian/dtu6" / scene
                scene_root.mkdir(parents=True)
                scene_root.joinpath("metrics_30000.txt").write_text(
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
                            "iterations": 30000,
                            "metricboost_mode": "dropgaussian",
                            "init_mode": "colmap_dense",
                            "training_time": 10,
                            "peak_vram": 100,
                            "gaussian_count": 100,
                        }
                    ),
                    encoding="utf-8",
                )
            rows, _ = collect(root, (1,), replay_final_policy=True)
            dtu6 = next(
                row
                for row in rows
                if row["tag"] == "dtu" and row["views"] == 6
            )
            self.assertTrue(dtu6["complete"])
            self.assertEqual(dtu6["evaluation_route"], "identity_passthrough")
            self.assertEqual(dtu6["delta_psnr"], 0.0)
            self.assertEqual(dtu6["ratio_training_time"], 1.0)

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
