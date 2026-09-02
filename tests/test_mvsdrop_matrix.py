import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mvsdrop_matrix import collect_matrix, markdown_table


class MVSDropMatrixTest(unittest.TestCase):
    def test_external_baseline_and_method_are_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "method"
            baseline = Path(directory) / "baseline"
            method_dir = root / "mvsdrop" / "llff3"
            baseline_dir = baseline / "llff3"
            method_dir.mkdir(parents=True)
            baseline_dir.mkdir(parents=True)
            baseline_dir.joinpath("metrics_mean.txt").write_text(
                "PSNR : 20\nSSIM : 0.7\nLPIPS : 0.2\n", encoding="utf-8"
            )
            method_dir.joinpath("metrics_mean.txt").write_text(
                "PSNR : 21\nSSIM : 0.8\nLPIPS : 0.1\n", encoding="utf-8"
            )
            for result_dir in (baseline_dir, method_dir):
                result_dir.joinpath("metrics_summary.json").write_text(
                    json.dumps({"scene_count": 8}), encoding="utf-8"
                )
            rows = collect_matrix(root, baseline)
            self.assertEqual(rows[0]["delta_psnr"], 1.0)
            self.assertAlmostEqual(rows[0]["delta_lpips"], -0.1)
            self.assertIn("MVSInit-DG", markdown_table(rows))

    def test_partial_aggregate_is_not_reported_as_full_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "mvsdrop" / "mipnerf36012"
            result_dir.mkdir(parents=True)
            result_dir.joinpath("metrics_mean.txt").write_text(
                "PSNR : 99\nSSIM : 0.99\nLPIPS : 0.01\n", encoding="utf-8"
            )
            result_dir.joinpath("metrics_summary.json").write_text(
                json.dumps({"scene_count": 1}), encoding="utf-8"
            )
            rows = collect_matrix(root)
            mip12 = next(
                row
                for row in rows
                if row["dataset"] == "Mip-NeRF360" and row["views"] == 12
            )
            self.assertIsNone(mip12["method_psnr"])

    def test_missing_external_baseline_falls_back_to_local_matched_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "matrix"
            external = Path(directory) / "older_baselines"
            local = root / "dropgaussian" / "llff6"
            local.mkdir(parents=True)
            external.mkdir(parents=True)
            local.joinpath("metrics_mean.txt").write_text(
                "PSNR : 24\nSSIM : 0.8\nLPIPS : 0.1\n", encoding="utf-8"
            )
            local.joinpath("metrics_summary.json").write_text(
                json.dumps({"scene_count": 8}), encoding="utf-8"
            )
            rows = collect_matrix(root, external)
            llff6 = next(
                row
                for row in rows
                if row["dataset"] == "LLFF" and row["views"] == 6
            )
            self.assertEqual(llff6["baseline_psnr"], 24.0)


if __name__ == "__main__":
    unittest.main()
