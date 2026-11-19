"""DPP-only shell dispatch, using the shared fake Python pipeline without GPUs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


# Reuse the real shell fixture and fake train/render/evaluate/summarize program.
# Loading by file also supports direct unittest execution without a tests package.
_spec = importlib.util.spec_from_file_location(
    "ablation_script_fixtures", Path(__file__).with_name("test_ablation_scripts.py")
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def clean_environment():
    return {
        key: value for key, value in fixtures.clean_environment().items()
        if not key.startswith(("DPP_", "FASTGS_")) and key != "VCP_STRATEGY"
    }


class DPPOnlyLauncherTest(unittest.TestCase):
    run_script = fixtures.AblationLauncherTest.run_script
    assert_success = fixtures.AblationLauncherTest.assert_success
    assert_rejected_without_dispatch = fixtures.AblationLauncherTest.assert_rejected_without_dispatch

    def fixture(self, directory, dataset):
        repo, env, log = fixtures.AblationLauncherTest.fixture(self, directory, dataset)
        env = {key: value for key, value in env.items()
               if not key.startswith(("DPP_", "FASTGS_")) and key != "VCP_STRATEGY"}
        return repo, env, log

    def test_entrypoints_help_and_dry_runs_need_no_data_or_python(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in fixtures.DATASETS.items():
                with self.subTest(dataset=dataset):
                    path = fixtures.ROOT / fixtures.SCRIPTS / "dpp_only" / f"train_{dataset}.sh"
                    subprocess.run(["bash", "-n", str(path)], check=True)
                    env = dict(clean_environment(), OUTPUT_ROOT=str(output),
                               DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python")
                    help_result = self.run_script("dpp_only", dataset, env=env, args=("--help",))
                    self.assert_success(help_result)
                    self.assertIn("native ADC", help_result.stdout)
                    self.assertIn("multiview support", help_result.stdout)
                    result = self.run_script("dpp_only", dataset, env=env, args=("--dry-run",))
                    self.assert_success(result)
                    command, = fixtures.commands(result.stdout)
                    for value in ("MODE=dropgaussian", "DENSITY_CONTROL=vcp", "GPU_ID=0",
                                  f"N_VIEWS={views}", f"OUTPUT_ROOT={output}/dpp_only/{dataset}{views}",
                                  "--dpp_only", "--fastgs_protect_egd"):
                        self.assertIn(value, command)
                    self.assertEqual(fixtures.last_value(command, "--vcp_strategy"), "fastgs")
            matrix_path = fixtures.ROOT / fixtures.SCRIPTS / "dpp_only/train_all_views.sh"
            subprocess.run(["bash", "-n", str(matrix_path)], check=True)
            self.assertFalse(output.exists())

    def test_full_pipeline_all_datasets_preserves_base_initialization_and_protocol(self):
        for dataset, (views, resolution, scene) in fixtures.DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(GPU_ID="2", MVS_INIT_PATH="/missing/cache.npz",
                           MVS_INIT_STRATEGY="adaptive", MVS_COMPLEMENT_MAX_POINTS="invalid",
                           MVS_COMPLEMENT_MODE="none", MODE="dropgaussian", DENSITY_CONTROL="vcp")
                self.assert_success(self.run_script("dpp_only", dataset, repo=repo, env=env))
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = ["train.py", "render.py"]
                if dataset == "dtu":
                    expected.append("evaluate_dtu.py")
                expected.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], expected)
                args = calls[0]["args"]
                for flag, value in {
                    "--init_mode": "colmap_dense", "--density_control": "vcp",
                    "--vcp_strategy": "fastgs", "--fastgs_prune_interval": "8000",
                    "--fastgs_loss_threshold": "0.1", "--fastgs_prune_threshold": "0.9",
                    "--fastgs_opacity_threshold": "0.1", "--n_views": str(views),
                    "--dataset_name": dataset, "--iterations": "10000", "-r": str(resolution),
                }.items():
                    self.assertEqual(fixtures.last_value(args, flag), value)
                self.assertEqual(args.count("--dpp_only"), 1)
                self.assertEqual(args.count("--fastgs_protect_egd"), 1)
                self.assertFalse(any(arg.startswith("--mvs") for arg in args))
                self.assertEqual("--rand_pcd" in args, dataset == "blender")
                self.assertEqual("--white_background" in args, dataset == "blender")
                self.assertEqual(calls[0]["gpu"], "2")
                setting = repo / "output/ablation/dpp_only" / f"{dataset}{views}"
                self.assertEqual(Path(fixtures.last_value(args, "-m")), setting / scene)
                self.assertEqual(fixtures.last_value(calls[1]["args"], "-r"), str(resolution))
                self.assertEqual(fixtures.last_value(calls[1]["args"], "--iteration"), "10000")
                self.assertEqual(Path(fixtures.last_value(calls[-1]["args"], "--experiment-root")), setting)
                self.assertTrue((setting / "metrics_mean.txt").is_file())
                if dataset == "dtu":
                    self.assertEqual(fixtures.last_value(calls[2]["args"], "--scene"), scene)

    def test_custom_numeric_settings_paths_and_literal_arguments_reach_training(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            output = Path(directory) / "results with spaces"
            env.update(OUTPUT_ROOT=str(output), N_VIEWS="6", DPP_PRUNE_INTERVAL="4000",
                       DPP_LOSS_THRESHOLD="0.2", DPP_PRUNE_THRESHOLD=".8", DPP_OPACITY_THRESHOLD="1e-2")
            literal = "$(touch should_not_exist); * ' quoted"
            self.assert_success(self.run_script("dpp_only", repo=repo, env=env, cwd=directory,
                                                args=("--literal", literal)))
            args = json.loads(log.read_text().splitlines()[0])["args"]
            for flag, value in {
                "--fastgs_prune_interval": "4000", "--fastgs_loss_threshold": "0.2",
                "--fastgs_prune_threshold": ".8", "--fastgs_opacity_threshold": "1e-2",
                "--literal": literal, "-m": str(output / "dpp_only/llff6/fern"),
            }.items():
                self.assertEqual(fixtures.last_value(args, flag), value)
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_dpp_overrides_cannot_mislabel_any_ablation(self):
        for variant in (*fixtures.VARIANTS, "dpp_only"):
            for option in ("--dpp_only", "--dp", "--fastgs_protect_egd", "--fas",
                           "--fastgs_prune_interval", "--vcp_strategy"):
                for args in ((option, "invalid"), (option + "=invalid",)):
                    with self.subTest(variant=variant, args=args):
                        result = self.run_script(variant, env=dict(clean_environment(), DRY_RUN="1"), args=args)
                        self.assert_rejected_without_dispatch(result)
            for changes in ({"DPP_ONLY": "0"}, {"DPP_ONLY": "1"},
                            {"FASTGS_PROTECT_EGD": "0"}, {"FASTGS_PRUNE_INTERVAL": "4000"},
                            {"VCP_STRATEGY": "tbp"}):
                with self.subTest(variant=variant, changes=changes):
                    result = self.run_script(variant, env=dict(clean_environment(), DRY_RUN="1", **changes))
                    self.assert_rejected_without_dispatch(result)
            if variant != "dpp_only":
                result = self.run_script(variant, env=dict(clean_environment(), DRY_RUN="1",
                                                          DPP_PRUNE_INTERVAL="4000"))
                self.assert_rejected_without_dispatch(result)

    def test_dpp_policy_environment_and_numeric_conflicts_fail_before_dispatch(self):
        invalid = (
            {"MODE": "mvs"}, {"DENSITY_CONTROL": "vcd_vcp"}, {"MVS_COMPLEMENT_MODE": "coverage"},
            {"DPP_PRUNE_INTERVAL": "0"}, {"DPP_PRUNE_INTERVAL": "2.5"},
            {"DPP_LOSS_THRESHOLD": "NaN"}, {"DPP_LOSS_THRESHOLD": "-0.1"},
            {"DPP_PRUNE_THRESHOLD": "1.1"}, {"DPP_OPACITY_THRESHOLD": "1e999"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                result = self.run_script("dpp_only", env=dict(clean_environment(), DRY_RUN="1", **changes))
                self.assert_rejected_without_dispatch(result)

    def test_default_matrix_and_gpu_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            result = self.run_script("dpp_only", matrix=True, args=("--dry-run",), env=dict(
                clean_environment(), OUTPUT_ROOT=str(output), N_VIEWS="999", GPU_ID="7",
                GPU_LLFF="2", GPU_LLFF_6="4", GPU_DTU_9="5"))
            self.assert_success(result)
            preview = fixtures.commands(result.stdout)
            expected = (("llff", 3, 2), ("llff", 6, 4), ("llff", 9, 2),
                        ("dtu", 3, 7), ("dtu", 6, 7), ("dtu", 9, 5),
                        ("mipnerf360", 12, 7), ("mipnerf360", 24, 7), ("blender", 8, 7))
            self.assertEqual(len(preview), len(expected))
            for cmd, (dataset, views, gpu) in zip(preview, expected):
                for value in (f"N_VIEWS={views}", f"GPU_ID={gpu}",
                              f"OUTPUT_ROOT={output}/dpp_only/{dataset}{views}", "--dpp_only"):
                    self.assertIn(value, cmd)
            self.assertFalse(output.exists())

    def test_blender_over_eight_views_rejected_before_any_matrix_training(self):
        for views in ("9", "12", "99999999999999999999"):
            with self.subTest(views=views):
                result = self.run_script("dpp_only", "blender", env=dict(clean_environment(),
                                                                                DRY_RUN="1", N_VIEWS=views))
                self.assert_rejected_without_dispatch(result)
                self.assertIn("N_VIEWS <= 8", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env["MATRIX"] = "llff:3 blender:9"
            result = self.run_script("dpp_only", repo=repo, env=env, matrix=True)
            self.assert_rejected_without_dispatch(result)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "output").exists())
        # This stricter new entry point must not change the historical policy.
        self.assert_success(self.run_script("baseline", "blender", env=dict(
            clean_environment(), DRY_RUN="1", N_VIEWS="9")))

    def test_nonempty_output_is_preserved_and_training_failure_stops_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            output = Path(directory) / "results"
            setting = output / "dpp_only/llff3"
            setting.mkdir(parents=True)
            marker = setting / "keep.txt"
            marker.write_text("prior results", encoding="utf-8")
            env["OUTPUT_ROOT"] = str(output)
            self.assert_rejected_without_dispatch(self.run_script("dpp_only", repo=repo, env=env))
            self.assertEqual(marker.read_text(), "prior results")
            self.assertFalse(log.exists())
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3 llff:6", FAKE_FAIL_TRAIN="1")
            result = self.run_script("dpp_only", repo=repo, env=env, matrix=True)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])


if __name__ == "__main__":
    unittest.main()
