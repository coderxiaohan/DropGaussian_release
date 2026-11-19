"""EGD + DPP ablation dispatch, exercised with fake Python and no GPUs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


# Reuse the shell fixture and fake train/render/evaluate/summarize pipeline.
# Loading by path also supports direct execution without a tests package.
_spec = importlib.util.spec_from_file_location(
    "egd_dpp_script_fixtures", Path(__file__).with_name("test_ablation_scripts.py")
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

VARIANT = "egd_dpp"
DPP_DEFAULTS = {
    "--vcp_strategy": "fastgs",
    "--fastgs_prune_interval": "8000",
    "--fastgs_loss_threshold": "0.1",
    "--fastgs_prune_threshold": "0.9",
    "--fastgs_opacity_threshold": "0.1",
}


def clean_environment():
    return {
        key: value for key, value in fixtures.clean_environment().items()
        if not key.startswith(("DPP_", "FASTGS_", "TBP_")) and key != "VCP_STRATEGY"
    }


class EGDDPPLauncherTest(unittest.TestCase):
    run_script = fixtures.AblationLauncherTest.run_script
    assert_success = fixtures.AblationLauncherTest.assert_success
    assert_rejected_without_dispatch = fixtures.AblationLauncherTest.assert_rejected_without_dispatch

    def fixture(self, directory, dataset):
        repo, env, log = fixtures.AblationLauncherTest.fixture(self, directory, dataset)
        env = {key: value for key, value in env.items()
               if not key.startswith(("DPP_", "FASTGS_", "TBP_")) and key != "VCP_STRATEGY"}
        return repo, env, log

    def assert_egd_dpp_policy(self, args):
        self.assertEqual(fixtures.last_value(args, "--vcp_strategy"), "fastgs")
        self.assertEqual(args.count("--fastgs_protect_egd"), 1)
        self.assertNotIn("--dpp_only", args)
        self.assertFalse(any(arg.startswith(("--mvs", "--tbp")) for arg in args))

    def test_entrypoints_have_valid_syntax_and_help_needs_no_data_or_python(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset in fixtures.DATASETS:
                with self.subTest(dataset=dataset):
                    path = fixtures.ROOT / fixtures.SCRIPTS / VARIANT / f"train_{dataset}.sh"
                    self.assertTrue(path.is_file(), path)
                    subprocess.run(["bash", "-n", str(path)], check=True)
                    result = self.run_script(VARIANT, dataset, args=("--help",), env=dict(
                        clean_environment(), OUTPUT_ROOT=str(output), DATA_ROOT="/missing-data",
                        PYTHON_BIN="/missing-python"))
                    self.assert_success(result)
                    for text in (VARIANT, "DPP_PRUNE_INTERVAL", "OUTPUT_ROOT", "GPU"):
                        self.assertIn(text, result.stdout)
            matrix_path = fixtures.ROOT / fixtures.SCRIPTS / VARIANT / "train_all_views.sh"
            self.assertTrue(matrix_path.is_file(), matrix_path)
            subprocess.run(["bash", "-n", str(matrix_path)], check=True)
            result = self.run_script(VARIANT, matrix=True, args=("--help",), env=dict(
                clean_environment(), OUTPUT_ROOT=str(output), DATA_ROOT="/missing-data",
                PYTHON_BIN="/missing-python"))
            self.assert_success(result)
            self.assertIn("serial", result.stdout)
            self.assertFalse(output.exists())

    def test_dry_runs_fix_egd_dpp_and_default_view_counts_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in fixtures.DATASETS.items():
                for args, preview_env in ((("--dry-run",), {}), ((), {"DRY_RUN": "1"})):
                    with self.subTest(dataset=dataset, args=args):
                        env = dict(clean_environment(), OUTPUT_ROOT=str(output),
                                   DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python", **preview_env)
                        result = self.run_script(VARIANT, dataset, env=env, args=args)
                        self.assert_success(result)
                        command, = fixtures.commands(result.stdout)
                        for value in ("MODE=dropgaussian", "DENSITY_CONTROL=vcd_vcp", "GPU_ID=0",
                                      f"N_VIEWS={views}", f"OUTPUT_ROOT={output}/{VARIANT}/{dataset}{views}"):
                            self.assertIn(value, command)
                        self.assert_egd_dpp_policy(command)
                        for flag, value in DPP_DEFAULTS.items():
                            self.assertEqual(fixtures.last_value(command, flag), value)
            self.assertFalse(output.exists())

    def test_full_pipeline_preserves_base_initialization_and_dataset_protocol(self):
        for dataset, (views, resolution, scene) in fixtures.DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(GPU_ID="2", MVS_INIT_PATH="/missing/cache.npz",
                           MVS_INIT_STRATEGY="adaptive", MVS_COMPLEMENT_MAX_POINTS="invalid",
                           MVS_COMPLEMENT_MODE="none", MODE="dropgaussian", DENSITY_CONTROL="vcd_vcp")
                self.assert_success(self.run_script(VARIANT, dataset, repo=repo, env=env))
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = ["train.py", "render.py"]
                if dataset == "dtu":
                    expected.append("evaluate_dtu.py")
                expected.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], expected)
                args = calls[0]["args"]
                self.assert_egd_dpp_policy(args)
                for flag, value in dict(DPP_DEFAULTS, **{
                    "--init_mode": "colmap_dense", "--density_control": "vcd_vcp",
                    "--n_views": str(views), "--dataset_name": dataset,
                    "--iterations": "10000", "-r": str(resolution),
                }).items():
                    self.assertEqual(fixtures.last_value(args, flag), value)
                self.assertEqual("--rand_pcd" in args, dataset == "blender")
                self.assertEqual("--white_background" in args, dataset == "blender")
                self.assertEqual(calls[0]["gpu"], "2")
                setting = repo / "output/ablation" / VARIANT / f"{dataset}{views}"
                self.assertEqual(Path(fixtures.last_value(args, "-m")), setting / scene)
                self.assertEqual(fixtures.last_value(calls[1]["args"], "-r"), str(resolution))
                self.assertEqual(fixtures.last_value(calls[1]["args"], "--iteration"), "10000")
                self.assertEqual(Path(fixtures.last_value(calls[-1]["args"], "--experiment-root")), setting)
                self.assertTrue((setting / "metrics_mean.txt").is_file())
                if dataset == "dtu":
                    self.assertEqual(fixtures.last_value(calls[2]["args"], "--scene"), scene)
                    self.assertEqual(Path(fixtures.last_value(calls[2]["args"], "--model-path")),
                                     setting / scene)

    def test_default_scene_lists_are_not_reduced(self):
        for dataset in fixtures.DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.pop("SCENES")
                for scene in fixtures.ALL_SCENES[dataset].split():
                    name = scene + "_colmap" if dataset == "dtu" else scene
                    (Path(env["DATA_ROOT"]) / name).mkdir(exist_ok=True)
                self.assert_success(self.run_script(VARIANT, dataset, repo=repo, env=env))
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                trained_scenes = [Path(fixtures.last_value(call["args"], "-m")).name
                                  for call in calls if call["kind"] == "train.py"]
                self.assertEqual(trained_scenes, fixtures.ALL_SCENES[dataset].split())

    def test_custom_numeric_settings_and_literal_arguments_reach_training(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            output = Path(directory) / "results with spaces"
            env.update(OUTPUT_ROOT=str(output), N_VIEWS="6", ITERATIONS="1000", RESOLUTION="4",
                       DPP_PRUNE_INTERVAL="4000", DPP_LOSS_THRESHOLD="0.2",
                       DPP_PRUNE_THRESHOLD=".8", DPP_OPACITY_THRESHOLD="1e-2")
            literal = "$(touch should_not_exist); * ' quoted"
            self.assert_success(self.run_script(VARIANT, repo=repo, env=env, cwd=directory,
                                                args=("--literal", literal, "--empty", "")))
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            args = calls[0]["args"]
            self.assert_egd_dpp_policy(args)
            for flag, value in {
                "--fastgs_prune_interval": "4000", "--fastgs_loss_threshold": "0.2",
                "--fastgs_prune_threshold": ".8", "--fastgs_opacity_threshold": "1e-2",
                "--literal": literal, "--empty": "", "-m": str(output / VARIANT / "llff6/fern"),
                "--n_views": "6", "--iterations": "1000", "-r": "4",
            }.items():
                self.assertEqual(fixtures.last_value(args, flag), value)
            self.assertEqual(fixtures.last_value(calls[1]["args"], "--iteration"), "1000")
            self.assertEqual(fixtures.last_value(calls[1]["args"], "-r"), "4")
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_method_and_protocol_overrides_fail_before_dispatch(self):
        for option in ("--dpp_only", "--dp", "--fastgs_protect_egd", "--fas",
                       "--fastgs_prune_interval", "--vcp_strategy", "--vcp_strat",
                       "--vcd_error_quantile", "--density_control", "--density_con",
                       "--init_mode", "--init_m", "--mvs_init_path", "--mv",
                       "--rand_pcd", "--white_background", "--n_views", "--n_v",
                       "--iterations", "--iter", "--dataset_name", "-m", "-s", "-r"):
            for args in ((option, "invalid"), (option + "=invalid",)):
                with self.subTest(args=args):
                    result = self.run_script(VARIANT, env=dict(clean_environment(), DRY_RUN="1"), args=args)
                    self.assert_rejected_without_dispatch(result)
        self.assert_rejected_without_dispatch(self.run_script(
            VARIANT, args=("--dry-run", "--", "--dpp_only"), env=clean_environment()))

    def test_conflicting_method_environment_is_rejected(self):
        for changes in ({"MODE": "mvs"}, {"MODE": "both"}, {"DENSITY_CONTROL": "vcp"},
                        {"DENSITY_CONTROL": "vcd"}, {"MVS_COMPLEMENT_MODE": "coverage"},
                        {"DPP_ONLY": "0"}, {"DPP_ONLY": "1"}, {"DPP_UNDOCUMENTED": "1"},
                        {"FASTGS_PROTECT_EGD": "0"}, {"FASTGS_PRUNE_INTERVAL": "4000"},
                        {"VCP_STRATEGY": "tbp"}, {"VCP_STRATEGY": "fastgs"}):
            with self.subTest(changes=changes):
                self.assert_rejected_without_dispatch(self.run_script(
                    VARIANT, env=dict(clean_environment(), DRY_RUN="1", **changes)))

    def test_numeric_validation_and_allowed_threshold_boundaries(self):
        invalid = (
            {"DPP_PRUNE_INTERVAL": "0"}, {"DPP_PRUNE_INTERVAL": "-1"},
            {"DPP_PRUNE_INTERVAL": "2.5"}, {"DPP_PRUNE_INTERVAL": "1e3"},
            {"N_VIEWS": "0"}, {"N_VIEWS": "3,6"}, {"GPU_ID": "0,1"},
            {"ITERATIONS": "0"}, {"RESOLUTION": "0"}, {"DRY_RUN": "yes"},
        )
        invalid += tuple({key: value} for key in (
            "DPP_LOSS_THRESHOLD", "DPP_PRUNE_THRESHOLD", "DPP_OPACITY_THRESHOLD"
        ) for value in ("NaN", "inf", "1e999", "-0.1", "1.1", "2e-"))
        for changes in invalid:
            with self.subTest(changes=changes):
                env = dict(clean_environment(), DRY_RUN="1")
                env.update(changes)
                self.assert_rejected_without_dispatch(self.run_script(VARIANT, env=env))
        for value in ("0", "1", ".5", "1e-2"):
            with self.subTest(valid_threshold=value):
                self.assert_success(self.run_script(VARIANT, env=dict(
                    clean_environment(), DRY_RUN="1", DPP_PRUNE_INTERVAL="1",
                    DPP_LOSS_THRESHOLD=value, DPP_PRUNE_THRESHOLD=value, DPP_OPACITY_THRESHOLD=value)))

    def test_default_matrix_has_nine_settings_and_gpu_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            result = self.run_script(VARIANT, matrix=True, args=("--dry-run",), env=dict(
                clean_environment(), OUTPUT_ROOT=str(output), N_VIEWS="999", GPU_ID="7",
                GPU_LLFF="2", GPU_LLFF_6="4", GPU_DTU_9="5",
                DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python"))
            self.assert_success(result)
            preview = fixtures.commands(result.stdout)
            expected = (("llff", 3, 2), ("llff", 6, 4), ("llff", 9, 2),
                        ("dtu", 3, 7), ("dtu", 6, 7), ("dtu", 9, 5),
                        ("mipnerf360", 12, 7), ("mipnerf360", 24, 7), ("blender", 8, 7))
            self.assertEqual(len(preview), len(expected))
            for command, (dataset, views, gpu) in zip(preview, expected):
                for value in (f"N_VIEWS={views}", f"GPU_ID={gpu}", "DENSITY_CONTROL=vcd_vcp",
                              f"OUTPUT_ROOT={output}/{VARIANT}/{dataset}{views}"):
                    self.assertIn(value, command)
                self.assert_egd_dpp_policy(command)
            self.assertFalse(output.exists())

    def test_blender_over_eight_views_is_rejected(self):
        for views in ("9", "12", "99999999999999999999"):
            with self.subTest(views=views):
                result = self.run_script(VARIANT, "blender", env=dict(
                    clean_environment(), DRY_RUN="1", N_VIEWS=views))
                self.assert_rejected_without_dispatch(result)
                self.assertIn("N_VIEWS <= 8", result.stderr)

    def test_actual_matrix_preflights_later_settings_before_any_training(self):
        invalid = (
            {"MATRIX": "llff:3 blender:9"}, {"MATRIX": "llff:3 invalid:6"},
            {"MATRIX": "llff:3 dtu:0"}, {"MATRIX": "llff:3 llff:3"},
            {"MATRIX": " "}, {"MATRIX": "llff:3 llff:6", "GPU_LLFF_6": "0,1"},
            {"MATRIX": "llff:3 llff:6", "DPP_PRUNE_THRESHOLD": "NaN"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "llff")
                env.update(changes)
                self.assert_rejected_without_dispatch(self.run_script(VARIANT, repo=repo, env=env, matrix=True))
                self.assertFalse(log.exists())
                self.assertFalse((repo / "output").exists())

    def test_actual_matrix_runs_settings_serially_with_selected_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3 llff:6", GPU_ID="7", GPU_LLFF="2", GPU_LLFF_6="4")
            self.assert_success(self.run_script(VARIANT, repo=repo, env=env, matrix=True))
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls],
                             ["train.py", "render.py", "summarize_mvsdrop_setting.py"] * 2)
            for offset, views, gpu in ((0, 3, "2"), (3, 6, "4")):
                self.assertEqual(calls[offset]["gpu"], gpu)
                self.assertEqual(calls[offset + 1]["gpu"], gpu)
                self.assertEqual(fixtures.last_value(calls[offset]["args"], "--n_views"), str(views))
                self.assert_egd_dpp_policy(calls[offset]["args"])
                setting = repo / "output/ablation" / VARIANT / f"llff{views}"
                self.assertEqual(Path(fixtures.last_value(calls[offset + 2]["args"], "--experiment-root")), setting)

    def test_nonempty_output_is_preserved_and_preview_does_not_mutate_it(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            output = Path(directory) / "results"
            setting = output / VARIANT / "llff3"
            setting.mkdir(parents=True)
            marker = setting / ".keep"
            marker.write_text("prior results", encoding="utf-8")
            env["OUTPUT_ROOT"] = str(output)
            self.assert_rejected_without_dispatch(self.run_script(VARIANT, repo=repo, env=env))
            self.assert_success(self.run_script(VARIANT, repo=repo, env=env, args=("--dry-run",)))
            self.assertEqual(marker.read_text(), "prior results")
            self.assertEqual(list(setting.iterdir()), [marker])
            self.assertFalse(log.exists())

    def test_training_failure_stops_matrix_before_render_or_next_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3 llff:6", FAKE_FAIL_TRAIN="1")
            result = self.run_script(VARIANT, repo=repo, env=env, matrix=True)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])
            self.assertFalse((repo / "output/ablation" / VARIANT / "llff6").exists())


if __name__ == "__main__":
    unittest.main()
