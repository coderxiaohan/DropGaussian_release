"""Complement launchers: CPU-only fake dispatch, no datasets, MVS inference or GPU."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path("scripts/mvs_complement")
DATASETS = {
    "llff": (3, 8, "fern"), "dtu": (3, 4, "scan8"),
    "mipnerf360": (12, 8, "bicycle"), "blender": (8, 1, "lego"),
}
ENV_KEYS = {
    "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
    "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
    "DTU_MASK_ROOT", "CALL_LOG", "FAKE_FAIL_TRAIN", "FAKE_REQUIRE_BLENDER_RANDOM",
    "DRY_RUN", "MATRIX", "MODES", "VCP_STRATEGY",
}


def clean_environment():
    return {
        key: value for key, value in os.environ.items()
        if key not in ENV_KEYS and not key.startswith(("GPU_", "MVS_", "FASTGS_", "DPP_", "TBP_"))
    }


def last_value(args, flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]


FAKE_PYTHON = r'''
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
kind = Path(args[0]).name
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"kind": kind, "args": args,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")

def value(flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]

if kind == "train.py":
    if os.environ.get("FAKE_REQUIRE_BLENDER_RANDOM"):
        assert value("--init_mode") == "colmap_dense", args
        assert "--rand_pcd" in args, args
        assert value("--density_control") == "vcd_vcp", args
        assert not any(arg.startswith("--mvs") for arg in args), args
    if os.environ.get("FAKE_FAIL_TRAIN"):
        sys.exit(23)
    root = Path(value("-m"))
    cloud = root / "point_cloud" / ("iteration_" + value("--iterations")) / "point_cloud.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("fake cloud", encoding="utf-8")
    (root / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
elif kind == "render.py":
    root = Path(value("-m"))
    iteration = value("--iteration")
    (root / "test" / ("ours_" + iteration) / "renders").mkdir(parents=True)
    (root / ("metrics_" + iteration + ".txt")).write_text("fake metrics", encoding="utf-8")
'''


class MvsComplementLauncherTest(unittest.TestCase):
    def run_script(self, name, *, repo=ROOT, env=None, args=(), cwd=None):
        return subprocess.run(
            ["bash", str(repo / SCRIPTS / name), *args], cwd=cwd or repo.parent,
            env=env or clean_environment(), text=True, capture_output=True, timeout=30,
        )

    def fixture(self, directory, dataset):
        root = Path(directory)
        repo = root / "repo with spaces"
        for relative in (SCRIPTS, Path("scripts/mvs_vcd_vcp"),
                         Path("scripts/dropgaussian_vcd_vcp"), Path("scripts/vcd_vcp")):
            shutil.copytree(ROOT / relative, repo / relative)
        common = repo / "scripts/all"
        common.mkdir()
        shutil.copyfile(ROOT / "scripts/all/train_mvsdrop_benchmark.sh",
                        common / "train_mvsdrop_benchmark.sh")
        scene = DATASETS[dataset][2]
        data = root / "data with spaces"
        (data / (scene + "_colmap" if dataset == "dtu" else scene)).mkdir(parents=True)
        fake = root / "fake python with spaces"
        fake.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON, encoding="utf-8")
        fake.chmod(0o755)
        log = root / "calls.jsonl"
        env = dict(clean_environment(), PYTHON_BIN=str(fake), DATA_ROOT=str(data),
                   SCENES=scene, CALL_LOG=str(log))
        return repo, env, log

    def test_shell_syntax(self):
        for path in (ROOT / SCRIPTS).glob("*.sh"):
            with self.subTest(script=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_help_explains_geometry_provenance_budgets_and_required_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in DATASETS.items():
                with self.subTest(dataset=dataset):
                    result = self.run_script(f"train_{dataset}.sh", args=("--help",), env=dict(
                        clean_environment(), OUTPUT_ROOT=str(output),
                        DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for expected in (
                        f"all configured {dataset} scenes, {views} training views",
                        "10000 iterations, GPU 0", "All scenes does not mean all input views",
                        "training-view depth caches", "does not preserve every original point",
                        "Random clouds", "record the fallback", "100000",
                        "same source quotas", "depth-supported, positive-novelty candidate pool",
                        "samples that pool uniformly", "depth-cache pixels",
                        "training-view-only provenance is not independently verified",
                        "none with TOTAL_POINTS=0 uses MVS_INIT_MAX_POINTS (default: 100000)",
                        "Blender main method: original DropGaussian random initialization (10000 points)",
                        "no MVS cache or depth preflight", "EGD/DPP",
                        "explicit direct-MVS ablation on every dataset, including Blender",
                        "ignores all MVS_COMPLEMENT_* budgets/selection settings and MVS_INIT_* settings",
                        "<root>/blender_random/blender<views>",
                    ):
                        self.assertIn(expected, result.stdout)
            self.assertFalse(output.exists())

    def test_defaults_are_real_dataset_coverage_and_blender_random_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in DATASETS.items():
                with self.subTest(dataset=dataset):
                    result = self.run_script(f"train_{dataset}.sh", env=dict(
                        clean_environment(), OUTPUT_ROOT=str(output), DRY_RUN="1",
                        DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cmd = shlex.split(result.stdout)
                    if dataset == "blender":
                        self.assertIn(f"OUTPUT_ROOT={output}/blender_random/blender{views}", cmd)
                        self.assertIn("MODE=dropgaussian", cmd)
                        self.assertIn("DENSITY_CONTROL=vcd_vcp", cmd)
                        self.assertTrue(any(token.endswith("/dropgaussian_vcd_vcp/train_blender.sh")
                                            for token in cmd))
                        self.assertFalse(any(token.startswith(("--mvs", "MVS_")) for token in cmd))
                        self.assertIn("random initialization (10000 points)", result.stderr)
                        self.assertIn("no MVS/cache requirement", result.stderr)
                        self.assertIn("EGD/DPP", result.stderr)
                        continue
                    self.assertIn(f"OUTPUT_ROOT={output}/coverage/keep_base/{dataset}{views}", cmd)
                    for token in ("DENSITY_CONTROL=vcd_vcp", "MODE=mvs", "MVS_INIT_STRATEGY=replace"):
                        self.assertIn(token, cmd)
                    for flag, value in {
                        "--mvs_complement_mode": "coverage", "--mvs_complement_max_points": "100000",
                        "--mvs_complement_total_points": "0", "--mvs_complement_candidate_points": "500000",
                        "--mvs_complement_coverage_pixels": "2.0", "--mvs_complement_depth_scale": "0.02",
                    }.items():
                        self.assertEqual(last_value(cmd, flag), value)
                    self.assertFalse(any(arg.startswith(("--mvs_geometry", "--mvs_surface"))
                                         for arg in cmd))
            self.assertFalse(output.exists())

    def test_modes_and_total_budgets_are_isolated_and_experiments_disabled(self):
        for mode in ("none", "coverage", "concat", "balanced"):
            for budget in ("0", "100000"):
                with self.subTest(mode=mode, budget=budget):
                    result = self.run_script("train_llff.sh", env=dict(
                        clean_environment(), MVS_COMPLEMENT_MODE=mode,
                        MVS_COMPLEMENT_TOTAL_POINTS=budget,
                        MVS_SURFACE_INIT="oas", MVS_GEOMETRY_CORRECTION="tps", DRY_RUN="1"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cmd = shlex.split(result.stdout)
                    self.assertEqual(last_value(cmd, "--mvs_complement_mode"), mode)
                    self.assertEqual(last_value(cmd, "--mvs_complement_total_points"), budget)
                    self.assertEqual(last_value(cmd, "--vcp_strategy"), "fastgs")
                    self.assertIn("--fastgs_protect_egd", cmd)
                    self.assertFalse(any(arg.startswith(("--mvs_geometry", "--mvs_surface"))
                                         for arg in cmd))
                    label = "keep_base" if budget == "0" else "budget" + budget
                    self.assertTrue(any(token.endswith(f"/{mode}/{label}/llff3") for token in cmd))

    def test_passthrough_quotes_parameters_views_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            literal = "$(touch should_not_exist); * ' quoted"
            result = self.run_script("train_llff.sh", cwd=directory, env=dict(
                clean_environment(), N_VIEWS="9", GPU_ID="3", MVS_COMPLEMENT_MAX_POINTS="120000",
                MVS_COMPLEMENT_CANDIDATE_POINTS="600000", MVS_COMPLEMENT_COVERAGE_PIXELS="3",
                MVS_COMPLEMENT_DEPTH_SCALE="2e-2"),
                args=("--dry-run", "--literal", literal, "--empty", ""))
            self.assertEqual(result.returncode, 0, result.stderr)
            cmd = shlex.split(result.stdout)
            self.assertEqual(last_value(cmd, "--literal"), literal)
            self.assertEqual(last_value(cmd, "--empty"), "")
            self.assertEqual(last_value(cmd, "--mvs_complement_max_points"), "120000")
            self.assertEqual(last_value(cmd, "--mvs_complement_candidate_points"), "600000")
            self.assertEqual(last_value(cmd, "--mvs_complement_coverage_pixels"), "3")
            self.assertEqual(last_value(cmd, "--mvs_complement_depth_scale"), "2e-2")
            self.assertIn("N_VIEWS=9", cmd)
            self.assertIn("GPU_ID=3", cmd)
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_default_matrix_has_nine_isolated_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            result = self.run_script("train_all_views.sh", env=dict(
                clean_environment(), OUTPUT_ROOT=str(output), N_VIEWS="999"), args=("--dry-run",))
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(commands), 9)
            paths = {next(token for token in cmd if token.startswith("OUTPUT_ROOT=")) for cmd in commands}
            self.assertEqual(len(paths), 9)
            self.assertTrue(all(last_value(cmd, "--vcp_strategy") == "fastgs" for cmd in commands))
            self.assertTrue(all("--fastgs_protect_egd" in cmd for cmd in commands))
            self.assertTrue(all(last_value(cmd, "--mvs_complement_mode") == "coverage"
                                for cmd in commands[:-1]))
            self.assertIn("MODE=dropgaussian", commands[-1])
            self.assertIn(f"OUTPUT_ROOT={output}/blender_random/blender8", commands[-1])
            self.assertFalse(any(token.startswith("--mvs") for token in commands[-1]))
            self.assertTrue(all("N_VIEWS=999" not in cmd for cmd in commands))
            self.assertFalse(output.exists())

    def test_matrix_gpu_precedence(self):
        result = self.run_script("train_all_views.sh", args=("--dry-run",), env=dict(
            clean_environment(), MATRIX="llff:3 llff:6 dtu:9 blender:8",
            GPU_ID="7", GPU_LLFF="2", GPU_LLFF_6="4", GPU_DTU_9="5"))
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 4)
        for cmd, gpu in zip(commands, (2, 4, 5, 7)):
            self.assertIn(f"GPU_ID={gpu}", cmd)

    def test_invalid_configuration_fails_without_dispatch(self):
        invalid = (
            {"MVS_COMPLEMENT_MODE": "adaptive"}, {"MVS_COMPLEMENT_MAX_POINTS": "0"},
            {"MVS_COMPLEMENT_MAX_POINTS": "2.5"}, {"MVS_COMPLEMENT_TOTAL_POINTS": "-1"},
            {"MVS_COMPLEMENT_CANDIDATE_POINTS": "0"}, {"MVS_COMPLEMENT_COVERAGE_PIXELS": "0"},
            {"MVS_COMPLEMENT_COVERAGE_PIXELS": "NaN"}, {"MVS_COMPLEMENT_DEPTH_SCALE": "inf"},
            {"MVS_COMPLEMENT_DEPTH_SCALE": "1e999"}, {"MVS_COMPLEMENT_DEPTH_SCALE": "-0.1"},
            {"N_VIEWS": "0"}, {"GPU_ID": "0,1"}, {"DRY_RUN": "yes"},
            {"VCP_STRATEGY": "tbp"}, {"TBP_MAX_TRIALS": "3"},
            {"FASTGS_PROTECT_EGD": "0"}, {"FASTGS_PRUNE_INTERVAL": "4000"},
            {"DPP_ONLY": "1"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                env = dict(clean_environment(), DRY_RUN="1")
                env.update(changes)
                result = self.run_script("train_llff.sh", env=env)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")
        for changes in (
            {"MATRIX": "llff:3 invalid:6"}, {"MATRIX": "llff:3 llff:3"}, {"MATRIX": " "},
            {"MATRIX": "llff:3 dtu:6", "GPU_DTU_6": "0,1"},
            {"MVS_COMPLEMENT_MODE": "invalid"},
        ):
            with self.subTest(matrix_changes=changes):
                result = self.run_script("train_all_views.sh", env=dict(clean_environment(), DRY_RUN="1", **changes))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")

    def test_owned_cli_and_abbreviations_cannot_change_protocol(self):
        options = (
            "-m", "-melsewhere", "--model_path", "--model", "-s", "-selsewhere", "--source",
            "-r", "-r2", "--resolution", "--resol", "--iterations", "--iter", "--n_v",
            "--dataset_name", "--dataset", "--mvs_complement_mode", "--mvs_complement_m",
            "--mvs_geometry_correction", "--mvs_geo", "--mvs_surface_init", "--mvs_s",
            "--mvs_init_max_points", "--mv", "--density_control", "--density_con",
            "--density_min_views", "--vcd_error_quantile", "--vcp", "--init_mode", "--init_m",
            "--vcp_max_trials", "--vcp_contribution_quantile", "--vcp_strategy", "--dpp_only",
            "--fastgs_protect_egd", "--fastgs_prune_interval", "--fastgs_p", "--fastg",
            "--rand_pcd", "--white_background", "-w", "--images", "-i", "--eval", "--seed",
            "--test_iterations", "--save_iterations", "--start_checkpoint", "--position_lr_init",
            "--densify_until_iter", "--opacity_reset_interval", "--sh_degree", "--random_background",
        )
        for dataset in ("llff", "blender"):
            for option in options:
                for args in ((option, "invalid"), (option + "=invalid",)):
                    with self.subTest(dataset=dataset, args=args):
                        result = self.run_script(f"train_{dataset}.sh", env=dict(
                            clean_environment(), DRY_RUN="1"), args=args)
                        self.assertEqual(result.returncode, 2)
                        self.assertEqual(result.stdout, "")
        result = self.run_script("train_all_views.sh", args=("--dry-run", "--n_v", "6"))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")

    def test_actual_fake_dispatch_preserves_training_and_evaluation_protocol(self):
        for dataset, (views, resolution, scene) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(MODE="dropgaussian", DENSITY_CONTROL="none", GPU_ID="4",
                           MVS_SURFACE_INIT="oas", MVS_GEOMETRY_CORRECTION="tps",
                           MVS_INIT_STRATEGY="adaptive")
                result = self.run_script(f"train_{dataset}.sh", repo=repo, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = ["train.py", "render.py"]
                if dataset == "dtu":
                    expected.append("evaluate_dtu.py")
                expected.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], expected)
                args = calls[0]["args"]
                for flag, value in {
                    "--density_control": "vcd_vcp",
                    "--vcp_strategy": "fastgs",
                    "--n_views": str(views), "-r": str(resolution), "--iterations": "10000",
                }.items():
                    self.assertEqual(last_value(args, flag), value)
                self.assertIn("--fastgs_protect_egd", args)
                if dataset == "blender":
                    self.assertEqual(last_value(args, "--init_mode"), "colmap_dense")
                    self.assertIn("--rand_pcd", args)
                    self.assertIn("--white_background", args)
                    self.assertFalse(any(arg.startswith("--mvs") for arg in args))
                    setting = "blender_random"
                else:
                    for flag, value in {
                        "--init_mode": "mvs_anywhere", "--mvs_init_strategy": "replace",
                        "--mvs_complement_mode": "coverage", "--mvs_complement_total_points": "0",
                        "--mvs_complement_max_points": "100000",
                    }.items():
                        self.assertEqual(last_value(args, flag), value)
                    setting = "coverage/keep_base"
                self.assertFalse(any(arg.startswith(("--mvs_geometry", "--mvs_surface"))
                                     for arg in args))
                self.assertEqual(calls[0]["gpu"], "4")
                output = repo / "output/mvs_complement_dpp" / setting / f"{dataset}{views}" / scene
                self.assertEqual(Path(last_value(args, "-m")), output)
                self.assertEqual(last_value(calls[1]["args"], "-r"), str(resolution))
                self.assertEqual(last_value(calls[1]["args"], "--iteration"), "10000")

    def test_blender_random_modes_ignore_mvs_settings_and_need_no_mvs_runner_or_cache(self):
        for mode in ("coverage", "concat", "balanced"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "blender")
                # Remove the copied direct-MVS runner so an accidental dispatch
                # or preflight there fails before the fake Python can mask it.
                shutil.rmtree(repo / "scripts/mvs_vcd_vcp")
                env.update(
                    MVS_COMPLEMENT_MODE=mode, MODE="mvs", DENSITY_CONTROL="none",
                    MVS_INIT_PATH="/missing/cache/that/must/not/be/read.npz",
                    MVS_INIT_STRATEGY="adaptive", MVS_INIT_MAX_POINTS="invalid",
                    MVS_COMPLEMENT_MAX_POINTS="invalid", MVS_COMPLEMENT_TOTAL_POINTS="-1",
                    MVS_COMPLEMENT_CANDIDATE_POINTS="0", MVS_COMPLEMENT_COVERAGE_PIXELS="NaN",
                    MVS_COMPLEMENT_DEPTH_SCALE="inf", FAKE_REQUIRE_BLENDER_RANDOM="1",
                    N_VIEWS="6", GPU_ID="2", ITERATIONS="1000", RESOLUTION="2",
                )
                result = self.run_script("train_blender.sh", repo=repo, env=env)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual([call["kind"] for call in calls],
                                 ["train.py", "render.py", "summarize_mvsdrop_setting.py"])
                args = calls[0]["args"]
                self.assertEqual(last_value(args, "--n_views"), "6")
                self.assertEqual(last_value(args, "--iterations"), "1000")
                self.assertEqual(last_value(args, "-r"), "2")
                self.assertEqual(calls[0]["gpu"], "2")
                self.assertIn("/blender_random/blender6/lego", last_value(args, "-m"))

    def test_dpp_default_output_preserves_historical_full_method_results(self):
        for dataset, (views, _, scene) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                suffix = f"blender_random/blender{views}" if dataset == "blender" else f"coverage/keep_base/{dataset}{views}"
                previous = repo / "output/mvs_complement" / suffix / scene / "historical.txt"
                previous.parent.mkdir(parents=True)
                previous.write_text("historical method", encoding="utf-8")
                result = self.run_script(f"train_{dataset}.sh", repo=repo, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(log.read_text().splitlines()[0])["args"]
                self.assertEqual(Path(last_value(args, "-m")), repo / "output/mvs_complement_dpp" / suffix / scene)
                self.assertEqual(previous.read_text(), "historical method")

    def test_blender_none_preserves_explicit_direct_mvs_ablation(self):
        for total, expected in (("0", "75000"), ("50000", "50000")):
            with self.subTest(total=total), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "blender")
                env.update(MVS_COMPLEMENT_MODE="none", MVS_COMPLEMENT_TOTAL_POINTS=total,
                           MVS_INIT_MAX_POINTS="75000", MVS_INIT_PATH="/explicit/cache.npz")
                result = self.run_script("train_blender.sh", repo=repo, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(log.read_text().splitlines()[0])["args"]
                self.assertEqual(last_value(args, "--init_mode"), "mvs_anywhere")
                self.assertEqual(last_value(args, "--mvs_init_strategy"), "replace")
                self.assertEqual(last_value(args, "--mvs_complement_mode"), "none")
                self.assertEqual(last_value(args, "--mvs_init_max_points"), expected)
                self.assertEqual(last_value(args, "--mvs_init_path"), "/explicit/cache.npz")
                self.assertEqual(last_value(args, "--density_control"), "vcd_vcp")
                label = "keep_base" if total == "0" else "budget" + total
                self.assertIn(f"/none/{label}/blender8/lego", last_value(args, "-m"))

    def test_blender_random_outputs_do_not_reuse_old_mvs_complement_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "blender")
            output = Path(directory) / "output with spaces"
            old_setting = output / "coverage/keep_base/blender8"
            old_setting.mkdir(parents=True)
            marker = old_setting / "existing-results.txt"
            marker.write_text("preserve old MVS results", encoding="utf-8")
            env["OUTPUT_ROOT"] = str(output)
            first = self.run_script("train_blender.sh", repo=repo, env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = log.read_text()
            self.assertEqual(marker.read_text(), "preserve old MVS results")
            self.assertTrue((output / "blender_random/blender8/lego/manifest.json").exists())
            for mode in ("coverage", "concat", "balanced"):
                repeated = self.run_script("train_blender.sh", repo=repo, env=dict(
                    env, MVS_COMPLEMENT_MODE=mode, MVS_COMPLEMENT_TOTAL_POINTS="50000"))
                self.assertEqual(repeated.returncode, 2)
                self.assertIn("fresh OUTPUT_ROOT", repeated.stderr)
                self.assertEqual(log.read_text(), original)

    def test_failed_training_stops_matrix_without_render_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3 llff:6", FAKE_FAIL_TRAIN="1")
            result = self.run_script("train_all_views.sh", repo=repo, env=env)
            self.assertEqual(result.returncode, 23)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])

    def test_none_positive_total_budget_reaches_the_actual_direct_loader(self):
        for mode, total, inherited, expected in (
            ("none", "0", None, "100000"),
            ("none", "0", "75000", "75000"),
            ("none", "50000", None, "50000"),
            ("none", "50000", "200000", "50000"),
            ("none", "200000", "50000", "200000"),
            ("coverage", "50000", "75000", "75000"),
        ):
            with self.subTest(mode=mode, total=total, inherited=inherited), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "llff")
                env.update(MVS_COMPLEMENT_MODE=mode, MVS_COMPLEMENT_TOTAL_POINTS=total)
                if inherited is not None:
                    env["MVS_INIT_MAX_POINTS"] = inherited
                result = self.run_script("train_llff.sh", repo=repo, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                args = calls[0]["args"]
                self.assertEqual(last_value(args, "--mvs_init_max_points"), expected)
                self.assertEqual(last_value(args, "--mvs_complement_total_points"), total)
                self.assertEqual(last_value(args, "--mvs_complement_mode"), mode)
                label = "keep_base" if total == "0" else "budget" + total
                self.assertIn(f"/{mode}/{label}/llff3/", last_value(args, "-m"))

    def test_nonempty_outputs_cannot_be_reused_but_preview_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            first = self.run_script("train_llff.sh", repo=repo, env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = log.read_text()
            repeated = self.run_script("train_llff.sh", repo=repo, env=env)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("fresh OUTPUT_ROOT", repeated.stderr)
            self.assertEqual(log.read_text(), original)
            preview = self.run_script("train_llff.sh", repo=repo, env=env, args=("--dry-run",))
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(log.read_text(), original)

    def test_modes_and_budgets_do_not_reuse_fake_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            for mode, budget in (("none", "0"), ("concat", "0"), ("balanced", "100000"), ("coverage", "100000")):
                with self.subTest(mode=mode, budget=budget):
                    result = self.run_script("train_llff.sh", repo=repo, env=dict(
                        env, MVS_COMPLEMENT_MODE=mode, MVS_COMPLEMENT_TOTAL_POINTS=budget,
                        ITERATIONS="1000", RESOLUTION="4"))
                    self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            trains = [call["args"] for call in calls if call["kind"] == "train.py"]
            self.assertEqual(len(trains), 4)
            self.assertEqual(len({last_value(args, "-m") for args in trains}), 4)
            self.assertTrue(all(last_value(args, "--iterations") == "1000" for args in trains))


if __name__ == "__main__":
    unittest.main()
