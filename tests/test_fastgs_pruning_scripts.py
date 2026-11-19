"""FastGS policy launchers, without CUDA compilation, real data or training."""

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
SCRIPTS = Path("scripts/fastgs_pruning")
DATASETS = {
    "llff": (3, 8, "fern"),
    "dtu": (3, 4, "scan8"),
    "mipnerf360": (12, 8, "bicycle"),
    "blender": (8, 1, "lego"),
}


def clean_environment():
    keys = {"MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
            "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
            "DTU_MASK_ROOT", "CALL_LOG", "FAKE_FAIL_TRAIN", "DRY_RUN", "CUDA_VISIBLE_DEVICES", "VCP_STRATEGY"}
    return {key: value for key, value in os.environ.items()
            if key not in keys and not key.startswith(("GPU_", "MVS_", "TBP_", "FASTGS_"))}


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
elif kind == "summarize_mvsdrop_setting.py":
    root = Path(value("--experiment-root"))
    (root / "metrics_mean.txt").write_text("fake mean", encoding="utf-8")
'''


FASTGS_ARGS = {
    "--vcp_strategy": "fastgs",
    "--fastgs_prune_interval": "8000",
    "--fastgs_loss_threshold": "0.1",
    "--fastgs_prune_threshold": "0.9",
    "--fastgs_opacity_threshold": "0.1",
}


class FastGSPruningLauncherTest(unittest.TestCase):
    def run_script(self, dataset="llff", *, repo=ROOT, env=None, args=(), full_method=False):
        scripts = Path("scripts/mvs_complement") if full_method else SCRIPTS
        return subprocess.run(
            ["bash", str(repo / scripts / f"train_{dataset}.sh"), *args],
            cwd=repo.parent, env=env or clean_environment(),
            text=True, capture_output=True, timeout=30,
        )

    def fixture(self, directory, dataset):
        root = Path(directory)
        repo = root / "repo with spaces"
        for relative in (SCRIPTS, Path("scripts/mvs_complement"),
                         Path("scripts/mvs_vcd_vcp"), Path("scripts/dropgaussian_vcd_vcp"),
                         Path("scripts/vcd_vcp")):
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

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_entrypoints_have_valid_shell_syntax(self):
        for name in ["_run.sh", *(f"train_{dataset}.sh" for dataset in DATASETS)]:
            with self.subTest(name=name):
                subprocess.run(["bash", "-n", str(ROOT / SCRIPTS / name)], check=True)

    def test_help_and_dry_run_do_not_write_or_require_data_python_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in DATASETS.items():
                with self.subTest(dataset=dataset):
                    env = dict(clean_environment(), OUTPUT_ROOT=str(output), GPU_ID="5",
                               DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python")
                    help_result = self.run_script(dataset, env=env, args=("--help",))
                    self.assert_success(help_result)
                    self.assertIn("FASTGS_PRUNE_INTERVAL", help_result.stdout)
                    result = self.run_script(dataset, env=env, args=("--dry-run",))
                    self.assert_success(result)
                    command = shlex.split(result.stdout)
                    suffix = (f"blender_random/blender{views}" if dataset == "blender"
                              else f"coverage/keep_base/{dataset}{views}")
                    self.assertIn(f"OUTPUT_ROOT={output}/{suffix}", command)
                    for flag, value in FASTGS_ARGS.items():
                        self.assertEqual(last_value(command, flag), value)
                    self.assertIn("GPU_ID=5", command)
                    self.assertIn("DENSITY_CONTROL=vcd_vcp", command)
                    self.assertNotIn("--vcp_max_trials", command)
            self.assertFalse(output.exists())

    def test_actual_dispatch_preserves_complete_method_protocol(self):
        for dataset, (views, resolution, scene) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(GPU_ID="4")
                result = self.run_script(dataset, repo=repo, env=env)
                self.assert_success(result)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = ["train.py", "render.py"]
                if dataset == "dtu":
                    expected.append("evaluate_dtu.py")
                expected.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], expected)
                args = calls[0]["args"]
                for flag, value in dict(FASTGS_ARGS, **{
                    "--density_control": "vcd_vcp", "--n_views": str(views),
                    "-r": str(resolution), "--iterations": "10000",
                    "--dataset_name": dataset,
                }).items():
                    self.assertEqual(last_value(args, flag), value)
                self.assertEqual(calls[0]["gpu"], "4")
                self.assertEqual(last_value(calls[1]["args"], "-r"), str(resolution))
                self.assertNotIn("--vcp_max_trials", args)
                if dataset == "blender":
                    suffix = f"blender_random/blender{views}"
                    self.assertEqual(last_value(args, "--init_mode"), "colmap_dense")
                    self.assertIn("--rand_pcd", args)
                    self.assertIn("--white_background", args)
                    self.assertFalse(any(arg.startswith("--mvs") for arg in args))
                else:
                    suffix = f"coverage/keep_base/{dataset}{views}"
                    self.assertEqual(last_value(args, "--init_mode"), "mvs_anywhere")
                    self.assertEqual(last_value(args, "--mvs_complement_mode"), "coverage")
                    self.assertEqual(last_value(args, "--mvs_init_strategy"), "replace")
                setting = repo / "output/fastgs_pruning" / suffix
                self.assertEqual(Path(last_value(args, "-m")), setting / scene)
                self.assertTrue((setting / "metrics_mean.txt").is_file())

    def test_matches_complete_method_except_pruning_options_and_output(self):
        def normalized(args):
            result = []
            skip = False
            for argument in args:
                if skip:
                    skip = False
                elif argument == "--fastgs_protect_egd":
                    continue
                elif argument in ("-m", *FASTGS_ARGS):
                    skip = True
                else:
                    result.append(argument)
            return result

        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(SEED="7", GPU_ID="2", N_VIEWS="6", ITERATIONS="20",
                           FASTGS_PRUNE_INTERVAL="10",
                           MVS_COMPLEMENT_TOTAL_POINTS="100000", MVS_COMPLEMENT_MAX_POINTS="20000",
                           MVS_COMPLEMENT_CANDIDATE_POINTS="200000",
                           MVS_COMPLEMENT_COVERAGE_PIXELS="3.0", MVS_COMPLEMENT_DEPTH_SCALE="0.04")
                self.assert_success(self.run_script(dataset, repo=repo, env=env))
                new_args = json.loads(log.read_text().splitlines()[0])["args"]
                full_env = {key: value for key, value in env.items() if not key.startswith("FASTGS_")}
                self.assert_success(self.run_script(dataset, repo=repo, env=full_env, full_method=True))
                trains = [json.loads(line) for line in log.read_text().splitlines()
                          if json.loads(line)["kind"] == "train.py"]
                self.assertEqual(len(trains), 2)
                self.assertEqual(normalized(new_args), normalized(trains[1]["args"]))
                self.assertEqual(last_value(trains[1]["args"], "--vcp_strategy"), "fastgs")
                self.assertIn("--fastgs_protect_egd", trains[1]["args"])
                self.assertNotIn("--fastgs_protect_egd", new_args)
                if dataset != "blender":
                    self.assertIn("/coverage/budget100000/", last_value(new_args, "-m"))

    def test_supported_fastgs_numeric_overrides_are_forwarded(self):
        env = dict(clean_environment(), FASTGS_PRUNE_INTERVAL="7000",
                   FASTGS_LOSS_THRESHOLD="0", FASTGS_PRUNE_THRESHOLD="1",
                   FASTGS_OPACITY_THRESHOLD="5e-2")
        result = self.run_script(env=env, args=("--dry-run",))
        self.assert_success(result)
        args = shlex.split(result.stdout)
        for flag, value in {
            "--fastgs_prune_interval": "7000", "--fastgs_loss_threshold": "0",
            "--fastgs_prune_threshold": "1", "--fastgs_opacity_threshold": "5e-2",
        }.items():
            self.assertEqual(last_value(args, flag), value)

    def test_conflicting_method_environment_is_rejected(self):
        for key, value in (("DENSITY_CONTROL", "none"), ("MVS_COMPLEMENT_MODE", "none"),
                           ("MODE", "both"), ("MVS_INIT_STRATEGY", "adaptive"),
                           ("VCP_STRATEGY", "tbp"), ("TBP_MAX_TRIALS", "3")):
            with self.subTest(key=key):
                result = self.run_script(env=dict(clean_environment(), **{key: value}), args=("--dry-run",))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")

    def test_method_protocol_cli_and_argparse_abbreviations_are_rejected(self):
        overrides = (
            ("--vcp_strategy", "tbp"), ("--vcp_s", "tbp"), ("--vcp_max_trials", "3"),
            ("--fastgs_prune_interval", "1"), ("--fastgs_prune_threshold=0.5",),
            ("--fast", "1"), ("--f", "1"), ("--density_min_views", "1"),
            ("--mvs_complement_mode", "none"), ("--init", "colmap_dense"),
            ("--model_path", "/tmp/elsewhere"), ("-m/tmp/elsewhere",),
            ("--source_path", "/missing"), ("--iter", "1"),
            ("--start_check", "/missing.pth"), ("--rand_pcd",), ("--",),
        )
        for args in overrides:
            with self.subTest(args=args):
                result = self.run_script(args=("--dry-run", *args))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_numeric_environment_and_final_step_interval_are_rejected(self):
        invalid = (
            ("FASTGS_PRUNE_INTERVAL", "0"), ("FASTGS_PRUNE_INTERVAL", "1.2"),
            ("FASTGS_PRUNE_INTERVAL", "10000"), ("FASTGS_PRUNE_INTERVAL", "11000"),
            ("FASTGS_LOSS_THRESHOLD", "-0.1"), ("FASTGS_LOSS_THRESHOLD", "NaN"),
            ("FASTGS_PRUNE_THRESHOLD", "1.1"), ("FASTGS_OPACITY_THRESHOLD", "inf"),
            ("N_VIEWS", "0"), ("GPU_ID", "0,1"), ("ITERATIONS", "-1"),
            ("RESOLUTION", "0"), ("DRY_RUN", "yes"),
            ("MVS_COMPLEMENT_MAX_POINTS", "0"), ("MVS_COMPLEMENT_TOTAL_POINTS", "-1"),
            ("MVS_COMPLEMENT_CANDIDATE_POINTS", "0"),
            ("MVS_COMPLEMENT_COVERAGE_PIXELS", "NaN"), ("MVS_COMPLEMENT_DEPTH_SCALE", "0"),
        )
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                env = dict(clean_environment(), DRY_RUN="1")
                env[key] = value
                result = self.run_script(env=env)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")
        result = self.run_script(env=dict(clean_environment(), ITERATIONS="8000"),
                                 args=("--dry-run",))
        self.assertEqual(result.returncode, 2)
        self.assertIn("smaller than ITERATIONS", result.stderr)

    def test_blender_ignores_mvs_cache_and_numeric_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "blender")
            env.update(MVS_INIT_PATH="/missing/cache.npz", MVS_INIT_MAX_POINTS="invalid",
                       MVS_INIT_STRATEGY="adaptive", MVS_COMPLEMENT_MAX_POINTS="invalid",
                       MVS_COMPLEMENT_TOTAL_POINTS="-1", MVS_COMPLEMENT_CANDIDATE_POINTS="0",
                       MVS_COMPLEMENT_COVERAGE_PIXELS="NaN", MVS_COMPLEMENT_DEPTH_SCALE="inf")
            self.assert_success(self.run_script("blender", repo=repo, env=env))
            args = json.loads(log.read_text().splitlines()[0])["args"]
            self.assertEqual(last_value(args, "--init_mode"), "colmap_dense")
            self.assertEqual(last_value(args, "--vcp_strategy"), "fastgs")
            self.assertFalse(any(arg.startswith("--mvs") for arg in args))

    def test_nonempty_setting_is_preserved_without_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            setting = repo / "output/fastgs_pruning/coverage/keep_base/llff3"
            setting.mkdir(parents=True)
            marker = setting / ".keep"
            marker.write_bytes(b"existing result")
            result = self.run_script(repo=repo, env=env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("nonempty", result.stderr)
            self.assertEqual(marker.read_bytes(), b"existing result")
            self.assertFalse(log.exists())

    def test_symlink_setting_is_rejected_even_when_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            setting = repo / "output/fastgs_pruning/coverage/keep_base/llff3"
            setting.parent.mkdir(parents=True)
            target = Path(directory) / "empty existing directory"
            target.mkdir()
            setting.symlink_to(target, target_is_directory=True)
            result = self.run_script(repo=repo, env=env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("symlink", result.stderr)
            self.assertFalse(log.exists())
            self.assertEqual(list(target.iterdir()), [])

    def test_train_failure_does_not_continue_to_render_or_summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env["FAKE_FAIL_TRAIN"] = "1"
            result = self.run_script(repo=repo, env=env)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            self.assertEqual([json.loads(line)["kind"] for line in log.read_text().splitlines()], ["train.py"])


if __name__ == "__main__":
    unittest.main()
