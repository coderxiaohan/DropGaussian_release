"""EGD-protected FastGS launchers, exercised without real training or CUDA."""

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
SCRIPTS = Path("scripts/fastgs_egd_pruning")
ORIGINAL_SCRIPTS = Path("scripts/fastgs_pruning")
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


class FastGSEGDPruningLauncherTest(unittest.TestCase):
    def run_script(self, dataset="llff", *, repo=ROOT, env=None, args=(), original=False):
        scripts = ORIGINAL_SCRIPTS if original else SCRIPTS
        return subprocess.run(
            ["bash", str(repo / scripts / f"train_{dataset}.sh"), *args],
            cwd=repo.parent, env=env or clean_environment(),
            text=True, capture_output=True, timeout=30,
        )

    def fixture(self, directory, dataset):
        root = Path(directory)
        repo = root / "repo with spaces"
        for relative in (SCRIPTS, ORIGINAL_SCRIPTS, Path("scripts/mvs_vcd_vcp"),
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

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_entrypoints_have_valid_shell_syntax(self):
        for name in ["_run.sh", *(f"train_{dataset}.sh" for dataset in DATASETS)]:
            with self.subTest(name=name):
                subprocess.run(["bash", "-n", str(ROOT / SCRIPTS / name)], check=True)

    def test_help_and_preview_are_read_only_with_correct_new_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in DATASETS.items():
                with self.subTest(dataset=dataset):
                    env = dict(clean_environment(), OUTPUT_ROOT=str(output), GPU_ID="5",
                               DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python")
                    help_result = self.run_script(dataset, env=env, args=("--help",))
                    self.assert_success(help_result)
                    self.assertIn("scripts/fastgs_egd_pruning/", help_result.stdout)
                    self.assertIn("FASTGS_PROTECT_EGD=1", help_result.stdout)
                    result = self.run_script(dataset, env=env, args=("--dry-run",))
                    self.assert_success(result)
                    command = shlex.split(result.stdout)
                    suffix = (f"blender_random/blender{views}" if dataset == "blender"
                              else f"coverage/keep_base/{dataset}{views}")
                    self.assertIn(f"OUTPUT_ROOT={output}/{suffix}", command)
                    self.assertEqual(command.count("--fastgs_protect_egd"), 1)
                    self.assertIn("GPU_ID=5", command)
                    self.assertIn(f"N_VIEWS={views}", command)
                    for flag, value in FASTGS_ARGS.items():
                        self.assertEqual(last_value(command, flag), value)
            self.assertFalse(output.exists())

    def test_default_old_preview_remains_unprotected_with_original_output(self):
        for dataset, (views, _, _) in DATASETS.items():
            with self.subTest(dataset=dataset):
                result = self.run_script(dataset, original=True, args=("--dry-run",))
                self.assert_success(result)
                command = shlex.split(result.stdout)
                self.assertNotIn("--fastgs_protect_egd", command)
                suffix = (f"blender_random/blender{views}" if dataset == "blender"
                          else f"coverage/keep_base/{dataset}{views}")
                self.assertIn(f"OUTPUT_ROOT={ROOT}/output/fastgs_pruning/{suffix}", command)

    def test_actual_dispatch_old_vs_new_only_protection_flag_and_output_differ(self):
        def normalized(args):
            result = []
            skip_path = False
            for argument in args:
                if skip_path:
                    skip_path = False
                elif argument == "-m":
                    skip_path = True
                elif argument != "--fastgs_protect_egd":
                    result.append(argument)
            return result

        for dataset, (_, resolution, scene) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(GPU_ID="4", N_VIEWS="6", SEED="9")
                self.assert_success(self.run_script(dataset, repo=repo, env=env, original=True))
                self.assert_success(self.run_script(dataset, repo=repo, env=env))
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                stages = ["train.py", "render.py"]
                if dataset == "dtu":
                    stages.append("evaluate_dtu.py")
                stages.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], stages * 2)
                trains = [call for call in calls if call["kind"] == "train.py"]
                old_args, new_args = (call["args"] for call in trains)
                self.assertNotIn("--fastgs_protect_egd", old_args)
                self.assertEqual(new_args.count("--fastgs_protect_egd"), 1)
                self.assertEqual(normalized(old_args), normalized(new_args))
                for args in (old_args, new_args):
                    for flag, value in dict(FASTGS_ARGS, **{
                        "--n_views": "6", "-r": str(resolution),
                        "--density_control": "vcd_vcp", "--seed": "9",
                    }).items():
                        self.assertEqual(last_value(args, flag), value)
                self.assertTrue(all(call["gpu"] == "4" for call in trains))
                suffix = "blender_random/blender6" if dataset == "blender" else f"coverage/keep_base/{dataset}6"
                old_setting = repo / "output/fastgs_pruning" / suffix
                new_setting = repo / "output/fastgs_egd_pruning" / suffix
                self.assertEqual(Path(last_value(old_args, "-m")), old_setting / scene)
                self.assertEqual(Path(last_value(new_args, "-m")), new_setting / scene)
                self.assertTrue((old_setting / "metrics_mean.txt").is_file())
                self.assertTrue((new_setting / "metrics_mean.txt").is_file())
                if dataset == "blender":
                    self.assertEqual(last_value(new_args, "--init_mode"), "colmap_dense")
                    self.assertIn("--rand_pcd", new_args)
                    self.assertFalse(any(arg.startswith("--mvs") for arg in new_args))
                else:
                    self.assertEqual(last_value(new_args, "--mvs_complement_mode"), "coverage")

    def test_explicit_old_protection_switch_matches_new_preview(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                env = dict(clean_environment(), FASTGS_PROTECT_EGD="1",
                           OUTPUT_ROOT="/a/fresh/parent", GPU_ID="2", N_VIEWS="9")
                old = self.run_script(dataset, env=env, args=("--dry-run",), original=True)
                new = self.run_script(dataset, env=env, args=("--dry-run",))
                self.assert_success(old)
                self.assert_success(new)
                self.assertEqual(shlex.split(old.stdout), shlex.split(new.stdout))

    def test_conflicting_new_protection_environment_and_invalid_old_values_rejected(self):
        for value in ("0", "false", "2", ""):
            with self.subTest(value=value):
                result = self.run_script(env=dict(clean_environment(), FASTGS_PROTECT_EGD=value),
                                         args=("--dry-run",))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
        for value in ("false", "2", "-1"):
            with self.subTest(old_value=value):
                result = self.run_script(env=dict(clean_environment(), FASTGS_PROTECT_EGD=value),
                                         original=True, args=("--dry-run",))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")

    def test_protection_cli_and_other_method_overrides_are_rejected(self):
        for args in (("--fastgs_protect_egd",), ("--fastgs_protect_egd=false",),
                     ("--fastgs_protect",), ("--fast",), ("--f",),
                     ("--vcp_strategy", "tbp"), ("--density_control", "none"),
                     ("--init_mode", "colmap_dense"), ("--mvs_complement_mode", "none"),
                     ("--start_checkpoint", "/missing"), ("--",)):
            with self.subTest(args=args):
                result = self.run_script(args=("--dry-run", *args))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")

    def test_numeric_overrides_and_positive_budget_use_shared_dispatch(self):
        env = dict(clean_environment(), FASTGS_PRUNE_INTERVAL="7000",
                   FASTGS_PRUNE_THRESHOLD="0.95", MVS_COMPLEMENT_TOTAL_POINTS="100000",
                   OUTPUT_ROOT="/fresh parent", N_VIEWS="6", GPU_ID="3")
        result = self.run_script(env=env, args=("--dry-run",))
        self.assert_success(result)
        args = shlex.split(result.stdout)
        self.assertEqual(last_value(args, "--fastgs_prune_interval"), "7000")
        self.assertEqual(last_value(args, "--fastgs_prune_threshold"), "0.95")
        self.assertIn("OUTPUT_ROOT=/fresh parent/coverage/budget100000/llff6", args)
        self.assertEqual(args.count("--fastgs_protect_egd"), 1)

    def test_blender_keeps_random_policy_with_irrelevant_invalid_mvs_settings(self):
        env = dict(clean_environment(), MVS_INIT_PATH="/missing.npz",
                   MVS_COMPLEMENT_TOTAL_POINTS="-1", MVS_COMPLEMENT_DEPTH_SCALE="NaN")
        result = self.run_script("blender", env=env, args=("--dry-run",))
        self.assert_success(result)
        args = shlex.split(result.stdout)
        self.assertIn("MODE=dropgaussian", args)
        self.assertIn("--fastgs_protect_egd", args)
        self.assertFalse(any(arg.startswith("--mvs") for arg in args))

    def test_nonempty_new_setting_does_not_dispatch_or_modify_old_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            setting = repo / "output/fastgs_egd_pruning/coverage/keep_base/llff3"
            setting.mkdir(parents=True)
            marker = setting / ".keep"
            marker.write_bytes(b"existing protected result")
            old = repo / "output/fastgs_pruning/coverage/keep_base/llff3"
            old.mkdir(parents=True)
            old_marker = old / "metrics_mean.txt"
            old_marker.write_bytes(b"old result unchanged")
            result = self.run_script(repo=repo, env=env)
            self.assertEqual(result.returncode, 2)
            self.assertIn("nonempty", result.stderr)
            self.assertFalse(log.exists())
            self.assertEqual(marker.read_bytes(), b"existing protected result")
            self.assertEqual(old_marker.read_bytes(), b"old result unchanged")

    def test_failed_training_does_not_render_or_summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env["FAKE_FAIL_TRAIN"] = "1"
            result = self.run_script(repo=repo, env=env)
            self.assertEqual(result.returncode, 23)
            self.assertEqual([json.loads(line)["kind"] for line in log.read_text().splitlines()], ["train.py"])


if __name__ == "__main__":
    unittest.main()
