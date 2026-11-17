"""TWINGS-inspired initializer launchers: no GPU, models, or real datasets."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path("scripts/twings")
DATASETS = {
    "llff": (3, 8, "fern"),
    "dtu": (3, 4, "scan8"),
    "mipnerf360": (12, 8, "bicycle"),
    "blender": (8, 1, "lego"),
}
ENV_KEYS = {
    "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
    "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
    "DTU_MASK_ROOT", "CALL_LOG", "FAKE_FAIL_TRAIN", "DRY_RUN", "MATRIX",
    "CORRECTIONS",
}


def clean_environment():
    return {
        key: value for key, value in os.environ.items()
        if key not in ENV_KEYS
        and not key.startswith(("GPU_", "MVS_INIT_", "MVS_GEOMETRY_"))
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
    stream.write(json.dumps({
        "kind": kind, "args": args,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) + "\n")

def value(flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]

if kind == "train.py":
    if os.environ.get("FAKE_FAIL_TRAIN"):
        print("Insufficient geometry controls (simulated)", file=sys.stderr)
        sys.exit(23)
    root = Path(value("-m"))
    iteration = value("--iterations")
    cloud = root / "point_cloud" / ("iteration_" + iteration) / "point_cloud.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("fake cloud", encoding="utf-8")
    (root / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
elif kind == "render.py":
    root = Path(value("-m"))
    iteration = value("--iteration")
    (root / "test" / ("ours_" + iteration) / "renders").mkdir(parents=True)
    (root / ("metrics_" + iteration + ".txt")).write_text("fake metrics", encoding="utf-8")
'''


class TwingsLauncherTest(unittest.TestCase):
    def run_script(self, name, *, repo=REPO_ROOT, env=None, args=(), cwd=None):
        return subprocess.run(
            ["bash", str(repo / SCRIPT_DIR / name), *args],
            cwd=cwd or repo.parent, env=env or clean_environment(),
            text=True, capture_output=True, timeout=30,
        )

    def fixture(self, directory, dataset):
        root = Path(directory)
        repo = root / "repo with spaces"
        for relative in (SCRIPT_DIR, Path("scripts/mvs_vcd_vcp"), Path("scripts/vcd_vcp")):
            shutil.copytree(REPO_ROOT / relative, repo / relative)
        common = repo / "scripts/all"
        common.mkdir()
        shutil.copyfile(
            REPO_ROOT / "scripts/all/train_mvsdrop_benchmark.sh",
            common / "train_mvsdrop_benchmark.sh",
        )
        scene = DATASETS[dataset][2]
        data = root / "data with spaces"
        (data / (scene + "_colmap" if dataset == "dtu" else scene)).mkdir(parents=True)
        fake = root / "fake python with spaces"
        fake.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON, encoding="utf-8")
        fake.chmod(0o755)
        log = root / "calls.jsonl"
        env = clean_environment()
        env.update(PYTHON_BIN=str(fake), DATA_ROOT=str(data), SCENES=scene, CALL_LOG=str(log))
        return repo, env, log

    def test_shell_syntax(self):
        for path in (REPO_ROOT / SCRIPT_DIR).glob("*.sh"):
            with self.subTest(script=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_four_dataset_previews_are_tps_with_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in DATASETS.items():
                with self.subTest(dataset=dataset):
                    env = dict(clean_environment(), OUTPUT_ROOT=str(output), DRY_RUN="1")
                    result = self.run_script(f"train_{dataset}.sh", env=env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command = shlex.split(result.stdout)
                    self.assertIn(f"OUTPUT_ROOT={output}/tps/{dataset}{views}", command)
                    self.assertIn("DENSITY_CONTROL=vcd_vcp", command)
                    self.assertIn("MODE=mvs", command)
                    self.assertIn("MVS_INIT_STRATEGY=replace", command)
                    self.assertEqual(last_value(command, "--mvs_geometry_correction"), "tps")
            self.assertFalse(output.exists())

    def test_correction_modes_separate_output_and_override_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for mode in ("none", "similarity", "tps", "tps_consistent", "tps_local"):
                with self.subTest(mode=mode):
                    env = dict(clean_environment(), OUTPUT_ROOT=str(output), MVS_GEOMETRY_CORRECTION=mode)
                    result = self.run_script("train_llff.sh", env=env, args=(
                        "--dry-run", "--mvs_geometry_correction", "invalid",
                        "--mvs_geometry_correction=adaptive", "--density_control", "none",
                    ))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command = shlex.split(result.stdout)
                    self.assertEqual(command.count("--mvs_geometry_correction"), 1)
                    self.assertEqual(last_value(command, "--mvs_geometry_correction"), mode)
                    self.assertIn(f"OUTPUT_ROOT={output}/{mode}/llff3", command)
                    self.assertNotIn("--density_control", command)
            self.assertFalse(output.exists())

    def test_preview_preserves_passthrough_spaces_and_optional_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                clean_environment(), N_VIEWS="6", GPU_ID="3",
                MVS_GEOMETRY_MAX_CONTROLS="128", MVS_GEOMETRY_SMOOTHING="0.01",
                OUTPUT_ROOT=str(Path(directory) / "not created"),
            )
            literal = "$(touch should_not_exist); * ' quoted"
            result = self.run_script("train_llff.sh", env=env, cwd=directory, args=(
                "--dry-run", "--literal", literal, "--empty", "",
                "--mvs_geometry_max_image_size", "800",
            ))
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout)
            for flag, value in {
                "--literal": literal, "--empty": "", "--mvs_geometry_max_controls": "128",
                "--mvs_geometry_smoothing": "0.01", "--mvs_geometry_max_image_size": "800",
            }.items():
                self.assertEqual(last_value(command, flag), value)
            self.assertIn("N_VIEWS=6", command)
            self.assertIn("GPU_ID=3", command)
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_comparison_defaults_to_27_dispatches_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            env = dict(clean_environment(), OUTPUT_ROOT=str(output), N_VIEWS="999")
            result = self.run_script("run_comparison.sh", env=env, args=("--dry-run",))
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(commands), 27)
            paths = {next(token for token in command if token.startswith("OUTPUT_ROOT=")) for command in commands}
            self.assertEqual(len(paths), 27)
            for mode in ("none", "similarity", "tps"):
                self.assertEqual(sum(last_value(command, "--mvs_geometry_correction") == mode for command in commands), 9)
            self.assertTrue(all("N_VIEWS=999" not in command for command in commands))
            self.assertFalse(output.exists())

    def test_comparison_subset_and_gpu_precedence(self):
        env = dict(
            clean_environment(), MATRIX="llff:3 llff:6 dtu:9 blender:8",
            CORRECTIONS="none tps", GPU_ID="7", GPU_LLFF="2", GPU_LLFF_6="4", GPU_DTU_9="5",
        )
        result = self.run_script("run_comparison.sh", env=env, args=("--dry-run",))
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 8)
        for command, gpu in zip(commands, (2, 2, 4, 4, 5, 5, 7, 7)):
            self.assertIn(f"GPU_ID={gpu}", command)

    def test_invalid_configuration_fails_before_dispatch(self):
        for name, changes, args in (
            ("train_llff.sh", {"MVS_GEOMETRY_CORRECTION": "adaptive"}, ()),
            ("train_llff.sh", {"N_VIEWS": "0"}, ()),
            ("train_llff.sh", {"GPU_ID": "0,1"}, ()),
            ("train_llff.sh", {}, ("--n_views", "6")),
            ("train_llff.sh", {}, ("--mvs_geometry_correction",)),
            ("run_comparison.sh", {"MATRIX": "llff:3 invalid:6"}, ()),
            ("run_comparison.sh", {"MATRIX": "llff:3 llff:3"}, ()),
            ("run_comparison.sh", {"CORRECTIONS": "none invalid"}, ()),
            ("run_comparison.sh", {"CORRECTIONS": "tps tps"}, ()),
            ("run_comparison.sh", {"DRY_RUN": "yes"}, ()),
        ):
            with self.subTest(script=name, changes=changes, args=args):
                env = dict(clean_environment(), DRY_RUN="1", **{key: value for key, value in changes.items() if key != "DRY_RUN"})
                env.update(changes)
                result = self.run_script(name, env=env, args=args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")

    def test_actual_dispatch_reuses_unchanged_training_and_evaluation(self):
        for dataset, (views, resolution, scene) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(MODE="dropgaussian", DENSITY_CONTROL="none", GPU_ID="4")
                result = self.run_script(f"train_{dataset}.sh", repo=repo, env=env, args=(
                    "--init_mode", "colmap_dense", "--mvs_init_strategy", "adaptive",
                    "--density_control", "none", "--mvs_geometry_correction", "none",
                ))
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = ["train.py", "render.py"]
                if dataset == "dtu":
                    expected.append("evaluate_dtu.py")
                expected.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], expected)
                train = calls[0]
                args = train["args"]
                for flag, value in {
                    "--init_mode": "mvs_anywhere", "--mvs_init_strategy": "replace",
                    "--density_control": "vcd_vcp", "--mvs_geometry_correction": "tps",
                    "--n_views": str(views), "-r": str(resolution), "--iterations": "10000",
                }.items():
                    self.assertEqual(last_value(args, flag), value)
                self.assertEqual(train["gpu"], "4")
                self.assertEqual(Path(last_value(args, "-m")), repo / "output/twings/tps" / f"{dataset}{views}" / scene)

    def test_failed_geometry_stops_comparison_without_fallback_or_render(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3", CORRECTIONS="tps none", FAKE_FAIL_TRAIN="1")
            result = self.run_script("run_comparison.sh", repo=repo, env=env)
            self.assertEqual(result.returncode, 23)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])
            self.assertEqual(last_value(calls[0]["args"], "--mvs_geometry_correction"), "tps")

    def test_existing_outputs_are_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            first = self.run_script("train_llff.sh", repo=repo, env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            original_calls = log.read_text()
            repeated = self.run_script("train_llff.sh", repo=repo, env=env)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("fresh OUTPUT_ROOT", repeated.stderr)
            self.assertEqual(log.read_text(), original_calls)
            preview = self.run_script("train_llff.sh", repo=repo, env=env, args=("--dry-run",))
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(log.read_text(), original_calls)

    def test_owned_cli_options_cannot_escape_output_or_change_protocol(self):
        variants = (
            ("-m", "elsewhere"), ("-melsewhere",), ("--model_path", "elsewhere"),
            ("--model_path=elsewhere",), ("--model", "elsewhere"), ("--model=elsewhere",),
            ("-s", "other-source"), ("-sother-source",), ("--source_path", "other-source"),
            ("--source_path=other-source",), ("--source", "other-source"),
            ("-r", "2"), ("-r2",), ("--resolution", "2"), ("--resolution=2",),
            ("--resol", "2"), ("--iterations", "2"), ("--iterations=2",),
            ("--iter", "2"), ("--iter=2",), ("--n_views", "6"), ("--n_views=6",),
            ("--n_v", "6"), ("--n_v=6",), ("--dataset_name", "blender"),
            ("--dataset_name=blender",), ("--dataset", "blender"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for args in variants:
                with self.subTest(args=args):
                    env = dict(clean_environment(), OUTPUT_ROOT=str(output))
                    result = self.run_script("train_llff.sh", env=env, args=args, cwd=directory)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("Use ", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertFalse(output.exists())
                    self.assertFalse((Path(directory) / "elsewhere").exists())

    def test_comparison_rejects_model_path_override_before_any_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            env = dict(clean_environment(), MATRIX="llff:3", OUTPUT_ROOT=str(output))
            result = self.run_script("run_comparison.sh", env=env, args=("--model=elsewhere",))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("OUTPUT_ROOT", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
