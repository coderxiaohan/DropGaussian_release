"""Baseline-initialized VCD/VCP dispatch, without data, MVS caches, or a GPU."""

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
SCRIPT_DIR = Path("scripts/dropgaussian_vcd_vcp")
DATASETS = {
    "llff": (3, 8, "fern flower fortress horns leaves orchids room trex"),
    "dtu": (3, 4, "scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41 scan45 scan55 scan63 scan82 scan103 scan110 scan114"),
    "mipnerf360": (12, 8, "bicycle bonsai counter garden kitchen room stump"),
    "blender": (8, 1, "chair drums ficus hotdog lego materials mic ship"),
}
SETTINGS = (
    ("llff", "3"), ("llff", "6"), ("llff", "9"),
    ("dtu", "3"), ("dtu", "6"), ("dtu", "9"),
    ("mipnerf360", "12"), ("mipnerf360", "24"), ("blender", "8"),
)


def clean_environment():
    ignored = {
        "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
        "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
        "DTU_MASK_ROOT", "CALL_LOG", "TEST_PYTHON", "TEST_RECORDER",
        "TEST_FAIL_SETTING", "TEST_FAIL_TRAIN", "DRY_RUN",
    }
    return {
        key: value for key, value in os.environ.items()
        if key not in ignored and not key.startswith(("GPU_", "MVS_INIT_"))
    }


def calls(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def last_value(args, flag):
    return args[max(index for index, value in enumerate(args) if value == flag) + 1]


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
        "env": {key: os.environ.get(key) for key in (
            "MODE", "DATASET", "CUDA_VISIBLE_DEVICES",
        )},
    }) + "\n")

def value(flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]

if kind == "train.py":
    # Exercise the dispatch contract: baseline initialization must not request
    # a cache even if stale MVS environment settings are inherited.
    if value("--init_mode") != "colmap_dense" or "--mvs_init_path" in args:
        sys.exit(31)
    if os.environ.get("TEST_FAIL_TRAIN"):
        sys.exit(23)
    root = Path(value("-m"))
    iteration = value("--iterations")
    cloud = root / "point_cloud" / ("iteration_" + iteration) / "point_cloud.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("fake baseline cloud", encoding="utf-8")
    (root / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
elif kind == "render.py":
    root = Path(value("-m"))
    iteration = value("--iteration")
    (root / "test" / ("ours_" + iteration) / "renders").mkdir(parents=True)
    (root / ("metrics_" + iteration + ".txt")).write_text("fake metrics", encoding="utf-8")
'''

RECORDER = r'''
import json
import os
import sys
record = {
    "dataset": sys.argv[1], "args": sys.argv[2:],
    "env": {key: os.environ.get(key) for key in (
        "N_VIEWS", "GPU_ID", "OUTPUT_ROOT", "MODE", "DENSITY_CONTROL",
    )},
}
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
if sys.argv[1] + os.environ.get("N_VIEWS", "") == os.environ.get("TEST_FAIL_SETTING"):
    sys.exit(47)
'''


class BaselineDensityLauncherTest(unittest.TestCase):
    def fixture(self, directory, dataset, scenes=None):
        root = Path(directory)
        repo = root / "repo with spaces"
        for relative in (SCRIPT_DIR, Path("scripts/vcd_vcp")):
            shutil.copytree(REPO_ROOT / relative, repo / relative)
        common = repo / "scripts/all"
        common.mkdir()
        shutil.copyfile(
            REPO_ROOT / "scripts/all/train_mvsdrop_benchmark.sh",
            common / "train_mvsdrop_benchmark.sh",
        )
        data = root / "data with spaces"
        for scene in (scenes or DATASETS[dataset][2].split()):
            if dataset == "dtu":
                scene = scene.removesuffix("_colmap") + "_colmap"
            (data / scene).mkdir(parents=True)
        fake = root / "fake python with spaces"
        fake.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON, encoding="utf-8")
        fake.chmod(0o755)
        log = root / "calls.jsonl"
        env = clean_environment()
        env.update(PYTHON_BIN=str(fake), DATA_ROOT=str(data), CALL_LOG=str(log))
        return repo, env, log

    @staticmethod
    def run_script(repo, dataset, env, *args):
        return subprocess.run(
            ["bash", str(repo / SCRIPT_DIR / f"train_{dataset}.sh"), *args],
            env=env, cwd=repo.parent, capture_output=True, text=True, timeout=30,
        )

    def test_shell_syntax_and_all_six_entrypoints(self):
        for name in ("_run.sh", "train_all_views.sh", *(f"train_{d}.sh" for d in DATASETS)):
            with self.subTest(name=name):
                script = REPO_ROOT / SCRIPT_DIR / name
                self.assertTrue(script.is_file())
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_defaults_cover_all_38_scenes_without_mvs_cache_arguments(self):
        for dataset, (views, resolution, scene_names) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                result = self.run_script(repo, dataset, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                recorded = calls(log)
                trains = [call for call in recorded if call["kind"] == "train.py"]
                self.assertEqual(
                    [Path(last_value(call["args"], "-m")).name for call in trains],
                    scene_names.split(),
                )
                for train in trains:
                    args = train["args"]
                    for flag, value in {
                        "--density_control": "vcd_vcp", "--init_mode": "colmap_dense",
                        "--n_views": str(views), "-r": str(resolution), "--iterations": "10000",
                    }.items():
                        self.assertEqual(last_value(args, flag), value)
                    self.assertFalse(any(arg.startswith("--mvs_init_") for arg in args))
                    self.assertEqual(train["env"]["MODE"], "dropgaussian")
                    self.assertEqual(train["env"]["DATASET"], dataset)
                    self.assertEqual(
                        Path(last_value(args, "-m")).parent,
                        repo / "output/dropgaussian_vcd_vcp" / f"{dataset}{views}",
                    )
                    if dataset == "blender":
                        self.assertIn("--white_background", args)
                        self.assertIn("--rand_pcd", args)
                self.assertEqual(sum(c["kind"] == "render.py" for c in recorded), len(trains))
                self.assertEqual(
                    sum(c["kind"] == "evaluate_dtu.py" for c in recorded),
                    len(trains) if dataset == "dtu" else 0,
                )
                self.assertEqual(recorded[-1]["kind"], "summarize_mvsdrop_setting.py")

    def test_mvs_environment_and_cli_cannot_override_baseline_initialization(self):
        for mode in ("mvs", "mvsdrop"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "llff", ["fern"])
                missing = Path(directory) / "nonexistent cache/points.ply"
                env.update(
                    SCENES="fern", MODE=mode, MVS_INIT_STRATEGY="adaptive",
                    MVS_INIT_PATH=str(missing), MVS_INIT_MAX_POINTS="1234",
                )
                result = self.run_script(
                    repo, "llff", env, "--init_mode", "mvs_anywhere",
                    "--mvs_init_strategy", "adaptive", "--density_control", "none",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                train = next(c for c in calls(log) if c["kind"] == "train.py")
                self.assertEqual(train["env"]["MODE"], "dropgaussian")
                self.assertEqual(last_value(train["args"], "--init_mode"), "colmap_dense")
                self.assertEqual(last_value(train["args"], "--density_control"), "vcd_vcp")
                self.assertNotIn("--mvs_init_path", train["args"])
                self.assertNotIn("--mvs_init_max_points", train["args"])
                self.assertFalse(missing.exists())

    def test_paths_gpu_views_and_extra_arguments_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "dtu", ["scan8_colmap"])
            output = Path(directory) / "custom output with spaces"
            mask = Path(directory) / "masks with spaces"
            env.update(
                SCENES="scan8_colmap", DENSITY_CONTROL="vcp", N_VIEWS="9", GPU_ID="5",
                ITERATIONS="42", RESOLUTION="2", SEED="9", PORT="6010",
                OUTPUT_ROOT=str(output), DTU_MASK_ROOT=str(mask),
            )
            result = self.run_script(
                repo, "dtu", env, "--test-argument", "value with spaces",
                "--literal", "$(touch should_not_exist); * ' quoted", "--empty", "",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = calls(log)
            train = next(c for c in recorded if c["kind"] == "train.py")
            for flag, value in {
                "--density_control": "vcp", "--n_views": "9", "--iterations": "42",
                "-r": "2", "--seed": "9", "--port": "6010",
                "--test-argument": "value with spaces", "--empty": "",
                "--literal": "$(touch should_not_exist); * ' quoted",
            }.items():
                self.assertEqual(last_value(train["args"], flag), value)
            self.assertEqual(train["env"]["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(Path(last_value(train["args"], "-m")).parent, output)
            evaluation = next(c for c in recorded if c["kind"] == "evaluate_dtu.py")
            self.assertEqual(last_value(evaluation["args"], "--mask-root"), str(mask))
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_density_ablations_use_distinct_baseline_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff", ["fern"])
            env["SCENES"] = "fern"
            modes = ("none", "vcd", "vcp", "vcd_vcp")
            for mode in modes:
                result = self.run_script(repo, "llff", dict(env, DENSITY_CONTROL=mode))
                self.assertEqual(result.returncode, 0, result.stderr)
            trains = [c for c in calls(log) if c["kind"] == "train.py"]
            self.assertEqual(len(trains), len(modes))
            for train, mode in zip(trains, modes):
                self.assertEqual(last_value(train["args"], "--density_control"), mode)
                self.assertEqual(
                    Path(last_value(train["args"], "-m")).parent,
                    repo / "output" / f"dropgaussian_{mode}" / "llff3",
                )

    def test_invalid_density_or_failed_training_stops_workflow(self):
        for fail in (False, True):
            with self.subTest(train_failure=fail), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "llff", ["fern"])
                env["SCENES"] = "fern"
                env["TEST_FAIL_TRAIN" if fail else "DENSITY_CONTROL"] = "invalid"
                result = self.run_script(repo, "llff", env)
                self.assertEqual(result.returncode, 23 if fail else 2)
                if fail:
                    self.assertEqual([c["kind"] for c in calls(log)], ["train.py"])
                else:
                    self.assertIn("DENSITY_CONTROL", result.stderr)
                    self.assertFalse(log.exists())
                    self.assertFalse((repo / "output").exists())

    def test_help_explains_no_mvs_without_dispatch_or_writes(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                before = set(Path(directory).rglob("*"))
                result = self.run_script(repo, dataset, env, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"scripts/dropgaussian_vcd_vcp/train_{dataset}.sh", result.stdout)
                self.assertIn("colmap_dense", result.stdout)
                self.assertIn("MVSAnywhere", result.stdout)
                self.assertIn("FastGS-inspired", result.stdout)
                self.assertIn("Not an exact reproduction", result.stdout)
                self.assertIn("train -> render -> evaluate -> summarize", result.stdout)
                self.assertFalse(log.exists())
                self.assertEqual(set(Path(directory).rglob("*")), before)


class BaselineDensityMatrixTest(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        repo = root / "repo with spaces"
        scripts = repo / SCRIPT_DIR
        scripts.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / SCRIPT_DIR / "train_all_views.sh", scripts / "train_all_views.sh")
        for dataset in DATASETS:
            (scripts / f"train_{dataset}.sh").write_text(
                '#!/usr/bin/env bash\n'
                f'exec "$TEST_PYTHON" "$TEST_RECORDER" {dataset} "$@"\n', encoding="utf-8",
            )
        recorder = root / "record dispatch.py"
        recorder.write_text(RECORDER, encoding="utf-8")
        log = root / "calls.jsonl"
        env = clean_environment()
        env.update(TEST_PYTHON=sys.executable, TEST_RECORDER=str(recorder), CALL_LOG=str(log))
        return repo, env, log

    @staticmethod
    def run_script(repo, env, *args):
        return subprocess.run(
            ["bash", str(repo / SCRIPT_DIR / "train_all_views.sh"), *args],
            env=env, cwd=repo.parent, capture_output=True, text=True, timeout=30,
        )

    def test_nine_settings_force_baseline_with_gpu_precedence_and_custom_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            output = Path(directory) / "results with spaces"
            env.update(
                MODE="mvsdrop", N_VIEWS="99", OUTPUT_ROOT=str(output), GPU_ID="7",
                GPU_LLFF="2", GPU_LLFF_6="0", GPU_DTU="3", GPU_DTU_9="4",
                GPU_MIPNERF360_24="5", GPU_BLENDER="6", GPU_BLENDER_8="1",
                DENSITY_CONTROL="vcd",
            )
            args = ("--test-argument", "value with spaces", "--literal", "$(touch no); *", "--empty", "")
            result = self.run_script(repo, env, *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = calls(log)
            self.assertEqual([(c["dataset"], c["env"]["N_VIEWS"]) for c in recorded], list(SETTINGS))
            self.assertEqual([c["env"]["GPU_ID"] for c in recorded], ["2", "0", "2", "3", "3", "4", "7", "5", "1"])
            for call, (dataset, views) in zip(recorded, SETTINGS):
                self.assertEqual(call["env"]["MODE"], "dropgaussian")
                self.assertEqual(call["env"]["OUTPUT_ROOT"], str(output / (dataset + views)))
                self.assertEqual(call["env"]["DENSITY_CONTROL"], "vcd")
                self.assertEqual(call["args"], list(args))
            self.assertFalse((Path(directory) / "no").exists())

    def test_dry_run_is_explicit_escaped_and_help_has_no_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            output = Path(directory) / "not created with spaces"
            env.update(OUTPUT_ROOT=str(output), GPU_LLFF_9="3", MODE="mvsdrop")
            args = ("--test-argument", "value with spaces", "--literal", "semi;$(touch no)")
            before = set(Path(directory).rglob("*"))
            result = self.run_script(repo, env, "--dry-run", *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stdout.splitlines() if "train_" in line and ".sh" in line]
            self.assertEqual(len(lines), 9, result.stdout)
            for line, (dataset, views) in zip(lines, SETTINGS):
                command = shlex.split(line)
                self.assertIn(str(repo / SCRIPT_DIR / f"train_{dataset}.sh"), command)
                self.assertIn("MODE=dropgaussian", command)
                self.assertIn(f"N_VIEWS={views}", command)
                self.assertIn(f"GPU_ID={'3' if (dataset, views) == ('llff', '9') else '0'}", command)
                self.assertIn(f"OUTPUT_ROOT={output / (dataset + views)}", command)
                self.assertEqual(command[-len(args):], list(args))
            help_result = self.run_script(repo, env, "--help")
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("scripts/dropgaussian_vcd_vcp/train_all_views.sh", help_result.stdout)
            self.assertIn("MVSAnywhere", help_result.stdout)
            self.assertIn("GPU_ID", help_result.stdout)
            self.assertIn("--dry-run", help_result.stdout)
            self.assertFalse(log.exists())
            self.assertEqual(set(Path(directory).rglob("*")), before)

    def test_invalid_gpu_and_view_override_fail_before_dispatch(self):
        for gpu, args in (
            ("0,1", ()), ("-1", ()), ("gpu0", ()), ("0; touch no", ()),
            ("0", ("--n_views", "6")), ("0", ("--n_views=6",)),
        ):
            with self.subTest(gpu=gpu, args=args), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory)
                env["GPU_BLENDER_8"] = gpu
                result = self.run_script(repo, env, *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--n_views" if args else "GPU", result.stderr)
                self.assertFalse(log.exists())

    def test_failed_setting_stops_later_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            env["TEST_FAIL_SETTING"] = "llff6"
            result = self.run_script(repo, env)
            self.assertEqual(result.returncode, 47)
            self.assertEqual(
                [(c["dataset"], c["env"]["N_VIEWS"]) for c in calls(log)],
                [("llff", "3"), ("llff", "6")],
            )


if __name__ == "__main__":
    unittest.main()
