"""Direct-MVS density-control launchers: exercise dispatch without GPU training."""

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
SCRIPT_DIR = Path("scripts/mvs_vcd_vcp")
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
ENV_KEYS = {
    "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
    "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
    "DTU_MASK_ROOT", "CALL_LOG", "FAKE_FAIL_TRAIN", "TEST_PYTHON",
    "TEST_RECORDER", "TEST_FAIL_SETTING", "DRY_RUN",
}


def clean_environment():
    return {
        key: value for key, value in os.environ.items()
        if key not in ENV_KEYS
        and not key.startswith(("GPU_", "MVS_INIT_"))
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
            "MODE", "DATASET", "MVS_INIT_STRATEGY", "CUDA_VISIBLE_DEVICES",
        )},
    }) + "\n")

def value(flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]

if kind == "train.py":
    if os.environ.get("FAKE_FAIL_TRAIN"):
        print("Missing MVS cache (simulated loader failure)", file=sys.stderr)
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


class DirectMVSDensityLauncherTest(unittest.TestCase):
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

    def test_shell_syntax_and_expected_entrypoints(self):
        for name in (
            "_run.sh", "train_all_views.sh",
            *(f"train_{dataset}.sh" for dataset in DATASETS),
        ):
            with self.subTest(name=name):
                script = REPO_ROOT / SCRIPT_DIR / name
                self.assertTrue(script.is_file())
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_defaults_run_all_38_scenes_with_direct_mvs_and_density_control(self):
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
                    self.assertEqual(last_value(args, "--density_control"), "vcd_vcp")
                    self.assertEqual(last_value(args, "--init_mode"), "mvs_anywhere")
                    self.assertEqual(last_value(args, "--mvs_init_strategy"), "replace")
                    self.assertEqual(last_value(args, "--mvs_init_max_points"), "100000")
                    self.assertEqual(last_value(args, "--n_views"), str(views))
                    self.assertEqual(last_value(args, "-r"), str(resolution))
                    self.assertEqual(last_value(args, "--iterations"), "10000")
                    self.assertEqual(train["env"]["MODE"], "mvs")
                    self.assertEqual(train["env"]["MVS_INIT_STRATEGY"], "replace")
                    self.assertEqual(train["env"]["DATASET"], dataset)
                    self.assertEqual(
                        Path(last_value(args, "-m")).parent,
                        repo / "output/mvs_vcd_vcp" / f"{dataset}{views}",
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

    def test_conflicting_environment_and_cli_cannot_reenable_adaptive_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff", ["fern"])
            env.update(SCENES="fern", MODE="mvsdrop", MVS_INIT_STRATEGY="auto")
            result = self.run_script(
                repo, "llff", env,
                "--init_mode", "colmap_dense", "--mvs_init_strategy", "adaptive",
                "--density_control", "none",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            train = next(c for c in calls(log) if c["kind"] == "train.py")
            self.assertEqual(train["env"]["MODE"], "mvs")
            self.assertEqual(train["env"]["MVS_INIT_STRATEGY"], "replace")
            self.assertEqual(last_value(train["args"], "--density_control"), "vcd_vcp")
            self.assertEqual(
                train["args"][-4:],
                ["--init_mode", "mvs_anywhere", "--mvs_init_strategy", "replace"],
            )

    def test_overrides_paths_spaces_cache_and_gpu_are_preserved(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                scene = "scan8_colmap" if dataset == "dtu" else DATASETS[dataset][2].split()[0]
                repo, env, log = self.fixture(directory, dataset, [scene])
                output = Path(directory) / "custom output with spaces"
                cache = Path(directory) / "cache with spaces/points.ply"
                mask = Path(directory) / "masks with spaces"
                env.update(
                    SCENES=scene, DENSITY_CONTROL="vcp", N_VIEWS="6", GPU_ID="5",
                    ITERATIONS="42", RESOLUTION="2", SEED="9", PORT="6010",
                    OUTPUT_ROOT=str(output), MVS_INIT_PATH=str(cache),
                    MVS_INIT_MAX_POINTS="4321", DTU_MASK_ROOT=str(mask),
                )
                result = self.run_script(
                    repo, dataset, env, "--test-argument", "value with spaces",
                    "--literal", "$(touch should_not_exist); * ' quoted", "--empty", "",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                recorded = calls(log)
                train = next(c for c in recorded if c["kind"] == "train.py")
                args = train["args"]
                for flag, value in {
                    "--density_control": "vcp", "--n_views": "6", "--iterations": "42",
                    "-r": "2", "--seed": "9", "--port": "6010",
                    "--mvs_init_path": str(cache), "--mvs_init_max_points": "4321",
                    "--test-argument": "value with spaces", "--empty": "",
                    "--literal": "$(touch should_not_exist); * ' quoted",
                }.items():
                    self.assertEqual(last_value(args, flag), value)
                self.assertEqual(train["env"]["CUDA_VISIBLE_DEVICES"], "5")
                self.assertEqual(Path(last_value(args, "-m")).parent, output)
                self.assertFalse((Path(directory) / "should_not_exist").exists())
                if dataset == "dtu":
                    evaluation = next(c for c in recorded if c["kind"] == "evaluate_dtu.py")
                    self.assertEqual(last_value(evaluation["args"], "--mask-root"), str(mask))

    def test_density_ablations_have_distinct_direct_mvs_outputs(self):
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
                self.assertEqual(last_value(train["args"], "--mvs_init_strategy"), "replace")
                self.assertEqual(
                    Path(last_value(train["args"], "-m")).parent,
                    repo / "output" / f"mvs_{mode}" / "llff3",
                )

    def test_invalid_density_mode_fails_before_training_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff", ["fern"])
            env["DENSITY_CONTROL"] = "invalid"
            result = self.run_script(repo, "llff", env)
            self.assertEqual(result.returncode, 2)
            self.assertIn("DENSITY_CONTROL", result.stderr)
            self.assertFalse(log.exists())
            self.assertFalse((repo / "output").exists())

    def test_missing_cache_training_failure_does_not_fallback_or_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff", ["fern"])
            env.update(SCENES="fern", FAKE_FAIL_TRAIN="1")
            result = self.run_script(repo, "llff", env)
            self.assertEqual(result.returncode, 23)
            self.assertIn("Missing MVS cache", result.stdout + result.stderr)
            self.assertEqual([c["kind"] for c in calls(log)], ["train.py"])

    def test_help_describes_direct_mvs_without_dispatch_or_output(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                result = self.run_script(repo, dataset, env, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"scripts/mvs_vcd_vcp/train_{dataset}.sh", result.stdout)
                self.assertIn("replace", result.stdout)
                self.assertIn("DENSITY_CONTROL", result.stdout)
                self.assertNotIn("Defaults: adaptive", result.stdout)
                self.assertIn("train -> render -> evaluate -> summarize", result.stdout)
                self.assertFalse(log.exists())
                self.assertFalse((repo / "output").exists())


RECORDER = r'''
import json
import os
import sys
record = {
    "dataset": sys.argv[1], "args": sys.argv[2:],
    "env": {key: os.environ.get(key) for key in (
        "N_VIEWS", "GPU_ID", "OUTPUT_ROOT", "MODE", "DENSITY_CONTROL", "MVS_INIT_STRATEGY",
    )},
}
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
if sys.argv[1] + os.environ.get("N_VIEWS", "") == os.environ.get("TEST_FAIL_SETTING"):
    sys.exit(47)
'''


class DirectMVSDensityMatrixTest(unittest.TestCase):
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

    def test_all_nine_settings_force_direct_mvs_and_override_inherited_views(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            env.update(MODE="mvsdrop", MVS_INIT_STRATEGY="auto", N_VIEWS="99")
            result = self.run_script(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = calls(log)
            self.assertEqual([(c["dataset"], c["env"]["N_VIEWS"]) for c in recorded], list(SETTINGS))
            for call in recorded:
                self.assertEqual(call["env"]["MODE"], "mvs")
                self.assertEqual(call["env"]["MVS_INIT_STRATEGY"], "replace")
                self.assertEqual(call["env"]["GPU_ID"], "0")
                self.assertEqual(call["args"], [])

    def test_gpu_precedence_custom_output_parent_and_extra_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            output = Path(directory) / "results with spaces"
            env.update(
                OUTPUT_ROOT=str(output), GPU_ID="7", GPU_LLFF="2", GPU_LLFF_6="0",
                GPU_DTU="3", GPU_DTU_9="4", GPU_MIPNERF360_24="5",
                GPU_BLENDER="6", GPU_BLENDER_8="1", DENSITY_CONTROL="vcd",
            )
            args = ("--test-argument", "value with spaces", "--literal", "$(touch no); *", "--empty", "")
            result = self.run_script(repo, env, *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = calls(log)
            self.assertEqual([c["env"]["GPU_ID"] for c in recorded], ["2", "0", "2", "3", "3", "4", "7", "5", "1"])
            for call, (dataset, views) in zip(recorded, SETTINGS):
                self.assertEqual(call["env"]["OUTPUT_ROOT"], str(output / (dataset + views)))
                self.assertEqual(call["env"]["DENSITY_CONTROL"], "vcd")
                self.assertEqual(call["args"], list(args))
            self.assertFalse((Path(directory) / "no").exists())

    def test_dry_run_is_explicit_escaped_and_has_no_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            output = Path(directory) / "not created with spaces"
            env.update(OUTPUT_ROOT=str(output), GPU_ID="5", GPU_LLFF_9="3", MODE="mvsdrop", MVS_INIT_STRATEGY="auto")
            args = ("--test-argument", "value with spaces", "--literal", "semi;$(touch no)")
            before = set(Path(directory).rglob("*"))
            result = self.run_script(repo, env, "--dry-run", *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stdout.splitlines() if "train_" in line and ".sh" in line]
            self.assertEqual(len(lines), 9, result.stdout)
            for line, (dataset, views) in zip(lines, SETTINGS):
                command = shlex.split(line)
                self.assertIn(str(repo / SCRIPT_DIR / f"train_{dataset}.sh"), command)
                self.assertIn("MODE=mvs", command)
                self.assertIn("MVS_INIT_STRATEGY=replace", command)
                self.assertIn(f"N_VIEWS={views}", command)
                self.assertIn(f"GPU_ID={'3' if (dataset, views) == ('llff', '9') else '5'}", command)
                self.assertIn(f"OUTPUT_ROOT={output / (dataset + views)}", command)
                self.assertEqual(command[-len(args):], list(args))
            self.assertFalse(log.exists())
            self.assertEqual(set(Path(directory).rglob("*")), before)

    def test_invalid_gpu_is_rejected_before_any_dispatch(self):
        for gpu in ("0,1", "-1", "gpu0", "1 2", "0; touch unexpected"):
            with self.subTest(gpu=gpu), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory)
                env["GPU_BLENDER_8"] = gpu
                result = self.run_script(repo, env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GPU", result.stderr)
                self.assertFalse(log.exists())

    def test_cli_view_override_is_rejected_to_avoid_mislabeled_results(self):
        for args in (("--n_views", "6"), ("--n_views=6",)):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory)
                result = self.run_script(repo, env, *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--n_views", result.stderr)
                self.assertFalse(log.exists())

    def test_help_describes_new_directory_and_direct_mvs_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory)
            before = set(Path(directory).rglob("*"))
            result = self.run_script(repo, env, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("scripts/mvs_vcd_vcp/train_all_views.sh", result.stdout)
            self.assertIn("GPU_ID", result.stdout)
            self.assertIn("--dry-run", result.stdout)
            self.assertIn("replace", result.stdout)
            self.assertNotIn("Default: adaptive", result.stdout)
            self.assertFalse(log.exists())
            self.assertEqual(set(Path(directory).rglob("*")), before)

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
