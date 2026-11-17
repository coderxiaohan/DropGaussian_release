"""Exercise real launcher dispatch with fake training/evaluation executables."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "llff": (3, 8, "fern flower fortress horns leaves orchids room trex"),
    "dtu": (3, 4, "scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41 scan45 scan55 scan63 scan82 scan103 scan110 scan114"),
    "mipnerf360": (12, 8, "bicycle bonsai counter garden kitchen room stump"),
    "blender": (8, 1, "chair drums ficus hotdog lego materials mic ship"),
}
ENV_KEYS = (
    "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
    "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "GPU_ID", "SEED",
    "PORT", "DTU_MASK_ROOT", "MVS_INIT_PATH", "MVS_INIT_STRATEGY",
    "MVS_INIT_MAX_POINTS", "MVS_INIT_TOTAL_MAX_POINTS",
    "MVS_INIT_DENSE_BASE_THRESHOLD", "MVS_INIT_DENSE_RESIDUAL_POINTS",
)

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
            "MODE", "DATASET", "N_VIEWS", "RESOLUTION", "ITERATIONS",
            "OUTPUT_ROOT", "CUDA_VISIBLE_DEVICES",
        )},
    }) + "\n")

def value(flag):
    return args[args.index(flag) + 1]

if kind == "train.py":
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


class ViewConsistentLauncherTest(unittest.TestCase):
    def _fixture(self, directory, dataset, scenes=None):
        root = Path(directory)
        repo = root / "repo with spaces"
        scripts = repo / "scripts"
        shutil.copytree(REPO_ROOT / "scripts/vcd_vcp", scripts / "vcd_vcp")
        (scripts / "all").mkdir()
        shutil.copyfile(
            REPO_ROOT / "scripts/all/train_mvsdrop_benchmark.sh",
            scripts / "all/train_mvsdrop_benchmark.sh",
        )
        data = root / "data with spaces"
        for scene in (scenes or DATASETS[dataset][2].split()):
            scene = scene.removesuffix("_colmap") if dataset == "dtu" else scene
            (data / (scene + "_colmap" if dataset == "dtu" else scene)).mkdir(parents=True)
        fake = root / "fake python with spaces"
        fake.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON, encoding="utf-8")
        fake.chmod(0o755)
        log = root / "calls.jsonl"
        env = {key: value for key, value in os.environ.items() if key not in ENV_KEYS}
        env.update(PYTHON_BIN=str(fake), DATA_ROOT=str(data), CALL_LOG=str(log))
        return repo, env, log

    def _run(self, repo, dataset, env, *args):
        return subprocess.run(
            ["bash", str(repo / "scripts/vcd_vcp" / f"train_{dataset}.sh"), *args],
            env=env, cwd=repo.parent, capture_output=True, text=True, timeout=30,
        )

    @staticmethod
    def _calls(log):
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def _last_value(args, flag):
        return args[max(index for index, value in enumerate(args) if value == flag) + 1]

    def test_shell_syntax(self):
        for script in (REPO_ROOT / "scripts/vcd_vcp").glob("*.sh"):
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_all_four_defaults_keep_full_scene_protocol_and_separate_output(self):
        for dataset, (views, resolution, scene_names) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self._fixture(directory, dataset)
                result = self._run(repo, dataset, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self._calls(log)
                trains = [call for call in calls if call["kind"] == "train.py"]
                self.assertEqual(len(trains), len(scene_names.split()))
                self.assertEqual(
                    [Path(self._last_value(call["args"], "-m")).name for call in trains],
                    scene_names.split(),
                )
                for call in trains:
                    args = call["args"]
                    self.assertEqual(self._last_value(args, "--density_control"), "vcd_vcp")
                    self.assertEqual(self._last_value(args, "--init_mode"), "mvs_anywhere")
                    self.assertEqual(self._last_value(args, "--mvs_init_strategy"), "replace")
                    self.assertEqual(self._last_value(args, "--n_views"), str(views))
                    self.assertEqual(self._last_value(args, "-r"), str(resolution))
                    self.assertEqual(self._last_value(args, "--iterations"), "10000")
                    self.assertEqual(call["env"]["MODE"], "mvs")
                    self.assertEqual(call["env"]["DATASET"], dataset)
                    self.assertEqual(
                        Path(self._last_value(args, "-m")).parent,
                        repo / "output/mvs_vcd_vcp" / f"{dataset}{views}",
                    )
                    if dataset == "blender":
                        self.assertIn("--white_background", args)
                        self.assertIn("--rand_pcd", args)
                self.assertEqual(sum(call["kind"] == "render.py" for call in calls), len(trains))
                self.assertEqual(
                    sum(call["kind"] == "evaluate_dtu.py" for call in calls),
                    len(trains) if dataset == "dtu" else 0,
                )
                self.assertEqual(calls[-1]["kind"], "summarize_mvsdrop_setting.py")

    def test_overrides_spaces_cache_options_and_authoritative_density_mode(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                scene = "scan8_colmap" if dataset == "dtu" else DATASETS[dataset][2].split()[0]
                repo, env, log = self._fixture(directory, dataset, [scene])
                output = Path(directory) / "custom results with spaces"
                cache = Path(directory) / "cache with spaces" / "points.ply"
                mask = Path(directory) / "masks with spaces"
                env.update(
                    SCENES=scene, DENSITY_CONTROL="vcd", N_VIEWS="6", ITERATIONS="42",
                    RESOLUTION="2", GPU_ID="3", OUTPUT_ROOT=str(output),
                    MVS_INIT_PATH=str(cache), MVS_INIT_MAX_POINTS="4321",
                    MVS_INIT_TOTAL_MAX_POINTS="5432", MVS_INIT_DENSE_BASE_THRESHOLD="3210",
                    MVS_INIT_DENSE_RESIDUAL_POINTS="1234", DTU_MASK_ROOT=str(mask),
                    SEED="9", PORT="6010",
                )
                result = self._run(
                    repo, dataset, env, "--density_control", "none",
                    "--density_vcp_relative_threshold", "0.001",
                    "--test-argument", "value with spaces",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self._calls(log)
                train = next(call for call in calls if call["kind"] == "train.py")
                args = train["args"]
                self.assertEqual(self._last_value(args, "--density_control"), "vcd")
                self.assertEqual(args[-4:], ["--init_mode", "mvs_anywhere", "--mvs_init_strategy", "replace"])
                self.assertEqual(self._last_value(args, "--test-argument"), "value with spaces")
                self.assertEqual(self._last_value(args, "--n_views"), "6")
                self.assertEqual(self._last_value(args, "--iterations"), "42")
                self.assertEqual(self._last_value(args, "-r"), "2")
                self.assertEqual(self._last_value(args, "--seed"), "9")
                self.assertEqual(self._last_value(args, "--port"), "6010")
                self.assertEqual(self._last_value(args, "--mvs_init_path"), str(cache))
                self.assertEqual(self._last_value(args, "--mvs_init_max_points"), "4321")
                self.assertNotIn("--mvs_init_total_max_points", args)
                self.assertNotIn("--mvs_init_dense_base_threshold", args)
                self.assertNotIn("--mvs_init_dense_residual_points", args)
                self.assertEqual(train["env"]["CUDA_VISIBLE_DEVICES"], "3")
                self.assertEqual(Path(self._last_value(args, "-m")).parent, output)
                if dataset == "dtu":
                    evaluation = next(call for call in calls if call["kind"] == "evaluate_dtu.py")
                    self.assertEqual(self._last_value(evaluation["args"], "--mask-root"), str(mask))

    def test_density_ablations_have_distinct_default_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory, "llff", ["fern"])
            env["SCENES"] = "fern"
            for mode in ("none", "vcd", "vcp", "vcd_vcp"):
                result = self._run(repo, "llff", dict(env, DENSITY_CONTROL=mode))
                self.assertEqual(result.returncode, 0, result.stderr)
            trains = [call for call in self._calls(log) if call["kind"] == "train.py"]
            self.assertEqual(len(trains), 4)
            outputs = [self._last_value(call["args"], "-m") for call in trains]
            self.assertEqual(len(set(outputs)), 4)
            for call, mode in zip(trains, ("none", "vcd", "vcp", "vcd_vcp")):
                self.assertIn(f"/mvs_{mode}/llff3/", self._last_value(call["args"], "-m"))

    def test_initialization_modes_are_forwarded_and_output_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory, "llff", ["fern"])
            env["SCENES"] = "fern"
            for mode in ("dropgaussian", "mvs"):
                result = self._run(repo, "llff", dict(env, MODE=mode))
                self.assertEqual(result.returncode, 0, result.stderr)
            trains = [call for call in self._calls(log) if call["kind"] == "train.py"]
            self.assertEqual(self._last_value(trains[0]["args"], "--init_mode"), "colmap_dense")
            self.assertEqual(self._last_value(trains[1]["args"], "--mvs_init_strategy"), "replace")
            for call, prefix in zip(trains, ("dropgaussian", "mvs")):
                self.assertIn(f"/{prefix}_vcd_vcp/llff3/", self._last_value(call["args"], "-m"))

    def test_invalid_mode_exits_before_dispatch_or_output_creation(self):
        for variable in ("DENSITY_CONTROL", "MODE"):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self._fixture(directory, "llff", ["fern"])
                env[variable] = "invalid"
                result = self._run(repo, "llff", env)
                self.assertEqual(result.returncode, 2)
                self.assertIn(variable, result.stderr)
                self.assertFalse(log.exists())
                self.assertFalse((repo / "output").exists())

    def test_removed_adaptive_mode_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory, "llff", ["fern"])
            env["MODE"] = "mvsdrop"
            result = self._run(repo, "llff", env)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unavailable", result.stderr)
            self.assertFalse(log.exists())

    def test_common_runner_both_keeps_original_baseline_density(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory, "llff", ["fern"])
            env.update(MODE="both", DATASET="llff", SCENES="fern")
            result = subprocess.run(
                ["bash", str(repo / "scripts/all/train_mvsdrop_benchmark.sh")],
                env=env, cwd=repo.parent, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            trains = [call for call in self._calls(log) if call["kind"] == "train.py"]
            self.assertEqual(len(trains), 2)
            self.assertEqual(self._last_value(trains[0]["args"], "--init_mode"), "colmap_dense")
            self.assertEqual(self._last_value(trains[0]["args"], "--density_control"), "none")
            self.assertEqual(self._last_value(trains[1]["args"], "--init_mode"), "mvs_anywhere")
            self.assertEqual(self._last_value(trains[1]["args"], "--density_control"), "vcd_vcp")
            for call, mode in zip(trains, ("dropgaussian", "mvs")):
                self.assertEqual(
                    Path(self._last_value(call["args"], "-m")),
                    repo / "output/mvs_vcd_vcp/paired" / mode / "llff3/fern",
                )

    def test_conflicting_initializer_flags_are_removed_before_parser_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory, "llff", ["fern"])
            env.update(SCENES="fern", MVS_INIT_STRATEGY="adaptive")
            result = self._run(
                repo, "llff", env, "--init_mode=colmap_dense",
                "--mvs_init_strategy=auto", "--mvs_init_strategy", "adaptive",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            train = next(call for call in self._calls(log) if call["kind"] == "train.py")
            args = train["args"]
            self.assertEqual(args.count("--mvs_init_strategy"), 1)
            self.assertEqual(self._last_value(args, "--mvs_init_strategy"), "replace")
            self.assertNotIn("adaptive", args)
            self.assertNotIn("--mvs_init_strategy=auto", args)

    def test_help_has_no_dispatch_or_output_side_effects(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self._fixture(directory, dataset)
                result = self._run(repo, dataset, env, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("DENSITY_CONTROL", result.stdout)
                self.assertIn("train -> render -> evaluate -> summarize", result.stdout)
                self.assertFalse(log.exists())
                self.assertFalse((repo / "output").exists())


if __name__ == "__main__":
    unittest.main()
