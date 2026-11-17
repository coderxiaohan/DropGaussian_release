"""Surface initialization launchers: isolated fake pipeline, no GPU or datasets."""

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
SCRIPTS = Path("scripts/surface_init")
DATASETS = {
    "llff": (3, 8, "fern"), "dtu": (3, 4, "scan8"),
    "mipnerf360": (12, 8, "bicycle"), "blender": (8, 1, "lego"),
}
ENV_KEYS = {
    "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
    "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
    "DTU_MASK_ROOT", "CALL_LOG", "FAKE_FAIL_TRAIN", "DRY_RUN", "MATRIX", "MODES",
}


def clean_environment():
    return {
        key: value for key, value in os.environ.items()
        if key not in ENV_KEYS and not key.startswith(("GPU_", "MVS_"))
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


class SurfaceInitLauncherTest(unittest.TestCase):
    def run_script(self, name, *, repo=ROOT, env=None, args=(), cwd=None):
        return subprocess.run(
            ["bash", str(repo / SCRIPTS / name), *args], cwd=cwd or repo.parent,
            env=env or clean_environment(), text=True, capture_output=True, timeout=30,
        )

    def fixture(self, directory, dataset):
        root = Path(directory)
        repo = root / "repo with spaces"
        for relative in (SCRIPTS, Path("scripts/mvs_vcd_vcp"), Path("scripts/vcd_vcp")):
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

    def test_four_dataset_dry_runs_default_to_oas_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for dataset, (views, _, _) in DATASETS.items():
                with self.subTest(dataset=dataset):
                    result = self.run_script(f"train_{dataset}.sh", env=dict(
                        clean_environment(), OUTPUT_ROOT=str(output), DRY_RUN="1"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cmd = shlex.split(result.stdout)
                    self.assertIn(f"OUTPUT_ROOT={output}/oas/{dataset}{views}", cmd)
                    for token in ("DENSITY_CONTROL=vcd_vcp", "MODE=mvs", "MVS_INIT_STRATEGY=replace"):
                        self.assertIn(token, cmd)
                    self.assertEqual(last_value(cmd, "--mvs_geometry_correction"), "none")
                    self.assertEqual(last_value(cmd, "--mvs_surface_init"), "oas")
                    self.assertEqual(last_value(cmd, "--mvs_surface_neighbors"), "32")
            self.assertFalse(output.exists())

    def test_modes_are_isolated_and_inherited_tps_does_not_apply(self):
        for mode in ("none", "oas"):
            with self.subTest(mode=mode):
                result = self.run_script("train_llff.sh", env=dict(
                    clean_environment(), MVS_SURFACE_INIT=mode,
                    MVS_GEOMETRY_CORRECTION="tps", DRY_RUN="1"))
                self.assertEqual(result.returncode, 0, result.stderr)
                cmd = shlex.split(result.stdout)
                self.assertEqual(last_value(cmd, "--mvs_surface_init"), mode)
                self.assertEqual(last_value(cmd, "--mvs_geometry_correction"), "none")
                self.assertTrue(any(token.endswith(f"/{mode}/llff3") for token in cmd))

    def test_passthrough_quotes_neighbors_views_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            literal = "$(touch should_not_exist); * ' quoted"
            result = self.run_script("train_llff.sh", cwd=directory, env=dict(
                clean_environment(), N_VIEWS="6", GPU_ID="3", MVS_SURFACE_NEIGHBORS="16"),
                args=("--dry-run", "--literal", literal, "--empty", ""))
            self.assertEqual(result.returncode, 0, result.stderr)
            cmd = shlex.split(result.stdout)
            self.assertEqual(last_value(cmd, "--literal"), literal)
            self.assertEqual(last_value(cmd, "--empty"), "")
            self.assertEqual(last_value(cmd, "--mvs_surface_neighbors"), "16")
            self.assertIn("N_VIEWS=6", cmd)
            self.assertIn("GPU_ID=3", cmd)
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_comparison_default_matrix_has_18_isolated_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            result = self.run_script("run_comparison.sh", env=dict(
                clean_environment(), OUTPUT_ROOT=str(output), N_VIEWS="999"), args=("--dry-run",))
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(commands), 18)
            paths = {next(token for token in cmd if token.startswith("OUTPUT_ROOT=")) for cmd in commands}
            self.assertEqual(len(paths), 18)
            for mode in ("none", "oas"):
                self.assertEqual(sum(last_value(cmd, "--mvs_surface_init") == mode for cmd in commands), 9)
            self.assertTrue(all("N_VIEWS=999" not in cmd for cmd in commands))
            self.assertFalse(output.exists())

    def test_comparison_gpu_precedence(self):
        result = self.run_script("run_comparison.sh", args=("--dry-run",), env=dict(
            clean_environment(), MATRIX="llff:3 llff:6 dtu:9 blender:8", MODES="none oas",
            GPU_ID="7", GPU_LLFF="2", GPU_LLFF_6="4", GPU_DTU_9="5"))
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 8)
        for cmd, gpu in zip(commands, (2, 2, 4, 4, 5, 5, 7, 7)):
            self.assertIn(f"GPU_ID={gpu}", cmd)

    def test_invalid_configuration_fails_without_dispatch(self):
        for name, changes in (
            ("train_llff.sh", {"MVS_SURFACE_INIT": "tps"}),
            ("train_llff.sh", {"MVS_SURFACE_NEIGHBORS": "2"}),
            ("train_llff.sh", {"MVS_SURFACE_NEIGHBORS": "3.5"}),
            ("train_llff.sh", {"N_VIEWS": "0"}),
            ("train_llff.sh", {"GPU_ID": "0,1"}),
            ("run_comparison.sh", {"MATRIX": "llff:3 invalid:6"}),
            ("run_comparison.sh", {"MATRIX": "llff:3 llff:3"}),
            ("run_comparison.sh", {"MODES": "none invalid"}),
            ("run_comparison.sh", {"MODES": "oas oas"}),
            ("run_comparison.sh", {"MODES": " "}),
            ("run_comparison.sh", {"DRY_RUN": "yes"}),
        ):
            with self.subTest(script=name, changes=changes):
                env = dict(clean_environment(), DRY_RUN="1")
                env.update(changes)
                result = self.run_script(name, env=env)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")

    def test_owned_cli_and_abbreviations_cannot_change_protocol(self):
        options = (
            "-m", "-melsewhere", "--model_path", "--model", "-s", "-selsewhere",
            "--source_path", "--source", "-r", "-r2", "--resolution", "--resol",
            "--iterations", "--iter", "--n_views", "--n_v", "--dataset_name", "--dataset",
            "--mvs_surface_init", "--mvs_surface_i", "--mvs_surface_neighbors",
            "--mvs_geometry_correction", "--mvs_geometry_smoothing", "--mvs_geo",
            "--density_control", "--density_con", "--init_mode", "--init_m",
            "--mvs_init_strategy", "--mvs_init_str",
        )
        for option in options:
            for args in ((option, "invalid"), (option + "=invalid",)):
                with self.subTest(args=args):
                    result = self.run_script("train_llff.sh", env=dict(clean_environment(), DRY_RUN="1"), args=args)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")

    def test_actual_fake_dispatch_keeps_training_evaluation_protocol(self):
        for dataset, (views, resolution, scene) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.update(MODE="dropgaussian", DENSITY_CONTROL="none", GPU_ID="4",
                           MVS_GEOMETRY_CORRECTION="tps", MVS_INIT_STRATEGY="adaptive")
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
                    "--init_mode": "mvs_anywhere", "--mvs_init_strategy": "replace",
                    "--density_control": "vcd_vcp", "--mvs_geometry_correction": "none",
                    "--mvs_surface_init": "oas", "--mvs_surface_neighbors": "32",
                    "--n_views": str(views), "-r": str(resolution), "--iterations": "10000",
                }.items():
                    self.assertEqual(last_value(args, flag), value)
                self.assertEqual(calls[0]["gpu"], "4")
                self.assertEqual(Path(last_value(args, "-m")), repo / "output/surface_init/oas" / f"{dataset}{views}" / scene)

    def test_failed_training_stops_without_fallback_or_render(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3", MODES="oas none", FAKE_FAIL_TRAIN="1")
            result = self.run_script("run_comparison.sh", repo=repo, env=env)
            self.assertEqual(result.returncode, 23)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])
            self.assertEqual(last_value(calls[0]["args"], "--mvs_surface_init"), "oas")

    def test_completed_outputs_cannot_be_reused_but_preview_is_read_only(self):
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

    def test_modes_do_not_reuse_each_others_fake_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3", ITERATIONS="1000", RESOLUTION="4")
            result = self.run_script("run_comparison.sh", repo=repo, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            trains = [call["args"] for call in calls if call["kind"] == "train.py"]
            self.assertEqual(len(trains), 2)
            self.assertEqual([last_value(args, "--mvs_surface_init") for args in trains], ["none", "oas"])
            self.assertNotEqual(last_value(trains[0], "-m"), last_value(trains[1], "-m"))
            for args in trains:
                self.assertEqual(last_value(args, "--iterations"), "1000")
                self.assertEqual(last_value(args, "-r"), "4")
                self.assertEqual(last_value(args, "--mvs_geometry_correction"), "none")


if __name__ == "__main__":
    unittest.main()
