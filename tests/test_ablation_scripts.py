"""Fixed ablation launchers, exercised without data, GPUs or model training."""

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
SCRIPTS = Path("scripts/ablation")
VARIANTS = {
    "baseline": (False, "none"),
    "complement_only": (True, "none"),
    "egd_only": (False, "vcd"),
    "tbp_only": (False, "vcp"),
    "complement_egd": (True, "vcd"),
    "complement_tbp": (True, "vcp"),
    "egd_tbp": (False, "vcd_vcp"),
}
DATASETS = {
    "llff": (3, 8, "fern"),
    "dtu": (3, 4, "scan8"),
    "mipnerf360": (12, 8, "bicycle"),
    "blender": (8, 1, "lego"),
}
ALL_SCENES = {
    "llff": "fern flower fortress horns leaves orchids room trex",
    "dtu": "scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41 scan45 scan55 scan63 scan82 scan103 scan110 scan114",
    "mipnerf360": "bicycle bonsai counter garden kitchen room stump",
    "blender": "chair drums ficus hotdog lego materials mic ship",
}
ENV_KEYS = {
    "MODE", "DENSITY_CONTROL", "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS",
    "ITERATIONS", "RESOLUTION", "OUTPUT_ROOT", "PYTHON_BIN", "SEED", "PORT",
    "DTU_MASK_ROOT", "CALL_LOG", "FAKE_FAIL_TRAIN", "DRY_RUN", "MATRIX", "MODES",
    "CUDA_VISIBLE_DEVICES",
}


def clean_environment():
    return {
        key: value for key, value in os.environ.items()
        if key not in ENV_KEYS and not key.startswith(("GPU_", "MVS_"))
    }


def last_value(args, flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]


def commands(stdout):
    return [shlex.split(line) for line in stdout.splitlines() if line.strip()]


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


class AblationLauncherTest(unittest.TestCase):
    def run_script(self, variant, dataset="llff", *, repo=ROOT, env=None,
                   args=(), cwd=None, matrix=False):
        filename = "train_all_views.sh" if matrix else f"train_{dataset}.sh"
        return subprocess.run(
            ["bash", str(repo / SCRIPTS / variant / filename), *args],
            cwd=cwd or repo.parent, env=env or clean_environment(),
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

    def assert_rejected_without_dispatch(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_all_variant_entrypoints_exist_and_have_valid_shell_syntax(self):
        for variant in VARIANTS:
            for name in [*(f"train_{dataset}.sh" for dataset in DATASETS), "train_all_views.sh"]:
                with self.subTest(variant=variant, name=name):
                    path = ROOT / SCRIPTS / variant / name
                    self.assertTrue(path.is_file(), path)
                    subprocess.run(["bash", "-n", str(path)], check=True)

    def test_help_is_available_without_data_python_or_output_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for variant in VARIANTS:
                for matrix in (False, True):
                    with self.subTest(variant=variant, matrix=matrix):
                        result = self.run_script(variant, matrix=matrix, args=("--help",), env=dict(
                            clean_environment(), OUTPUT_ROOT=str(output), DATA_ROOT="/missing-data",
                            PYTHON_BIN="/missing-python"))
                        self.assert_success(result)
                        self.assertIn("OUTPUT_ROOT", result.stdout)
                        self.assertIn("GPU", result.stdout)
                        self.assertIn(variant, result.stdout)
            self.assertFalse(output.exists())

    def test_preview_defaults_have_isolated_method_setting_paths_and_no_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for variant in VARIANTS:
                for dataset, (views, _, _) in DATASETS.items():
                    with self.subTest(variant=variant, dataset=dataset):
                        result = self.run_script(variant, dataset, env=dict(
                            clean_environment(), OUTPUT_ROOT=str(output), DRY_RUN="1",
                            DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python"))
                        self.assert_success(result)
                        preview = commands(result.stdout)
                        self.assertEqual(len(preview), 1)
                        self.assertIn(f"OUTPUT_ROOT={output}/{variant}/{dataset}{views}", preview[0])
                        self.assertIn(f"N_VIEWS={views}", preview[0])
                        self.assertIn("GPU_ID=0", preview[0])
            self.assertFalse(output.exists())

    def test_actual_dispatch_has_fixed_method_and_matching_evaluation_protocol(self):
        for variant, (complement, density) in VARIANTS.items():
            for dataset, (views, resolution, scene) in DATASETS.items():
                with self.subTest(variant=variant, dataset=dataset), tempfile.TemporaryDirectory() as directory:
                    repo, env, log = self.fixture(directory, dataset)
                    env.update(GPU_ID="4", MODE="mvs" if complement else "dropgaussian",
                               DENSITY_CONTROL=density,
                               MVS_COMPLEMENT_MODE="coverage" if complement else "none")
                    result = self.run_script(variant, dataset, repo=repo, env=env)
                    self.assert_success(result)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    expected = ["train.py", "render.py"]
                    if dataset == "dtu":
                        expected.append("evaluate_dtu.py")
                    expected.append("summarize_mvsdrop_setting.py")
                    self.assertEqual([call["kind"] for call in calls], expected)
                    args = calls[0]["args"]
                    for flag, value in {
                        "--density_control": density, "--n_views": str(views),
                        "-r": str(resolution), "--iterations": "10000", "--dataset_name": dataset,
                    }.items():
                        self.assertEqual(last_value(args, flag), value)
                    effective_mvs = complement and dataset != "blender"
                    self.assertEqual(last_value(args, "--init_mode"),
                                     "mvs_anywhere" if effective_mvs else "colmap_dense")
                    if effective_mvs:
                        for flag, value in {
                            "--mvs_init_strategy": "replace", "--mvs_complement_mode": "coverage",
                            "--mvs_complement_max_points": "100000", "--mvs_complement_total_points": "0",
                            "--mvs_complement_candidate_points": "500000",
                            "--mvs_complement_coverage_pixels": "2.0", "--mvs_complement_depth_scale": "0.02",
                        }.items():
                            self.assertEqual(last_value(args, flag), value)
                    else:
                        self.assertFalse(any(arg.startswith("--mvs") for arg in args))
                    if dataset == "blender":
                        self.assertIn("--rand_pcd", args)
                        self.assertIn("--white_background", args)
                    else:
                        self.assertNotIn("--rand_pcd", args)
                    self.assertFalse(any(arg.startswith(("--mvs_surface", "--mvs_geometry")) for arg in args))
                    self.assertEqual(calls[0]["gpu"], "4")
                    setting = repo / "output/ablation" / variant / f"{dataset}{views}"
                    self.assertEqual(Path(last_value(args, "-m")), setting / scene)
                    self.assertEqual(last_value(calls[1]["args"], "-r"), str(resolution))
                    self.assertEqual(last_value(calls[1]["args"], "--iteration"), "10000")
                    self.assertEqual(Path(last_value(calls[-1]["args"], "--experiment-root")), setting)
                    self.assertTrue((setting / "metrics_mean.txt").is_file())
                    if dataset == "dtu":
                        evaluation = calls[2]["args"]
                        self.assertEqual(last_value(evaluation, "--scene"), scene)
                        self.assertEqual(Path(last_value(evaluation, "--model-path")), setting / scene)
                        self.assertEqual(last_value(evaluation, "--iteration"), "10000")

    def test_default_scene_lists_are_not_silently_reduced(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, dataset)
                env.pop("SCENES")
                for scene in ALL_SCENES[dataset].split():
                    name = scene + "_colmap" if dataset == "dtu" else scene
                    (Path(env["DATA_ROOT"]) / name).mkdir(exist_ok=True)
                result = self.run_script("baseline", dataset, repo=repo, env=env)
                self.assert_success(result)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                trained_scenes = [Path(last_value(call["args"], "-m")).name
                                  for call in calls if call["kind"] == "train.py"]
                self.assertEqual(trained_scenes, ALL_SCENES[dataset].split())

    def test_no_mvs_variants_and_all_blender_variants_ignore_irrelevant_mvs_settings(self):
        for variant, (complement, _) in VARIANTS.items():
            datasets = ("blender",) if complement else ("llff", "blender")
            for dataset in datasets:
                with self.subTest(variant=variant, dataset=dataset), tempfile.TemporaryDirectory() as directory:
                    repo, env, log = self.fixture(directory, dataset)
                    env.update(
                        MVS_INIT_PATH="/missing/cache.npz", MVS_INIT_STRATEGY="adaptive",
                        MVS_INIT_MAX_POINTS="invalid", MVS_COMPLEMENT_MAX_POINTS="invalid",
                        MVS_COMPLEMENT_TOTAL_POINTS="-1", MVS_COMPLEMENT_CANDIDATE_POINTS="0",
                        MVS_COMPLEMENT_COVERAGE_PIXELS="NaN", MVS_COMPLEMENT_DEPTH_SCALE="inf",
                        MVS_GEOMETRY_CORRECTION="tps", MVS_SURFACE_INIT="oas",
                    )
                    result = self.run_script(variant, dataset, repo=repo, env=env)
                    self.assert_success(result)
                    args = json.loads(log.read_text().splitlines()[0])["args"]
                    self.assertEqual(last_value(args, "--init_mode"), "colmap_dense")
                    self.assertFalse(any(arg.startswith("--mvs") for arg in args))

    def test_matching_method_environment_is_allowed_and_conflicts_rejected(self):
        for variant, (complement, density) in VARIANTS.items():
            expected = {"MODE": "mvs" if complement else "dropgaussian",
                        "DENSITY_CONTROL": density,
                        "MVS_COMPLEMENT_MODE": "coverage" if complement else "none"}
            for dataset in ("llff", "blender"):
                with self.subTest(variant=variant, dataset=dataset):
                    result = self.run_script(variant, dataset, env=dict(
                        clean_environment(), DRY_RUN="1", **expected))
                    self.assert_success(result)
            for key, bad in (
                ("MODE", "both"), ("MODE", "dropgaussian" if complement else "mvs"),
                ("DENSITY_CONTROL", "not-the-fixed-density"),
                ("DENSITY_CONTROL", "vcd" if density == "vcp" else "vcp"),
                ("MVS_COMPLEMENT_MODE", "concat"),
                ("MVS_COMPLEMENT_MODE", "none" if complement else "coverage"),
            ):
                with self.subTest(variant=variant, key=key):
                    env = dict(clean_environment(), DRY_RUN="1", **expected)
                    env[key] = bad
                    self.assert_rejected_without_dispatch(self.run_script(variant, env=env))

    def test_method_and_protocol_cli_overrides_cannot_be_silently_accepted(self):
        options = (
            "--init_mode", "--init_m", "--density_control", "--density_con",
            "--density_min_views", "--vcd", "--vcp", "--mvs_complement_mode", "--mvs_complement_m",
            "--mvs_init_path", "--mv", "--rand_pcd", "--white_background", "--dataset_name",
            "-m", "-melsewhere", "--model", "-s", "--source", "-r", "--resol",
            "--n_views", "--n_v", "--iterations", "--iter", "--seed", "--test_iterations",
            "--save_iterations", "--start_checkpoint", "--eval", "--images", "-w",
        )
        for variant in ("complement_only", "egd_only", "tbp_only"):
            for option in options:
                for args in ((option, "invalid"), (option + "=invalid",)):
                    with self.subTest(variant=variant, args=args):
                        result = self.run_script(variant, env=dict(clean_environment(), DRY_RUN="1"), args=args)
                        self.assert_rejected_without_dispatch(result)
            result = self.run_script(variant, args=("--dry-run", "--", "--density_control", "none"))
            self.assert_rejected_without_dispatch(result)

    def test_custom_views_gpu_budget_and_quoted_arguments_reach_actual_training(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            output = Path(directory) / "output with spaces"
            env.update(OUTPUT_ROOT=str(output), N_VIEWS="9", GPU_ID="3", ITERATIONS="1000", RESOLUTION="4",
                       MVS_COMPLEMENT_MAX_POINTS="120000", MVS_COMPLEMENT_TOTAL_POINTS="150000",
                       MVS_COMPLEMENT_CANDIDATE_POINTS="600000", MVS_COMPLEMENT_COVERAGE_PIXELS="3",
                       MVS_COMPLEMENT_DEPTH_SCALE="2e-2", MVS_INIT_PATH="/cache with spaces/cloud.npz")
            literal = "$(touch should_not_exist); * ' quoted"
            result = self.run_script("complement_egd", repo=repo, env=env, cwd=directory,
                                     args=("--literal", literal, "--empty", ""))
            self.assert_success(result)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            args = calls[0]["args"]
            for flag, value in {
                "--literal": literal, "--empty": "", "--n_views": "9", "--iterations": "1000", "-r": "4",
                "--mvs_complement_max_points": "120000", "--mvs_complement_total_points": "150000",
                "--mvs_complement_candidate_points": "600000", "--mvs_complement_coverage_pixels": "3",
                "--mvs_complement_depth_scale": "2e-2", "--mvs_init_path": "/cache with spaces/cloud.npz",
            }.items():
                self.assertEqual(last_value(args, flag), value)
            self.assertEqual(last_value(args, "-m"), str(output / "complement_egd/llff9/fern"))
            self.assertEqual(calls[0]["gpu"], "3")
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_default_matrix_has_nine_distinct_setting_paths_for_each_variant(self):
        expected_settings = {"llff3", "llff6", "llff9", "dtu3", "dtu6", "dtu9",
                             "mipnerf36012", "mipnerf36024", "blender8"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not created"
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    result = self.run_script(variant, matrix=True, args=("--dry-run",), env=dict(
                        clean_environment(), OUTPUT_ROOT=str(output), N_VIEWS="999"))
                    self.assert_success(result)
                    preview = commands(result.stdout)
                    self.assertEqual(len(preview), 9)
                    paths = {next(token.removeprefix("OUTPUT_ROOT=") for token in cmd
                                  if token.startswith("OUTPUT_ROOT=")) for cmd in preview}
                    self.assertEqual(paths, {str(output / variant / setting) for setting in expected_settings})
                    self.assertTrue(all("N_VIEWS=999" not in cmd for cmd in preview))
            self.assertFalse(output.exists())

    def test_matrix_subset_and_per_setting_gpu_precedence(self):
        result = self.run_script("complement_tbp", matrix=True, args=("--dry-run",), env=dict(
            clean_environment(), MATRIX="llff:3 llff:6 dtu:9 blender:8", GPU_ID="7",
            GPU_LLFF="2", GPU_LLFF_6="4", GPU_DTU_9="5"))
        self.assert_success(result)
        preview = commands(result.stdout)
        self.assertEqual(len(preview), 4)
        for cmd, gpu in zip(preview, (2, 4, 5, 7)):
            self.assertIn(f"GPU_ID={gpu}", cmd)

    def test_matrix_validates_every_entry_before_first_dispatch(self):
        invalid = (
            {"MATRIX": "llff:3 invalid:6"}, {"MATRIX": "llff:3 llff:3"}, {"MATRIX": " "},
            {"MATRIX": "llff:3 dtu:0"}, {"MATRIX": "llff:3 dtu:not-a-number"},
            {"MATRIX": "llff:3 dtu:6", "GPU_DTU_6": "0,1"},
            {"DENSITY_CONTROL": "invalid"}, {"MVS_COMPLEMENT_MODE": "invalid"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                result = self.run_script("complement_only", matrix=True,
                                         env=dict(clean_environment(), DRY_RUN="1", **changes))
                self.assert_rejected_without_dispatch(result)
        result = self.run_script("egd_only", matrix=True, args=("--dry-run", "--n_v", "6"))
        self.assert_rejected_without_dispatch(result)

    def test_actual_matrix_rejects_later_invalid_entries_before_training(self):
        invalid = (
            {"MATRIX": "llff:3 invalid:6"},
            {"MATRIX": "llff:3 dtu:6", "GPU_DTU_6": "not-a-gpu"},
            {"MATRIX": "llff:3 llff:3"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self.fixture(directory, "llff")
                env.update(changes)
                result = self.run_script("complement_only", repo=repo, env=env, matrix=True)
                self.assert_rejected_without_dispatch(result)
                self.assertFalse(log.exists())
                self.assertFalse((repo / "output").exists())

    def test_invalid_numeric_configuration_rejected_before_dispatch(self):
        invalid = (
            {"MVS_COMPLEMENT_MAX_POINTS": "0"}, {"MVS_COMPLEMENT_MAX_POINTS": "2.5"},
            {"MVS_COMPLEMENT_TOTAL_POINTS": "-1"}, {"MVS_COMPLEMENT_CANDIDATE_POINTS": "0"},
            {"MVS_COMPLEMENT_COVERAGE_PIXELS": "0"}, {"MVS_COMPLEMENT_COVERAGE_PIXELS": "NaN"},
            {"MVS_COMPLEMENT_DEPTH_SCALE": "inf"}, {"MVS_COMPLEMENT_DEPTH_SCALE": "1e999"},
            {"MVS_COMPLEMENT_DEPTH_SCALE": "2e"}, {"MVS_COMPLEMENT_COVERAGE_PIXELS": "2e-"},
            {"N_VIEWS": "0"}, {"GPU_ID": "0,1"}, {"DRY_RUN": "yes"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                env = dict(clean_environment(), DRY_RUN="1")
                env.update(changes)
                self.assert_rejected_without_dispatch(self.run_script("complement_only", env=env))

    def test_nonempty_outputs_are_preserved_and_different_variants_remain_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            output = Path(directory) / "results"
            env["OUTPUT_ROOT"] = str(output)
            old = output / "old_full_method"
            old.mkdir(parents=True)
            marker = old / "keep.txt"
            marker.write_text("original results", encoding="utf-8")
            self.assert_success(self.run_script("complement_only", repo=repo, env=env))
            original_log = log.read_text()
            repeated = self.run_script("complement_only", repo=repo, env=env)
            self.assertEqual(repeated.returncode, 2, repeated.stderr)
            self.assertEqual(log.read_text(), original_log)
            self.assertEqual(marker.read_text(), "original results")
            self.assert_success(self.run_script("complement_only", repo=repo, env=env, args=("--dry-run",)))
            self.assertEqual(log.read_text(), original_log)
            self.assert_success(self.run_script("egd_only", repo=repo, env=env))
            self.assertTrue((output / "complement_only/llff3/metrics_mean.txt").is_file())
            self.assertTrue((output / "egd_only/llff3/metrics_mean.txt").is_file())
            self.assertEqual(marker.read_text(), "original results")

    def test_training_failure_stops_matrix_without_render_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self.fixture(directory, "llff")
            env.update(MATRIX="llff:3 llff:6", FAKE_FAIL_TRAIN="1")
            result = self.run_script("complement_tbp", repo=repo, env=env, matrix=True)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])


if __name__ == "__main__":
    unittest.main()
