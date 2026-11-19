"""Exercise workflow paths and dispatch without datasets, CUDA, or training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_ROOT = REPO_ROOT / "scripts" / "all"
WORKFLOW_ENV = (
    "MATRIX", "MODES", "SEEDS", "SEED", "MASK_SEED", "DRY_RUN", "OUTPUT_ROOT",
    "DATASET", "DATA_ROOT", "SCENES", "N_VIEWS", "ITERATIONS", "RESOLUTION",
    "PYTHON_BIN", "PYTHONPATH", "DTU_MASK_ROOT",
    "MODE", "MVS_INIT_PATH", "MVS_INIT_MAX_POINTS", "MVS_INIT_STRATEGY",
    "TEST_TRAIN_EXIT", "TEST_DIRTY",
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
            "DATASET", "N_VIEWS", "MODE", "SEED", "MASK_SEED", "OUTPUT_ROOT"
        )},
    }) + "\n")

def value(flag):
    return args[args.index(flag) + 1]

if kind == "train.py":
    if os.environ.get("TEST_TRAIN_EXIT"):
        sys.exit(int(os.environ["TEST_TRAIN_EXIT"]))
    root = Path(value("-m"))
    iteration = value("--iterations")
    cloud = root / "point_cloud" / ("iteration_" + iteration) / "point_cloud.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("mock point cloud", encoding="utf-8")
    (root / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
elif kind == "render.py":
    root = Path(value("-m"))
    iteration = value("--iteration")
    (root / "test" / ("ours_" + iteration) / "renders").mkdir(parents=True)
    (root / ("metrics_" + iteration + ".txt")).write_text("mock metrics", encoding="utf-8")
'''


class AllWorkflowTest(unittest.TestCase):
    def _env(self, **updates):
        env = {key: value for key, value in os.environ.items() if key not in WORKFLOW_ENV}
        env.update(updates)
        return env

    def _run(self, path, *, cwd, env, args=()):
        return subprocess.run(
            ["bash", str(path), *args], cwd=cwd, env=env,
            text=True, capture_output=True, timeout=30,
        )

    def _fixture(self, directory):
        root = Path(directory)
        script_dir = root / "repo with spaces" / "scripts" / "all"
        script_dir.mkdir(parents=True)
        for name in ("run_all.sh", "run_mvsdrop_multiseed.sh", "run_mvsdrop_matrix.sh"):
            shutil.copyfile(ALL_ROOT / name, script_dir / name)
        (script_dir / "train_mvsdrop_benchmark.sh").write_text(
            '#!/usr/bin/env bash\nexec "$PYTHON_BIN" record_call.py "$@"\n',
            encoding="utf-8",
        )
        fake_bin = root / "bin"
        fake_bin.mkdir()
        git = fake_bin / "git"
        git.write_text(
            '#!/usr/bin/env bash\n'
            'case "$3" in\n'
            '  status) if [[ ${TEST_DIRTY:-0} == 1 ]]; then printf " M train.py\\n"; fi ;;\n'
            '  rev-parse) printf "test-frozen-commit\\n" ;;\n'
            '  *) exit 2 ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        git.chmod(0o755)
        fake_python = fake_bin / "record-python"
        fake_python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON, encoding="utf-8")
        fake_python.chmod(0o755)
        log = root / "calls.jsonl"
        env = self._env(
            PYTHON_BIN=str(fake_python), CALL_LOG=str(log),
            PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
            OUTPUT_ROOT=str(root / "results with spaces"),
        )
        return script_dir, env, log

    def _mvs_fixture(self, directory):
        script_dir, env, log = self._fixture(directory)
        shutil.copyfile(
            ALL_ROOT / "train_mvsdrop_benchmark.sh",
            script_dir / "train_mvsdrop_benchmark.sh",
        )
        mvs_dir = script_dir.parent / "mvs"
        shutil.copytree(REPO_ROOT / "scripts" / "mvs", mvs_dir)
        return mvs_dir, env, log

    def test_all_shell_scripts_parse(self):
        for path in sorted((REPO_ROOT / "scripts").rglob("*.sh")):
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_default_preview_is_single_run_all_settings_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "not-created"
            result = self._run(
                ALL_ROOT / "run_all.sh", cwd=directory,
                env=self._env(OUTPUT_ROOT=str(output_root), SEEDS="0 1 2"),
                args=("--dry-run",),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith("DRY-RUN")]
            self.assertEqual(len(lines), 18)
            self.assertTrue(all("seed=0 " in line for line in lines))
            for setting in (
                "llff:3", "llff:6", "llff:9", "dtu:3", "dtu:6", "dtu:9",
                "mipnerf360:12", "mipnerf360:24", "blender:8",
            ):
                self.assertEqual(sum(f"setting={setting} " in line for line in lines), 2)
            self.assertFalse(output_root.exists())

    def test_single_run_entry_rejects_multiple_seed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                ALL_ROOT / "run_all.sh", cwd=directory,
                env=self._env(SEED="0 1"), args=("--dry-run",),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("one non-negative integer", result.stderr)

    def test_single_run_dispatch_preserves_arguments_and_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            script_dir, env, log = self._fixture(directory)
            env.update(MATRIX="llff:3 blender:8", SEEDS="0 1 2")
            result = self._run(
                script_dir / "run_all.sh", cwd=directory, env=env,
                args=("--example", "argument with spaces"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 5)
            for call, (dataset, mode) in zip(calls[:4], (
                ("llff", "dropgaussian"), ("llff", "mvs"),
                ("blender", "dropgaussian"), ("blender", "mvs"),
            )):
                self.assertEqual(call["kind"], "record_call.py")
                self.assertEqual(call["args"][1:], ["--example", "argument with spaces"])
                self.assertEqual(call["env"]["SEED"], "0")
                self.assertEqual(call["env"]["MASK_SEED"], "0")
                self.assertEqual(call["env"]["DATASET"], dataset)
                self.assertEqual(call["env"]["MODE"], mode)
                views = "3" if dataset == "llff" else "8"
                self.assertEqual(call["env"]["OUTPUT_ROOT"], str(
                    Path(env["OUTPUT_ROOT"]) / "seed_0" / mode / (dataset + views)
                ))
            self.assertEqual(calls[-1]["args"], [
                str(script_dir / "summarize_mvsdrop_multiseed.py"),
                "--root", env["OUTPUT_ROOT"], "--seeds", "0",
            ])

    def test_dirty_checkout_does_not_start_work(self):
        with tempfile.TemporaryDirectory() as directory:
            script_dir, env, log = self._fixture(directory)
            env["TEST_DIRTY"] = "1"
            result = self._run(script_dir / "run_all.sh", cwd=directory, env=env)
            self.assertEqual(result.returncode, 2)
            self.assertIn("clean committed checkout", result.stderr)
            self.assertFalse(log.exists())

    def test_flat_matrix_custom_output_is_separated_per_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            script_dir, env, log = self._fixture(directory)
            env["MATRIX"] = "llff:3 blender:8"
            result = self._run(script_dir / "run_mvsdrop_matrix.sh", cwd=directory, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 5)
            expected = {
                str(Path(env["OUTPUT_ROOT"]) / mode / setting)
                for mode in ("dropgaussian", "mvs")
                for setting in ("llff3", "blender8")
            }
            self.assertEqual({call["env"]["OUTPUT_ROOT"] for call in calls[:4]}, expected)
            self.assertEqual(calls[-1]["args"], [
                str(script_dir / "summarize_mvsdrop_matrix.py"), "--root", env["OUTPUT_ROOT"],
            ])

    def test_relocated_benchmark_reaches_train_render_evaluate_and_summary(self):
        for dataset, scene in (("llff", "fern"), ("dtu", "scan8"),
                               ("mipnerf360", "bicycle"), ("blender", "lego")):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                _, env, log = self._fixture(directory)
                source = Path(directory) / "data" / (scene + "_colmap" if dataset == "dtu" else scene)
                source.mkdir(parents=True)
                env.update(DATASET=dataset, DATA_ROOT=str(source.parent), SCENES=scene)
                result = self._run(ALL_ROOT / "train_mvsdrop_benchmark.sh", cwd=directory, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                expected = ["train.py", "render.py"]
                if dataset == "dtu":
                    expected.append("evaluate_dtu.py")
                expected.append("summarize_mvsdrop_setting.py")
                self.assertEqual([call["kind"] for call in calls], expected)
                self.assertEqual(calls[0]["args"][0], str(REPO_ROOT / "train.py"))
                self.assertEqual(calls[1]["args"][0], str(REPO_ROOT / "render.py"))
                for call in calls[2:]:
                    self.assertEqual(Path(call["args"][0]).parent, ALL_ROOT)
                # Completed outputs must not be relabeled by a newer policy.
                log.unlink()
                result = self._run(ALL_ROOT / "train_mvsdrop_benchmark.sh", cwd=directory, env=env)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("fresh OUTPUT_ROOT", result.stderr)
                self.assertFalse(log.exists())

    def test_shared_runner_rejects_historical_outputs_for_every_density_policy(self):
        cases = (
            ("dropgaussian", "none", ()),
            ("mvs", "vcd", ()),
            ("mvs", "vcp", ()),
            ("mvs", "vcd_vcp", ()),
            ("mvs", "none", ("--density_control", "vcp", "--vcp_strategy", "fastgs")),
            ("mvs", "none", ("--density_control=vcd_vcp",)),
            ("mvs", "none", ("--density_con=vcd_vcp",)),
            ("mvs", "none", ("--density_c", "vcp")),
        )
        for mode, density, args in cases:
            with self.subTest(mode=mode, density=density, args=args), tempfile.TemporaryDirectory() as directory:
                _, env, log = self._fixture(directory)
                output = Path(env["OUTPUT_ROOT"])
                output.mkdir()
                marker = output / ".historical-policy"
                marker.write_text("unchanged", encoding="utf-8")
                env.update(MODE=mode, DENSITY_CONTROL=density, DATA_ROOT="/missing-data")
                result = self._run(ALL_ROOT / "train_mvsdrop_benchmark.sh", cwd=directory, env=env, args=args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("fresh OUTPUT_ROOT", result.stderr)
                self.assertFalse(log.exists())
                self.assertEqual(list(output.iterdir()), [marker])
                self.assertEqual(marker.read_text(), "unchanged")

    def test_paired_runner_checks_later_output_before_starting_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            _, env, log = self._fixture(directory)
            output = Path(env["OUTPUT_ROOT"])
            later = output / "mvs/llff3"
            later.mkdir(parents=True)
            marker = later / "historical.txt"
            marker.write_text("preserve", encoding="utf-8")
            env.update(MODE="both", DATASET="llff", N_VIEWS="3", DATA_ROOT="/missing-data")
            result = self._run(ALL_ROOT / "train_mvsdrop_benchmark.sh", cwd=directory, env=env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("fresh OUTPUT_ROOT", result.stderr)
            self.assertFalse((output / "dropgaussian").exists())
            self.assertFalse(log.exists())
            self.assertEqual(marker.read_text(), "preserve")

    def test_shared_runner_rejects_file_and_symlink_output_targets(self):
        for kind in ("file", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                _, env, log = self._fixture(directory)
                output = Path(env["OUTPUT_ROOT"])
                if kind == "file":
                    output.write_text("preserve", encoding="utf-8")
                else:
                    target = Path(directory) / "untouched"
                    target.mkdir()
                    output.symlink_to(target, target_is_directory=True)
                result = self._run(ALL_ROOT / "train_mvsdrop_benchmark.sh", cwd=directory, env=env)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("real directory", result.stderr)
                self.assertFalse(log.exists())
                if kind == "file":
                    self.assertEqual(output.read_text(), "preserve")
                else:
                    self.assertEqual(list(target.iterdir()), [])

    def test_shared_runner_help_documents_fresh_output_policy_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, env, log = self._fixture(directory)
            result = self._run(ALL_ROOT / "train_mvsdrop_benchmark.sh", cwd=directory, env=env, args=("--help",))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("including baseline runs", result.stdout)
            self.assertIn("MODE=both checks both", result.stdout)
            self.assertFalse(Path(env["OUTPUT_ROOT"]).exists())
            self.assertFalse(log.exists())

    def test_summary_clis_work_outside_repository_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("summarize_mvsdrop_setting.py", "summarize_mvsdrop_matrix.py",
                         "summarize_mvsdrop_multiseed.py"):
                with self.subTest(script=name):
                    result = subprocess.run(
                        [sys.executable, "-I", str(ALL_ROOT / name), "--help"],
                        cwd=directory, env=self._env(), capture_output=True,
                        text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_direct_mvs_wrappers_cover_all_scenes_and_force_replace(self):
        from scripts.all.summarize_mvsdrop_multiseed import SCENES

        settings = (("llff", 3, 8, 8), ("dtu", 3, 4, 15),
                    ("mipnerf360", 12, 8, 7), ("blender", 8, 1, 8))
        for dataset, views, resolution, count in settings:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                mvs_dir, env, log = self._mvs_fixture(directory)
                data_root = Path(directory) / "data"
                for scene in SCENES[dataset]:
                    (data_root / (scene + "_colmap" if dataset == "dtu" else scene)).mkdir(parents=True)
                env.pop("OUTPUT_ROOT")
                env.update(DATA_ROOT=str(data_root), MODE="dropgaussian", MVS_INIT_STRATEGY="adaptive")
                result = self._run(
                    mvs_dir / f"train_{dataset}.sh", cwd=directory, env=env,
                    args=("--init_mode", "colmap_dense", "--mvs_init_strategy", "auto"),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                train_calls = [call for call in calls if call["kind"] == "train.py"]
                self.assertEqual(len(train_calls), count)
                output_root = mvs_dir.parents[1] / "output/mvs" / f"{dataset}{views}"
                for call in train_calls:
                    args = call["args"]
                    self.assertEqual(call["env"]["MODE"], "mvs")
                    self.assertEqual(args[args.index("--n_views") + 1], str(views))
                    self.assertEqual(args[args.index("-r") + 1], str(resolution))
                    self.assertEqual(args[args.index("--iterations") + 1], "10000")
                    self.assertEqual(args[args.index("--mvs_init_max_points") + 1], "100000")
                    self.assertEqual(args[args.index("--density_control") + 1], "none")
                    self.assertEqual(args[-4:], ["--init_mode", "mvs_anywhere", "--mvs_init_strategy", "replace"])
                    self.assertEqual(Path(args[args.index("-m") + 1]).parent, output_root)
                self.assertEqual(sum(call["kind"] == "render.py" for call in calls), count)
                self.assertEqual(
                    sum(call["kind"] == "evaluate_dtu.py" for call in calls),
                    count if dataset == "dtu" else 0,
                )
                self.assertEqual(calls[-1]["kind"], "summarize_mvsdrop_setting.py")
                self.assertEqual(calls[-1]["args"][-4:], ["--experiment-root", str(output_root), "--iteration", "10000"])

    def test_direct_mvs_forwards_view_cache_and_resource_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            mvs_dir, env, log = self._mvs_fixture(directory)
            source = Path(directory) / "data" / "fern"
            source.mkdir(parents=True)
            cache = str(Path(directory) / "cache with spaces" / "points.ply")
            env.update(
                DATA_ROOT=str(source.parent), SCENES="fern", N_VIEWS="6",
                ITERATIONS="2000", RESOLUTION="4", MVS_INIT_MAX_POINTS="1234",
                MVS_INIT_PATH=cache,
            )
            result = self._run(mvs_dir / "train_llff.sh", cwd=directory, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            args = calls[0]["args"]
            for flag, expected in (
                ("--n_views", "6"), ("--iterations", "2000"), ("-r", "4"),
                ("--mvs_init_max_points", "1234"), ("--mvs_init_path", cache),
            ):
                self.assertEqual(args[args.index(flag) + 1], expected)
            self.assertEqual(Path(args[args.index("-m") + 1]), Path(env["OUTPUT_ROOT"]) / "fern")

    def test_direct_mvs_training_error_stops_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            mvs_dir, env, log = self._mvs_fixture(directory)
            source = Path(directory) / "data" / "lego"
            source.mkdir(parents=True)
            env.update(DATA_ROOT=str(source.parent), SCENES="lego", TEST_TRAIN_EXIT="7")
            result = self._run(mvs_dir / "train_blender.sh", cwd=directory, env=env)
            self.assertEqual(result.returncode, 7)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([call["kind"] for call in calls], ["train.py"])


if __name__ == "__main__":
    unittest.main()
