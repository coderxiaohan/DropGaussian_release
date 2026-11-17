"""Check the view/GPU launcher using recorded dispatches, without CUDA or data."""

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
SCRIPT = REPO_ROOT / "scripts/vcd_vcp/train_all_views.sh"
SETTINGS = (
    ("llff", "3"), ("llff", "6"), ("llff", "9"),
    ("dtu", "3"), ("dtu", "6"), ("dtu", "9"),
    ("mipnerf360", "12"), ("mipnerf360", "24"), ("blender", "8"),
)
ENV_KEYS = {
    "GPU_ID", "N_VIEWS", "OUTPUT_ROOT", "DENSITY_CONTROL", "MODE",
    "MVS_INIT_STRATEGY", "DRY_RUN", "CALL_LOG", "TEST_FAIL_SETTING",
    "TEST_PYTHON", "TEST_RECORDER",
}

RECORDER = r'''
import json
import os
import sys

dataset = sys.argv[1]
record = {
    "dataset": dataset,
    "args": sys.argv[2:],
    "env": {key: os.environ.get(key) for key in (
        "N_VIEWS", "GPU_ID", "OUTPUT_ROOT", "MODE", "DENSITY_CONTROL",
        "MVS_INIT_STRATEGY",
    )},
}
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
if dataset + os.environ.get("N_VIEWS", "") == os.environ.get("TEST_FAIL_SETTING"):
    sys.exit(47)
'''


class ViewConsistentMatrixTest(unittest.TestCase):
    def _fixture(self, directory):
        root = Path(directory)
        repo = root / "repo with spaces"
        scripts = repo / "scripts/vcd_vcp"
        scripts.mkdir(parents=True)
        shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
        for dataset in dict(SETTINGS):
            (scripts / f"train_{dataset}.sh").write_text(
                '#!/usr/bin/env bash\n'
                f'exec "$TEST_PYTHON" "$TEST_RECORDER" {dataset} "$@"\n',
                encoding="utf-8",
            )
        recorder = root / "record dispatch.py"
        recorder.write_text(RECORDER, encoding="utf-8")
        log = root / "calls.jsonl"
        env = {
            key: value for key, value in os.environ.items()
            if key not in ENV_KEYS and not key.startswith("GPU_")
        }
        env.update(
            TEST_PYTHON=sys.executable, TEST_RECORDER=str(recorder), CALL_LOG=str(log),
        )
        return repo, env, log

    @staticmethod
    def _run(repo, env, *args):
        return subprocess.run(
            ["bash", str(repo / "scripts/vcd_vcp" / SCRIPT.name), *args],
            cwd=repo.parent, env=env, text=True, capture_output=True, timeout=30,
        )

    @staticmethod
    def _calls(log):
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_cli_view_override_cannot_mislabel_matrix_outputs(self):
        for args in (("--n_views", "6"), ("--n_views=6",)):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self._fixture(directory)
                result = self._run(repo, env, *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--n_views", result.stderr)
                self.assertFalse(log.exists())

    def test_defaults_dispatch_all_nine_settings_from_an_external_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            env["N_VIEWS"] = "99"
            result = self._run(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = self._calls(log)
            self.assertEqual(
                [(call["dataset"], call["env"]["N_VIEWS"]) for call in calls],
                list(SETTINGS),
            )
            self.assertTrue(all(call["env"]["GPU_ID"] == "0" for call in calls))
            self.assertTrue(all(call["args"] == [] for call in calls))

    def test_gpu_override_precedence_including_explicit_gpu_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            env.update(
                GPU_ID="7", GPU_LLFF="2", GPU_LLFF_6="0", GPU_DTU="3",
                GPU_DTU_9="4", GPU_MIPNERF360_24="5", GPU_BLENDER="6",
                GPU_BLENDER_8="1",
            )
            result = self._run(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [call["env"]["GPU_ID"] for call in self._calls(log)],
                ["2", "0", "2", "3", "3", "4", "7", "5", "1"],
            )

    def test_global_gpu_fallback_is_used_for_all_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            env["GPU_ID"] = "12"
            result = self._run(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [call["env"]["GPU_ID"] for call in self._calls(log)], ["12"] * 9,
            )

    def test_custom_output_parent_is_separated_by_dataset_and_view_count(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            parent = Path(directory) / "results with spaces"
            env["OUTPUT_ROOT"] = str(parent)
            result = self._run(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs = [call["env"]["OUTPUT_ROOT"] for call in self._calls(log)]
            self.assertEqual(
                outputs, [str(parent / (dataset + views)) for dataset, views in SETTINGS],
            )
            self.assertEqual(len(set(outputs)), 9)

    def test_extra_arguments_and_existing_modes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            env.update(MODE="mvs", DENSITY_CONTROL="vcd", MVS_INIT_STRATEGY="auto")
            args = (
                "--iterations", "42", "--test-argument", "argument with spaces",
                "--literal", "$(touch should_not_exist); * ' quoted", "--empty", "",
            )
            result = self._run(repo, env, *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = self._calls(log)
            self.assertEqual(len(calls), 9)
            for call in calls:
                self.assertEqual(call["args"], list(args))
                self.assertEqual(call["env"]["MODE"], "mvs")
                self.assertEqual(call["env"]["DENSITY_CONTROL"], "vcd")
                self.assertEqual(call["env"]["MVS_INIT_STRATEGY"], "auto")
            self.assertFalse((Path(directory) / "should_not_exist").exists())

    def test_invalid_gpu_in_last_setting_is_rejected_before_any_dispatch(self):
        for gpu in ("0,1", "-1", "gpu0", "1 2", "0; touch unexpected"):
            with self.subTest(gpu=gpu), tempfile.TemporaryDirectory() as directory:
                repo, env, log = self._fixture(directory)
                output = Path(directory) / "not created"
                env.update(GPU_BLENDER_8=gpu, OUTPUT_ROOT=str(output))
                result = self._run(repo, env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GPU", result.stderr)
                self.assertFalse(log.exists())
                self.assertFalse(output.exists())
                self.assertFalse((Path(directory) / "unexpected").exists())

    def test_invalid_fallback_gpu_is_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            env["GPU_ID"] = "0,1"
            result = self._run(repo, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists())

    def test_dry_run_previews_nine_escaped_commands_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            output = Path(directory) / "not created with spaces"
            env.update(OUTPUT_ROOT=str(output), GPU_ID="2", GPU_LLFF_9="3")
            args = ("--test-argument", "value with spaces", "--literal", "semi;$(touch no)")
            before = set(Path(directory).rglob("*"))
            result = self._run(repo, env, "--dry-run", *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [
                line for line in result.stdout.splitlines()
                if any(f"train_{dataset}.sh" in line for dataset in dict(SETTINGS))
            ]
            self.assertEqual(len(lines), 9, result.stdout)
            for line, (dataset, views) in zip(lines, SETTINGS):
                command = shlex.split(line)
                self.assertIn(str(repo / "scripts/vcd_vcp" / f"train_{dataset}.sh"), command)
                self.assertIn(f"N_VIEWS={views}", command)
                self.assertIn(f"GPU_ID={'3' if (dataset, views) == ('llff', '9') else '2'}", command)
                self.assertIn(f"OUTPUT_ROOT={output / (dataset + views)}", command)
                self.assertEqual(command[-len(args):], list(args))
            self.assertFalse(log.exists())
            self.assertFalse(output.exists())
            self.assertEqual(set(Path(directory).rglob("*")), before)

    def test_help_has_no_writes_or_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            before = set(Path(directory).rglob("*"))
            result = self._run(repo, env, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GPU_ID", result.stdout)
            self.assertIn("--dry-run", result.stdout)
            self.assertFalse(log.exists())
            self.assertEqual(set(Path(directory).rglob("*")), before)

    def test_failure_aborts_before_later_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env, log = self._fixture(directory)
            env["TEST_FAIL_SETTING"] = "llff6"
            result = self._run(repo, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                [(call["dataset"], call["env"]["N_VIEWS"]) for call in self._calls(log)],
                [("llff", "3"), ("llff", "6")],
            )


if __name__ == "__main__":
    unittest.main()
