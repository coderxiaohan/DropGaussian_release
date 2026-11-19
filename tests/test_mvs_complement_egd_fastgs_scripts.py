"""Descriptive combined-method entry points reuse the protected FastGS policy."""

from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import test_fastgs_egd_pruning_scripts as shared


SCRIPTS = Path("scripts/mvs_complement_egd_fastgs")


class MVSComplementEGDFastGSLauncherTest(unittest.TestCase):
    def run_script(self, dataset="llff", *, env=None, args=("--dry-run",), old=False):
        scripts = shared.SCRIPTS if old else SCRIPTS
        return subprocess.run(
            ["bash", str(shared.ROOT / scripts / f"train_{dataset}.sh"), *args],
            cwd=shared.ROOT.parent, env=env or shared.clean_environment(),
            text=True, capture_output=True, timeout=30,
        )

    def successful_command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return shlex.split(result.stdout)

    def test_all_entrypoints_syntax_help_and_default_policy(self):
        for name in ("_run.sh", *(f"train_{dataset}.sh" for dataset in shared.DATASETS)):
            subprocess.run(["bash", "-n", str(shared.ROOT / SCRIPTS / name)], check=True)
        for dataset, (views, _, _) in shared.DATASETS.items():
            with self.subTest(dataset=dataset):
                help_result = self.run_script(dataset, args=("--help",))
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn(str(SCRIPTS), help_result.stdout)
                self.assertIn("FASTGS_PROTECT_EGD=1", help_result.stdout)
                command = self.successful_command(self.run_script(dataset))
                suffix = (f"blender_random/blender{views}" if dataset == "blender"
                          else f"coverage/keep_base/{dataset}{views}")
                self.assertIn(f"OUTPUT_ROOT={shared.ROOT}/output/mvs_complement_egd_fastgs/{suffix}", command)
                self.assertIn(f"N_VIEWS={views}", command)
                self.assertEqual(command.count("--fastgs_protect_egd"), 1)
                for flag, value in shared.FASTGS_ARGS.items():
                    self.assertEqual(shared.last_value(command, flag), value)
                if dataset == "blender":
                    self.assertIn("MODE=dropgaussian", command)
                    self.assertFalse(any(arg.startswith("--mvs") for arg in command))
                else:
                    self.assertEqual(shared.last_value(command, "--mvs_complement_mode"), "coverage")

    def test_custom_gpu_views_output_and_exact_old_policy_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never created"
            env = dict(shared.clean_environment(), OUTPUT_ROOT=str(output), GPU_ID="5",
                       N_VIEWS="6", DATA_ROOT="/missing-data", PYTHON_BIN="/missing-python")
            for dataset in shared.DATASETS:
                with self.subTest(dataset=dataset):
                    new_command = self.successful_command(self.run_script(dataset, env=env))
                    old_command = self.successful_command(self.run_script(dataset, env=env, old=True))
                    self.assertEqual(new_command, old_command)
                    self.assertIn("GPU_ID=5", new_command)
                    self.assertIn("N_VIEWS=6", new_command)
                    suffix = ("blender_random/blender6" if dataset == "blender"
                              else f"coverage/keep_base/{dataset}6")
                    self.assertIn(f"OUTPUT_ROOT={output}/{suffix}", new_command)
            self.assertFalse(output.exists())

    def test_conflicting_environment_and_method_cli_are_rejected(self):
        env = dict(shared.clean_environment(), FASTGS_PROTECT_EGD="0")
        result = self.run_script(env=env)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        for override in (("--fastgs_protect_egd=false",), ("--vcp_strategy", "tbp"),
                         ("--density_control", "none"), ("--mvs_complement_mode", "none")):
            with self.subTest(override=override):
                result = self.run_script(args=("--dry-run", *override))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
