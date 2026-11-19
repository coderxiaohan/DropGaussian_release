"""Standalone DPP disables densification gating, not its protection evidence."""

from argparse import ArgumentParser
from dataclasses import replace
from types import SimpleNamespace
import unittest

from arguments import ModelParams, OptimizationParams
from utils.fastgs_pruning import (
    FastGSPruningConfig, dpp_protection_config, periodic_density_config,
    validate_dpp_initialization, validate_training_schedule,
)
from utils.view_consistent_density import DensityControlConfig


class DPPOnlyConfigurationTest(unittest.TestCase):
    def config(self, **kwargs):
        values = dict(strategy="fastgs", protect_egd=True, dpp_only=True)
        values.update(kwargs)
        return FastGSPruningConfig(**values)

    def test_cli_opt_in_and_safe_defaults(self):
        parser = ArgumentParser()
        model = ModelParams(parser)
        optimization = OptimizationParams(parser)
        self.assertFalse(FastGSPruningConfig.from_options(parser.parse_args([])).dpp_only)
        args = parser.parse_args([
            "--dpp_only", "--vcp_strategy", "fastgs", "--fastgs_protect_egd",
            "--density_control", "vcp", "--init_mode", "colmap_dense",
        ])
        config = FastGSPruningConfig.from_options(optimization.extract(args))
        self.assertEqual(config, self.config())
        validate_dpp_initialization(config, model.extract(args))
        self.assertTrue(config.as_dict()["dpp_only"])
        automatic = FastGSPruningConfig.from_options(SimpleNamespace(
            density_control="vcp", vcp_strategy="auto", dpp_only=True))
        self.assertEqual(automatic, self.config())
        for invalid in (None, 0, 1, "true"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "must be a bool"):
                self.config(dpp_only=invalid)

    def test_standalone_requires_protection_and_the_native_adc_mode(self):
        with self.assertRaisesRegex(ValueError, "requires protected pruning"):
            self.config(protect_egd=False)
        with self.assertRaisesRegex(ValueError, "requires vcp_strategy=fastgs"):
            self.config(strategy="none")
        for mode in ("none", "vcd", "vcd_vcp"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                periodic_density_config(DensityControlConfig(mode), self.config())
        self.assertEqual(periodic_density_config(DensityControlConfig("vcp"), self.config()).mode, "none")

    def test_protection_reuses_egd_support_without_densification_gating(self):
        density = DensityControlConfig("vcp", 0.65, 3)
        evidence = dpp_protection_config(density, self.config())
        self.assertEqual(evidence, replace(density, mode="vcd"))
        self.assertTrue(evidence.use_vcd)
        self.assertFalse(evidence.use_vcp)
        self.assertEqual(density.mode, "vcp")  # Immutable requested config.
        self.assertEqual(periodic_density_config(density, self.config()).mode, "none")

    def test_existing_combined_and_unprotected_paths_stay_unchanged(self):
        for config in (FastGSPruningConfig(), self.config(dpp_only=False),
                       FastGSPruningConfig(strategy="fastgs")):
            self.assertIsNone(dpp_protection_config(DensityControlConfig("vcd_vcp"), config))
        combined = self.config(dpp_only=False)
        self.assertEqual(periodic_density_config(DensityControlConfig("vcd_vcp"), combined).mode, "vcd")
        with self.assertRaisesRegex(ValueError, "requires density_control=vcd_vcp"):
            periodic_density_config(DensityControlConfig("vcp"), combined)

    def test_direct_cli_cannot_silently_use_mvs_initialization(self):
        for init_mode, complement in (("mvs_anywhere", "none"), ("mvs_anywhere", "coverage"),
                                      ("colmap_dense", "coverage"), (None, "none")):
            dataset = SimpleNamespace(init_mode=init_mode, mvs_complement_mode=complement)
            with self.subTest(init=init_mode, complement=complement), self.assertRaisesRegex(ValueError, "original initialization"):
                validate_dpp_initialization(self.config(), dataset)
            validate_dpp_initialization(self.config(dpp_only=False), dataset)
        validate_dpp_initialization(self.config(), SimpleNamespace(init_mode="colmap_dense", mvs_complement_mode="none"))

    def test_dpp_requires_fresh_native_adc_evidence_and_avoids_opacity_reset(self):
        options = SimpleNamespace(iterations=10000, densify_from_iter=500, densify_until_iter=10000,
                                  densification_interval=100, opacity_reset_interval=3000)
        validate_training_schedule(self.config(), options, True)
        with self.assertRaisesRegex(ValueError, "opacity reset"):
            validate_training_schedule(self.config(interval=3000), options)
        options.iterations = 20000
        with self.assertRaisesRegex(ValueError, "every FastGS pruning event"):
            validate_training_schedule(self.config(), options)


if __name__ == "__main__":
    unittest.main()
