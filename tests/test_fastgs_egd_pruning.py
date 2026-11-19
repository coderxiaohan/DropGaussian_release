"""CPU checks for the event-local EGD veto of FastGS candidates."""

from argparse import ArgumentParser
from dataclasses import replace
import json
import random
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from arguments import OptimizationParams
from utils.fastgs_pruning import (
    FastGSPruningConfig, evaluate_fastgs_pruning, periodic_density_config,
    validate_training_schedule,
)
from utils.view_consistent_density import DensityControlConfig


class EGDProtectionConfigTest(unittest.TestCase):
    def test_cli_default_protection_and_explicit_strategy_are_serializable(self):
        parser = ArgumentParser()
        options = OptimizationParams(parser)
        default = options.extract(parser.parse_args([]))
        self.assertEqual(default.vcp_strategy, "auto")
        self.assertTrue(FastGSPruningConfig.from_options(default).protect_egd)
        explicit = options.extract(parser.parse_args(["--vcp_strategy", "fastgs"]))
        self.assertFalse(FastGSPruningConfig.from_options(explicit).protect_egd)
        args = options.extract(parser.parse_args(["--vcp_strategy", "fastgs", "--fastgs_protect_egd"]))
        config = FastGSPruningConfig.from_options(args)
        self.assertTrue(config.protect_egd)
        self.assertTrue(json.loads(json.dumps(config.as_dict()))["protect_egd"])
        with self.assertRaisesRegex(ValueError, "requires vcp_strategy=fastgs"):
            FastGSPruningConfig.from_options(options.extract(parser.parse_args([
                "--vcp_strategy", "none", "--fastgs_protect_egd"])))

    def test_only_boolean_flag_and_egd_plus_pruning_allowed(self):
        for invalid in (None, 0, 1, "true"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "must be a bool"):
                FastGSPruningConfig(strategy="fastgs", protect_egd=invalid)
        config = FastGSPruningConfig(strategy="fastgs", protect_egd=True)
        for mode in ("none", "vcp", "vcd"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                periodic_density_config(DensityControlConfig(mode), config)
        original = DensityControlConfig("vcd_vcp", 0.6, 3)
        self.assertEqual(periodic_density_config(original, config), replace(original, mode="vcd"))

    def options(self, **kwargs):
        values = dict(iterations=10000, densify_until_iter=10000,
                      opacity_reset_interval=3000, densify_from_iter=500,
                      densification_interval=100)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_default_and_multiple_active_events_allowed(self):
        config = FastGSPruningConfig(strategy="fastgs", protect_egd=True)
        validate_training_schedule(config, self.options(), True)
        # Both 8k and 16k have ADC, with no joint opacity-reset event.
        validate_training_schedule(config, self.options(iterations=20000, densify_until_iter=19000))

    def test_every_pruning_event_must_have_current_egd_not_stale_mask(self):
        config = FastGSPruningConfig(strategy="fastgs", protect_egd=True)
        invalid_options = (
            self.options(densification_interval=300),
            self.options(densify_from_iter=8000),
            self.options(densify_until_iter=8000),
            self.options(iterations=20000),  # First event valid, second outside ADC.
            self.options(iterations=20000, densify_until_iter=16000),
        )
        for options in invalid_options:
            with self.subTest(options=vars(options)), self.assertRaisesRegex(ValueError, "every FastGS pruning event"):
                validate_training_schedule(config, options)
            # The unprotected schedule remains permitted (no current-ADC dependency).
            validate_training_schedule(replace(config, protect_egd=False), options)
        for interval in (False, 0, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                validate_training_schedule(config, self.options(densification_interval=interval))

    def test_protection_does_not_permit_reset_collisions_or_final_step_prune(self):
        for interval in (3000, 500, 10000):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                validate_training_schedule(FastGSPruningConfig(strategy="fastgs", interval=interval,
                                                                protect_egd=True), self.options(), True)


class EGDProtectionDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def probe(self, counts=(0, 2, 10, 0), opacity=(0.5, 0.5, 0.5, 0.05)):
        model = SimpleNamespace(get_xyz=torch.nn.Parameter(torch.zeros(len(counts), 3)),
                                get_opacity=torch.tensor(opacity)[:, None])
        camera = SimpleNamespace(uid=0, image_name="train", original_image=torch.zeros(3, 4, 4))
        render = mock.Mock(return_value={"render": torch.full((3, 4, 4), 0.2),
                                         "high_error_counts": torch.tensor(counts, dtype=torch.int32)})
        return model, [camera], render

    def evaluate(self, probe, mask=None, protect=True):
        model, cameras, render = probe
        return evaluate_fastgs_pruning(
            model, cameras, None, torch.zeros(3),
            FastGSPruningConfig(strategy="fastgs", protect_egd=protect),
            render_fn=render, protection_mask=mask,
        )

    def test_candidate_veto_does_not_modify_score_opacity_or_diagnostic_images(self):
        probe = self.probe()
        base = self.evaluate(probe, protect=False)
        mask = torch.tensor([True, False, True, False])
        decision = self.evaluate(probe, mask)
        self.assertEqual(base.prune_mask.tolist(), [False, False, True, True])
        self.assertEqual(decision.prune_mask.tolist(), [False, False, False, True])
        self.assertTrue(torch.equal(mask, torch.tensor([True, False, True, False])))
        for field in ("raw_score_min", "raw_score_max", "score_candidates", "opacity_candidates",
                      "view_loss_min", "view_loss_max", "evaluated_views", "score_backend"):
            self.assertEqual(decision.diagnostics[field], base.diagnostics[field])
        log = decision.diagnostics
        self.assertEqual(log["egd_protected_count"], 2)  # One protected noncandidate.
        self.assertEqual(log["candidates_before_protection"], 2)
        self.assertEqual(log["blocked_by_egd"], 1)
        self.assertEqual(log["blocked_score_candidates"], 1)
        self.assertEqual(log["blocked_opacity_candidates"], 0)
        self.assertEqual(log["requested_after_protection"], 1)
        self.assertEqual(log["requested"], 1)
        self.assertEqual(log["trial_renders"], 0)
        self.assertEqual(probe[2].call_count, 2)  # One per evaluation, no retries.
        json.dumps(log, allow_nan=False)

    def test_veto_covers_late_opacity_rule_but_does_not_change_native_cleanup(self):
        # These rows are post-ADC. Opacity .05 survived native .005 cleanup but
        # is a candidate for the additional FastGS .1 rule.
        decision = self.evaluate(self.probe(), torch.tensor([False, False, False, True]))
        self.assertEqual(decision.prune_mask.tolist(), [False, False, True, False])
        self.assertEqual(decision.diagnostics["blocked_opacity_candidates"], 1)
        self.assertEqual(decision.diagnostics["blocked_score_candidates"], 0)

    def test_overlapping_reasons_count_candidate_once(self):
        probe = self.probe(opacity=(0.5, 0.5, 0.05, 0.5))
        decision = self.evaluate(probe, torch.tensor([False, False, True, False]))
        self.assertFalse(decision.prune_mask.any())
        for field in ("blocked_by_egd", "blocked_score_candidates", "blocked_opacity_candidates",
                      "candidates_before_protection", "candidate_overlap"):
            self.assertEqual(decision.diagnostics[field], 1)
        self.assertEqual(decision.diagnostics["requested_after_protection"], 0)

    def test_guard_follows_veto_and_does_not_keep_unnecessary_second_point(self):
        probe = self.probe(counts=(0, 2, 1), opacity=(0.01, 0.08, 0.02))
        decision = self.evaluate(probe, torch.tensor([True, False, False]))
        self.assertEqual(decision.prune_mask.tolist(), [False, True, True])
        self.assertFalse(decision.diagnostics["prune_all_guard"])
        self.assertEqual(decision.diagnostics["requested_after_protection"], 2)
        # Without a protected survivor the existing highest-opacity guard applies.
        zero = self.evaluate(probe, torch.zeros(3, dtype=torch.bool))
        base = self.evaluate(probe, protect=False)
        self.assertTrue(torch.equal(zero.prune_mask, base.prune_mask))
        self.assertEqual(zero.diagnostics["requested_after_protection"], 3)
        self.assertEqual(zero.diagnostics["requested"], 2)
        self.assertTrue(zero.diagnostics["prune_all_guard"])

    def test_zero_protection_exactly_preserves_legacy_decision_and_rng(self):
        probe = self.probe()
        probe[0].get_xyz.grad = torch.ones_like(probe[0].get_xyz)
        rng, py_rng = torch.random.get_rng_state().clone(), random.getstate()
        protected = self.evaluate(probe, torch.zeros(4, dtype=torch.bool))
        base = self.evaluate(probe, protect=False)
        self.assertTrue(torch.equal(protected.prune_mask, base.prune_mask))
        differences = {key for key in base.diagnostics if base.diagnostics[key] != protected.diagnostics[key]}
        self.assertEqual(differences, {"egd_protection_enabled"})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        self.assertTrue(torch.equal(probe[0].get_xyz.grad, torch.ones(4, 3)))

    def test_stale_missing_wrong_type_or_device_mask_fails_before_render(self):
        probe = self.probe()
        for mask in (None, [False] * 4, torch.ones(4), torch.ones(3, dtype=torch.bool),
                     torch.ones(4, 1, dtype=torch.bool), torch.empty(4, dtype=torch.bool, device="meta")):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "aligned post-ADC"):
                self.evaluate(probe, mask)
        with self.assertRaisesRegex(ValueError, "only valid"):
            self.evaluate(probe, torch.zeros(4, dtype=torch.bool), protect=False)
        probe[2].assert_not_called()

    def test_nonfinite_or_empty_evidence_safe_and_logs_remain_finite(self):
        probe = self.probe()
        probe[2].side_effect = FloatingPointError("nonfinite render")
        decision = self.evaluate(probe, torch.tensor([False, True, False, False]))
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["blocked_by_egd"], 0)
        self.assertEqual(decision.diagnostics["requested"], 0)
        json.dumps(decision.diagnostics, allow_nan=False)
        empty = self.probe(counts=(), opacity=())
        self.assertEqual(self.evaluate(empty, torch.zeros(0, dtype=torch.bool)).prune_mask.numel(), 0)
        empty[2].assert_not_called()


if __name__ == "__main__":
    unittest.main()
