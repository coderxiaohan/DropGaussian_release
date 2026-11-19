"""Exercise actual training event routing for DPP-only without running training."""

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "dpp_shared_training_harness", Path(__file__).with_name("test_fastgs_egd_training.py"),
)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
torch = shared.torch


class DPPOnlyTrainingHarness(shared.TrainingEventHarness):
    def __init__(self, code):
        from utils.fastgs_pruning import (
            FastGSPruningConfig, dpp_protection_config, periodic_density_config,
        )
        from utils.view_consistent_density import DensityControlConfig

        super().__init__(code, protect=True, periodic_mode="none")
        self.density_config = DensityControlConfig(mode="vcp", error_quantile=0.81, min_views=3)
        pruning = FastGSPruningConfig(strategy="fastgs", protect_egd=True, dpp_only=True)
        self.scope["pruning_config"] = pruning
        self.scope["periodic_config"] = periodic_density_config(self.density_config, pruning)
        self.scope["protection_config"] = dpp_protection_config(self.density_config, pruning)
        self.model.densify_and_prune = mock.Mock(side_effect=self.native_adc)

    def native_adc(self, *args, **kwargs):
        self.events.append("native_adc")
        # Always change topology, including when support is entirely false.
        # This fixture models an event that clones the first two input rows;
        # actual mixed clone/split lineage is covered by test_dpp_only_model.
        count = len(self.model.get_xyz) + 2
        self.model.get_xyz = torch.zeros(count, 3)
        self.model.max_radii2D = torch.zeros(count)
        support = kwargs.get("protection_support")
        if support is not None:
            mask = torch.cat((support, support[:2])).clone()
            self.post_adc_masks.append(mask)
            return {"output_count": count, "cloned": 2}, mask

    def run(self, iteration):
        with torch.no_grad():
            super().run(iteration)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DPPOnlyTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = shared.training_event_code()

    def test_dpp_event_diagnoses_support_then_native_adc_and_forwards_post_adc_mask(self):
        harness = DPPOnlyTrainingHarness(self.code)
        harness.scope["saving_iterations"] = [8000]
        harness.scope["checkpoint_iterations"] = [8000]
        # Diagnostic pruning requests must not become topology decisions in
        # this ablation, even if a substitute evaluator were to emit them.
        harness.density_decision.prune_mask.fill_(True)
        harness.run(8000)

        self.assertEqual(harness.scope["periodic_config"].mode, "none")
        self.assertFalse(harness.scope["periodic_config"].use_vcd)
        evidence = harness.scope["protection_config"]
        self.assertEqual(evidence.mode, "vcd")
        self.assertEqual(evidence.error_quantile, 0.81)
        self.assertEqual(evidence.min_views, 3)
        self.assertFalse(evidence.use_vcp)
        harness.evaluate_density.assert_called_once()
        self.assertIs(harness.evaluate_density.call_args.args[4], evidence)
        self.assertEqual(harness.evaluate_density.call_args.kwargs, {"native_opacity_threshold": 0.005})

        native = harness.model.densify_and_prune
        native.assert_called_once_with(
            0.0002, 0.005, 1.0, None,
            protection_support=harness.density_decision.densify_mask,
        )
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        mask, score_count = harness.score_inputs[0]
        self.assertIs(mask, harness.post_adc_masks[0])
        self.assertIsNot(mask, harness.density_decision.densify_mask)
        self.assertEqual(mask.tolist(), [True, True, False, True, True])
        self.assertEqual(mask.shape, (score_count,))
        self.assertEqual(score_count, 5)
        self.assertEqual(harness.events, [
            "density", "native_adc", "score", "prune", ("report", 4), ("save", 4),
            "optimizer", ("checkpoint", 4),
        ])
        evidence_event = harness.scope["density_events"][0]
        self.assertEqual(evidence_event["event"], "dpp_protection_evidence")
        self.assertFalse(evidence_event["densification_gate"])
        self.assertEqual(evidence_event["topology"]["output_count"], 5)
        pruning_event = harness.scope["fastgs_events"][0]
        self.assertFalse(pruning_event["densification_gate"])
        self.assertEqual(pruning_event["protection_evidence"], "multi_view_error_support_intersect_native_adc")

    def test_ordinary_adc_does_not_diagnose_support_or_request_protection(self):
        harness = DPPOnlyTrainingHarness(self.code)
        harness.scope["egd_protection"] = torch.ones(99, dtype=torch.bool)
        harness.run(8100)
        harness.model.densify_and_prune.assert_called_once_with(0.0002, 0.005, 1.0, None)
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        harness.evaluate_density.assert_not_called()
        harness.evaluate_fastgs.assert_not_called()
        harness.apply_fastgs.assert_not_called()
        self.assertEqual(harness.post_adc_masks, [])
        self.assertEqual(harness.scope["density_events"], [])
        self.assertIsNone(harness.scope["egd_protection"])
        self.assertEqual(harness.events, [("report", 3), "native_adc", "optimizer"])

    def test_zero_support_still_runs_native_adc_and_late_pruning(self):
        harness = DPPOnlyTrainingHarness(self.code)
        harness.density_decision.densify_mask.zero_()
        harness.run(8000)
        native = harness.model.densify_and_prune
        native.assert_called_once()
        self.assertIs(native.call_args.kwargs["protection_support"], harness.density_decision.densify_mask)
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        mask, count = harness.score_inputs[0]
        self.assertEqual(count, 5)
        self.assertEqual(mask.shape, (5,))
        self.assertFalse(mask.any())
        self.assertEqual(harness.scope["density_events"][0]["topology"]["cloned"], 2)
        harness.apply_fastgs.assert_called_once()

    def test_missing_support_fails_before_topology_pruning_or_save(self):
        harness = DPPOnlyTrainingHarness(self.code)
        harness.density_decision.densify_mask = None
        with self.assertRaisesRegex(RuntimeError, "this event's multiview protection evidence"):
            harness.run(8000)
        harness.evaluate_density.assert_called_once()
        harness.model.densify_and_prune.assert_not_called()
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        harness.evaluate_fastgs.assert_not_called()
        harness.apply_fastgs.assert_not_called()
        harness.scope["scene"].save.assert_not_called()
        harness.model.optimizer.step.assert_not_called()

    def test_protection_is_cleared_between_events_and_recomputed_for_current_rows(self):
        harness = DPPOnlyTrainingHarness(self.code)
        harness.run(8000)
        first = harness.scope["egd_protection"]
        harness.run(8001)
        self.assertIsNone(harness.scope["egd_protection"])
        self.assertEqual(harness.evaluate_density.call_count, 1)
        self.assertEqual(harness.model.densify_and_prune.call_count, 1)
        harness.density_decision.densify_mask = torch.tensor([False, True, False, True])
        harness.run(16000)
        second = harness.scope["egd_protection"]
        self.assertIsNot(first, second)
        self.assertIs(second, harness.post_adc_masks[1])
        self.assertIs(second, harness.score_inputs[1][0])
        self.assertEqual(second.tolist(), [False, True, False, True, False, True])
        self.assertEqual(second.shape, (harness.score_inputs[1][1],))


if __name__ == "__main__":
    unittest.main()
