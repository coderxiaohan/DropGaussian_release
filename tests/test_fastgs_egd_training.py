"""Execute the real training event block with CPU-only diagnostic substitutes.

The AST extraction avoids importing train.py's CUDA/data dependencies and never
runs training. Event routing, keyword arguments, mask lifetime, reporting and
checkpoint ordering are the actual statements from the training loop.
"""

import ast
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def training_event_code():
    tree = ast.parse((ROOT / "train.py").read_text())
    training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training")
    report = next(node for node in training.body if isinstance(node, ast.FunctionDef) and node.name == "report_and_save")
    loop = next(node for node in training.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "iteration")
    event_block = next(
        node for node in loop.body
        if isinstance(node, ast.With)
        and any(isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "no_grad" for item in node.items)
    )
    first = next(
        index for index, node in enumerate(event_block.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "fastgs_due" for target in node.targets)
    )
    # Include the real nested report/save function and EVERY statement from
    # event scheduling through the end of this loop's no_grad block.
    module = ast.Module(body=copy.deepcopy([report, *event_block.body[first:]]), type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(ROOT / "train.py"), "exec")


class TrainingEventHarness:
    def __init__(self, code, protect=True, due_iterations=(8000, 16000), periodic_mode="vcd"):
        self.code = code
        self.events = []
        self.post_adc_masks = []
        self.score_inputs = []
        self.model = SimpleNamespace(get_xyz=torch.zeros(3, 3), max_radii2D=torch.zeros(3))
        self.model.add_densification_stats = mock.Mock()
        self.model.reset_opacity = mock.Mock()
        self.model.densify_and_prune = mock.Mock(side_effect=lambda *args: self.events.append("legacy_adc"))
        self.model.densify_and_prune_view_consistent = mock.Mock(side_effect=self.adc)
        self.model.optimizer = SimpleNamespace(
            step=mock.Mock(side_effect=lambda: self.events.append("optimizer")),
            zero_grad=mock.Mock(),
        )
        self.model.capture = mock.Mock(side_effect=lambda: {"count": len(self.model.get_xyz)})
        self.density_decision = SimpleNamespace(
            densify_mask=torch.tensor([True, True, False]),
            prune_mask=torch.zeros(3, dtype=torch.bool), diagnostics={},
        )
        self.evaluate_density = mock.Mock(side_effect=self.density)
        self.evaluate_fastgs = mock.Mock(side_effect=self.score)
        self.apply_fastgs = mock.Mock(side_effect=self.prune)
        scene = SimpleNamespace(
            cameras_extent=1.0, getTrainCameras=mock.Mock(return_value=[object(), object()]),
            save=mock.Mock(side_effect=lambda iteration: self.events.append(("save", len(self.model.get_xyz)))),
            model_path="unused-test-output",
        )
        ticks = itertools.count()
        self.scope = {
            "gaussians": self.model, "scene": scene, "pipe": object(), "background": torch.zeros(3),
            "opt": SimpleNamespace(
                iterations=20000, densify_until_iter=19900, densify_from_iter=500,
                densification_interval=100, densify_grad_threshold=0.0002,
                lambda_dssim=0.2, opacity_reset_interval=3000,
            ),
            "dataset": SimpleNamespace(white_background=False),
            "periodic_config": SimpleNamespace(mode=periodic_mode, use_vcd=periodic_mode in {"vcd", "vcd_vcp"}),
            "protection_config": None,
            "pruning_config": SimpleNamespace(
                protect_egd=protect, dpp_only=False, should_prune=lambda iteration, total: iteration in due_iterations,
            ),
            "evaluate_density_control": self.evaluate_density,
            "evaluate_fastgs_pruning": self.evaluate_fastgs,
            "apply_fastgs_pruning": self.apply_fastgs,
            "torch": SimpleNamespace(
                max=torch.max, cuda=SimpleNamespace(synchronize=mock.Mock()),
                save=mock.Mock(side_effect=lambda state, path: self.events.append(("checkpoint", state[0]["count"]))),
            ),
            "time": SimpleNamespace(perf_counter=lambda: next(ticks)),
            "open": mock.mock_open(), "json": json, "print": mock.Mock(),
            "density_events": [], "fastgs_events": [], "convergence_curve": [],
            "density_log_path": "unused-density.jsonl", "fastgs_log_path": "unused-fastgs.jsonl",
            "tb_writer": None, "loss": torch.tensor(0.1),
            "iter_start": SimpleNamespace(elapsed_time=mock.Mock(return_value=1.0)), "iter_end": object(),
            "training_report": mock.Mock(side_effect=lambda *args: self.events.append(("report", len(self.model.get_xyz)))),
            "l1_loss": object(), "testing_iterations": [], "saving_iterations": [],
            "checkpoint_iterations": [], "render": object(),
        }

    def density(self, *args, **kwargs):
        self.events.append("density")
        return self.density_decision

    def adc(self, *args, **kwargs):
        self.events.append("adc")
        # Simulate a changed population, so accidentally forwarding the
        # pre-ADC support vector cannot pass the alignment assertions.
        count = len(self.model.get_xyz) + 2
        self.model.get_xyz = torch.zeros(count, 3)
        self.model.max_radii2D = torch.zeros(count)
        summary = {"output_count": count, "cloned": 2}
        if kwargs.get("return_densify_protection", False):
            mask = torch.arange(count) % 2 == 0
            self.post_adc_masks.append(mask)
            return summary, mask
        return summary

    def score(self, model, cameras, pipe, background, config, **kwargs):
        self.events.append("score")
        self.score_inputs.append((kwargs.get("protection_mask"), len(model.get_xyz)))
        return SimpleNamespace(diagnostics={})

    def prune(self, model, decision):
        self.events.append("prune")
        model.get_xyz = model.get_xyz[:-1]
        model.max_radii2D = model.max_radii2D[:-1]
        return {"pruned": 1, "output_count": len(model.get_xyz)}

    def run(self, iteration):
        self.scope.update(
            iteration=iteration, visibility_filter=torch.ones(len(self.model.get_xyz), dtype=torch.bool),
            radii=torch.ones(len(self.model.get_xyz)), viewspace_point_tensor=object(),
        )
        # Reuse the SAME namespace across events, just as loop locals persist.
        # The source itself must reset egd_protection on each iteration.
        exec(self.code, self.scope)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class FastGSEGDTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = training_event_code()

    def test_protected_prune_event_requests_and_forwards_fresh_post_adc_mask(self):
        harness = TrainingEventHarness(self.code)
        harness.run(8000)
        adc = harness.model.densify_and_prune_view_consistent
        adc.assert_called_once()
        self.assertIs(adc.call_args.kwargs["return_densify_protection"], True)
        self.assertIs(adc.call_args.kwargs["densify_mask"], harness.density_decision.densify_mask)
        mask, count = harness.score_inputs[0]
        self.assertIs(mask, harness.post_adc_masks[0])
        self.assertIsNot(mask, harness.density_decision.densify_mask)
        self.assertEqual(mask.shape, (count,))
        self.assertEqual(count, 5)  # Input support was length 3.
        self.assertLess(harness.events.index("adc"), harness.events.index("score"))
        self.assertEqual(harness.scope["density_events"][0]["topology"]["output_count"], 5)

    def test_unprotected_prune_keeps_original_adc_call_without_new_flag(self):
        harness = TrainingEventHarness(self.code, protect=False)
        harness.run(8000)
        self.assertNotIn("return_densify_protection", harness.model.densify_and_prune_view_consistent.call_args.kwargs)
        harness.evaluate_fastgs.assert_called_once()
        self.assertIsNone(harness.score_inputs[0][0])
        self.assertEqual(harness.post_adc_masks, [])

    def test_ordinary_adc_does_not_request_protection_and_keeps_legacy_report_order(self):
        harness = TrainingEventHarness(self.code, protect=True)
        harness.run(8100)
        self.assertNotIn("return_densify_protection", harness.model.densify_and_prune_view_consistent.call_args.kwargs)
        harness.evaluate_fastgs.assert_not_called()
        self.assertIsNone(harness.scope["egd_protection"])
        self.assertLess(harness.events.index(("report", 3)), harness.events.index("adc"))

    def test_egd_tensorboard_logging_needs_only_egd_evidence(self):
        harness = TrainingEventHarness(self.code)
        harness.density_decision.diagnostics = {"vcd_supported": 2}
        writer = mock.Mock()
        harness.scope["tb_writer"] = writer
        harness.run(8100)
        writer.add_scalar.assert_any_call("density/vcd_supported", 2, 8100)
        self.assertEqual(
            {call.args[0] for call in writer.add_scalar.call_args_list},
            {"density/vcd_supported", "density/seconds"},
        )
        self.assertNotIn("prune_mask", harness.model.densify_and_prune_view_consistent.call_args.kwargs)

    def test_non_adc_iteration_never_requests_lineage_or_uses_stale_mask(self):
        harness = TrainingEventHarness(self.code)
        harness.scope["egd_protection"] = torch.ones(99, dtype=torch.bool)
        harness.run(8001)
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        harness.evaluate_fastgs.assert_not_called()
        self.assertIsNone(harness.scope["egd_protection"])

    def test_mask_lifetime_is_one_iteration_and_next_event_uses_new_mask(self):
        harness = TrainingEventHarness(self.code)
        harness.run(8000)
        first_mask = harness.scope["egd_protection"]
        harness.run(8001)
        self.assertIsNone(harness.scope["egd_protection"])
        harness.run(16000)
        second_mask = harness.scope["egd_protection"]
        self.assertIsNot(first_mask, second_mask)
        self.assertIs(second_mask, harness.post_adc_masks[1])
        self.assertIs(harness.score_inputs[1][0], second_mask)
        self.assertEqual(second_mask.shape, (harness.score_inputs[1][1],))

    def test_missing_current_egd_decision_fails_before_adc_or_fastgs_pruning(self):
        harness = TrainingEventHarness(self.code)
        harness.density_decision.densify_mask = None
        with self.assertRaisesRegex(RuntimeError, "this event's EGD decision"):
            harness.run(8000)
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        harness.evaluate_fastgs.assert_not_called()
        harness.apply_fastgs.assert_not_called()
        harness.scope["scene"].save.assert_not_called()

    def test_report_ply_and_checkpoint_all_follow_actual_fastgs_pruning(self):
        harness = TrainingEventHarness(self.code)
        harness.scope["saving_iterations"] = [8000]
        harness.scope["checkpoint_iterations"] = [8000]
        harness.run(8000)
        self.assertEqual(harness.events, [
            "density", "adc", "score", "prune", ("report", 4), ("save", 4),
            "optimizer", ("checkpoint", 4),
        ])
        harness.scope["scene"].save.assert_called_once_with(8000)
        self.assertEqual(harness.scope["torch"].save.call_args.args[0][1], 8000)

    def test_legacy_density_mode_does_not_gain_new_arguments(self):
        harness = TrainingEventHarness(self.code, protect=False, periodic_mode="none")
        harness.run(8000)
        harness.model.densify_and_prune.assert_called_once()
        self.assertEqual(harness.model.densify_and_prune.call_args.kwargs, {})
        harness.model.densify_and_prune_view_consistent.assert_not_called()
        self.assertIsNone(harness.score_inputs[0][0])


if __name__ == "__main__":
    unittest.main()
