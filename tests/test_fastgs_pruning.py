"""CPU regression tests of the isolated FastGS late-pruning policy."""

import ast
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from utils.fastgs_pruning import (
    FastGSPruningConfig, apply_fastgs_pruning, evaluate_fastgs_pruning,
    normalized_pruning_score, periodic_density_config, validate_training_schedule,
)
from utils.loss_utils import l1_loss, ssim
from utils.view_consistent_density import DensityControlConfig, DensityDecision


REPO = Path(__file__).resolve().parents[1]


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


def make_camera(index):
    return SimpleNamespace(uid=index, image_name="train%d" % index,
                           original_image=torch.zeros(3, 4, 4))


class Harness:
    def __init__(self, counts, opacities=None, images=None):
        n = len(next(iter(counts.values())))
        self.model = SimpleNamespace(
            get_xyz=torch.nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)),
            get_opacity=torch.nn.Parameter(torch.tensor(opacities or [0.5] * n)[:, None]),
        )
        self.counts = counts
        self.images = images or {}
        self.calls = []

    def render(self, camera, model, pipe, background, **kwargs):
        self.calls.append((camera.uid, model, kwargs, torch.is_grad_enabled()))
        return {"render": self.images.get(camera.uid, torch.full((3, 4, 4), 0.2)),
                "high_error_counts": torch.tensor(self.counts[camera.uid], dtype=torch.int32)}

    def run(self, cameras=None, **kwargs):
        return evaluate_fastgs_pruning(
            self.model, cameras if cameras is not None else [make_camera(i) for i in self.counts],
            SimpleNamespace(), torch.zeros(3), FastGSPruningConfig(strategy="fastgs", **kwargs),
            render_fn=self.render,
        )


class FastGSConfigTest(unittest.TestCase):
    def test_defaults_and_cli_options(self):
        self.assertFalse(FastGSPruningConfig().enabled)
        options = SimpleNamespace(vcp_strategy="fastgs", fastgs_prune_interval=6000,
                                  fastgs_loss_threshold=0.2, fastgs_prune_threshold=0.8,
                                  fastgs_opacity_threshold=0.05)
        self.assertEqual(FastGSPruningConfig.from_options(options),
                         FastGSPruningConfig("fastgs", 6000, 0.2, 0.8, 0.05))
        self.assertEqual(FastGSPruningConfig.from_options(SimpleNamespace()), FastGSPruningConfig())
        json.dumps(FastGSPruningConfig().as_dict(), allow_nan=False)

    def test_invalid_options(self):
        for strategy in ("tbp", "auto", "lightgaussian", "", "FASTGS", None):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                FastGSPruningConfig(strategy=strategy)
        for interval in (0, -1, 1.5, True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                FastGSPruningConfig(interval=interval)
        for key in ("loss_threshold", "score_threshold", "opacity_threshold"):
            for value in (-0.1, 1.1, float("inf"), float("nan"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    FastGSPruningConfig(**{key: value})

    def test_late_schedule_excludes_final_step_and_resume_does_not_repeat(self):
        config = FastGSPruningConfig(strategy="fastgs")
        self.assertEqual([i for i in range(1, 10001) if config.should_prune(i, 10000)], [8000])
        self.assertEqual([i for i in range(8001, 10001) if config.should_prune(i, 10000)], [])
        self.assertFalse(config.should_prune(0, 10000))
        self.assertFalse(config.should_prune(8000, 8000))
        self.assertFalse(FastGSPruningConfig().should_prune(8000, 10000))

    def test_auto_resolves_pruning_without_changing_explicit_fastgs(self):
        for mode in ("none", "vcd"):
            with self.subTest(mode=mode):
                config = FastGSPruningConfig.from_options(SimpleNamespace(density_control=mode))
                self.assertEqual(config.strategy, "none")
                self.assertFalse(config.protect_egd)
        for mode, protected in (("vcp", False), ("vcd_vcp", True)):
            with self.subTest(mode=mode):
                config = FastGSPruningConfig.from_options(SimpleNamespace(density_control=mode))
                self.assertEqual(config.strategy, "fastgs")
                self.assertEqual(config.protect_egd, protected)
        explicit = FastGSPruningConfig.from_options(
            SimpleNamespace(density_control="vcd_vcp", vcp_strategy="fastgs"))
        self.assertFalse(explicit.protect_egd)
        with self.assertRaises(ValueError):
            FastGSPruningConfig.from_options(SimpleNamespace(vcp_strategy="tbp"))

    def test_pruning_runs_separately_without_changing_egd_options(self):
        density = DensityControlConfig("vcd_vcp", 0.6, 3)
        with self.assertRaises(ValueError):
            periodic_density_config(density, FastGSPruningConfig())
        adapted = periodic_density_config(density, FastGSPruningConfig(strategy="fastgs"))
        self.assertEqual(adapted, DensityControlConfig("vcd", 0.6, 3))
        self.assertTrue(density.use_vcp)
        adapted = periodic_density_config(DensityControlConfig("vcp"), FastGSPruningConfig(strategy="fastgs"))
        self.assertEqual(adapted.mode, "none")
        for mode in ("none", "vcd"):
            disabled_density = DensityControlConfig(mode)
            self.assertIs(periodic_density_config(disabled_density, FastGSPruningConfig()),
                          disabled_density)
            with self.assertRaises(ValueError):
                periodic_density_config(DensityControlConfig(mode), FastGSPruningConfig(strategy="fastgs"))
        with self.assertRaises(ValueError):
            periodic_density_config(DensityControlConfig("vcp"), FastGSPruningConfig())

    def test_reset_collision_rejected_without_changing_training_schedule(self):
        options = SimpleNamespace(iterations=10000, densify_until_iter=10000,
                                  opacity_reset_interval=3000, densify_from_iter=500)
        validate_training_schedule(FastGSPruningConfig(strategy="fastgs"), options, True)
        for interval in (2000, 3000, 6000, 500):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "opacity reset"):
                validate_training_schedule(FastGSPruningConfig("fastgs", interval), options, True)
        with self.assertRaisesRegex(ValueError, "smaller than iterations"):
            validate_training_schedule(FastGSPruningConfig("fastgs", 10000), options)
        # A future collision after the native reset phase is harmless.
        options.iterations = 30000
        validate_training_schedule(FastGSPruningConfig(strategy="fastgs"), options)
        validate_training_schedule(FastGSPruningConfig(), options)


class FastGSDecisionTest(unittest.TestCase):
    def test_high_score_or_low_opacity_without_trial_renders(self):
        probe = Harness({0: [0, 2, 10, 0], 1: [0, 2, 10, 0]}, [0.5, 0.5, 0.5, 0.05])
        decision = probe.run()
        self.assertEqual(decision.prune_mask.tolist(), [False, False, True, True])
        self.assertIsNone(decision.densify_mask)
        self.assertEqual(decision.diagnostics["score_candidates"], 1)
        self.assertEqual(decision.diagnostics["opacity_candidates"], 1)
        self.assertEqual(decision.diagnostics["requested"], 2)
        self.assertEqual(decision.diagnostics["trial_renders"], 0)
        self.assertEqual(len(probe.calls), 2)
        self.assertTrue(all(call[1] is probe.model for call in probe.calls))
        self.assertTrue(all(call[2] == {"loss_threshold": 0.1} for call in probe.calls))
        self.assertTrue(all(not call[3] for call in probe.calls))
        self.assertNotIn("vcp_verified", decision.diagnostics)
        json.dumps(decision.diagnostics, allow_nan=False)

    def test_view_losses_weight_counts_instead_of_low_contribution_tail(self):
        probe = Harness({0: [0, 10, 0], 1: [0, 0, 10]},
                        images={0: torch.full((3, 4, 4), 0.02), 1: torch.full((3, 4, 4), 0.8)})
        decision = probe.run()
        self.assertEqual(decision.prune_mask.tolist(), [False, False, True])
        expected_loss = (0.8 * l1_loss(probe.images[1], torch.zeros(3, 4, 4))
                         + 0.2 * (1 - ssim(probe.images[1], torch.zeros(3, 4, 4))))
        self.assertAlmostEqual(decision.diagnostics["raw_score_max"], expected_loss.item() * 10, places=5)

    def test_one_view_evidence_is_not_vetoed_by_other_views(self):
        probe = Harness({0: [0, 10, 0], 1: [0, 0, 0], 2: [0, 0, 0]})
        self.assertEqual(probe.run().prune_mask.tolist(), [False, True, False])

    def test_duplicate_cameras_are_counted_once(self):
        probe = Harness({0: [0, 10, 0], 1: [0, 0, 0]})
        views = [make_camera(0), make_camera(0), make_camera(1)]
        decision = probe.run(cameras=views)
        self.assertEqual(decision.diagnostics["evaluated_views"], 2)
        self.assertEqual([call[0] for call in probe.calls], [0, 1])

    def test_constant_score_disables_score_pruning_not_opacity_pruning(self):
        for counts in ([0, 0, 0], [4, 4, 4]):
            decision = Harness({0: counts}, [0.5, 0.05, 0.5]).run()
            self.assertTrue(decision.diagnostics["constant_score"])
            self.assertEqual(decision.prune_mask.tolist(), [False, True, False])

    def test_score_and_opacity_thresholds_are_strict(self):
        probe = Harness({0: [0, 1, 2]}, [0.1, 0.1, 0.1])
        decision = probe.run(score_threshold=0.5, opacity_threshold=0.1)
        self.assertEqual(decision.prune_mask.tolist(), [False, False, True])

    def test_all_pruned_guard_keeps_highest_opacity_and_singleton_survives(self):
        decision = Harness({0: [0, 2, 1]}, [0.01, 0.08, 0.02]).run()
        self.assertEqual(decision.prune_mask.tolist(), [True, False, True])
        self.assertTrue(decision.diagnostics["prune_all_guard"])
        self.assertFalse(Harness({0: [3]}, [0.01]).run().prune_mask.any())

    def test_nonfinite_evidence_never_becomes_opacity_deletion(self):
        for invalid in (float("nan"), float("inf")):
            probe = Harness({0: [0, 10, 0]}, [0.5, 0.05, 0.5], {0: torch.full((3, 4, 4), invalid)})
            decision = probe.run()
            self.assertFalse(decision.prune_mask.any())
            self.assertEqual(decision.diagnostics["skip_reason"], "nonfinite_diagnostic_image")
        probe = Harness({0: [0, 1, 2]}, [0.5, float("nan"), 0.5])
        self.assertFalse(probe.run().prune_mask.any())
        self.assertFalse(probe.calls)

    def test_numeric_backend_failure_is_safe_but_programming_errors_propagate(self):
        probe = Harness({0: [0, 1, 2]})
        probe.render = mock.Mock(side_effect=FloatingPointError("probe failed"))
        self.assertFalse(probe.run().prune_mask.any())
        probe.render = mock.Mock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            probe.run()

    def test_wrong_or_weighted_counts_fail_before_deletion(self):
        for counts in (torch.zeros(3), torch.ones(2, dtype=torch.int32),
                       torch.tensor([0, -1, 1], dtype=torch.int32)):
            probe = Harness({0: [0, 1, 2]})
            probe.render = lambda *args, **kwargs: {"render": torch.zeros(3, 4, 4), "high_error_counts": counts}
            with self.assertRaises(ValueError):
                probe.run()

    def test_no_views_no_points_or_disabled_needs_no_renderer(self):
        probe = Harness({0: [0, 1, 2]})
        self.assertFalse(probe.run(cameras=[]).prune_mask.any())
        self.assertFalse(probe.calls)
        decision = evaluate_fastgs_pruning(probe.model, [make_camera(0)], None, None,
                                           FastGSPruningConfig(), render_fn=probe.render)
        self.assertEqual(decision.diagnostics["skip_reason"], "disabled")
        self.assertFalse(probe.calls)
        probe.model.get_xyz = torch.zeros(0, 3)
        self.assertEqual(probe.run().prune_mask.numel(), 0)

    def test_rng_and_existing_parameters_gradients_are_untouched(self):
        probe = Harness({0: [0, 1, 2]})
        probe.model.get_xyz.grad = torch.ones_like(probe.model.get_xyz)
        before = probe.model.get_xyz.detach().clone()
        state, python_state = torch.random.get_rng_state().clone(), random.getstate()
        probe.run()
        torch.testing.assert_close(probe.model.get_xyz, before)
        self.assertTrue(torch.equal(probe.model.get_xyz.grad, torch.ones_like(before)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        self.assertEqual(random.getstate(), python_state)

    def test_normalization_invariant_and_safe_empty(self):
        score, degenerate = normalized_pruning_score(torch.tensor([3., 5., 7.]))
        torch.testing.assert_close(score, torch.tensor([0., 0.5, 1.]))
        self.assertFalse(degenerate)
        self.assertTrue(normalized_pruning_score(torch.tensor([]))[1])
        for values in ([float("inf")], [-1.0]):
            with self.assertRaises(FloatingPointError):
                normalized_pruning_score(torch.tensor(values))


class FastGSApplyTest(unittest.TestCase):
    def test_noop_preserves_parameter_identity_and_pending_gradients(self):
        model = SimpleNamespace(get_xyz=torch.nn.Parameter(torch.zeros(3, 3)), prune_points=mock.Mock())
        model.get_xyz.grad = torch.ones(3, 3)
        parameter = model.get_xyz
        result = apply_fastgs_pruning(model, DensityDecision(None, torch.zeros(3, dtype=torch.bool), {}))
        model.prune_points.assert_not_called()
        self.assertIs(model.get_xyz, parameter)
        self.assertTrue(torch.equal(model.get_xyz.grad, torch.ones(3, 3)))
        self.assertEqual(result, {"pruned": 0, "output_count": 3})

    def test_real_adam_rows_and_egd_accumulators_survive_masking(self):
        spec = importlib.util.spec_from_file_location("fastgs_model_fixture", REPO / "tests/test_view_consistent_model.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        model = fixture._model()
        names = {"xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
                 "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}
        model.optimizer = torch.optim.Adam([
            {"params": [getattr(model, attr)], "name": name} for name, attr in names.items()
        ], lr=0.0)
        before = {}
        for group in model.optimizer.param_groups:
            param = group["params"][0]
            param.grad = torch.ones_like(param)
        model.optimizer.step()
        for group in model.optimizer.param_groups:
            param = group["params"][0]
            before[group["name"]] = (param.detach().clone(), model.optimizer.state[param]["exp_avg"].clone())
        model.xyz_gradient_accum = torch.tensor([[1.], [2.], [3.]])
        model.denom = torch.tensor([[4.], [5.], [6.]])
        model.max_radii2D = torch.tensor([7., 8., 9.])
        result = apply_fastgs_pruning(model, DensityDecision(None, torch.tensor([False, True, False]), {}))
        self.assertEqual(result, {"pruned": 1, "output_count": 2})
        for group in model.optimizer.param_groups:
            param = group["params"][0]
            old_param, old_moment = before[group["name"]]
            torch.testing.assert_close(param, old_param[[0, 2]])
            torch.testing.assert_close(model.optimizer.state[param]["exp_avg"], old_moment[[0, 2]])
            self.assertIsNone(param.grad)  # Existing pruning helper's semantics.
        torch.testing.assert_close(model.xyz_gradient_accum, torch.tensor([[1.], [3.]]))
        torch.testing.assert_close(model.denom, torch.tensor([[4.], [6.]]))
        torch.testing.assert_close(model.max_radii2D, torch.tensor([7., 9.]))

    def test_stale_or_all_deleted_masks_fail(self):
        model = SimpleNamespace(get_xyz=torch.zeros(3, 3), prune_points=mock.Mock())
        for mask in (None, torch.ones(3), torch.ones(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                apply_fastgs_pruning(model, DensityDecision(None, mask, {}))
        model.prune_points.assert_not_called()

    def test_training_calls_fresh_late_score_after_adc_and_reports_after_deletion(self):
        source = (REPO / "train.py").read_text()
        ast.parse(source)
        adc = source.index("changes = gaussians.densify_and_prune_view_consistent(")
        late = source.index("decision = evaluate_fastgs_pruning(")
        apply = source.index("changes = apply_fastgs_pruning(")
        report = source.index("report_and_save(iteration, loss, iter_start.elapsed_time(iter_end))", apply)
        optimizer = source.index("gaussians.optimizer.step()", apply)
        self.assertLess(adc, late)
        self.assertLess(late, apply)
        self.assertLess(apply, report)
        self.assertLess(report, optimizer)
        self.assertIn("periodic_config, opt.lambda_dssim", source)


if __name__ == "__main__":
    unittest.main()
