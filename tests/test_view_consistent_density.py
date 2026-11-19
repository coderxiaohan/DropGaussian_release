"""CPU tests of view-consistent decisions, without a rasterizer or training run.

The renderer and contribution pass are deterministic injected fixtures. The
controller still uses its real PyTorch masks and EGD quantiles; no score
calculation is mocked.
"""

import random
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from utils.view_consistent_density import (
        DensityControlConfig,
        distinct_training_cameras,
        evaluate_density_control,
        high_error_mask,
        view_evidence,
    )


def setUpModule():
    if torch is not None:
        global _previous_threads
        _previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)


def tearDownModule():
    if torch is not None:
        torch.set_num_threads(_previous_threads)


class FakeGaussianModel:
    """Small parameterized model for checking diagnostic immutability."""

    def __init__(self, count=3):
        ids = torch.arange(count, dtype=torch.float32)
        self._xyz = torch.nn.Parameter(torch.stack((ids, ids + 1, ids + 2), dim=1))
        self._features = torch.nn.Parameter(ids[:, None, None].expand(count, 4, 3).clone())
        self._opacity = torch.nn.Parameter(torch.full((count, 1), 0.1))
        self._scaling = torch.nn.Parameter(torch.ones(count, 3))
        rotations = torch.zeros(count, 4)
        rotations[:, 0] = 1.0
        self._rotation = torch.nn.Parameter(rotations)

    @property
    def get_xyz(self):
        return self._xyz

    def parameters(self):
        return (self._xyz, self._features, self._opacity, self._scaling, self._rotation)


def camera(index):
    return SimpleNamespace(
        uid=index,
        image_name="train_%d" % index,
        original_image=torch.zeros(3, 4, 4),
    )


def error_image():
    image = torch.zeros(3, 4, 4)
    image[:, -1, -1] = 1.0
    image[:, -1, -2] = 0.5
    return image


def evidence(mass, high=None, visible=None):
    count = len(mass)
    return {
        "weight_sum": torch.tensor(mass, dtype=torch.float32),
        "high_error_weight": torch.tensor([0.0] * count if high is None else high, dtype=torch.float32),
        "frustum_visible": torch.tensor([True] * count if visible is None else visible, dtype=torch.bool),
    }


class ProbeHarness:
    def __init__(self, model, contributions, images=None):
        self.model = model
        self.contributions = contributions
        self.images = images or {}
        self.render_calls = []
        self.contribution_calls = []

    def render(self, view, model, pipe, background, **kwargs):
        self.render_calls.append((view.uid, model, kwargs, torch.is_grad_enabled()))
        if model is not self.model:
            raise AssertionError("EGD must render the unmodified model")
        return {"render": self.images.get(view.uid, torch.zeros_like(view.original_image))}

    def contribute(self, view, model, pipe, mask):
        self.contribution_calls.append((view.uid, model, mask.clone(), torch.is_grad_enabled()))
        return self.contributions[view.uid]

    def run(self, cameras=None, mode="vcd", pipe=None, **config):
        if cameras is None:
            cameras = [camera(index) for index in sorted(self.contributions)]
        return evaluate_density_control(
            self.model,
            cameras,
            pipe or SimpleNamespace(compute_cov3D_python=False),
            torch.zeros(3),
            DensityControlConfig(mode=mode, **config),
            lambda_dssim=0.2,
            render_fn=self.render,
            contribution_fn=self.contribute,
        )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DensityConfigurationTest(unittest.TestCase):
    def test_default_is_off_and_options_preserve_defaults(self):
        default = DensityControlConfig()
        self.assertEqual(default.mode, "none")
        self.assertFalse(default.use_vcd)
        self.assertFalse(default.use_vcp)
        self.assertEqual(DensityControlConfig.from_options(SimpleNamespace()), default)
        self.assertEqual(default.as_dict(), {
            "mode": "none", "error_quantile": 0.75,
            "min_views": 2,
        })

    def test_explicit_options_and_independent_modes(self):
        options = SimpleNamespace(
            density_control="vcp", vcd_error_quantile=0.6,
            density_min_views=3,
        )
        self.assertEqual(
            DensityControlConfig.from_options(options),
            DensityControlConfig("vcp", 0.6, 3),
        )
        for mode, vcd, vcp in (("vcd", True, False), ("vcp", False, True), ("vcd_vcp", True, True)):
            with self.subTest(mode=mode):
                config = DensityControlConfig(mode)
                self.assertEqual(config.use_vcd, vcd)
                self.assertEqual(config.use_vcp, vcp)

    def test_invalid_mode_and_error_quantiles_are_rejected(self):
        for mode in ("all", "VCD", "", None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                DensityControlConfig(mode=mode)
        for value in (0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DensityControlConfig(error_quantile=value)

    def test_at_least_two_integer_views_are_required(self):
        for minimum in (True, False, 0, 1, -1, 2.0, "2"):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                DensityControlConfig(min_views=minimum)

    def test_off_never_touches_model_cameras_or_render_functions(self):
        render = mock.Mock(side_effect=AssertionError("default-off must not render"))
        contribute = mock.Mock(side_effect=AssertionError("default-off must not score"))
        decision = evaluate_density_control(
            None, None, None, None, DensityControlConfig(), 0.2,
            render_fn=render, contribution_fn=contribute,
        )
        self.assertIsNone(decision.densify_mask)
        self.assertIsNone(decision.prune_mask)
        self.assertEqual(decision.diagnostics, {"mode": "none"})
        render.assert_not_called()
        contribute.assert_not_called()


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DensityEvidenceTest(unittest.TestCase):
    def test_camera_identity_prefers_image_name_then_uid_then_object(self):
        named = SimpleNamespace(image_name="frame", uid=1)
        duplicate_name = SimpleNamespace(image_name="frame", uid=2)
        other_name = SimpleNamespace(image_name="other", uid=1)
        unnamed = SimpleNamespace(uid=3)
        duplicate_uid = SimpleNamespace(uid=3)
        object_only = SimpleNamespace()
        other_object = SimpleNamespace()
        result = distinct_training_cameras([
            named, duplicate_name, other_name, unnamed, duplicate_uid,
            object_only, object_only, other_object,
        ])
        self.assertEqual([id(item) for item in result], [
            id(named), id(other_name), id(unnamed), id(object_only), id(other_object),
        ])

    def test_constant_error_including_zero_has_no_artificial_upper_tail(self):
        for constant in (0.0, 0.5, 1.0):
            with self.subTest(constant=constant):
                image = torch.full((3, 4, 4), constant, requires_grad=True)
                mask = high_error_mask(image, torch.zeros_like(image), 0.75)
                self.assertEqual(mask.dtype, torch.bool)
                self.assertFalse(mask.any())
                self.assertFalse(mask.requires_grad)

    def test_high_error_tail_is_strict_finite_and_does_not_mutate_inputs(self):
        image = error_image().requires_grad_()
        target = torch.zeros_like(image, requires_grad=True)
        before = image.detach().clone()
        mask = high_error_mask(image, target, 0.75)
        expected = torch.zeros(4, 4, dtype=torch.bool)
        expected[-1, -2:] = True
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.equal(image, before))
        self.assertIsNone(image.grad)
        self.assertIsNone(target.grad)

    def test_invalid_error_images_fail_explicitly(self):
        for image, target in (
            (torch.zeros(4, 4), torch.zeros(4, 4)),
            (torch.zeros(1, 4, 4), torch.zeros(1, 4, 4)),
            (torch.zeros(3, 4, 4), torch.zeros(3, 2, 2)),
            (torch.full((3, 4, 4), float("nan")), torch.zeros(3, 4, 4)),
            (torch.zeros(3, 4, 4), torch.full((3, 4, 4), float("inf"))),
        ):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                high_error_mask(image, target, 0.75)

    def test_image_wide_splat_does_not_vote_without_error_concentration(self):
        mask = torch.tensor([[True, False], [False, False]])
        vote = view_evidence(
            torch.ones(3), torch.tensor([0.25, 0.5, 0.0]),
            torch.tensor([True, True, False]), mask,
        )
        self.assertEqual(vote.dtype, torch.bool)
        self.assertEqual(vote.tolist(), [False, True, False])

    def test_zero_mass_never_votes_even_when_in_frustum(self):
        vote = view_evidence(
            torch.zeros(2), torch.zeros(2), torch.tensor([True, False]),
            torch.zeros(4, 4, dtype=torch.bool),
        )
        self.assertEqual(vote.tolist(), [False, False])

    def test_invalid_per_point_evidence_is_rejected(self):
        mask = torch.zeros(4, 4, dtype=torch.bool)
        examples = (
            (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1, dtype=torch.bool)),
            (torch.ones(2), torch.ones(3), torch.ones(2, dtype=torch.bool)),
            (torch.ones(2), torch.ones(2), torch.ones(2)),
            (torch.tensor([-1.0, 1.0]), torch.zeros(2), torch.ones(2, dtype=torch.bool)),
            (torch.ones(2), torch.tensor([0.0, -1.0]), torch.ones(2, dtype=torch.bool)),
            (torch.tensor([float("nan"), 1.0]), torch.zeros(2), torch.ones(2, dtype=torch.bool)),
            (torch.ones(2), torch.tensor([float("inf"), 0.0]), torch.ones(2, dtype=torch.bool)),
        )
        for values in examples:
            with self.subTest(values=values), self.assertRaises(ValueError):
                view_evidence(*values, mask)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DensityControllerTest(unittest.TestCase):
    def test_vcd_requires_votes_from_multiple_distinct_views(self):
        model = FakeGaussianModel()
        probe = ProbeHarness(model, {
            0: evidence([1, 1, 1], [0.8, 0.8, 0]),
            1: evidence([1, 1, 1], [0.8, 0, 0]),
        }, images={0: error_image(), 1: error_image()})
        first, second = camera(0), camera(1)
        decision = probe.run([first, camera(0), first, second], mode="vcd")
        self.assertEqual(decision.densify_mask.tolist(), [True, False, False])
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["evaluated_views"], 2)
        self.assertEqual(decision.diagnostics["vcd_supported"], 1)
        self.assertEqual(len(probe.render_calls), 2)

    def test_configured_minimum_views_is_respected(self):
        probe = ProbeHarness(FakeGaussianModel(), {
            0: evidence([1, 1, 1], [1, 1, 0]),
            1: evidence([1, 1, 1], [1, 1, 0]),
            2: evidence([1, 1, 1], [1, 0, 0]),
        }, images={index: error_image() for index in range(3)})
        decision = probe.run(mode="vcd", min_views=3)
        self.assertEqual(decision.densify_mask.tolist(), [True, False, False])

    def test_insufficient_distinct_views_or_empty_model_are_conservative(self):
        for count, cameras in ((3, []), (3, [camera(0)]), (3, [camera(0), camera(0)]), (0, [camera(0), camera(1)])):
            with self.subTest(count=count, views=len(cameras)):
                probe = ProbeHarness(FakeGaussianModel(count), {})
                decision = probe.run(cameras)
                self.assertEqual(decision.prune_mask.numel(), count)
                self.assertFalse(decision.prune_mask.any())
                self.assertFalse(decision.densify_mask.any())
                self.assertIn("insufficient", decision.diagnostics["skip_reason"])
                self.assertEqual(probe.render_calls, [])

    def test_egd_only_renders_each_unmodified_training_view(self):
        model = FakeGaussianModel()
        probe = ProbeHarness(model, {index: evidence([0, 1, 2]) for index in range(2)})
        decision = probe.run()
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual([call[0] for call in probe.render_calls], [0, 1])
        self.assertTrue(all(call[1] is model for call in probe.render_calls))
        self.assertEqual(decision.diagnostics, {
            "mode": "vcd", "input_count": 3, "evaluated_views": 2,
            "vcd_supported": 0, "observed_in_multiple_views": 3,
        })

    def test_legacy_pruning_modes_require_separate_controller_before_rendering(self):
        for mode in ("vcp", "vcd_vcp"):
            with self.subTest(mode=mode):
                probe = ProbeHarness(FakeGaussianModel(), {})
                with self.assertRaisesRegex(ValueError, "separate pruning controller"):
                    probe.run(mode=mode)
                self.assertEqual(probe.render_calls, [])
                self.assertEqual(probe.contribution_calls, [])

    def test_legacy_loss_and_opacity_arguments_do_not_change_egd(self):
        model = FakeGaussianModel()
        probe = ProbeHarness(
            model,
            {index: evidence([1, 1, 1], [1, 0, 0]) for index in range(2)},
            images={index: error_image() for index in range(2)},
        )
        expected = probe.run()
        actual = evaluate_density_control(
            model, [camera(0), camera(1)], SimpleNamespace(), torch.zeros(3),
            DensityControlConfig("vcd"),
            lambda_dssim=0.5, native_opacity_threshold=0.5,
            render_fn=probe.render, contribution_fn=probe.contribute,
        )
        self.assertTrue(torch.equal(actual.densify_mask, expected.densify_mask))
        self.assertTrue(torch.equal(actual.prune_mask, expected.prune_mask))
        self.assertEqual(actual.diagnostics, expected.diagnostics)

    def test_nonfinite_images_or_targets_disable_egd(self):
        for invalid_target in (False, True):
            with self.subTest(invalid_target=invalid_target):
                cameras = [camera(0), camera(1)]
                images = {0: error_image(), 1: error_image()}
                if invalid_target:
                    cameras[1].original_image[0, 0, 0] = float("nan")
                else:
                    images[1][0, 0, 0] = float("inf")
                probe = ProbeHarness(FakeGaussianModel(), {
                    index: evidence([1, 1, 1], [1, 0, 0]) for index in range(2)
                }, images=images)
                decision = probe.run(cameras)
                self.assertFalse(decision.densify_mask.any())
                self.assertFalse(decision.prune_mask.any())
                self.assertEqual(decision.diagnostics["skip_reason"], "nonfinite_diagnostic_image")

    def test_nonfinite_contributions_disable_new_actions(self):
        for field in ("weight_sum", "high_error_weight"):
            for bad in (float("nan"), float("inf")):
                with self.subTest(field=field, bad=bad):
                    contributions = {index: evidence([1, 1, 1], [1, 0, 0]) for index in range(2)}
                    contributions[1][field][0] = bad
                    probe = ProbeHarness(FakeGaussianModel(), contributions, {index: error_image() for index in range(2)})
                    decision = probe.run()
                    self.assertFalse(decision.densify_mask.any())
                    self.assertFalse(decision.prune_mask.any())
                    self.assertEqual(decision.diagnostics["skip_reason"], "nonfinite_diagnostic_contribution")

    def test_numeric_contribution_exception_conservatively_discards_partial_evidence(self):
        model = FakeGaussianModel()
        probe = ProbeHarness(
            model,
            {index: evidence([0, 1, 1], [0, 1, 0]) for index in range(2)},
            {index: error_image() for index in range(2)},
        )
        with mock.patch.object(probe, "contribute", side_effect=[
            probe.contributions[0], FloatingPointError("nonfinite color gradient"),
        ]) as contribute:
            decision = probe.run()
        self.assertEqual(contribute.call_count, 2)
        self.assertFalse(decision.prune_mask.any())
        self.assertFalse(decision.densify_mask.any())
        self.assertEqual(decision.diagnostics["skip_reason"], "nonfinite_diagnostic_contribution")
        self.assertTrue(all(call[1] is model for call in probe.render_calls))

    def test_unrelated_contribution_runtime_error_is_not_silently_swallowed(self):
        probe = ProbeHarness(FakeGaussianModel(), {index: evidence([0, 1, 1]) for index in range(2)})
        failure = RuntimeError("CUDA diagnostic failure")
        with mock.patch.object(probe, "contribute", side_effect=failure) as contribute:
            with self.assertRaises(RuntimeError) as caught:
                probe.run()
        self.assertIs(caught.exception, failure)
        contribute.assert_called_once()
        self.assertEqual(len(probe.render_calls), 1)

    def test_probe_preserves_rng_parameters_existing_gradients_and_input_evidence(self):
        model = FakeGaussianModel()
        parameters = model.parameters()
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, index + 0.5)
        values_before = [parameter.detach().clone() for parameter in parameters]
        gradients_before = [parameter.grad.clone() for parameter in parameters]
        contributions = {index: evidence([0, 1, 1]) for index in range(2)}
        evidence_before = {key: {name: value.clone() for name, value in item.items()} for key, item in contributions.items()}
        cameras = [camera(0), camera(1)]
        for view in cameras:
            view.original_image.requires_grad_()
            view.original_image.grad = torch.ones_like(view.original_image)
        torch_rng = torch.random.get_rng_state().clone()
        python_rng = random.getstate()
        probe = ProbeHarness(model, contributions)
        decision = probe.run(cameras)
        self.assertFalse(decision.prune_mask.any())
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for parameter, before, gradient in zip(parameters, values_before, gradients_before):
            self.assertTrue(torch.equal(parameter, before))
            self.assertTrue(torch.equal(parameter.grad, gradient))
        for key, item in contributions.items():
            for name, value in item.items():
                self.assertTrue(torch.equal(value, evidence_before[key][name]))
        for view in cameras:
            self.assertTrue(torch.equal(view.original_image.grad, torch.ones_like(view.original_image)))
        self.assertTrue(all(not call[3] for call in probe.render_calls))
        self.assertTrue(all(not call[3] for call in probe.contribution_calls))
        self.assertTrue(all(call[2] == {"is_train": False} for call in probe.render_calls))
        self.assertTrue(all(call[1] is model for call in probe.render_calls))

if __name__ == "__main__":
    unittest.main()
