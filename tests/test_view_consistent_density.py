"""CPU tests of view-consistent decisions, without a rasterizer or training run.

The renderer and contribution pass are deterministic injected fixtures. The
controller still uses its real PyTorch masks, quantiles, L1/SSIM objective and
filtered Gaussian proxy; no score calculation is mocked.
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
    """Expose precisely the public Gaussian attributes the probe consumes."""

    def __init__(self, count=3):
        ids = torch.arange(count, dtype=torch.float32)
        self._xyz = torch.nn.Parameter(torch.stack((ids, ids + 1, ids + 2), dim=1))
        self._features = torch.nn.Parameter(ids[:, None, None].expand(count, 4, 3).clone())
        self._opacity = torch.nn.Parameter(torch.full((count, 1), 0.1))
        self._scaling = torch.nn.Parameter(torch.ones(count, 3))
        rotations = torch.zeros(count, 4)
        rotations[:, 0] = 1.0
        self._rotation = torch.nn.Parameter(rotations)
        self.active_sh_degree = 0
        self.max_sh_degree = 1
        self.covariance_calls = 0

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    def get_covariance(self, scaling_modifier=1.0):
        self.covariance_calls += 1
        diagonal = (self._scaling * scaling_modifier).square()
        zero = torch.zeros_like(diagonal[:, 0])
        return torch.stack((diagonal[:, 0], zero, zero, diagonal[:, 1], zero, diagonal[:, 2]), dim=1)

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
    def __init__(self, model, contributions, images=None, after=None):
        self.model = model
        self.contributions = contributions
        self.images = images or {}
        self.after = after
        self.render_calls = []
        self.contribution_calls = []

    def render(self, view, model, pipe, background, **kwargs):
        self.render_calls.append((view.uid, model, kwargs, torch.is_grad_enabled()))
        if model is not self.model and self.after is not None:
            return {"render": self.after(view, model)}
        return {"render": self.images.get(view.uid, torch.zeros_like(view.original_image))}

    def contribute(self, view, model, pipe, mask):
        self.contribution_calls.append((view.uid, model, mask.clone(), torch.is_grad_enabled()))
        return self.contributions[view.uid]

    def run(self, cameras=None, mode="vcd_vcp", pipe=None, **config):
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
            "contribution_quantile": 0.10, "min_views": 2,
        })

    def test_explicit_options_and_independent_modes(self):
        options = SimpleNamespace(
            density_control="vcp", vcd_error_quantile=0.6,
            vcp_contribution_quantile=0.2, density_min_views=3,
        )
        self.assertEqual(
            DensityControlConfig.from_options(options),
            DensityControlConfig("vcp", 0.6, 0.2, 3),
        )
        for mode, vcd, vcp in (("vcd", True, False), ("vcp", False, True), ("vcd_vcp", True, True)):
            with self.subTest(mode=mode):
                config = DensityControlConfig(mode)
                self.assertEqual(config.use_vcd, vcd)
                self.assertEqual(config.use_vcp, vcp)

    def test_invalid_mode_and_quantiles_are_rejected(self):
        for mode in ("all", "VCD", "", None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                DensityControlConfig(mode=mode)
        for name in ("error_quantile", "contribution_quantile"):
            for value in (0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    DensityControlConfig(**{name: value})

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
        vote, low, useful = view_evidence(
            torch.ones(3), torch.tensor([0.25, 0.5, 0.0]),
            torch.tensor([True, True, False]), mask, 0.1,
        )
        self.assertEqual(vote.tolist(), [False, True, False])
        self.assertEqual(low.tolist(), [False, False, False])
        self.assertEqual(useful.tolist(), [True, True, False])

    def test_equal_positive_contributions_never_force_a_lower_tail(self):
        for quantile in (0.1, 0.5, 0.9):
            with self.subTest(quantile=quantile):
                _, low, useful = view_evidence(
                    torch.ones(4), torch.zeros(4), torch.ones(4, dtype=torch.bool),
                    torch.zeros(4, 4, dtype=torch.bool), quantile,
                )
                self.assertFalse(low.any())
                self.assertTrue(useful.all())

    def test_zero_mass_is_low_only_when_in_frustum_and_never_votes(self):
        vote, low, useful = view_evidence(
            torch.zeros(2), torch.zeros(2), torch.tensor([True, False]),
            torch.zeros(4, 4, dtype=torch.bool), 0.1,
        )
        self.assertEqual(vote.tolist(), [False, False])
        self.assertEqual(low.tolist(), [True, False])
        self.assertEqual(useful.tolist(), [False, False])

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
                view_evidence(*values, mask, 0.1)


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
            for mode in ("vcd", "vcp", "vcd_vcp"):
                with self.subTest(count=count, views=len(cameras), mode=mode):
                    probe = ProbeHarness(FakeGaussianModel(count), {})
                    decision = probe.run(cameras, mode=mode)
                    self.assertEqual(decision.prune_mask.numel(), count)
                    self.assertFalse(decision.prune_mask.any())
                    if mode != "vcp":
                        self.assertFalse(decision.densify_mask.any())
                    else:
                        self.assertIsNone(decision.densify_mask)
                    self.assertIn("insufficient", decision.diagnostics["skip_reason"])
                    self.assertEqual(probe.render_calls, [])

    def test_equal_positive_mass_proposes_no_pruning(self):
        probe = ProbeHarness(FakeGaussianModel(), {index: evidence([1, 1, 1]) for index in range(2)})
        decision = probe.run(mode="vcp")
        self.assertIsNone(decision.densify_mask)
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["vcp_check"], "no_candidates")
        self.assertEqual(len(probe.render_calls), 2)

    def test_one_useful_view_vetoes_multiple_low_contribution_views(self):
        probe = ProbeHarness(FakeGaussianModel(), {
            0: evidence([0.01, 1, 2]),
            1: evidence([0.01, 1, 2]),
            2: evidence([2, 1, 0.01]),
        })
        decision = probe.run(mode="vcp")
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["vcp_candidates"], 0)
        self.assertEqual(len(probe.render_calls), 3)

    def test_unseen_and_single_view_points_are_not_pruned(self):
        probe = ProbeHarness(FakeGaussianModel(4), {
            0: evidence([0, 0, 0, 1], visible=[False, True, True, True]),
            1: evidence([0, 0, 0, 1], visible=[False, False, True, True]),
        })
        decision = probe.run(mode="vcp")
        self.assertEqual(decision.prune_mask.tolist(), [False, False, True, False])
        self.assertEqual(decision.diagnostics["vcp_verified"], 1)

    def test_zero_contribution_joint_removal_is_accepted_and_preserves_one_point(self):
        model = FakeGaussianModel()
        probe = ProbeHarness(model, {index: evidence([0, 0, 0]) for index in range(2)})
        decision = probe.run(mode="vcp")
        self.assertEqual(decision.prune_mask.tolist(), [False, True, True])
        self.assertEqual(decision.diagnostics["vcp_check"], "accepted_all_training_views")
        self.assertEqual(decision.diagnostics["vcp_verified"], 2)
        proxies = [item[1] for item in probe.render_calls if item[1] is not model]
        self.assertEqual(len(proxies), 2)
        self.assertTrue(all(proxy.get_xyz.shape[0] == 1 for proxy in proxies))
        self.assertEqual(model.get_xyz.shape[0], 3)

    def test_single_gaussian_is_never_proposed_for_removal(self):
        probe = ProbeHarness(FakeGaussianModel(1), {index: evidence([0]) for index in range(2)})
        decision = probe.run(mode="vcp")
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["vcp_candidates"], 0)
        self.assertEqual(len(probe.render_calls), 2)

    def test_any_training_view_loss_increase_rejects_entire_batch(self):
        model = FakeGaussianModel(4)
        probe = ProbeHarness(model, {index: evidence([0.01, 0.02, 1, 2]) for index in range(3)},
            after=lambda view, proxy: torch.full_like(view.original_image, 0.2 if view.uid == 1 else 0.0))
        decision = probe.run(mode="vcp", contribution_quantile=0.5)
        self.assertEqual(decision.diagnostics["vcp_candidates"], 2)
        self.assertEqual(decision.diagnostics["vcp_verified"], 0)
        self.assertEqual(decision.diagnostics["vcp_check"], "rejected_training_loss_increase")
        self.assertGreater(decision.diagnostics["vcp_max_train_loss_delta"], 0)
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual([item[0] for item in probe.render_calls if item[1] is not model], [0, 1, 2])

    def test_batch_is_checked_jointly_not_by_individually_harmless_deletions(self):
        model = FakeGaussianModel(4)

        def jointly_harmful(view, proxy):
            removed = model.get_xyz.shape[0] - proxy.get_xyz.shape[0]
            return torch.full_like(view.original_image, 0.3 if removed >= 2 else 0.0)

        probe = ProbeHarness(model, {index: evidence([0.01, 0.02, 1, 2]) for index in range(2)}, after=jointly_harmful)
        decision = probe.run(mode="vcp", contribution_quantile=0.5)
        self.assertFalse(decision.prune_mask.any())
        proxies = [item[1] for item in probe.render_calls if item[1] is not model]
        self.assertEqual([proxy.get_xyz[:, 0].tolist() for proxy in proxies], [[2.0, 3.0], [2.0, 3.0]])

    def test_harmless_lower_tail_batch_is_verified_without_mutating_model(self):
        model = FakeGaussianModel(4)
        probe = ProbeHarness(model, {index: evidence([0.01, 0.02, 1, 2]) for index in range(2)})
        decision = probe.run(mode="vcp", contribution_quantile=0.5)
        self.assertEqual(decision.prune_mask.tolist(), [True, True, False, False])
        self.assertEqual(decision.diagnostics["vcp_max_train_loss_delta"], 0.0)
        self.assertEqual(model.get_xyz[:, 0].tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_joint_removal_that_improves_every_view_is_accepted(self):
        probe = ProbeHarness(
            FakeGaussianModel(),
            {index: evidence([0.01, 1, 2]) for index in range(2)},
            images={index: torch.full((3, 4, 4), 0.2) for index in range(2)},
            after=lambda view, proxy: torch.zeros_like(view.original_image),
        )
        decision = probe.run(mode="vcp")
        self.assertEqual(decision.prune_mask.tolist(), [True, False, False])
        self.assertLess(decision.diagnostics["vcp_max_train_loss_delta"], 0.0)
        self.assertEqual(decision.diagnostics["vcp_check"], "accepted_all_training_views")

    def test_vcd_only_never_runs_pruning_verification(self):
        probe = ProbeHarness(FakeGaussianModel(), {index: evidence([0, 1, 2]) for index in range(2)},
            after=lambda view, proxy: (_ for _ in ()).throw(AssertionError("VCD cannot prune")))
        decision = probe.run(mode="vcd")
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["vcp_check"], "not_requested")
        self.assertEqual(len(probe.render_calls), 2)

    def test_combined_mode_protects_vcd_supported_points_from_vcp(self):
        contributions = {index: evidence([0.01, 1, 2], [0.01, 0, 0]) for index in range(2)}
        images = {index: error_image() for index in range(2)}
        combined = ProbeHarness(FakeGaussianModel(), contributions, images).run()
        self.assertEqual(combined.densify_mask.tolist(), [True, False, False])
        self.assertFalse(combined.prune_mask.any())
        vcp_only = ProbeHarness(FakeGaussianModel(), contributions, images).run(mode="vcp")
        self.assertIsNone(vcp_only.densify_mask)
        self.assertEqual(vcp_only.prune_mask.tolist(), [True, False, False])

    def test_nonfinite_baseline_images_or_targets_disable_new_actions(self):
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
        for mode in ("vcd", "vcp", "vcd_vcp"):
            with self.subTest(mode=mode):
                model = FakeGaussianModel()
                probe = ProbeHarness(
                    model,
                    {index: evidence([0, 1, 1], [0, 1, 0]) for index in range(2)},
                    {index: error_image() for index in range(2)},
                )
                with mock.patch.object(probe, "contribute", side_effect=[
                    probe.contributions[0], FloatingPointError("nonfinite color gradient"),
                ]) as contribute:
                    decision = probe.run(mode=mode)
                self.assertEqual(contribute.call_count, 2)
                self.assertFalse(decision.prune_mask.any())
                if mode == "vcp":
                    self.assertIsNone(decision.densify_mask)
                else:
                    self.assertFalse(decision.densify_mask.any())
                self.assertEqual(decision.diagnostics["skip_reason"], "nonfinite_diagnostic_contribution")
                self.assertEqual(decision.diagnostics["vcp_verified"], 0)
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

    def test_nonfinite_pruned_render_rejects_batch(self):
        probe = ProbeHarness(FakeGaussianModel(), {index: evidence([0, 1, 2]) for index in range(2)},
            after=lambda view, proxy: torch.full_like(view.original_image, float("nan")))
        decision = probe.run(mode="vcp")
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["vcp_check"], "rejected_nonfinite")
        self.assertEqual(decision.diagnostics["vcp_verified"], 0)

    def test_finite_images_with_overflowed_baseline_loss_disable_actions(self):
        # All pixel values are finite, but SSIM's squared terms overflow.
        probe = ProbeHarness(
            FakeGaussianModel(),
            {index: evidence([0, 1, 1]) for index in range(2)},
            images={0: torch.full((3, 4, 4), 1e20)},
        )
        decision = probe.run()
        self.assertFalse(decision.densify_mask.any())
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["skip_reason"], "nonfinite_diagnostic_loss")

    def test_finite_pruned_images_with_overflowed_loss_reject_batch(self):
        probe = ProbeHarness(
            FakeGaussianModel(),
            {index: evidence([0, 1, 1]) for index in range(2)},
            after=lambda view, proxy: torch.full_like(view.original_image, 1e20),
        )
        decision = probe.run(mode="vcp")
        self.assertFalse(decision.prune_mask.any())
        self.assertEqual(decision.diagnostics["vcp_check"], "rejected_nonfinite")
        self.assertEqual(decision.diagnostics["vcp_verified"], 0)

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
        self.assertTrue(decision.prune_mask[0])
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
        for _, proxy, _, _ in probe.render_calls:
            if proxy is model:
                continue
            for attribute in ("get_xyz", "get_features", "get_opacity", "get_scaling", "get_rotation"):
                self.assertFalse(getattr(proxy, attribute).requires_grad)
                self.assertEqual(getattr(proxy, attribute).shape[0], 2)
            self.assertEqual(proxy.active_sh_degree, model.active_sh_degree)
            self.assertEqual(proxy.max_sh_degree, model.max_sh_degree)

    def test_python_covariance_proxy_is_filtered_and_detached(self):
        model = FakeGaussianModel()
        probe = ProbeHarness(model, {index: evidence([0, 1, 1]) for index in range(2)})
        decision = probe.run(mode="vcp", pipe=SimpleNamespace(compute_cov3D_python=True))
        self.assertTrue(decision.prune_mask[0])
        self.assertEqual(model.covariance_calls, 1)
        proxies = [item[1] for item in probe.render_calls if item[1] is not model]
        for proxy in proxies:
            self.assertEqual(proxy.get_covariance().shape, (2, 6))
            self.assertFalse(proxy.get_covariance().requires_grad)
            with self.assertRaises(ValueError):
                proxy.get_covariance(2.0)


if __name__ == "__main__":
    unittest.main()
