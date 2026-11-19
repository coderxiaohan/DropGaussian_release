"""DPP-only tracks native ADC lineage without changing native ADC behavior.

The model, topology operations, split sampling, and Adam state are real. Only
the existing hard-coded CUDA zero allocations are redirected to the CPU.
"""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_model():
    package, extension = ModuleType("simple_knn"), ModuleType("simple_knn._C")
    extension.distCUDA2 = mock.Mock(side_effect=AssertionError("initializer not used"))
    spec = importlib.util.spec_from_file_location(
        "dpp_only_model_under_test", ROOT / "scene" / "gaussian_model.py",
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict("sys.modules", {"simple_knn": package, "simple_knn._C": extension}):
        spec.loader.exec_module(module)
    return module.GaussianModel


@contextmanager
def cpu_allocations():
    zeros = torch.zeros

    def allocate(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return zeros(*args, **kwargs)

    with mock.patch("torch.zeros", side_effect=allocate), mock.patch("torch.cuda.empty_cache"):
        yield


def original_native_event(model, max_grad, min_opacity, extent, max_screen_size):
    """Frozen native ADC sequence, independent of the new optional branch."""
    grads = model.xyz_gradient_accum / model.denom
    grads[grads.isnan()] = 0.0
    model.densify_and_clone(grads, max_grad, extent)
    model.densify_and_split(grads, max_grad, extent)
    prune_mask = (model.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = model.max_radii2D > max_screen_size
        big_points_ws = model.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    model.prune_points(prune_mask)
    torch.cuda.empty_cache()


def copy_model(model):
    # Parameter.__deepcopy__ omits .grad; preserve pending gradients so the
    # baseline comparison also exercises the real pre-ADC optimizer schedule.
    duplicate = copy.deepcopy(model)
    for source, target in zip(model.optimizer.param_groups, duplicate.optimizer.param_groups):
        gradient = source["params"][0].grad
        if gradient is not None:
            target["params"][0].grad = gradient.detach().clone()
    return duplicate


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DPPOnlyModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = load_model()
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, scales=(0.005, 0.02, 0.005), opacity=None, gradients=None):
        count = len(scales)
        model = self.model_class(1)
        values = torch.arange(count, dtype=torch.float32)
        model._xyz = torch.nn.Parameter(torch.stack((values, values + 1, values + 2), dim=1))
        model._features_dc = torch.nn.Parameter(values[:, None, None].expand(count, 1, 3).clone())
        model._features_rest = torch.nn.Parameter(torch.zeros(count, 3, 3))
        model._scaling = torch.nn.Parameter(torch.tensor(scales).log()[:, None].repeat(1, 3))
        rotation = torch.zeros(count, 4)
        rotation[:, 0] = 1
        model._rotation = torch.nn.Parameter(rotation)
        opacity = [0.2] * count if opacity is None else opacity
        model._opacity = torch.nn.Parameter(torch.logit(torch.tensor(opacity))[:, None])
        model.max_radii2D = torch.full((count,), 30.0)
        model.spatial_lr_scale = 1.0
        options = SimpleNamespace(
            percent_dense=0.01, position_lr_init=0.00016,
            position_lr_final=0.0000016, position_lr_delay_mult=0.01,
            position_lr_max_steps=10000, feature_lr=0.0025, opacity_lr=0.05,
            scaling_lr=0.005, rotation_lr=0.001,
        )
        with cpu_allocations():
            model.training_setup(options)
        for group in model.optimizer.param_groups:
            group["params"][0].grad = torch.zeros_like(group["params"][0])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            model.optimizer.state[parameter]["exp_avg"].fill_(3)
            model.optimizer.state[parameter]["exp_avg_sq"].fill_(5)
            parameter.grad = torch.ones_like(parameter)
        gradients = [1.0] * count if gradients is None else gradients
        model.xyz_gradient_accum = torch.tensor(gradients, dtype=torch.float32)[:, None]
        model.denom = torch.ones(count, 1)
        return model

    def event(self, model, support=None, max_screen_size=None):
        with cpu_allocations(), torch.no_grad():
            return model.densify_and_prune(
                0.5, 0.005, 1.0, max_screen_size, protection_support=support,
            )

    def assert_models_equal(self, expected, actual):
        self.assertEqual(set(vars(expected)), set(vars(actual)))
        self.assertEqual(len(expected.capture()), 12)
        self.assertEqual(len(actual.capture()), 12)
        for left, right in zip(expected.optimizer.param_groups, actual.optimizer.param_groups):
            lp, rp = left["params"][0], right["params"][0]
            torch.testing.assert_close(lp, rp, rtol=0, atol=0, equal_nan=True)
            self.assertIsNone(lp.grad)
            self.assertIsNone(rp.grad)
            self.assertEqual(
                {key: value for key, value in left.items() if key != "params"},
                {key: value for key, value in right.items() if key != "params"},
            )
            ls, rs = expected.optimizer.state[lp], actual.optimizer.state[rp]
            self.assertEqual(ls.keys(), rs.keys())
            for key in ls:
                self.assertTrue(torch.equal(ls[key], rs[key]), key)
        for name in ("xyz_gradient_accum", "denom", "max_radii2D"):
            self.assertTrue(torch.equal(getattr(expected, name), getattr(actual, name)), name)

    def assert_protection(self, model, protection):
        self.assertEqual(protection.shape, (model.get_xyz.shape[0],))
        self.assertEqual(protection.dtype, torch.bool)
        self.assertEqual(protection.device, model.get_xyz.device)
        self.assertFalse(protection.requires_grad)

    def test_support_does_not_change_native_topology_adam_gradients_or_rng(self):
        cases = (
            dict(scales=(0.005, 0.02, 0.005, 0.02, 0.005, 0.02),
                 gradients=(1, 1, 0.1, 0.1, 1, 1)),
            dict(scales=(0.005, 0.02, 0.005, 0.02, 0.005, 0.02),
                 gradients=(0.1, 0.1, 0, 0, 0.1, 0.1)),
            dict(scales=(0.005, 0.2, 0.02, 0.005, 0.005, 0.02),
                 opacity=(0.2, 0.2, 0.2, 0.001, 0.2, 0.001),
                 gradients=(1, 1, 0.1, 1, 1, 1)),
            dict(scales=(0.005, 0.02, 0.005, 0.02, 0.005, 0.02),
                 opacity=(0.001,) * 6),
            dict(scales=(0.005, 0.02, 0.005, 0.02, 0.005, 0.02),
                 gradients=(float("inf"), float("inf"), float("nan"),
                            float("nan"), -float("inf"), -float("inf"))),
        )
        for case_id, options in enumerate(cases):
            for screen_size in (None, 20):
                initial = self.model(**options)
                baseline = copy_model(initial)
                rng_before = torch.random.get_rng_state().clone()
                with cpu_allocations(), torch.no_grad():
                    original_native_event(baseline, 0.5, 0.005, 1.0, screen_size)
                expected_rng = torch.random.get_rng_state().clone()
                supports = (None, [False] * 6, [True] * 6,
                            [True, False, True, False, False, True])
                for support_values in supports:
                    with self.subTest(case=case_id, screen_size=screen_size, support=support_values):
                        tracked = copy_model(initial)
                        support = None if support_values is None else torch.tensor(support_values)
                        torch.random.set_rng_state(rng_before)
                        result = self.event(tracked, support, screen_size)
                        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
                        self.assert_models_equal(baseline, tracked)
                        if support is None:
                            self.assertIsNone(result)
                        else:
                            summary, protection = result
                            self.assert_protection(tracked, protection)
                            self.assertEqual(support.tolist(), support_values)
                            self.assertEqual(summary["output_count"], tracked.get_xyz.shape[0])
                            self.assertFalse(summary["prune_all_guard"])
                            if not any(support_values):
                                self.assertFalse(protection.any())
                        # Pending gradients are discarded at the same native
                        # topology boundary, so the scheduled Adam step agrees.
                        after_step = copy_model(baseline)
                        after_step.optimizer.step()
                        tracked.optimizer.step()
                        self.assert_models_equal(after_step, tracked)

    def test_only_supported_actual_clone_and_split_lineage_is_protected(self):
        model = self.model(
            scales=(0.005, 0.02, 0.005, 0.02, 0.005, 0.02, 0.005),
            gradients=(1, 1, 1, 1, 0.1, 0.1, 1),
        )
        support = torch.tensor([True, True, False, False, True, True, True])
        summary, protection = self.event(model, support)
        self.assertEqual(summary["cloned"], 3)
        self.assertEqual(summary["split_parents"], 2)
        self.assertEqual(summary["split_children"], 4)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [0, 2, 4, 5, 6, 0, 2, 6, 1, 3, 1, 3])
        self.assertEqual(protection.tolist(), [True, False, False, False, True, True, False, True,
                                              True, False, True, False])
        self.assert_protection(model, protection)
        for group in model.optimizer.param_groups:
            moment = model.optimizer.state[group["params"][0]]["exp_avg"]
            self.assertTrue(torch.all(moment[:5] == 3))
            self.assertTrue(torch.all(moment[5:] == 0))

    def test_native_cleanup_still_removes_supported_lineage_and_can_empty_population(self):
        model = self.model(scales=(0.005, 0.2, 0.005), opacity=(0.001, 0.2, 0.2))
        summary, protection = self.event(model, torch.tensor([True, True, False]), 20)
        self.assertEqual(summary["cleanup_pruned"], 4)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [2, 2])
        self.assertEqual(protection.tolist(), [False, False])
        model = self.model(opacity=(0.001, 0.001, 0.001))
        summary, protection = self.event(model, torch.ones(3, dtype=torch.bool))
        self.assertEqual(summary["output_count"], 0)
        self.assertEqual(protection.tolist(), [])
        self.assertFalse(summary["prune_all_guard"])
        self.assert_protection(model, protection)

    def test_no_selection_keeps_native_parameter_replacement_and_stat_resets(self):
        model = self.model(gradients=(0.1, 0.1, 0.1))
        before = [group["params"][0] for group in model.optimizer.param_groups]
        summary, protection = self.event(model, torch.ones(3, dtype=torch.bool), 20)
        self.assertEqual(summary["cloned"], 0)
        self.assertEqual(summary["split_parents"], 0)
        self.assertEqual(summary["cleanup_pruned"], 0)
        self.assertFalse(protection.any())
        for old, group in zip(before, model.optimizer.param_groups):
            self.assertIsNot(old, group["params"][0])
            self.assertIsNone(group["params"][0].grad)
        self.assertTrue(torch.all(model.xyz_gradient_accum == 0))
        self.assertTrue(torch.all(model.denom == 0))
        self.assertTrue(torch.all(model.max_radii2D == 0))

    def test_protection_is_event_local_and_does_not_change_checkpoint_format(self):
        model = self.model()
        attributes = set(vars(model))
        _, first = self.event(model, torch.ones(3, dtype=torch.bool))
        self.assertTrue(first.any())
        _, second = self.event(model, torch.ones(model.get_xyz.shape[0], dtype=torch.bool))
        self.assertFalse(second.any())
        self.assertEqual(set(vars(model)), attributes)
        self.assertEqual(len(model.capture()), 12)

    def test_invalid_support_fails_before_native_mutation(self):
        invalid = (
            ([True, True, True], TypeError),
            (torch.ones(3), TypeError),
            (torch.ones(3, 1, dtype=torch.bool), ValueError),
            (torch.ones(2, dtype=torch.bool), ValueError),
            (torch.ones(3, dtype=torch.bool, device="meta"), ValueError),
        )
        for support, error in invalid:
            with self.subTest(support=support):
                model = self.model()
                parameter = model._xyz
                rng_before = torch.random.get_rng_state().clone()
                with self.assertRaises(error):
                    self.event(model, support)
                self.assertIs(model._xyz, parameter)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))


if __name__ == "__main__":
    unittest.main()
