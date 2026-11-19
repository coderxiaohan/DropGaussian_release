"""CPU tests of event-local EGD lineage using real topology and Adam helpers.

Only the existing hard-coded CUDA allocations are redirected to CPU. Clone,
split sampling, optimizer state indexing and native cleanup remain real.
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
        "egd_pruning_protection_model_under_test", ROOT / "scene" / "gaussian_model.py",
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


def optimization_args():
    return SimpleNamespace(
        percent_dense=0.01, position_lr_init=0.00016,
        position_lr_final=0.0000016, position_lr_delay_mult=0.01,
        position_lr_max_steps=10000, feature_lr=0.0025, opacity_lr=0.05,
        scaling_lr=0.005, rotation_lr=0.001,
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class EGDPruningProtectionModelTest(unittest.TestCase):
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
        model.max_radii2D = torch.zeros(count)
        model.spatial_lr_scale = 1.0
        with cpu_allocations():
            model.training_setup(optimization_args())
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
        model.xyz_gradient_accum = torch.tensor(gradients)[:, None]
        model.denom = torch.ones(count, 1)
        return model

    def event(self, model, **kwargs):
        with cpu_allocations():
            return model.densify_and_prune_view_consistent(
                0.5, 0.005, 1.0, kwargs.pop("max_screen_size", None), **kwargs,
            )

    def assert_consistent(self, model, protection):
        count = model.get_xyz.shape[0]
        self.assertEqual(protection.shape, (count,))
        self.assertEqual(protection.dtype, torch.bool)
        self.assertEqual(protection.device, model.get_xyz.device)
        self.assertFalse(protection.requires_grad)
        self.assertEqual(model.xyz_gradient_accum.shape, (count, 1))
        self.assertEqual(model.denom.shape, (count, 1))
        self.assertEqual(model.max_radii2D.shape, (count,))
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            self.assertEqual(parameter.shape[0], count)
            self.assertIsNone(parameter.grad)
            state = model.optimizer.state[parameter]
            self.assertEqual(state["exp_avg"].shape, parameter.shape)
            self.assertEqual(state["exp_avg_sq"].shape, parameter.shape)

    def test_true_selection_lineage_tracks_clone_split_and_low_gradient(self):
        model = self.model(
            scales=(0.005, 0.02, 0.005, 0.02, 0.005, 0.02, 0.005),
            gradients=(1, 1, 0.1, 0.1, 1, 1, 1),
        )
        support = torch.tensor([True, True, True, True, False, False, False])
        summary, protection = self.event(
            model, densify_mask=support, return_densify_protection=True,
        )
        self.assertEqual(summary["cloned"], 1)
        self.assertEqual(summary["split_parents"], 1)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [0, 2, 3, 4, 5, 6, 0, 1, 1])
        self.assertEqual(protection.tolist(), [True, False, False, False, False, False, True, True, True])
        self.assertEqual(support.tolist(), [True, True, True, True, False, False, False])
        self.assert_consistent(model, protection)
        for group in model.optimizer.param_groups:
            moment = model.optimizer.state[group["params"][0]]["exp_avg"]
            self.assertTrue(torch.all(moment[:6] == 3))
            self.assertTrue(torch.all(moment[6:] == 0))

    def test_native_opacity_cleanup_removes_protected_parents_and_children(self):
        model = self.model(opacity=(0.001, 0.001, 0.2))
        summary, protection = self.event(
            model, densify_mask=torch.tensor([True, True, False]), return_densify_protection=True,
        )
        self.assertEqual(summary["cloned"], 1)
        self.assertEqual(summary["split_children"], 2)
        self.assertEqual(summary["cleanup_pruned"], 4)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [2])
        self.assertEqual(protection.tolist(), [False])
        self.assert_consistent(model, protection)

    def test_native_world_size_cleanup_removes_protected_split_children(self):
        model = self.model(scales=(0.2, 0.005))
        summary, protection = self.event(
            model, densify_mask=torch.tensor([True, False]), max_screen_size=20,
            return_densify_protection=True,
        )
        self.assertEqual(summary["split_children"], 2)
        self.assertEqual(summary["cleanup_pruned"], 2)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [1])
        self.assertEqual(protection.tolist(), [False])
        self.assert_consistent(model, protection)

    def test_noop_screen_cleanup_and_original_parameter_identity_are_unchanged(self):
        model = self.model()
        parameters = [group["params"][0] for group in model.optimizer.param_groups]
        summary, protection = self.event(
            model, densify_mask=torch.zeros(3, dtype=torch.bool), return_densify_protection=True,
        )
        self.assertEqual(summary["output_count"], 3)
        self.assertFalse(protection.any())
        self.assertTrue(all(group["params"][0] is old for group, old in zip(model.optimizer.param_groups, parameters)))
        model.max_radii2D[1] = 30
        summary, protection = self.event(
            model, densify_mask=torch.zeros(3, dtype=torch.bool), max_screen_size=20,
            return_densify_protection=True,
        )
        self.assertEqual(summary["cleanup_pruned"], 1)
        self.assertEqual(protection.tolist(), [False, False])
        self.assert_consistent(model, protection)

    def test_prune_all_guard_and_empty_population_return_aligned_masks(self):
        for supported in (False, True):
            with self.subTest(supported=supported):
                model = self.model(opacity=(0.001, 0.003, 0.002))
                summary, protection = self.event(
                    model, densify_mask=torch.full((3,), supported, dtype=torch.bool),
                    return_densify_protection=True,
                )
                self.assertTrue(summary["prune_all_guard"])
                self.assertEqual(model._features_dc[:, 0, 0].tolist(), [1])
                self.assertEqual(protection.tolist(), [supported])
                self.assert_consistent(model, protection)
        empty = self.model_class(1)
        empty._xyz = torch.nn.Parameter(torch.empty(0, 3))
        summary, protection = self.event(empty, return_densify_protection=True)
        self.assertEqual(summary["output_count"], 0)
        self.assertEqual(protection.shape, (0,))
        self.assertEqual(protection.dtype, torch.bool)

    def test_event_mask_is_not_persistent_and_checkpoint_format_is_unchanged(self):
        model = self.model()
        attributes = set(vars(model))
        _, first = self.event(
            model, densify_mask=torch.tensor([True, True, False]), return_densify_protection=True,
        )
        self.assertTrue(first.any())
        count = model.get_xyz.shape[0]
        _, second = self.event(
            model, densify_mask=torch.ones(count, dtype=torch.bool), return_densify_protection=True,
        )
        self.assertFalse(second.any())  # No fresh accumulated gradients.
        self.assertEqual(set(vars(model)), attributes)
        self.assertEqual(len(model.capture()), 12)

    def test_opt_in_is_identical_to_default_for_summary_topology_adam_gradients_and_rng(self):
        for cleanup in (None, 20):
            with self.subTest(max_screen_size=cleanup):
                initial = self.model(
                    scales=(0.005, 0.2, 0.02, 0.005, 0.005),
                    opacity=(0.2, 0.2, 0.2, 0.001, 0.2),
                    gradients=(1, 1, 0.1, 1, 1),
                )
                default, tracked = copy.deepcopy(initial), copy.deepcopy(initial)
                support = torch.tensor([True, True, True, True, True])
                random_state = torch.random.get_rng_state().clone()
                expected = self.event(default, densify_mask=support, max_screen_size=cleanup)
                expected_rng = torch.random.get_rng_state().clone()
                torch.random.set_rng_state(random_state)
                actual, protection = self.event(
                    tracked, densify_mask=support, max_screen_size=cleanup,
                    return_densify_protection=True,
                )
                self.assertIsInstance(expected, dict)
                self.assertEqual(actual, expected)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
                self.assertEqual(set(vars(default)), set(vars(tracked)))
                self.assert_consistent(tracked, protection)
                for left, right in zip(default.optimizer.param_groups, tracked.optimizer.param_groups):
                    lp, rp = left["params"][0], right["params"][0]
                    self.assertTrue(torch.equal(lp, rp))
                    self.assertIsNone(lp.grad)
                    self.assertIsNone(rp.grad)
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        self.assertTrue(torch.equal(default.optimizer.state[lp][key], tracked.optimizer.state[rp][key]))
                for name in ("xyz_gradient_accum", "denom", "max_radii2D"):
                    self.assertTrue(torch.equal(getattr(default, name), getattr(tracked, name)))

    def test_helper_default_return_contracts_and_exact_selection(self):
        for method_name, scales in (("densify_and_clone", (0.005, 0.005)), ("densify_and_split", (0.02, 0.02))):
            with self.subTest(method=method_name):
                for selection in (None, torch.tensor([True, False])):
                    model = self.model(scales=scales)
                    with cpu_allocations(), torch.no_grad():
                        result = getattr(model, method_name)(model.xyz_gradient_accum, 0.5, 1.0, selection_mask=selection)
                    self.assertEqual(result, None if selection is None else 1)
                model = self.model(scales=scales, gradients=(0.5, 0.1))
                with cpu_allocations(), torch.no_grad():
                    count, selected = getattr(model, method_name)(
                        model.xyz_gradient_accum, 0.5, 1.0,
                        selection_mask=torch.ones(2, dtype=torch.bool), return_selection=True,
                    )
                self.assertEqual(count, 1)
                self.assertEqual(selected.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
