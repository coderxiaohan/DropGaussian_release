"""Density-control indexing tests; CUDA fixtures exercise the real Adam helpers."""

import copy
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from utils.graphics_utils import BasicPointCloud

try:
    import torch
except ImportError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_model():
    # Loading this single module avoids importing the full dataset/renderer
    # stack. These tests never call the CUDA nearest-neighbour initializer.
    package = ModuleType("simple_knn")
    extension = ModuleType("simple_knn._C")
    extension.distCUDA2 = mock.Mock(side_effect=AssertionError("initializer not used"))
    spec = importlib.util.spec_from_file_location(
        "view_consistent_model_test_target", REPO_ROOT / "scene/gaussian_model.py",
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict("sys.modules", {"simple_knn": package, "simple_knn._C": extension}):
        spec.loader.exec_module(module)
    return module.GaussianModel


GaussianModel = _load_model() if torch is not None else None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DirectMvsInitializationTest(unittest.TestCase):
    def test_direct_points_get_original_gaussian_attributes(self):
        # Run the real initializer on CPU with only its device allocation and
        # CUDA KNN primitive substituted. No adaptive geometry is generated.
        cloud = BasicPointCloud(
            points=np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            colors=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            normals=np.zeros((3, 3), dtype=np.float32),
        )
        model = GaussianModel(1)
        original_zeros, original_ones = torch.zeros, torch.ones

        def allocate_cpu(factory):
            def allocate(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return factory(*args, **kwargs)
            return allocate

        knn = mock.Mock(return_value=torch.tensor([1.0, 4.0, 9.0]))
        with mock.patch.dict(GaussianModel.create_from_pcd.__globals__, {"distCUDA2": knn}), \
                mock.patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor), \
                mock.patch("torch.zeros", side_effect=allocate_cpu(original_zeros)), \
                mock.patch("torch.ones", side_effect=allocate_cpu(original_ones)), \
                mock.patch("builtins.print"):
            model.create_from_pcd(cloud, spatial_lr_scale=2.5)

        self.assertEqual(model.spatial_lr_scale, 2.5)
        torch.testing.assert_close(model.get_xyz, torch.from_numpy(cloud.points))
        knn.assert_called_once()
        torch.testing.assert_close(knn.call_args.args[0], torch.from_numpy(cloud.points))
        torch.testing.assert_close(
            model.get_scaling, torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
        )
        torch.testing.assert_close(
            model.get_rotation, torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
        )
        torch.testing.assert_close(model.get_opacity, torch.full((3, 1), 0.1))
        self.assertEqual(tuple(model.get_features.shape), (3, 4, 3))
        self.assertTrue(torch.all(model._features_rest == 0))
        self.assertEqual(tuple(model.max_radii2D.shape), (3,))


def _optimization_args():
    return SimpleNamespace(
        percent_dense=0.01, position_lr_init=0.00016,
        position_lr_final=0.0000016, position_lr_delay_mult=0.01,
        position_lr_max_steps=10000, feature_lr=0.0025, opacity_lr=0.05,
        scaling_lr=0.005, rotation_lr=0.001,
    )


def _model(device="cpu", scales=(0.005, 0.2, 0.005), opacity=None, optimizer=False):
    count = len(scales)
    model = GaussianModel(1)
    values = torch.arange(count, device=device, dtype=torch.float32)
    model._xyz = torch.nn.Parameter(torch.stack((values, values + 1, values + 2), dim=1))
    model._features_dc = torch.nn.Parameter(values[:, None, None].expand(count, 1, 3).clone())
    model._features_rest = torch.nn.Parameter(torch.zeros(count, 3, 3, device=device))
    model._scaling = torch.nn.Parameter(torch.tensor(scales, device=device).log()[:, None].repeat(1, 3))
    rotation = torch.zeros(count, 4, device=device)
    rotation[:, 0] = 1
    model._rotation = torch.nn.Parameter(rotation)
    opacity = [0.2] * count if opacity is None else opacity
    model._opacity = torch.nn.Parameter(torch.logit(torch.tensor(opacity, device=device))[:, None])
    model.max_radii2D = torch.zeros(count, device=device)
    model.spatial_lr_scale = 1.0
    model.percent_dense = 0.01
    if optimizer:
        model.training_setup(_optimization_args())
        # Initialize Adam state without moving the fixture geometry.
        for group in model.optimizer.param_groups:
            group["params"][0].grad = torch.zeros_like(group["params"][0])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    model.xyz_gradient_accum = torch.ones(count, 1, device=device)
    model.denom = torch.ones(count, 1, device=device)
    return model


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DensityMaskValidationTest(unittest.TestCase):
    def test_masks_require_bool_vector_and_exact_model_device(self):
        model = _model()
        valid = torch.tensor([True, False, True])
        model._validate_density_mask(valid, 3, "mask")
        for invalid in ([True, False, True], torch.ones(3)):
            with self.assertRaises(TypeError):
                model._validate_density_mask(invalid, 3, "mask")
        for invalid in (valid[:2], valid[:, None]):
            with self.assertRaises(ValueError):
                model._validate_density_mask(invalid, 3, "mask")
        with self.assertRaises(ValueError):
            model._validate_density_mask(torch.empty(3, dtype=torch.bool, device="meta"), 3, "mask")

    def test_all_pruned_keeps_finite_highest_opacity_without_mutating_input(self):
        model = _model(opacity=[0.1, 0.8, 0.9])
        with torch.no_grad():
            model._xyz[2, 0] = torch.nan
        requested = torch.ones(3, dtype=torch.bool)
        actual, protected = model._protect_last_density_point(requested)
        self.assertTrue(protected)
        self.assertEqual(actual.tolist(), [True, False, True])
        self.assertTrue(requested.all())

    def test_all_nonfinite_guard_fails_explicitly(self):
        model = _model()
        with torch.no_grad():
            model._xyz[:] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "finite Gaussian"):
            model._protect_last_density_point(torch.ones(3, dtype=torch.bool))

    def test_legacy_density_call_has_no_new_selection_arguments(self):
        model = _model()
        with mock.patch.object(model, "densify_and_clone") as clone, \
                mock.patch.object(model, "densify_and_split") as split, \
                mock.patch.object(model, "prune_points") as prune, \
                mock.patch("torch.cuda.empty_cache"):
            model.densify_and_prune(0.5, 0.005, 1.0, None)
        self.assertEqual(clone.call_count, 1)
        self.assertEqual(split.call_count, 1)
        self.assertEqual(clone.call_args.kwargs, {})
        self.assertEqual(split.call_args.kwargs, {})
        self.assertEqual(len(clone.call_args.args), 3)
        self.assertEqual(len(split.call_args.args), 3)
        self.assertEqual(prune.call_args.args[0].tolist(), [False, False, False])

    def test_invalid_public_masks_fail_before_any_topology_mutation(self):
        model = _model()
        with mock.patch.object(model, "prune_points") as prune, \
                mock.patch.object(model, "densify_and_clone") as clone:
            with self.assertRaises(ValueError):
                model.densify_and_prune_view_consistent(
                    0.5, 0.005, 1.0, None,
                    torch.ones(3, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                )
        prune.assert_not_called()
        clone.assert_not_called()


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
class ViewConsistentDensityModelTest(unittest.TestCase):
    def assert_consistent_optimizer(self, model):
        count = model.get_xyz.shape[0]
        self.assertEqual(model.xyz_gradient_accum.shape, (count, 1))
        self.assertEqual(model.denom.shape, (count, 1))
        self.assertEqual(model.max_radii2D.shape, (count,))
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            self.assertEqual(parameter.shape[0], count)
            state = model.optimizer.state[parameter]
            self.assertEqual(state["exp_avg"].shape, parameter.shape)
            self.assertEqual(state["exp_avg_sq"].shape, parameter.shape)

    def test_vcp_then_clone_and_split_preserves_original_index_identity(self):
        model = _model("cuda", scales=(0.005, 0.2, 0.005, 0.2), optimizer=True)
        # Make existing rows identifiable in Adam moments as well as features.
        for group in model.optimizer.param_groups:
            state = model.optimizer.state[group["params"][0]]
            state["exp_avg"].fill_(3.0)
            state["exp_avg_sq"].fill_(5.0)
        densify = torch.tensor([True, True, True, False], device="cuda")
        prune = torch.tensor([False, False, True, False], device="cuda")
        summary = model.densify_and_prune_view_consistent(
            0.5, 0.005, 1.0, None, densify, prune,
        )
        self.assertEqual(summary["vcp_pruned"], 1)
        self.assertEqual(summary["cloned"], 1)
        self.assertEqual(summary["split_parents"], 1)
        self.assertEqual(summary["split_children"], 2)
        self.assertEqual(summary["output_count"], 5)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [0, 3, 0, 1, 1])
        self.assertEqual(densify.tolist(), [True, True, True, False])
        self.assertEqual(prune.tolist(), [False, False, True, False])
        self.assert_consistent_optimizer(model)
        for group in model.optimizer.param_groups:
            moment = model.optimizer.state[group["params"][0]]["exp_avg"]
            self.assertTrue(torch.all(moment[:2] == 3))
            self.assertTrue(torch.all(moment[2:] == 0))

    def test_adaptive_noop_preserves_adam_and_legacy_skipped_update(self):
        model = _model("cuda", optimizer=True)
        parameters = [group["params"][0] for group in model.optimizer.param_groups]
        states = [model.optimizer.state[p] for p in parameters]
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        with mock.patch.object(model, "prune_points", wraps=model.prune_points) as prune:
            summary = model.densify_and_prune_view_consistent(
                0.5, 0.005, 1.0, None,
                torch.zeros(3, dtype=torch.bool, device="cuda"),
                torch.zeros(3, dtype=torch.bool, device="cuda"),
            )
        self.assertEqual(summary["output_count"], 3)
        prune.assert_not_called()
        for group, old, state in zip(model.optimizer.param_groups, parameters, states):
            self.assertIs(group["params"][0], old)
            self.assertIs(model.optimizer.state[old], state)
            self.assertIsNone(old.grad)
        self.assertEqual(model.xyz_gradient_accum.sum().item(), 0)
        self.assertEqual(model.denom.sum().item(), 0)

    def test_fully_pruned_population_keeps_one_even_after_opacity_cleanup(self):
        model = _model("cuda", opacity=[0.0001, 0.0003, 0.0002], optimizer=True)
        summary = model.densify_and_prune_view_consistent(
            0.5, 0.005, 1.0, None,
            torch.ones(3, dtype=torch.bool, device="cuda"),
            torch.ones(3, dtype=torch.bool, device="cuda"),
        )
        self.assertTrue(summary["prune_all_guard"])
        self.assertEqual(summary["output_count"], 1)
        self.assertEqual(model._features_dc[0, 0, 0].item(), 1)
        self.assertEqual(summary["cloned"], 0)
        self.assertEqual(summary["split_parents"], 0)
        self.assert_consistent_optimizer(model)

    def test_noop_event_matches_legacy_optimizer_update(self):
        legacy = _model("cuda", optimizer=True)
        legacy.xyz_gradient_accum.zero_()
        adaptive = GaussianModel(1)
        adaptive.restore(copy.deepcopy(legacy.capture()), _optimization_args())
        for model in (legacy, adaptive):
            for group in model.optimizer.param_groups:
                group["params"][0].grad = torch.ones_like(group["params"][0])
        with torch.no_grad():
            legacy.densify_and_prune(0.5, 0.005, 1.0, None)
            adaptive.densify_and_prune_view_consistent(
                0.5, 0.005, 1.0, None,
                torch.zeros(3, dtype=torch.bool, device="cuda"),
            )
            legacy.optimizer.step()
            adaptive.optimizer.step()
        for left, right in zip(legacy.optimizer.param_groups, adaptive.optimizer.param_groups):
            left_parameter, right_parameter = left["params"][0], right["params"][0]
            self.assertTrue(torch.equal(left_parameter, right_parameter))
            left_state = legacy.optimizer.state[left_parameter]
            right_state = adaptive.optimizer.state[right_parameter]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(left_state[key], right_state[key]))

    def test_existing_opacity_cleanup_still_applies(self):
        model = _model("cuda", opacity=[0.2, 0.001, 0.3], optimizer=True)
        summary = model.densify_and_prune_view_consistent(
            0.5, 0.005, 1.0, None,
            torch.zeros(3, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(summary["cleanup_pruned"], 1)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [0, 2])
        self.assert_consistent_optimizer(model)

    def test_gradient_threshold_is_still_required_for_vcd(self):
        model = _model("cuda", optimizer=True)
        model.xyz_gradient_accum.fill_(0.01)
        summary = model.densify_and_prune_view_consistent(
            0.5, 0.005, 1.0, None,
            torch.ones(3, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(summary["cloned"], 0)
        self.assertEqual(summary["split_parents"], 0)
        self.assertEqual(summary["output_count"], 3)

    def test_existing_size_cleanup_still_applies_when_enabled(self):
        model = _model("cuda", scales=(0.005, 0.2, 0.005), optimizer=True)
        model.max_radii2D[2] = 30
        summary = model.densify_and_prune_view_consistent(
            0.5, 0.005, 1.0, 20,
            torch.zeros(3, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(summary["cleanup_pruned"], 2)
        self.assertEqual(model._features_dc[:, 0, 0].tolist(), [0])
        self.assert_consistent_optimizer(model)

    def test_split_masks_exclude_post_clone_rows_even_with_zero_threshold(self):
        model = _model("cuda", scales=(0.005, 0.2), optimizer=True)
        gradients = model.xyz_gradient_accum.clone()
        selection = torch.tensor([True, False], device="cuda")
        with torch.no_grad():
            model.densify_and_clone(gradients, 0.0, 1.0, selection_mask=selection)
            # Artificially enlarge the newborn to test mask protection, not
            # merely the ordinary clone-vs-split size partition.
            model._scaling[-1].fill_(torch.log(torch.tensor(0.2)).item())
            count = model.densify_and_split(gradients, 0.0, 1.0, selection_mask=selection)
        self.assertEqual(count, 0)
        self.assertEqual(model.get_xyz.shape[0], 3)
        self.assert_consistent_optimizer(model)

    def test_legacy_checkpoint_roundtrip_after_density_event(self):
        model = _model("cuda", optimizer=True)
        model.densify_and_prune_view_consistent(
            0.5, 0.005, 1.0, None,
            torch.tensor([True, True, False], device="cuda"),
        )
        captured = copy.deepcopy(model.capture())
        self.assertEqual(len(captured), 12)
        restored = GaussianModel(1)
        restored.restore(captured, _optimization_args())
        self.assertTrue(torch.equal(restored.get_xyz, model.get_xyz))
        self.assertTrue(torch.equal(restored._features_dc, model._features_dc))
        self.assert_consistent_optimizer(restored)
        for left, right in zip(model.optimizer.param_groups, restored.optimizer.param_groups):
            self.assertTrue(torch.equal(
                model.optimizer.state[left["params"][0]]["exp_avg"],
                restored.optimizer.state[right["params"][0]]["exp_avg"],
            ))


if __name__ == "__main__":
    unittest.main()
