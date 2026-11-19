"""CPU interface tests; opt-in CUDA tests check the real unweighted counter.

CUDA tests run only with RUN_FASTGS_CUDA_TESTS=1. The default test run never
builds the extension or requests a CUDA device.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_backend_module():
    spec = importlib.util.spec_from_file_location(
        "fastgs_pruning_backend_under_test", ROOT / "gaussian_renderer" / "fastgs_pruning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FastGSPruningBackendCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_backend_module()

    def setUp(self):
        self.camera = SimpleNamespace(
            image_height=2, image_width=2, FoVx=1.0, FoVy=1.0,
            world_view_transform=torch.eye(4, requires_grad=True),
            full_proj_transform=torch.eye(4, requires_grad=True),
            camera_center=torch.zeros(3, requires_grad=True),
            original_image=torch.zeros((3, 2, 2), requires_grad=True),
        )
        self.pc = SimpleNamespace(
            get_xyz=torch.tensor([[0.0, 0, 2], [0.1, 0, 3]], requires_grad=True),
            get_opacity=torch.tensor([[0.98], [0.02]], requires_grad=True),
            get_scaling=torch.ones((2, 3), requires_grad=True),
            get_rotation=torch.tensor([[1.0, 0, 0, 0]] * 2, requires_grad=True),
            get_features=torch.zeros((2, 1, 3), requires_grad=True),
            active_sh_degree=0, max_sh_degree=0,
        )
        self.pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        self.background = torch.zeros(3, requires_grad=True)

    def test_mask_is_minmax_not_quantile_and_uses_strict_threshold(self):
        image = torch.tensor([[0.0, 0.1], [0.2, 1.0]]).expand(3, -1, -1)
        mask = self.module.fastgs_error_mask(image, torch.zeros_like(image), 0.1)
        self.assertTrue(torch.equal(mask, torch.tensor([[False, False], [True, True]])))

    def test_constant_error_maps_are_empty_including_nonzero_constant(self):
        for value in (0.0, 0.5, 1.0):
            with self.subTest(value=value):
                mask = self.module.fastgs_error_mask(torch.full((3, 2, 2), value), torch.zeros(3, 2, 2))
                self.assertFalse(mask.any())
                self.assertEqual(mask.dtype, torch.bool)

    def test_invalid_errors_and_thresholds_fail_explicitly(self):
        image = torch.zeros(3, 2, 2)
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.module.fastgs_error_mask(image, image, threshold)
        with self.assertRaises(TypeError):
            self.module.fastgs_error_mask(image, image, True)
        with self.assertRaises(ValueError):
            self.module.fastgs_error_mask(image, torch.zeros(2, 2))
        with self.assertRaises(FloatingPointError):
            self.module.fastgs_error_mask(torch.full_like(image, float("nan")), image)

    def test_public_interface_clamps_rgb_detaches_and_keeps_counts_unweighted(self):
        test = self
        raw_image = torch.tensor([[0.0, 0.1], [0.5, 2.0]]).expand(3, -1, -1)
        calls = []

        class Context:
            image = raw_image
            radii = torch.tensor([2, 0], dtype=torch.int32)

            def count(self, mask):
                test.assertTrue(torch.equal(mask, torch.tensor([[False, False], [True, True]])))
                # Equal integers remain equal despite the different opacities.
                return torch.tensor([2, 2], dtype=torch.int32)

        def render(*args):
            calls.append(args)
            test.assertFalse(torch.is_grad_enabled())
            for value in args:
                if isinstance(value, torch.Tensor):
                    test.assertFalse(value.requires_grad)
            return Context()

        self.pc.get_xyz.grad = torch.full_like(self.pc.get_xyz, 7.0)
        state = torch.random.get_rng_state().clone()
        with patch.object(self.module, "_load_backend", return_value=SimpleNamespace(render=render)):
            result = self.module.render_fastgs_pruning_view(self.camera, self.pc, self.pipe, self.background)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(result["high_error_counts"], torch.tensor([2, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(result["frustum_visible"], torch.tensor([True, False])))
        torch.testing.assert_close(result["render"], raw_image.clamp(0, 1))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        self.assertTrue(torch.equal(self.pc.get_xyz.grad, torch.full_like(self.pc.get_xyz, 7.0)))
        self.assertIsNone(self.pc.get_opacity.grad)
        self.assertIsNone(self.camera.original_image.grad)
        self.assertIsNone(self.background.grad)
        self.assertTrue(all(not value.requires_grad for value in result.values()))

    def test_covariance_path_does_not_send_scales_or_rotations(self):
        self.pipe.compute_cov3D_python = True
        self.pc.get_covariance = lambda factor: torch.ones((2, 6), requires_grad=True) * factor
        context = SimpleNamespace(
            image=torch.zeros(3, 2, 2), radii=torch.ones(2, dtype=torch.int32),
            count=lambda mask: torch.zeros(2, dtype=torch.int32),
        )
        seen = []
        def render(*args):
            seen.append(args)
            return context
        with patch.object(self.module, "_load_backend", return_value=SimpleNamespace(render=render)):
            self.module.render_fastgs_pruning_view(self.camera, self.pc, self.pipe, self.background)
        self.assertEqual(seen[0][4].numel(), 0)
        self.assertEqual(seen[0][5].numel(), 0)
        self.assertEqual(tuple(seen[0][6].shape), (2, 6))
        self.assertFalse(seen[0][6].requires_grad)

    def test_empty_points_do_not_compile_or_load_cuda(self):
        self.pc.get_xyz = torch.empty((0, 3), requires_grad=True)
        with patch.object(self.module, "_load_backend", side_effect=AssertionError("must stay lazy")):
            result = self.module.render_fastgs_pruning_view(self.camera, self.pc, self.pipe, self.background)
        self.assertEqual(tuple(result["high_error_counts"].shape), (0,))
        self.assertEqual(tuple(result["render"].shape), (3, 2, 2))

    def test_invalid_model_or_ground_truth_fails_before_loading(self):
        with patch.object(self.module, "_load_backend", side_effect=AssertionError("must stay lazy")):
            self.pc.get_xyz = torch.zeros(2, 4)
            with self.assertRaises(ValueError):
                self.module.render_fastgs_pruning_view(self.camera, self.pc, self.pipe, self.background)
            self.pc.get_xyz = torch.zeros(2, 3)
            self.camera.original_image = torch.zeros(3, 3, 3)
            with self.assertRaises(ValueError):
                self.module.render_fastgs_pruning_view(self.camera, self.pc, self.pipe, self.background)

    def test_nonfinite_native_inputs_fail_before_loading(self):
        for field in ("get_xyz", "get_opacity", "get_scaling", "get_rotation", "get_features"):
            with self.subTest(field=field):
                original = getattr(self.pc, field)
                setattr(self.pc, field, torch.full_like(original, float("nan")))
                with patch.object(self.module, "_load_backend", side_effect=AssertionError("must stay lazy")):
                    with self.assertRaises(FloatingPointError):
                        self.module.render_fastgs_pruning_view(self.camera, self.pc, self.pipe, self.background)
                setattr(self.pc, field, original)

    def test_prepare_backend_only_loads_the_isolated_module(self):
        sentinel = object()
        with patch.object(self.module, "_load_backend", return_value=sentinel) as load:
            self.assertIs(self.module.prepare_fastgs_pruning_backend(), sentinel)
        load.assert_called_once_with()


@unittest.skipUnless(
    torch is not None and os.environ.get("RUN_FASTGS_CUDA_TESTS") == "1",
    "opt-in CUDA tests: set RUN_FASTGS_CUDA_TESTS=1",
)
class FastGSPruningBackendCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        cls.module = load_backend_module()

    def make_scene(self, opacities, size=1):
        from utils.graphics_utils import getProjectionMatrix

        count = len(opacities)
        xyz = torch.zeros((count, 3), device="cuda")
        xyz[:, 2] = torch.arange(count, device="cuda") + 2.0
        camera = SimpleNamespace(
            image_height=size, image_width=size, FoVx=1.0, FoVy=1.0,
            world_view_transform=torch.eye(4, device="cuda"),
            full_proj_transform=getProjectionMatrix(0.01, 100.0, 1.0, 1.0).transpose(0, 1).cuda(),
            camera_center=torch.zeros(3, device="cuda"),
            original_image=torch.zeros((3, size, size), device="cuda"),
        )
        pc = SimpleNamespace(
            get_xyz=xyz, get_opacity=torch.tensor(opacities, device="cuda")[:, None],
            get_scaling=torch.full((count, 3), 0.1, device="cuda"),
            get_rotation=torch.tensor([[1.0, 0, 0, 0]] * count, device="cuda"),
            get_features=torch.zeros((count, 1, 3), device="cuda"),
            active_sh_degree=0, max_sh_degree=0,
        )
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        return camera, pc, pipe, torch.zeros(3, device="cuda")

    def test_real_counter_counts_pixels_not_alpha_weights_and_stops_before_termination(self):
        scene = self.make_scene([0.98, 0.98, 0.9])
        context = self.module._render_context(*scene)
        counts = context.count(torch.ones((1, 1), dtype=torch.bool, device="cuda"))
        # The first two splats have weights .98 and .0196, but BOTH count 1.
        # The third crosses T=1e-4 and must not count (nor blend).
        self.assertEqual(counts.cpu().tolist(), [1, 1, 0])

    def test_real_counter_skips_below_alpha_cutoff_and_empty_mask(self):
        scene = self.make_scene([0.003, 0.2])
        context = self.module._render_context(*scene)
        self.assertEqual(context.count(torch.ones((1, 1), dtype=torch.bool, device="cuda")).cpu().tolist(), [0, 1])
        self.assertEqual(context.count(torch.zeros((1, 1), dtype=torch.bool, device="cuda")).cpu().tolist(), [0, 0])

    def test_render_matches_normal_local_rasterizer_and_nondefault_stream(self):
        from gaussian_renderer import render

        scene = self.make_scene([0.6, 0.3], size=8)
        camera, pc, pipe, background = scene
        expected = render(camera, pc, pipe, background, is_train=False)["render"].detach()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = self.module.render_fastgs_pruning_view(*scene)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(result["render"], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(result["high_error_counts"].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
