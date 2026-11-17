"""CPU analytic tests of the optional diagnostic rasterizer interface."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DensityRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "density_renderer_under_test", REPO_ROOT / "gaussian_renderer/density.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.camera = SimpleNamespace(
            image_height=2,
            image_width=2,
            FoVx=1.0,
            FoVy=1.0,
            world_view_transform=torch.eye(4, requires_grad=True),
            full_proj_transform=torch.eye(4, requires_grad=True),
            camera_center=torch.zeros(3, requires_grad=True),
        )
        self.pipe = SimpleNamespace(compute_cov3D_python=False, debug=False)
        self.pc = SimpleNamespace(
            get_xyz=torch.zeros((3, 3), requires_grad=True),
            get_opacity=torch.full((3, 1), 0.2, requires_grad=True),
            get_scaling=torch.ones((3, 3), requires_grad=True),
            get_rotation=torch.tensor([[1.0, 0, 0, 0]] * 3, requires_grad=True),
            active_sh_degree=0,
        )
        self.weights = torch.tensor(
            [[[0.2, 0.0], [0.1, 0.0]], [[0.3, 0.4], [0.0, 0.15]], [[0.0, 0.0], [0.0, 0.0]]]
        )
        test = self

        class AnalyticRasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings
                test.settings = raster_settings

            def __call__(self, **kwargs):
                test.raster_inputs = kwargs
                colors = kwargs["colors_precomp"]
                for key in ("means3D", "means2D", "opacities", "scales", "rotations", "cov3D_precomp"):
                    if kwargs[key] is not None:
                        test.assertFalse(kwargs[key].requires_grad, key)
                test.assertTrue(colors.is_leaf)
                test.assertTrue(colors.requires_grad)
                test.assertTrue(torch.equal(colors, torch.full_like(colors, 0.5)))
                image = torch.einsum("nhw,nc->chw", test.weights, colors)
                image = image + self.settings.bg[:, None, None] * (1 - test.weights.sum(dim=0))
                return image, torch.tensor([2, 3, 0])

        self.extension = SimpleNamespace(
            GaussianRasterizationSettings=lambda **kwargs: SimpleNamespace(**kwargs),
            GaussianRasterizer=AnalyticRasterizer,
        )

    def render(self, mask):
        with patch.dict(sys.modules, {"diff_gaussian_rasterization": self.extension}):
            return self.module.render_density_contributions(self.camera, self.pc, self.pipe, mask)

    def test_exact_weighted_sums_and_no_gradient_or_rng_mutation(self):
        self.pc.get_xyz.grad = torch.full_like(self.pc.get_xyz, 7.0)
        self.pc.get_opacity.grad = torch.full_like(self.pc.get_opacity, 9.0)
        mask = torch.tensor([[1.0, 0.0], [0.0, 0.5]], requires_grad=True)
        rng_state = torch.random.get_rng_state().clone()
        with torch.no_grad():
            result = self.render(mask)
        torch.testing.assert_close(result["weight_sum"], torch.tensor([0.3, 0.85, 0.0]))
        torch.testing.assert_close(result["high_error_weight"], torch.tensor([0.2, 0.375, 0.0]))
        self.assertTrue(torch.equal(result["frustum_visible"], torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertTrue(torch.equal(self.pc.get_xyz.grad, torch.full_like(self.pc.get_xyz, 7)))
        self.assertTrue(torch.equal(self.pc.get_opacity.grad, torch.full_like(self.pc.get_opacity, 9)))
        self.assertIsNone(self.pc.get_scaling.grad)
        self.assertIsNone(self.pc.get_rotation.grad)
        self.assertIsNone(mask.grad)
        self.assertIsNone(self.camera.world_view_transform.grad)
        for value in result.values():
            self.assertFalse(value.requires_grad)
        self.assertFalse(self.settings.viewmatrix.requires_grad)
        self.assertFalse(self.settings.projmatrix.requires_grad)
        self.assertFalse(self.settings.campos.requires_grad)

    def test_boolean_all_and_zero_masks(self):
        result = self.render(torch.ones((2, 2), dtype=torch.bool))
        torch.testing.assert_close(result["high_error_weight"], result["weight_sum"])
        result = self.render(torch.zeros((2, 2), dtype=torch.int64))
        self.assertTrue(torch.equal(result["high_error_weight"], torch.zeros(3)))

    def test_python_covariance_path(self):
        covariance = torch.ones((3, 6), requires_grad=True)
        self.pipe.compute_cov3D_python = True
        self.pc.get_covariance = lambda scale: covariance * scale
        self.render(torch.ones((2, 2)))
        self.assertIsNone(self.raster_inputs["scales"])
        self.assertIsNone(self.raster_inputs["rotations"])
        torch.testing.assert_close(self.raster_inputs["cov3D_precomp"], covariance)
        self.assertIsNone(covariance.grad)

    def test_empty_point_cloud_needs_no_rasterizer(self):
        self.pc.get_xyz = torch.empty((0, 3), requires_grad=True)
        result = self.module.render_density_contributions(
            self.camera, self.pc, self.pipe, torch.zeros((2, 2))
        )
        for value in result.values():
            self.assertEqual(tuple(value.shape), (0,))
            self.assertFalse(value.requires_grad)
        self.assertEqual(result["frustum_visible"].dtype, torch.bool)

    def test_invalid_masks_raise_before_rasterizing(self):
        with self.assertRaises(TypeError):
            self.render([[0, 0], [0, 0]])
        for mask in (
            torch.zeros((1, 2, 2)),
            torch.full((2, 2), float("nan")),
            torch.full((2, 2), float("inf")),
            torch.full((2, 2), -0.1),
            torch.full((2, 2), 1.1),
        ):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.render(mask)
        with self.assertRaises(TypeError):
            self.render(torch.zeros((2, 2), dtype=torch.complex64))


if __name__ == "__main__":
    unittest.main()
