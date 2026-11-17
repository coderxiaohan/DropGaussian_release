"""Read-only compositing attribution for view-consistent density control.

FastGS motivates using reconstruction errors across training views to guide
density control: https://arxiv.org/abs/2511.04283.  This helper is an independent
adaptation to the existing rasterizer, not its unweighted footprint counter.
It measures alpha-times-transmittance attribution, not a causal improvement
in image quality from adding or deleting an individual Gaussian.
"""

import math

import torch


@torch.enable_grad()
def render_density_contributions(camera, pc, pipe, high_error_mask):
    """Return detached per-Gaussian compositing weights for one training view.

    ``high_error_mask`` must have shape ``(H, W)`` and contain finite real
    values in ``[0, 1]``; Boolean masks are accepted.  The returned dictionary
    contains ``weight_sum`` and ``high_error_weight`` (both shape ``(N,)``),
    and ``frustum_visible`` (Boolean shape ``(N,)``).  The last field follows
    the existing renderer's ``radii > 0`` convention: it does not establish
    that a Gaussian contributes through occluders.

    For fixed geometry and opacity, rendered color is linear in per-Gaussian
    RGB.  Differentiating channel 0 with pixel weights one gives
    ``sum_p alpha_i(p) * T_i(p)``; channel 1, weighted by the error mask, gives
    its high-error counterpart.  Constant RGB/background 0.5 keep the probe
    independent of appearance; the raw rasterizer output is never clamped.

    All model and camera tensors are detached.  ``autograd.grad`` targets
    only a fresh color tensor, so existing training gradients, optimizer
    state, random-number state, and DropGaussian masks are untouched.  The
    local enable-grad context also supports callers inside ``no_grad``.
    """
    height = int(camera.image_height)
    width = int(camera.image_width)
    if height <= 0 or width <= 0:
        raise ValueError("Density diagnostic image dimensions must be positive")
    if not isinstance(high_error_mask, torch.Tensor):
        raise TypeError("high_error_mask must be a torch.Tensor")
    if tuple(high_error_mask.shape) != (height, width):
        raise ValueError("high_error_mask must have shape (image_height, image_width)")
    if high_error_mask.is_complex() or high_error_mask.is_quantized:
        raise TypeError("high_error_mask must contain real, non-quantized values")
    if high_error_mask.layout != torch.strided:
        raise TypeError("high_error_mask must be a dense tensor")
    if not bool(torch.isfinite(high_error_mask).all()):
        raise ValueError("high_error_mask must contain only finite values")
    if bool(((high_error_mask < 0) | (high_error_mask > 1)).any()):
        raise ValueError("high_error_mask values must lie in [0, 1]")

    with torch.no_grad():
        xyz = pc.get_xyz.detach()
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not xyz.is_floating_point():
            raise ValueError("Gaussian positions must be a floating-point (N, 3) tensor")
        mask = high_error_mask.detach().to(device=xyz.device, dtype=xyz.dtype)
        point_count = int(xyz.shape[0])
        if point_count == 0:
            return {
                "weight_sum": xyz.new_empty((0,)),
                "high_error_weight": xyz.new_empty((0,)),
                "frustum_visible": torch.empty((0,), dtype=torch.bool, device=xyz.device),
            }

        opacity = pc.get_opacity.detach()
        if pipe.compute_cov3D_python:
            covariance = pc.get_covariance(1.0).detach()
            scales = rotations = None
        else:
            covariance = None
            scales = pc.get_scaling.detach()
            rotations = pc.get_rotation.detach()

    # Lazy import keeps the optional controller decoupled from CUDA extension
    # loading until a nonempty diagnostic render is actually requested.
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=xyz.new_full((3,), 0.5),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform.detach(),
        projmatrix=camera.full_proj_transform.detach(),
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center.detach(),
        prefiltered=False,
        debug=pipe.debug,
    )
    colors = xyz.new_full((point_count, 3), 0.5, requires_grad=True)
    diagnostic_image, radii = GaussianRasterizer(raster_settings=settings)(
        means3D=xyz,
        means2D=torch.zeros_like(xyz),
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance,
    )
    pixel_weights = torch.stack((torch.ones_like(mask), mask, torch.zeros_like(mask)))
    color_gradient = torch.autograd.grad(
        outputs=diagnostic_image,
        inputs=colors,
        grad_outputs=pixel_weights,
        create_graph=False,
        retain_graph=False,
    )[0].detach()
    if not bool(torch.isfinite(color_gradient).all()):
        raise FloatingPointError("Density diagnostic returned non-finite compositing weights")
    return {
        "weight_sum": color_gradient[:, 0].clamp_min(0).contiguous(),
        "high_error_weight": color_gradient[:, 1].clamp_min(0).contiguous(),
        "frustum_visible": (radii.detach() > 0).reshape(-1),
    }
