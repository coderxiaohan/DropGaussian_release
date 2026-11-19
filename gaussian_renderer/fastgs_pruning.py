"""Isolated, forward-only FastGS pruning evidence.

Fast-DropGaussian counts *pixels*, not alpha-times-transmittance weights:
https://github.com/fastgs/FastGS/blob/fast-dropgaussian/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu
Its error mask is a min-max-normalized RGB L1 map above ``loss_thresh``:
https://github.com/fastgs/FastGS/blob/fast-dropgaussian/utils/fast_utils.py

This backend compiles the repository's existing rasterizer sources into a
separately named extension. A read-only count pass uses buffers created by
that SAME extension; it never interprets the installed rasterizer's private
ABI. Ordinary rendering and the existing EGD contribution backend are untouched.
The local rasterizer's geometry/tile coverage is retained (not FastGS's compact
tile-box optimization). There is no backward pass, dropout, or training RNG use.
"""

from functools import lru_cache
import hashlib
import math
from pathlib import Path

import torch


def fastgs_error_mask(image, ground_truth, loss_threshold=0.1):
    """Return the official strict error mask, with constant maps safely empty."""
    if isinstance(loss_threshold, bool) or not isinstance(loss_threshold, (int, float)):
        raise TypeError("loss_threshold must be a real number")
    if not math.isfinite(loss_threshold) or not 0 <= loss_threshold <= 1:
        raise ValueError("loss_threshold must be finite and in [0, 1]")
    if image.ndim != 3 or image.shape[0] != 3 or image.shape != ground_truth.shape:
        raise ValueError("FastGS pruning requires matching RGB [3,H,W] images")
    if image.shape[1] == 0 or image.shape[2] == 0:
        raise ValueError("FastGS pruning requires nonempty images")
    if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(ground_truth).all()):
        raise FloatingPointError("FastGS pruning received a nonfinite image")
    errors = (image.detach() - ground_truth.detach()).abs().mean(dim=0)
    if not bool(torch.isfinite(errors).all()):
        raise FloatingPointError("FastGS pruning received nonfinite RGB errors")
    low, high = errors.min(), errors.max()
    if bool(high == low):
        return torch.zeros_like(errors, dtype=torch.bool)
    return ((errors - low) / (high - low) > loss_threshold).contiguous()


@lru_cache(maxsize=1)
def _load_backend():
    # Import/build lazily: baseline and EGD-only do not need this extension.
    from torch.utils.cpp_extension import load

    local = Path(__file__).resolve().parent / "fastgs_count"
    rasterizer = local.parents[1] / "submodules" / "diff-gaussian-rasterization"
    cuda = rasterizer / "cuda_rasterizer"
    sources = [
        local / "bindings.cpp",
        local / "count.cu",
        cuda / "rasterizer_impl.cu",
        cuda / "forward.cu",
        # rasterizer_impl also defines backward; link its dependencies without
        # exporting or invoking a backward interface from this extension.
        cuda / "backward.cu",
    ]
    dependencies = sources + [local / "count.h"] + sorted(cuda.glob("*.h"))
    digest = hashlib.sha256()
    for path in dependencies:
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    name = "dropgaussian_fastgs_count_" + digest.hexdigest()[:12]
    # Hide/rename reused symbols so an already imported training rasterizer
    # cannot interpose its functions. No installation or global cache override.
    definitions = [
        "-DCudaRasterizer=FastGSPruningRasterizer",
        "-DFORWARD=FastGSPruningForward",
        "-DBACKWARD=FastGSPruningBackward",
    ]
    try:
        return load(
            name=name,
            sources=[str(path) for path in sources],
            extra_include_paths=[str(rasterizer), str(rasterizer / "third_party" / "glm")],
            extra_cflags=["-O2", "-fvisibility=hidden", *definitions],
            # The old source assumes transitive <cstdint> includes. Supply it
            # explicitly for newer CUDA/GCC without editing that source tree.
            extra_cuda_cflags=["-O2", "-include", "cstdint", "-Xcompiler=-fvisibility=hidden", *definitions],
            with_cuda=True,
            verbose=False,
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "The isolated FastGS pixel-count backend could not be built/loaded. "
            "It needs this checkout's rasterizer sources, PyTorch's CUDA build "
            "toolchain, and ninja; no alpha*T approximation is substituted."
        ) from error


def prepare_fastgs_pruning_backend():
    """Build/load the isolated backend without rendering or launching kernels.

    A build-only process can set CUDA_VISIBLE_DEVICES='' together with an
    explicit TORCH_CUDA_ARCH_LIST, so compilation does not inspect a GPU.
    """
    return _load_backend()


def _detached_float(value, device, name="input"):
    result = value.detach().to(device=device, dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("FastGS pruning received nonfinite " + name)
    return result


def _render_context(camera, pc, pipe, background):
    """Create an opaque native context whose buffers belong to this backend."""
    xyz = _detached_float(pc.get_xyz, pc.get_xyz.device, "positions")
    device = xyz.device
    empty = xyz.new_empty((0,))
    if pipe.compute_cov3D_python:
        covariance = _detached_float(pc.get_covariance(1.0), device, "covariance")
        scales, rotations = empty, empty
    else:
        covariance = empty
        scales = _detached_float(pc.get_scaling, device, "scales")
        rotations = _detached_float(pc.get_rotation, device, "rotations")
    features = _detached_float(pc.get_features, device, "SH coefficients")
    colors, sh = empty, features
    if getattr(pipe, "convert_SHs_python", False):
        from utils.sh_utils import eval_sh

        directions = xyz - _detached_float(camera.camera_center, device, "camera center")
        directions = directions / directions.norm(dim=1, keepdim=True)
        colors = _detached_float(
            (eval_sh(pc.active_sh_degree, features.transpose(1, 2), directions) + 0.5).clamp_min(0),
            device, "SH colors",
        )
        sh = empty
    # Materialize/validate every input BEFORE loading or calling native code.
    inputs = (
        _detached_float(background, device, "background"),
        xyz.contiguous(),
        colors,
        _detached_float(pc.get_opacity, device, "opacity"),
        scales,
        rotations,
        covariance,
        _detached_float(camera.world_view_transform, device, "view matrix"),
        _detached_float(camera.full_proj_transform, device, "projection matrix"),
        math.tan(camera.FoVx * 0.5),
        math.tan(camera.FoVy * 0.5),
        int(camera.image_height),
        int(camera.image_width),
        sh,
        int(pc.active_sh_degree),
        _detached_float(camera.camera_center, device, "camera center"),
        bool(getattr(pipe, "debug", False)),
    )
    return _load_backend().render(*inputs)


@torch.no_grad()
def render_fastgs_pruning_view(camera, pc, pipe, background, loss_threshold=0.1):
    """Return RGB, exact unweighted high-error pixel counts, and visibility.

    Outputs are detached. The RGB image has the same [0,1] clamp as the normal
    DropGaussian renderer. Each counted pixel must pass alpha >= 1/255 and the
    *post-update* transmittance test T*(1-alpha) >= 1e-4. A point contributes at
    most one integer count per masked pixel, irrespective of alpha*T magnitude.
    """
    xyz = pc.get_xyz.detach()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype != torch.float32:
        raise ValueError("FastGS pruning requires float32 Gaussian positions [N,3]")
    height, width = int(camera.image_height), int(camera.image_width)
    if height <= 0 or width <= 0:
        raise ValueError("FastGS pruning requires positive image dimensions")
    if background.shape != (3,) or not bool(torch.isfinite(background).all()):
        raise ValueError("FastGS pruning requires a finite RGB background")
    ground_truth = _detached_float(camera.original_image, xyz.device, "ground truth")
    if ground_truth.shape != (3, height, width):
        raise ValueError("camera ground truth must match its RGB image dimensions")
    if xyz.shape[0] == 0:
        image = _detached_float(background, xyz.device)[:, None, None].expand(3, height, width).clamp(0, 1)
        # Validate the same public inputs without compiling a CUDA extension.
        fastgs_error_mask(image, ground_truth, loss_threshold)
        return {
            "render": image,
            "high_error_counts": torch.empty((0,), dtype=torch.int32, device=xyz.device),
            "frustum_visible": torch.empty((0,), dtype=torch.bool, device=xyz.device),
        }
    context = _render_context(camera, pc, pipe, background)
    image = context.image.detach().clamp(0, 1)
    mask = fastgs_error_mask(image, ground_truth, loss_threshold)
    counts = context.count(mask).detach()
    if counts.shape != (xyz.shape[0],) or counts.dtype != torch.int32:
        raise RuntimeError("FastGS count backend returned invalid integer counts")
    if bool((counts < 0).any()):
        raise FloatingPointError("FastGS pixel counts overflowed")
    return {
        "render": image,
        "high_error_counts": counts,
        "frustum_visible": (context.radii.detach() > 0).reshape(-1),
    }
