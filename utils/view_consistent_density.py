"""Training-view-only error-guided densification (EGD).

EGD uses alpha-compositing-weighted error-region participation in DISTINCT
views, not a claim that splitting has a known counterfactual benefit.
No test images or MVS scores are used. Pruning is handled separately.

Reference: https://github.com/fastgs/FastGS/tree/fast-dropgaussian
This is an independent adaptation, not the upstream count-based CUDA policy.
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class DensityControlConfig:
    """Density selection, including legacy names used by the pruning router."""

    mode: str = "none"
    error_quantile: float = 0.75
    min_views: int = 2

    def __post_init__(self):
        if self.mode not in {"none", "vcd", "vcp", "vcd_vcp"}:
            raise ValueError("density_control must be none, vcd, vcp, or vcd_vcp")
        if not math.isfinite(self.error_quantile) or not 0.0 < self.error_quantile < 1.0:
            raise ValueError("error_quantile must be finite and strictly between 0 and 1")
        if isinstance(self.min_views, bool) or not isinstance(self.min_views, int) or self.min_views < 2:
            raise ValueError("density_min_views must be an integer >= 2")

    @classmethod
    def from_options(cls, options):
        return cls(
            mode=getattr(options, "density_control", "none"),
            error_quantile=getattr(options, "vcd_error_quantile", 0.75),
            min_views=getattr(options, "density_min_views", 2),
        )

    @property
    def use_vcd(self):
        return self.mode in {"vcd", "vcd_vcp"}

    @property
    def use_vcp(self):
        return self.mode in {"vcp", "vcd_vcp"}

    def as_dict(self):
        return asdict(self)


@dataclass
class DensityDecision:
    densify_mask: Optional[torch.Tensor]
    prune_mask: Optional[torch.Tensor]
    diagnostics: dict


def distinct_training_cameras(cameras):
    """Do not allow repeated camera objects/IDs to create multiview evidence."""
    seen = set()
    result = []
    for camera in cameras:
        name = getattr(camera, "image_name", None)
        uid = getattr(camera, "uid", None)
        key = ("image", name) if name else (("uid", uid) if uid is not None else ("object", id(camera)))
        if key not in seen:
            seen.add(key)
            result.append(camera)
    return result


def high_error_mask(image, ground_truth, quantile):
    """Relative high-error regions; a constant error map has no upper tail."""
    if image.shape != ground_truth.shape or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("density scoring expects matching RGB [3,H,W] images")
    if not torch.isfinite(image).all() or not torch.isfinite(ground_truth).all():
        raise ValueError("non-finite density diagnostic image")
    errors = (image.detach() - ground_truth.detach()).abs().mean(dim=0)
    cutoff = torch.quantile(errors.reshape(-1), quantile)
    return errors > cutoff


def view_evidence(weight_sum, high_error_weight, frustum_visible, error_mask):
    """Return an EGD support vote for one view.

    A vote requires more high-error participation than the image-wide
    high-error area fraction. A large splat covering the whole image therefore
    does not get a vote just because it covers a few difficult pixels.
    Frustum presence alone is not interpreted as compositing contribution.
    """
    if weight_sum.ndim != 1 or high_error_weight.shape != weight_sum.shape or frustum_visible.shape != weight_sum.shape:
        raise ValueError("density contributions must be aligned [N] tensors")
    if frustum_visible.dtype != torch.bool:
        raise ValueError("frustum visibility must be boolean")
    if not torch.isfinite(weight_sum).all() or not torch.isfinite(high_error_weight).all():
        raise ValueError("non-finite per-Gaussian density contribution")
    if (weight_sum < 0).any() or (high_error_weight < 0).any():
        raise ValueError("density contributions must be nonnegative")
    eps = torch.finfo(weight_sum.dtype).eps
    positive = weight_sum > 0
    high_fraction = high_error_weight / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)
    return frustum_visible & positive & (high_fraction > error_mask.float().mean() + eps)


@torch.no_grad()
def evaluate_density_control(
    gaussians, train_cameras, pipe, background, config, lambda_dssim=None,
    render_fn=None, contribution_fn=None, native_opacity_threshold=None,
):
    """Recompute event-local EGD evidence without modifying the model.

    The default-off path makes no renders. Fresh evidence requires no extra
    checkpoint state and never consumes training RNG or changes .grad fields.
    A numeric diagnostic failure conservatively disables EGD for this event.
    The returned pruning mask is always empty; pruning has its own controller.

    lambda_dssim and native_opacity_threshold remain accepted for caller
    compatibility and do not affect EGD.
    """
    if config.mode == "none":
        return DensityDecision(None, None, {"mode": "none"})
    if config.use_vcp:
        raise ValueError(
            "pruning requires the separate pruning controller; "
            "pass the periodic EGD configuration to evaluate_density_control"
        )
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    if contribution_fn is None:
        from gaussian_renderer.density import render_density_contributions as contribution_fn

    xyz = gaussians.get_xyz
    count = xyz.shape[0]
    zeros = torch.zeros(count, dtype=torch.bool, device=xyz.device)
    cameras = distinct_training_cameras(train_cameras)
    diagnostics = {
        "mode": config.mode,
        "input_count": int(count),
        "evaluated_views": len(cameras),
        "vcd_supported": 0,
    }

    def conservative(reason):
        diagnostics["skip_reason"] = reason
        return DensityDecision(zeros.clone(), zeros.clone(), diagnostics)

    if count == 0 or len(cameras) < config.min_views:
        return conservative("insufficient_distinct_training_views_or_points")

    votes = torch.zeros(count, dtype=torch.int32, device=xyz.device)
    seen = torch.zeros_like(votes)
    for camera in cameras:
        gt = camera.original_image.to(device=xyz.device)
        image = render_fn(camera, gaussians, pipe, background, is_train=False)["render"]
        if not torch.isfinite(image).all() or not torch.isfinite(gt).all():
            return conservative("nonfinite_diagnostic_image")
        mask = high_error_mask(image, gt, config.error_quantile)
        try:
            values = contribution_fn(camera, gaussians, pipe, mask)
        except FloatingPointError:
            # Do not hide unrelated CUDA/OOM/programming exceptions.
            return conservative("nonfinite_diagnostic_contribution")
        mass = values["weight_sum"]
        high = values["high_error_weight"]
        visible = values["frustum_visible"]
        if not torch.isfinite(mass).all() or not torch.isfinite(high).all():
            return conservative("nonfinite_diagnostic_contribution")
        vote = view_evidence(mass, high, visible, mask)
        votes += vote.to(votes.dtype)
        seen += visible.to(seen.dtype)

    densify_mask = votes >= config.min_views
    diagnostics["vcd_supported"] = int(densify_mask.sum().item())
    diagnostics["observed_in_multiple_views"] = int((seen >= config.min_views).sum().item())
    return DensityDecision(densify_mask, zeros.clone(), diagnostics)
