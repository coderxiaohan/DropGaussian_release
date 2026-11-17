"""Training-view-only, FastGS-inspired density control.

VCD uses alpha-compositing-weighted error-region participation in DISTINCT
views, not a claim that splitting has a known counterfactual benefit. VCP
proposes low-contribution points and checks their JOINT removal against every
training view before deleting them. No test images or MVS scores are used.

Reference: https://github.com/fastgs/FastGS/tree/fast-dropgaussian
This is an independent adaptation, not the upstream count-based CUDA policy.
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch

from utils.loss_utils import l1_loss, ssim


@dataclass(frozen=True)
class DensityControlConfig:
    mode: str = "none"
    error_quantile: float = 0.75
    contribution_quantile: float = 0.10
    min_views: int = 2

    def __post_init__(self):
        if self.mode not in {"none", "vcd", "vcp", "vcd_vcp"}:
            raise ValueError("density_control must be none, vcd, vcp, or vcd_vcp")
        for name in ("error_quantile", "contribution_quantile"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(name + " must be finite and strictly between 0 and 1")
        if isinstance(self.min_views, bool) or not isinstance(self.min_views, int) or self.min_views < 2:
            raise ValueError("density_min_views must be an integer >= 2")

    @classmethod
    def from_options(cls, options):
        return cls(
            mode=getattr(options, "density_control", "none"),
            error_quantile=getattr(options, "vcd_error_quantile", 0.75),
            contribution_quantile=getattr(options, "vcp_contribution_quantile", 0.10),
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


def view_evidence(weight_sum, high_error_weight, frustum_visible, error_mask, contribution_quantile):
    """Return a support vote and low-contribution evidence for one view.

    A VCD vote requires more high-error participation than the image-wide
    high-error area fraction. A large splat covering the whole image therefore
    does not get a vote just because it covers a few difficult pixels.

    A VCP candidate must be in the strict lower contribution tail (or have
    zero compositing mass). Equal positive contributions never force pruning.
    Frustum presence alone is NOT interpreted as compositing contribution.
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
    vote = frustum_visible & positive & (high_fraction > error_mask.float().mean() + eps)
    positive_weights = weight_sum[frustum_visible & positive]
    low = ~positive
    if positive_weights.numel():
        cutoff = torch.quantile(positive_weights, contribution_quantile)
        low = low | (weight_sum < cutoff)
    # Even a single useful view vetoes VCP; all training views are evaluated.
    useful = frustum_visible & positive & ~low
    return vote, frustum_visible & low, useful


class _PrunedGaussianView:
    """Read-only, physically filtered render proxy; never mutate model/Adam."""

    def __init__(self, model, keep, compute_covariance):
        self.active_sh_degree = model.active_sh_degree
        self.max_sh_degree = model.max_sh_degree
        self.get_xyz = model.get_xyz.detach()[keep]
        self.get_features = model.get_features.detach()[keep]
        self.get_opacity = model.get_opacity.detach()[keep]
        self.get_scaling = model.get_scaling.detach()[keep]
        self.get_rotation = model.get_rotation.detach()[keep]
        self._covariance = model.get_covariance().detach()[keep] if compute_covariance else None

    def get_covariance(self, scaling_modifier=1.0):
        if self._covariance is None or scaling_modifier != 1.0:
            raise ValueError("density probe only supports its fixed unit-scale covariance")
        return self._covariance


def _photometric_loss(image, ground_truth, lambda_dssim):
    # Exactly the current train.py objective, without changing the train loss.
    return l1_loss(image, ground_truth) + lambda_dssim * (1.0 - ssim(image, ground_truth))


@torch.no_grad()
def evaluate_density_control(
    gaussians, train_cameras, pipe, background, config, lambda_dssim,
    render_fn=None, contribution_fn=None,
):
    """Recompute event-local evidence, then verify a proposed VCP batch.

    The default-off path makes no renders. Fresh evidence requires no extra
    checkpoint state and never consumes training RNG or changes .grad fields.
    A failed/unsupported diagnostic conservatively disables both new actions
    for this event; ordinary opacity pruning is still the caller's policy.
    """
    if config.mode == "none":
        return DensityDecision(None, None, {"mode": "none"})
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
        "vcp_candidates": 0,
        "vcp_verified": 0,
        "vcp_check": "not_requested",
    }

    def conservative(reason):
        diagnostics["skip_reason"] = reason
        return DensityDecision(zeros.clone() if config.use_vcd else None, zeros.clone(), diagnostics)

    if count == 0 or len(cameras) < config.min_views:
        return conservative("insufficient_distinct_training_views_or_points")

    votes = torch.zeros(count, dtype=torch.int32, device=xyz.device)
    seen = torch.zeros_like(votes)
    low_views = torch.zeros_like(votes)
    useful_any = zeros.clone()
    total_mass = torch.zeros(count, dtype=xyz.dtype, device=xyz.device)
    baseline_losses = []
    for camera in cameras:
        gt = camera.original_image.to(device=xyz.device)
        image = render_fn(camera, gaussians, pipe, background, is_train=False)["render"]
        if not torch.isfinite(image).all() or not torch.isfinite(gt).all():
            return conservative("nonfinite_diagnostic_image")
        mask = high_error_mask(image, gt, config.error_quantile)
        try:
            values = contribution_fn(camera, gaussians, pipe, mask)
        except FloatingPointError:
            # A numeric probe failure must not become evidence to delete a
            # point. Do not hide unrelated CUDA/OOM/programming exceptions.
            return conservative("nonfinite_diagnostic_contribution")
        mass = values["weight_sum"]
        high = values["high_error_weight"]
        visible = values["frustum_visible"]
        if not torch.isfinite(mass).all() or not torch.isfinite(high).all():
            return conservative("nonfinite_diagnostic_contribution")
        vote, low, useful = view_evidence(mass, high, visible, mask, config.contribution_quantile)
        votes += vote.to(votes.dtype)
        seen += visible.to(seen.dtype)
        low_views += low.to(low_views.dtype)
        useful_any |= useful
        total_mass += mass / mask.numel()
        if config.use_vcp:
            loss = _photometric_loss(image, gt, lambda_dssim)
            if not torch.isfinite(loss):
                return conservative("nonfinite_diagnostic_loss")
            baseline_losses.append(loss.detach())

    densify_mask = votes >= config.min_views if config.use_vcd else None
    diagnostics["vcd_supported"] = int(densify_mask.sum().item()) if densify_mask is not None else 0
    diagnostics["observed_in_multiple_views"] = int((seen >= config.min_views).sum().item())
    prune_mask = zeros.clone()
    if config.use_vcp:
        candidates = (seen >= config.min_views) & (low_views == seen) & ~useful_any
        if densify_mask is not None:
            candidates &= ~densify_mask
        # A nonempty representation is required by the existing rasterizer.
        if candidates.all():
            candidates[total_mass.argmax()] = False
        diagnostics["vcp_candidates"] = int(candidates.sum().item())
        diagnostics["vcp_check"] = "no_candidates"
        if candidates.any():
            proxy = _PrunedGaussianView(gaussians, ~candidates, pipe.compute_cov3D_python)
            accepted = True
            worst_delta = -math.inf
            # Check the entire proposed deletion jointly, not one-by-one:
            # individually redundant Gaussians can be collectively necessary.
            for camera, before in zip(cameras, baseline_losses):
                gt = camera.original_image.to(device=xyz.device)
                after_image = render_fn(camera, proxy, pipe, background, is_train=False)["render"]
                if not torch.isfinite(after_image).all():
                    accepted = False
                    diagnostics["vcp_check"] = "rejected_nonfinite"
                    break
                after = _photometric_loss(after_image, gt, lambda_dssim)
                if not torch.isfinite(after):
                    accepted = False
                    diagnostics["vcp_check"] = "rejected_nonfinite"
                    break
                delta = float((after - before).item())
                worst_delta = max(worst_delta, delta)
                # Numerical roundoff only, not a dataset-specific loss budget.
                tolerance = torch.finfo(before.dtype).eps * torch.maximum(before.abs(), before.new_tensor(1.0))
                if after > before + tolerance:
                    accepted = False
            diagnostics["vcp_max_train_loss_delta"] = worst_delta if math.isfinite(worst_delta) else None
            if accepted:
                prune_mask = candidates
                diagnostics["vcp_check"] = "accepted_all_training_views"
                diagnostics["vcp_verified"] = int(prune_mask.sum().item())
            elif diagnostics["vcp_check"] != "rejected_nonfinite":
                diagnostics["vcp_check"] = "rejected_training_loss_increase"

    return DensityDecision(densify_mask, prune_mask, diagnostics)
