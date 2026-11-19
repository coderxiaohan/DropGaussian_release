"""Late error-score pruning with optional event-local EGD protection (DPP).

Reference: https://github.com/fastgs/FastGS/tree/fast-dropgaussian
``utils/fast_utils.py`` defines the loss-weighted high-error footprint count;
``GaussianModel.final_prune_fastgs`` removes high-score OR low-opacity points.
This adaptation uses ALL distinct training views, the existing rasterizer's
footprints and the existing SSIM implementation. It does not import upstream
random three-view sampling, compact boxes, SH updates or densification rules.
The score is an association with reconstruction error, not a causal estimate
of the benefit of removing a point. There is deliberately no trial-render gate.
An optional event-local EGD veto protects actual densification lineages only;
it does not modify scoring, EGD decisions or ordinary low-opacity/size cleanup.
"""

from dataclasses import asdict, dataclass, replace
import math

import torch

from utils.loss_utils import l1_loss, ssim
from utils.view_consistent_density import DensityDecision, distinct_training_cameras


@dataclass(frozen=True)
class FastGSPruningConfig:
    strategy: str = "none"
    interval: int = 8000
    loss_threshold: float = 0.1
    score_threshold: float = 0.9
    opacity_threshold: float = 0.1
    protect_egd: bool = False
    dpp_only: bool = False

    def __post_init__(self):
        if self.strategy not in {"none", "fastgs"}:
            raise ValueError("resolved vcp_strategy must be none or fastgs")
        if not isinstance(self.protect_egd, bool):
            raise ValueError("fastgs_protect_egd must be a bool")
        if self.protect_egd and self.strategy != "fastgs":
            raise ValueError("fastgs_protect_egd requires vcp_strategy=fastgs")
        if not isinstance(self.dpp_only, bool):
            raise ValueError("dpp_only must be a bool")
        if self.dpp_only and not self.protect_egd:
            raise ValueError("dpp_only requires protected pruning, not unprotected candidate deletion")
        if isinstance(self.interval, bool) or not isinstance(self.interval, int) or self.interval < 1:
            raise ValueError("fastgs_prune_interval must be a positive integer")
        for name in ("loss_threshold", "score_threshold", "opacity_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("fastgs " + name + " must be finite and in [0, 1]")

    @classmethod
    def from_options(cls, options):
        strategy = getattr(options, "vcp_strategy", "auto")
        if strategy not in {"auto", "none", "fastgs"}:
            raise ValueError("vcp_strategy must be auto, none or fastgs")
        protect_egd = getattr(options, "fastgs_protect_egd", False)
        dpp_only = getattr(options, "dpp_only", False)
        # Validate before auto-resolution so malformed flags are not masked.
        if not isinstance(protect_egd, bool):
            raise ValueError("fastgs_protect_egd must be a bool")
        if not isinstance(dpp_only, bool):
            raise ValueError("dpp_only must be a bool")
        if strategy == "auto":
            mode = getattr(options, "density_control", "none")
            strategy = "fastgs" if mode in {"vcp", "vcd_vcp"} else "none"
            if strategy == "fastgs" and (mode == "vcd_vcp" or dpp_only):
                protect_egd = True
        return cls(
            strategy=strategy,
            interval=getattr(options, "fastgs_prune_interval", 8000),
            loss_threshold=getattr(options, "fastgs_loss_threshold", 0.1),
            score_threshold=getattr(options, "fastgs_prune_threshold", 0.9),
            opacity_threshold=getattr(options, "fastgs_opacity_threshold", 0.1),
            protect_egd=protect_egd,
            dpp_only=dpp_only,
        )

    @property
    def enabled(self):
        return self.strategy == "fastgs"

    def should_prune(self, iteration, total_iterations):
        # No post-save final-step mutation: leave at least one optimization step.
        return self.enabled and 0 < iteration < total_iterations and iteration % self.interval == 0

    def as_dict(self):
        return asdict(self)


def periodic_density_config(density_config, pruning_config):
    """Route only EGD to periodic diagnostics; late pruning has its own event."""
    if not pruning_config.enabled:
        if density_config.use_vcp:
            raise ValueError("density_control=vcp or vcd_vcp requires an enabled pruning strategy")
        return density_config
    if not density_config.use_vcp:
        raise ValueError("vcp_strategy=fastgs requires density_control=vcp or vcd_vcp")
    if pruning_config.dpp_only and density_config.mode != "vcp":
        raise ValueError("dpp_only requires density_control=vcp: native densification, no EGD gate")
    if pruning_config.protect_egd and not density_config.use_vcd and not pruning_config.dpp_only:
        raise ValueError("fastgs_protect_egd requires density_control=vcd_vcp for this-event EGD evidence")
    return replace(density_config, mode="vcd" if density_config.use_vcd else "none")


def dpp_protection_config(density_config, pruning_config):
    """Shared multiview evidence for DPP, NOT a gate on native densification.

    Only the dpp_only ablation needs a separate diagnostic configuration. The
    combined method already gets this evidence from its normal EGD update.
    """
    if not pruning_config.dpp_only:
        return None
    periodic_density_config(density_config, pruning_config)  # Validate the ablation.
    return replace(density_config, mode="vcd")


def validate_dpp_initialization(config, dataset):
    """The standalone ablation must not silently use the default MVS init."""
    if config.dpp_only and (
        getattr(dataset, "init_mode", None) != "colmap_dense"
        or getattr(dataset, "mvs_complement_mode", "none") != "none"
    ):
        raise ValueError("dpp_only requires original initialization: init_mode=colmap_dense and mvs_complement_mode=none")


def validate_training_schedule(config, options, white_background=False):
    """Reject late pruning immediately after an opacity reset, before training.

    Native reset caps every opacity at 0.01; applying the official 0.1 pruning
    threshold in that same event would destroy nearly the entire population.
    The default 8k event in the 10k protocol does not collide with a reset.
    """
    if not config.enabled:
        return
    if config.interval >= options.iterations:
        raise ValueError("fastgs_prune_interval must be smaller than iterations to leave optimization after pruning")
    reset_limit = min(options.iterations, options.densify_until_iter)
    first_joint_event = math.lcm(config.interval, options.opacity_reset_interval)
    white_reset_collision = (
        white_background and 0 < options.densify_from_iter < reset_limit
        and options.densify_from_iter % config.interval == 0
    )
    if first_joint_event < reset_limit or white_reset_collision:
        raise ValueError(
            "FastGS pruning and opacity reset must not occur in the same iteration; "
            "choose fastgs_prune_interval without a reset collision (default 8000 for 10000 iterations)"
        )
    if config.protect_egd:
        adc_interval = options.densification_interval
        if isinstance(adc_interval, bool) or not isinstance(adc_interval, int) or adc_interval < 1:
            raise ValueError("EGD protection requires a positive integer densification_interval")
        last_event = ((options.iterations - 1) // config.interval) * config.interval
        if (config.interval % adc_interval != 0
                or config.interval <= options.densify_from_iter
                or last_event >= options.densify_until_iter):
            raise ValueError(
                "EGD protection requires every FastGS pruning event to coincide with active "
                "densification: iteration > densify_from_iter, iteration < densify_until_iter "
                "and iteration % densification_interval == 0"
            )


def normalized_pruning_score(raw_score):
    """Upstream min-max score, with an explicit constant-score zero policy."""
    if raw_score.ndim != 1 or not raw_score.is_floating_point():
        raise ValueError("FastGS scores must be a floating-point [N] tensor")
    if not torch.isfinite(raw_score).all() or (raw_score < 0).any():
        raise FloatingPointError("FastGS scores must be finite and nonnegative")
    if raw_score.numel() == 0:
        return raw_score.clone(), True
    low, high = raw_score.min(), raw_score.max()
    if high == low:
        return torch.zeros_like(raw_score), True
    return (raw_score - low) / (high - low), False


@torch.no_grad()
def evaluate_fastgs_pruning(gaussians, train_cameras, pipe, background, config,
                           render_fn=None, protection_mask=None):
    """Compute a direct deletion mask on the current, post-ADC population.

    No model/optimizer mutation, test images, random camera selection, temporary
    removal, or additional quality-loss acceptance threshold. Non-finite
    evidence disables this event instead of turning a failed probe into deletion.
    ``render_fn`` is injectable for CPU tests; production returns true integer
    high-error pixel counts, never alpha*T compositing mass.
    A protected run requires the current event's post-ADC/post-cleanup boolean
    mask; it is never reconstructed from gradients that ADC already reset.
    """
    xyz = gaussians.get_xyz
    count = int(xyz.shape[0])
    empty_mask = torch.zeros(count, dtype=torch.bool, device=xyz.device)
    if config.protect_egd:
        if (not isinstance(protection_mask, torch.Tensor)
                or protection_mask.dtype != torch.bool
                or protection_mask.shape != (count,)
                or protection_mask.device != xyz.device):
            raise ValueError("EGD protection requires this event's aligned post-ADC bool [N] mask")
    elif protection_mask is not None:
        raise ValueError("protection_mask is only valid with fastgs_protect_egd enabled")
    diagnostics = {
        "event": "fastgs_pruning",
        "strategy": config.strategy,
        "input_count": count,
        "evaluated_views": 0,
        "score_candidates": 0,
        "opacity_candidates": 0,
        "egd_protection_enabled": config.protect_egd,
        "egd_protected_count": int(protection_mask.sum().item()) if config.protect_egd else 0,
        "candidates_before_protection": 0,
        "blocked_by_egd": 0,
        "blocked_score_candidates": 0,
        "blocked_opacity_candidates": 0,
        "requested_after_protection": 0,
        "requested": 0,
        "prune_all_guard": False,
        "trial_renders": 0,
        "score_backend": "unweighted_high_error_pixel_count",
        "diagnostic_views": "all_distinct_training_views",
        "dropout_during_diagnostics": False,
    }

    def skip(reason):
        diagnostics["skip_reason"] = reason
        return DensityDecision(None, empty_mask.clone(), diagnostics)

    if not config.enabled:
        return skip("disabled")
    cameras = distinct_training_cameras(train_cameras)
    if not count or not cameras:
        return skip("no_points_or_training_views")
    if not torch.isfinite(xyz).all():
        return skip("nonfinite_geometry")
    opacity = gaussians.get_opacity.detach().reshape(-1)
    if opacity.shape != (count,) or opacity.device != xyz.device:
        raise ValueError("FastGS opacity must be aligned with the current Gaussian population")
    if not torch.isfinite(opacity).all():
        return skip("nonfinite_opacity")
    if render_fn is None:
        from gaussian_renderer.fastgs_pruning import render_fastgs_pruning_view
        render_fn = render_fastgs_pruning_view

    raw_score = torch.zeros(count, device=xyz.device, dtype=xyz.dtype)
    view_losses = []
    for camera in cameras:
        try:
            result = render_fn(camera, gaussians, pipe, background, loss_threshold=config.loss_threshold)
        except FloatingPointError:
            return skip("nonfinite_diagnostic")
        image = result["render"]
        gt = camera.original_image.to(device=xyz.device)
        counts = result["high_error_counts"]
        if image.shape != gt.shape or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("FastGS diagnostic expects matching RGB [3,H,W] images")
        if counts.shape != (count,) or counts.device != xyz.device:
            raise ValueError("FastGS pixel counts must be aligned with the current Gaussian population")
        if counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("FastGS requires integer pixel counts, not compositing weights")
        if (counts < 0).any():
            raise ValueError("FastGS pixel counts must be nonnegative")
        if not torch.isfinite(image).all() or not torch.isfinite(gt).all():
            return skip("nonfinite_diagnostic_image")
        # Official scoring loss. Deliberately NOT a change to train.py's loss.
        # Clamp only guards tiny negative SSIM roundoff at perfect reconstruction.
        loss = (0.8 * l1_loss(image, gt) + 0.2 * (1.0 - ssim(image, gt))).clamp_min(0)
        if not torch.isfinite(loss):
            return skip("nonfinite_diagnostic_loss")
        raw_score += loss * counts.to(raw_score.dtype)
        view_losses.append(float(loss.item()))
        diagnostics["evaluated_views"] += 1

    try:
        score, degenerate = normalized_pruning_score(raw_score)
    except FloatingPointError:
        return skip("nonfinite_aggregated_score")
    score_mask = score > config.score_threshold
    opacity_mask = opacity < config.opacity_threshold
    mask = score_mask | opacity_mask
    diagnostics.update({
        "score_candidates": int(score_mask.sum().item()),
        "opacity_candidates": int(opacity_mask.sum().item()),
        "candidate_overlap": int((score_mask & opacity_mask).sum().item()),
        "raw_score_min": float(raw_score.min().item()),
        "raw_score_max": float(raw_score.max().item()),
        "constant_score": degenerate,
        "view_loss_min": min(view_losses),
        "view_loss_max": max(view_losses),
        # The ordinary 0.005 cleanup is unchanged; report possible overlap.
        "native_opacity_overlap": int((mask & (opacity < 0.005)).sum().item()),
        "candidates_before_protection": int(mask.sum().item()),
    })
    if config.protect_egd:
        diagnostics.update({
            "blocked_by_egd": int((mask & protection_mask).sum().item()),
            "blocked_score_candidates": int((score_mask & protection_mask).sum().item()),
            "blocked_opacity_candidates": int((opacity_mask & protection_mask).sum().item()),
        })
        mask = mask & ~protection_mask
    diagnostics["requested_after_protection"] = int(mask.sum().item())
    # Apply the nonempty-model guard AFTER the EGD veto. A protected survivor
    # already satisfies this guard; do not keep an unnecessary second point.
    if mask.all():
        mask[opacity.argmax()] = False
        diagnostics["prune_all_guard"] = True
    diagnostics["requested"] = int(mask.sum().item())
    return DensityDecision(None, mask, diagnostics)


@torch.no_grad()
def apply_fastgs_pruning(gaussians, decision):
    """Use the existing Adam-aware row removal without altering ADC statistics."""
    mask = decision.prune_mask
    count = int(gaussians.get_xyz.shape[0])
    if mask is None or mask.dtype != torch.bool or mask.shape != (count,) or mask.device != gaussians.get_xyz.device:
        raise ValueError("FastGS deletion mask must match the current population")
    if count and mask.all():
        raise ValueError("FastGS deletion must preserve a nonempty representation")
    pruned = int(mask.sum().item())
    if pruned:
        # prune_points masks parameters, Adam moments, radii and accumulated
        # densification statistics together. Do not reset surviving EGD stats.
        gaussians.prune_points(mask)
    return {"pruned": pruned, "output_count": int(gaussians.get_xyz.shape[0])}
