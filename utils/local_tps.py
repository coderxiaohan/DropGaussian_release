"""Compact-support localization and sampled diagnostics for initialization warps.

This module does not change TPS fitting, Gaussian training, or density control.
The support radius at an anchor comes from neighboring *unique input points*,
not from distances to sparsely distributed anchors.  Consequently a lone anchor
cannot make its influence radius arbitrarily large merely because other anchors
are far away.  Localization limits unsupported extrapolation; it does not make a
TPS warp injective, remove erroneous correspondences, or guarantee better images.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from utils.tps_warp import _normalization, _points, _positive_integer, _readonly


def _validate_warp(warp):
    if not callable(getattr(warp, "transform", None)):
        raise ValueError("warp must provide a callable transform(points) method")


def _transformed(warp, points):
    result = _points(warp.transform(points), "transformed points")
    if result.shape != points.shape:
        raise ValueError("warp.transform must preserve the input [N, 3] shape")
    return result


@dataclass(frozen=True)
class LocalTPSWarp:
    """A supplied warp blended with identity inside compact anchor supports."""

    warp: object
    controls: np.ndarray
    radii: np.ndarray
    source_center: np.ndarray
    source_scale: float
    normalized_controls: np.ndarray
    normalized_radii: np.ndarray
    diagnostics: dict

    def _support(self, points):
        query = (points - self.source_center) / self.source_scale
        squared = np.zeros((len(query), len(self.normalized_controls)), dtype=np.float64)
        for axis in range(3):
            delta = query[:, axis, None] - self.normalized_controls[None, :, axis]
            squared += delta * delta
        distance = np.sqrt(squared) / self.normalized_radii[None, :]
        # Clipping gives an exactly zero basis outside each compact support.
        inside_distance = np.minimum(distance, 1.0)
        basis = (1.0 - inside_distance) ** 4 * (1.0 + 4.0 * inside_distance)
        return np.clip(1.0 - np.prod(1.0 - basis, axis=1), 0.0, 1.0)

    def support(self, points, chunk_size: int = 8192) -> np.ndarray:
        """Return union support weights in [0, 1], without applying the warp."""
        values = _points(points, "points")
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        result = np.empty(len(values), dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for start in range(0, len(values), chunk_size):
                    end = start + chunk_size
                    result[start:end] = self._support(values[start:end])
        except FloatingPointError as exc:
            raise ValueError("local TPS support produced nonfinite arithmetic") from exc
        if not np.isfinite(result).all():
            raise ValueError("local TPS support produced nonfinite weights")
        return result

    def transform(self, points, chunk_size: int = 8192) -> np.ndarray:
        """Transform supported points only; return exact originals elsewhere."""
        values = _points(points, "points")
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        result = np.array(values, copy=True)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for start in range(0, len(values), chunk_size):
                    end = min(start + chunk_size, len(values))
                    query = values[start:end]
                    support = self._support(query)
                    active = np.flatnonzero(support > 0)
                    if not len(active):
                        continue
                    warped = _transformed(self.warp, query[active])
                    active_support = support[active]
                    blended = np.array(warped, copy=True)
                    fractional = active_support < 1.0
                    if fractional.any():
                        original = query[active[fractional]]
                        blended[fractional] = original + active_support[fractional, None] * (
                            warped[fractional] - original
                        )
                    result[start + active] = blended
        except FloatingPointError as exc:
            raise ValueError("local TPS warp produced nonfinite coordinates") from exc
        if not np.isfinite(result).all():
            raise ValueError("local TPS warp produced nonfinite coordinates")
        return result


def localize_tps(warp, source_cloud, controls, neighbors: int = 64) -> LocalTPSWarp:
    """Localize a warp using data-scaled, compact C2 Wendland supports.

    Controls must be distinct, exact members of ``source_cloud``.  Duplicate
    input cloud locations count only once when estimating the neighborhood.
    Each radius is the distance to the k-th non-self unique point, where
    ``k=min(neighbors, number_of_unique_points-1)``.  The wrapper equals the
    original warp at every anchor, and identity outside all support balls.
    """
    _validate_warp(warp)
    points = _points(source_cloud, "source_cloud", minimum=2)
    controls = _points(controls, "controls", minimum=1)
    neighbors = _positive_integer(neighbors, "neighbors")
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 2:
        raise ValueError("local TPS support needs at least two distinct source cloud locations")
    if len(np.unique(controls, axis=0)) != len(controls):
        raise ValueError("local TPS controls must be distinct")
    center, scale, normalized = _normalization(unique_points)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized_controls = (controls - center) / scale
            tree = cKDTree(normalized)
            effective_neighbors = min(neighbors, len(unique_points) - 1)
            distances, _ = tree.query(normalized_controls, k=effective_neighbors + 1)
            if np.any(distances[:, 0] != 0):
                raise ValueError("each local TPS control must be an exact source_cloud point")
            normalized_radii = distances[:, -1]
            radii = normalized_radii * scale
    except FloatingPointError as exc:
        raise ValueError("local TPS support radius computation is not finite") from exc
    if (not np.isfinite(radii).all() or not np.isfinite(normalized_radii).all()
            or np.any(radii <= 0) or np.any(normalized_radii <= 0)):
        raise ValueError("local TPS support radii must be finite and positive")
    diagnostics = {
        "method": "compact_support_localized_tps",
        "base_fit": copy.deepcopy(getattr(warp, "diagnostics", {})),
        "control_fit_policy": "support_is_exactly_one_at_controls_base_fit_residuals_are_unchanged",
        "support_basis": "C2_Wendland_phi(r)=(1-r)^4*(1+4*r)_for_0<=r<1_else_0",
        "support_union": "g(x)=1-product_i(1-phi(norm(x-control_i)/radius_i))",
        "transform": "x+g(x)*(base_warp(x)-x)",
        "radius_rule": "kth_nonself_unique_source_cloud_neighbor",
        "neighbors_requested": neighbors,
        "neighbors_effective": effective_neighbors,
        "source_point_count": len(points),
        "unique_source_point_count": len(unique_points),
        "control_count": len(controls),
        "radius_min": float(np.min(radii)),
        "radius_median": float(np.median(radii)),
        "radius_p95": float(np.quantile(radii, 0.95)),
        "radius_max": float(np.max(radii)),
        "normalization_scale": scale,
        "outside_support_policy": "exact_identity_base_warp_not_evaluated",
        "injectivity_guaranteed": False,
    }
    localized = LocalTPSWarp(
        warp, _readonly(controls), _readonly(radii), _readonly(center), scale,
        _readonly(normalized_controls), _readonly(normalized_radii), diagnostics,
    )
    source_support = localized.support(points)
    diagnostics.update({
        "support_diagnostic_scope": "all_source_cloud_points_including_duplicate_locations",
        "support_nonzero_count": int(np.count_nonzero(source_support > 0)),
        "support_nonzero_fraction": float(np.mean(source_support > 0)),
        "support_gt_half_fraction": float(np.mean(source_support > 0.5)),
        "support_quantiles": {
            "min": float(np.min(source_support)),
            "p05": float(np.quantile(source_support, 0.05)),
            "median": float(np.median(source_support)),
            "p95": float(np.quantile(source_support, 0.95)),
            "max": float(np.max(source_support)),
        },
    })
    return localized


def _control_locations(warp):
    if isinstance(warp, LocalTPSWarp):
        return warp.controls
    controls = getattr(warp, "normalized_controls", None)
    if controls is None:
        return None
    center = getattr(warp, "source_center", None)
    scale = getattr(warp, "source_scale", None)
    if center is None or scale is None:
        return None
    return _points(np.asarray(controls) * scale + center, "warp controls")


def sampled_warp_diagnostics(warp, points, max_samples: int = 1024) -> dict:
    """Estimate displacement and local Jacobians on deterministic cloud samples.

    Uses central differences with step ``1e-5 * cloud_rms_radius``.  Locations
    within two steps of TPS controls are excluded because the ``-r`` TPS kernel
    is not differentiable exactly at a control.  Samples are evenly spaced in
    the remaining input order, not randomized.  These diagnostics can reveal
    local orientation reversals; they cannot prove absence of folds between
    samples or global injectivity.  Positive determinants alone are insufficient.
    """
    _validate_warp(warp)
    values = _points(points, "points", minimum=2)
    max_samples = _positive_integer(max_samples, "max_samples")
    _, radius, _ = _normalization(values)
    step = radius * 1e-5
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Jacobian finite-difference step must be finite and positive")
    candidates = np.arange(len(values), dtype=np.int64)
    controls = _control_locations(warp)
    if controls is not None and len(controls):
        distances, _ = cKDTree(controls).query(values, k=1)
        candidates = candidates[distances > 2 * step]
    excluded = len(values) - len(candidates)
    if not len(candidates):
        return {
            "method": "deterministic_linspace_central_finite_difference",
            "status": "unavailable_no_noncontrol_samples",
            "scope": "sampled_local_diagnostic_not_a_global_injectivity_or_quality_guarantee",
            "source_point_count": len(values),
            "sample_count": 0,
            "max_samples": max_samples,
            "excluded_near_control_count": excluded,
            "step": step,
            "relative_rms_step": 1e-5,
        }
    selected = candidates[np.linspace(0, len(candidates) - 1, min(max_samples, len(candidates)), dtype=np.int64)]
    query = values[selected]
    jacobian = np.empty((len(query), 3, 3), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            transformed = _transformed(warp, query)
            for axis in range(3):
                plus, minus = query.copy(), query.copy()
                plus[:, axis] += step
                minus[:, axis] -= step
                if np.any(plus[:, axis] == query[:, axis]) or np.any(minus[:, axis] == query[:, axis]):
                    raise ValueError("Jacobian step is below input coordinate precision")
                jacobian[:, :, axis] = (_transformed(warp, plus) - _transformed(warp, minus)) / (2 * step)
            determinant = np.linalg.det(jacobian)
            singular = np.linalg.svd(jacobian, compute_uv=False)
            displacement = np.linalg.norm(transformed - query, axis=1)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("sampled warp Jacobian diagnostic is not numerically finite") from exc
    if not all(np.isfinite(array).all() for array in (jacobian, determinant, singular, displacement)):
        raise ValueError("sampled warp Jacobian diagnostic produced nonfinite values")
    active = warp.support(query) > 0 if isinstance(warp, LocalTPSWarp) else np.ones(len(query), dtype=bool)
    active_count = int(np.count_nonzero(active))
    active_nonpositive_count = int(np.count_nonzero(determinant[active] <= 0))
    return {
        "method": "deterministic_linspace_central_finite_difference",
        "status": "available",
        "scope": "sampled_local_diagnostic_not_a_global_injectivity_or_quality_guarantee",
        "source_point_count": len(values),
        "sample_count": len(query),
        "max_samples": max_samples,
        "excluded_near_control_count": excluded,
        "step": step,
        "relative_rms_step": 1e-5,
        "active_support_policy": "positive_local_support" if isinstance(warp, LocalTPSWarp) else "all_samples_for_global_warp",
        "active_sample_count": active_count,
        "active_sample_fraction": float(np.mean(active)),
        "nonpositive_count_among_active_support": active_nonpositive_count,
        "nonpositive_fraction_among_active_support": (
            active_nonpositive_count / active_count if active_count else None
        ),
        "minimum_determinant_among_active_support": float(np.min(determinant[active])) if active_count else None,
        "determinant_nonpositive_count": int(np.count_nonzero(determinant <= 0)),
        "determinant_nonpositive_fraction": float(np.mean(determinant <= 0)),
        "determinant_min": float(np.min(determinant)),
        "determinant_p05": float(np.quantile(determinant, 0.05)),
        "determinant_median": float(np.median(determinant)),
        "determinant_p95": float(np.quantile(determinant, 0.95)),
        "determinant_max": float(np.max(determinant)),
        "smallest_singular_value_min": float(np.min(singular[:, -1])),
        "smallest_singular_value_median": float(np.median(singular[:, -1])),
        "largest_singular_value_median": float(np.median(singular[:, 0])),
        "largest_singular_value_p95": float(np.quantile(singular[:, 0], 0.95)),
        "largest_singular_value_max": float(np.max(singular[:, 0])),
        "sample_displacement_median": float(np.median(displacement)),
        "sample_displacement_p95": float(np.quantile(displacement, 0.95)),
        "sample_displacement_max": float(np.max(displacement)),
    }
