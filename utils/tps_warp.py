"""Small, deterministic geometric warps for MVS initialization experiments.

This is an independent implementation of similarity alignment and a 3D
polyharmonic (often called thin-plate) spline, not the TWINGS training pipeline.
The spline uses ``phi(r) = -r`` with an affine tail.  This sign convention makes
the bending kernel positive on the affine-nullspace, so positive diagonal
regularization has the usual smoothing interpretation.  It is NOT the 2D
``r**2 log(r)`` kernel.

All fitting happens in float64, with scene coordinates normalized by the source
RMS radius.  Smoothing is consequently dimensionless and unchanged under scene
translation, rotation, or uniform scaling.  No geometry is rounded or silently
replaced with the uncorrected input when fitting fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _points(value, name: str, *, minimum: int = 0) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real coordinates")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric [N, 3] array") from exc
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if len(result) < minimum:
        raise ValueError(f"{name} needs at least {minimum} points")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return result


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normalization(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            center = points.mean(axis=0)
            centered = points - center
            # Scale before squaring to avoid overflow for large scene units.
            extent = float(np.max(np.abs(centered)))
            if extent == 0:
                raise ValueError("source controls are degenerate: zero spatial extent")
            radius = extent * float(np.sqrt(np.mean(np.sum((centered / extent) ** 2, axis=1))))
            normalized = centered / radius
    except FloatingPointError as exc:
        raise ValueError("source normalization is not numerically finite") from exc
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("source normalization scale must be finite and positive")
    return center, radius, normalized


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Coordinate-wise subtraction avoids cancellation of nearby controls that
    # can occur in ||x||^2 + ||y||^2 - 2 x.y, without allocating an [M,N,3] tensor.
    squared = np.zeros((len(left), len(right)), dtype=np.float64)
    for axis in range(3):
        delta = left[:, axis, None] - right[None, :, axis]
        squared += delta * delta
    return -np.sqrt(squared)


def _residual_diagnostics(source, target, fitted) -> dict:
    before = np.linalg.norm(source - target, axis=1)
    after = np.linalg.norm(fitted - target, axis=1)
    return {
        "finite": bool(np.all(np.isfinite(fitted))),
        "control_rmse_before": float(np.sqrt(np.mean(before ** 2))),
        "control_rmse_after": float(np.sqrt(np.mean(after ** 2))),
        "control_max_residual_after": float(np.max(after)),
    }


@dataclass(frozen=True)
class SimilarityWarp:
    """Proper-rotation similarity transform, with row-vector input points."""

    source_center: np.ndarray
    target_center: np.ndarray
    rotation: np.ndarray
    scale: float
    diagnostics: dict

    def transform(self, points, chunk_size: int = 8192) -> np.ndarray:
        values = _points(points, "points")
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        result = np.empty_like(values)
        try:
            with np.errstate(over="raise", invalid="raise"):
                for start in range(0, len(values), chunk_size):
                    end = start + chunk_size
                    result[start:end] = (
                        self.scale * ((values[start:end] - self.source_center) @ self.rotation.T)
                        + self.target_center
                    )
        except FloatingPointError as exc:
            raise ValueError("similarity warp produced nonfinite coordinates") from exc
        if not np.all(np.isfinite(result)):
            raise ValueError("similarity warp produced nonfinite coordinates")
        return result


@dataclass(frozen=True)
class TPSWarp:
    """Normalized 3D spline displacement with an affine tail."""

    source_center: np.ndarray
    source_scale: float
    normalized_controls: np.ndarray
    weights: np.ndarray
    affine: np.ndarray
    diagnostics: dict

    def transform(self, points, chunk_size: int = 8192) -> np.ndarray:
        values = _points(points, "points")
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        result = np.empty_like(values)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for start in range(0, len(values), chunk_size):
                    end = start + chunk_size
                    query = (values[start:end] - self.source_center) / self.source_scale
                    polynomial = np.column_stack((np.ones(len(query)), query))
                    displacement = (
                        _kernel(query, self.normalized_controls) @ self.weights
                        + polynomial @ self.affine
                    )
                    result[start:end] = values[start:end] + self.source_scale * displacement
        except FloatingPointError as exc:
            raise ValueError("TPS warp produced nonfinite coordinates") from exc
        if not np.all(np.isfinite(result)):
            raise ValueError("TPS warp produced nonfinite coordinates")
        return result


def fit_similarity(source, target) -> SimilarityWarp:
    """Least-squares positive-scale, proper-rotation alignment (Umeyama).

    At least three non-collinear source and target correspondences are needed.
    Reflection is not allowed.  Fitting is deterministic and never mutates the
    supplied arrays or the global random state.
    """

    source = _points(source, "source", minimum=3)
    target = _points(target, "target", minimum=3)
    if source.shape != target.shape:
        raise ValueError("source and target must have matching shapes")
    center, radius, normalized = _normalization(source)
    target_center, _, _ = _normalization(target)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        target_normalized = (target - target_center) / radius
    if np.linalg.matrix_rank(normalized) < 2 or np.linalg.matrix_rank(target_normalized) < 2:
        raise ValueError("similarity controls are degenerate: non-collinear correspondences required")
    try:
        left, singular, right = np.linalg.svd(target_normalized.T @ normalized / len(source))
    except np.linalg.LinAlgError as exc:
        raise ValueError("similarity alignment SVD did not converge") from exc
    signs = np.ones(3)
    if np.linalg.det(left @ right) < 0:
        signs[-1] = -1
    rotation = (left * signs) @ right
    scale = float(np.sum(singular * signs) / np.mean(np.sum(normalized ** 2, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("similarity alignment needs a finite positive scale")
    diagnostics = {
        "method": "similarity",
        "control_count": int(len(source)),
        "normalization_center": center.tolist(),
        "normalization_scale": radius,
        "fitted_scale": scale,
        "rotation_determinant": float(np.linalg.det(rotation)),
    }
    warp = SimilarityWarp(_readonly(center), _readonly(target_center), _readonly(rotation), scale, diagnostics)
    diagnostics.update(_residual_diagnostics(source, target, warp.transform(source)))
    return warp


def fit_tps(source, target, smoothing: float = 1e-3) -> TPSWarp:
    """Fit a regularized 3D spline to paired source/target control points.

    Solve ``[K + lambda I, P; P.T, 0] [W; A] = [D; 0]``, where ``K=-r``,
    ``P=[1,x,y,z]`` and ``D=(target-source)/source_rms_radius``.  The affine tail
    is unpenalized, preserving exact affine maps even when smoothing is positive.
    Lambda zero interpolates distinct controls.  Positive lambda trades fitting
    accuracy for smaller non-affine bending; it is not a confidence threshold.

    Source controls must contain at least four distinct non-coplanar points.
    Callers should cap controls before fitting: solve memory is O(N^2), time
    O(N^3); transform memory is O(chunk_size * N).
    """

    source = _points(source, "source", minimum=4)
    target = _points(target, "target", minimum=4)
    if source.shape != target.shape:
        raise ValueError("source and target must have matching shapes")
    if isinstance(smoothing, bool) or not isinstance(smoothing, Real) or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be a finite non-negative number")
    smoothing = float(smoothing)
    if len(np.unique(source, axis=0)) != len(source):
        raise ValueError("TPS source controls are degenerate: duplicate coordinates")
    center, radius, normalized = _normalization(source)
    polynomial = np.column_stack((np.ones(len(source)), normalized))
    affine_rank = int(np.linalg.matrix_rank(polynomial))
    if affine_rank != 4:
        raise ValueError("TPS source controls are degenerate: affine rank must be 4 (non-coplanar)")
    count = len(source)
    kernel = _kernel(normalized, normalized)
    kernel.flat[::count + 1] += smoothing
    system = np.zeros((count + 4, count + 4), dtype=np.float64)
    system[:count, :count] = kernel
    system[:count, count:] = polynomial
    system[count:, :count] = polynomial.T
    rhs = np.zeros((count + 4, 3), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            rhs[:count] = (target - source) / radius
        coefficients = np.linalg.solve(system, rhs)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError("TPS controls do not admit a finite stable solve") from exc
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("TPS solve produced nonfinite coefficients")
    solve_residual = np.linalg.norm(system @ coefficients - rhs)
    rhs_norm = np.linalg.norm(rhs)
    diagnostics = {
        "method": "tps",
        "kernel": "3d_polyharmonic_negative_r",
        "smoothing": smoothing,
        "control_count": count,
        "affine_rank": affine_rank,
        "normalization_center": center.tolist(),
        "normalization_scale": radius,
        "linear_system_relative_residual": float(solve_residual / max(rhs_norm, np.finfo(np.float64).tiny)),
        "affine_constraint_residual": float(np.linalg.norm(polynomial.T @ coefficients[:count])),
    }
    warp = TPSWarp(
        _readonly(center), radius, _readonly(normalized), _readonly(coefficients[:count]),
        _readonly(coefficients[count:]), diagnostics,
    )
    diagnostics.update(_residual_diagnostics(source, target, warp.transform(source)))
    return warp


def select_control_indices(points, max_controls: int) -> np.ndarray:
    """Select spatially spread, unique controls without random sampling.

    Exact duplicate locations are represented by their first input index.  The
    first selected point is farthest from the centroid; subsequent points are
    farthest from the selected set.  Ties use input order.  Memory is O(N), time
    O(N * max_controls), and no dense N-by-N distance array is constructed.
    """

    points = _points(points, "points", minimum=1)
    max_controls = _positive_integer(max_controls, "max_controls")
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    unique_indices.sort()
    if len(unique_indices) <= max_controls:
        return unique_indices.astype(np.int64, copy=False)
    _, _, normalized = _normalization(points[unique_indices])
    chosen = np.empty(max_controls, dtype=np.int64)
    closest_squared = np.full(len(normalized), np.inf)
    current = int(np.argmax(np.sum(normalized ** 2, axis=1)))
    for step in range(max_controls):
        chosen[step] = current
        delta = normalized - normalized[current]
        np.minimum(closest_squared, np.sum(delta * delta, axis=1), out=closest_squared)
        closest_squared[chosen[:step + 1]] = -np.inf
        current = int(np.argmax(closest_squared))
    return unique_indices[chosen].astype(np.int64, copy=False)
