"""Initialization-only local covariance transfer from a fixed MVS point cloud.

This is not a position correction or a learned MVS-confidence estimator. Local
PCA supplies shape/orientation; the original 3-NN Gaussian scale supplies total
variance. The p=3 Oracle Approximating Shrinkage (OAS) formula regularizes noisy
finite neighborhoods. Its Gaussian/iid optimality assumptions are not claimed
for spatial KNN neighborhoods.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _points_array(points):
    array = np.asarray(points)
    if (array.ndim != 2 or array.shape[1] != 3 or len(array) == 0
            or array.dtype.kind not in "fiu"):
        raise ValueError("points must be a nonempty real numeric (N, 3) array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    return array


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _quantiles(array):
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(array, (0., .05, .5, .95, 1.)).tolist()))


def estimate_surface_frames(points, neighbors=32, chunk_size=4096):
    """Return dimensionless axis scales, wxyz rotations, and JSON metadata.

    Exact duplicate positions are removed ONLY from neighborhood estimation;
    the returned arrays retain the original point count/order. Each neighborhood
    includes its query point and contains min(neighbors, unique_count) distinct
    positions. KNN is deterministic for repeated identical inputs; tied-distance
    neighborhoods need not be rotation-equivariant because their membership is
    not geometrically unique. All computation is CPU, with no random draws.

    For centered covariance C and n neighbors, use the original OAS Eq. (23):
      rho = min(1, ((1-2/3) tr(C^2) + tr(C)^2) /
                   ((n+1-2/3) (tr(C^2) - tr(C)^2/3))).
    The high-dimensional approximation omitting 2/p is NOT used. The regularized
    covariance is normalized to trace 3. This preserves the original total
    variance after multiplication by the original isotropic scale squared, not
    necessarily its projected image footprint or opacity integral.
    """
    started = time.perf_counter()
    points = _points_array(points)
    neighbors = _positive_integer(neighbors, "neighbors")
    if neighbors < 3:
        raise ValueError("neighbors must be an integer of at least 3")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    count = len(unique)
    effective_neighbors = min(neighbors, count)
    factors = np.ones((count, 3), dtype=np.float64)
    rotations = np.zeros((count, 4), dtype=np.float64)
    rotations[:, 0] = 1.
    shrinkage = np.ones(count, dtype=np.float64)
    zero_covariance = np.zeros(count, dtype=bool)

    # Normalize the search coordinates for scene-unit invariance and to avoid
    # squared-distance under/overflow. Actual Gaussian positions are untouched.
    with np.errstate(over="ignore", invalid="ignore"):
        search_points = unique - unique[0]
    if not np.isfinite(search_points).all():
        raise ValueError("point coordinate range cannot be represented safely")
    coordinate_scale = float(np.max(np.abs(search_points)))
    if coordinate_scale > 0:
        search_points = search_points / coordinate_scale
    tree = cKDTree(search_points)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        query = search_points[start:stop]
        _, indices = tree.query(query, k=effective_neighbors, workers=1)
        if effective_neighbors == 1:
            indices = indices[:, None]
        offsets = search_points[indices] - query[:, None, :]
        centered = offsets - offsets.mean(axis=1, keepdims=True)
        local_scale = np.max(np.abs(centered), axis=(1, 2))
        valid = local_scale > 0
        zero_covariance[start:stop] = ~valid
        if not np.any(valid):
            continue
        centered = centered[valid] / local_scale[valid, None, None]
        covariance = np.einsum("nki,nkj->nij", centered, centered) / effective_neighbors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # PSD roundoff can produce tiny negative eigenvalues in exact planes.
        eigenvalues = np.maximum(eigenvalues, 0.)
        trace = eigenvalues.sum(axis=1)
        eigenvalues = eigenvalues / trace[:, None]
        trace_square = np.square(eigenvalues).sum(axis=1)
        denominator = (effective_neighbors + 1. - 2. / 3.) * (trace_square - 1. / 3.)
        numerator = (1. - 2. / 3.) * trace_square + 1.
        rho = np.ones(len(trace), dtype=np.float64)
        np.divide(numerator, denominator, out=rho, where=denominator > 0)
        rho = np.clip(rho, 0., 1.)
        variances = 3. * (1. - rho[:, None]) * eigenvalues + rho[:, None]
        # Remove only floating-point trace drift, not any geometric feature.
        variances *= 3. / variances.sum(axis=1, keepdims=True)
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("surface covariance estimation produced a nonpositive variance")

        # Columns are world-space principal axes: Sigma = R diag(s^2) R^T.
        # Flip one axis for a proper rotation; its sign does not change Sigma.
        negative = np.linalg.det(eigenvectors) < 0
        eigenvectors[negative, :, 2] *= -1.
        quaternion_xyzw = Rotation.from_matrix(eigenvectors).as_quat()
        quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
        quaternion_wxyz[quaternion_wxyz[:, 0] < 0] *= -1.
        isotropic = rho == 1.
        quaternion_wxyz[isotropic] = (1., 0., 0., 0.)
        locations = start + np.flatnonzero(valid)
        factors[locations] = np.sqrt(variances)
        rotations[locations] = quaternion_wxyz
        shrinkage[locations] = rho

    factors, rotations = factors[inverse], rotations[inverse]
    shrinkage = shrinkage[inverse]
    zero_covariance = zero_covariance[inverse]
    metadata = {
        "method": "mvs_surface_covariance_oas_v1",
        "point_count": int(len(points)),
        "unique_support_point_count": int(count),
        "duplicate_position_count": int(len(points) - count),
        "requested_neighbors": neighbors,
        "effective_neighbors": effective_neighbors,
        "neighborhood": "exact_unique_xyz_knn_including_self",
        "duplicate_policy": "deduplicate_estimation_support_only_preserve_all_gaussians",
        "chunk_size": chunk_size,
        "covariance": "centered_population_covariance",
        "shrinkage": "original_p3_OAS_equation_23_including_2_over_p",
        "reference": "https://arxiv.org/abs/0907.4698",
        "normalization": "trace_3_times_original_isotropic_variance",
        "rotation_convention": "wxyz_columns_are_world_space_principal_axes",
        "zero_covariance_count": int(zero_covariance.sum()),
        "isotropic_count": int(np.count_nonzero(shrinkage == 1.)),
        "shrinkage_coefficient": _quantiles(shrinkage),
        "scale_factor": _quantiles(factors),
        "scale_factor_per_axis_median": np.median(factors, axis=0).tolist(),
        "covariance_condition_number": _quantiles(np.square(factors[:, 2] / factors[:, 0])),
        "max_dimensionless_trace_error": float(np.max(np.abs(np.square(factors).sum(axis=1) - 3.))),
        "points_sha256": _digest(points),
        "scale_factors_sha256": _digest(factors),
        "rotations_sha256": _digest(rotations),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return factors, rotations, metadata


def initialize_surface_covariance(gaussians, points, neighbors=32):
    """Transfer local MVS covariance after create_from_pcd, before training.

    Parameter objects are retained, not replaced. XYZ, SH/color, opacity, point
    count, and the subsequent optimizer/training/density-control code are not
    modified. Reject non-isotropic base scales or an already-created optimizer
    to prevent accidentally changing a trained model.
    """
    import torch

    started = time.perf_counter()
    points = _points_array(points)
    if not hasattr(gaussians, "optimizer") or gaussians.optimizer is not None:
        raise ValueError("surface initialization requires optimizer=None before training_setup")
    names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    attributes = {}
    for name in names:
        parameter = getattr(gaussians, name, None)
        if not isinstance(parameter, torch.nn.Parameter) or not parameter.is_floating_point():
            raise ValueError(f"{name} must be an initialized floating nn.Parameter")
        array = parameter.detach().cpu().numpy().copy()
        if array.shape[0] != len(points) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite and have the same point count")
        attributes[name] = array
    if attributes["_xyz"].shape != points.shape or gaussians._xyz.dtype != torch.float32:
        raise ValueError("Gaussian XYZ must have shape (N, 3) and dtype float32")
    with np.errstate(over="ignore", invalid="ignore"):
        points_float32 = points.astype(np.float32)
    if not np.isfinite(points_float32).all() or not np.array_equal(attributes["_xyz"], points_float32):
        raise ValueError("points must exactly match initialized Gaussian XYZ in float32")
    if attributes["_scaling"].shape != (len(points), 3) or attributes["_rotation"].shape != (len(points), 4):
        raise ValueError("Gaussian scaling/rotation shapes must be (N, 3)/(N, 4)")
    if not np.all(attributes["_scaling"] == attributes["_scaling"][:, :1]):
        raise ValueError("surface initialization requires original isotropic base scales")
    if gaussians._scaling.device != gaussians._rotation.device:
        raise ValueError("Gaussian scaling and rotation must be on the same device")

    # Use the actual float32 model coordinates, rather than any extra precision
    # in a caller's array that create_from_pcd has already rounded away.
    factors, rotations, metadata = estimate_surface_frames(points_float32, neighbors=neighbors)
    with torch.no_grad():
        increment = torch.as_tensor(np.log(factors), device=gaussians._scaling.device,
                                    dtype=gaussians._scaling.dtype)
        new_logs = gaussians._scaling.detach() + increment
        new_rotations = torch.as_tensor(rotations, device=gaussians._rotation.device,
                                        dtype=gaussians._rotation.dtype)
        base_scales = torch.exp(gaussians._scaling.detach())
        initialized_scales = torch.exp(new_logs)
        if (not torch.isfinite(base_scales).all() or not torch.isfinite(initialized_scales).all()
                or not torch.all(base_scales > 0) or not torch.all(initialized_scales > 0)):
            raise ValueError("Gaussian scales must remain finite and positive")
        base_scales = base_scales.cpu().numpy().copy()
        initialized_scales = initialized_scales.cpu().numpy().copy()
        gaussians._scaling.copy_(new_logs)
        gaussians._rotation.copy_(new_rotations)

    before_hashes = {name: _digest(array) for name, array in attributes.items()}
    after_hashes = {name: _digest(getattr(gaussians, name).detach().cpu().numpy()) for name in names}
    base_trace = np.square(base_scales.astype(np.float64)).sum(axis=1)
    initialized_trace = np.square(initialized_scales.astype(np.float64)).sum(axis=1)
    metadata.update({
        "application": "after_create_from_pcd_before_training_setup",
        "preserved_attributes": ["_xyz", "_features_dc", "_features_rest", "_opacity"],
        "changed_attributes": ["_scaling", "_rotation"],
        "parameter_objects_preserved": True,
        "attributes_before_sha256": before_hashes,
        "attributes_after_sha256": after_hashes,
        "base_scale": _quantiles(base_scales),
        "initialized_scale": _quantiles(initialized_scales),
        "max_relative_covariance_trace_error": float(np.max(np.abs(initialized_trace / base_trace - 1.))),
        "total_elapsed_seconds": float(time.perf_counter() - started),
    })
    arrays = {
        "scale_factors": factors,
        "rotations": gaussians._rotation.detach().cpu().numpy().copy(),
        "base_scales": base_scales,
        "initialized_scales": initialized_scales,
    }
    return metadata, arrays
