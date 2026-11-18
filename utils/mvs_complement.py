"""Optional, train-view-driven selection of MVS points to complement a base cloud.

This module selects existing points; it never moves points, changes their RGB,
or initializes Gaussian shapes/opacities.  A reconstruction-backed base cloud is
an explicit caller requirement, not something inferred from a dataset name.
Its reconstruction provenance is *not* verified by this helper.  Only the new
coverage computation is restricted to the supplied training cameras/depths.

``coverage`` and ``balanced`` use the same source quotas and geometric support
test.  The former samples by uncovered image-space area; the latter samples
uniformly from the same supported, positive-novelty candidate pool.  ``concat``
is a deliberately unfiltered source-mixture control.  A positive total budget
subsamples the base cloud, so only total_budget=0 preserves the complete base.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from utils.graphics_utils import BasicPointCloud
from utils.train_view_controls import camera_geometry


_CHUNK_SIZE = 131_072


def _integer(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < (0 if allow_zero else 1)):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer.")
    return int(value)


def _cloud_arrays(pcd, name):
    if pcd is None:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    arrays = tuple(np.asarray(getattr(pcd, key)) for key in ("points", "colors", "normals"))
    if any(array.ndim != 2 or array.shape[1] != 3 for array in arrays):
        raise ValueError(f"{name} points, colors and normals must have Nx3 shapes.")
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError(f"{name} attributes have inconsistent point counts.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{name} contains non-finite point attributes.")
    return arrays


def _uniform_indices(size, count):
    if count >= size:
        return np.arange(size, dtype=np.int64)
    return np.sort(np.random.default_rng(0).choice(size, count, replace=False))


def _quotas(base_count, candidate_count, mvs_budget, total_budget):
    candidate_limit = min(candidate_count, mvs_budget)
    if total_budget == 0:
        return base_count, candidate_limit
    # The odd extra slot belongs to MVS.  Missing source capacity is borrowed
    # before geometry filtering; invalid MVS candidates are never backfilled.
    base_quota = min(base_count, total_budget // 2)
    mvs_quota = min(candidate_limit, total_budget - total_budget // 2)
    free = total_budget - base_quota - mvs_quota
    extra_base = min(free, base_count - base_quota)
    base_quota += extra_base
    free -= extra_base
    mvs_quota += min(free, candidate_limit - mvs_quota)
    return base_quota, mvs_quota


def _training_depths(path, cameras):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("coverage/balanced require an explicit mvs_depths_masks.npz cache.")
    names = [str(getattr(camera, "image_name", "")) for camera in cameras]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Training camera image_names must be nonempty and unique.")
    with np.load(path, allow_pickle=False) as cache:
        required = {"image_names", "depths", "masks"}
        if not required.issubset(cache.files):
            raise ValueError("Depth cache requires image_names, depths and masks.")
        cached_names = np.asarray(cache["image_names"])
        if cached_names.ndim != 1 or [str(name) for name in cached_names] != names:
            raise ValueError("Depth cache image_names must exactly match the ordered training cameras.")
        depths = np.asarray(cache["depths"])
        masks = np.asarray(cache["masks"])
    if (depths.ndim != 3 or depths.shape[0] != len(names)
            or min(depths.shape[1:]) <= 0 or masks.shape != depths.shape):
        raise ValueError("Depths and masks must have equal, nonempty VxHxW shapes.")
    if not np.issubdtype(depths.dtype, np.number) or np.iscomplexobj(depths):
        raise ValueError("Depths must be real numeric arrays.")
    if not (np.issubdtype(masks.dtype, np.number) or masks.dtype == np.bool_) or np.iscomplexobj(masks):
        raise ValueError("Masks must be boolean or real numeric arrays.")
    valid = np.isfinite(depths) & (depths > 0) & np.isfinite(masks) & (masks > 0)
    return depths, valid, names


def _depth_geometry(camera, width, height):
    intrinsics, extrinsics = camera_geometry(camera)
    image = camera.image
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        image_width, image_height = image.size
    else:
        image_height, image_width = np.asarray(image).shape[:2]
    intrinsics = intrinsics.copy()
    intrinsics[0] *= width / image_width
    intrinsics[1] *= height / image_height
    return intrinsics, extrinsics


def _project_support(points, intrinsics, extrinsics, depth, valid, depth_scale):
    """Project one batch and return uv plus a bounded camera-z agreement weight.

    The compact biweight goes continuously to zero at depth_scale relative
    error.  depth_scale is a user-visible geometric tolerance, NOT a calibrated
    probability.  An in-frame projection alone does not establish visibility.
    """
    camera_points = np.asarray(points, dtype=np.float64) @ extrinsics[:, :3].T + extrinsics[:, 3]
    z = camera_points[:, 2]
    pixels = camera_points @ intrinsics.T
    uv = np.zeros((len(points), 2), dtype=np.float64)
    front = np.isfinite(camera_points).all(axis=1) & (z > 0)
    uv[front] = pixels[front, :2] / pixels[front, 2:3]
    height, width = depth.shape
    inside = (front & np.isfinite(uv).all(axis=1)
              & (uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    indices = np.flatnonzero(inside)
    weight = np.zeros(len(points), dtype=np.float64)
    if len(indices):
        # Cache pixel coordinates follow its K scaling convention.  Flooring
        # assigns a continuous projected location to the corresponding cell.
        xy = np.floor(uv[indices]).astype(np.int64)
        observed = depth[xy[:, 1], xy[:, 0]]
        usable = valid[xy[:, 1], xy[:, 0]]
        indices, observed = indices[usable], observed[usable]
        residual = np.abs(z[indices] - observed) / observed
        normalized = residual / depth_scale
        weight[indices] = np.maximum(1.0 - normalized * normalized, 0.0) ** 2
    return uv, weight


def _candidate_scores(base_points, candidates, cameras, depths, valid, coverage_pixels, depth_scale):
    count = len(candidates)
    support = np.zeros(count, dtype=np.int32)
    score_sum = np.zeros(count, dtype=np.float64)
    view_records = []
    height, width = depths.shape[1:]
    for view, camera in enumerate(cameras):
        intrinsics, extrinsics = _depth_geometry(camera, width, height)
        supported_base = []
        for start in range(0, len(base_points), _CHUNK_SIZE):
            uv, weights = _project_support(
                base_points[start:start + _CHUNK_SIZE], intrinsics, extrinsics,
                depths[view], valid[view], depth_scale,
            )
            supported_base.append(uv[weights > 0])
        projected_base = (np.concatenate(supported_base) if supported_base
                          else np.empty((0, 2), dtype=np.float64))
        tree = cKDTree(projected_base) if len(projected_base) else None
        view_supported = 0
        for start in range(0, count, _CHUNK_SIZE):
            uv, weights = _project_support(
                candidates[start:start + _CHUNK_SIZE], intrinsics, extrinsics,
                depths[view], valid[view], depth_scale,
            )
            local = np.flatnonzero(weights > 0)
            if not len(local):
                continue
            novelty = np.ones(len(local), dtype=np.float64)
            if tree is not None:
                distance, _ = tree.query(uv[local], k=1)
                novelty = -np.expm1(-0.5 * (distance / coverage_pixels) ** 2)
            indices = start + local
            score_sum[indices] += weights[local] * novelty
            support[indices] += 1
            view_supported += len(local)
        view_records.append({
            "image_name": str(camera.image_name),
            "supported_base_projection_count": int(len(projected_base)),
            "supported_mvs_projection_count": int(view_supported),
        })
    # Average over supporting views, not a raw sum that grows with view count.
    scores = np.divide(score_sum, support, out=np.zeros_like(score_sum), where=support > 0)
    scores[support < 2] = 0.0
    return scores, support, view_records


def build_complement_cloud(
    base_pcd, mvs_pcd, train_cameras, *, mode="coverage", mvs_budget=100_000,
    total_budget=0, depth_cache_path=None, base_is_geometry=True,
    coverage_pixels=2.0, depth_relative_scale=0.02, direct_pcd=None,
):
    """Return ``(selected BasicPointCloud, JSON metadata, diagnostic arrays)``.

    ``direct_pcd`` must be the exact original-loader output when the base is
    random, absent or otherwise not reconstruction geometry.  That fallback is
    returned untouched, without depth-cache access or a second sampling step.
    The caller must load it with the intended direct-MVS point budget.

    Positive total_budget establishes a cap, not a promise to fill invalid
    slots.  Geometry-rejected candidates never get inserted to meet a count.
    For strict same-count comparisons, inspect the returned source counts and
    budget_shortfall rather than assuming the requested cap was achieved.
    """
    if mode not in {"coverage", "balanced", "concat"}:
        raise ValueError("mode must be coverage, balanced, or concat.")
    mvs_budget = _integer(mvs_budget, "mvs_budget")
    total_budget = _integer(total_budget, "total_budget", allow_zero=True)
    if not isinstance(base_is_geometry, (bool, np.bool_)):
        raise ValueError("base_is_geometry must be a boolean provenance decision.")
    for value, name in ((coverage_pixels, "coverage_pixels"), (depth_relative_scale, "depth_relative_scale")):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    base = _cloud_arrays(base_pcd, "base_pcd")
    mvs = _cloud_arrays(mvs_pcd, "mvs_pcd")
    cameras = list(train_cameras)
    metadata = {
        "method": "training_view_coverage_complement", "mode": mode,
        "base_source_count": int(len(base[0])), "mvs_candidate_count": int(len(mvs[0])),
        "mvs_budget": mvs_budget, "total_budget": total_budget,
        "base_is_geometry": bool(base_is_geometry),
        "base_reconstruction_camera_provenance": "not_verified_by_complement_helper",
        "depth_cache_calibration_provenance": "image_names_checked_but_legacy_cache_has_no_pose_or_intrinsic_hash",
        "coverage_pixels": float(coverage_pixels), "depth_relative_scale": float(depth_relative_scale),
        "depth_weight": "compact_biweight_of_relative_camera_z_residual",
        "minimum_supported_training_views": 2,
        "score_normalization": "mean_over_supported_views",
        "candidate_pool_policy": "two_view_depth_support_and_positive_coverage_novelty",
        "new_observations": "supplied_training_cameras_and_matching_depth_cache_only",
        "point_coordinates_or_colors_modified": False,
        "gaussian_attributes_modified": False,
        "random_generator": "local_numpy_default_rng_seed0",
    }
    if not base_is_geometry or len(base[0]) == 0:
        if direct_pcd is None:
            raise ValueError("No reliable base geometry: provide exact direct_pcd for an unchanged direct-MVS fallback.")
        direct = _cloud_arrays(direct_pcd, "direct_pcd")
        expected_cap = min(mvs_budget, total_budget) if total_budget else mvs_budget
        if len(direct[0]) == 0 or len(direct[0]) > expected_cap:
            raise ValueError("direct_pcd must be nonempty and already selected within the requested direct-MVS budget.")
        metadata.update({
            "fallback": "no_reconstruction_geometry_direct_mvs_unchanged",
            "effective_mode": "direct_mvs",
            "selected_base_count": 0, "selected_mvs_count": int(len(direct[0])),
            "output_point_count": int(len(direct[0])), "complete_base_preserved": False,
            "depth_cache_checked": False, "mvs_index_space": "exact_direct_pcd",
            "base_quota": 0, "mvs_quota": expected_cap,
            "budget_shortfall": max(0, expected_cap - len(direct[0])),
            "candidate_pool_policy": "original_direct_loader_selection",
        })
        return direct_pcd, metadata, {
            "selected_base_indices": np.empty(0, dtype=np.int64),
            "selected_mvs_indices": np.arange(len(direct[0]), dtype=np.int64),
            "candidate_scores": np.empty(0, dtype=np.float64),
            "candidate_support_views": np.empty(0, dtype=np.int32),
        }

    base_quota, mvs_quota = _quotas(len(base[0]), len(mvs[0]), mvs_budget, total_budget)
    base_indices = _uniform_indices(len(base[0]), base_quota)
    scores = np.zeros(len(mvs[0]), dtype=np.float64)
    support = np.zeros(len(mvs[0]), dtype=np.int32)
    if mode == "concat":
        eligible = np.arange(len(mvs[0]), dtype=np.int64)
        selected_mvs = _uniform_indices(len(mvs[0]), mvs_quota)
        metadata.update({"depth_cache_checked": False, "candidate_pool_policy": "unfiltered", "views": []})
    else:
        depths, valid, names = _training_depths(depth_cache_path, cameras)
        scores, support, views = _candidate_scores(
            base[0][base_indices], mvs[0], cameras, depths, valid,
            float(coverage_pixels), float(depth_relative_scale),
        )
        eligible = np.flatnonzero(scores > 0)
        selected_count = min(mvs_quota, len(eligible))
        if mode == "balanced":
            selected_mvs = eligible[_uniform_indices(len(eligible), selected_count)]
        elif selected_count == len(eligible):
            selected_mvs = eligible
        elif selected_count == 0:
            selected_mvs = np.empty(0, dtype=np.int64)
        else:
            # Exponential races implement weighted sampling without replacement
            # and avoid forming a huge view-by-candidate matrix.
            random_times = np.random.default_rng(0).exponential(size=len(eligible))
            keys = random_times / scores[eligible]
            chosen = np.argpartition(keys, selected_count - 1)[:selected_count]
            selected_mvs = np.sort(eligible[chosen])
        metadata.update({
            "depth_cache_checked": True, "depth_cache_path": str(Path(depth_cache_path).resolve()),
            "training_image_names": names, "depth_shape": list(depths.shape), "views": views,
            "valid_depth_pixel_count": int(valid.sum()),
        })
    attributes = [np.concatenate((value[base_indices], other[selected_mvs]), axis=0)
                  for value, other in zip(base, mvs)]
    cloud = BasicPointCloud(*attributes)
    if len(cloud.points) == 0:
        raise ValueError(
            "Complement selection produced an empty cloud: the base quota is zero "
            "and no MVS candidate passed. Increase the total budget or inspect the depth cache."
        )
    planned_count = base_quota + mvs_quota
    metadata.update({
        "fallback": None, "base_quota": base_quota, "mvs_quota": mvs_quota,
        "effective_mode": mode,
        "eligible_mvs_count": int(len(eligible)),
        "selected_base_count": int(len(base_indices)), "selected_mvs_count": int(len(selected_mvs)),
        "output_point_count": int(len(cloud.points)),
        "complete_base_preserved": bool(len(base_indices) == len(base[0])),
        "budget_shortfall": int((total_budget or planned_count) - len(cloud.points)),
        "geometry_filter_shortfall": int(planned_count - len(cloud.points)),
        "mvs_index_space": "input_mvs_candidate_pool",
        "selection": "coverage_weighted_without_replacement" if mode == "coverage" else "uniform_without_replacement",
        "base_first_output_order": True,
    })
    return cloud, metadata, {
        "selected_base_indices": base_indices,
        "selected_mvs_indices": selected_mvs,
        "candidate_scores": scores,
        "candidate_support_views": support,
    }
