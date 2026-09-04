"""Load a precomputed, cross-view-consistent MVS initialization."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from utils.graphics_utils import BasicPointCloud


_RESOLUTION_PATTERN = re.compile(r"_H_(\d+)_W_(\d+)_")
_RELIABILITY_GRID_MAX_DIM = 256
_RELIABILITY_NEGATIVE_DRAWS = 3
_RELIABILITY_SEED = 20_250_903


@dataclass(frozen=True)
class MvsReliabilityScore:
    """Scene-calibrated reliability for one candidate point cloud."""

    global_reliability: float
    local_reliability: np.ndarray
    geometry_evidence: float
    appearance_evidence: float
    support: np.ndarray
    backend: str


@dataclass(frozen=True)
class AdaptiveMvsPrior:
    """Residual Gaussian attributes produced by the adaptive MVS gate."""

    point_cloud: BasicPointCloud
    reliability: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    global_reliability: float
    metadata: dict

    @property
    def point_count(self) -> int:
        return int(self.point_cloud.points.shape[0])


def _positive_excess(value):
    """C1 map from a soft win probability to evidence above chance."""

    value = np.asarray(value, dtype=np.float64)
    return np.square(np.maximum(2.0 * value - 1.0, 0.0))


def _linear_positive_excess(value):
    value = np.asarray(value, dtype=np.float64)
    return np.maximum(2.0 * value - 1.0, 0.0)


def _sigmoid(value):
    value = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _soft_win_probability(positive, negatives):
    """Calibrate errors against within-scene counterfactual correspondences."""

    positive = np.asarray(positive, dtype=np.float64)
    negatives = np.asarray(negatives, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        differences = negatives - positive[None]
    finite = np.isfinite(differences)
    values = differences[finite]
    if values.size == 0:
        return np.full(positive.shape, 0.5, dtype=np.float32)
    center = np.median(values)
    temperature = 1.4826 * np.median(np.abs(values - center))
    temperature = max(float(temperature), np.finfo(np.float32).eps)
    probabilities = _sigmoid(differences / temperature)
    probabilities[~finite] = np.nan
    counts = np.sum(finite, axis=0)
    result = np.full(positive.shape, 0.5, dtype=np.float64)
    supported = counts > 0
    result[supported] = np.nansum(probabilities[:, supported], axis=0) / counts[
        supported
    ]
    return np.asarray(result, dtype=np.float32)


def _weighted_color_dispersion(colors, weights):
    finite = np.isfinite(colors).all(axis=-1) & np.isfinite(weights)
    effective = np.where(finite, weights, 0.0).astype(np.float64)
    safe_colors = np.where(finite[..., None], colors, 0.0).astype(np.float64)
    denominator = np.sum(effective, axis=0)
    mean = np.sum(effective[..., None] * safe_colors, axis=0)
    mean /= np.maximum(denominator[:, None], np.finfo(np.float32).eps)
    residual = np.mean(np.abs(safe_colors - mean[None]), axis=-1)
    dispersion = np.sum(effective * residual, axis=0)
    dispersion /= np.maximum(denominator, np.finfo(np.float32).eps)
    dispersion[denominator <= 0.0] = np.inf
    return np.asarray(dispersion, dtype=np.float32)


def _weighted_color_reference_error(colors, reference_colors, weights):
    finite = np.isfinite(colors).all(axis=-1) & np.isfinite(weights)
    residual = np.mean(
        np.abs(
            np.where(finite[..., None], colors, 0.0)
            - np.asarray(reference_colors, dtype=np.float32)[None]
        ),
        axis=-1,
    )
    effective = np.where(finite, weights, 0.0).astype(np.float64)
    denominator = np.sum(effective, axis=0)
    result = np.sum(effective * residual, axis=0)
    result /= np.maximum(denominator, np.finfo(np.float32).eps)
    result[denominator <= 0.0] = np.inf
    return np.asarray(result, dtype=np.float32)


def _camera_grid(camera, max_dimension):
    width = int(camera.width)
    height = int(camera.height)
    scale = min(1.0, float(max_dimension) / max(width, height))
    grid_width = max(1, int(round(width * scale)))
    grid_height = max(1, int(round(height * scale)))
    image = camera.image.convert("RGB").resize((grid_width, grid_height))
    image = np.asarray(image, dtype=np.float32) / 255.0
    return image, grid_width, grid_height


def _load_compatible_depth_cache(cache_path, camera_names):
    cache_path = Path(cache_path)
    depth_path = cache_path.parent / "mvs_depths_masks.npz"
    if not depth_path.is_file():
        return None
    with np.load(depth_path, allow_pickle=False) as values:
        required = {"depths", "masks", "image_names"}
        if not required.issubset(values.files):
            return None
        names = [str(name) for name in values["image_names"]]
        lookup = {name: index for index, name in enumerate(names)}
        if any(name not in lookup for name in camera_names):
            return None
        indices = [lookup[name] for name in camera_names]
        depths = np.asarray(values["depths"][indices], dtype=np.float32)
        masks = np.asarray(values["masks"][indices], dtype=bool)
    return depths, masks


def _project_points(points, camera):
    camera_points = points @ np.asarray(camera.R, dtype=np.float64)
    camera_points += np.asarray(camera.T, dtype=np.float64)[None]
    depth = camera_points[:, 2]
    projected = camera_points @ np.asarray(camera.K, dtype=np.float64).T
    denominator = np.where(
        np.abs(projected[:, 2]) > np.finfo(np.float32).eps,
        projected[:, 2],
        np.nan,
    )
    pixels = projected[:, :2] / denominator[:, None]
    valid = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < int(camera.width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < int(camera.height))
    )
    return pixels, depth, valid


def _cross_partition_depth(depth, cell_index, valid, point_partition, cell_count):
    buffers = []
    for partition in (0, 1):
        buffer = np.full(cell_count, np.inf, dtype=np.float32)
        selected = valid & (point_partition == partition)
        np.minimum.at(
            buffer,
            cell_index[selected],
            np.asarray(depth[selected], dtype=np.float32),
        )
        buffers.append(buffer)
    reference = np.full(depth.shape, np.nan, dtype=np.float32)
    for partition in (0, 1):
        selected = valid & (point_partition == partition)
        reference[selected] = buffers[1 - partition][cell_index[selected]]
    reference[~np.isfinite(reference)] = np.nan
    return reference


def score_mvs_reliability(
    point_cloud: BasicPointCloud,
    cameras,
    *,
    cache_path: str | Path | None = None,
    grid_max_dimension: int = _RELIABILITY_GRID_MAX_DIM,
    negative_draws: int = _RELIABILITY_NEGATIVE_DRAWS,
) -> MvsReliabilityScore:
    """Estimate MVS reliability from training views without dataset labels.

    Actual point/view correspondences are compared to deterministic deranged
    correspondences from the same scene.  Consequently, all error scales and
    the chance boundary are inferred from the current input rather than from a
    benchmark-specific threshold.
    """

    points = np.asarray(point_cloud.points, dtype=np.float64)
    point_colors = np.asarray(point_cloud.colors, dtype=np.float32)
    point_count = int(points.shape[0])
    view_count = len(cameras)
    if point_count == 0 or view_count < 2:
        return MvsReliabilityScore(
            global_reliability=0.0,
            local_reliability=np.zeros(point_count, dtype=np.float32),
            geometry_evidence=0.0,
            appearance_evidence=0.0,
            support=np.zeros(point_count, dtype=np.float32),
            backend="insufficient_views",
        )

    camera_names = [str(camera.image_name) for camera in cameras]
    depth_cache = (
        _load_compatible_depth_cache(cache_path, camera_names)
        if cache_path is not None
        else None
    )
    backend = "depth_cache" if depth_cache is not None else "cross_splat"
    projected_log_depth = np.full(
        (view_count, point_count), np.nan, dtype=np.float32
    )
    reference_log_depth = np.full_like(projected_log_depth, np.nan)
    sampled_colors = np.full(
        (view_count, point_count, 3), np.nan, dtype=np.float32
    )
    point_partition = np.arange(point_count, dtype=np.int64) & 1

    for view_index, camera in enumerate(cameras):
        image, grid_width, grid_height = _camera_grid(
            camera, grid_max_dimension
        )
        pixels, depth, in_frame = _project_points(points, camera)
        grid_x = np.clip(
            np.floor(pixels[:, 0] * grid_width / int(camera.width)),
            0,
            grid_width - 1,
        ).astype(np.int64)
        grid_y = np.clip(
            np.floor(pixels[:, 1] * grid_height / int(camera.height)),
            0,
            grid_height - 1,
        ).astype(np.int64)
        cell_index = grid_y * grid_width + grid_x

        if depth_cache is not None:
            depths, masks = depth_cache
            depth_height, depth_width = depths[view_index].shape
            depth_x = np.clip(
                np.floor(pixels[:, 0] * depth_width / int(camera.width)),
                0,
                depth_width - 1,
            ).astype(np.int64)
            depth_y = np.clip(
                np.floor(pixels[:, 1] * depth_height / int(camera.height)),
                0,
                depth_height - 1,
            ).astype(np.int64)
            reference = depths[view_index, depth_y, depth_x]
            valid = (
                in_frame
                & masks[view_index, depth_y, depth_x]
                & np.isfinite(reference)
                & (reference > 0.0)
            )
        else:
            valid = in_frame
            reference = _cross_partition_depth(
                depth,
                cell_index,
                valid,
                point_partition,
                grid_width * grid_height,
            )
            valid &= np.isfinite(reference) & (reference > 0.0)

        projected_log_depth[view_index, valid] = np.log(depth[valid])
        reference_log_depth[view_index, valid] = np.log(reference[valid])
        sampled_colors[view_index, valid] = image[grid_y[valid], grid_x[valid]]

    geometry_matrix = np.abs(projected_log_depth - reference_log_depth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        geometry_positive = np.nanmedian(geometry_matrix, axis=0)
    supported_views = np.sum(np.isfinite(geometry_matrix), axis=0)
    support = np.clip(
        (supported_views.astype(np.float32) - 1.0) / max(view_count - 1, 1),
        0.0,
        1.0,
    )

    finite_geometry = geometry_matrix[np.isfinite(geometry_matrix)]
    if finite_geometry.size:
        geometry_center = np.median(finite_geometry)
        geometry_scale = geometry_center + 1.4826 * np.median(
            np.abs(finite_geometry - geometry_center)
        )
    else:
        geometry_scale = 0.0
    geometry_scale = max(
        float(geometry_scale), np.finfo(np.float32).eps
    )
    visibility_weights = np.exp(
        -np.nan_to_num(geometry_matrix, nan=np.inf) / geometry_scale
    ).astype(np.float32)
    appearance_positive = 0.5 * (
        _weighted_color_dispersion(sampled_colors, visibility_weights)
        + _weighted_color_reference_error(
            sampled_colors, point_colors, visibility_weights
        )
    )

    geometry_negatives = []
    appearance_negatives = []
    for draw in range(int(negative_draws)):
        deranged_depth = np.full_like(reference_log_depth, np.nan)
        deranged_colors = np.full_like(sampled_colors, np.nan)
        for view_index in range(view_count):
            finite = np.flatnonzero(
                np.isfinite(reference_log_depth[view_index])
            )
            if finite.size < 2:
                continue
            offset = 1 + (
                (draw + 1) * (finite.size // (negative_draws + 1) + 1)
            ) % (finite.size - 1)
            deranged_depth[view_index, finite] = np.roll(
                reference_log_depth[view_index, finite], offset
            )
            deranged_colors[view_index, finite] = np.roll(
                sampled_colors[view_index, finite], offset, axis=0
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            geometry_negatives.append(
                np.nanmedian(
                    np.abs(projected_log_depth - deranged_depth), axis=0
                )
            )
        appearance_negatives.append(
            0.5
            * (
                _weighted_color_dispersion(
                    deranged_colors, visibility_weights
                )
                + _weighted_color_reference_error(
                    deranged_colors, point_colors, visibility_weights
                )
            )
        )

    geometry_q = _soft_win_probability(
        geometry_positive, np.asarray(geometry_negatives)
    )
    appearance_q = _soft_win_probability(
        appearance_positive, np.asarray(appearance_negatives)
    )
    evidence_weight = support.astype(np.float64)
    weight_sum = float(np.sum(evidence_weight))
    if weight_sum > 0.0:
        geometry_auc = float(np.sum(evidence_weight * geometry_q) / weight_sum)
        appearance_auc = float(
            np.sum(evidence_weight * appearance_q) / weight_sum
        )
    else:
        geometry_auc = 0.5
        appearance_auc = 0.5
    geometry_evidence = float(_positive_excess(geometry_auc))
    appearance_evidence = float(_positive_excess(appearance_auc))
    global_reliability = float(
        np.sqrt(geometry_evidence * appearance_evidence)
    )

    epsilon = np.finfo(np.float32).eps
    geometry_logit = np.log(
        np.clip(geometry_q, epsilon, 1.0 - epsilon)
        / np.clip(1.0 - geometry_q, epsilon, 1.0)
    )
    appearance_logit = np.log(
        np.clip(appearance_q, epsilon, 1.0 - epsilon)
        / np.clip(1.0 - appearance_q, epsilon, 1.0)
    )
    joint_q = _sigmoid(0.5 * (geometry_logit + appearance_logit))
    local_reliability = (
        _linear_positive_excess(joint_q) * support.astype(np.float64)
    )
    return MvsReliabilityScore(
        global_reliability=global_reliability,
        local_reliability=np.asarray(local_reliability, dtype=np.float32),
        geometry_evidence=geometry_evidence,
        appearance_evidence=appearance_evidence,
        support=support,
        backend=backend,
    )


def _local_planarity(points, query_points):
    points = np.asarray(points, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    if points.shape[0] < 3 or query_points.shape[0] == 0:
        return np.zeros(query_points.shape[0], dtype=np.float32)
    neighbor_count = min(9, points.shape[0])
    tree = cKDTree(points)
    neighbors = tree.query(
        query_points, k=neighbor_count, workers=-1
    )[1]
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    neighborhood = points[neighbors]
    centered = neighborhood - np.mean(neighborhood, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    covariance /= neighbor_count
    eigenvalues = np.linalg.eigvalsh(covariance)
    total = np.maximum(
        np.sum(eigenvalues, axis=1), np.finfo(np.float32).eps
    )
    planarity = 1.0 - 3.0 * eigenvalues[:, 0] / total
    return np.asarray(np.clip(planarity, 0.0, 1.0), dtype=np.float32)


def score_surface_structure(
    point_cloud: BasicPointCloud,
    *,
    max_points: int = 50_000,
    negative_draws: int = _RELIABILITY_NEGATIVE_DRAWS,
) -> float:
    """Test whether a cloud contains surfaces rather than volume samples.

    Independently permuting its coordinate axes preserves the scene bounds and
    marginal coordinate distributions while destroying local surfaces.  The
    returned score is therefore a scene-calibrated structural anchor signal,
    not a dataset or point-count heuristic.
    """

    points = np.asarray(point_cloud.points, dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0
    sample_count = min(int(max_points), int(points.shape[0]))
    indices = np.linspace(
        0, points.shape[0] - 1, sample_count, dtype=np.int64
    )
    sampled = points[indices]
    positive_error = 1.0 - _local_planarity(points, sampled)
    generator = np.random.default_rng(_RELIABILITY_SEED)
    negative_errors = []
    for _ in range(int(negative_draws)):
        deranged = np.column_stack(
            [
                points[generator.permutation(points.shape[0]), axis]
                for axis in range(3)
            ]
        )
        negative_errors.append(
            1.0 - _local_planarity(deranged, deranged[indices])
        )
    draw_evidence = []
    for negative_error in negative_errors:
        q = _soft_win_probability(
            positive_error, np.asarray([negative_error])
        )
        draw_evidence.append(float(_positive_excess(np.mean(q))))
    return float(np.prod(draw_evidence) ** (1.0 / len(draw_evidence)))


def score_spatial_alignment(
    base: BasicPointCloud,
    mvs: BasicPointCloud,
    *,
    max_anchor_points: int = 50_000,
    negative_draws: int = _RELIABILITY_NEGATIVE_DRAWS,
) -> float:
    """Measure SfM-to-MVS alignment against within-scene null clouds.

    The positive statistic is the distance from each sampled SfM anchor to
    the dense MVS cloud.  Null clouds independently permute the MVS x/y/z
    coordinates, retaining its bounds and one-dimensional marginals while
    destroying its surfaces and alignment.  A soft rank against those nulls
    removes the need for a metric-distance threshold across scene scales.
    """

    base_points = np.asarray(base.points, dtype=np.float64)
    mvs_points = np.asarray(mvs.points, dtype=np.float64)
    if base_points.shape[0] == 0 or mvs_points.shape[0] == 0:
        return 0.0
    sample_count = min(int(max_anchor_points), int(base_points.shape[0]))
    indices = np.linspace(
        0, base_points.shape[0] - 1, sample_count, dtype=np.int64
    )
    anchors = base_points[indices]
    positive_distance = cKDTree(mvs_points).query(
        anchors, k=1, workers=-1
    )[0]
    generator = np.random.default_rng(_RELIABILITY_SEED)
    negative_distances = []
    for _ in range(int(negative_draws)):
        deranged = np.column_stack(
            [
                mvs_points[
                    generator.permutation(mvs_points.shape[0]), axis
                ]
                for axis in range(3)
            ]
        )
        negative_distances.append(
            cKDTree(deranged).query(anchors, k=1, workers=-1)[0]
        )
    q = _soft_win_probability(
        positive_distance, np.asarray(negative_distances)
    )
    return float(_linear_positive_excess(np.mean(q)))


def _empty_adaptive_prior(metadata):
    empty = np.empty((0, 3), dtype=np.float32)
    return AdaptiveMvsPrior(
        point_cloud=BasicPointCloud(
            points=empty, colors=empty.copy(), normals=empty.copy()
        ),
        reliability=np.empty(0, dtype=np.float32),
        scales=empty.copy(),
        rotations=np.empty((0, 4), dtype=np.float32),
        opacities=np.empty((0, 1), dtype=np.float32),
        global_reliability=0.0,
        metadata=metadata,
    )


def _estimate_gaussian_shapes(
    points,
    selected_indices,
    reliability,
    tree,
    base_points,
):
    selected_points = points[selected_indices]
    neighbor_count = min(9, points.shape[0])
    distances, neighbors = tree.query(
        selected_points, k=neighbor_count, workers=-1
    )
    if neighbor_count == 1:
        distances = np.ones((selected_points.shape[0], 1), dtype=np.float64)
        neighbors = np.zeros_like(distances, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    neighborhood = points[neighbors]
    centered = neighborhood - np.mean(neighborhood, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    covariance /= max(neighbor_count, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # PCA is estimated on the dense candidate cloud, but the weak-prior scale
    # must reflect the points that will actually be instantiated.  Using the
    # full MVS spacing here makes a thinned residual cloud nearly invisible and
    # was especially harmful when the independent anchor cloud was sparse.
    instantiated = selected_points
    if np.asarray(base_points).shape[0]:
        instantiated = np.concatenate((base_points, selected_points), axis=0)
    instantiated_tree = cKDTree(instantiated)
    instantiated_distances = instantiated_tree.query(
        selected_points,
        k=min(2, instantiated.shape[0]),
        workers=-1,
    )[0]
    if instantiated_distances.ndim == 1:
        spacing = instantiated_distances
    else:
        spacing = instantiated_distances[:, -1]
    spacing = np.maximum(spacing, np.finfo(np.float32).eps)
    prior_scales = np.sqrt(
        np.maximum(
            eigenvalues,
            np.finfo(np.float32).eps * np.square(spacing[:, None]),
        )
    )
    blended_scales = np.exp(
        (1.0 - reliability[:, None]) * np.log(spacing[:, None])
        + reliability[:, None] * np.log(prior_scales)
    )
    determinants = np.linalg.det(eigenvectors)
    eigenvectors[determinants < 0.0, :, 0] *= -1.0
    xyzw = Rotation.from_matrix(eigenvectors).as_quat()
    rotations = np.column_stack((xyzw[:, 3], xyzw[:, :3]))
    normals = eigenvectors[:, :, 0]
    return (
        np.asarray(blended_scales, dtype=np.float32),
        np.asarray(rotations, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
    )


def build_adaptive_mvs_prior(
    base: BasicPointCloud,
    mvs: BasicPointCloud,
    cameras,
    *,
    cache_path: str | Path,
    max_residual_points: int,
) -> AdaptiveMvsPrior:
    """Build a baseline-preserving, scene-adaptive residual MVS prior."""

    mvs_points = np.asarray(mvs.points, dtype=np.float64)
    base_points = np.asarray(base.points, dtype=np.float64)
    score = score_mvs_reliability(mvs, cameras, cache_path=cache_path)
    anchor_structure = score_surface_structure(base)
    spatial_alignment = score_spatial_alignment(base, mvs)
    # Each calibrated view is another opportunity for independently anchored
    # surface evidence to be observed.  The probabilistic union increases the
    # useful residual budget with view support without a view-count branch or
    # an absolute quality threshold.  Squaring the per-view evidence still
    # makes chance-level fluctuations vanish fast enough for an exact discrete
    # fallback on unstructured anchor clouds.
    per_view_evidence = anchor_structure * spatial_alignment
    anchor_reliability = 1.0 - np.power(
        1.0 - per_view_evidence, max(len(cameras), 1)
    )
    # MVS self-consistency may be correlated and therefore cannot open the
    # gate by itself.  Once independent anchor evidence exists, however, it is
    # useful for deciding how much dense coverage to admit.  Treating it as an
    # effective number of additional observations gives a smooth boost while
    # retaining O(anchor) behavior near the exact-fallback limit.
    evidence_multiplicity = 1.0 + len(cameras) * score.global_reliability
    global_reliability = 1.0 - np.power(
        1.0 - anchor_reliability, evidence_multiplicity
    )
    metadata = {
        "reliability_policy": "counterfactual_geometry_appearance",
        "reliability_backend": score.backend,
        "global_reliability": global_reliability,
        "anchor_reliability": anchor_reliability,
        "evidence_multiplicity": evidence_multiplicity,
        "mvs_self_reliability": score.global_reliability,
        "base_structure_evidence": anchor_structure,
        "spatial_alignment_evidence": spatial_alignment,
        "geometry_evidence": score.geometry_evidence,
        "appearance_evidence": score.appearance_evidence,
        "candidate_point_count": int(mvs_points.shape[0]),
        "max_residual_points": int(max_residual_points),
    }
    if (
        global_reliability <= 0.0
        or max_residual_points <= 0
        or mvs_points.shape[0] == 0
    ):
        metadata.update(
            {
                "fusion_policy": "adaptive_exact_baseline",
                "mvs_point_count": 0,
            }
        )
        return _empty_adaptive_prior(metadata)

    mvs_tree = cKDTree(mvs_points)

    if base_points.shape[0] > 1:
        base_tree = cKDTree(base_points)
        base_distances, base_indices = base_tree.query(
            mvs_points, k=1, workers=-1
        )
        base_neighbors = base_tree.query(
            base_points, k=2, workers=-1
        )[0]
        base_spacing = np.maximum(
            base_neighbors[:, 1], np.finfo(np.float32).eps
        )
        normalized_distance = base_distances / base_spacing[base_indices]
        novelty = 1.0 - np.exp(-0.5 * np.square(normalized_distance))

        anchor = anchor_structure * np.exp(
            -0.5 * np.square(normalized_distance)
        )
        local_reliability = 1.0 - (
            1.0 - score.local_reliability
        ) * (1.0 - anchor)
        metadata["base_anchor_reliability"] = anchor_structure
    else:
        novelty = np.ones(mvs_points.shape[0], dtype=np.float64)
        local_reliability = score.local_reliability.astype(np.float64)
        metadata["base_anchor_reliability"] = 0.0

    capacity = min(int(max_residual_points), int(mvs_points.shape[0]))
    # Correct the conservative independent-anchor gate by the probability that
    # the MVS evidence is *not* self-consistent.  This is an anchored
    # importance correction: self-consistency can increase coverage only when
    # the independent gate is already non-zero, so a correlated synthetic
    # failure cannot activate itself.
    anchored_admission = min(
        1.0,
        global_reliability
        / max(1.0 - score.global_reliability, np.finfo(np.float32).eps),
    )
    # Once the independent anchor has opened the gate, repeated calibrated
    # evidence may cover complementary parts of the surface.  This is the
    # probability that at least one of the effective observations supports a
    # candidate.  It is continuous, parameter-free, and remains exactly zero
    # when the independent anchor evidence is zero.
    admission_probability = 1.0 - np.power(
        1.0 - anchored_admission, evidence_multiplicity
    )
    activation_mass = admission_probability * capacity
    if int(np.floor(activation_mass)) == 0:
        metadata.update(
            {
                "fusion_policy": "adaptive_exact_baseline",
                "mvs_point_count": 0,
            }
        )
        return _empty_adaptive_prior(metadata)

    # Once the scene gate opens, retain dense surface coverage and encode its
    # confidence as opacity mass.  Thinning uncertain geometry inflates KNN
    # scales, so a small number of large splats can occlude more of the image
    # than the original dense prior.  Keeping the resource-bounded population
    # makes scale estimation stable while total prior opacity still tends
    # continuously to zero with admission_probability.
    instantiation_probability = 1.0
    selected_count = capacity
    sampling_reliability = (
        score.global_reliability
        + (1.0 - score.global_reliability)
        * np.asarray(local_reliability, dtype=np.float64)
    )
    adaptive_sampling_weight = sampling_reliability * novelty
    positive_adaptive = adaptive_sampling_weight > 0.0
    if np.any(positive_adaptive):
        adaptive_sampling_weight /= np.mean(
            adaptive_sampling_weight[positive_adaptive]
        )
    coverage_reliability = 1.0 - np.power(
        1.0 - score.global_reliability, max(len(cameras), 1)
    )
    uniform_mixture = 1.0 - (
        1.0 - coverage_reliability
    ) * (1.0 - admission_probability)
    sampling_weight = (
        (1.0 - uniform_mixture) * adaptive_sampling_weight
        + uniform_mixture
    )
    positive = sampling_weight > 0.0
    if not np.any(positive):
        metadata.update(
            {
                "fusion_policy": "adaptive_exact_baseline",
                "mvs_point_count": 0,
            }
        )
        return _empty_adaptive_prior(metadata)
    selected_count = min(selected_count, int(np.count_nonzero(positive)))
    if selected_count == mvs_points.shape[0]:
        selected = np.arange(selected_count, dtype=np.int64)
    else:
        generator = np.random.default_rng(_RELIABILITY_SEED)
        random_values = np.maximum(
            generator.random(mvs_points.shape[0]), np.finfo(np.float64).tiny
        )
        priority = np.full(mvs_points.shape[0], np.inf, dtype=np.float64)
        priority[positive] = -np.log(random_values[positive]) / sampling_weight[
            positive
        ]
        selected = np.argpartition(
            priority, selected_count - 1
        )[:selected_count]
        selected = selected[np.argsort(priority[selected], kind="stable")]

    selected_reliability = np.clip(
        np.asarray(local_reliability[selected], dtype=np.float64),
        0.0,
        1.0,
    )
    # Anisotropy is a stronger geometric commitment than admitting a point.
    # It therefore uses the joint scene/local confidence.  When the scene gate
    # is only moderately certain, accepted points start near the standard
    # isotropic 3DGS scale and retain a larger optimization basin.
    joint_reliability = global_reliability * selected_reliability
    shape_reliability = np.square(joint_reliability)
    scales, rotations, normals = _estimate_gaussian_shapes(
        mvs_points,
        selected,
        shape_reliability,
        mvs_tree,
        base_points,
    )
    base_opacity = 0.1
    opacity_support = (
        score.global_reliability
        + (1.0 - score.global_reliability) * selected_reliability
    )
    opacity_support /= max(
        float(np.mean(opacity_support)), np.finfo(np.float32).eps
    )
    opacity_scene_evidence = admission_probability
    opacity_evidence = np.clip(
        opacity_scene_evidence * opacity_support, 0.0, 1.0
    )
    opacities = 1.0 - np.power(
        1.0 - base_opacity, opacity_evidence[:, None]
    )
    selected_pcd = BasicPointCloud(
        points=np.asarray(mvs.points, dtype=np.float32)[selected],
        colors=np.asarray(mvs.colors, dtype=np.float32)[selected],
        normals=normals,
    )
    metadata.update(
        {
            "fusion_policy": "adaptive_residual",
            "mvs_point_count": selected_count,
            "activation_mass": activation_mass,
            "instantiation_probability": instantiation_probability,
            "opacity_scene_evidence": opacity_scene_evidence,
            "effective_opacity_mass": float(
                np.sum(opacities) / base_opacity
            ),
            "anchored_admission": anchored_admission,
            "admission_probability": admission_probability,
            "coverage_reliability": coverage_reliability,
            "local_reliability_mean": float(
                np.mean(selected_reliability)
            ),
            "local_reliability_median": float(
                np.median(selected_reliability)
            ),
        }
    )
    return AdaptiveMvsPrior(
        point_cloud=selected_pcd,
        reliability=np.asarray(selected_reliability, dtype=np.float32),
        scales=scales,
        rotations=rotations,
        opacities=np.asarray(opacities, dtype=np.float32),
        global_reliability=global_reliability,
        metadata=metadata,
    )


def resolve_mvs_init_strategy(requested: str) -> str:
    """Resolve automatic fusion without dataset or view-count routing."""

    if requested not in ("auto", "adaptive", "replace", "augment"):
        raise ValueError(
            "mvs_init_strategy must be 'auto', 'adaptive', 'replace', "
            "or 'augment'"
        )
    return "adaptive" if requested == "auto" else requested


def resolve_mvs_init_point_budget(
    requested_strategy: str,
    resolved_strategy: str,
    configured_max_points: int,
) -> int:
    """Validate the candidate budget; reliability is handled separately."""

    del requested_strategy, resolved_strategy
    return int(configured_max_points)


def resolve_mvs_fusion_budget(
    requested_strategy: str,
    configured_total_points: int,
    *,
    base_point_count: int,
    dense_base_threshold: int,
    dense_residual_points: int,
) -> int:
    """Return the explicit resource budget for legacy ablations."""

    del requested_strategy, base_point_count, dense_base_threshold
    del dense_residual_points
    return int(configured_total_points)


def _resolution_area(path: Path) -> int:
    match = _RESOLUTION_PATTERN.search(path.parent.name)
    return int(match.group(1)) * int(match.group(2)) if match else 0


def _metadata_compatibility(path: Path, train_camera_names: list[str]):
    metadata_path = path.parent / "mvs_cache_meta.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        depth_path = path.parent / "mvs_depths_masks.npz"
        if not depth_path.is_file():
            return None, None
        with np.load(depth_path, allow_pickle=False) as values:
            if "image_names" not in values:
                return None, None
            metadata = {
                "view_names": [str(name) for name in values["image_names"]],
                "provenance_source": "mvs_depths_masks.npz",
            }
    cached_names = [str(name) for name in metadata.get("view_names", [])]
    return cached_names == list(train_camera_names), metadata


def resolve_mvs_init_path(
    source_path: str | Path,
    *,
    n_views: int,
    train_camera_names: list[str],
    explicit_path: str | Path = "",
) -> tuple[Path, dict | None, bool]:
    """Resolve a compatible cache, rejecting provenance-known view leakage."""

    source_path = Path(source_path).resolve()
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "mvs_input.ply"
        candidates = [candidate]
    else:
        candidates = sorted(
            source_path.glob(
                f"MVS*_{int(n_views)}_views_*_init/mvs_input.ply"
            )
        )
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"no MVS {n_views}-view mvs_input.ply below {source_path}"
        )

    verified = []
    unverified = []
    incompatible = []
    for candidate in candidates:
        compatible, metadata = _metadata_compatibility(
            candidate, train_camera_names
        )
        if compatible is True:
            verified.append((candidate, metadata))
        elif compatible is False:
            incompatible.append(candidate)
        else:
            unverified.append((candidate, metadata))
    pool = verified or unverified
    if not pool:
        raise ValueError(
            "all MVS caches were generated from cameras that differ "
            f"from the current training split: {incompatible}"
        )
    candidate, metadata = max(
        pool,
        key=lambda item: (
            item[0].parent.name.startswith("MVSAnywhere_"),
            _resolution_area(item[0]),
        ),
    )
    return candidate, metadata, bool(verified)


def load_mvs_point_cloud(
    source_path: str | Path,
    *,
    n_views: int,
    train_camera_names: list[str],
    explicit_path: str | Path = "",
    max_points: int = 250_000,
):
    if (
        not isinstance(max_points, int)
        or isinstance(max_points, bool)
        or max_points <= 0
    ):
        raise ValueError("mvs_init_max_points must be a positive integer")
    path, cache_metadata, provenance_verified = resolve_mvs_init_path(
        source_path,
        n_views=n_views,
        train_camera_names=train_camera_names,
        explicit_path=explicit_path,
    )
    vertices = PlyData.read(path)["vertex"]
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32) / 255.0
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points, colors = points[finite], colors[finite]
    source_count = int(points.shape[0])
    if source_count == 0:
        raise ValueError(f"MVSAnywhere point cloud is empty: {path}")
    if source_count > max_points:
        generator = np.random.default_rng(0)
        indices = np.sort(generator.choice(source_count, max_points, replace=False))
        points, colors = points[indices], colors[indices]
    digest = hashlib.sha256()
    digest.update(points.tobytes())
    digest.update(colors.tobytes())
    metadata = {
        "path": str(path),
        "backend": path.parent.name.split("_", 1)[0],
        "source_point_count": source_count,
        "selected_point_count": int(points.shape[0]),
        "max_points": int(max_points),
        "selection": (
            "all"
            if source_count <= max_points
            else "seed0_uniform_without_replacement"
        ),
        "selected_sha256": digest.hexdigest(),
        "camera_provenance_verified": provenance_verified,
        "cache_generation_id": (cache_metadata or {}).get("generation_id"),
    }
    return (
        BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points),
        ),
        path,
        metadata,
    )


def fuse_residual_mvs_point_cloud(
    base: BasicPointCloud,
    mvs: BasicPointCloud,
    *,
    total_max_points: int = 350_000,
) -> tuple[BasicPointCloud, dict]:
    """Preserve a reliable base cloud and fill its point budget with MVS."""

    if (
        not isinstance(total_max_points, int)
        or isinstance(total_max_points, bool)
        or total_max_points <= 0
    ):
        raise ValueError("mvs_init_total_max_points must be a positive integer")

    base_points = np.asarray(base.points, dtype=np.float32)
    base_colors = np.asarray(base.colors, dtype=np.float32)
    mvs_points = np.asarray(mvs.points, dtype=np.float32)
    mvs_colors = np.asarray(mvs.colors, dtype=np.float32)
    if base_points.ndim != 2 or base_points.shape[1] != 3:
        raise ValueError("base point cloud must have shape [N, 3]")
    if base_colors.shape != base_points.shape:
        raise ValueError("base colors must match base point positions")
    base_finite = np.isfinite(base_points).all(axis=1) & np.isfinite(
        base_colors
    ).all(axis=1)
    base_points, base_colors = base_points[base_finite], base_colors[base_finite]
    base_count = int(base_points.shape[0])
    if base_count == 0:
        raise ValueError("base point cloud is empty")

    if base_count >= total_max_points:
        points, colors = base_points, base_colors
        mvs_count = 0
        policy = "base_budget_passthrough"
    else:
        mvs_count = min(int(mvs_points.shape[0]), total_max_points - base_count)
        if mvs_count < mvs_points.shape[0]:
            generator = np.random.default_rng(1)
            mvs_indices = np.sort(
                generator.choice(mvs_points.shape[0], mvs_count, replace=False)
            )
            selected_mvs_points = mvs_points[mvs_indices]
            selected_mvs_colors = mvs_colors[mvs_indices]
        else:
            selected_mvs_points = mvs_points
            selected_mvs_colors = mvs_colors
        points = np.concatenate((base_points, selected_mvs_points), axis=0)
        colors = np.concatenate((base_colors, selected_mvs_colors), axis=0)
        policy = "preserve_base_then_fill_with_mvs"

    digest = hashlib.sha256()
    digest.update(points.tobytes())
    digest.update(colors.tobytes())
    return (
        BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points),
        ),
        {
            "fusion_policy": policy,
            "base_point_count": base_count,
            "mvs_point_count": mvs_count,
            "fused_point_count": int(points.shape[0]),
            "total_max_points": int(total_max_points),
            "fused_sha256": digest.hexdigest(),
        },
    )
