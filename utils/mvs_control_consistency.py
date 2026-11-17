"""Train-only multiview association for fused-MVS geometry controls.

Being close to a feature in one camera does not identify its 3D point: another
surface on that ray can be arbitrarily far away. This module requires the SAME
source sample to explain every observation of a triangulated track. It does not
establish ground-truth correspondence, recover missing geometry, or certify
occlusion. The point z-buffer is only an approximate candidate generator.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree


def _finite_array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real coordinates")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _validated_tracks(tracks, camera_count):
    normalized = []
    for track in tracks:
        if not isinstance(track, Mapping) or not {"point", "camera_indices", "pixels"} <= track.keys():
            raise ValueError("each track requires point, camera_indices and pixels")
        target = _finite_array(track["point"], (3,), "triangulated target")
        try:
            indices, pixels = list(track["camera_indices"]), list(track["pixels"])
        except TypeError as exc:
            raise ValueError("track camera_indices and pixels must be sequences") from exc
        if (len(indices) < 2 or len(indices) != len(pixels)
                or any(isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) for index in indices)):
            raise ValueError("each track requires matching observations in at least two integer camera indices")
        if len(set(indices)) != len(indices) or any(index < 0 or index >= camera_count for index in indices):
            raise ValueError("track must refer to distinct cameras from the supplied training list")
        pixel_array = _finite_array(pixels, (len(indices), 2), "track pixels")
        error = track.get("reprojection_error_px", 0.0)
        if (isinstance(error, (bool, np.bool_)) or not isinstance(error, Real)
                or not np.isfinite(error) or error < 0):
            raise ValueError("track reprojection error must be finite and nonnegative")
        normalized.append((target, [int(index) for index in indices], pixel_array, float(error)))
    return normalized


def associate_consistent_control_points(points, cameras, tracks, *, max_image_size=1600,
                                       match_radius_px=3.0):
    """Return one-to-one multiview-consistent ``(source, target, ids, metadata)``.

    For each track, enumerate the union of frontmost source samples within its
    pixel radius in every observed camera. Project each candidate into ALL of
    those cameras and require positive depth, image bounds and bounded error in
    matching-resolution pixels. No distance to the target in world units is
    used to generate or admit a correspondence.

    Sort admissible edges by (maximum error, mean error, scaled triangulation
    reprojection error, track id, source id), then greedily assign one source to
    one track. Exact source-coordinate and exact target-coordinate duplicates
    are also forbidden. Alternative edges remain eligible when a preferred
    source was assigned to a better track. This deterministic greedy matching
    is not claimed to maximize cardinality or to be globally optimal.

    Return pairs sorted by source index. All ``chosen_*`` metadata arrays follow
    that same order. Projection memory is bounded by candidates of one track,
    rather than constructing a full point-by-track distance matrix.
    """
    # Local imports permit mvs_geometry to select this helper without a cycle.
    from utils.mvs_geometry import projected_cloud_index
    from utils.train_view_controls import camera_geometry

    if (isinstance(max_image_size, (bool, np.bool_)) or not isinstance(max_image_size, Integral)
            or max_image_size <= 0):
        raise ValueError("max_image_size must be a positive integer")
    if (isinstance(match_radius_px, (bool, np.bool_)) or not isinstance(match_radius_px, Real)
            or not np.isfinite(match_radius_px) or match_radius_px <= 0):
        raise ValueError("match_radius_px must be finite and positive")
    max_image_size, radius = int(max_image_size), float(match_radius_px)
    if np.iscomplexobj(points):
        raise ValueError("MVS points must contain real coordinates")
    try:
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("MVS points must be finite [N, 3]") from exc
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("MVS points must be finite [N, 3]")
    cameras = list(cameras)
    normalized = _validated_tracks(tracks, len(cameras))
    geometries = [camera_geometry(camera) for camera in cameras]
    indexes = [projected_cloud_index(points, camera, max_image_size) for camera in cameras]
    sizes = [np.asarray(camera.image).shape[:2] for camera in cameras]
    trees = [cKDTree(projected) for projected, _, _ in indexes]
    edges = []
    proposed_count = 0
    tracks_with_proposals = 0
    tracks_with_consistent_edges = 0
    maximum_candidate_count = 0
    for track_id, (target, view_indices, pixels, track_error) in enumerate(normalized):
        candidate_ids = set()
        for camera_index, pixel in zip(view_indices, pixels):
            _, source_ids, scale = indexes[camera_index]
            if not len(source_ids):
                continue
            neighbors = trees[camera_index].query_ball_point(pixel * scale, radius)
            candidate_ids.update(int(index) for index in source_ids[neighbors])
        if not candidate_ids:
            continue
        tracks_with_proposals += 1
        ids = np.asarray(sorted(candidate_ids), dtype=np.int64)
        proposed_count += len(ids)
        maximum_candidate_count = max(maximum_candidate_count, len(ids))
        candidates = points[ids]
        valid = np.ones(len(ids), dtype=bool)
        per_view_errors = []
        for camera_index, pixel in zip(view_indices, pixels):
            k, rt = geometries[camera_index]
            camera_points = candidates @ rt[:, :3].T + rt[:, 3]
            homogeneous = camera_points @ k.T
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                projected = homogeneous[:, :2] / homogeneous[:, 2:]
                error = np.linalg.norm(projected - pixel, axis=1) * indexes[camera_index][2]
            height, width = sizes[camera_index]
            valid &= (
                (camera_points[:, 2] > np.finfo(np.float64).eps)
                & np.isfinite(projected).all(axis=1) & np.isfinite(error)
                & (projected[:, 0] >= 0) & (projected[:, 0] < width)
                & (projected[:, 1] >= 0) & (projected[:, 1] < height)
                & (error <= radius)
            )
            per_view_errors.append(error)
        if not valid.any():
            continue
        tracks_with_consistent_edges += 1
        errors = np.stack(per_view_errors, axis=1)
        scaled_track_error = track_error * max(indexes[index][2] for index in view_indices)
        for index in np.flatnonzero(valid):
            # Keep all admissible edges, not only one preferred source per track.
            per_view = tuple(float(value) for value in errors[index])
            score = (max(per_view), float(np.mean(per_view)), scaled_track_error,
                     track_id, int(ids[index]))
            edges.append((score, per_view))

    used_tracks, used_sources, used_source_xyz, used_target_xyz = set(), set(), set(), set()
    chosen = []
    for score, per_view in sorted(edges, key=lambda edge: edge[0]):
        track_id, source_id = score[3:]
        source_key = tuple(points[source_id])
        target_key = tuple(normalized[track_id][0])
        if (track_id in used_tracks or source_id in used_sources
                or source_key in used_source_xyz or target_key in used_target_xyz):
            continue
        used_tracks.add(track_id)
        used_sources.add(source_id)
        used_source_xyz.add(source_key)
        used_target_xyz.add(target_key)
        chosen.append((source_id, track_id, score, per_view))
    chosen.sort(key=lambda item: item[0])
    source_ids = np.asarray([item[0] for item in chosen], dtype=np.int64)
    targets = np.asarray([normalized[item[1]][0] for item in chosen], dtype=np.float64).reshape(-1, 3)
    metadata = {
        "association": "training_track_all_view_radius_candidates_one_to_one",
        "association_radius_matching_pixels": radius,
        "matching_max_image_size": max_image_size,
        "visible_mvs_points_per_camera": [int(len(item[1])) for item in indexes],
        "track_count": len(normalized),
        "tracks_with_candidate_sources": tracks_with_proposals,
        "tracks_with_multiview_consistent_sources": tracks_with_consistent_edges,
        "proposed_candidate_edges": proposed_count,
        "accepted_multiview_candidate_edges": len(edges),
        "max_candidate_sources_per_track": maximum_candidate_count,
        "accepted_pairs": len(chosen),
        "unique_source_candidates": len(chosen),
        "one_to_one_policy": "deterministic_global_greedy_all_eligible_edges",
        "duplicate_coordinate_policy": "exact_source_xyz_and_target_xyz_unique",
        "multiview_policy": "same_source_positive_depth_inbounds_bounded_error_in_all_track_views",
        "chosen_track_indices": [item[1] for item in chosen],
        "chosen_max_reprojection_matching_px": [item[2][0] for item in chosen],
        "chosen_mean_reprojection_matching_px": [item[2][1] for item in chosen],
        "chosen_per_view_reprojection_matching_px": [list(item[3]) for item in chosen],
        "chosen_camera_indices": [normalized[item[1]][1] for item in chosen],
        "visibility_caveat": "point_zbuffer_candidates_only_not_an_occlusion_or_geometry_certificate",
    }
    return points[source_ids], targets, source_ids, metadata
