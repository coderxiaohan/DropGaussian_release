"""Optional, initialization-only TWINGS-inspired correction of a fused MVS cloud.

Unlike original TWINGS (predicted depths + MASt3R), this experiment associates
the existing fused MVS points with SIFT tracks using camera projection. Targets
are triangulated exclusively from supplied training images. No SfM cloud, test
image, training loss, or density-control operation is used here.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.spatial import cKDTree

from utils.graphics_utils import BasicPointCloud
from utils.tps_warp import fit_similarity, fit_tps, select_control_indices
from utils.train_view_controls import build_training_tracks, camera_geometry


def cloud_digest(points, colors):
    digest = hashlib.sha256()
    for array in (points, colors):
        digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()


def projected_cloud_index(points, camera, max_image_size=1600):
    """Return approximate frontmost points per matching-resolution pixel.

    This is a point z-buffer, not a mesh visibility oracle. Nearest association
    is subsequently bounded in image space, never by target 3D distance.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("MVS points must be finite [N, 3]")
    if not isinstance(max_image_size, int) or max_image_size <= 0:
        raise ValueError("max_image_size must be a positive integer")
    K, extrinsic = camera_geometry(camera)
    height, width = np.asarray(camera.image).shape[:2]
    scale = min(1.0, max_image_size / max(width, height))
    raster_width = max(1, int(round(width * scale)))
    raster_height = max(1, int(round(height * scale)))
    camera_points = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    positive = np.flatnonzero(camera_points[:, 2] > np.finfo(np.float64).eps)
    homogeneous = camera_points[positive] @ K.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    indices = positive[inside]
    pixels = pixels[inside] * scale
    if not len(indices):
        return pixels, indices, scale
    bins = np.floor(pixels).astype(np.int64)
    bins[:, 0] = np.minimum(bins[:, 0], raster_width - 1)
    bins[:, 1] = np.minimum(bins[:, 1], raster_height - 1)
    pixel_ids = bins[:, 1] * raster_width + bins[:, 0]
    # Primary key pixel id, secondary positive depth, final deterministic index.
    order = np.lexsort((indices, camera_points[indices, 2], pixel_ids))
    sorted_ids = pixel_ids[order]
    front = order[np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]]
    return pixels[front], indices[front], scale


def associate_control_points(points, cameras, tracks, *, max_image_size=1600,
                             match_radius_px=3.0):
    """Pair visible MVS samples with train-only triangulation targets.

    Repeated observations of one MVS sample keep the smallest image-space
    association + reprojection error, rather than create inconsistent duplicate
    TPS source nodes. Points behind the camera/outside the image are excluded.
    """
    if not np.isfinite(match_radius_px) or match_radius_px <= 0:
        raise ValueError("match_radius_px must be finite and positive")
    points = np.asarray(points, dtype=np.float64)
    observations = [[] for _ in cameras]
    for track_id, track in enumerate(tracks):
        target = np.asarray(track["point"], dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("triangulated targets must be finite 3D points")
        if len(track["camera_indices"]) != len(track["pixels"]):
            raise ValueError("track camera/pixel lengths differ")
        view_indices = track["camera_indices"]
        if (len(view_indices) < 2 or len(set(view_indices)) != len(view_indices)
                or any(isinstance(index, (bool, np.bool_))
                       or not isinstance(index, (int, np.integer)) for index in view_indices)):
            raise ValueError("each track requires at least two distinct integer camera indices")
        reprojection_error = track.get("reprojection_error_px", 0.0)
        if not np.isfinite(reprojection_error) or reprojection_error < 0:
            raise ValueError("track reprojection error must be finite and nonnegative")
        for camera_index, pixel in zip(track["camera_indices"], track["pixels"]):
            if camera_index < 0 or camera_index >= len(cameras):
                raise ValueError("track references a camera outside the training set")
            pixel = np.asarray(pixel, dtype=np.float64)
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise ValueError("track pixels must be finite 2D coordinates")
            observations[camera_index].append((track_id, pixel, target))
    best = {}
    association_count = 0
    visible_counts = []
    for camera_index, camera in enumerate(cameras):
        projected, indices, scale = projected_cloud_index(points, camera, max_image_size)
        visible_counts.append(int(len(indices)))
        if not len(indices) or not observations[camera_index]:
            continue
        query_pixels = np.asarray([item[1] for item in observations[camera_index]]) * scale
        distances, neighbors = cKDTree(projected).query(query_pixels, k=1)
        for observation, distance, neighbor in zip(observations[camera_index], distances, neighbors):
            if distance > match_radius_px:
                continue
            track_id, _, target = observation
            point_index = int(indices[neighbor])
            score = float(distance + scale * tracks[track_id].get("reprojection_error_px", 0.0))
            candidate = (score, track_id, camera_index, target, float(distance))
            if point_index not in best or candidate[:3] < best[point_index][:3]:
                best[point_index] = candidate
            association_count += 1
    indices = np.asarray(sorted(best), dtype=np.int64)
    targets = np.asarray([best[index][3] for index in indices], dtype=np.float64).reshape(-1, 3)
    metadata = {
        "association": "training_pixel_nearest_point_zbuffer",
        "association_radius_matching_pixels": float(match_radius_px),
        "matching_max_image_size": int(max_image_size),
        "visible_mvs_points_per_camera": visible_counts,
        "accepted_observations": association_count,
        "unique_source_candidates": int(len(indices)),
        "duplicate_source_policy": "lowest_pixel_association_plus_reprojection_error",
    }
    return points[indices], targets, indices, metadata


def correct_mvs_geometry(pcd, cameras, *, mode="tps", max_controls=256,
                         smoothing=1e-3, max_image_size=1600, max_features=8000,
                         ratio=0.75, reprojection_px=2.0,
                         min_angle_deg=1.0, match_radius_px=3.0,
                         local_neighbors=64):
    """Return corrected cloud, JSON-safe diagnostics, and audit control arrays.

    Point count/order/RGB and the ordinary Gaussian initializer remain intact.
    Modes are explicit; failures never silently become an uncorrected run.
    """
    if mode not in ("similarity", "tps", "tps_consistent", "tps_local"):
        raise ValueError("geometry correction must be similarity, tps, tps_consistent, or tps_local")
    consistent = mode in ("tps_consistent", "tps_local")
    if isinstance(max_controls, bool) or not isinstance(max_controls, int) or max_controls < 4:
        raise ValueError("mvs_geometry_max_controls must be an integer >= 4")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("mvs_geometry_smoothing must be finite and nonnegative")
    started = time.perf_counter()
    names = [str(camera.image_name) for camera in cameras]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("geometry correction requires distinct training cameras")
    points = np.asarray(pcd.points)
    if not len(points):
        raise ValueError("cannot correct an empty MVS cloud")
    tracks, matching_metadata = build_training_tracks(
        cameras, max_image_size=max_image_size, max_features=max_features,
        ratio=ratio, reprojection_px=reprojection_px,
        min_triangulation_angle_deg=min_angle_deg,
    )
    association = associate_control_points
    if consistent:
        from utils.mvs_control_consistency import associate_consistent_control_points
        association = associate_consistent_control_points
    source, target, indices, association_metadata = association(
        points, cameras, tracks, max_image_size=max_image_size,
        match_radius_px=match_radius_px,
    )
    if len(source) < 4:
        raise ValueError(
            f"only {len(source)} MVS/triangulation control pairs; need at least 4. "
            "Check training overlap, camera calibration and matching diagnostics. "
            "No uncorrected fallback was used."
        )
    candidate_source, candidate_target = source, target
    selected = select_control_indices(source, max_controls)
    source, target, indices = source[selected], target[selected], indices[selected]
    model = (fit_similarity(source, target) if mode == "similarity"
             else fit_tps(source, target, smoothing=smoothing))
    if mode == "tps_local":
        from utils.local_tps import localize_tps
        model = localize_tps(model, points, source, neighbors=local_neighbors)
    transformed = model.transform(points).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("geometry correction produced nonfinite float32 points")
    displacement = np.linalg.norm(transformed.astype(np.float64) - points, axis=1)
    calibration = hashlib.sha256()
    image_digest = hashlib.sha256()
    for camera in cameras:
        calibration.update(str(camera.image_name).encode("utf-8"))
        for matrix in camera_geometry(camera):
            calibration.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
        image_digest.update(str(camera.image_name).encode("utf-8"))
        image_digest.update(np.asarray(camera.image).tobytes())
    metadata = {
        "mode": mode,
        "implementation": ("twings_inspired_fused_mvs_sift_v2" if consistent
                           else "twings_inspired_fused_mvs_sift_v1"),
        "reference": "https://github.com/sandokim/TWINGS-official",
        "control_source": "same_budgeted_fused_mvs_cloud",
        "target_source": "training_only_sift_triangulation",
        "train_camera_names": names,
        "camera_sha256": calibration.hexdigest(),
        "training_images_sha256": image_digest.hexdigest(),
        "max_controls": max_controls,
        "selected_controls": int(len(source)),
        "smoothing": float(smoothing) if mode != "similarity" else None,
        "matching": matching_metadata,
        "association": association_metadata,
        "fit": model.diagnostics,
        "input_sha256": cloud_digest(points, pcd.colors),
        "output_sha256": cloud_digest(transformed, pcd.colors),
        "point_count": int(len(points)),
        "displacement_median": float(np.median(displacement)),
        "displacement_p95": float(np.quantile(displacement, 0.95)),
        "displacement_max": float(np.max(displacement)),
        "elapsed_seconds": time.perf_counter() - started,
        "scale_rule": "unchanged_knn_recomputed_from_corrected_positions",
    }
    unused = np.ones(len(candidate_source), dtype=bool)
    unused[selected] = False
    if unused.any():
        before = np.linalg.norm(candidate_source[unused] - candidate_target[unused], axis=1)
        after = np.linalg.norm(model.transform(candidate_source[unused]) - candidate_target[unused], axis=1)
        metadata["unused_control_diagnostic"] = {
            "count": int(unused.sum()),
            "rmse_before": float(np.sqrt(np.mean(before ** 2))),
            "rmse_after": float(np.sqrt(np.mean(after ** 2))),
            "note": "training-image matches not fitted; not ground truth or test-view validation",
        }
    arrays = {"source": source, "target": target, "source_indices": indices,
              "transformed_controls": model.transform(source)}
    if consistent:
        from utils.local_tps import sampled_warp_diagnostics
        metadata["warp_diagnostic"] = sampled_warp_diagnostics(model, points)
        metadata["unchanged_point_fraction"] = float(np.mean(np.all(
            transformed == points.astype(np.float32), axis=1)))
        arrays["track_indices"] = np.asarray(
            association_metadata["chosen_track_indices"], dtype=np.int64)[selected]
        # Store the actual observations as well as 3D pairs for independent audit.
        for name in ("camera_indices", "pixels"):
            selected_tracks = [tracks[index] for index in arrays["track_indices"]]
            # Current matcher emits pairwise tracks, hence fixed-size numeric arrays.
            arrays["track_" + name] = np.asarray([track[name] for track in selected_tracks])
    corrected = BasicPointCloud(points=transformed, colors=pcd.colors, normals=pcd.normals)
    return corrected, metadata, arrays
