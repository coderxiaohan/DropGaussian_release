"""Train-view-only controls for an optional TWINGS-inspired geometry correction.

This is deliberately *not* a reproduction of TWINGS' MASt3R matcher.  It uses
CPU SIFT correspondences and the supplied training-camera calibration to obtain
independent triangulated targets.  It never opens a COLMAP point/track file,
discovers additional images, or changes a random-number generator's state.

Coordinates returned by this module refer to ``camera.image`` pixels, before
the training renderer's image-resolution reduction.  The quality thresholds
below are ordinary configurable matching/triangulation parameters, not
dataset-specific decisions or guarantees of correct geometry.
"""

import copy
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


def geometry_cameras_with_intrinsics(cameras, intrinsics):
    """Copy training cameras with correctly decoded COLMAP intrinsics.

    ``CameraInfo.uid`` is the COLMAP *intrinsic camera id*, not image id.
    Only the supplied intrinsic records are read.  Training camera objects,
    images, extrinsics, FoVs and the renderer remain unchanged.  In particular,
    off-center principal points in PINHOLE calibration are preserved rather
    than replaced with a centered FoV approximation.

    SIMPLE_RADIAL is treated as a pinhole camera, matching the existing MVS
    cache-generation convention.  Its ignored coefficient is explicitly
    recorded; this helper does not undistort images or refit camera poses.
    """
    corrected = []
    records = []
    supported_lengths = {"SIMPLE_PINHOLE": 3, "SIMPLE_RADIAL": 4, "PINHOLE": 4}
    for camera in cameras:
        uid = getattr(camera, "uid", None)
        if uid not in intrinsics:
            raise ValueError(f"Training camera intrinsic id {uid!r} is absent from COLMAP calibration.")
        record = intrinsics[uid]
        model = str(getattr(record, "model", ""))
        if model not in supported_lengths:
            raise ValueError(f"Unsupported COLMAP geometry camera model {model!r} for id {uid}.")
        parameters = np.asarray(record.params, dtype=np.float64)
        if parameters.shape != (supported_lengths[model],) or not np.isfinite(parameters).all():
            raise ValueError(f"Invalid or non-finite {model} calibration parameters for id {uid}.")
        width, height = record.width, record.height
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0
               for value in (width, height)):
            raise ValueError(f"Invalid calibration image dimensions for id {uid}.")
        width, height = int(width), int(height)
        if model == "PINHOLE":
            fx, fy, cx, cy = parameters
        else:
            fx, cx, cy = parameters[:3]
            fy = fx
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Non-positive focal length in COLMAP calibration for id {uid}.")
        calibrated_k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        original_k = np.asarray(camera.K, dtype=np.float64)
        if original_k.shape != (3, 3) or not np.isfinite(original_k).all():
            raise ValueError(f"Training camera has malformed/non-finite original K for id {uid}.")
        # CameraInfo is a NamedTuple. A shallow-copy fallback also supports
        # standalone camera fixtures without importing scene/CUDA modules.
        if hasattr(camera, "_replace"):
            geometry_camera = camera._replace(K=calibrated_k, width=width, height=height)
        else:
            geometry_camera = copy.copy(camera)
            geometry_camera.K, geometry_camera.width, geometry_camera.height = calibrated_k, width, height
        actual_k, _ = camera_geometry(geometry_camera)
        changed = (not np.array_equal(original_k, calibrated_k)
                   or camera.width != width or camera.height != height)
        records.append({
            "image_name": str(getattr(camera, "image_name", "")),
            "intrinsic_camera_id": int(uid),
            "model": model,
            "original_K": original_k.tolist(),
            "geometry_K": calibrated_k.tolist(),
            "actual_image_K": actual_k.tolist(),
            "original_calibration_image_size": [int(camera.width), int(camera.height)],
            "calibration_image_size": [width, height],
            "changed": bool(changed),
            "distortion_policy": "pinhole_approximation_no_undistortion" if model == "SIMPLE_RADIAL" else "none",
            "ignored_radial_coefficient": float(parameters[3]) if model == "SIMPLE_RADIAL" else None,
        })
        corrected.append(geometry_camera)
    return corrected, {
        "source": "colmap_camera_model_parameters",
        "camera_count": len(corrected),
        "changed_camera_count": sum(record["changed"] for record in records),
        "training_camera_objects_modified": False,
        "cameras": records,
    }


def _image_size(camera):
    image = camera.image
    if isinstance(image, Image.Image):
        return image.size
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("Training camera image must be a PIL image or HWC array.")
    return int(array.shape[1]), int(array.shape[0])


def camera_geometry(camera):
    """Return ``(K, [R|t])`` in actual training-image pixel coordinates.

    CameraInfo stores the transpose of the world-to-camera rotation.  Its K
    and width/height can describe an original COLMAP image while its loaded
    image is a downsampled copy, so the two intrinsic rows must be rescaled.
    """
    width, height = _image_size(camera)
    original_width, original_height = int(camera.width), int(camera.height)
    if min(width, height, original_width, original_height) <= 0:
        raise ValueError("Camera image and calibration dimensions must be positive.")
    intrinsics = np.array(camera.K, dtype=np.float64, copy=True)
    rotation = np.asarray(camera.R, dtype=np.float64).T
    translation = np.asarray(camera.T, dtype=np.float64).reshape(-1)
    if intrinsics.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected camera K/R matrices of size 3x3 and a 3-vector T.")
    if not all(np.isfinite(value).all() for value in (intrinsics, rotation, translation)):
        raise ValueError("Camera calibration contains non-finite values.")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or abs(np.linalg.det(intrinsics)) < 1e-15:
        raise ValueError("Camera intrinsics are singular or have invalid focal lengths.")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("Camera R is not an orthogonal rotation.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("Camera R is not a proper rotation.")
    intrinsics[0] *= width / original_width
    intrinsics[1] *= height / original_height
    return intrinsics, np.column_stack((rotation, translation))


def _mask_array(value, size):
    """Normalize an explicitly supplied foreground/alpha mask to uint8."""
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("Training camera mask must have shape HxW or HxWxC.")
    if not np.isfinite(array).all():
        raise ValueError("Training camera mask contains non-finite values.")
    # The convention is nonzero=foreground for both alpha and supplied masks.
    mask = Image.fromarray((array > 0).astype(np.uint8) * 255)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask)


def _image_and_mask(camera):
    value = camera.image
    image = value if isinstance(value, Image.Image) else Image.fromarray(np.asarray(value))
    image.load()
    masks = []
    if "A" in image.getbands():
        masks.append(_mask_array(image.getchannel("A"), image.size))
    else:
        # Synthetic readers may already have composited alpha into RGB.  Only
        # this explicitly supplied training image is inspected; no directory
        # search or held-out image is allowed here.
        path = getattr(camera, "image_path", None)
        if path and Path(path).is_file():
            with Image.open(path) as original:
                if "A" in original.getbands():
                    masks.append(_mask_array(original.getchannel("A"), image.size))
    mask = getattr(camera, "mask", None)
    if mask is not None:
        masks.append(_mask_array(mask, image.size))
    foreground = None
    if masks:
        foreground = np.logical_and.reduce([item > 0 for item in masks]).astype(np.uint8) * 255
    return np.asarray(image.convert("RGB")), foreground


def _extract_features(camera, cv2, max_image_size, max_features):
    rgb, mask = _image_and_mask(camera)
    original_height, original_width = rgb.shape[:2]
    ratio = min(1.0, max_image_size / max(original_height, original_width))
    width = max(1, round(original_width * ratio))
    height = max(1, round(original_height * ratio))
    if (width, height) != (original_width, original_height):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(gray, mask)
    pixels = np.asarray([point.pt for point in keypoints], dtype=np.float64).reshape(-1, 2)
    # Use independent resize ratios because integer rounding can make them
    # slightly different.  OpenCV pixel centers map by (p+0.5)/s-0.5.
    if len(pixels):
        pixels = (pixels + 0.5) * np.array([original_width / width, original_height / height]) - 0.5
    return pixels, descriptors, {
        "image_name": str(getattr(camera, "image_name", "")),
        "image_size": [original_width, original_height],
        "matching_size": [width, height],
        "features": int(len(pixels)),
        "foreground_mask_used": mask is not None,
    }


def _mutual_ratio_matches(first, second, cv2, ratio):
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def accepted(query, target):
        result = {}
        for neighbors in matcher.knnMatch(query, target, k=2):
            if len(neighbors) == 2 and neighbors[0].distance < ratio * neighbors[1].distance:
                result[int(neighbors[0].queryIdx)] = int(neighbors[0].trainIdx)
        return result

    forward = accepted(first, second)
    backward = accepted(second, first)
    return [(i, j) for i, j in sorted(forward.items()) if backward.get(j) == i]


def triangulate_correspondences(
    first_geometry, second_geometry, first_pixels, second_pixels,
    reprojection_px=2.0, min_triangulation_angle_deg=1.0,
):
    """DLT triangulation with cheirality, reprojection and parallax checks.

    Returns arrays ``(points, valid, reprojection_errors, angles_deg)``.  Each
    point corresponds to an input match; invalid points must never be used as
    controls.  Reprojection errors are the maximum error of the two views.
    The angle test uses min(angle, 180-angle), rejecting both parallel and
    antiparallel rays; no scene-dependent world-distance cutoff is used.
    """
    if not np.isfinite(reprojection_px) or reprojection_px <= 0:
        raise ValueError("reprojection_px must be finite and positive.")
    if not np.isfinite(min_triangulation_angle_deg) or not 0 < min_triangulation_angle_deg < 90:
        raise ValueError("min_triangulation_angle_deg must lie strictly between 0 and 90.")
    first_pixels = np.asarray(first_pixels, dtype=np.float64)
    second_pixels = np.asarray(second_pixels, dtype=np.float64)
    if first_pixels.shape != second_pixels.shape or first_pixels.ndim != 2 or first_pixels.shape[1] != 2:
        raise ValueError("Correspondence coordinates must have equal Nx2 shapes.")
    count = len(first_pixels)
    points = np.full((count, 3), np.nan, dtype=np.float64)
    errors = np.full(count, np.inf, dtype=np.float64)
    angles = np.zeros(count, dtype=np.float64)
    valid = np.isfinite(first_pixels).all(axis=1) & np.isfinite(second_pixels).all(axis=1)
    if not valid.any():
        return points, valid, errors, angles
    first_k, first_rt = first_geometry
    second_k, second_rt = second_geometry
    first_p, second_p = first_k @ first_rt, second_k @ second_rt
    ids = np.flatnonzero(valid)
    a, b = first_pixels[ids], second_pixels[ids]
    equations = np.stack((
        a[:, 0, None] * first_p[2] - first_p[0],
        a[:, 1, None] * first_p[2] - first_p[1],
        b[:, 0, None] * second_p[2] - second_p[0],
        b[:, 1, None] * second_p[2] - second_p[1],
    ), axis=1)
    _, _, vt = np.linalg.svd(equations)
    homogeneous = vt[:, -1]
    finite = np.abs(homogeneous[:, 3]) > np.finfo(np.float64).eps
    valid[ids[~finite]] = False
    ids, homogeneous = ids[finite], homogeneous[finite]
    points[ids] = homogeneous[:, :3] / homogeneous[:, 3, None]
    valid &= np.isfinite(points).all(axis=1)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return points, valid, errors, angles
    xyz = points[ids]
    first_camera = xyz @ first_rt[:, :3].T + first_rt[:, 3]
    second_camera = xyz @ second_rt[:, :3].T + second_rt[:, 3]
    positive = (first_camera[:, 2] > 0) & (second_camera[:, 2] > 0)
    valid[ids[~positive]] = False
    ids = ids[positive]
    if not len(ids):
        return points, valid, errors, angles
    first_h = first_camera[positive] @ first_k.T
    second_h = second_camera[positive] @ second_k.T
    first_error = np.linalg.norm(first_h[:, :2] / first_h[:, 2, None] - first_pixels[ids], axis=1)
    second_error = np.linalg.norm(second_h[:, :2] / second_h[:, 2, None] - second_pixels[ids], axis=1)
    errors[ids] = np.maximum(first_error, second_error)
    center_a = -first_rt[:, :3].T @ first_rt[:, 3]
    center_b = -second_rt[:, :3].T @ second_rt[:, 3]
    ray_a, ray_b = points[ids] - center_a, points[ids] - center_b
    norm_product = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
    cosine = np.sum(ray_a * ray_b, axis=1) / np.maximum(norm_product, np.finfo(np.float64).tiny)
    angles[ids] = np.degrees(np.arccos(np.clip(np.abs(cosine), 0, 1)))
    valid[ids] &= (errors[ids] <= reprojection_px) & (angles[ids] >= min_triangulation_angle_deg)
    return points, valid, errors, angles


def build_training_tracks(
    cameras, max_image_size=1600, max_features=8000, ratio=0.75,
    reprojection_px=2.0, min_triangulation_angle_deg=1.0,
):
    """Match and triangulate all pairs from the *provided training list only*.

    Returns pairwise tracks (not merged global SfM tracks) and JSON-compatible
    diagnostics.  No feature/matching caches are written.  Callers determine
    whether the resulting spatial distribution is sufficient for TPS fitting.
    An empty set is an error, not a silent substitution of baseline geometry.
    """
    cameras = list(cameras)
    if len(cameras) < 2:
        raise ValueError("Train-view triangulation requires at least two training cameras.")
    if int(max_image_size) != max_image_size or max_image_size <= 0:
        raise ValueError("max_image_size must be a positive integer.")
    if int(max_features) != max_features or max_features < 2:
        raise ValueError("max_features must be an integer of at least two.")
    if not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("SIFT Lowe ratio must lie strictly between 0 and 1.")
    # Validate triangulation settings before running feature extraction.
    if not np.isfinite(reprojection_px) or reprojection_px <= 0:
        raise ValueError("reprojection_px must be finite and positive.")
    if not np.isfinite(min_triangulation_angle_deg) or not 0 < min_triangulation_angle_deg < 90:
        raise ValueError("min_triangulation_angle_deg must lie strictly between 0 and 90.")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Geometry correction requires OpenCV with SIFT support.") from exc
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("Installed OpenCV has no SIFT_create; SIFT support is required.")
    geometries = [camera_geometry(camera) for camera in cameras]
    features = [_extract_features(camera, cv2, int(max_image_size), int(max_features)) for camera in cameras]
    metadata = {
        "method": "train_only_sift_mutual_ratio_calibrated_dlt",
        "matcher_is_twings_mast3r": False,
        "opencv_version": str(cv2.__version__),
        "max_image_size": int(max_image_size),
        "max_features": int(max_features),
        "ratio": float(ratio),
        "reprojection_px": float(reprojection_px),
        "min_triangulation_angle_deg": float(min_triangulation_angle_deg),
        "cameras": [item[2] for item in features],
        "pairs": [],
    }
    tracks = []
    for first, second in combinations(range(len(cameras)), 2):
        matches = _mutual_ratio_matches(features[first][1], features[second][1], cv2, ratio)
        pair = {"camera_indices": [first, second], "mutual_matches": len(matches), "accepted_tracks": 0}
        metadata["pairs"].append(pair)
        if not matches:
            continue
        matches = np.asarray(matches, dtype=np.int64)
        first_pixels = features[first][0][matches[:, 0]]
        second_pixels = features[second][0][matches[:, 1]]
        points, valid, errors, angles = triangulate_correspondences(
            geometries[first], geometries[second], first_pixels, second_pixels,
            reprojection_px, min_triangulation_angle_deg,
        )
        pair["accepted_tracks"] = int(valid.sum())
        if valid.any():
            pair["median_reprojection_error_px"] = float(np.median(errors[valid]))
            pair["median_triangulation_angle_deg"] = float(np.median(angles[valid]))
        for index in np.flatnonzero(valid):
            tracks.append({
                "camera_indices": [first, second],
                "pixels": [first_pixels[index].tolist(), second_pixels[index].tolist()],
                "point": points[index].tolist(),
                "reprojection_error_px": float(errors[index]),
                "triangulation_angle_deg": float(angles[index]),
            })
    metadata["track_count"] = len(tracks)
    if not tracks:
        raise RuntimeError(
            "No valid train-view triangulated controls: check image overlap, camera "
            "calibration, masks, matching settings and triangulation thresholds. "
            f"Feature counts={[item[2]['features'] for item in features]}; "
            f"mutual matches={sum(pair['mutual_matches'] for pair in metadata['pairs'])}."
        )
    return tracks, metadata
