"""Camera geometry shared by direct MVS and geometry-complement initialization.

These helpers decode the supplied training-camera calibration and map it to
loaded image coordinates without changing the training/rendering cameras.
"""

import copy

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
