#!/usr/bin/env python3
"""Train-only, CPU correspondence audit for the TWINGS-inspired initializer.

Reproduces the original one-view source association independently of the active
initializer, then compares two-view geometric association. This is a diagnostic,
not a correctness oracle: small reprojection errors do not prove true geometry.
Only selected training images are opened; no SfM 3D points, GT depths or test
images are used. Output must be new so earlier evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.dataset_splits import DTU_TRAIN_INDICES, NERF_SYNTHETIC_TRAIN_INDICES
from utils.mvs_geometry import projected_cloud_index
from utils.mvs_init import load_mvs_point_cloud
from utils.tps_warp import select_control_indices
from utils.train_view_controls import (
    build_training_tracks, camera_geometry, geometry_cameras_with_intrinsics,
)


def training_cameras(source, dataset, n_views, images="images", white_background=True):
    """Load camera metadata and only requested training RGBs, without scene import."""
    source = Path(source)
    if dataset == "blender":
        values = json.loads((source / "transforms_train.json").read_text())
        selected = NERF_SYNTHETIC_TRAIN_INDICES[:n_views]
        if not 1 <= n_views <= len(NERF_SYNTHETIC_TRAIN_INDICES):
            raise ValueError("Blender n_views must follow the existing 1..8 protocol")
        cameras = []
        for index in selected:
            frame = values["frames"][index]
            path = source / (frame["file_path"] + ".png")
            with Image.open(path) as original:
                rgba = np.asarray(original.convert("RGBA"), dtype=np.float64) / 255
            rgb = rgba[..., :3] * rgba[..., 3:] + float(white_background) * (1 - rgba[..., 3:])
            image = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
            width, height = image.size
            focal = width / (2 * np.tan(float(values["camera_angle_x"]) / 2))
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
            c2w[:3, 1:3] *= -1
            rt = np.linalg.inv(c2w)
            cameras.append(SimpleNamespace(
                uid=index, R=rt[:3, :3].T, T=rt[:3, 3],
                K=np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
                image=image, image_path=str(path), image_name=path.stem,
                width=width, height=height, mask=None,
            ))
        return cameras, {"source": "transforms_train.json", "training_images_only": True}

    # Import the file directly: importing scene would load Gaussian/CUDA code.
    spec = importlib.util.spec_from_file_location("diagnostic_colmap_loader", ROOT / "scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    sparse = source / "sparse/0"
    if (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file():
        intrinsics = loader.read_intrinsics_binary(sparse / "cameras.bin")
        extrinsics = loader.read_extrinsics_binary(sparse / "images.bin")
    else:
        intrinsics = loader.read_intrinsics_text(sparse / "cameras.txt")
        extrinsics = loader.read_extrinsics_text(sparse / "images.txt")
    records = sorted(extrinsics.values(), key=lambda record: record.name)
    if dataset == "dtu":
        if not 1 <= n_views <= len(DTU_TRAIN_INDICES):
            raise ValueError("DTU n_views must follow the existing 1..9 protocol")
        selected = list(DTU_TRAIN_INDICES[:n_views])
    else:
        pool = [i for i in range(len(records)) if i % 8 != 0]
        selected = [pool[round(i)] for i in np.linspace(0, len(pool) - 1, n_views)]
        if len(set(selected)) != n_views:
            raise ValueError("Requested n_views exceeds the training pool")
    cameras = []
    for index in selected:
        extrinsic = records[index]
        intrinsic = intrinsics[extrinsic.camera_id]
        path = source / images / Path(extrinsic.name).name
        with Image.open(path) as value:
            image = value.copy()
        # Match the reader's pre-correction record when its legacy indexing is
        # defined, so calibration audit metadata is comparable to the probe.
        legacy_k = np.eye(3)
        if len(intrinsic.params) >= 4:
            legacy_k[0, 0], legacy_k[1, 1], legacy_k[0, 2], legacy_k[1, 2] = intrinsic.params[:4]
        else:
            focal, cx, cy = intrinsic.params[:3]
            legacy_k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        cameras.append(SimpleNamespace(
            uid=intrinsic.id, R=loader.qvec2rotmat(extrinsic.qvec).T,
            T=np.asarray(extrinsic.tvec), K=legacy_k, image=image,
            image_path=str(path), image_name=path.stem,
            width=intrinsic.width, height=intrinsic.height, mask=None,
        ))
    return geometry_cameras_with_intrinsics(cameras, intrinsics)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = {"count": len(values), "nonfinite": int(len(values) - len(finite))}
    if len(finite):
        for name, q in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75),
                        ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1)):
            result[name] = float(np.quantile(finite, q))
        result["rmse"] = float(np.sqrt(np.mean(finite ** 2)))
    return result


def projected_errors(points, geometries, scales, track):
    errors, depths = [], []
    for camera_index, pixel in zip(track["camera_indices"], track["pixels"]):
        k, rt = geometries[camera_index]
        camera_points = points @ rt[:, :3].T + rt[:, 3]
        homogeneous = camera_points @ k.T
        with np.errstate(divide="ignore", invalid="ignore"):
            projection = homogeneous[:, :2] / homogeneous[:, 2:]
        error = np.linalg.norm(projection - np.asarray(pixel), axis=1) * scales[camera_index]
        error[(camera_points[:, 2] <= 0) | ~np.isfinite(error)] = np.inf
        errors.append(error)
        depths.append(camera_points[:, 2])
    return np.stack(errors, axis=1), np.stack(depths, axis=1)


def associate_legacy(points, cameras, tracks, max_image_size, radius):
    """Frozen original association, returning provenance for every source node."""
    indexes = [projected_cloud_index(points, camera, max_image_size) for camera in cameras]
    trees = [cKDTree(item[0]) for item in indexes]
    best = {}
    accepted = 0
    for camera_index, (projected, point_ids, scale) in enumerate(indexes):
        if not len(projected):
            continue
        for track_id, track in enumerate(tracks):
            if camera_index not in track["camera_indices"]:
                continue
            track_index = track["camera_indices"].index(camera_index)
            pixel = np.asarray(track["pixels"][track_index]) * scale
            distance, nearest = trees[camera_index].query(pixel)
            if distance > radius:
                continue
            point_id = int(point_ids[nearest])
            candidate = (float(distance + scale * track["reprojection_error_px"]),
                         track_id, camera_index, float(distance))
            if point_id not in best or candidate[:3] < best[point_id][:3]:
                best[point_id] = candidate
            accepted += 1
    rows = [(point_id, *best[point_id][1:]) for point_id in sorted(best)]
    return rows, indexes, trees, accepted


def associate_both_views(points, tracks, geometries, indexes, trees, radius):
    """Compare union-of-radius candidates by reprojection in all track views."""
    scales = [item[2] for item in indexes]
    best = {}
    tracks_with_candidates = 0
    for track_id, track in enumerate(tracks):
        candidates = set()
        for camera_index, pixel in zip(track["camera_indices"], track["pixels"]):
            local_ids = trees[camera_index].query_ball_point(np.asarray(pixel) * scales[camera_index], radius)
            candidates.update(map(int, indexes[camera_index][1][local_ids]))
        if not candidates:
            continue
        candidate_ids = np.asarray(sorted(candidates), dtype=np.int64)
        errors, _ = projected_errors(points[candidate_ids], geometries, scales, track)
        maximum = errors.max(axis=1)
        valid = maximum <= radius
        if not valid.any():
            continue
        tracks_with_candidates += 1
        candidate_ids, errors, maximum = candidate_ids[valid], errors[valid], maximum[valid]
        chosen = int(np.lexsort((candidate_ids, errors.sum(axis=1), maximum))[0])
        point_id = int(candidate_ids[chosen])
        candidate = (float(maximum[chosen]), float(errors[chosen].sum()), track_id)
        if point_id not in best or candidate < best[point_id]:
            best[point_id] = candidate
    rows = [(point_id, best[point_id][2], -1, best[point_id][0]) for point_id in sorted(best)]
    return rows, tracks_with_candidates


def summarize(rows, points, tracks, geometries, scales, radius, max_controls):
    if not rows:
        return {"count": 0}, {}
    point_ids = np.array([row[0] for row in rows], dtype=np.int64)
    targets = np.array([tracks[row[1]]["point"] for row in rows])
    source = points[point_ids]
    errors = np.array([projected_errors(points[[row[0]]], geometries, scales, tracks[row[1]])[0][0]
                       for row in rows])
    target_depths = np.array([projected_errors(np.array([tracks[row[1]]["point"]]), geometries,
                                              scales, tracks[row[1]])[1][0] for row in rows])
    displacement = np.linalg.norm(source - targets, axis=1)
    selected = select_control_indices(source, max_controls)
    unused = np.ones(len(rows), dtype=bool)
    unused[selected] = False

    def section(indices):
        maximum_error = errors[indices].max(axis=1)
        return {
            "count": len(indices),
            "displacement_world": distribution(displacement[indices]),
            "source_max_reprojection_matching_px": distribution(maximum_error),
            "source_fails_other_view_radius_count": int((maximum_error > radius).sum()),
            "source_exceeds_10px_count": int((maximum_error > 10).sum()),
            "source_exceeds_50px_count": int((maximum_error > 50).sum()),
            "target_reprojection_original_px": distribution(
                [tracks[rows[i][1]]["reprojection_error_px"] for i in indices]),
            "triangulation_angle_deg": distribution(
                [tracks[rows[i][1]]["triangulation_angle_deg"] for i in indices]),
            "target_min_depth_world": distribution(target_depths[indices].min(axis=1)),
        }

    return {"all": section(np.arange(len(rows))), "fps_selected": section(selected),
            "not_selected": section(np.flatnonzero(unused))}, {
        "source_indices": point_ids, "track_ids": np.array([row[1] for row in rows]),
        "association_camera": np.array([row[2] for row in rows]), "source": source,
        "target": targets, "reprojection_errors_matching_px": errors,
        "selected_indices": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", choices=("llff", "dtu", "mipnerf360", "blender"), required=True)
    parser.add_argument("--n-views", type=int, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--mvs-path", default="")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--max-controls", type=int, default=256)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output must be a new directory; earlier diagnostics are never overwritten")
    cameras, calibration = training_cameras(args.source, args.dataset, args.n_views, args.images)
    names = [camera.image_name for camera in cameras]
    pcd, path, cache = load_mvs_point_cloud(args.source, n_views=args.n_views,
        train_camera_names=names, explicit_path=args.mvs_path, max_points=args.max_points)
    if not cache["camera_provenance_verified"]:
        raise ValueError("Diagnostic requires cache provenance matching the selected training cameras")
    tracks, matching = build_training_tracks(cameras, max_image_size=args.max_image_size)
    points = np.asarray(pcd.points, dtype=np.float64)
    geometries = [camera_geometry(camera) for camera in cameras]
    legacy, indexes, trees, accepted = associate_legacy(points, cameras, tracks, args.max_image_size, args.radius)
    scales = [item[2] for item in indexes]
    multiview, supported_tracks = associate_both_views(points, tracks, geometries, indexes, trees, args.radius)
    legacy_report, legacy_arrays = summarize(legacy, points, tracks, geometries, scales, args.radius, args.max_controls)
    multiview_report, multiview_arrays = summarize(multiview, points, tracks, geometries, scales, args.radius, args.max_controls)
    if legacy_arrays:
        valid = legacy_arrays["reprojection_errors_matching_px"].max(axis=1) <= args.radius
        nearest_filter = {
            "remaining_sources": int(valid.sum()),
            "distinct_track_ids": int(len(np.unique(legacy_arrays["track_ids"][valid]))),
        }
    else:
        nearest_filter = {"remaining_sources": 0, "distinct_track_ids": 0}
    source_center = points.mean(axis=0)
    report = {
        "dataset": args.dataset, "source": str(args.source.resolve()), "n_views": args.n_views,
        "train_camera_names": names, "cache": cache, "calibration": calibration,
        "matching": matching, "max_image_size": args.max_image_size, "radius_matching_px": args.radius,
        "cloud_rms_radius": float(np.sqrt(np.mean(np.sum((points - source_center) ** 2, axis=1)))),
        "legacy_accepted_observations": accepted, "legacy_one_view": legacy_report,
        "legacy_nearest_then_all_view_filter": nearest_filter,
        "both_views_supported_tracks": supported_tracks, "both_views_radius_candidates": multiview_report,
        "caveat": "Training-only diagnostics, not ground truth, not proof of correct geometry or rendering improvement.",
    }
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "correspondence_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "training_tracks.json").write_text(json.dumps(tracks, indent=2) + "\n")
        np.savez_compressed(args.output / "legacy_associations.npz", **legacy_arrays)
        np.savez_compressed(args.output / "both_view_associations.npz", **multiview_arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
