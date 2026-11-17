#!/usr/bin/env python3
"""Read-only geometric audit of a global TPS initialization run.

No GPU, training, target views, or original cache writes are used. A JSON report
is written only to the requested output. Metrics are geometric diagnostics,
not ground-truth geometry accuracy or explanations of PSNR causality.
Only ``tps`` and ``tps_consistent`` runs are supported. Localized runs must use
``diagnose_local_warp.py`` so their support blending is not silently omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.mvs_init import load_mvs_point_cloud
from utils.tps_warp import fit_tps


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values), "min": float(values.min()),
        "p05": float(np.quantile(values, .05)),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "max": float(values.max()),
    }


def jacobian(model, points):
    q = (points - model.source_center) / model.source_scale
    diff = q[:, None, :] - model.normalized_controls[None, :, :]
    radius = np.linalg.norm(diff, axis=2)
    # phi=-r is not differentiable exactly at a control. Exclude such queries
    # from Jacobian summary rather than claim a derivative there.
    valid = np.all(radius > 1e-12, axis=1)
    direction = -diff[valid] / radius[valid, :, None]
    matrices = np.eye(3)[None] + model.affine[1:].T[None]
    matrices = matrices + np.einsum("mcj,co->moj", direction, model.weights)
    return matrices, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=4096)
    args = parser.parse_args()
    if args.sample_count < 4:
        parser.error("sample-count must be >= 4")
    manifest = json.loads((args.run / "manifest.json").read_text())
    correction = json.loads((args.run / "geometry_correction.json").read_text())
    if correction.get("mode") not in ("tps", "tps_consistent"):
        parser.error(
            "diagnose_warp.py supports only global tps/tps_consistent runs; "
            "use diagnose_local_warp.py for tps_local"
        )
    provenance = manifest["initialization_source"]
    with np.load(args.run / "geometry_controls.npz", allow_pickle=False) as arrays:
        source, target = arrays["source"], arrays["target"]
        source_indices = arrays["source_indices"]
    pcd, _, meta = load_mvs_point_cloud(
        manifest["source_path"], n_views=len(correction["train_camera_names"]),
        train_camera_names=correction["train_camera_names"],
        explicit_path=provenance["path"], max_points=provenance["max_points"],
    )
    if meta["selected_sha256"] != correction["input_sha256"]:
        raise ValueError("source cloud hash does not reproduce audited run")
    points = np.asarray(pcd.points, dtype=np.float64)
    if not np.array_equal(points[source_indices], source):
        raise ValueError("control sources are not the recorded source cloud samples")
    model = fit_tps(source, target, smoothing=correction["smoothing"])
    warped = model.transform(points)
    displacement = np.linalg.norm(warped - points, axis=1)
    inside = Delaunay(source).find_simplex(points, tol=1e-10) >= 0
    tree = cKDTree(source)
    control_spacing = tree.query(source, k=2)[0][:, 1]
    distance, nearest = tree.query(points, k=1)
    normalized_distance = distance / control_spacing[nearest]
    sample = np.unique(np.linspace(0, len(points)-1, min(args.sample_count, len(points)), dtype=int))
    matrices, derivative_valid = jacobian(model, points[sample])
    det = np.linalg.det(matrices)
    singular = np.linalg.svd(matrices, compute_uv=False)
    check_points = points[sample][derivative_valid][:64]
    finite_difference = np.empty((len(check_points), 3, 3))
    step = 1e-6 * model.source_scale
    for axis in range(3):
        offset = np.eye(3)[axis] * step
        finite_difference[:, :, axis] = (
            model.transform(check_points + offset) - model.transform(check_points - offset)
        ) / (2 * step)
    derivative_error = float(np.max(np.abs(finite_difference - matrices[:len(check_points)])))
    source_tree, warped_tree = cKDTree(points), cKDTree(warped)
    before, neighbors = source_tree.query(points[sample], k=4)
    after = warped_tree.query(warped[sample], k=4)[0]
    before = np.sqrt(np.mean(before[:, 1:]**2, axis=1))
    after = np.sqrt(np.mean(after[:, 1:]**2, axis=1))
    original_edges = np.linalg.norm(points[neighbors[:, 1:]] - points[sample, None, :], axis=2)
    warped_edges = np.linalg.norm(warped[neighbors[:, 1:]] - warped[sample, None, :], axis=2)
    relative = after / np.maximum(before, np.finfo(float).eps)
    q = (points[sample] - model.source_center) / model.source_scale
    affine_displacement = model.source_scale * (np.column_stack((np.ones(len(q)), q)) @ model.affine)
    actual_displacement = warped[sample] - points[sample]
    weight_norm = np.linalg.norm(model.weights, axis=1)
    target_displacement = np.linalg.norm(target - source, axis=1)
    leverage = []
    for index in np.argsort(weight_norm)[-5:][::-1]:
        keep = np.arange(len(source)) != index
        other = fit_tps(source[keep], target[keep], smoothing=correction["smoothing"])
        delta = np.linalg.norm(other.transform(points[sample]) - warped[sample], axis=1)
        leverage.append({
            "control_index": int(index), "source_cloud_index": int(source_indices[index]),
            "control_displacement": float(target_displacement[index]),
            "normalized_weight_norm": float(weight_norm[index]),
            "source": source[index].tolist(), "target": target[index].tolist(),
            "leave_one_control_out_query_change": stats(delta),
        })
    result = {
        "run": str(args.run.resolve()), "source_hash_verified": True,
        "point_count": len(points), "control_count": len(source), "fit": model.diagnostics,
        "scene_source_axis_extent": np.ptp(points, axis=0).tolist(),
        "control_axis_extent": np.ptp(source, axis=0).tolist(),
        "control_spacing": stats(control_spacing),
        "control_displacement": stats(target_displacement),
        "inside_control_convex_hull_fraction": float(inside.mean()),
        "displacement_all": stats(displacement),
        "displacement_inside_hull": stats(displacement[inside]),
        "displacement_outside_hull": stats(displacement[~inside]),
        "nearest_control_distance": stats(distance),
        "nearest_control_distance_over_nearest_control_spacing": stats(normalized_distance),
        "displacement_over_nearest_control_spacing": stats(displacement / control_spacing[nearest]),
        "jacobian": {
            "sample_count": len(sample), "differentiable_count": int(derivative_valid.sum()),
            "analytic_vs_central_difference_max_abs_error": derivative_error,
            "determinant": stats(det), "orientation_reversal_fraction": float(np.mean(det < 0)),
            "determinant_outside_0p1_to_10_fraction": float(np.mean((det < .1) | (det > 10))),
            "min_singular_value": stats(singular[:, -1]),
            "max_singular_value": stats(singular[:, 0]),
            "inside_hull_reversal_fraction": float(np.mean(det[inside[sample][derivative_valid]] < 0)),
            "outside_hull_reversal_fraction": float(np.mean(det[~inside[sample][derivative_valid]] < 0)),
        },
        "knn_scale_recomputed_ratio": stats(relative),
        "knn_scale_ratio_outside_0p5_to_2_fraction": float(np.mean((relative < .5) | (relative > 2))),
        "fixed_original_neighbor_edge_length_ratio": stats(warped_edges / np.maximum(original_edges, np.finfo(float).eps)),
        "affine_tail": {
            "displacement_matrix": model.affine.tolist(),
            "linear_transform_determinant": float(np.linalg.det(np.eye(3) + model.affine[1:].T)),
            "linear_transform_singular_values": np.linalg.svd(np.eye(3) + model.affine[1:].T, compute_uv=False).tolist(),
            "sample_affine_displacement": stats(np.linalg.norm(affine_displacement, axis=1)),
            "sample_nonaffine_displacement": stats(np.linalg.norm(actual_displacement - affine_displacement, axis=1)),
        },
        "largest_weight_control_leave_one_out": leverage,
        "interpretation_limits": [
            "Hull support does not certify correct geometry or correct control correspondence.",
            "Negative Jacobian determinant detects local orientation reversal, not all global self-intersections.",
            "Metrics do not identify the causal contribution to test PSNR without controlled training ablations.",
            "Scale statistics approximate the ordinary RMS-three-neighbor initializer on a deterministic subset.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
