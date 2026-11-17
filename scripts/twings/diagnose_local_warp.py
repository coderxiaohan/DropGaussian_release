#!/usr/bin/env python3
"""Reconstruct a saved localized TPS and compare matched-sample derivatives.

Reads the run's train-only controls and verified source MVS cache. Reproduces
the source and corrected float32 cloud hashes before doing any diagnostics.
Writes only a new JSON report; never changes cached geometry or run metadata.
No rendered images, test targets, GPU, or training are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.local_tps import localize_tps, sampled_warp_diagnostics
from utils.mvs_geometry import cloud_digest
from utils.mvs_init import load_mvs_point_cloud
from utils.tps_warp import _normalization, fit_tps


def summarize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values), "min": float(values.min()),
        "p05": float(np.quantile(values, .05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "max": float(values.max()),
    }


def matched_diagnostics(model, query, step, active):
    jacobians = np.empty((len(query), 3, 3), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for axis in range(3):
            offset = np.eye(3)[axis] * step
            plus, minus = query + offset, query - offset
            if np.any(plus[:, axis] == query[:, axis]) or np.any(minus[:, axis] == query[:, axis]):
                raise ValueError("finite-difference step is below coordinate precision")
            jacobians[:, :, axis] = (model.transform(plus) - model.transform(minus)) / (2 * step)
    determinants = np.linalg.det(jacobians)
    singular = np.linalg.svd(jacobians, compute_uv=False)
    if not np.isfinite(determinants).all() or not np.isfinite(singular).all():
        raise ValueError("nonfinite matched-sample derivatives")
    reversal = determinants <= 0
    return {
        "sample_count": len(query),
        "determinant": summarize(determinants),
        "determinant_nonpositive_count": int(reversal.sum()),
        "determinant_nonpositive_fraction": float(reversal.mean()),
        "same_local_support_active_sample_count": int(active.sum()),
        "determinant_nonpositive_count_in_same_local_support": int(reversal[active].sum()),
        "determinant_nonpositive_fraction_in_same_local_support": (
            float(reversal[active].mean()) if active.any() else None
        ),
        "determinant_in_same_local_support": summarize(determinants[active]),
        "smallest_singular_value": summarize(singular[:, -1]),
        "largest_singular_value": summarize(singular[:, 0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=4096)
    args = parser.parse_args()
    if args.sample_count < 4:
        parser.error("sample-count must be >= 4")
    if args.output.exists():
        parser.error("output already exists; choose a new audit report path")
    correction = json.loads((args.run / "geometry_correction.json").read_text())
    if correction.get("mode") != "tps_local":
        parser.error("diagnose_local_warp.py requires a tps_local run")
    manifest = json.loads((args.run / "manifest.json").read_text())
    provenance = manifest["initialization_source"]
    with np.load(args.run / "geometry_controls.npz", allow_pickle=False) as arrays:
        controls, target = arrays["source"], arrays["target"]
        source_indices = arrays["source_indices"]
        saved_transformed_controls = arrays["transformed_controls"]
    pcd, _, source_metadata = load_mvs_point_cloud(
        manifest["source_path"], n_views=len(correction["train_camera_names"]),
        train_camera_names=correction["train_camera_names"],
        explicit_path=provenance["path"], max_points=provenance["max_points"],
    )
    if source_metadata["selected_sha256"] != correction["input_sha256"]:
        raise ValueError("source cloud hash differs from the saved run")
    points = np.asarray(pcd.points, dtype=np.float64)
    if not np.array_equal(points[source_indices], controls):
        raise ValueError("control sources differ from the saved source-cloud samples")
    smoothing = correction["smoothing"]
    neighbors = correction["fit"]["neighbors_requested"]
    base = fit_tps(controls, target, smoothing=smoothing)
    localized = localize_tps(base, points, controls, neighbors=neighbors)
    transformed = localized.transform(points).astype(np.float32)
    output_hash = cloud_digest(transformed, pcd.colors)
    if output_hash != correction["output_sha256"]:
        raise ValueError("localized float32 output hash differs from the saved run")
    control_error = float(np.max(np.abs(localized.transform(controls) - saved_transformed_controls)))
    if not np.allclose(localized.transform(controls), saved_transformed_controls, rtol=1e-12, atol=1e-12):
        raise ValueError("reconstructed control outputs differ from the saved run")
    support = localized.support(points)
    _, radius, _ = _normalization(points)
    step = radius * 1e-5
    distances, _ = cKDTree(controls).query(points, k=1)
    candidates = np.flatnonzero(distances > 2 * step)
    if not len(candidates):
        parser.error("no differentiable non-control samples for a matched comparison")
    selected = candidates[np.linspace(0, len(candidates)-1, min(args.sample_count, len(candidates)), dtype=int)]
    query = points[selected]
    active = support[selected] > 0
    result = {
        "run": str(args.run.resolve()),
        "source_cloud_hash_verified": correction["input_sha256"],
        "localized_output_hash_verified": output_hash,
        "reconstructed_control_max_abs_error": control_error,
        "point_count": len(points), "control_count": len(controls),
        "smoothing": smoothing, "local_neighbors": neighbors,
        "full_cloud": {
            "support_nonzero_count": int((support > 0).sum()),
            "support_nonzero_fraction": float((support > 0).mean()),
            "support_gt_half_fraction": float((support > .5).mean()),
            "support": summarize(support),
            "float32_changed_point_fraction": float(np.any(transformed != pcd.points, axis=1).mean()),
            "localized_displacement": summarize(np.linalg.norm(transformed.astype(float) - points, axis=1)),
        },
        "matched_sampling": {
            "method": "deterministic_linspace_after_excluding_control_neighborhoods",
            "sample_count": len(selected),
            "excluded_near_control_count": len(points) - len(candidates),
            "sample_indices_sha256": hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest(),
            "derivative_method": "central_difference_of_complete_transform_including_support_blend",
            "step": step, "relative_cloud_rms_step": 1e-5,
            "local_support_active_count": int(active.sum()),
            "local_support_active_fraction": float(active.mean()),
        },
        "matched_global_tps": matched_diagnostics(base, query, step, active),
        "matched_localized_tps": matched_diagnostics(localized, query, step, active),
        "current_module_global_diagnostic": sampled_warp_diagnostics(base, points, max_samples=args.sample_count),
        "current_module_local_diagnostic": sampled_warp_diagnostics(localized, points, max_samples=args.sample_count),
        "reconstructed_localization_diagnostics": localized.diagnostics,
        "interpretation_limits": [
            "Global and local comparisons use exactly the same source controls and sample indices.",
            "Overall reversal rates include identity regions; active-only rates condition on the same LOCAL support for both warps.",
            "Compact support limits affected geometry but its spatial gradient can introduce local orientation reversals.",
            "Finite sampling cannot prove global injectivity; determinant positivity alone is also insufficient.",
            "This is a geometry audit, not ground-truth correctness or a causal explanation of rendering quality.",
            "Historical run metadata is preserved; current-module diagnostics are recorded only in this new report.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "point_count": len(points),
        "source_and_output_hashes_verified": True,
        "full_support_fraction": result["full_cloud"]["support_nonzero_fraction"],
        "global_fold_fraction": result["matched_global_tps"]["determinant_nonpositive_fraction"],
        "local_fold_fraction": result["matched_localized_tps"]["determinant_nonpositive_fraction"],
        "global_fold_fraction_in_local_support": result["matched_global_tps"]["determinant_nonpositive_fraction_in_same_local_support"],
        "local_fold_fraction_in_local_support": result["matched_localized_tps"]["determinant_nonpositive_fraction_in_same_local_support"],
    }, indent=2))


if __name__ == "__main__":
    main()
