#!/usr/bin/env python3
"""Aggregate completed per-scene runs for one MVSInit-DG setting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


METRIC_PATTERN = re.compile(r"^(PSNR|SSIM|LPIPS)\s*:\s*([-+0-9.eE]+)")


def parse_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if match:
            values[match.group(1).lower()] = float(match.group(2))
    if set(values) != {"psnr", "ssim", "lpips"}:
        raise ValueError(f"incomplete metrics file: {path}")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite metrics file: {path}")
    return values


def collect(experiment_root: Path, iteration: int) -> list[dict]:
    rows = []
    for scene_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        metrics_path = scene_dir / f"metrics_{iteration}.txt"
        manifest_path = scene_dir / "manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        rows.append(
            {
                "scene": scene_dir.name,
                **parse_metrics(metrics_path),
                "gaussians": int(manifest["gaussian_count"]),
                "training_time": float(manifest["training_time"]),
                "peak_vram": int(manifest["peak_vram"]),
                "mode": str(manifest["metricboost_mode"]),
                "init_mode": str(manifest["init_mode"]),
                "n_views": int(manifest["n_views"]),
                "iterations": int(manifest["iterations"]),
            }
        )
    if not rows:
        raise ValueError(f"no completed runs found below {experiment_root}")
    for field in ("mode", "init_mode", "n_views", "iterations"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"experiment root mixes {field} values")
    if rows[0]["iterations"] != iteration:
        raise ValueError("completed iteration differs from requested iteration")
    return rows


def aggregate(rows: list[dict]) -> dict:
    return {
        "scene_count": len(rows),
        "mode": rows[0]["mode"],
        "init_mode": rows[0]["init_mode"],
        "n_views": rows[0]["n_views"],
        "iteration": rows[0]["iterations"],
        **{
            name: mean(row[name] for row in rows)
            for name in (
                "psnr",
                "ssim",
                "lpips",
                "gaussians",
                "training_time",
                "peak_vram",
            )
        },
    }


def write_outputs(root: Path, rows: list[dict], summary: dict) -> None:
    with (root / "metrics_per_scene.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "metrics_mean.txt").write_text(
        "PSNR : {psnr}\nSSIM : {ssim}\nLPIPS : {lpips}\n".format(**summary),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    rows = collect(root, args.iteration)
    summary = aggregate(rows)
    write_outputs(root, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
