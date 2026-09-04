#!/usr/bin/env python3
"""Audit complete paired multi-seed MVSInit-DG benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.summarize_mvsdrop_setting import parse_metrics


SETTINGS = (
    ("LLFF", "llff", 3, 10000),
    ("LLFF", "llff", 6, 10000),
    ("LLFF", "llff", 9, 10000),
    ("DTU", "dtu", 3, 30000),
    ("DTU", "dtu", 6, 30000),
    ("DTU", "dtu", 9, 30000),
    ("Mip-NeRF360", "mipnerf360", 12, 10000),
    ("Mip-NeRF360", "mipnerf360", 24, 10000),
    ("Blender", "blender", 8, 7000),
)

SCENES = {
    "llff": ("fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"),
    "dtu": ("scan8", "scan21", "scan30", "scan31", "scan34", "scan38", "scan40", "scan41", "scan45", "scan55", "scan63", "scan82", "scan103", "scan110", "scan114"),
    "mipnerf360": ("bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"),
    "blender": ("chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"),
}

METRICS = ("psnr", "ssim", "lpips")
RESOURCES = ("training_time", "peak_vram", "gaussian_count")


def _summarize_scene_rows(rows: list[dict]) -> dict:
    return {
        "scene_count": len(rows),
        "scenes": rows,
        **{
            name: mean(row[name] for row in rows)
            for name in (*METRICS, *RESOURCES)
        },
        "adaptive_fallback_scenes": [
            row["scene"]
            for row in rows
            if (row.get("initialization_source") or {}).get("mvs_point_count")
            == 0
        ],
    }


def _finite_number(value, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value}")
    return number


def load_setting(
    root: Path,
    *,
    seed: int,
    mode: str,
    tag: str,
    views: int,
    iteration: int,
) -> dict | None:
    setting_root = root / f"seed_{seed}" / mode / f"{tag}{views}"
    rows = []
    for scene in SCENES[tag]:
        scene_root = setting_root / scene
        manifest_path = scene_root / "manifest.json"
        metrics_path = scene_root / f"metrics_{iteration}.txt"
        if not manifest_path.is_file() or not metrics_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            return None
        expected_init = "colmap_dense" if mode == "dropgaussian" else "mvs_anywhere"
        checks = {
            "seed": int(manifest.get("seed", -1)) == seed,
            "mask_seed": int(manifest.get("mask_seed", -1)) == seed,
            "views": int(manifest.get("n_views", -1)) == views,
            "iterations": int(manifest.get("iterations", -1)) == iteration,
            "loss": manifest.get("metricboost_mode") == "dropgaussian",
            "initialization": manifest.get("init_mode") == expected_init,
        }
        if not all(checks.values()):
            raise ValueError(f"protocol mismatch in {manifest_path}: {checks}")
        metrics = parse_metrics(metrics_path)
        rows.append(
            {
                "scene": scene,
                **metrics,
                "training_time": _finite_number(
                    manifest["training_time"], label="training_time"
                ),
                "peak_vram": _finite_number(
                    manifest["peak_vram"], label="peak_vram"
                ),
                "gaussian_count": _finite_number(
                    manifest["gaussian_count"], label="gaussian_count"
                ),
                "initialization_source": manifest.get("initialization_source"),
            }
        )
    return _summarize_scene_rows(rows)


def collect(
    root: Path,
    seeds: tuple[int, ...],
    *,
    max_time_ratio: float = 1.5,
    max_vram_ratio: float = 2.0,
    max_gaussian_ratio: float = 2.0,
) -> tuple[list[dict], dict]:
    rows = []
    complete_deltas = []
    for dataset, tag, views, iteration in SETTINGS:
        row = {
            "dataset": dataset,
            "tag": tag,
            "views": views,
            "evaluation_route": "trained_pair",
        }
        seed_deltas = []
        for seed in seeds:
            baseline = load_setting(
                root,
                seed=seed,
                mode="dropgaussian",
                tag=tag,
                views=views,
                iteration=iteration,
            )
            method = load_setting(
                root,
                seed=seed,
                mode="mvsdrop",
                tag=tag,
                views=views,
                iteration=iteration,
            )
            if baseline is None or method is None:
                continue
            delta = {
                "seed": seed,
                **{
                    f"baseline_{metric}": baseline[metric]
                    for metric in METRICS
                },
                **{
                    f"method_{metric}": method[metric]
                    for metric in METRICS
                },
                **{
                    f"delta_{metric}": method[metric] - baseline[metric]
                    for metric in METRICS
                },
                **{
                    f"ratio_{resource}": method[resource] / baseline[resource]
                    for resource in RESOURCES
                },
            }
            seed_deltas.append(delta)
        row["complete_seeds"] = len(seed_deltas)
        row["required_seeds"] = len(seeds)
        complete = len(seed_deltas) == len(seeds)
        row["complete"] = complete
        for metric in METRICS:
            for label in ("baseline", "method"):
                row[f"{label}_{metric}"] = (
                    mean(item[f"{label}_{metric}"] for item in seed_deltas)
                    if complete
                    else None
                )
            row[f"delta_{metric}"] = (
                mean(item[f"delta_{metric}"] for item in seed_deltas)
                if complete
                else None
            )
            row[f"delta_{metric}_std"] = (
                stdev(item[f"delta_{metric}"] for item in seed_deltas)
                if complete and len(seed_deltas) > 1
                else (0.0 if complete else None)
            )
        for resource in RESOURCES:
            row[f"ratio_{resource}"] = (
                mean(item[f"ratio_{resource}"] for item in seed_deltas)
                if complete
                else None
            )
        if complete:
            improved = (
                int(row["delta_psnr"] > 0)
                + int(row["delta_ssim"] > 0)
                + int(row["delta_lpips"] < 0)
            )
            row["improved_metric_count"] = improved
            row["setting_pass"] = improved >= 2
            row["resource_pass"] = (
                row["ratio_training_time"] <= max_time_ratio
                and row["ratio_peak_vram"] <= max_vram_ratio
                and row["ratio_gaussian_count"] <= max_gaussian_ratio
            )
            complete_deltas.append(row)
        else:
            row["improved_metric_count"] = None
            row["setting_pass"] = None
            row["resource_pass"] = None
        rows.append(row)

    all_settings_complete = len(complete_deltas) == len(SETTINGS)
    audit = {
        "requested_seeds": list(seeds),
        "policy": "scene_adaptive_reliability",
        "resource_ratio_limits": {
            "training_time": max_time_ratio,
            "peak_vram": max_vram_ratio,
            "gaussian_count": max_gaussian_ratio,
        },
        "complete_setting_count": len(complete_deltas),
        "all_settings_complete": all_settings_complete,
        "passing_setting_count": (
            sum(bool(row["setting_pass"]) for row in complete_deltas)
            if all_settings_complete
            else None
        ),
        "average_psnr_delta": (
            mean(row["delta_psnr"] for row in complete_deltas)
            if all_settings_complete
            else None
        ),
    }
    audit["all_resource_gates_pass"] = (
        all(bool(row["resource_pass"]) for row in complete_deltas)
        if all_settings_complete
        else None
    )
    audit["goal_pass"] = (
        audit["passing_setting_count"] >= 6
        and audit["average_psnr_delta"] >= 0.25
        and audit["all_resource_gates_pass"]
        if all_settings_complete
        else None
    )
    return rows, audit


def _fmt(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown(rows: list[dict], audit: dict) -> str:
    method_name = "MVSInit-DG-Adaptive"
    lines = [
        "| Dataset | Views | Method | Seeds | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for label, name in (
            ("baseline", "DropGaussian"),
            ("method", method_name),
        ):
            lines.append(
                "| {dataset} | {views} | {name} | {complete_seeds}/{required_seeds} | {psnr} | {ssim} | {lpips} |".format(
                    dataset=row["dataset"],
                    views=row["views"],
                    name=name,
                    complete_seeds=row["complete_seeds"],
                    required_seeds=row["required_seeds"],
                    psnr=_fmt(row[f"{label}_psnr"], 3),
                    ssim=_fmt(row[f"{label}_ssim"], 4),
                    lpips=_fmt(row[f"{label}_lpips"], 4),
                )
            )
    lines.extend((
        "",
        "| Dataset | Views | Seeds | ΔPSNR | ΔSSIM | ΔLPIPS | Time × | VRAM × | Gaussians × | ≥2 metrics | Resources |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ))
    for row in rows:
        lines.append(
            "| {dataset} | {views} | {complete_seeds}/{required_seeds} | {psnr} | {ssim} | {lpips} | {time} | {vram} | {gaussians} | {passed} | {resources} |".format(
                dataset=row["dataset"],
                views=row["views"],
                complete_seeds=row["complete_seeds"],
                required_seeds=row["required_seeds"],
                psnr=_fmt(row["delta_psnr"]),
                ssim=_fmt(row["delta_ssim"]),
                lpips=_fmt(row["delta_lpips"]),
                time=_fmt(row["ratio_training_time"], 3),
                vram=_fmt(row["ratio_peak_vram"], 3),
                gaussians=_fmt(row["ratio_gaussian_count"], 3),
                passed=("yes" if row["setting_pass"] else "no")
                if row["setting_pass"] is not None
                else "—",
                resources=("yes" if row["resource_pass"] else "no")
                if row["resource_pass"] is not None
                else "—",
            )
        )
    lines.extend(("", "```json", json.dumps(audit, indent=2), "```", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/mvsdrop/multiseed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--max-time-ratio", type=float, default=1.5)
    parser.add_argument("--max-vram-ratio", type=float, default=2.0)
    parser.add_argument("--max-gaussian-ratio", type=float, default=2.0)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise SystemExit("--seeds must be unique non-negative integers")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    limits = (
        args.max_time_ratio,
        args.max_vram_ratio,
        args.max_gaussian_ratio,
    )
    if not all(math.isfinite(value) and value >= 1.0 for value in limits):
        raise SystemExit("resource ratio limits must be finite and at least 1.0")
    rows, audit = collect(
        root,
        seeds,
        max_time_ratio=args.max_time_ratio,
        max_vram_ratio=args.max_vram_ratio,
        max_gaussian_ratio=args.max_gaussian_ratio,
    )
    with (root / "multiseed_matrix_adaptive.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "multiseed_audit_adaptive.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = markdown(rows, audit)
    (root / "multiseed_matrix_adaptive.md").write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
