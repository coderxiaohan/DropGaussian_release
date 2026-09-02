#!/usr/bin/env python3
"""Create the requested DropGaussian versus MVSInit-DG result matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_mvsdrop_setting import parse_metrics


MATRIX = (
    ("LLFF", "llff", 3),
    ("LLFF", "llff", 6),
    ("LLFF", "llff", 9),
    ("DTU", "dtu", 3),
    ("DTU", "dtu", 6),
    ("DTU", "dtu", 9),
    ("Mip-NeRF360", "mipnerf360", 12),
    ("Mip-NeRF360", "mipnerf360", 24),
    ("Blender", "blender", 8),
)


EXPECTED_SCENES = {
    "llff": 8,
    "dtu": 15,
    "mipnerf360": 7,
    "blender": 8,
}


def parse_complete_metrics(directory: Path, tag: str) -> dict[str, float]:
    """Return aggregate metrics only for a complete benchmark scene set."""
    metrics_path = directory / "metrics_mean.txt"
    summary_path = directory / "metrics_summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scene_count = int(summary["scene_count"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}
    if scene_count != EXPECTED_SCENES[tag]:
        return {}
    return parse_metrics(metrics_path)


def collect_matrix(root: Path, baseline_root: Path | None = None) -> list[dict]:
    rows = []
    for dataset, tag, n_views in MATRIX:
        row = {"dataset": dataset, "views": n_views}
        for label, directory in (("baseline", "dropgaussian"), ("method", "mvsdrop")):
            local_result_dir = root / directory / f"{tag}{n_views}"
            if label == "baseline" and baseline_root:
                values = parse_complete_metrics(
                    baseline_root / f"{tag}{n_views}", tag
                )
                # A separate baseline root may contain only previously completed
                # cells.  Fall back to the matched runs beside this matrix.
                if not values:
                    values = parse_complete_metrics(local_result_dir, tag)
            else:
                values = parse_complete_metrics(local_result_dir, tag)
            for metric in ("psnr", "ssim", "lpips"):
                row[f"{label}_{metric}"] = values.get(metric)
        for metric in ("psnr", "ssim", "lpips"):
            baseline = row[f"baseline_{metric}"]
            method = row[f"method_{metric}"]
            row[f"delta_{metric}"] = (
                float(method) - float(baseline)
                if baseline is not None and method is not None
                else None
            )
        rows.append(row)
    return rows


def _number(value, digits):
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Dataset | Views | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        for label, name in (
            ("baseline", "DropGaussian"),
            ("method", "MVSInit-DG"),
        ):
            lines.append(
                "| {dataset} | {views} | {name} | {psnr} | {ssim} | {lpips} |".format(
                    dataset=row["dataset"],
                    views=row["views"],
                    name=name,
                    psnr=_number(row[f"{label}_psnr"], 3),
                    ssim=_number(row[f"{label}_ssim"], 4),
                    lpips=_number(row[f"{label}_lpips"], 4),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/mvsdrop/final"))
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Optional directory containing baseline llff3, dtu3, ... roots",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline_root = args.baseline_root.resolve() if args.baseline_root else None
    rows = collect_matrix(root, baseline_root)
    with (root / "benchmark_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = markdown_table(rows)
    (root / "benchmark_matrix.md").write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
