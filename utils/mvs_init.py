"""Load a precomputed MVS cloud for direct Gaussian initialization.

Cache provenance checks prevent known training/test-view leakage. They do not
score geometry quality or gate MVS against the baseline initialization.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData

from utils.graphics_utils import BasicPointCloud


_RESOLUTION_PATTERN = re.compile(r"_H_(\d+)_W_(\d+)_")


def resolve_mvs_init_strategy(requested: str) -> str:
    """Allow direct replacement only; adaptive/fusion modes are absent."""

    if requested != "replace":
        raise ValueError(
            "this branch supports only mvs_init_strategy='replace'; "
            "adaptive, auto, and augment initialization are not available"
        )
    return "replace"


def _resolution_area(path: Path) -> int:
    match = _RESOLUTION_PATTERN.search(path.parent.name)
    return int(match.group(1)) * int(match.group(2)) if match else 0


def _metadata_compatibility(path: Path, train_camera_names: list[str]):
    metadata_path = path.parent / "mvs_cache_meta.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        depth_path = path.parent / "mvs_depths_masks.npz"
        if not depth_path.is_file():
            return None, None
        with np.load(depth_path, allow_pickle=False) as values:
            if "image_names" not in values:
                return None, None
            metadata = {
                "view_names": [str(name) for name in values["image_names"]],
                "provenance_source": "mvs_depths_masks.npz",
            }
    cached_names = [str(name) for name in metadata.get("view_names", [])]
    return cached_names == list(train_camera_names), metadata


def resolve_mvs_init_path(
    source_path: str | Path,
    *,
    n_views: int,
    train_camera_names: list[str],
    explicit_path: str | Path = "",
) -> tuple[Path, dict | None, bool]:
    """Resolve a compatible cache, rejecting provenance-known view leakage."""

    source_path = Path(source_path).resolve()
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "mvs_input.ply"
        candidates = [candidate]
    else:
        candidates = sorted(
            source_path.glob(
                f"MVS*_{int(n_views)}_views_*_init/mvs_input.ply"
            )
        )
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"no MVS {n_views}-view mvs_input.ply below {source_path}"
        )

    verified = []
    unverified = []
    incompatible = []
    for candidate in candidates:
        compatible, metadata = _metadata_compatibility(
            candidate, train_camera_names
        )
        if compatible is True:
            verified.append((candidate, metadata))
        elif compatible is False:
            incompatible.append(candidate)
        else:
            unverified.append((candidate, metadata))
    pool = verified or unverified
    if not pool:
        raise ValueError(
            "all MVS caches were generated from cameras that differ "
            f"from the current training split: {incompatible}"
        )
    candidate, metadata = max(
        pool,
        key=lambda item: (
            item[0].parent.name.startswith("MVSAnywhere_"),
            _resolution_area(item[0]),
        ),
    )
    return candidate, metadata, bool(verified)


def load_mvs_point_cloud(
    source_path: str | Path,
    *,
    n_views: int,
    train_camera_names: list[str],
    explicit_path: str | Path = "",
    max_points: int = 250_000,
):
    if (
        not isinstance(max_points, int)
        or isinstance(max_points, bool)
        or max_points <= 0
    ):
        raise ValueError("mvs_init_max_points must be a positive integer")
    path, cache_metadata, provenance_verified = resolve_mvs_init_path(
        source_path,
        n_views=n_views,
        train_camera_names=train_camera_names,
        explicit_path=explicit_path,
    )
    vertices = PlyData.read(path)["vertex"]
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32) / 255.0
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points, colors = points[finite], colors[finite]
    source_count = int(points.shape[0])
    if source_count == 0:
        raise ValueError(f"MVSAnywhere point cloud is empty: {path}")
    if source_count > max_points:
        generator = np.random.default_rng(0)
        indices = np.sort(generator.choice(source_count, max_points, replace=False))
        points, colors = points[indices], colors[indices]
    digest = hashlib.sha256()
    digest.update(points.tobytes())
    digest.update(colors.tobytes())
    metadata = {
        "path": str(path),
        "backend": path.parent.name.split("_", 1)[0],
        "source_point_count": source_count,
        "selected_point_count": int(points.shape[0]),
        "max_points": int(max_points),
        "selection": (
            "all"
            if source_count <= max_points
            else "seed0_uniform_without_replacement"
        ),
        "selected_sha256": digest.hexdigest(),
        "camera_provenance_verified": provenance_verified,
        "cache_generation_id": (cache_metadata or {}).get("generation_id"),
    }
    return (
        BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points),
        ),
        path,
        metadata,
    )
