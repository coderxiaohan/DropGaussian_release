"""Load a precomputed, cross-view-consistent MVS initialization."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData

from utils.graphics_utils import BasicPointCloud


_RESOLUTION_PATTERN = re.compile(r"_H_(\d+)_W_(\d+)_")
_AUTO_REPLACE_MAX_POINTS = 100_000
_AUTO_DTU_TOTAL_POINTS = 100_000


def resolve_mvs_init_strategy(
    requested: str,
    *,
    n_views: int,
    is_synthetic: bool,
    dataset_name: str = "auto",
) -> str:
    """Resolve the public initialization policy without looking at test data."""

    if requested not in ("auto", "replace", "augment"):
        raise ValueError(
            "mvs_init_strategy must be 'auto', 'replace', or 'augment'"
        )
    if requested != "auto":
        return requested
    if is_synthetic:
        return "passthrough"
    if str(dataset_name).lower() == "dtu":
        return "augment"
    return "replace" if n_views <= 3 else "augment"


def resolve_mvs_init_point_budget(
    requested_strategy: str,
    resolved_strategy: str,
    configured_max_points: int,
    *,
    dataset_name: str = "auto",
    dense_base: bool = False,
) -> int:
    """Apply the preregistered lightweight budget to auto 3-view replacement."""

    if requested_strategy == "auto" and (
        resolved_strategy == "replace"
        or str(dataset_name).lower() == "dtu"
        or dense_base
    ):
        return min(int(configured_max_points), _AUTO_REPLACE_MAX_POINTS)
    return int(configured_max_points)


def resolve_mvs_fusion_budget(
    requested_strategy: str,
    configured_total_points: int,
    *,
    base_point_count: int,
    dense_base_threshold: int,
    dense_residual_points: int,
    dataset_name: str = "auto",
) -> int:
    """Resolve a fixed, training-input-only residual fusion budget."""

    if requested_strategy != "auto":
        return int(configured_total_points)
    if str(dataset_name).lower() == "dtu":
        return max(int(base_point_count), _AUTO_DTU_TOTAL_POINTS)
    if int(base_point_count) >= int(dense_base_threshold):
        return int(base_point_count) + int(dense_residual_points)
    return int(configured_total_points)


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
    candidate, metadata = max(pool, key=lambda item: _resolution_area(item[0]))
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


def fuse_residual_mvs_point_cloud(
    base: BasicPointCloud,
    mvs: BasicPointCloud,
    *,
    total_max_points: int = 350_000,
) -> tuple[BasicPointCloud, dict]:
    """Preserve a reliable base cloud and fill its point budget with MVS."""

    if (
        not isinstance(total_max_points, int)
        or isinstance(total_max_points, bool)
        or total_max_points <= 0
    ):
        raise ValueError("mvs_init_total_max_points must be a positive integer")

    base_points = np.asarray(base.points, dtype=np.float32)
    base_colors = np.asarray(base.colors, dtype=np.float32)
    mvs_points = np.asarray(mvs.points, dtype=np.float32)
    mvs_colors = np.asarray(mvs.colors, dtype=np.float32)
    if base_points.ndim != 2 or base_points.shape[1] != 3:
        raise ValueError("base point cloud must have shape [N, 3]")
    if base_colors.shape != base_points.shape:
        raise ValueError("base colors must match base point positions")
    base_finite = np.isfinite(base_points).all(axis=1) & np.isfinite(
        base_colors
    ).all(axis=1)
    base_points, base_colors = base_points[base_finite], base_colors[base_finite]
    base_count = int(base_points.shape[0])
    if base_count == 0:
        raise ValueError("base point cloud is empty")

    if base_count >= total_max_points:
        points, colors = base_points, base_colors
        mvs_count = 0
        policy = "base_budget_passthrough"
    else:
        mvs_count = min(int(mvs_points.shape[0]), total_max_points - base_count)
        if mvs_count < mvs_points.shape[0]:
            generator = np.random.default_rng(1)
            mvs_indices = np.sort(
                generator.choice(mvs_points.shape[0], mvs_count, replace=False)
            )
            selected_mvs_points = mvs_points[mvs_indices]
            selected_mvs_colors = mvs_colors[mvs_indices]
        else:
            selected_mvs_points = mvs_points
            selected_mvs_colors = mvs_colors
        points = np.concatenate((base_points, selected_mvs_points), axis=0)
        colors = np.concatenate((base_colors, selected_mvs_colors), axis=0)
        policy = "preserve_base_then_fill_with_mvs"

    digest = hashlib.sha256()
    digest.update(points.tobytes())
    digest.update(colors.tobytes())
    return (
        BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points),
        ),
        {
            "fusion_policy": policy,
            "base_point_count": base_count,
            "mvs_point_count": mvs_count,
            "fused_point_count": int(points.shape[0]),
            "total_max_points": int(total_max_points),
            "fused_sha256": digest.hexdigest(),
        },
    )
