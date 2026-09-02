"""Sparse-view protocols shared by scene loading and Nexus flow generation."""

from __future__ import annotations

from collections.abc import Sequence


# StereoGS protocols. Indices are zero-based positions after sorting cameras by
# image name, even though the DTU mask files themselves use one-based names.
DTU_TRAIN_INDICES = (25, 22, 28, 40, 44, 48, 0, 8, 13)
DTU_TEST_INDICES = (
    1,
    2,
    9,
    10,
    11,
    12,
    14,
    15,
    23,
    24,
    26,
    27,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    41,
    42,
    43,
    45,
    46,
    47,
)
NERF_SYNTHETIC_TRAIN_INDICES = (2, 16, 26, 55, 73, 76, 86, 93)
NERF_SYNTHETIC_TEST_HOLD = 8


def take_indices(values: Sequence, indices: Sequence[int], *, label: str) -> list:
    """Select ordered protocol indices and fail clearly on incomplete datasets."""

    if not indices:
        return []
    maximum = max(indices)
    if maximum >= len(values):
        raise ValueError(
            f"{label} requires index {maximum}, but the dataset contains only "
            f"{len(values)} entries"
        )
    return [values[index] for index in indices]


def dtu_split(cameras: Sequence, n_views: int) -> tuple[list, list]:
    if n_views < 1 or n_views > len(DTU_TRAIN_INDICES):
        raise ValueError(
            f"DTU n_views must be in [1, {len(DTU_TRAIN_INDICES)}], got {n_views}"
        )
    train = take_indices(
        cameras, DTU_TRAIN_INDICES[:n_views], label="StereoGS DTU training split"
    )
    test = take_indices(cameras, DTU_TEST_INDICES, label="StereoGS DTU test split")
    return train, test


def nerf_synthetic_split(cameras: Sequence, n_views: int) -> list:
    if n_views < 1 or n_views > len(NERF_SYNTHETIC_TRAIN_INDICES):
        raise ValueError(
            "NeRF-Synthetic n_views must be in "
            f"[1, {len(NERF_SYNTHETIC_TRAIN_INDICES)}], got {n_views}"
        )
    return take_indices(
        cameras,
        NERF_SYNTHETIC_TRAIN_INDICES[:n_views],
        label="StereoGS NeRF-Synthetic training split",
    )


def nerf_synthetic_test_split(cameras: Sequence) -> list:
    return [
        camera
        for index, camera in enumerate(cameras)
        if index % NERF_SYNTHETIC_TEST_HOLD == 0
    ]
