#!/usr/bin/env python3
"""Evaluate rendered DTU test views with the StereoGS IDR-mask protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lpipsPyTorch import lpips
from utils.dataset_splits import DTU_TEST_INDICES
from utils.image_utils import psnr
from utils.loss_utils import ssim


MASK_SUBDIRECTORY_SCENES = {"scan40", "scan55", "scan63", "scan110", "scan114"}


def normalize_scene_name(value: str) -> str:
    name = Path(value).name
    return name[: -len("_colmap")] if name.endswith("_colmap") else name


def resolve_mask_path(mask_root: Path, scene: str, test_index: int) -> Path:
    scene = normalize_scene_name(scene)
    root = mask_root / scene
    if scene in MASK_SUBDIRECTORY_SCENES:
        root = root / "mask"
    return root / f"{test_index:03d}.png"


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


def evaluate(args) -> dict:
    model_path = Path(args.model_path).resolve()
    method_root = model_path / "test" / f"ours_{args.iteration}"
    render_paths = sorted((method_root / "renders").glob("*.png"))
    gt_paths = sorted((method_root / "gt").glob("*.png"))
    if len(render_paths) != len(DTU_TEST_INDICES) or len(gt_paths) != len(
        DTU_TEST_INDICES
    ):
        raise ValueError(
            "StereoGS DTU evaluation expects "
            f"{len(DTU_TEST_INDICES)} renders/GTs, found "
            f"{len(render_paths)}/{len(gt_paths)} below {method_root}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = normalize_scene_name(args.scene or model_path.name)
    mask_root = Path(args.mask_root).resolve()
    per_view = []
    for render_path, gt_path, test_index in zip(
        render_paths, gt_paths, DTU_TEST_INDICES
    ):
        render = load_rgb(render_path, device)
        gt = load_rgb(gt_path, device)
        if render.shape != gt.shape:
            raise ValueError(f"render/GT shape mismatch: {render_path} and {gt_path}")
        mask_path = resolve_mask_path(mask_root, scene, test_index)
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing DTU IDR mask: {mask_path}")
        with Image.open(mask_path) as image:
            mask_image = image.convert("RGB").resize(
                (render.shape[-1], render.shape[-2])
            )
            mask = to_tensor(mask_image).unsqueeze(0).to(device)
        masked_render = render * mask + (1.0 - mask)
        masked_gt = gt * mask + (1.0 - mask)
        values = {
            "test_index": int(test_index),
            "render": render_path.name,
            "mask": str(mask_path),
            "PSNR": float(psnr(masked_render, masked_gt, mask=mask).item()),
            "SSIM": float(ssim(masked_render, masked_gt).item()),
            "LPIPS": float(lpips(masked_render, masked_gt, net_type="vgg").item()),
        }
        per_view.append(values)

    summary = {
        name: sum(view[name] for view in per_view) / len(per_view)
        for name in ("PSNR", "SSIM", "LPIPS")
    }
    payload = {
        "dataset": "DTU",
        "protocol": "StereoGS fixed split + IDR foreground masks",
        "scene": scene,
        "iteration": args.iteration,
        "summary": summary,
        "per_view": per_view,
    }
    json_path = model_path / f"metrics_{args.iteration}_dtu.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metrics_path = model_path / f"metrics_{args.iteration}.txt"
    metrics_path.write_text(
        "PSNR : {:>12.7f}\nSSIM : {:>12.7f}\nLPIPS : {:>12.7f}\n".format(
            summary["PSNR"], summary["SSIM"], summary["LPIPS"]
        ),
        encoding="utf-8",
    )
    print(metrics_path.read_text(encoding="utf-8"), end="")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--iteration", type=int, default=30000)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
