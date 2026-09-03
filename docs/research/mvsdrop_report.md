# MVSInit-DG: bounded residual MVS initialization for DropGaussian

## Scope

MVSInit-DG changes only the initial point cloud and, for DTU, applies a fixed
final opacity budget. DropGaussian's renderer, reconstruction loss, optimizer,
dropout schedule, ADC densification, and iteration count remain unchanged.

The design is intentionally small. It borrows the MVS-assisted initialization
direction explored by StereoGS, but it does not add an online stereo network,
depth loss, virtual cameras, or a new CUDA kernel. MVSAnywhere and MVSFormer
point clouds produced offline from the selected training cameras are both
accepted.

Primary references:

- [DropGaussian, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.html)
- [MVSAnywhere, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Izquierdo_MVSAnywhere_Zero-Shot_Multi-View_Stereo_CVPR_2025_paper.html)
- [StereoGS preprint](https://arxiv.org/abs/2606.30545) and [code](https://github.com/StringerYwh00/StereoGS)

## Frozen automatic policy

The policy depends only on the dataset label, training-view count, and input
point statistics. It never reads a held-out image or evaluation metric.

- Blender keeps the released synthetic initializer.
- LLFF 3-view replaces the weak dense-fusion cloud with a deterministic 100k
  subset of the MVS cloud.
- Other real scenes preserve every COLMAP point and add deterministic MVS
  residual points. An input below 200k uses a 350k total budget; an input at or
  above 200k receives only 100k extra MVS points.
- DTU adds a quality gate because its full-scene pilot exposed heterogeneous
  cache quality. Automatic fusion is used only at 3 views when COLMAP contains
  fewer than 10k points and the verified MVS cache contains at least 100k
  candidates. Selected scenes use a 100k total input budget; all other DTU
  scenes follow the exact baseline initialization path.
- Selected DTU scenes keep the 70k highest-opacity Gaussians at the final
  checkpoint. This happens after optimization and before final
  evaluation/saving; the pruning step is disabled when the quality gate rejects
  the MVS prior.

Every cache selection records its path, backend, selected-point hash, and
camera provenance. A cache whose recorded camera names differ from the active
training split is rejected. The resolver looks for
`MVS*_<views>_views_*_init/mvs_input.ply`; `MVS_INIT_PATH` can select an
explicit cache.

## Ablation evidence

These pilot results selected fixed budgets before the final multi-seed matrix.
They are not substituted for missing full-setting results.

LLFF 3-view full-set budget comparison:

| Variant | PSNR | SSIM | LPIPS | Time × | VRAM × | Gaussians × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DropGaussian | 20.5421 | 0.70938 | 0.19951 | 1.000 | 1.000 | 1.000 |
| MVS 250k | 20.7899 | 0.74423 | 0.16458 | 1.069 | 2.398 | 3.361 |
| MVS 100k | **20.9223** | **0.74899** | **0.16388** | 0.855 | 1.385 | 1.765 |

The smaller 100k initializer improves all metrics and is the only tested
replacement budget that passes the 2× VRAM and Gaussian gates.

DTU scan8 showed why COLMAP preservation and final pruning are both needed:

| Variant | ΔPSNR | ΔSSIM | ΔLPIPS | Time × | VRAM × | Gaussians × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Replace 250k | +1.0087 | +0.02033 | -0.03664 | 1.366 | 3.721 | 6.599 |
| Replace 100k | +0.6292 | +0.01581 | -0.02613 | 1.055 | 1.829 | 2.887 |
| Replace 100k, prune 70k | +2.8368 | -0.00481 | -0.00131 | 1.055 | 1.829 | 1.926 |
| Preserve + MVS to 100k, prune 70k | **+3.7203** | **+0.03044** | **-0.04200** | 1.099 | 1.797 | 1.927 |

The selected residual policy also transferred to dense-input pilots:

| Setting / scene | ΔPSNR | ΔSSIM | ΔLPIPS | Time × | VRAM × | Gaussians × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLFF 9 / fern | +0.1485 | +0.00253 | -0.01145 | 1.102 | 1.205 | 1.213 |
| Mip-NeRF360 24 / bonsai | +0.1980 | +0.02597 | -0.03182 | 1.066 | 1.110 | 1.087 |
| Mip-NeRF360 24 / room | +0.3771 | +0.00040 | -0.00401 | 1.057 | 1.092 | 1.090 |

Replacement without COLMAP preservation was rejected on DTU 6/9, and full
MVS replacement was rejected on Blender. Those negative controls are retained
as ablation artifacts but are not part of the final module set.

## Reproduction and audit

Run a single matched pair on one GPU:

```bash
MODE=dropgaussian DATASET=llff N_VIEWS=3 GPU_ID=1 \
  bash scripts/train_mvsdrop_benchmark.sh
MODE=mvsdrop DATASET=llff N_VIEWS=3 GPU_ID=1 \
  bash scripts/train_mvsdrop_benchmark.sh
```

Run three seeds over all nine requested settings:

```bash
SEEDS="0 1 2" GPU_ID=1 bash scripts/run_mvsdrop_multiseed.sh
```

For parallel execution, launch one seed per GPU. Within a seed, do not split
the baseline and method across GPUs: CUDA splatting and densification are not
bit-deterministic. The runner always executes the matched baseline before the
method on the same device.

The audit requires exact scene sets, seed/view/iteration agreement, at least
two improved metrics in 6/9 settings, mean ΔPSNR of at least +0.25 dB, and
resource ratios no larger than 1.5× time, 2× peak allocated VRAM, and 2× final
Gaussian count. Incomplete seeds/settings are shown as missing and never
silently averaged.

Final three-seed measurements are written to
`output/mvsdrop/multiseed/multiseed_matrix.md` and
`output/mvsdrop/multiseed/multiseed_audit.json`. This report does not fill
those cells until the corresponding scene sets have actually completed.

## Limitations

Offline MVS preprocessing time is excluded from training time. The final DTU
opacity budget can trade SSIM/LPIPS for PSNR on individual runs, so only the
predeclared full-setting multi-seed aggregate is used for the final claim.
Camera-provenance checks prevent known split leakage, but an old cache without
any metadata is necessarily marked unverified rather than treated as verified.
