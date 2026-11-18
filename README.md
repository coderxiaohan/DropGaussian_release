<h1 align="center">DropGaussian: Structural Regularization<br>for Sparse-view Gaussian Splatting</h1>

Official Pytorch implementation [**"DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting"**](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.pdf)
[Hyunwoo Park](https://github.com/HWP97?tab=repositories), [Gun Ryu](https://github.com/jerry-ryu), and [Wonjun Kim](https://sites.google.com/view/dcvl) (Corresponding Author) <br>
🎸***IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)***, Jun. 2025.🎸

[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/EU6rPxXFV5k)
<p align="center"><img src='figures/Fig2.gif'></p>
<p align="center"><img src='figures/Fig1.jpg'></p>
<p align="center">[ Training pipeline ]</p>

## :eyes: Overview 
We propose a simple yet powerful regularization technique, **DropGaussian**, for neural rendering with sparse input views.

By randomly eliminating Gaussians during the training process, DropGaussian gives the opportunity for the remaining Gaussians to be more visible with larger gradients, which make them to meaningfully contribute to the optimization process of 3DGS.
This is fairly desirable to alleviate the overfitting problem occurring in sparse-view conditions.

We provide:

- 🚀 **Minimal plug-and-play code snippet** for quick integration
- ✅ **Full implementation** of DropGaussian

## 🚀 Quick Snippet

Here's a minimal example of how to use `DropGaussian` in your training loop:

```python
import torch
# (Assume the rest of the 3DGS pipeline is already set up)
# Create initial compensation factor (1 for each Gaussian)
compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

# Apply DropGaussian with compensation
drop_rate = 0.1
d = torch.nn.Dropout(p=drop_rate)
compensation = d(compensation)

# Apply to opacity
opacity = opacity * compensation[:, None]
```
## ✅ Full implementation

## Geometry-complementary initialization + density control

The current working implementation on `exp/mvs-vcd-vcp` has two modules:
**geometry-complementary MVS initialization** and **multi-view density
control** (EGD/TBP; the existing CLI names remain `vcd`/`vcp`). The density
controller is adapted from FastGS ideas, not an exact upstream reproduction.
DropGaussian dropout, the photometric objective, learning rates, camera splits,
and the densification event schedule are unchanged.

The main entry points are `scripts/mvs_complement/train_<dataset>.sh`.
For Blender, the main protocol now retains the original 10,000-point random
DropGaussian initialization and applies the same EGD/TBP density control, without
MVS. LLFF, DTU, and Mip-NeRF360 still use geometry-complementary initialization.
`scripts/mvs_vcd_vcp/` is the direct-MVS comparison, not the main initialization
method. TPS/TWINGS geometry correction and OAS surface-covariance experiments,
their CLI options, and their dedicated scripts/tests have been removed.

Adaptive MVS reliability scoring, counterfactual calibration, anchor-based
admission, baseline-preserving residual fusion, and confidence-dependent
Gaussian attributes are not part of this branch. The historical
`docs/research/mvsdrop_report.md` describes an earlier adaptive method, not the
current implementation. The earlier adaptive working state was preserved in Git stash
`0f821ef35ee65c3953462a62104dfd7af4255c63`; do not apply it on top of this branch
unless you intend to restore the adaptive implementation.

### Main initialization: preserve geometry, fill coverage gaps

With `--mvs_complement_mode coverage`, the initializer retains reconstructed
baseline geometry and adds MVS candidates supported by at least two training
views, prioritizing regions not covered by that geometry. Positions and colors
are not warped; ordinary Gaussian scale, rotation, and opacity initialization
is retained. Random or unknown-source base clouds are not treated as reliable
reconstructions.

**Blender initialization exception:** for the main complement modes (`coverage`,
`concat`, or `balanced`), Blender/NeRF-Synthetic input uses the original random
initializer: 10,000 points in `[-1, 1]^3`, original colors, and ordinary Gaussian
attributes. Existing `points3d.ply` files are neither used as a reconstruction
nor overwritten. No MVS point/depth cache is loaded and no MVS points are added.
DropGaussian training and EGD/TBP are unchanged. This is an explicit experimental
protocol choice, not a learned reliability gate or a guarantee of baseline PSNR;
report it in experimental settings rather than claiming MVS is used on all four
datasets. `Scene` also enforces this behavior for direct `train.py` calls with
Blender-format input and an enabled complement mode.

The Blender launcher writes new runs to
`output/mvs_complement/blender_random/blender<N>/` (or below the supplied
`OUTPUT_ROOT`). MVS complement budgets do not apply to this random initializer.
Initialization metadata records the actual base/MVS counts and effective mode.
The saved results under `coverage/keep_base/blender8` and
`docs/results/2026-11-18/` predate this change and are still direct-MVS runs;
they must not be relabeled as random-initialization results.

For other input formats without reconstructed base geometry, the existing
direct-MVS fallback remains unchanged. On paths that actually use MVS, missing
MVS caches remain errors, not silent fallbacks.

The default `MVS_COMPLEMENT_TOTAL_POINTS=0` keeps every base point and adds up
to `MVS_COMPLEMENT_MAX_POINTS=100000` MVS points. A positive total budget can
subsample the base cloud; use equal positive budgets when testing whether gains
come from selection rather than point count. `concat` and `balanced` remain as
initialization ablations on the real datasets, and `none` explicitly selects the
direct-MVS comparison on every dataset, including Blender.
`coverage` and `balanced` require compatible depth caches. Reused baseline
geometry assets have not been independently audited for training-view-only
provenance.

### Comparison initialization: direct MVS

For an explicit direct-MVS run (`--mvs_complement_mode none`), `Scene` loads the
cached MVS point cloud and **replaces** the original initialization point cloud.
All datasets, including Blender, use the same comparison policy. The Gaussian initializer uses the ordinary nearest-neighbor
isotropic scales, identity rotations, and uniform opacity 0.1. There is no
reliability weighting or automatic fallback to the baseline.

Defaults for direct `train.py` invocation are:

```text
--init_mode mvs_anywhere --mvs_init_strategy replace
--mvs_init_max_points 100000 --density_control vcd_vcp
```

Only `replace` is accepted for a fresh MVS initialization; `auto`, `adaptive`,
and `augment` are unavailable. The point cap is a uniform resource limit, not
a learned or confidence-dependent threshold. Original dataset/SfM assets are
still loaded for cameras and protocol bookkeeping, but their point cloud is
not appended to the MVS initializer.

Matching MVS caches must already exist; the training scripts **do not run the
MVSAnywhere network**. The loader searches
`MVS*_<N>_views_*_init/mvs_input.ply` below the scene, or uses
`MVS_INIT_PATH`. It rejects known training-view mismatches and missing/empty
caches. Historical caches without camera metadata are still accepted, with
`camera_provenance_verified=false`; supply provenance metadata for an audited
comparison. Nonfinite points are removed. No additional reliability filtering
is applied to the valid cache points.

### Density control: EGD and TBP (`vcd` / `vcp` in code)

This is the project's independent adaptation of
[FastGS](https://github.com/fastgs/FastGS) and its
[sparse-view implementation](https://github.com/fastgs/FastGS/tree/fast-dropgaussian),
**not an exact reproduction** of their CUDA scoring rules.

- **EGD / `vcd`:** keep the original gradient/size criterion for clone/split, but also
  require support in high-error regions of at least two distinct training
  views. A Gaussian's compositing mass must be more concentrated in the
  high-error mask than the image-wide mask area. This identifies refinement
  opportunities; it does not prove that every proposed split will improve PSNR.
- **TBP / `vcp`:** propose points with low contribution in every observed training
  view, requiring at least two observations; usefulness in any view vetoes a
  proposal. Remove the candidate batch in a read-only render proxy, then accept
  only if the current `L1 + lambda_dssim * (1 - SSIM)` loss does not increase
  in any training view, up to floating-point roundoff.

Diagnostics use all distinct **training** cameras with dropout disabled only
for those diagnostic renders. Per-point contribution is the rasterizer's
`alpha * transmittance` mass, measured through a detached color-gradient probe.
The normal training pass still uses DropGaussian. Original low-opacity/size
cleanup remains separate and is not covered by VCP's acceptance check.
No FastGS optimizer, renderer replacement, AbsGS gradients, or loss changes are
imported.

Shared controls are `--density_min_views 2`,
`--vcd_error_quantile 0.75`, and `--vcp_contribution_quantile 0.10`.
These are configurable hyperparameters, not learned quantities.
Additional diagnostic rendering has a runtime cost: neither FastGS-equivalent
speed nor improved held-out metrics is guaranteed.

### Run the four datasets

Use the `DropGaussian` environment. GPU IDs below are examples, not a claim
about free devices:

```bash
N_VIEWS=3 GPU_ID=0 bash scripts/mvs_complement/train_llff.sh
N_VIEWS=3 GPU_ID=1 bash scripts/mvs_complement/train_dtu.sh
N_VIEWS=12 GPU_ID=2 bash scripts/mvs_complement/train_mipnerf360.sh
N_VIEWS=8 GPU_ID=3 bash scripts/mvs_complement/train_blender.sh
```

Each command runs all configured scenes once, then renders and summarizes;
DTU uses foreground-mask evaluation. Real-dataset outputs default to
`output/mvs_complement/coverage/keep_base/<dataset><views>/<scene>/`;
Blender main-method outputs use `output/mvs_complement/blender_random/blender8/<scene>/`.
For MVS runs, changing the complement mode or total budget changes the
intermediate output directories. Existing nonempty main-method settings are rejected; use a fresh
`OUTPUT_ROOT` for another experiment. To run the direct-MVS comparison, use the
corresponding `scripts/mvs_vcd_vcp/` command instead.

| Dataset | Configured scenes | Training view counts | Downsampling | Iterations |
| --- | ---: | --- | ---: | ---: |
| LLFF | 8 | 3 / 6 / 9 | 8 | 10,000 |
| DTU | 15 | 3 / 6 / 9 | 4 | 10,000 |
| Mip-NeRF360 | 7 | 12 / 24 | 8 | 10,000 |
| Blender | 8 | 8 | 1 | 10,000 |

`N_VIEWS`, `GPU_ID`, `SCENES`, `DATA_ROOT`, `RESOLUTION`, `ITERATIONS`,
`PYTHON_BIN`, `OUTPUT_ROOT`, `DTU_MASK_ROOT`, `MVS_INIT_PATH`, and
`MVS_INIT_MAX_POINTS` remain available. On MVS paths, every requested view count
requires its corresponding MVS cache; the main Blender random path does not.
The direct-MVS launchers force the replacement
policy even if inherited variables or extra CLI flags select another initializer.

Preview all nine settings (91 scene-setting runs, one method):

```bash
GPU_LLFF=0 GPU_DTU=1 GPU_MIPNERF360=2 GPU_BLENDER=3 \
  bash scripts/mvs_complement/train_all_views.sh --dry-run
```

Remove `--dry-run` to execute **sequentially**. GPU priority is
`GPU_<DATASET>_<VIEWS>` > `GPU_<DATASET>` > `GPU_ID` > 0.
Different GPU IDs do not enable parallel or distributed training. A batch
`OUTPUT_ROOT` is a parent directory with separate dataset/view subdirectories.
Do not share one dataset-specific `DATA_ROOT`, `SCENES`, or `MVS_INIT_PATH`
across a mixed-dataset batch.

### Module ablations and evaluation

Keep `scripts/mvs_complement/` as the complete-method entry point. The dedicated
[`scripts/ablation/`](scripts/ablation/README.md) launchers fix module choices and
separate results; they do not modify the training algorithms:

| Directory under `scripts/ablation/` | Complementary MVS initialization | EGD | TBP |
| --- | --- | --- | --- |
| `baseline/` | No | No | No |
| `complement_only/` | Yes | No | No |
| `egd_only/` | No | Yes | No |
| `tbp_only/` | No | No | Yes |
| `complement_egd/` | Yes | Yes | No |
| `complement_tbp/` | Yes | No | Yes |
| `egd_tbp/` | No | Yes | Yes |

Each directory contains `train_llff.sh`, `train_dtu.sh`, `train_mipnerf360.sh`,
`train_blender.sh`, and `train_all_views.sh`. Results default to
`output/ablation/<variant>/<dataset><views>/<scene>/`. An explicit `OUTPUT_ROOT`
is a common parent and still receives the variant and setting suffixes. Existing
nonempty settings are rejected. Both single-setting and matrix launchers support
`--dry-run`; matrix launchers also accept `MATRIX` and per-setting GPU mappings.

```bash
# All LLFF scenes at three views; EGD and TBP are both disabled.
GPU_ID=0 N_VIEWS=3 bash scripts/ablation/complement_only/train_llff.sh
# Preview every setting for the EGD-only ablation, without training.
bash scripts/ablation/egd_only/train_all_views.sh --dry-run
```

These are fixed experiments: leave `MODE`, `DENSITY_CONTROL`, and
`MVS_COMPLEMENT_MODE` unset (conflicting values are rejected), and use the
appropriate directory instead. The original DropGaussian regularization and
native density operations are retained. Complementary initialization is not
direct MVS replacement. On Blender every ablation preserves the complete
method's original random initialization: `complement_only` equals `baseline`,
`complement_egd` equals `egd_only`, `complement_tbp` equals `tbp_only`, and
`egd_tbp` equals the complete-method configuration. Do not treat these as
distinct MVS initialization experiments on Blender.

Baseline and module-disable switches remain only as experimental controls;
there is no adaptive MVS implementation to enable.

```bash
# Main initializer, selection and point-budget controls
MVS_COMPLEMENT_MODE=concat bash scripts/mvs_complement/train_llff.sh
MVS_COMPLEMENT_MODE=balanced bash scripts/mvs_complement/train_llff.sh
MVS_COMPLEMENT_TOTAL_POINTS=100000 bash scripts/mvs_complement/train_llff.sh
# Direct MVS comparison at the same positive point budget
MVS_COMPLEMENT_MODE=none MVS_COMPLEMENT_TOTAL_POINTS=100000 \
  bash scripts/mvs_complement/train_llff.sh
# Direct MVS, original density control
DENSITY_CONTROL=none bash scripts/mvs_vcd_vcp/train_llff.sh
# Direct MVS + one density module
DENSITY_CONTROL=vcd bash scripts/mvs_vcd_vcp/train_llff.sh
DENSITY_CONTROL=vcp bash scripts/mvs_vcd_vcp/train_llff.sh
# Original initialization + VCD/VCP, no added MVSAnywhere
bash scripts/dropgaussian_vcd_vcp/train_llff.sh
# Original DropGaussian initialization and density control
bash scripts/train_llff.sh
# Main Blender protocol: random initialization + EGD/TBP (no MVS cache required)
GPU_ID=0 N_VIEWS=8 bash scripts/mvs_complement/train_blender.sh
# Historical direct-MVS Blender comparison remains available explicitly
bash scripts/mvs_vcd_vcp/train_blender.sh
```

Density variants use separate `output/mvs_<mode>/` or
`output/dropgaussian_<mode>/` roots. Always use a fresh `OUTPUT_ROOT` when
changing other hyperparameters: completed outputs may otherwise be skipped.
`scripts/vcd_vcp/` and `scripts/train_mvsinit_<dataset>.sh` are direct-MVS
compatibility entries on this branch, not adaptive entries.
`scripts/mvs/` retains direct MVS with original density control as an ablation.

The paired workflow in `scripts/all/` now compares original DropGaussian with
direct MVS + VCD/VCP, using matching splits, resolution, iterations, and recorded
random state. It runs once per method/scene/setting (182 runs), not multiple
trials:

```bash
bash scripts/all/run_all.sh --dry-run
# Requires a clean committed checkout before actual paired evaluation:
GPU_ID=0 bash scripts/all/run_all.sh
```

Some helper filenames retain `mvsdrop` for compatibility, but `MODE=mvsdrop`
is rejected. Supported per-setting modes are `mvs`, `dropgaussian`, and
`both`. Historical adaptive results must not be reused as direct-MVS results.
Paper/report files have not been rewritten or presented as new experimental
evidence.

Enabled runs record `density_control.jsonl` and a `manifest.json` containing
initialization provenance, density settings/events, protocol, wall-clock time,
peak VRAM, and final point count. Checkpoint formats remain unchanged: repeat
the same density settings on resume. Checkpoint training still constructs the
initial scene before restoring its state, so keep the initialization cache;
rendering an already trained PLY does not load an MVS cache.

### 📦 Installation
We provide an installation using Conda package and environment management:
```
git clone https://github.com/DCVL-3D/DropGaussian_release
cd DropGaussian_release
conda env create --file environment.yaml
conda activate DropGaussian
```

**Note:** This Conda environment assumes that **CUDA 12.1** is already installed on your system.

### 🗂️ Data Preparation

In the data preparation stage, we first reconstruct sparse-view inputs using **Structure-from-Motion (SfM)** with the provided camera poses from the datasets. Then, we perform dense stereo matching using COLMAP’s `patch_match_stereo` function, followed by `stereo_fusion` to generate the dense stereo point cloud.

<details>
<summary><strong> Setup Instructions</strong></summary>

```bash
mkdir dataset
cd dataset

# Download LLFF dataset
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g

# Generate sparse point cloud using COLMAP (limited views) for LLFF
python tools/colmap_llff.py

# Download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# Generate sparse point cloud using COLMAP (limited views) for MipNeRF-360
python tools/colmap_360.py
```

We also provide preprocessed sparse and dense point clouds for convenience.
You can download them via the link below:

👉 [Download Preprocessed Point Clouds](https://drive.google.com/drive/folders/1P3I9m_HU0jF50qwxIIhXhegOVk-kihdI?usp=sharing)
</details>

### 🧪 Training

The geometry-complementary commands and all-view matrix above are the recommended
entries for this branch. Original `scripts/train_<dataset>.sh` commands are
baseline controls and explicitly disable the two density modules. The original
COLMAP preprocessing below is still relevant to baseline assets and camera
loading; direct MVS experiments additionally require the precomputed MVS cache.

### 🎬 Rendering & Evaluation
You can perform **rendering and evaluation in a single step** using the following command:
#### 🔹 LLFF Dataset
```
python render.py -m ${MODEL_PATH} --eval -r 8
```
#### 🔹 MipNeRF-360 Dataset
```
python render.py -m ${MODEL_PATH} --eval -r 8
```

## License

This project is licensed under the **Apache License 2.0**, with the exception of certain components derived from the [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) project.

- **Apache License 2.0**: All original code written for DropGaussian is released under the Apache 2.0 license. See [LICENSE](./LICENSE).
- **Non-commercial License (Inria & MPII)**: Some parts of the code are based on Gaussian Splatting, which is licensed for **non-commercial research use only**. See [LICENSE_GAUSSIAN_SPLATTING.md](./LICENSE_GAUSSIAN_SPLATTING.md) for full terms.

Please ensure that you comply with both licenses when using this repository.

## Acknowledgments
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2023-NR076462) and Institute of Information Communications Technology Planning Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2018-0-00207, RS-2018-II180207, Immersive Media Research Laboratory).

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[GraphDeco-INRIA/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)  </br>
[VITA-Group/FSGS](https://github.com/VITA-Group/FSGS)  </br>

## Citation
If you find our work useful for your project, please consider citing the following paper.
```
@inproceedings{park2025dropgaussian,
  title={Dropgaussian: Structural regularization for sparse-view gaussian splatting},
  author={Park, Hyunwoo and Ryu, Gun and Kim, Wonjun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21600--21609},
  year={2025}
}
```
