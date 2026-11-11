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

#### 🔹 LLFF Dataset

To train on a single LLFF scene, use the following command:

```
python train.py -s ${DATASET_PATH} -m ${OUTPUT_PATH} --eval -r 8 --n_views {3 or 6 or 9}
```
To train and evaluate on **all LLFF scenes**, simply run the script below:
```
bash scripts/train_llff.sh
```
#### 🔹 MipNeRF-360 Dataset

To train on a single MipNeRF-360 scene, use the following command:

```
python train.py -s ${DATASET_PATH} -m ${OUTPUT_PATH} --eval -r 8 --n_views {12 or 24}
```
To train and evaluate on **all MipNeRF-360 scenes**, simply run the script below:
```
bash scripts/train_mipnerf360.sh
```

#### 🔹 Blender / NeRF-Synthetic Dataset

Use the NeRF-Synthetic scene folders containing `transforms_train.json`,
`transforms_test.json`, and their images. The script selects the first `N_VIEWS`
of the fixed training indices `[2, 16, 26, 55, 73, 76, 86, 93]`, evaluates every
eighth test frame, and uses the original 10,000-point random initialization and
white background.

```bash
conda activate DropGaussian
N_VIEWS=8 SCENES="lego chair" GPU_ID=0 bash scripts/train_blender.sh
```

#### 🔹 DTU Dataset

Use the prepared `DTU_colmap/<scan>_colmap` folders with `images/`,
`poses_bounds.npy`, COLMAP cameras, and `sparse/0/points3D.ply` or `points3D.bin`.
The script uses the fixed DTU training/test split from `exp/mvsinit-dg-gated`,
initializes from the COLMAP sparse cloud, and evaluates with IDR foreground masks.
Supported training view counts are 3, 6, and 9.

```bash
N_VIEWS=3 SCENES="scan21 scan30" GPU_ID=0 bash scripts/train_dtu.sh
```

Both scripts run training, rendering, and metric aggregation with the original
DropGaussian training code and its fixed random seed 0. Set `DATA_ROOT`,
`OUTPUT_ROOT`, `PYTHON_BIN`, `RESOLUTION`, or `ITERATIONS` to override their
defaults; DTU also accepts `DTU_MASK_ROOT` (default: `$DATA_ROOT/idrmasks`).
Their default outputs are `output/raw/blender8` and `output/raw/dtu3`.
Use `DRY_RUN=1` to check input paths and print commands without starting training.
These scripts consume prepared datasets; they do not run COLMAP preprocessing.

### 🎬 Rendering & Evaluation
You can perform **rendering and evaluation in a single step** using the following command:
#### 🔹 LLFF Dataset
```
python render.py -s -m ${MODEL_PATH} --eval -r 8
```
#### 🔹 MipNeRF-360 Dataset
```
python render.py -s -m ${MODEL_PATH} --eval -r 8
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
