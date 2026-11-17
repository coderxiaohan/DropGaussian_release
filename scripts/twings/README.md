# TWINGS-inspired MVS 几何校正实验

这个目录只增加 **Gaussian 初始化前的点云几何校正**，不改 DropGaussian 的训练损失、随机丢弃机制、学习率、训练相机划分，也不改现有 VCD/VCP 的增密和剪枝逻辑。所有脚本复用 `scripts/mvs_vcd_vcp/` 和 `scripts/all/train_mvsdrop_benchmark.sh` 的训练 → 渲染 → 评估 → 汇总流程。

## 做了什么，没做什么

借鉴 [TWINGS 官方项目](https://sandokim.github.io/twings/) 和 [作者代码](https://github.com/sandokim/TWINGS-official) 中“用多视图三角化控制点校正预测几何”的思想。这里是面向本项目的独立实现，**不是原论文的完整复现，也不使用其 MASt3R 匹配器**：

1. 仅在本次指定的训练图像之间进行 SIFT 双向匹配，用已知相机参数三角化控制点；不读取测试图像、GT 深度或完整场景的 SfM 轨迹作为控制点。
2. 将本次已经选定、完成点数截断的 MVS 点云投影到训练图像，通过可见性／像素邻近关联，找到对应的 MVS 源控制点。目标控制点来自上一步的三角化。
3. 拟合全局相似变换或带正则化的三维 TPS，变换同一份 MVS 点云的位置，随后进入原有 Gaussian 初始化和训练。

使用的是融合后的 MVS PLY，**不需要额外的 MVS 深度 NPZ 文件**。源控制点与图像特征的像素关联只是近似对应关系，不应当作真实三维对应关系的保证。纹理弱、重复纹理、遮挡边缘、错误匹配或控制点分布不均可能造成校正失败，甚至降低质量。

校正不增加／删除点，不改变点颜色，也不改变给定的初始化点数预算。Gaussian 的 scale 仍由原有 KNN 距离规则初始化：点位置变化后会按同一规则重新计算距离，**不意味着 scale 数值保持不变**。opacity、SH 和其他初始化规则不变。

匹配筛选、重投影筛选、像素关联半径和 TPS 正则化均包含显式超参数；本实验不声称“无阈值”或“自动保证不退化”。缓存不兼容、控制点不足或几何求解失败时应报错停止，不能把失败的 `tps` 实验静默记成 `none`。

## 可选对照（默认仍运行原三组）

| `MVS_GEOMETRY_CORRECTION` | 初始化操作 | 实验目的 |
|---|---|---|
| `none` | 原来的直接 MVS 初始化，不运行校正 | 确定对照结果 |
| `similarity` | 同一类匹配控制点上的全局相似变换 | 判断简单整体对齐是否已经足够 |
| `tps`，本目录默认 | 正则化三维 TPS 校正 | 检验局部非刚性调整的额外收益 |
| `tps_consistent` | 所有匹配视角一致的源控制点＋原全局 TPS | 单独检验控制点对应修复 |
| `tps_local` | 同上控制点＋紧支撑局部 TPS | 单独检验限制外推的作用 |

所有组都固定 `--init_mode mvs_anywhere --mvs_init_strategy replace --density_control vcd_vcp`。`none` 不是原始 DropGaussian 初始化，而是“直接 MVS＋原有 VCD/VCP”。原始 DropGaussian 基线仍使用项目原来的入口。

## 已完成的短程验证：当前结果为负

2026-11-17 在 **LLFF / fern / 3 views** 上完成三组各一次的 1000-iteration 验证。三组使用相同的训练／测试协议、分辨率参数 `8`、初始点数 `100000` 和固定 `seed=0`，均执行了 4 次原有 VCD/VCP 密度控制事件。下面是保存模型后独立重新渲染并评估得到的指标，**不是训练日志中最后一次显示的 PSNR**：

| 校正模式 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| `none`：直接 MVS | 22.4321003 | 0.7659114 | 0.1631543 |
| `similarity` | 19.3653564 | 0.5632265 | 0.3064851 |
| `tps` | 20.3283710 | 0.6444800 | 0.2477480 |

在这次短程实验中，两种校正的三项指标都差于不校正，TPS 的 PSNR 低约 2.10 dB。**目前没有证据支持把这版校正作为有效改进推广到全部实验。** 这只是一个场景、一个固定 seed、1000 iterations 的结果，不是正式的 10000-iteration 结果，不是 LLFF 场景平均值，也不能据此否定 TWINGS 原方法。尚未完成 Blender 的校正训练对照。

结果保存在 `output/twings_probe_20261117/<mode>/llff3/fern/`；各模式的 `output/twings_probe_20261117/<mode>/llff3/metrics_mean.txt` 对应本次唯一场景的指标。

此次实现还审计了校正所用的相机内参：原有 `CameraInfo.K` 对 COLMAP `SIMPLE_RADIAL` 的参数顺序存在误解。新增辅助函数只在**校正模块使用的相机副本**上，按实际 COLMAP 相机参数重新构造内参；没有改变原有训练／渲染相机。这里读取的是相机标定记录，不是 SfM 三维点或轨迹，控制点仍只由选定训练图像的 SIFT 匹配三角化得到。径向畸变仍按已有 MVS 流程近似为针孔模型，不进行额外去畸变；被忽略的径向系数和内参修正记录在校正元数据中。

下方脚本的正式默认预算仍为 10000 iterations，并未改成短程预算。

## 对应与外推审计，以及两个独立修复对照

2026-11-17 的训练图像几何审计发现了两个可复现的问题：

- fern 旧版 FPS 选中的 256 个控制点中，128 个在另一匹配视角的投影误差超过既有的 3 像素关联半径，最大 117.37 像素。lego 旧版 129 个控制对应中，80 个超限。这里的像素是长边最多 1600 的匹配分辨率，不是训练渲染分辨率。
- fern 全局 TPS 的 29.75% 输入点位于控制点凸包外；凸包外位移 P95 为 22.715，凸包内也有 11.915（场景世界单位）。在 4086 个非控制点确定性样本上，27.29% 的 Jacobian 行列式为负，且凸包内部的比例更高。说明问题不仅在外推，错误或互相冲突的控制对应也会破坏内部几何。有限差分交叉核验支持该 Jacobian 计算。

对应审计见 `output/twings_diagnostics_20261117/{fern3,lego8}_correspondences_v2/`；旧形变审计见 `output/twings_diagnostics_20261117/fern3_tps_warp.json`。脚本 `diagnose_controls.py` 和 `diagnose_warp.py` 只读输入，不修改缓存或训练。这些几何证据不是 GT 正确性证明，也不是 PSNR 下降各因素的因果占比。

新增模式保留旧 `none`／`similarity`／`tps` 行为和默认值，显式选择后才生效：

1. `tps_consistent`：枚举训练特征附近的源点候选；同一个源点必须在该 track 的**所有观测视角**满足正深度、图像内与像素半径约束。按多视角投影误差进行确定性一对一分配，源点、track 和完全相同的源／目标坐标不重复绑定。再使用原来的 TPS 求解和控制点预算。这并不要求源点在每个视角都是 z-buffer 第一名，也不等同于已经解决遮挡或重复纹理。
2. `tps_local`：使用同一套一致控制点，在有局部控制支撑的位置施加 TPS，其余点严格保持原位：

   \[
   T_{\mathrm{local}}(x)=x+s(x)(T_{\mathrm{TPS}}(x)-x),\qquad
   s(x)=1-\prod_j[1-\phi(\|x-c_j\|/h_j)],\quad
   \phi(r)=(1-r)_+^4(1+4r).
   \]

   每个支持半径 \(h_j\) 来自该控制点到原始 MVS 点云第 64 个不同邻点的距离，不使用数据集名称或固定世界单位距离。点不足时用全部可用不同邻点。`--mvs_geometry_local_neighbors` 可显式调整该统一实验超参数，不能将其宣称为无超参数方法。控制点处 \(s=1\)，所有支持球外 \(s=0\)，过渡平滑。点数、顺序、RGB 不变；无支撑区域保留直接 MVS，**不是退回原始 DropGaussian 点云**。

原 TWINGS 还会在形变后采样靠近目标控制点的点；这里为保持初始化预算和点序，采用原位局部化而非复制其采样过程。仍然是借鉴实现，不是原论文复现。

局部化只能限制形变影响范围，不能保证不翻折：支持权重的空间梯度本身也影响 Jacobian。两个新模式均保存确定性采样的 Jacobian／奇异值诊断，不能将采样结果当作全局单射性证明，也不会按测试 PSNR 自动选择参数或静默回退。

尤其要区分“跨视角不一致比例”和“真实错配率”：没有 GT 时，另一视角投影偏差大，既可能是绑定了错误表面，也可能是正确表面的 MVS 深度本来就错得很大。新的全视角约束是保守筛选，不是错配真值分类器；它也会拒绝一些本来需要大幅修复的几何，因此恢复到基线并不等于已学会校正严重 MVS 错误。

最小消融命令（新的输出根目录，旧结果不会覆盖）：

```bash
MATRIX="llff:3" SCENES=fern GPU_ID=0 ITERATIONS=1000 \
  CORRECTIONS="none tps_consistent tps_local" \
  OUTPUT_ROOT=output/twings_repair_trial \
  bash scripts/twings/run_comparison.sh
```

各模式训练预算和输入点云必须一致，再比较独立渲染指标。先修复对应、再比较局部支持，避免把两个改变的作用混在一起。

### 修复后的 fern 短程结果

同一天重新运行直接 MVS、新对应＋全局 TPS、新对应＋局部 TPS，各一次 1000 iterations。协议与上方首轮一致，三组校正前点云 SHA256、训练／测试相机和所有训练参数均核对一致；未按这些测试指标调整匹配或支持参数。

| 模式 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| `none`（本轮重新训练） | 22.4536915 | 0.7660033 | 0.1643883 |
| `tps_consistent` | 22.3044281 | 0.7565423 | 0.1669559 |
| `tps_local` | 22.4309540 | 0.7665775 | 0.1616893 |

结果目录：`output/twings_repair_probe_20261117/<mode>/llff3/`。旧 `tps` 的 20.3283710 dB 来自前一轮同协议实验，并非本轮重跑。固定 seed 下 GPU 训练也存在数值波动；本轮与上轮 `none` 相差约 0.022 dB。

新对应修复后，fern 的 399 个候选中选 256 个；源—目标距离 RMSE 从 6.859 降至 0.650，这主要说明排除了大位移、不一致对应，不能当作真实深度误差下降。局部版 PSNR 相对本轮 `none` 为 **−0.0227 dB**，SSIM **+0.000574**，LPIPS **−0.002699**。目前结论是“避免原 TPS 的明显负优化、恢复到接近直接 MVS”，**不是已证明全面优于直接 MVS**。训练曲线也未显示明确加速。

局部版全点云非零支持比例为 10.596%，float32 实际移动点比例为 10.162%。约 90% 点未移动，对照接近也可能来自形变范围大幅缩小。独立 4096 点诊断仍能检测到局部翻折：在同一组 431 个有效支持样本内，全局版约 2.088%，局部版约 2.784%；不能只用整体负行列式比例很低来宣称形变安全。该结果不支持直接扩大到所有场景正式实验。

独立报告 `output/twings_diagnostics_20261117/fern3_repaired_support.json` 由 `diagnose_local_warp.py` 重建并核对校正前后点云哈希得到，没有改写训练时的元数据。`diagnose_warp.py` 只允许全局 `tps`／`tps_consistent`，避免将局部方法误诊断为底层全局 TPS。

### 不调参的 lego 迁移检查

沿用相同的匹配、TPS 与支持参数，对 Blender / lego / 8 views 跑一次 `none` 与一次 `tps_local`，均为 1000 iterations、分辨率参数 1、100000 个初始点；相机划分、输入哈希及训练协议核对一致。评估使用该协议的 200 张测试图像，不以测试图像构建控制点。

| 模式 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| `none` | 21.2242851 | 0.8694589 | 0.1319910 |
| `tps_local` | 21.2029743 | 0.8693848 | 0.1318441 |

结果目录：`output/twings_repair_probe_20261117/<mode>/blender8/`。局部版使用 54 个控制点，实际移动 2.179% 输入点，PSNR **−0.0213 dB**，SSIM **−0.0000741**，LPIPS **−0.0001469**。这是接近不校正的表现，不是改善证据；1024 个采样点未检出翻折也不能证明全局无翻折。没有训练旧 TPS 的 lego 对照，因此不能宣称在 lego 上相对旧 TPS 提升了多少。

两场景均未显示明确 PSNR 收益；未跑本轮全数据集／正式长程实验。当前建议保留直接 MVS＋VCD/VCP 为对照主线，将新模式用于对应与外推消融，而非作为已验证有效的新主方法。最终 CPU 测试共 273 项，264 项通过、9 项 CUDA-only 测试因 CPU 测试进程屏蔽显卡而跳过；此外已完成上述五次真实 GPU 训练、保存模型及独立渲染评估。DropGaussian／VCD/VCP 的三个核心实现文件哈希未改变。

## 先做小规模验证

先进入可运行本项目的 Python 环境。默认 `PYTHON_BIN=python`；若没有激活环境，可显式设置 `PYTHON_BIN=/home/wuxiaohan/miniforge3/envs/DropGaussian/bin/python`。

可选校正模块还使用 NumPy、SciPy 和支持 SIFT 的 OpenCV；原有训练环境未必包含 OpenCV。缺失依赖时会明确报错，不会自动联网安装。本次本地验证环境的 OpenCV 为 `5.0.0` 且 SIFT 可用，这不是对其他环境的版本要求。

以下命令仅预览，不检查数据和缓存、不启动训练、不创建输出目录：

```bash
MATRIX="llff:3" SCENES=fern GPU_ID=0 \
  bash scripts/twings/run_comparison.sh --dry-run
```

在 LLFF 的一个场景依次跑三组；任何一组失败都会停止：

```bash
MATRIX="llff:3" SCENES=fern GPU_ID=0 \
  bash scripts/twings/run_comparison.sh
```

再检查此前比较容易出现 MVS 负优化的合成场景；方法不对 Blender 做特判：

```bash
MATRIX="blender:8" SCENES=lego GPU_ID=1 \
  bash scripts/twings/run_comparison.sh
```

这里只给出实验命令，并不表示已经跑过完整训练或证实指标提升。先检查校正诊断和渲染结果，再决定是否扩大实验。

## 四个数据集的单组入口

省略 `SCENES` 时，每个入口使用现有完整场景列表；下列默认运行 TPS 校正。视角数和卡号仍通过 `N_VIEWS`、`GPU_ID` 指定：

```bash
N_VIEWS=3  GPU_ID=0 bash scripts/twings/train_llff.sh
N_VIEWS=6  GPU_ID=1 bash scripts/twings/train_dtu.sh
N_VIEWS=12 GPU_ID=2 bash scripts/twings/train_mipnerf360.sh
N_VIEWS=8  GPU_ID=3 bash scripts/twings/train_blender.sh
```

只跑指定对照组：

```bash
MVS_GEOMETRY_CORRECTION=none N_VIEWS=3 GPU_ID=0 \
  bash scripts/twings/train_llff.sh
MVS_GEOMETRY_CORRECTION=similarity N_VIEWS=3 GPU_ID=0 \
  bash scripts/twings/train_llff.sh
```

其他已有环境变量仍可用，例如 `DATA_ROOT`、`SCENES`、`ITERATIONS`、`RESOLUTION`、`MVS_INIT_PATH`、`MVS_INIT_MAX_POINTS` 和 `DTU_MASK_ROOT`。几何参数可通过 `MVS_GEOMETRY_MAX_CONTROLS`、`MVS_GEOMETRY_SMOOTHING` 或额外 `train.py` 参数传入。CLI 选项以 `python train.py --help` 为准。

`MVS_GEOMETRY_CORRECTION` 对模式有最终决定权，额外 CLI 的 `--mvs_geometry_correction` 和 `--density_control` 不能改变这一对照设计。请用 `N_VIEWS` 而不是 `--n_views`，保证输出目录标记与实际训练视角数一致。

为避免绕过独立输出目录或使训练与评估设置不一致，新增入口拒绝 CLI 的 `-m`／`--model_path`、`-s`／`--source_path`、`-r`／`--resolution`、`--iterations`、`--n_views`、`--dataset_name`，包括其参数附着形式和长选项缩写。请使用 `OUTPUT_ROOT`、`DATA_ROOT`／`SCENES`、`RESOLUTION`、`ITERATIONS`、`N_VIEWS` 或对应数据集脚本；几何参数及其他训练额外参数仍可传入。

## 完整九种视角设置的比较

```bash
# 先预览：27 个数据集／视角／方法任务，共 273 次逐场景训练。
bash scripts/twings/run_comparison.sh --dry-run

# 确认小规模实验有效后，再运行全部。
GPU_LLFF=0 GPU_DTU=1 GPU_MIPNERF360=2 GPU_BLENDER=3 \
  bash scripts/twings/run_comparison.sh
```

默认设置是 LLFF 3/6/9、DTU 3/6/9、Mip-NeRF360 12/24、Blender 8；每个场景每组只跑一次，不做多 seed 重复。沿用原脚本的固定 seed 只是保证可复现，不是额外重复训练。默认均为 10000 iterations，分辨率参数依次为 8、4、8、1，DTU 仍走 IDR mask 评估。

可以用 `MATRIX="llff:3 dtu:3"` 缩小设置，用 `CORRECTIONS="none tps"` 缩小方法。GPU 优先级为 `GPU_<DATASET>_<VIEWS>` > `GPU_<DATASET>` > `GPU_ID` > `0`，例如 `GPU_LLFF_9=2`。所有任务**串行**运行，不会自动多卡并行；不同卡号只是指定每个任务使用哪张卡。`MATRIX` 覆盖继承的 `N_VIEWS`。

跨数据集比较时不要全局指定一个仅适用某场景的 `DATA_ROOT`、`SCENES` 或 `MVS_INIT_PATH`，否则会产生不兼容输入。

## 输出和判断标准

默认输出独立保存：

```text
output/twings/none/<dataset><views>/<scene>/
output/twings/similarity/<dataset><views>/<scene>/
output/twings/tps/<dataset><views>/<scene>/
```

`OUTPUT_ROOT` 在这些新脚本中始终表示上层目录，脚本会自动追加 `<mode>/<dataset><views>`，避免三组覆盖。每种设置沿用公共流程生成 `metrics_mean.txt` 和逐场景指标。

新入口在实际运行前要求对应的 `<mode>/<dataset><views>` 目录为空或不存在；若已有文件则报错，而不是让公共流程静默跳过旧结果。修改参数、重新运行或从中断实验重启时，请指定新的 `OUTPUT_ROOT`，例如 `OUTPUT_ROOT=output/twings_trial2`。`--dry-run` 不受此限制。

启用校正时，每个场景额外保存：

- `geometry_correction.json`：所用训练相机、模式、控制点数量、匹配／关联统计、拟合诊断、位移统计、输入／输出哈希和校正耗时。
- `geometry_controls.npz`：源控制点、目标控制点、源点索引、变换后的控制点，便于检查对应关系与形变。
- `input.ply`：变换后的初始化点云；原始 MVS 缓存 PLY 不会被覆盖。
- `manifest.json` 的 `initialization_source.geometry_correction`：记录同一份校正元信息；同时保留校正前后点云的哈希。

`none` 保持原有初始化行为，不额外生成校正控制点文件。

首先比较同点数、同训练设置下的 PSNR／SSIM／LPIPS，并检查每场景而非只看平均值。若 `similarity` 与 `tps` 相当，就没有证据表明局部形变是必要的；若 TPS 改善匹配误差却损害新视角质量，也不应将其当作有效改进。收敛比较需同时报告训练迭代曲线与端到端耗时，后者应包含匹配、三角化和校正开销。单次结果只能提供初步证据，不能证明统计显著性或保证 Blender 不退化。
