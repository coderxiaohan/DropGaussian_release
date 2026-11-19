# EGD 保护的 FastGS 剪枝实验

本目录在已有 `scripts/fastgs_pruning/` 实验上，只增加**同一次密度更新中 EGD 选出的增密点保护**，避免刚执行的增密结果被随后发生的 FastGS 剪枝直接撤销。MVS 几何互补初始化、EGD 判定与增密方式、FastGS 评分、阈值和调度均不改变；展示名为 DPP。

保护对象必须满足 EGD 的完整增密条件，包括原有梯度、尺寸与多视角误差支持条件，不是所有高误差区域中的点都被保护。克隆时保护保留的原点及其新增点；分裂时保护生成的子点。保护只作用于**同次 FastGS 剪枝事件**，不是永久保护，也不阻止原生的不透明度/尺寸清理。因而“受保护”不等于该点必定保留到训练结束。

这是待验证实验，不能预先认定重建质量、模型规模或速度更优。原有 FastGS 对照和历史输出保持不变。

## 运行

先预览命令、卡号、视角数与输出位置：

```bash
GPU_ID=0 N_VIEWS=3 bash scripts/fastgs_egd_pruning/train_llff.sh --dry-run
```

四类数据集入口：

```bash
GPU_ID=0 N_VIEWS=3 bash scripts/fastgs_egd_pruning/train_llff.sh
GPU_ID=1 N_VIEWS=3 bash scripts/fastgs_egd_pruning/train_dtu.sh
GPU_ID=2 N_VIEWS=12 bash scripts/fastgs_egd_pruning/train_mipnerf360.sh
GPU_ID=3 N_VIEWS=8 bash scripts/fastgs_egd_pruning/train_blender.sh
```

GPU 编号只是示例，不代表当前空闲。LLFF、DTU 可选 3/6/9 视图，Mip-NeRF360 可选 12/24，Blender 默认 8。每条命令包含该设置下全部配置场景，可用 `SCENES=fern` 等选单场景；沿用原有单次实验与 10,000 次迭代预算。

本目录固定 `FASTGS_PROTECT_EGD=1`，通常不必额外设置。冲突值或手动传入 `--fastgs_protect_egd` 等方法 CLI 参数均会被拒绝。原 `scripts/fastgs_pruning/` 默认仍为 `FASTGS_PROTECT_EGD=0`；只有用户显式设为 `1` 才改变旧入口行为。

## 结果与匹配对照

默认结果保存在新目录：

```text
output/fastgs_egd_pruning/
├── coverage/keep_base/llff3/fern/
├── coverage/keep_base/dtu3/scan8/
├── coverage/keep_base/mipnerf36012/bicycle/
└── blender_random/blender8/lego/
```

`OUTPUT_ROOT` 是共同父目录。正的 `MVS_COMPLEMENT_TOTAL_POINTS` 将 `keep_base` 改为 `budget<点数>`。非空设置目录会拒绝运行；重跑或修改参数请更换父目录，不覆盖、重标记历史结果。

建议用相同场景与配置比较以下两组，例如：

```bash
GPU_ID=0 N_VIEWS=6 SCENES=fern FASTGS_PROTECT_EGD=0 \
  bash scripts/fastgs_pruning/train_llff.sh

GPU_ID=0 N_VIEWS=6 SCENES=fern \
  bash scripts/fastgs_egd_pruning/train_llff.sh
```

若曾全局导出 `FASTGS_PROTECT_EGD=0`，运行新目录前应取消该环境设置；新入口会拒绝冲突值，避免目录名与实际方法不一致。

## 保持不变的设置

共用原 FastGS 实验启动器，不另复制一套训练流程：默认剪枝间隔 8,000，像素误差阈值 `0.1`、剪枝分数阈值 `0.9`、不透明度阈值 `0.1`；评分使用全部训练视角。输入、分辨率、原优化器、球谐参数和训练预算不变。Blender 沿用原始 10,000 点随机初始化，不使用 MVS 缓存。

原有数值环境变量仍可配置，但匹配实验应保持一致。剪枝间隔必须小于总迭代数。保护模式还要求每一次剪枝都与实际 ADC 密度更新同轮：该迭代须严格大于 `densify_from_iter`、严格小于 `densify_until_iter`，且能被 `densification_interval` 整除。任一剪枝事件不满足该条件，或与不透明度重置同轮，训练程序均会在启动前拒绝配置，避免出现没有同轮 EGD 证据的“保护”或重置引发的误删。默认 8,000/10,000 及原有密度调度满足这些要求。

首次训练启动时，在读取场景前预先准备独立 CUDA 诊断后端，需要可用的 CUDA 编译工具链；缓存和已安装渲染器隔离。`--dry-run` 不编译、不写文件、不训练。其他配置和后端说明参见 [FastGS 实验说明](../fastgs_pruning/README.md)。

## 核对保护是否生效

查看 `fastgs_pruning.jsonl` 中保护前的拟删除量 `candidates_before_protection`、本轮受保护的总点数 `egd_protected_count`、实际阻止删除的交集数量 `blocked_by_egd`、保护后的拟删除量 `requested_after_protection` 和最终实际删除量 `pruned`。其中 `blocked_by_egd` 只计算“受 EGD 保护且原本会被 FastGS 删除”的点，不等于全部受保护点数 `egd_protected_count`。

实际阻止删除的数量可能为零：例如该事件没有满足完整增密条件的点，或增密点没有落入 FastGS 拟删除集合，不能只凭开关已开启认定产生了收益。

同时核对 `manifest.json` 的保护配置、最终高斯数及训练耗时，再比较 PSNR、SSIM、LPIPS 和渲染细节。累计删除量不等于净压缩，保护有效触发也不保证新视角指标提高。后端准备时间、纯训练时间与含准备时间的统计口径沿用原 FastGS 实验。
