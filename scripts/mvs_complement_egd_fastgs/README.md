# 补充 MVS + EGD 增密保护 + FastGS 剪枝

本目录是这三个部分的统一实验入口，复用 `scripts/fastgs_egd_pruning/` 的实现，不改变训练参数或算法；单独设置默认结果父目录，便于辨认和管理实验。原目录与历史结果保留。

## 包含的模块

1. **补充 MVS**：保留原始重建几何，通过跨视角深度支持及覆盖信息选择 MVS 候选点，补充未覆盖区域，不直接用整份 MVS 点云替换原始结构。
2. **EGD 增密与保护**：继续使用多视角误差支持、原有梯度及尺寸条件决定克隆或分裂。在同次 FastGS 剪枝中，保护实际选中的克隆原点和新增点、分裂生成的子点。不是永久保护，也不阻止原生低不透明度或尺寸清理。
3. **DPP 剪枝**：沿用 FastGS 高误差像素计数加权评分及不透明度候选规则，再排除本轮受 EGD 保护的点。

**Blender 协议例外**：按现有实验设置，Blender 使用原始 10,000 点随机初始化，不运行 MVS 补点；EGD 增密、增密保护与 FastGS 剪枝仍启用。不能将该入口的 Blender 结果描述为使用了 MVS。

## 训练命令

在项目根目录、已配置好的训练环境中运行：

```bash
GPU_ID=0 N_VIEWS=3 bash scripts/mvs_complement_egd_fastgs/train_llff.sh
GPU_ID=1 N_VIEWS=3 bash scripts/mvs_complement_egd_fastgs/train_dtu.sh
GPU_ID=2 N_VIEWS=12 bash scripts/mvs_complement_egd_fastgs/train_mipnerf360.sh
GPU_ID=3 N_VIEWS=8 bash scripts/mvs_complement_egd_fastgs/train_blender.sh
```

卡号仅为示例，不代表当前空闲。每条命令运行所选数据集当前配置的全部场景，训练预算默认 10,000 次迭代，并沿用底层脚本的渲染、评测和汇总流程。

| 数据集 | 常用 `N_VIEWS` | 默认值 |
| --- | --- | --- |
| LLFF | 3 / 6 / 9 | 3 |
| DTU | 3 / 6 / 9 | 3 |
| Mip-NeRF360 | 12 / 24 | 12 |
| Blender | 8 | 8 |

先检查卡号、视角数和输出路径，不启动训练：

```bash
GPU_ID=0 N_VIEWS=6 bash scripts/mvs_complement_egd_fastgs/train_llff.sh --dry-run
```

可用 `SCENES=fern` 选择单个场景，或用 `OUTPUT_ROOT=/path/to/new_results` 指定新的结果父目录。其他数据路径、MVS 参数及 CUDA 后端要求见[共用实验说明](../fastgs_egd_pruning/README.md)。

## 输出目录

默认结果对应关系如下（场景名只是示例）：

| 入口 | 场景结果路径 |
| --- | --- |
| `train_llff.sh` | `output/mvs_complement_egd_fastgs/coverage/keep_base/llff3/fern/` |
| `train_dtu.sh` | `output/mvs_complement_egd_fastgs/coverage/keep_base/dtu3/scan8/` |
| `train_mipnerf360.sh` | `output/mvs_complement_egd_fastgs/coverage/keep_base/mipnerf36012/bicycle/` |
| `train_blender.sh` | `output/mvs_complement_egd_fastgs/blender_random/blender8/lego/` |

`N_VIEWS` 会改变对应路径中的视角数。正的 `MVS_COMPLEMENT_TOTAL_POINTS` 会将 `keep_base` 改为 `budget<点数>`；默认值为 0。非空设置目录会拒绝运行，重复实验请使用新的 `OUTPUT_ROOT`，不会覆盖旧结果。

## 配置与核对

保护开关固定为 `FASTGS_PROTECT_EGD=1`，无需额外传参；冲突环境变量或手动方法参数会被拒绝。默认剪枝在第 8,000 次迭代执行，所有剪枝事件必须与有效 EGD 密度更新同轮，且不能与不透明度重置冲突。

每个场景的 `fastgs_pruning.jsonl` 记录原候选数量 `candidates_before_protection`、实际被保护挡下的数量 `blocked_by_egd` 与最终删除量 `pruned`。结合 `manifest.json`、最终高斯数、耗时和渲染指标验证效果；开启保护不保证一定触发或提升质量。

与 `scripts/fastgs_egd_pruning/` 在相同配置下属于**同一种方法**，不是新的消融变体；未加保护的对照入口仍是 `scripts/fastgs_pruning/`。
