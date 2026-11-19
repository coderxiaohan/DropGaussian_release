# EGD＋DPP 消融

这个入口使用原始初始化，开启 EGD 增密门控与带同轮保护的 DPP 剪枝，不使用 MVSAnywhere 或几何互补初始化。真实数据集复用原有重建点云；这些资产仍可能来自传统多视图重建，不能因此称为完全没有传统多视图立体步骤。

EGD 继续用多视角误差支持及原有梯度、尺寸条件筛选 clone/split。DPP 沿用既有评分与删除实现，并保护同轮实际选中的克隆原点和新增点、分裂生成的子点；保护仅作用于该轮 DPP 删除候选，原生 opacity/size 清理继续照常。这里保留完整 EGD 增密门控，与全程保留原生 ADC、只在晚期剪枝同轮计算保护的 `dpp_only` 不同。

内部通过 `DENSITY_CONTROL=vcd_vcp` 向训练管线传入 `--density_control vcd_vcp`，固定追加 `--vcp_strategy fastgs --fastgs_protect_egd`，不传 `--dpp_only`。`fastgs` 参数、日志及完整方法目录名保留兼容性与实现来源，展示名为 DPP。完整方法入口为 `scripts/mvs_complement/`，固定 coverage 的 `scripts/mvs_complement_egd_fastgs/` 也继续保留；历史结果保持原有标记。

## 运行

```bash
# 预览不启动训练、不写结果、不编译 CUDA。
GPU_ID=0 N_VIEWS=3 bash scripts/ablation/egd_dpp/train_llff.sh --dry-run
GPU_ID=1 N_VIEWS=3 bash scripts/ablation/egd_dpp/train_dtu.sh --dry-run
GPU_ID=2 N_VIEWS=12 bash scripts/ablation/egd_dpp/train_mipnerf360.sh --dry-run
GPU_ID=3 N_VIEWS=8 bash scripts/ablation/egd_dpp/train_blender.sh --dry-run

# 单场景训练 → 渲染 → 评估 → 汇总。
GPU_ID=0 N_VIEWS=3 SCENES=fern \
  bash scripts/ablation/egd_dpp/train_llff.sh

# 默认九种视角设置，包含全部配置场景，串行执行。
GPU_ID=0 bash scripts/ablation/egd_dpp/train_all_views.sh --dry-run

# MATRIX 选择子集，并覆盖继承的 N_VIEWS。
MATRIX="llff:3 llff:6 dtu:3 mipnerf360:12 blender:8" \
GPU_ID=0 GPU_LLFF_6=1 GPU_MIPNERF360=2 \
  bash scripts/ablation/egd_dpp/train_all_views.sh --dry-run
```

默认视角数为 LLFF 3、DTU 3、Mip-NeRF360 12、Blender 8，默认训练 10,000 次迭代。卡号仅为示例。移除 `--dry-run` 才会启动训练、渲染、评估和汇总。

批量入口默认包含 LLFF 3/6/9、DTU 3/6/9、Mip-NeRF360 12/24、Blender 8，共九种设置、91 次场景实验，每个场景/设置只运行一次。GPU 优先级为 `GPU_<DATASET>_<VIEWS>` > `GPU_<DATASET>` > `GPU_ID` > `0`；即使配置不同 GPU，设置仍串行执行。所有设置先完成参数预检，失败时停止批量执行。

结果保存到 `output/ablation/egd_dpp/<dataset><views>/<scene>/`。`OUTPUT_ROOT` 始终是共同父目录，脚本自动追加 `egd_dpp/<dataset><views>`；训练时已有非空设置目录会被拒绝，重跑需使用新的父目录。场景保存训练日志和 `manifest.json` 等输出，设置完成后汇总指标。

## 固定组合与数值设置

可通过 `GPU_ID`、`N_VIEWS`、`SCENES`、`DATA_ROOT`、`ITERATIONS`、`RESOLUTION`、`PYTHON_BIN`、`OUTPUT_ROOT`、`DTU_MASK_ROOT`、`SEED`、`PORT` 调整既有运行设置。DPP 数值设置如下：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `DPP_PRUNE_INTERVAL` | `8000` | 晚期剪枝间隔 |
| `DPP_LOSS_THRESHOLD` | `0.1` | 高误差像素阈值 |
| `DPP_PRUNE_THRESHOLD` | `0.9` | 归一化剪枝分数阈值 |
| `DPP_OPACITY_THRESHOLD` | `0.1` | 低不透明度剪枝阈值 |

阈值范围为 `[0, 1]`。剪枝间隔必须为正整数，并满足训练中的调度校验：与有效 EGD 增密轮次对齐、避免 opacity reset 同轮、剪枝后保留优化迭代。修改间隔或总迭代数时须同时核对这些约束，并记录实际配置。

固定方法参数不能用 CLI 覆盖，包括 `--dpp*`、`--fastgs*`、`--density*`、`--vcp*`、`--vcd*`、`--mvs*` 及初始化/训练协议参数，缩写和 `--参数=值` 同样拒绝。`MODE`、`DENSITY_CONTROL`、`MVS_COMPLEMENT_MODE` 仅接受固定组合对应的 `dropgaussian`、`vcd_vcp`、`none`；`DPP_ONLY`、`FASTGS_*`、`VCP_STRATEGY` 环境变量禁用。MVS 路径和数值环境设置不会在本组激活 MVS。

## 对照与 Blender 解释

| 入口 | 初始化 | EGD 增密门控 | 带保护的 DPP |
| --- | --- | :---: | :---: |
| `baseline` | 原始 | — | — |
| `egd_only` | 原始 | ✓ | — |
| `dpp_only` | 原始 | — | ✓ |
| `egd_dpp` | 原始 | ✓ | ✓ |
| 完整方法 `scripts/mvs_complement_egd_fastgs/` | 真实数据集使用几何互补；Blender 使用原始随机点云 | ✓ | ✓ |

- `egd_dpp` 对 `egd_only`：原始初始化＋EGD 基础上，DPP 的条件贡献。
- `egd_dpp` 对 `dpp_only`：原始初始化＋带保护 DPP 基础上，启用 EGD 增密门控的贡献。
- `egd_dpp` 对 `baseline`：原始初始化上，EGD＋DPP 组合的整体作用。
- 完整方法对 `egd_dpp`：真实数据集上几何互补初始化的条件贡献；须匹配其他训练与评测设置。

Blender 保留原始 10,000 点随机初始化，不使用 MVS 点云或深度缓存，默认 8 视角；超过 8 视角会在训练前报错，批量矩阵也会先验证。在数据、视角、分辨率、迭代数、随机种子及其他协议均相同时，本入口与完整方法的 Blender 配置等价，不能重复列作独立有效消融，也不能用于证明 MVS 补缺效果。配置等价不保证结果逐位相同。

真实数据集对照还需核对输入资产、DTU 掩码与评测集合。目录与脚本只定义实验配置，不表示实验已完成，也不承诺质量或速度提升。
