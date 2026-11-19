# DPP 单独消融

这个入口使用原始初始化、关闭 MVS，且全程保留原生 ADC（自适应增密与原生清理规则），仅增加 DPP 晚期剪枝。真实数据集复用原有重建点云，Blender 使用原有 10,000 点随机初始化。它不加载 MVSAnywhere 点云或深度缓存；真实数据集原始重建资产仍可能来自传统多视图重建。

DPP 保留**同轮多视角支持保护**：仅在晚期剪枝发生的迭代计算训练视角支持，用于保护该轮实际增密谱系；不会把支持条件用于筛选原生 ADC 的 clone/split。保护只作用于该轮 DPP 删除候选，原生 opacity/size 清理继续照常。因此“关闭 EGD 增密门控”不表示完全没有 EGD/多视角计算，DPP 单独消融也不是未保护的普通剪枝。

内部固定参数为 `--dpp_only --density_control vcp --vcp_strategy fastgs --fastgs_protect_egd`。`fastgs` 参数、日志和既有路径保留兼容性；DPP 的评分/删除实现沿用项目对 [FastGS fast-dropgaussian](https://github.com/fastgs/FastGS/tree/fast-dropgaussian) 的适配，来源不因展示名改变。

## 运行

```bash
# 预览不启动训练、不写结果、不编译 CUDA。
GPU_ID=0 bash scripts/ablation/dpp_only/train_llff.sh --dry-run
GPU_ID=0 bash scripts/ablation/dpp_only/train_dtu.sh --dry-run
GPU_ID=0 bash scripts/ablation/dpp_only/train_mipnerf360.sh --dry-run
GPU_ID=0 bash scripts/ablation/dpp_only/train_blender.sh --dry-run

# 单场景训练 → 渲染 → 评估 → 汇总。
GPU_ID=0 N_VIEWS=3 SCENES=fern \
  bash scripts/ablation/dpp_only/train_llff.sh

# 默认九种视角设置，串行执行；MATRIX 可选择子集。
GPU_ID=0 bash scripts/ablation/dpp_only/train_all_views.sh --dry-run
MATRIX="llff:3 dtu:3 mipnerf360:12 blender:8" GPU_ID=0 \
  bash scripts/ablation/dpp_only/train_all_views.sh --dry-run
```

默认视角数：LLFF 3、DTU 3、Mip-NeRF360 12、Blender 8；默认 10,000 次迭代。各数据集默认包含全部已配置场景。Blender 超过 8 视角会在训练前报错，整个批量矩阵也会先验证再执行。

结果统一保存到 `output/ablation/dpp_only/<dataset><views>/<scene>/`。`OUTPUT_ROOT` 是共同父目录，会自动追加 `dpp_only/<dataset><views>`；已有非空设置目录会被拒绝，重跑需使用新的父目录。

## 固定组合与数值设置

允许 `GPU_ID`、`N_VIEWS`、`SCENES`、`DATA_ROOT`、`ITERATIONS`、`RESOLUTION`、`PYTHON_BIN`、`OUTPUT_ROOT`、`DTU_MASK_ROOT`、`SEED`、`PORT` 等既有运行设置。DPP 数值设置如下：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `DPP_PRUNE_INTERVAL` | `8000` | 晚期剪枝间隔 |
| `DPP_LOSS_THRESHOLD` | `0.1` | 高误差像素阈值 |
| `DPP_PRUNE_THRESHOLD` | `0.9` | 归一化剪枝分数阈值 |
| `DPP_OPACITY_THRESHOLD` | `0.1` | 低不透明度剪枝阈值 |

阈值范围为 `[0, 1]`。剪枝间隔必须为正整数，并满足训练中的调度校验：发生在有效 ADC 轮次、避免 opacity reset 同轮、剪枝后保留优化迭代。默认 8,000/10,000 的组合沿用完整 DPP 方法。修改数值请另选输出父目录并记录实际配置。

不能用 CLI 覆盖 `--dpp*`、`--fastgs*`、`--density*`、`--vcp*`、`--vcd*`、`--mvs*` 或初始化/训练协议参数；缩写和 `--参数=值` 同样拒绝。`MODE`、`DENSITY_CONTROL`、`MVS_COMPLEMENT_MODE` 仅接受与固定组合一致的值；`DPP_ONLY`、`FASTGS_*`、`VCP_STRATEGY` 环境变量禁用。MVS 路径和数值环境设置不会在本组激活 MVS。

## 对照与解释

- `dpp_only` 对 `baseline`：原始初始化＋原生 ADC 基线上，DPP 的单独作用。
- DPP 完整方法 `scripts/mvs_complement_egd_fastgs/` 对 `complement_egd`：MVS＋EGD 基础上，DPP 的条件贡献。

对照需匹配数据、训练视角、分辨率、迭代数、随机种子、DTU 掩码与评测集合。Blender 无 MVS 初始化，不能用于证明 MVS 补缺效果。旧版 TBP 的代码和入口已移除，其历史结果与本组独立保存，不能重标为 DPP。这些入口尚不构成性能或速度提升的实验结论。
