# 模块消融实验

`scripts/mvs_complement/` 当前完整方法为几何互补初始化＋EGD 增密＋DPP 剪枝，默认写入新的 `output/mvs_complement_dpp/`。固定 coverage 的完整方法入口 `scripts/mvs_complement_egd_fastgs/` 继续保留，其兼容目录名保留 `fastgs`。本目录提供固定组合的消融入口，不迁移或重标已有实验，也不代表这些实验已经完成。

这里的“补充 MVS”是**保留原始重建点云，再用通过深度支持与覆盖筛选的 MVS 候选点补缺**，不是用 MVS 点云整体替换原始点云。

## 目录与结果对应关系

下表的模块开关指额外引入的模块，EGD 列专指用于增密的门控。各组保留 DropGaussian 正则化与相应的原有高斯优化、增密和剪枝流程；关闭 EGD/DPP 不等于停止一切增密/剪枝。

| 脚本目录 | 几何互补初始化 | EGD 增密门控 | DPP | 含义 | 默认结果设置目录 |
| --- | :---: | :---: | :---: | --- | --- |
| `baseline/` | — | — | — | 原始初始化与原始训练，作为匹配基线 | `output/ablation/baseline/<dataset><views>/` |
| `complement_only/` | ✓ | — | — | 仅加入几何互补初始化 | `output/ablation/complement_only/<dataset><views>/` |
| `egd_only/` | — | ✓ | — | 原始初始化，仅加入误差引导增密 | `output/ablation/egd_only/<dataset><views>/` |
| `dpp_only/` | — | — | ✓ | 原始初始化、原生 ADC，仅加入带同轮多视角支持保护的 DPP | `output/ablation/dpp_only/<dataset><views>/` |
| `egd_dpp/` | — | ✓ | ✓ | 原始初始化，开启 EGD 增密门控与带同轮保护的 DPP；完整方法去掉几何互补初始化 | `output/ablation/egd_dpp/<dataset><views>/` |
| `complement_egd/` | ✓ | ✓ | — | 几何互补＋EGD，用于完整方法的去 DPP 对照 | `output/ablation/complement_egd/<dataset><views>/` |

`dpp_only` 全程使用原生 ADC 的增密规则，不使用 EGD 增密门控；仅在 DPP 晚期剪枝同轮计算多视角支持，为该轮实际增密谱系生成保护掩码。因此不能将它描述为“完全没有 EGD/多视角计算”，也不能用未保护的普通剪枝替代它。内部 `--vcp_strategy fastgs`、`--fastgs_protect_egd` 和历史日志字段保留兼容性与 FastGS 来源，展示名为 DPP。详见 [DPP 单独消融说明](dpp_only/README.md)。

`egd_dpp` 保留 EGD 增密门控与同轮实际增密谱系的 DPP 保护；内部通过 `DENSITY_CONTROL=vcd_vcp` 启用 `--density_control vcd_vcp`，并固定 `--vcp_strategy fastgs --fastgs_protect_egd`。它不使用 MVSAnywhere 或几何互补，也不传 `--dpp_only`。详见 [EGD＋DPP 消融说明](egd_dpp/README.md)。

每个设置目录下再按场景保存，例如：

```text
output/ablation/complement_only/llff3/fern/
output/ablation/egd_only/llff6/fern/
output/ablation/dpp_only/llff3/fern/
output/ablation/egd_dpp/llff3/fern/
```

每个设置完成后，设置目录下会生成 `metrics_mean.txt`、`metrics_per_scene.csv`
和 `metrics_summary.json`；场景子目录保存 `manifest.json`、训练日志、点云及渲染结果。
例如，仅补充 MVS 的 LLFF 三视图平均指标位于
`output/ablation/complement_only/llff3/metrics_mean.txt`。

真实数据集的“原始初始化”复用项目原始重建点云，不接入 MVSAnywhere。原始资产可能来自 COLMAP 的稠密重建，不能因此称为“完全没有任何传统多视图立体步骤”。

当前完整方法使用新目录，不迁移、覆盖或重新标记已有结果：

- LLFF、DTU、Mip-NeRF360 默认：`output/mvs_complement_dpp/coverage/keep_base/<dataset><views>/<scene>/`。
- Blender 默认：`output/mvs_complement_dpp/blender_random/blender8/<scene>/`。
- 完整方法设置正的初始化总预算时，真实数据集路径中的 `keep_base` 改为 `budget<点数>`。

## Blender 的消融解释

遵循项目既有策略，Blender 使用原始的 10,000 点随机初始化，不使用 MVS 点云或 MVS 深度缓存。不能把这一设置记为“Blender 上验证了 MVS 补缺”。因此，在其他配置相同时：

- `complement_only` 与 `baseline` 方法配置相同。
- `complement_egd` 与 `egd_only` 方法配置相同。
- `egd_dpp` 与 `scripts/mvs_complement/` 和 `scripts/mvs_complement_egd_fastgs/` 的 Blender 完整方法配置相同。

`dpp_only` 在 Blender 同样保留随机初始化和原生 ADC，仅加入带保护的 DPP；默认 8 视角，并在启动训练前拒绝 `N_VIEWS > 8`，批量矩阵也执行该预检。这个新增限制不改变旧消融入口。

`egd_dpp` 在 Blender 保留同样的 10,000 点随机初始化，开启 EGD 增密与带保护 DPP；默认 8 视角，也在单数据集和批量入口拒绝 `N_VIEWS > 8`。上述与完整方法的等价关系要求其他训练和评测协议一致。

这些入口仍为四类数据集提供统一使用方式，但不应把等价配置重复列为独立有效消融。运行结果可能存在数值波动；配置等价不等于保证逐位复现。

真实数据集如缺少有效重建基础点云，继续遵循现有实现的直接 MVS 回退策略，须检查运行记录，不能只凭目录名称认定执行了几何互补。

## 单数据集运行

每个子目录都包含以下五个入口：

```text
train_llff.sh
train_dtu.sh
train_mipnerf360.sh
train_blender.sh
train_all_views.sh
```

先预览将要运行的方法、视角数、卡号与结果目录：

```bash
GPU_ID=0 N_VIEWS=3 bash scripts/ablation/complement_only/train_llff.sh --dry-run
GPU_ID=1 N_VIEWS=6 bash scripts/ablation/egd_only/train_llff.sh --dry-run
GPU_ID=2 N_VIEWS=9 bash scripts/ablation/dpp_only/train_dtu.sh --dry-run
GPU_ID=0 N_VIEWS=3 bash scripts/ablation/dpp_only/train_llff.sh --dry-run
GPU_ID=0 N_VIEWS=3 bash scripts/ablation/egd_dpp/train_llff.sh --dry-run
```

移除 `--dry-run` 后才执行训练、渲染、评估及汇总。卡号仅为示例，不表示当前空闲；脚本不会自动占用多个 GPU。默认每种设置包含该数据集的全部配置场景，而不是使用全部训练视角。默认视角数为 LLFF 3、DTU 3、Mip-NeRF360 12、Blender 8，默认训练 10,000 次迭代。

只跑一个场景时使用 `SCENES`：

```bash
GPU_ID=0 N_VIEWS=3 SCENES=fern \
  bash scripts/ablation/complement_only/train_llff.sh
```

`OUTPUT_ROOT` **始终表示共同父目录**，脚本会自动追加 `<variant>/<dataset><views>`，不需要自己追加：

```bash
GPU_ID=0 N_VIEWS=6 OUTPUT_ROOT="$PWD/output/ablation_v2" \
  bash scripts/ablation/egd_only/train_llff.sh
```

该命令写入 `output/ablation_v2/egd_only/llff6/<scene>/`。设置结果目录非空时会拒绝运行，避免复用旧指标或覆盖结果；重跑请使用新的父目录。

## 批量运行全部视角设置

```bash
GPU_ID=0 bash scripts/ablation/complement_only/train_all_views.sh --dry-run
GPU_ID=0 bash scripts/ablation/egd_only/train_all_views.sh
GPU_ID=0 bash scripts/ablation/dpp_only/train_all_views.sh
GPU_ID=0 bash scripts/ablation/egd_dpp/train_all_views.sh --dry-run
```

每个 `train_all_views.sh` 默认包含九种设置：

| 数据集 | 训练视角数 | 场景数 | 场景实验数 |
| --- | --- | ---: | ---: |
| LLFF | 3、6、9 | 8 | 24 |
| DTU | 3、6、9 | 15 | 45 |
| Mip-NeRF360 | 12、24 | 7 | 14 |
| Blender | 8 | 8 | 8 |

合计 91 次场景实验，逐个执行，每个场景/视角组合仅运行一次，不进行多随机种子重复。训练仍沿用项目默认随机种子，以记录设置；“只跑一次”不等于禁用随机种子。

批量入口可用 `MATRIX` 指定子集，用 `GPU_<DATASET>_<VIEWS>` 或 `GPU_<DATASET>` 指定对应卡号：

```bash
MATRIX="llff:3 llff:6 mipnerf360:12" \
GPU_ID=0 GPU_LLFF_6=1 GPU_MIPNERF360=2 \
  bash scripts/ablation/complement_only/train_all_views.sh --dry-run
```

批量卡号优先级为 `GPU_<DATASET>_<VIEWS>` > `GPU_<DATASET>` > `GPU_ID` > `0`。即便设置不同 GPU，批量入口仍串行执行，不会自动后台并行。`MATRIX` 的视角数覆盖继承的 `N_VIEWS`；单数据集入口用 `N_VIEWS` 与 `GPU_ID`。

## 配置约束与公平比较

- 子目录固定方法组合，不要传 `--density_control`、`--mvs_complement_mode`、`--init_mode`、`--dpp*`、`--fastgs*` 等方法覆盖参数，包括缩写和 `--参数=值` 形式。与入口冲突的 `MODE`、`DENSITY_CONTROL`、`MVS_COMPLEMENT_MODE` 环境变量会报错；`DPP_ONLY`、`FASTGS_*`、`VCP_STRATEGY` 环境变量在本目录均禁用，防止目录名与实际运行方法不一致。
- `dpp_only` 和 `egd_dpp` 允许数值环境变量 `DPP_PRUNE_INTERVAL`、`DPP_LOSS_THRESHOLD`、`DPP_PRUNE_THRESHOLD`、`DPP_OPACITY_THRESHOLD`，默认分别为 `8000`、`0.1`、`0.9`、`0.1`。这些设置仅属于 DPP 组，其他消融入口会拒绝。修改间隔或总迭代数时，仍须满足训练代码的同轮增密和 opacity reset 时间约束。
- 含几何互补的真实数据集组固定使用 `coverage`。可通过 `MVS_COMPLEMENT_MAX_POINTS`、`MVS_COMPLEMENT_TOTAL_POINTS`、`MVS_COMPLEMENT_CANDIDATE_POINTS`、`MVS_COMPLEMENT_COVERAGE_PIXELS`、`MVS_COMPLEMENT_DEPTH_SCALE` 调整既有数值设置；Blender 的这些 MVS 设置不参与初始化。
- 默认 `MVS_COMPLEMENT_TOTAL_POINTS=0`，保留全部原始点云并最多补充 100,000 个 MVS 点。因此不同方法初始点数不一定相同，应记录实际点数，不要将该组实验描述为严格等点数对照。正总预算可能采样原始点云，候选不足时实际点数也可能少于预算。
- `DATA_ROOT`、`SCENES`、显式缓存路径和 `RESOLUTION` 等配置在跨数据集批量运行时须与所有选中设置兼容；通常保留默认数据配置，或限制 `MATRIX` 为兼容子集。
- 本目录复用现有数据划分、训练、渲染和评测流程。比较时仍需核对输入资产、视角、分辨率、训练迭代数、DTU 掩码和测试集合，不能仅凭目录名宣称协议完全一致。
- 新脚本不会启动时自动跑一轮测试，也不会自动汇总尚未完成的结果；不预先承诺质量或速度提升。

`dpp_only` 与相同初始化和训练协议的 `baseline` 对比，回答 DPP 在基线上的单独作用；DPP 完整方法与 `complement_egd`（MVS＋EGD）对比，回答 DPP 在完整组合中的条件贡献。两组比较回答不同问题，不能互相替代。旧版 TBP 的代码和入口已移除，历史结果保持原有标记，不改名为 DPP。

`egd_dpp` 与 `egd_only` 对比，回答原始初始化＋EGD 上 DPP 的条件贡献；与 `dpp_only` 对比，回答原始初始化＋带保护 DPP 上 EGD 增密门控的贡献；与 `baseline` 对比，回答 EGD＋DPP 组合的整体作用。DPP 完整方法与 `egd_dpp` 对比，回答真实数据集上几何互补初始化的条件贡献；Blender 按既有协议不启用 MVS，因此不能用于这一初始化消融。
