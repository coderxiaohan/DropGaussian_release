# FastGS 剪枝策略的独立实验

本目录保留现有几何互补初始化和 EGD 增密，剪枝采用 FastGS 的高误差像素计数评分与阈值删点，默认不保护同轮增密结果。它是剪枝策略适配实验，**不是 FastGS 整套训练流程复现，也没有预先验证质量或速度提升**。

本目录固定传入 `--vcp_strategy fastgs`，默认不传保护开关，算法与已有实验保持一致。`scripts/mvs_complement/` 当前使用 EGD＋受保护 DPP，并写入新的 `output/mvs_complement_dpp/`；历史输出保持原有标记。

## 剪枝规则与适配范围

在指定间隔的迭代完成原有密度更新后，重新渲染并统计高斯对高误差像素的实际贡献次数。各视角计数按 `0.8 × L1 + 0.2 × DSSIM` 的视角损失加权，汇总后在全部高斯上做 min–max 归一化。归一化分数大于 `0.9` **或者**不透明度小于 `0.1` 的高斯直接进入删除集合；不存在“先试删、损失不能升高才接受”的条件，也不添加至少两视角可见或 EGD 保护条件。

默认每 8,000 次迭代执行一次，保持现有 10,000 次迭代训练预算，不延长训练。为保留删点后的优化步骤并保证最终保存一致，要求 `FASTGS_PRUNE_INTERVAL < ITERATIONS`，且不在最终迭代执行剪枝。因此默认设置在第 8,000 步执行，随后继续优化。

训练程序还会按实际训练参数检查剪枝与不透明度重置的调度：若两者在增密阶段的同一迭代发生，将拒绝该配置，避免刚重置的不透明度触发大规模误删。默认 `8000/10000` 配置安全；并非所有小于总迭代数的间隔都可用，例如默认重置周期为 3,000 步时，2,000 步的剪枝间隔会在第 6,000 步与重置重合。最终以训练启动时的调度校验为准。

与官方实现的重要差异：本实验使用**全部不重复训练视角**评分，官方实现随机采样三个视角；这样不引入新的视角采样随机数。MVS 补缺、EGD、DropGaussian 正则化、原有优化器和球谐设置保持不变，也不引入官方其他训练调整或紧凑包围盒策略。真实数据集沿用 coverage 几何互补；Blender 仍使用原始 10,000 点随机初始化，无 MVS 点云与深度缓存要求。

参考：[FastGS 官方 fast-dropgaussian 分支](https://github.com/fastgs/FastGS/tree/fast-dropgaussian)、[官方评分与相机采样代码](https://github.com/fastgs/FastGS/blob/fast-dropgaussian/utils/fast_utils.py)、[官方高斯模型代码](https://github.com/fastgs/FastGS/blob/fast-dropgaussian/scene/gaussian_model.py)。

## 运行入口

先预览，`--dry-run` 不编译 CUDA、不读取训练数据、不创建输出文件，也不启动训练：

```bash
GPU_ID=0 N_VIEWS=3 bash scripts/fastgs_pruning/train_llff.sh --dry-run
```

四个数据集均有独立入口：

```bash
GPU_ID=0 N_VIEWS=3 bash scripts/fastgs_pruning/train_llff.sh
GPU_ID=1 N_VIEWS=3 bash scripts/fastgs_pruning/train_dtu.sh
GPU_ID=2 N_VIEWS=12 bash scripts/fastgs_pruning/train_mipnerf360.sh
GPU_ID=3 N_VIEWS=8 bash scripts/fastgs_pruning/train_blender.sh
```

GPU 号只是示例，不代表当前空闲。LLFF、DTU 分别使用 `N_VIEWS=3/6/9`，Mip-NeRF360 使用 `12/24`，Blender 默认 `8`。每条命令串行运行该设置下全部配置场景；可用 `SCENES=fern` 等限制场景。每个场景只跑一次，不自动开展多随机种子实验。

## 参数

| 环境变量 | 默认值 | 约束与含义 |
| --- | --- | --- |
| `FASTGS_PRUNE_INTERVAL` | `8000` | 正整数，小于总迭代数；周期性剪枝间隔 |
| `FASTGS_LOSS_THRESHOLD` | `0.1` | 有限 `[0,1]`；归一化像素误差阈值 |
| `FASTGS_PRUNE_THRESHOLD` | `0.9` | 有限 `[0,1]`；高斯归一化评分阈值 |
| `FASTGS_OPACITY_THRESHOLD` | `0.1` | 有限 `[0,1]`；低不透明度删除阈值 |

使用环境变量调整上述参数；不要直接传 `--fastgs_*` 或 `--vcp_strategy`。方法和协议 CLI 参数及缩写均被保护，冲突的 `DENSITY_CONTROL`、`MVS_COMPLEMENT_MODE`、`MODE`、`VCP_STRATEGY` 等环境设置会报错。已经移除的历史剪枝配置也会明确报错。

其他设置沿用完整方法：`GPU_ID`、`N_VIEWS`、`SCENES`、`DATA_ROOT`、`ITERATIONS`、`RESOLUTION`、`PYTHON_BIN`、`DTU_MASK_ROOT`、`SEED`、`PORT`，以及真实数据集的 `MVS_INIT_PATH` 与原有 MVS 点数/coverage 参数。MVS 相关设置对 Blender 初始化不生效。

## 输出目录

默认结果与旧方法完全分开：

```text
output/fastgs_pruning/
├── coverage/keep_base/llff3/fern/
├── coverage/keep_base/dtu3/scan8/
├── coverage/keep_base/mipnerf36012/bicycle/
└── blender_random/blender8/lego/
```

`OUTPUT_ROOT` 始终是共同父目录，不要重复填写 `coverage/keep_base` 等后缀。设定正的 `MVS_COMPLEMENT_TOTAL_POINTS` 后，`keep_base` 改为 `budget<点数>`，和原完整方法相同。

已有非空设置目录会拒绝运行，不复用或覆盖旧结果。改变 FastGS 阈值或重复实验时，请使用新的 `OUTPUT_ROOT`，例如：

```bash
GPU_ID=0 N_VIEWS=6 SCENES=fern \
OUTPUT_ROOT="$PWD/output/fastgs_pruning_trial2" FASTGS_PRUNE_THRESHOLD=0.95 \
  bash scripts/fastgs_pruning/train_llff.sh
```

## CUDA 诊断与结果检查

像素贡献次数使用独立的 CUDA 诊断后端，不以 alpha 合成质量代替计数。训练启动时、读取场景数据前，会预先检查和加载后端；首次运行从项目本地源码即时编译并使用隔离缓存，不会等到第 8,000 步才发现编译环境缺失。需要与当前 PyTorch 环境兼容的 CUDA toolkit、编译器及构建工具。它不覆盖或重新安装当前训练渲染器。

运行后检查独立的 `fastgs_pruning.jsonl` 中实际剪枝事件与删除数量；`density_control.jsonl` 在该策略下记录周期性的 EGD 事件，不是 FastGS 剪枝日志。`manifest.json` 中的 `fastgs_pruning.backend_setup_seconds` 单独记录后端准备耗时，`training_time` 不含该准备耗时，`training_time_including_backend_setup` 则包含。比较效率时应明确是否包含首次编译，避免把环境准备成本与稳态训练时间混在一起。

还应核对初始化与策略配置、最终高斯数量，并比较 PSNR、SSIM、LPIPS 和代表性渲染图。累计删除量不等于最终净压缩量；直接删点也不保证训练损失或新视角质量不下降。应在相同初始化、视角、点数预算和评测协议下与原完整方法比较，再决定是否采用。脚本或后端微测试通过只说明相应实现检查通过，不代表实际重建效果已经验证。
