# Scene-Adaptive MVS Prior Fusion for DropGaussian

## 1. 目标与约束

本方法只改变 3D Gaussian 的初始化，不修改 DropGaussian 的渲染器、损失函数、
dropout、ADC densification 或优化器。自动策略仅使用当前场景的训练图像、相机、
SfM 点云和 MVS 输出；不读取数据集名称、验证图像、测试图像或最终指标。

设计目标是同时满足：

1. MVS 可靠时提供密集、各向异性的几何先验；
2. MVS 证据较弱时降低先验的 opacity 和形状约束，而不是突然开关；
3. MVS 与独立 SfM 几何不一致时，残差点集为空，初始化和原始
   DropGaussian 逐属性一致。

`auto` 因而总是解析为同一个 `adaptive` 算法。`replace` 和 `augment` 仅保留为
显式消融选项，不属于自动策略。

## 2. 无监督可靠性估计

### 2.1 场景内反事实标定

对 MVS 候选点 $x_i$ 和训练视角 $v$，将点投影到图像并计算：

$$
e^g_i=\operatorname{median}_v\left|\log z_{iv}-\log \hat z_{iv}\right|,
$$

其中 $z_{iv}$ 是投影深度；$\hat z_{iv}$ 优先来自与训练相机严格匹配的 MVS
深度缓存，否则由点云的交叉分组 z-buffer 产生。外观误差 $e^a_i$ 是可见视角间
颜色离散度与投影颜色到点颜色误差的平均。

绝对误差不能跨场景使用固定阈值。为此，在每个视角内确定性地错排深度与颜色
对应，生成同一场景的反事实负样本 $e^-_{di}$。正样本优于反事实的软概率为：

$$
q_i=\frac{1}{D}\sum_{d=1}^{D}
\sigma\!\left(\frac{e^-_{di}-e^+_i}{T}\right),
\qquad
T=1.4826\,\operatorname{MAD}(e^- - e^+).
$$

$T$ 完全由当前场景估计。$q=0.5$ 是随机对应的统计机会水平，而非人为质量阈值。
令 $w_i$ 为多视角有效支持度，则几何与外观证据为：

$$
E_g=[\max(2\bar q_g-1,0)]^2,\quad
E_a=[\max(2\bar q_a-1,0)]^2,
\quad
\bar q=\frac{\sum_i w_iq_i}{\sum_iw_i},
$$

MVS 自一致性为 $r_{self}=\sqrt{E_gE_a}$。局部可靠性使用几何和外观 logit 的
平均，并乘以视角支持度：

$$
\ell_i=w_i\max\!\left(
2\sigma\!\left(\frac{\operatorname{logit}q^g_i+
\operatorname{logit}q^a_i}{2}\right)-1,0\right).
$$

### 2.2 独立 SfM 锚点

仅有 MVS 自一致性不够，因为 domain gap 可能产生跨视角一致但整体错误的表面。
因此，全局 gate 必须由独立 SfM 点云开启。

首先用局部协方差特征值定义点云平面性：

$$
P_i=1-\frac{3\lambda_{i,\min}}
{\lambda_{i,1}+\lambda_{i,2}+\lambda_{i,3}}.
$$

将 SfM 点云的 $x/y/z$ 坐标分别随机置换，保持包围盒和每个坐标的边缘分布、但
破坏真实表面。真实点云和这些 null cloud 的平面性软排序得到结构证据
$r_{struct}$。

其次，计算每个 SfM 锚点到 MVS 点云的最近距离；再与坐标独立置换后的 MVS
null cloud 比较：

$$
r_{align}=\max\left(2\,\mathbb E_i[q(d_i,d^-_i)]-1,0\right).
$$

这是一个无量纲、场景内标定的 SfM--MVS 空间对齐分数，不需要米制距离阈值。
独立锚点的每视角证据、视角联合证据及最终全局可靠性为：

$$
p=r_{struct}r_{align},\qquad
r_{anchor}=1-(1-p)^V,
$$

$$
m=1+Vr_{self},\qquad
r_G=1-(1-r_{anchor})^m.
$$

这里 MVS 自一致性只能在 $r_{anchor}>0$ 后增加证据 multiplicity，不能独自开启
gate。这个非对称性是抵抗合成域上“自洽但错误”MVS 的关键。

### 2.3 局部 SfM 支持

令 $\delta_i$ 为 MVS 点到最近 SfM 点的距离除以该 SfM 点的局部间距。定义：

$$
a_i=r_{struct}\exp(-\delta_i^2/2),\qquad
\ell'_i=1-(1-\ell_i)(1-a_i),
$$

$$
\nu_i=1-\exp(-\delta_i^2/2).
$$

$\ell'_i$ 表示局部可靠性，$\nu_i$ 表示相对 SfM 的新覆盖区域。若输入 MVS 点数
超过资源预算，采样权重由二者平滑组合；采样本身使用固定 seed，保证可复现。

## 3. 可靠性到 Gaussian 属性的映射

### 3.1 场景级先验质量

锚点 gate 的保守估计先经过只在 gate 已开启时有效的自一致性校正：

$$
\tilde A=\min\left(1,\frac{r_G}{1-r_{self}}\right),\qquad
A=1-(1-\tilde A)^m.
$$

$A\in[0,1]$ 是场景级先验质量。设资源上限内候选数为 $N$。当
$\lfloor AN\rfloor=0$ 时，不实例化任何 MVS Gaussian；这是离散点数必须具备的
数值条件，而非数据集质量阈值。

当 gate 开启时保留资源上限内的稠密点位，而不按 $A$ 稀疏抽点。原因是抽稀会
增大 KNN 距离，使少数低置信点初始化成大尺度 splat，反而产生更强遮挡。先验的
连续强度由总 opacity mass 表达。

### 3.2 位置与 opacity

高可靠 MVS 点以 $x_i^{MVS}$ 初始化，并在后续训练中保持可学习。原始 SfM
Gaussian 全部保留且使用未经修改的 DropGaussian 初始化属性。

定义归一化局部支持：

$$
u_i=r_{self}+(1-r_{self})\ell'_i,\qquad
\hat u_i=\frac{u_i}{\mathbb E_j[u_j]},
$$

$$
o_i=\operatorname{clip}(A\hat u_i,0,1),\qquad
\alpha_i=1-(1-\alpha_0)^{o_i},
$$

其中 $\alpha_0=0.1$ 是原始 3DGS 的标准初始化 opacity，而不是质量阈值。
该式可解释为 fractional alpha compositing：可靠区域趋近标准 opacity，较弱区域
的可见贡献和位置梯度连续减小，并且 $A\to0\Rightarrow\alpha_i\to0$。

### 3.3 scale 与 rotation

MVS 局部协方差给出 PCA 尺度 $s_i^{PCA}$ 和旋转 $R_i^{PCA}$；当前实例化点集
中的最近邻距离给出无方向尺度 $d_i$。几何承诺系数为：

$$
\gamma_i=(r_G\ell'_i)^2.
$$

尺度在 log-space 内插：

$$
s_i=\exp\left[(1-\gamma_i)\log d_i+
\gamma_i\log s_i^{PCA}\right].
$$

高可靠区域得到贴合局部表面的各向异性 Gaussian；低可靠区域接近各向同性
DropGaussian 尺度，扩大优化搜索盆地。$\gamma_i\to0$ 时旋转对各向同性 Gaussian
没有实际作用。

## 4. 平滑及精确退化

若 MVS 与 SfM 不对齐，则 $r_{align}=0$，从而：

$$
r_{anchor}=r_G=A=0
\Longrightarrow N_{MVS}=0.
$$

此时执行路径直接把原始 SfM 点云传给 `create_from_pcd`，不会先拼接再赋予零
opacity，因此位置、颜色、scale、rotation、opacity、优化器状态和随机数轨迹都
与 baseline 相同。若 $A$ 很小但非零，位置集合仍保持稠密以稳定 KNN，影响则由
$\alpha_i$ 和 $\gamma_i$ 连续趋零。这给出了“平滑弱化”和“端点精确相等”同时成立
的退化机制。

## 5. 当前验证结果

### 5.0 冻结的配对实验协议

正式比较不再把历史 baseline 均值与新方法均值直接相减。Baseline 和 MVSInit
必须由同一个 `scripts/train_mvsdrop_benchmark.sh` 产生，并满足：

1. 相同源码 commit，且运行时 tracked working tree 必须干净；
2. 相同场景、训练/测试相机名称、分辨率、迭代数和 seed；
3. 相同基础 SfM/随机点云 SHA-256；
4. 唯一允许不同的实验变量是 `init_mode`；
5. 每个设置必须包含完整官方场景集合，正式表格使用 seed 0/1/2 的配对均值。

DropGaussian-matched 主协议固定为：

| Dataset | Views | Resolution | Iterations |
| --- | --- | ---: | ---: |
| LLFF | 3 / 6 / 9 | 8 | 10,000 |
| DTU | 3 / 6 / 9 | 4 | 10,000 |
| Mip-NeRF360 | 12 / 24 | 8 | 10,000 |
| Blender | 8 | 1 | 10,000 |

Blender 两个分支都按相同 seed 在各自进程内生成相同的 10k 随机基础点云。随机
点云不再写入数据集共享的 `points3d.ply`，因此并行 seed 不会互相覆盖。manifest
记录完整 protocol identity；`summarize_mvsdrop_multiseed.py` 会拒绝 commit、相机
划分或基础点云哈希不一致的配对。

推荐从已提交的干净分支运行：

```bash
SEEDS="0 1 2" bash scripts/run_mvsdrop_multiseed.sh
```

先验证 Blender 配对身份时可只跑一个 setting：

```bash
MATRIX="blender:8" SEEDS="0" \
OUTPUT_ROOT="output/mvsdrop/protocol_v1" \
bash scripts/run_mvsdrop_multiseed.sh
```

FastGS/StereoGS 等外部结果只作为参考行；除非使用完全相同的数据划分、训练预算
和 evaluator 重跑，否则不参与本方法的 paired delta 与显著性检验。

协议修订前生成的 `output/mvsinit/*` 不包含完整 protocol identity，且其 manifest
记录的 commit 早于最终冻结版本；严格汇总器会有意拒绝这些结果。它们可以保留
为机制诊断，但不能继续与历史 DropGaussian 均值组成正式主表。

以下是 seed 0 的开发集机制验证，不属于上述冻结后的正式配对主表。
LLFF/Mip-NeRF360/DTU 使用各自既定评估协议；DTU 使用固定 split 和 IDR
foreground mask。LPIPS 越低越好。

| Dataset | Scene | $r_G$ / $A$ | Baseline PSNR / SSIM / LPIPS | Adaptive PSNR / SSIM / LPIPS | Delta PSNR / SSIM / LPIPS |
| --- | --- | ---: | --- | --- | --- |
| LLFF-3 | fortress | .565 / .576 | 23.1493 / .7328 / .1785 | 24.9461 / .7872 / .1353 | +1.7968 / +.0544 / -.0432 |
| LLFF-3 | fern | .234 / .234 | 22.9243 / .7569 / .1857 | 23.4689 / .7847 / .1502 | +.5447 / +.0279 / -.0355 |
| LLFF-3 | flower | .141 / .145 | 20.4053 / .6464 / .2416 | 21.0331 / .6934 / .2066 | +.6278 / +.0470 / -.0350 |
| Mip-12 | bicycle | 1.000 / 1.000 | 19.9223 / .4201 / .5160 | 20.2490 / .4556 / .4397 | +.3267 / +.0356 / -.0763 |
| Mip-12 | room | 1.000 / 1.000 | 20.8633 / .7499 / .2467 | 21.2195 / .7718 / .2224 | +.3562 / +.0219 / -.0243 |
| DTU-3 | scan30 | .086 / .233 | 13.7249 / .9203 / .0849 | 14.7482 / .9228 / .0795 | +1.0233 / +.0024 / -.0054 |
| DTU-3 | scan21 | 0 / 0 | 14.9760 / .7274 / .2211 | exact baseline trajectory | 0 / 0 / 0 |
| Blender-8 | lego | 0 / 0 | 24.0414 / .8730 / .1083 | 24.0414 / .8730 / .1083 | 0 / 0 / 0 |
| Blender-8 | chair | 0 / 0 | 26.6514 / .9307 / .0583 | 26.6514 / .9307 / .0583 | 0 / 0 / 0 |

结果为 6/9 场景严格改善，且这 6 个场景的 PSNR、SSIM、LPIPS 三项全部改善；
其余 3 个场景精确退化，因而没有负优化。六个改善场景的平均 PSNR 增益为
+0.7792 dB。

Blender 的等价性不依赖“重复运行看起来接近”。在当前代码、相同 seed 和同一
GPU 上，lego 与 chair 的 baseline/adaptive 均产生 10,000 个初始点；input PLY
和 iteration-1 PLY 的 SHA-256 分别完全一致，iteration-1 PSNR 也逐位相同。
两场景的 $r_{self}$ 分别仍为 .336 和 .275，但 $r_{align}=0$ 使它们无法自行开启
gate，直接验证了独立锚点设计。

### 5.1 收敛

按同一 test-view evaluator 的记录 checkpoint，以 baseline 的 10k 最终 PSNR
为目标：

| Scene | Baseline target | Baseline first recorded hit | Adaptive first recorded hit |
| --- | ---: | ---: | ---: |
| fern | 22.9369 | 10,000 | 1,000 |
| flower | 20.4144 | 10,000 | 1,000 |
| bicycle | 19.9396 | 10,000 | 7,000 |

因此完整配对曲线上的目标迭代数分别减少 90%、90% 和 30%。另外，fortress 在
500 iteration、room 在 4,000 iteration 已超过各自 baseline 的最终渲染 PSNR；
这两个旧 baseline 没有保存中间曲线，所以不把它们用于严格的速度比值。

## 6. 工程落点与审计

- 可靠性、反事实 null、属性映射：`utils/mvs_init.py`
- 通用自动策略和 exact-baseline 路由：`scene/__init__.py`
- base/prior Gaussian 属性拼接：`scene/gaussian_model.py`
- checkpoint 收敛曲线和 manifest：`train.py`
- 四数据集统一 runner：`scripts/train_mvsdrop_benchmark.sh`

所有 MVS cache 都检查相机 provenance；已知训练 split 不匹配的缓存会被拒绝，
不会为了得到更高 reliability 而偷换相机。完整单元测试当前为 27/27 通过，包含
无数据集特定 MVS policy、反事实 alignment、cache provenance、确定性采样和零
可靠性退化测试。

## 7. 学术表述与局限

核心贡献应表述为：**利用场景内反事实标定和独立 SfM 锚点，估计 MVS 先验的
可迁移可靠性；再通过 opacity mass 与 log-scale interpolation 将可靠性连续映射
到 3D Gaussian 初始化，并在证据为零时与无先验 baseline 精确等价。**

当前 9 场景结果是单 seed 的机制验证与性能筛选，不足以支持统计显著性声明。
正式论文表格仍应冻结代码后运行至少 3 个 seed，并报告均值、标准差或置信区间。
离线 MVS 推理时间未计入训练时间；当前实现也保留一个可配置的 100k 残差资源
预算，它限制计算量但不参与判断某类数据是否可靠。

相关工作：

- [DropGaussian, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.html)
- [MVSAnywhere, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Izquierdo_MVSAnywhere_Zero-Shot_Multi-View_Stereo_CVPR_2025_paper.html)
- [StereoGS preprint](https://arxiv.org/abs/2606.30545)
