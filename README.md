# LIBS Spectral Analysis Toolkit

[中文](#中文) | [English](#english)

Research code for an auditable laser-induced breakdown spectroscopy (LIBS)
study of industrial dust. The project combines locally acquired spectra with
NASA ChemCam calibration data to examine physically informed feature
extraction, small-sample quantitative regression, and conservative
cross-instrument transfer.

> **Research status.** This is an exploratory research baseline, not an
> industrial quantitative-analysis product. The strict local target set used
> for the final transfer assessment contains six independent physical samples.
> Multiple laser shots from one specimen are technical replicates, never new
> independent samples.

---

## 中文

### 1. 项目要解决什么问题

本项目研究粉末状工业除尘灰的 LIBS 多元素定量分析，核心问题不是单纯追求网络的名义评分，而是：在本地样本稀缺、谱线受仪器覆盖和基体效应制约、且与 NASA ChemCam 存在显著跨仪器差异时，哪些元素能够被诚实地评估，NASA 标定知识能否在严格样品隔离下提供增益。

因此，仓库把工作重点放在以下四件事上：

1. 修正波长轴和数据切分等会制造虚高结果的工程问题；
2. 以 NIST 原子谱线、连续背景扣除和谱段覆盖为约束构造特征；
3. 把独立物理样品而不是激光打点作为统计单位；
4. 在极小样本下报告误差、置换检验和多重比较结果，而不把单个正 `R2` 当作可部署证据。

项目当前应被理解为一个**可审计的可行性研究与后续实验基线**。它不发布、也不声称已经形成工业现场可用的标定产品。

### 2. 研究演进与审计结论

早期工作曾尝试全谱深度网络、单流 ECNN 和双流门控结构。代码审计表明，部分早期高分受方法性缺陷影响，不能作为性能证据。当前代码与文档采用以下修复后的原则：

| 审计问题 | 风险 | 当前处理 |
| --- | --- | --- |
| Excel 索引列被当作波长 | 元素峰窗与真实物理波长错位 | 在数据加载中使用校准系数或真实波长列。 |
| 测试集参与早停或阈值选择 | 测试泄露、乐观偏差 | 模型选择必须留在训练折内；评估使用按样品的 GroupKFold/LOSO。 |
| 同一试样的 shots 跨折 | 网络记忆样品背景指纹而非浓度规律 | 先按物理样品聚合，测试样品与训练样品零重叠。 |
| 对无覆盖主线的元素建模 | 没有可识别的物理信号 | 逐元素检查波长覆盖、线强、SNR 和浓度方差；不满足条件则标记为不可评估。 |
| 小波重构和零填充处理不稳 | 人工制造异常 SNR 或数值问题 | 限制重构长度，并在特征提取中使用有效波长范围。 |

审计后的一个重要负结果是：在只保留物理峰区、并消除泄露后，当前散铺粉末条件下 Mn、P、S、Ti、K 等微量元素没有可支持的分类或定量信号。这是数据和测量条件的约束，不应用更复杂的网络掩盖。

### 3. 当前方法主线

最终的定量基线是“本地原生波长上的物理特征 + NASA 特征域迁移”，而不是端到端深度网络：

```text
本地 LIBS shots
  -> 质量控制与样品级稳健聚合
  -> SNIP 连续背景扣除
  -> NIST 覆盖感知的峰高、SNR、峰面积与强度比特征
  -> Ridge/SVR 本地基线，或 NASA ChemCam 标样 Ridge
  -> 仅在训练折拟合 slope/bias calibration (SBC)
  -> 留一样品交叉验证、RMSE/MAE/R2、精确置换检验
```

`feature_extractor_v2.py` 提取有明确物理窗口的特征；`train_regression_v2.py` 与 `eval_peak_ratio_regression.py` 给出本地基线；`train_transfer_benchmark.py` 实现从 ChemCam 标样到本地训练折的特征域斜率/偏置校正（SBC）。

ChemCam 源域不会把 744 条单次测量视为 744 个独立训练样本。当前 T5 源模型先按标准样品聚合，使用 59 个匹配的地质标准样品和 235 条原始光谱。这个选择避免把重复测量的数量误当成化学组成的独立信息量。

### 4. 严格评估协议与当前结果

最终本地迁移评估使用 Dataset B 的 6 个独立 TR 工业灰样品。每个外层折留出一个物理样品；样本聚合、预处理、特征选择、缩放、SBC 映射和任何超参数选择均不应使用该留出样本的信息。对正 `R2` 结果，以全部 `6! = 720` 种标签索引置换计算精确经验 p 值。

下表为冻结方案的 T5（NIST 特征域 SBC）结果。`NoDrift` 是默认主线，因为 Dataset B 已提供波长列；`WithDrift` 仅作为逐样品漂移的探索性对照。

| 配置 | 元素 | R2 | RMSE (wt%) | 未校正精确 p | Bonferroni p | BH-FDR p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NoDrift | Ca | +0.827 | 1.255 | 0.018 | 0.216 | 0.112 |
| NoDrift | Si | +0.125 | 1.197 | 0.037 | 0.449 | 0.112 |
| NoDrift | Fe | -0.171 | 9.766 | 0.087 | 1.000 | 0.210 |
| NoDrift | Mg | -0.860 | 1.372 | 0.458 | 1.000 | 0.676 |
| NoDrift | Al | -1.554 | 1.268 | 0.595 | 1.000 | 0.676 |
| NoDrift | Zn | -0.440 | 3.029 | 0.620 | 1.000 | 0.676 |
| WithDrift | Fe | +0.242 | 7.858 | 0.035 | 0.416 | 0.112 |
| WithDrift | Ca | +0.427 | 2.283 | 0.035 | 0.416 | 0.112 |

两种配置和六个元素构成 12 项比较。经 Bonferroni 或 Benjamini-Hochberg FDR 校正后，没有任何结果达到统计显著。因此，上表中 Ca、Si 与 Fe 的正向结果只能称为**未校正的探索性名义信号**，不能被表述为已经验证的工业定量精度。

负 `R2` 也不自动意味着“所有研究方向失败”：在 N=6、浓度范围很窄或谱线弱的元素上，约 1 wt% 的绝对误差即可带来很负的 `R2`。更有意义的判断是，该元素是否具有被仪器覆盖且 SNR 可用的谱线、足够的浓度方差，以及相对均值/线强比基线的稳定提升。

### 5. 代码地图

| 模块 | 文件 | 用途与状态 |
| --- | --- | --- |
| 配置和读取 | `config.py`, `dataset_loader.py` | 本地/NASA 路径约定、加载、波长修正/对齐、样品信息。 |
| 物理预处理 | `industrial_preprocess.py`, `wavelet_preprocess.py`, `four_step_spectral_transform.py` | 连续背景、去噪和跨仪器启发式对照；四步变换不是已验证的仪器响应校正。 |
| 特征工程 | `feature_extractor_v2.py`, `check_nist_peaks.py` | NIST 谱线、覆盖感知峰特征和峰位检查。 |
| 严格回归 | `train_regression_v2.py`, `train_regression_rigorous.py`, `eval_peak_ratio_regression.py` | 样品级聚合、LOSO、Ridge/SVR/峰强比基线。 |
| 跨域迁移 | `train_transfer_benchmark.py`, `generate_exact_permutation_csv.py` | ChemCam 到本地的 T5/SBC 与精确置换检验。 |
| 深度学习探索 | `model_dual_stream.py`, `train_dual_stream_ecnn.py`, `train_calibration_ecnn.py` | ECNN/双流实验代码，仅作研究储备和对照。 |
| 复核工具 | `verify_audit.py`, `verify_full_reproducibility.py` | 审计规则和对本地生成工件的检查。 |

#### 关于双流代码

`train_dual_stream_ecnn.py` 当前是元素存在性检测：它使用二元标签、`BCEWithLogitsLoss`、sigmoid 和 0.5 阈值，**不是 wt% 连续回归实现**。此外，旧双流路径把冻结 NASA 流的投影层置于 `torch.no_grad()` 中，使随机初始化的投影不能学习；两个卷积分支还使用 `Flatten -> Linear(24608, 64)`，对当前独立样本数而言参数量过大。它不应被原样复用于定量回归。

未来若研究双视图深度模型，应先固定 E0--E2 小模型基线，再测试低容量、带观测通道掩码的物理谱线/全局谱图编码器。提议中的 P-MAE-LIBS（physics-masked autoencoder）是未来研究假设，不是本仓库已验证的方法：只在已观测通道上做掩码重建，以冻结编码器配合极小的本地下游回归头，并在相同的嵌套 LOSO 下比较。

### 6. 环境、数据契约与复现

环境定义以 Conda `dl1` 为准：

```powershell
conda env create -f environment.yml
conda activate dl1

# 从本地、获授权的化验信息生成统一标签
python parse_and_unify_labels.py

# 本地覆盖感知 LOSO 基线
python train_regression_v2.py

# ChemCam 到本地的迁移基线
python train_transfer_benchmark.py

# N=6 的精确索引置换与多重比较汇总
python generate_exact_permutation_csv.py

# 检查本地生成的结果工件；--full-rerun 会从本地数据重跑
python verify_full_reproducibility.py
```

完整运行需要在本地、按权利和使用条款准备下列目录；它们不会也不应提交到本仓库：

```text
NASA data/
  msl_ccam_libs_calib.csv
  ccam_calibration_compositions.csv
2026.07.16wl_corr/
  <TR 样品目录及已定标 xlsx 光谱>
20260422data/
20260422数据标注/
labels/
  unified_wt.csv
```

`labels/unified_wt.example.csv` **不是数据集，不能直接训练**。它是零数据、仅含列名的 schema 模板，公开了 `unified_wt.csv` 所需的数据契约，防止协作者误填列顺序。复制为 `labels/unified_wt.csv` 后，只能填写有授权、可追溯的样品标识、数据来源、核验状态和元素质量分数。详见 [labels/README.md](labels/README.md)。

NASA ChemCam 产品应从 [NASA PDS](https://pds.nasa.gov/) 获取，并遵守集合的产品说明、引用和再分发条款。NIST ASD 是谱线元数据来源，不是带化验标签的 LIBS 训练集。

### 7. 数据公开范围与隐私边界

本公开仓库只包含源代码、环境定义、零数据标签模板和项目文档。下列内容未公开：

- 原始或处理后的本地光谱、化验单、真实质量分数及样品对应关系；
- NASA 或其他第三方数据的重新分发副本；
- 模型权重、检查点、图像、预测结果及其他生成工件；
- 本地路径、实验记录中可能暴露样品来源的信息。

### 8. 参考文献与公开数据来源

这些条目说明了每项外部工作与本仓库的对应关系；它们不是对本项目结果的背书。

1. Clegg, S. M. et al. *Recalibration of the Mars Science Laboratory ChemCam instrument with an expanded geochemical database.* **Spectrochimica Acta Part B: Atomic Spectroscopy** 129, 64--85 (2017). [DOI](https://doi.org/10.1016/j.sab.2016.12.003). ChemCam 标定数据库与源域聚合的科学背景。
2. Jin, G. et al. *A New Spectral Transformation Approach and Quantitative Analysis for MarSCoDe LIBS Data.* **Remote Sensing** 14, 3960 (2022). [DOI](https://doi.org/10.3390/rs14163960). 跨仪器光谱变换与响应校正的背景；本仓库四步管线不构成其复现。
3. Jia, L. et al. *Initial Drift Correction and Spectral Calibration of MarSCoDe LIBS on the Zhurong Rover.* **Remote Sensing** 14, 5964 (2022). [DOI](https://doi.org/10.3390/rs14235964). 锚线与漂移校正的物理动机。
4. Yu, Y. & Yao, M. *When Convolutional Neural Networks Meet LIBS: End-to-End Quantitative Analysis Modeling of ChemCam Spectral Data for Major Elements Based on ECNN.* **Remote Sensing** 15, 3422 (2023). [DOI](https://doi.org/10.3390/rs15133422). ECNN 对照实验的架构参考，不证明本地小样本 CNN 有效。
5. Roh, Y. et al. *LEVI: Generalizable Fine-tuning via Layer-wise Ensemble of Different Views.* **ICML 2024**, PMLR 235, 42500--42520. [Paper](https://proceedings.mlr.press/v235/roh24a.html). 分层/多视图迁移的通用思路，非 LIBS 方法验证。
6. [NIST Atomic Spectra Database](https://physics.nist.gov/asd). 原子谱线窗口和跃迁信息；可见性仍受仪器范围、分辨率、基体和等离子体条件限制。
7. [NASA PDS ChemCam Derived Calibration Collection](https://pds.nasa.gov/ds-view/pds/viewCollection.jsp?identifier=urn%3Anasa%3Apds%3Amsl_chemcam_derived%3Acalib&version=1.0). ChemCam 公开标定产品入口；使用者必须阅读产品说明。

### 9. AI 协作与作者责任

Gemini、DeepSeek 和 ChatGPT 被用作辅助工具，支持代码审阅和调试思路、文字表达与结构整理、文献检索线索以及技术讨论。所有 AI 生成或建议的内容均由项目作者核查、修改和最终确认。

研究问题、整体技术路线、实验与数据工作安排、关键科学判断、结果解释、参考文献筛选与取舍，以及公开内容的最终责任均由项目作者完成并承担。AI 工具不是作者，也不替代实验、数据核验、统计判断或引用核查。

### 10. 许可证

尚未选择开源许可证。代码目前用于审阅与研究交流；在仓库所有者添加许可证前，不授予额外的复制、修改或再分发许可。

---

## English

### 1. Research Scope

This repository studies multi-element LIBS quantification for powdered
industrial dust. The practical question is not whether a larger neural network
can produce an attractive score; it is whether an element is quantitatively
identifiable when local samples are scarce, line visibility is constrained by
instrument coverage and matrix effects, and the local instrument differs
substantially from NASA ChemCam.

The repository therefore prioritizes corrected wavelength handling,
NIST-informed coverage-aware features, physical-sample-level evaluation, and
error, permutation, and multiplicity-aware reporting. It is an auditable
feasibility baseline, not a validated field-deployment calibration.

### 2. Audit-Informed Design

Earlier full-spectrum CNN, ECNN, and gated dual-stream experiments identified
important failure modes: spreadsheet index columns treated as wavelengths,
test-set-assisted model selection, same-specimen shots crossing folds, and
claims for elements without adequately covered diagnostic lines. The current
workflow corrects those defects by using calibrated/physical wavelength
columns, group-aware splits, sample-level aggregation, and per-element
coverage/SNR/variance triage.

An important negative result is retained rather than hidden: after restricting
models to physical peak regions and removing leakage, the present loose-powder
measurements do not support credible Mn, P, S, Ti, or K inference. A more
complex model cannot create an identifiable signal from inadequate measurement
conditions.

### 3. Implemented Quantitative Baseline

The principal baseline is native-grid physical feature extraction followed by
local regression or ChemCam-to-local feature-space transfer:

```text
local LIBS shots
  -> quality control and robust physical-sample aggregation
  -> SNIP continuum removal
  -> NIST coverage-aware peak-height, SNR, area, and ratio features
  -> local Ridge/SVR or a Ridge model trained on aggregated ChemCam standards
  -> slope/bias calibration (SBC) fitted within each training fold only
  -> leave-one-physical-sample-out evaluation and exact permutation tests
```

`feature_extractor_v2.py` implements physically defined peak features.
`train_regression_v2.py` and `eval_peak_ratio_regression.py` provide local
baselines. `train_transfer_benchmark.py` implements the T5 slope/bias
calibration transfer protocol. The ChemCam source model aggregates repeated
measurements before source fitting: the current T5 implementation uses 59
matched geological standards and 235 raw spectra, not 744 shots as independent
chemical samples.

### 4. Evidence and Statistical Boundary

The strict transfer assessment is LOSO over six independent Dataset-B physical
samples. The held-out sample must not affect aggregation, preprocessing,
feature selection, scaling, SBC fitting, or hyperparameter selection. Exact
empirical p-values for positive R2 values enumerate all `6! = 720` label
permutations.

| Configuration | Element | R2 | RMSE (wt%) | Nominal exact p | Bonferroni p | BH-FDR p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NoDrift (default) | Ca | +0.827 | 1.255 | 0.018 | 0.216 | 0.112 |
| NoDrift (default) | Si | +0.125 | 1.197 | 0.037 | 0.449 | 0.112 |
| NoDrift (default) | Fe | -0.171 | 9.766 | 0.087 | 1.000 | 0.210 |
| NoDrift (default) | Mg | -0.860 | 1.372 | 0.458 | 1.000 | 0.676 |
| NoDrift (default) | Al | -1.554 | 1.268 | 0.595 | 1.000 | 0.676 |
| NoDrift (default) | Zn | -0.440 | 3.029 | 0.620 | 1.000 | 0.676 |
| WithDrift (control) | Fe | +0.242 | 7.858 | 0.035 | 0.416 | 0.112 |
| WithDrift (control) | Ca | +0.427 | 2.283 | 0.035 | 0.416 | 0.112 |

Across two configurations and six elements, none of the 12 comparisons remains
significant after Bonferroni or Benjamini-Hochberg FDR correction. The positive
Ca, Si, and Fe values are therefore **uncorrected exploratory nominal signals**,
not validated industrial performance. Conversely, a negative R2 can be driven
by N=6, low target variance, or weak lines; it must be interpreted with line
coverage, SNR, concentration range, absolute error, and a mean/line-ratio
baseline.

### 5. Repository Map and Legacy Code Status

| Area | Files | Role |
| --- | --- | --- |
| Loading and configuration | `config.py`, `dataset_loader.py` | Local/NASA conventions, loading, wavelength handling, and sample metadata. |
| Preprocessing | `industrial_preprocess.py`, `wavelet_preprocess.py`, `four_step_spectral_transform.py` | Continuum/denoising utilities and an exploratory cross-instrument comparison. |
| Physical features | `feature_extractor_v2.py`, `check_nist_peaks.py` | NIST line windows, coverage-aware features, and peak checks. |
| Strict regression | `train_regression_v2.py`, `train_regression_rigorous.py`, `eval_peak_ratio_regression.py` | Sample aggregation, LOSO, Ridge/SVR, and line-ratio baselines. |
| Transfer and statistics | `train_transfer_benchmark.py`, `generate_exact_permutation_csv.py` | T5/SBC transfer and exact permutation summaries. |
| Deep-learning exploration | `model_dual_stream.py`, `train_dual_stream_ecnn.py`, `train_calibration_ecnn.py` | ECNN and dual-stream research controls, not deployment code. |
| Verification | `verify_audit.py`, `verify_full_reproducibility.py` | Audit rules and checks for locally generated artifacts. |

The legacy `train_dual_stream_ecnn.py` is an element-presence classifier: it
uses binary labels, `BCEWithLogitsLoss`, sigmoid, and a 0.5 threshold. It is
**not** a continuous wt% regression implementation. The older dual-stream path
also contains a frozen NASA projection evaluated under `torch.no_grad()` and
very large `Flatten -> Linear(24608, 64)` branches. It must not be revived
unchanged for quantitative claims.

P-MAE-LIBS, a compact physics-masked autoencoder with observed-channel masks
and separate line/global spectral views, is a proposed future hypothesis only.
It would require fixed small-model baselines, fewer than roughly 0.3M
parameters, frozen-encoder downstream heads, and the same nested-LOSO protocol
before any quantitative claim.

### 6. Setup, Data Contract, and Reproduction

The supported environment is Conda `dl1`:

```powershell
conda env create -f environment.yml
conda activate dl1
python parse_and_unify_labels.py
python train_regression_v2.py
python train_transfer_benchmark.py
python generate_exact_permutation_csv.py
python verify_full_reproducibility.py
```

Full execution requires authorized local data under the directory contract in
the Chinese section. Raw local spectra, assay sheets, measured labels, model
weights, generated figures, and results are intentionally absent from this
repository.

`labels/unified_wt.example.csv` is a **header-only schema template**, not a
dataset and not a trainable file. Copy it to `labels/unified_wt.csv` only when
you have authorized, traceable measurements, then add one row per independent
physical sample following [labels/README.md](labels/README.md). The real label
file is Git-ignored and must remain local.

ChemCam products must be obtained from [NASA PDS](https://pds.nasa.gov/) and
used under their collection-specific documentation, citation, and distribution
terms. NIST ASD supplies atomic metadata, not assay-labelled LIBS training
data.

### 7. Next Work Under Current Constraints

1. **Make every experiment auditable:** separate detection from regression and
   fit all data-driven decisions within the training fold.
2. **Recover traceable labels already measured:** audit sample IDs and add only
   authorized assay values missing from the local unified table.
3. **Establish small-model baselines first:** native-window line ratios,
   Ridge/elastic net, PLS/interval-PLS, and uncertainty-aware kernel/Gaussian
   process models.
4. **Expand independent physical samples:** acquire 30--50 matrix-matched,
   standardized pressed-pellet standards spanning continuous concentrations,
   with a never-tuned blind set.
5. **Revisit representation learning only after those gates:** evaluate frozen
   NASA-pretrained encoders or dual-view models against the same nested-LOSO
   baselines.

### 8. References and Data Sources

1. Clegg, S. M. et al. *Recalibration of the Mars Science Laboratory ChemCam instrument with an expanded geochemical database.* **Spectrochimica Acta Part B: Atomic Spectroscopy** 129, 64--85 (2017). [DOI](https://doi.org/10.1016/j.sab.2016.12.003). ChemCam calibration and source-domain context.
2. Jin, G. et al. *A New Spectral Transformation Approach and Quantitative Analysis for MarSCoDe LIBS Data.* **Remote Sensing** 14, 3960 (2022). [DOI](https://doi.org/10.3390/rs14163960). Cross-instrument transformation context; not reproduced here.
3. Jia, L. et al. *Initial Drift Correction and Spectral Calibration of MarSCoDe LIBS on the Zhurong Rover.* **Remote Sensing** 14, 5964 (2022). [DOI](https://doi.org/10.3390/rs14235964). Physical motivation for anchor-line drift handling.
4. Yu, Y. & Yao, M. *When Convolutional Neural Networks Meet LIBS: End-to-End Quantitative Analysis Modeling of ChemCam Spectral Data for Major Elements Based on ECNN.* **Remote Sensing** 15, 3422 (2023). [DOI](https://doi.org/10.3390/rs15133422). ECNN control architecture reference.
5. Roh, Y. et al. *LEVI: Generalizable Fine-tuning via Layer-wise Ensemble of Different Views.* **ICML 2024**, PMLR 235, 42500--42520. [Paper](https://proceedings.mlr.press/v235/roh24a.html). General multi-view fine-tuning context, not LIBS validation.
6. [NIST Atomic Spectra Database](https://physics.nist.gov/asd). Atomic line metadata used for feature curation.
7. [NASA PDS ChemCam Derived Calibration Collection](https://pds.nasa.gov/ds-view/pds/viewCollection.jsp?identifier=urn%3Anasa%3Apds%3Amsl_chemcam_derived%3Acalib&version=1.0). Public ChemCam calibration entry point.

### 9. AI Assistance and Author Responsibility

Gemini, DeepSeek, and ChatGPT were used as assistive tools for code-review and
debugging ideas, writing and structural editing, literature-discovery leads,
and technical discussion. The project author reviewed, revised, and approved
all AI-assisted material.

The research question, overall study plan, experimental and data-work
planning, scientific judgement, result interpretation, reference selection,
and final responsibility remain with the project author. AI tools are not
authors and do not replace experiments, data validation, statistical judgement,
or citation verification.

### 10. License

No open-source license has been selected. The source is shared for review and
research discussion only; no additional permission to copy, modify, or
redistribute is granted until the repository owner adds a license.
