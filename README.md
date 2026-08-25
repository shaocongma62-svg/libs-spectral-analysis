# LIBS Spectral Analysis Toolkit

[中文](#中文说明) | [English](#english)

## 中文说明

### 项目定位

本仓库提供激光诱导击穿光谱（LIBS）的研究代码，覆盖本地工业除尘灰光谱与 NASA ChemCam 标定光谱之间的数据读取、波长对齐、物理特征提取、小样本定量回归和跨域迁移基线。

项目目标是建立可审计的研究基线，而不是发布可直接用于工业现场的定量产品。当前 Dataset B 仅含 6 个独立物理样品，所有结果均属于探索性证据；同一试样的多次激光打点仅为技术重复，不能视为新增独立样本。

### 代码内容

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 配置与数据读取 | `config.py`, `dataset_loader.py` | 本地/NASA 数据路径、光谱加载、波长对齐和数据增强。 |
| 光谱物理特征 | `feature_extractor_v2.py`, `industrial_preprocess.py` | SNIP 扣底、锚线漂移估计、覆盖感知 NIST 峰特征和样品级聚合。 |
| 严格定量评估 | `train_regression_v2.py`, `eval_peak_ratio_regression.py` | Ridge、SVR、峰强比模型和按物理样品的 LOSO 评估。 |
| 跨域迁移 | `train_transfer_benchmark.py`, `generate_exact_permutation_csv.py` | ChemCam 到本地的 NIST 特征域 SBC 基线、精确索引置换检验和多重比较校正。 |
| 预处理对照 | `four_step_spectral_transform.py`, `plot_preprocessing_effect.py` | 自研跨仪器预处理对照实验。 |
| 深度学习探索 | `model_dual_stream.py`, `train_dual_stream_ecnn.py`, `train_calibration_ecnn.py` | ECNN、双流网络与消融实验；仅作为研究对照，不应绕过严格小样本评估。 |
| 质量控制 | `check_nist_peaks.py`, `verify_full_reproducibility.py` | 峰位检查、结果工件检查和可选全流程重跑。 |

### 环境与运行

```powershell
conda env create -f environment.yml
conda activate dl1

# 根据本地化验表生成统一标签
python parse_and_unify_labels.py

# 本地数据集的覆盖感知 LOSO 基线
python train_regression_v2.py

# ChemCam 到本地的迁移基线
python train_transfer_benchmark.py

# N=6 目标集的精确索引置换与多重比较结果
python generate_exact_permutation_csv.py

# 检查已生成的结果工件；--full-rerun 会从本地原始数据重跑
python verify_full_reproducibility.py
```

环境定义以 Conda `dl1` 为准。CPU 可完成基线复现；CUDA 仅用于部分神经网络实验。

### 数据、标签与隐私边界

原始光谱、化验单、派生的真实标签、模型权重和实验输出均未上传。它们可能包含敏感实验信息、体积较大，或受第三方数据使用条款约束。运行完整流程时，请在本地准备：

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

`labels/unified_wt.example.csv` **不是数据集，也不能直接用于训练**。它只保留 `unified_wt.csv` 所需的列名，作用是公开标签表的数据契约：复现实验的人可以复制该文件为 `labels/unified_wt.csv`，然后按 [labels/README.md](labels/README.md) 填入自己拥有、并获授权使用的样品标识、来源、核验状态和元素质量分数。真实标签不会提交到 GitHub。

NASA ChemCam 产品应从 [NASA PDS](https://pds.nasa.gov/) 获取，并遵守相应集合的分发和引用要求。

### 结果解释与方法边界

- T5 源模型在训练前会将 ChemCam 原始光谱按标准样品聚合；当前实现使用 59 个匹配地质标准样品和 235 条原始光谱，而不是把 744 点当作独立 Ridge 训练样本。
- 当前 T5/NoDrift、WithDrift 与四步管线的比较必须在预先固定的方案下报告；不能按元素事后挑选配置。
- N=6 下的精确索引置换检验可给出未校正的探索性信号，但 12 项比较经 Bonferroni 或 Benjamini-Hochberg FDR 校正后均不显著。报告 RMSE、偏差、置信区间和独立盲测比单独报告 R² 更重要。
- `four_step_spectral_transform.py` 是受跨仪器 LIBS 文献启发的启发式对照。它在重采样至 NASA 网格后执行平滑，且没有共测标样响应比矩阵；因此它不是已验证的仪器响应校正方法。

### 参考文献与数据来源

下表给出每篇参考文献与本仓库代码的具体关系，而非仅列出书目信息。项目的整体研究规划、问题定义、参考文献筛选与最终取舍均由项目作者独立完成。

| 参考文献/资源 | 在本项目中的用途 | 使用边界 |
| --- | --- | --- |
| [Clegg et al., 2017](https://doi.org/10.1016/j.sab.2016.12.003), *Recalibration of the Mars Science Laboratory ChemCam instrument with an expanded geochemical database* | ChemCam 标定数据、氧化物真值转换和跨域源域建模的科学背景。 | 本仓库使用的是本地匹配的数据子集；不能将论文中的完整标样库规模等同于当前 Ridge 的有效训练单位。 |
| [Jin et al., 2022](https://doi.org/10.3390/rs14163960), *A New Spectral Transformation Approach and Quantitative Analysis for MarSCoDe LIBS Data* | 说明跨仪器变换可涉及强度单位转换、波长重定标、插值和共测标样响应校正。 | 本仓库没有 MarSCoDe 的共测 Norite 响应比，因此四步代码仅受其思路启发，不是复现。 |
| [Jia et al., 2022](https://doi.org/10.3390/rs14235964), *Initial Drift Correction and Spectral Calibration of MarSCoDe LIBS on the Zhurong Rover* | 锚线匹配与漂移校正的物理动机。 | Dataset B 已包含波长列，`correct_drift=False` 是默认主线；漂移开启只作探索性对照。 |
| [Yu and Yao, 2023](https://doi.org/10.3390/rs15133422), *When CNNs Meet LIBS: End-to-End Quantitative Analysis Modeling of ChemCam Spectral Data for Major Elements Based on ECNN* | `train_calibration_ecnn.py` 与 ECNN 对照实验的网络设计参考。 | 本地小样本不足以证明深度模型优于简化的物理特征基线。 |
| [Roh et al., 2024](https://proceedings.mlr.press/v235/roh24a.html), *LEVI: Generalizable Fine-tuning via Layer-wise Ensemble of Different Views* | 双流/分层集成实验的通用迁移学习思路。 | 该论文不是 LIBS 领域方法验证；本仓库中的适配仅是探索性实现。 |
| [NIST Atomic Spectra Database](https://physics.nist.gov/asd) | `feature_extractor_v2.py` 中元素发射线与峰窗的物理参考。 | 谱线可见性仍取决于仪器覆盖、分辨率、基体和等离子体条件，不能因存在于 NIST 而假定可定量。 |
| [NASA PDS ChemCam derived data collection](https://pds.nasa.gov/ds-view/pds/viewCollection.jsp?identifier=urn%3Anasa%3Apds%3Amsl_chemcam_derived%3Acalib&version=1.0) | 可公开获取 ChemCam 衍生标定产品的入口。 | 下载、处理和再分发必须遵守 PDS 目录和产品说明。 |

### AI 协作与作者责任说明

本项目使用 Gemini、DeepSeek 与 ChatGPT 作为辅助工具，用于代码审阅与调试思路、文字表达与结构整理、文献检索线索和技术讨论。AI 生成或建议的内容均须经项目作者核查、修改和最终确认。

项目的研究目标、整体技术路线、实验与数据工作的安排、关键科学判断、结果解释、参考文献的选择与取舍，以及对公开内容的最终责任，均由项目作者完成并承担。AI 工具不是作者，也不替代实验、数据核验、统计判断或引用核查。

### 许可证

尚未选择开源许可证。代码目前仅用于审阅与研究交流；在仓库所有者添加许可证前，不授予额外的复制、修改或再分发许可。

---

## English

### Scope

This repository contains research code for laser-induced breakdown spectroscopy
(LIBS): local industrial-dust spectra and NASA ChemCam calibration spectra,
wavelength alignment, physically informed feature extraction, small-sample
quantitative regression, and calibration-transfer baselines.

It is an auditable research baseline, not a production quantitative-analysis
system. Dataset B contains only six independent physical samples; repeated laser
shots from a specimen are technical replicates rather than additional samples.

### Setup and data contract

Create the `dl1` Conda environment with `environment.yml`, then provide the
local data tree shown in the Chinese section above. Raw spectra, assay sheets,
measured labels, weights, and generated outputs are intentionally excluded.

`labels/unified_wt.example.csv` is a **header-only schema template**, not a
training file. Copy it to `labels/unified_wt.csv` and populate it only with
authorized measurements, following [labels/README.md](labels/README.md).

### Reproducible workflow

```powershell
conda env create -f environment.yml
conda activate dl1
python parse_and_unify_labels.py
python train_regression_v2.py
python train_transfer_benchmark.py
python generate_exact_permutation_csv.py
python verify_full_reproducibility.py
```

The default verification command checks generated artifacts. Use
`--full-rerun` to regenerate selected benchmarks from locally supplied data.

### Interpretation and references

The Chinese sections above provide the complete method map, interpretation
guardrails, linked references, data sources, and applicability limits. In
particular, T5 uses 59 aggregated ChemCam standards in the current code, the
N=6 results are exploratory after multiple-testing correction, and the
four-step transform is not a validated instrument-response correction.

### AI assistance and authorship

Gemini, DeepSeek, and ChatGPT were used as assistive tools for code review and
debugging ideas, writing and structural editing, literature-discovery leads,
and technical discussion. The project author reviewed, revised, and approved
all AI-assisted material.

The research aims, overall study plan, experimental and data-work planning,
scientific judgement, result interpretation, reference selection, and final
responsibility remain with the project author. AI tools are not authors and do
not replace experimental work, data validation, statistical judgement, or
reference verification.

### License

No open-source license has been selected. The source is shared for review and
research discussion only until the repository owner adds a license.
