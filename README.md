# LIBS Spectral Analysis Toolkit

Research code for laser-induced breakdown spectroscopy (LIBS) preprocessing,
spectral feature extraction, and small-sample quantitative calibration transfer.
The repository combines local industrial-dust spectra with NASA ChemCam
calibration spectra for reproducible, group-aware benchmark experiments.

> **Research status:** this is an experimental analysis toolkit, not an
> industrial deployment package. The local transfer benchmark has six
> independent target samples, so reported results are exploratory and must not
> be interpreted as validated production performance.

## Included methods

- Physical wavelength alignment and multi-window spectrum loading.
- SNIP baseline subtraction, wavelength-shift estimation, and coverage-aware
  NIST emission-line features.
- Leave-one-sample-out (LOSO) Ridge/SVR benchmarks for local spectra.
- ChemCam-to-local NIST-feature slope/bias calibration (SBC) transfer baseline.
- Exact **index-permutation** tests for the N=6 transfer experiment, with
  Bonferroni and Benjamini-Hochberg corrections.
- Historical ECNN and dual-stream model implementations for comparison.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config.py` | Project paths, target elements, and default hyperparameters. |
| `dataset_loader.py` | NASA/local spectrum readers, alignment, and augmentation utilities. |
| `feature_extractor_v2.py` | SNIP preprocessing and coverage-aware NIST features. |
| `train_regression_v2.py` | Local Dataset A/B LOSO Ridge and SVR benchmarks. |
| `train_transfer_benchmark.py` | Cross-domain neural baselines and T5 NIST-SBC transfer benchmark. |
| `generate_exact_permutation_csv.py` | Exact index-permutation statistics for T5. |
| `four_step_spectral_transform.py` | Exploratory four-step cross-instrument preprocessing benchmark. |
| `model_dual_stream.py`, `train_dual_stream_ecnn.py` | Historical dual-stream ECNN experiments. |
| `environment.yml` | Conda environment definition (`dl1`). |
| `labels/unified_wt.example.csv` | Required label-table schema; contains no measurements. |

## Environment

```powershell
conda env create -f environment.yml
conda activate dl1
```

The code has been developed with Python 3.10, NumPy, Pandas, SciPy,
scikit-learn, and PyTorch. CUDA is optional; the CPU-only environment definition
is deliberate for reproducible baseline execution.

## Data setup

Raw spectra, assay tables, derived labels, trained weights, and experiment
outputs are intentionally excluded. They may be confidential, large, or subject
to third-party distribution terms. Create the following layout locally before
running the full workflows:

```text
NASA data/
  msl_ccam_libs_calib.csv
  ccam_calibration_compositions.csv
2026.07.16wl_corr/
  <TR sample folders and wavelength-calibrated xlsx files>
20260422data/
20260422数据标注/
labels/
  unified_wt.csv
```

Use `labels/unified_wt.example.csv` as the schema reference. NASA ChemCam
products should be retrieved from the relevant NASA PDS collections and used in
accordance with their distribution terms. Local assay labels must be supplied by
the data owner and are not included in this repository.

## Typical workflow

```powershell
# Build labels from locally available assay files.
python parse_and_unify_labels.py

# Run local coverage-aware LOSO baselines.
python train_regression_v2.py

# Run transfer baselines and write benchmark CSVs.
python train_transfer_benchmark.py

# Recompute exact index-permutation results for the six-sample target set.
python generate_exact_permutation_csv.py

# Check generated benchmark artifacts.
python verify_full_reproducibility.py
```

`verify_full_reproducibility.py --full-rerun` reruns the selected benchmark
pipelines from the locally supplied raw data. The default command is an artifact
consistency check, not a substitute for an independent reproduction.

## Interpretation safeguards

- The T5 source model is trained after aggregating ChemCam spectra by standard;
  it is not a 744-independent-observation Ridge fit.
- Dataset B has N=6 independent physical samples. Ten shots from the same
  specimen are technical replicates, not independent samples.
- Nominal exact-permutation signals do not survive the 12-test Bonferroni or
  BH-FDR correction. Treat them as directions for new calibration data, not as
  validated analytical capability.
- The four-step transform is an exploratory, self-developed preprocessing
  comparison inspired by cross-instrument LIBS literature. It is not a
  validated instrument-response correction without common reference standards.

## References

- Clegg et al. *Spectrochimica Acta Part B* 129, 64-85 (2017).
- Jin et al. *Remote Sensing* 14, 3960 (2022).
- Jia et al. *Remote Sensing* 14, 5964 (2022).

## License

No open-source license has been selected yet. Source code is shared for review
and research discussion only until a license is added by the repository owner.
