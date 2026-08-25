"""
Dataset B ONLY Quantitative Regression with Spectral Coverage Awareness.

This script performs Leave-One-Sample-Out (LOSO) regression on Dataset B (TR samples)
using only NIST emission lines that have REAL spectral signal in the 385-644nm coverage.

Elements whose NIST lines fall outside the spectrometer coverage are explicitly excluded
and reported as "N/A — no signal coverage" rather than producing misleading negative R².
"""
import os
import sys
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_aligned_dataset_b, load_nasa_calibration_data
from model_dual_stream import NISTFeatureExtractor
from wavelet_preprocess import wavelet_denoise

print("=" * 80)
print("=== Dataset B ONLY: Coverage-Aware Quantitative Regression (TR-01 ~ TR-20) ===")
print("=" * 80)

REGRESSION_ELEMENTS = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn']

# Stoichiometry conversion factors
OXIDE_TO_ELEM_FACTOR = {
    'Fe2O3': (2 * 55.845) / 159.69,
    'ZnO': 65.38 / 81.38,
    'SiO2': 0.4674,
    'CaO': 0.7147,
    'MgO': 0.6031,
    'Al2O3': 0.5293
}

# 1. Load real Dataset B labels from 样品表.xlsx
excel_path = os.path.join(config.BASE_DIR, "2026.07.16wl_corr", "样品表.xlsx")
DATASET_B_WT_MAP = {}

def load_sample_table_native(fp):
    import zipfile
    from xml.etree import ElementTree as ET
    z = zipfile.ZipFile(fp)
    strings = []
    if 'xl/sharedStrings.xml' in z.namelist():
        stree = ET.parse(z.open('xl/sharedStrings.xml'))
        for t in stree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
            strings.append(t.text if t.text else '')
    tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows = tree.findall('.//x:row', ns)
    data = []
    for r in rows:
        row_data = []
        for c in r.findall('x:c', ns):
            t = c.attrib.get('t', '')
            v_elem = c.find('x:v', ns)
            v_val = v_elem.text if v_elem is not None else ''
            if t == 's' and v_val.isdigit():
                v_val = strings[int(v_val)]
            row_data.append(v_val)
        if row_data:
            data.append(row_data)
    return data

if os.path.exists(excel_path):
    table_data = load_sample_table_native(excel_path)
    header = [str(h).strip() for h in table_data[0]]
    col_idx = {h: i for i, h in enumerate(header)}
    for row in table_data[1:]:
        if len(row) <= max(col_idx.values()): continue
        def get_val_by_keyword(*keywords):
            for h, i in col_idx.items():
                for kw in keywords:
                    if kw.lower() in h.lower():
                        if len(row) > i:
                            try: return float(row[i])
                            except ValueError: return 0.0
            return 0.0
        sample_id = str(row[col_idx['样品编号']]).strip() if '样品编号' in col_idx else str(row[0]).strip()
        fe2o3 = get_val_by_keyword('Fe2O3', 'Fe₂O₃')
        zno = get_val_by_keyword('ZnO')
        sio2 = get_val_by_keyword('SiO2', 'SiO₂')
        caco3 = get_val_by_keyword('CaCO3', 'CaCO₃')
        cao = get_val_by_keyword('CaO')
        mgo = get_val_by_keyword('MgO')
        al2o3 = get_val_by_keyword('Al2O3', 'Al₂O₃')
        ca_wt = (caco3 * 0.4004) if caco3 > 0 else (cao * OXIDE_TO_ELEM_FACTOR['CaO'])
        DATASET_B_WT_MAP[sample_id] = {
            'Fe': fe2o3 * OXIDE_TO_ELEM_FACTOR['Fe2O3'],
            'Zn': zno * OXIDE_TO_ELEM_FACTOR['ZnO'],
            'Si': sio2 * OXIDE_TO_ELEM_FACTOR['SiO2'],
            'Ca': ca_wt,
            'Mg': mgo * OXIDE_TO_ELEM_FACTOR['MgO'],
            'Al': al2o3 * OXIDE_TO_ELEM_FACTOR['Al2O3']
        }

# 2. Load aligned Dataset B ONLY
_, _, _, target_wvl = load_nasa_calibration_data()
Xb, yb, names_b = load_aligned_dataset_b(target_wvl=target_wvl)

raw_sample_dict = {}
for i in range(len(Xb)):
    name = names_b[i]
    sub = name.split('_')[1]
    if sub in DATASET_B_WT_MAP:
        if sub not in raw_sample_dict:
            raw_sample_dict[sub] = {"spectra": [], "wt": [DATASET_B_WT_MAP[sub][e] for e in REGRESSION_ELEMENTS]}
        raw_sample_dict[sub]["spectra"].append(Xb[i])

sample_names = sorted(list(raw_sample_dict.keys()))
X_samples = np.zeros((len(sample_names), len(target_wvl)), dtype=np.float32)
Y_samples = np.zeros((len(sample_names), len(REGRESSION_ELEMENTS)), dtype=np.float32)

for i, sname in enumerate(sample_names):
    X_samples[i] = np.mean(raw_sample_dict[sname]["spectra"], axis=0)
    Y_samples[i] = raw_sample_dict[sname]["wt"]

print(f"\nDataset B Sample Aggregation Complete: N = {len(sample_names)} UNIQUE TR Physical Samples.")
for i, sname in enumerate(sample_names):
    print(f"  Sample {i+1:02d}: {sname:<8} (Aggregated from {len(raw_sample_dict[sname]['spectra'])} shots)")

# 3. Crop to valid wavelength range and SNV normalize
valid_wvl_mask = (target_wvl >= 240.0) & (target_wvl <= 760.0)
target_wvl_valid = target_wvl[valid_wvl_mask]
X_samples_valid = X_samples[:, valid_wvl_mask]

X_snv = np.zeros_like(X_samples_valid)
for i in range(len(X_samples_valid)):
    spec_mean = np.mean(X_samples_valid[i])
    spec_std = np.std(X_samples_valid[i]) + 1e-8
    X_snv[i] = (X_samples_valid[i] - spec_mean) / spec_std

X_samples_clean = np.nan_to_num(X_snv, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nApplying SNV Preprocessing & Wavelet Denoising ({target_wvl_valid.min():.1f}nm - {target_wvl_valid.max():.1f}nm)...")
X_denoised = wavelet_denoise(X_samples_clean, wavelet='db4', level=1, mode='soft')

# 4. COVERAGE-AWARE NIST Feature Extraction
extractor = NISTFeatureExtractor(wavelengths=target_wvl_valid)

# Run coverage diagnostic on RAW spectrum (before SNV, so zero-padded regions remain 0.0)
valid_elements, blind_elements = extractor.get_coverage_report(X_samples_valid)

# Extract only covered features
F_covered, covered_lines = extractor.transform_covered(X_denoised)

print(f"\nExtracted {F_covered.shape[1]} COVERED features (from {len(covered_lines)} NIST lines with signal).")

# 5. Determine which elements are modelable
modelable_elements = []
non_modelable_elements = []
for elem in REGRESSION_ELEMENTS:
    if elem in valid_elements:
        modelable_elements.append(elem)
    else:
        non_modelable_elements.append(elem)

print(f"\nModelable elements (have NIST lines in 385-644nm): {modelable_elements}")
if non_modelable_elements:
    print(f"Non-modelable elements (NO NIST lines in coverage): {non_modelable_elements}")

# 6. Leave-One-Sample-Out (LOSO) Cross-Validation — ONLY for modelable elements
logo = LeaveOneGroupOut()
groups = np.arange(len(sample_names))

# Build Y matrix for modelable elements only
modelable_indices = [REGRESSION_ELEMENTS.index(e) for e in modelable_elements]
Y_modelable = Y_samples[:, modelable_indices]

oof_pred_svr = np.zeros_like(Y_modelable)
oof_pred_ridge = np.zeros_like(Y_modelable)

for train_idx, test_idx in logo.split(F_covered, Y_modelable, groups=groups):
    F_tr, Y_tr = F_covered[train_idx], Y_modelable[train_idx]
    F_te, Y_te = F_covered[test_idx], Y_modelable[test_idx]

    # Per-fold feature selection: variance mask is fit on train only.
    fold_var_mask = np.var(F_tr, axis=0) > 1e-6
    if not fold_var_mask.any():
        fold_var_mask = np.ones(F_tr.shape[1], dtype=bool)
    F_tr = F_tr[:, fold_var_mask]
    F_te = F_te[:, fold_var_mask]

    scaler = StandardScaler()
    F_tr_scaled = scaler.fit_transform(F_tr)
    F_te_scaled = scaler.transform(F_te)

    ridge = Ridge(alpha=10.0)
    ridge.fit(F_tr_scaled, Y_tr)
    oof_pred_ridge[test_idx] = np.maximum(ridge.predict(F_te_scaled), 0.0)

    svr = MultiOutputRegressor(SVR(C=10.0, epsilon=0.1))
    svr.fit(F_tr_scaled, Y_tr)
    oof_pred_svr[test_idx] = np.maximum(svr.predict(F_te_scaled), 0.0)

# 7. Evaluation Results
print("\n" + "=" * 85)
print("=== Dataset B ONLY: Coverage-Aware LOSO Evaluation (N={}) ===".format(len(sample_names)))
print("=" * 85)
print(f"{'Element':<10} | {'Coverage':<12} | {'Ridge RMSE':<12} | {'Ridge R^2':<10} | {'SVR RMSE':<12} | {'SVR R^2':<10}")
print("-" * 85)

for elem in REGRESSION_ELEMENTS:
    if elem in modelable_elements:
        idx = modelable_elements.index(elem)
        r_rmse = np.sqrt(mean_squared_error(Y_modelable[:, idx], oof_pred_ridge[:, idx]))
        r_r2 = r2_score(Y_modelable[:, idx], oof_pred_ridge[:, idx])
        s_rmse = np.sqrt(mean_squared_error(Y_modelable[:, idx], oof_pred_svr[:, idx]))
        s_r2 = r2_score(Y_modelable[:, idx], oof_pred_svr[:, idx])
        print(f"{elem:<10} | {'COVERED':<12} | {r_rmse:<12.3f} | {r_r2:<10.3f} | {s_rmse:<12.3f} | {s_r2:<10.3f}")
    else:
        print(f"{elem:<10} | {'NO SIGNAL':<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':<12} | {'N/A':<10}")

print("=" * 85)

# 8. Per-sample Residual Report (only for modelable elements)
print("\n=== Per-Sample Residual Report (SVR, modelable elements only) ===")
for i, sname in enumerate(sample_names):
    print(f"\n--- Sample: {sname} ---")
    for j, elem in enumerate(modelable_elements):
        orig_idx = REGRESSION_ELEMENTS.index(elem)
        truth = Y_samples[i, orig_idx]
        pred = oof_pred_svr[i, j]
        err = pred - truth
        print(f"  {elem:<4}: True = {truth:6.2f} wt% | SVR Pred = {pred:6.2f} wt% | Residual = {err:+6.2f} wt%")
