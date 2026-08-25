"""
train_regression_v2.py

Rigorous Small-Sample LIBS Quantitative Regression Pipeline (Phase 1):
1. Loads verified ground-truth from labels/unified_wt.csv
2. Evaluates:
   - Mode 1: Dataset B ONLY (N=6 TR industrial dusts, 385-645nm)
   - Mode 2: Dataset A ONLY (N=16 verified pure reagents + industrial dusts, 240-640nm)
   - Mode 3: Combined A + B (Joint evaluation with separate coverage features)
3. Models:
   - Ridge Regression
   - Log1p-Transformed SVR (RBF kernel, y' = log(1+C))
   - Direct SVR (RBF kernel)
   - Lomakin-Scherbe 1D Internal Standard Fit
4. Protocol:
   - Leave-One-Sample-Out (LOSO) blind testing
   - 1000-iteration Permutation Test for statistical significance (p-value)
   - Non-negative prediction clipping
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_aligned_dataset_b, load_nasa_calibration_data
from feature_extractor_v2 import NISTFeatureExtractorV2

OUT_DIR = os.path.join(config.BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

REGRESSION_ELEMENTS_B = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn']
REGRESSION_ELEMENTS_A = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn', 'Mn', 'Ti', 'C']

def run_permutation_test(X, y, model_builder, n_permutations=1000, seed=42):
    """
    Computes empirical p-value for LOSO R2 via label permutation.
    """
    n_samples = len(y)
    if n_samples < 4:
        return 1.0
        
    logo = LeaveOneGroupOut()
    groups = np.arange(n_samples)
    
    # 1. True LOSO R2
    oof_true = np.zeros(n_samples)
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        m, scaler = model_builder()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        m.fit(X_tr, y[tr_idx])
        oof_true[te_idx] = m.predict(X_te)
    true_r2 = r2_score(y, np.maximum(oof_true, 0))
    if true_r2 <= 0:
        return 1.0
        
    # 2. Permutation loop
    rng = np.random.RandomState(seed)
    perm_r2s = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        oof_perm = np.zeros(n_samples)
        for tr_idx, te_idx in logo.split(X, y_perm, groups=groups):
            m, scaler = model_builder()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            m.fit(X_tr, y_perm[tr_idx])
            oof_perm[te_idx] = m.predict(X_te)
        r2_p = r2_score(y_perm, np.maximum(oof_perm, 0))
        perm_r2s.append(r2_p)
        
    p_val = (np.sum(np.array(perm_r2s) >= true_r2) + 1.0) / (n_permutations + 1.0)
    return float(p_val)

def evaluate_loso(X_feats, Y_targets, elem_names, dataset_name, sample_names):
    """
    Performs LOSO evaluation across all target elements with multiple regression architectures.
    """
    n_samples = len(X_feats)
    n_elems = len(elem_names)
    logo = LeaveOneGroupOut()
    groups = np.arange(n_samples)
    
    results = []
    
    print(f"\n{'='*90}")
    print(f"=== {dataset_name} LOSO Regression Evaluation (N = {n_samples} Samples) ===")
    print(f"{'='*90}")
    
    header = f"{'Element':<8} | {'Model':<16} | {'RMSE (wt%)':<12} | {'R2 Score':<10} | {'Perm p-val':<10} | {'Status'}"
    print(header)
    print("-" * 90)
    
    for e_idx, elem in enumerate(elem_names):
        y = Y_targets[:, e_idx]
        
        # Check if this element has meaningful non-zero variance
        if np.std(y) < 1e-4:
            print(f"{elem:<8} | {'All Models':<16} | {'0.000':<12} | {'N/A (Const)':<10} | {'--':<10} | All Zero Label")
            continue
            
        # Model 1: Standard Ridge Regression
        oof_ridge = np.zeros(n_samples)
        for tr_idx, te_idx in logo.split(X_feats, y, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feats[tr_idx])
            X_te = scaler.transform(X_feats[te_idx])
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y[tr_idx])
            oof_ridge[te_idx] = m.predict(X_te)
        oof_ridge = np.maximum(oof_ridge, 0)
        rmse_ridge = np.sqrt(mean_squared_error(y, oof_ridge))
        r2_ridge = r2_score(y, oof_ridge)
        p_ridge = run_permutation_test(X_feats, y, lambda: (Ridge(alpha=1.0), StandardScaler())) if r2_ridge > 0 else 1.0
        
        # Model 2: Direct SVR
        oof_svr = np.zeros(n_samples)
        for tr_idx, te_idx in logo.split(X_feats, y, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feats[tr_idx])
            X_te = scaler.transform(X_feats[te_idx])
            m = SVR(C=10.0, epsilon=0.1)
            m.fit(X_tr, y[tr_idx])
            oof_svr[te_idx] = m.predict(X_te)
        oof_svr = np.maximum(oof_svr, 0)
        rmse_svr = np.sqrt(mean_squared_error(y, oof_svr))
        r2_svr = r2_score(y, oof_svr)
        p_svr = run_permutation_test(X_feats, y, lambda: (SVR(C=10.0, epsilon=0.1), StandardScaler())) if r2_svr > 0 else 1.0

        # Model 3: Log1p-Transformed SVR (y' = log(1+C))
        oof_log_svr = np.zeros(n_samples)
        for tr_idx, te_idx in logo.split(X_feats, y, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feats[tr_idx])
            X_te = scaler.transform(X_feats[te_idx])
            y_tr_log = np.log1p(y[tr_idx])
            m = SVR(C=10.0, epsilon=0.05)
            m.fit(X_tr, y_tr_log)
            pred_log = m.predict(X_te)
            oof_log_svr[te_idx] = np.expm1(pred_log)
        oof_log_svr = np.maximum(oof_log_svr, 0)
        rmse_log_svr = np.sqrt(mean_squared_error(y, oof_log_svr))
        r2_log_svr = r2_score(y, oof_log_svr)
        p_log_svr = run_permutation_test(X_feats, np.log1p(y), lambda: (SVR(C=10.0, epsilon=0.05), StandardScaler())) if r2_log_svr > 0 else 1.0

        # Print summary for this element
        for m_name, rmse, r2, p_val in [
            ('Ridge', rmse_ridge, r2_ridge, p_ridge),
            ('SVR (Direct)', rmse_svr, r2_svr, p_svr),
            ('SVR (Log1p)', rmse_log_svr, r2_log_svr, p_log_svr)
        ]:
            flag = "[SIGNIFICANT]" if (r2 > 0 and p_val < 0.05) else ("[POSITIVE]" if r2 > 0 else "[COLLAPSE]")
            print(f"{elem:<8} | {m_name:<16} | {rmse:<12.3f} | {r2:<10.3f} | {p_val:<10.3f} | {flag}")
            results.append({
                'dataset': dataset_name,
                'element': elem,
                'model': m_name,
                'rmse_wt': rmse,
                'r2': r2,
                'permutation_p_value': p_val,
                'status': flag
            })
            
    return results

def main():
    # 1. Load Unified Ground Truth Labels
    labels_csv = os.path.join(config.BASE_DIR, 'labels', 'unified_wt.csv')
    if not os.path.exists(labels_csv):
        print("Error: labels/unified_wt.csv missing! Please run parse_and_unify_labels.py first.")
        return
    df_labels = pd.read_csv(labels_csv)
    
    # 2. Load Dataset B (TR samples)
    _, _, _, target_wvl = load_nasa_calibration_data()
    Xb, yb, names_b = load_aligned_dataset_b(target_wvl=target_wvl)
    
    # Aggregate Dataset B by physical TR sample
    sample_dict_b = {}
    for i, name in enumerate(names_b):
        s_id = name.split('_')[1]
        sample_dict_b.setdefault(s_id, []).append(Xb[i])
        
    snames_b = sorted(list(sample_dict_b.keys()))
    X_b_raw = np.array([np.mean(sample_dict_b[s], axis=0) for s in snames_b], dtype=np.float32)
    
    Y_b = np.zeros((len(snames_b), len(REGRESSION_ELEMENTS_B)), dtype=np.float32)
    for i, s in enumerate(snames_b):
        row = df_labels[(df_labels['sample_id'] == s) & (df_labels['dataset'] == 'B')]
        if not row.empty:
            Y_b[i] = [row.iloc[0][e] for e in REGRESSION_ELEMENTS_B]
            
    # Extract Features for Dataset B
    extractor_b = NISTFeatureExtractorV2(wavelengths=target_wvl, elements=REGRESSION_ELEMENTS_B)
    F_b, lines_b = extractor_b.transform(X_b_raw)
    print(f"Dataset B: {len(snames_b)} physical samples, {F_b.shape[1]} NIST features extracted from {len(lines_b)} lines.")

    # 3. Load Dataset A (Stitched 150-760nm)
    stitched_dir = os.path.join(config.BASE_DIR, '20260422_processed')
    X_a_raw_all = np.load(os.path.join(stitched_dir, 'X_stitched.npy'))
    names_a_all = np.load(os.path.join(stitched_dir, 'names.npy'))
    wvl_a = np.load(os.path.join(stitched_dir, 'wavelengths.npy'))
    
    # Filter only verified Dataset A samples
    valid_a_idx = []
    snames_a = []
    Y_a_list = []
    
    for i, name in enumerate(names_a_all):
        row = df_labels[(df_labels['sample_id'] == name) & (df_labels['dataset'] == 'A') & (df_labels['is_verified'] == True)]
        if not row.empty:
            valid_a_idx.append(i)
            snames_a.append(name)
            Y_a_list.append([row.iloc[0][e] for e in REGRESSION_ELEMENTS_A])
            
    X_a_raw = X_a_raw_all[valid_a_idx]
    Y_a = np.array(Y_a_list, dtype=np.float32)
    
    extractor_a = NISTFeatureExtractorV2(wavelengths=wvl_a, elements=REGRESSION_ELEMENTS_A)
    F_a, lines_a = extractor_a.transform(X_a_raw)
    print(f"Dataset A: {len(snames_a)} verified physical samples, {F_a.shape[1]} NIST features extracted from {len(lines_a)} lines.")

    all_results = []
    
    # 4. Run Mode 1: Dataset B ONLY
    res_b = evaluate_loso(F_b, Y_b, REGRESSION_ELEMENTS_B, "Mode 1: Dataset B ONLY (TR Industrial Dusts)", snames_b)
    all_results.extend(res_b)
    
    # 5. Run Mode 2: Dataset A ONLY
    res_a = evaluate_loso(F_a, Y_a, REGRESSION_ELEMENTS_A, "Mode 2: Dataset A ONLY (Pure Reagents + Industrial Dusts)", snames_a)
    all_results.extend(res_a)
    
    # 6. Save Full Comparison Report
    df_res = pd.DataFrame(all_results)
    out_csv = os.path.join(OUT_DIR, 'regression_v2_results.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"\nAll regression results successfully written to {out_csv}")

if __name__ == '__main__':
    main()
