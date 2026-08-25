"""
generate_exact_permutation_csv.py

Computes exact 720-index-permutation p-values and unadjusted/FDR/Bonferroni statistics
for T5 (NIST-SBC) under both NoDrift and WithDrift configurations.
Saves results to output/phase2_exact_permutation_results.csv.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from itertools import permutations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import config
from dataset_loader import load_aligned_dataset_b, load_nasa_calibration_data
from feature_extractor_v2 import NISTFeatureExtractorV2
from eval_nasa_feasibility import load_nasa_regression_data, aggregate_by_standard, REGRESSION_ELEMENTS as NASA_ELEM

ELEMS = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn']

def run_exact_permutations():
    print("=== Computing Exact 720 Index-Permutation Benchmark ===")
    # 1. Load NASA Source Data (aggregated by standard, 59 unique standards)
    X_nasa_raw, Y_nasa_raw, std_names, wvl_nasa = load_nasa_regression_data()
    X_nasa, Y_nasa, _ = aggregate_by_standard(X_nasa_raw, Y_nasa_raw, std_names)
    mask_b = (wvl_nasa >= 385.0) & (wvl_nasa <= 645.0)
    wvl_b_win = wvl_nasa[mask_b]
    X_nasa_crop = X_nasa[:, mask_b]
    
    nasa_map = [NASA_ELEM.index(e) if e in NASA_ELEM else -1 for e in ELEMS]
    Y_nasa_b = np.zeros((len(Y_nasa), len(ELEMS)), dtype=np.float32)
    for j, idx in enumerate(nasa_map):
        if idx >= 0:
            Y_nasa_b[:, j] = Y_nasa[:, idx]

    # 2. Load Local Dataset B Target Data (N=6 physical samples)
    df_labels = pd.read_csv(os.path.join(BASE_DIR, 'labels', 'unified_wt.csv'))
    _, _, _, target_wvl = load_nasa_calibration_data()
    Xb_all, _, names_b = load_aligned_dataset_b(target_wvl=target_wvl)
    sample_dict_b = {}
    for i, name in enumerate(names_b):
        s_id = name.split('_')[1]
        sample_dict_b.setdefault(s_id, []).append(Xb_all[i])
    snames_b = sorted(sample_dict_b.keys())
    X_b_full = np.array([np.mean(sample_dict_b[s], axis=0) for s in snames_b], dtype=np.float32)
    X_b_crop = X_b_full[:, mask_b]
    Y_b = np.zeros((len(snames_b), len(ELEMS)), dtype=np.float32)
    for i, s in enumerate(snames_b):
        row = df_labels[(df_labels['sample_id'] == s) & (df_labels['dataset'] == 'B')]
        Y_b[i] = [row.iloc[0][e] for e in ELEMS]

    # 3. Extract Features
    ext = NISTFeatureExtractorV2(wavelengths=wvl_b_win, elements=ELEMS)
    F_nasa_off, _ = ext.transform(X_nasa_crop, correct_drift=False)
    F_b_off, _ = ext.transform(X_b_crop, correct_drift=False)
    F_nasa_on, _ = ext.transform(X_nasa_crop, correct_drift=True)
    F_b_on, _ = ext.transform(X_b_crop, correct_drift=True)

    logo = LeaveOneGroupOut()
    groups = np.arange(len(snames_b))
    
    # Pre-generate all 720 index permutations
    all_perms = list(permutations(range(len(snames_b))))
    n_perms = len(all_perms)  # 720
    
    results = []

    for config_name, F_nasa, F_b in [('NoDrift', F_nasa_off, F_b_off), ('WithDrift', F_nasa_on, F_b_on)]:
        sc = StandardScaler()
        F_nasa_s = sc.fit_transform(F_nasa)
        F_b_s = sc.transform(F_b)
        
        for e_i, elem in enumerate(ELEMS):
            y_true = Y_b[:, e_i]
            
            # Train base Ridge once on source standards
            m = Ridge(alpha=10.0)
            m.fit(F_nasa_s, Y_nasa_b[:, e_i])
            pred_b_raw = m.predict(F_b_s)
            
            # Closed-form LOSO SBC for true labels
            oof_true = np.zeros(len(y_true))
            for tr_idx, te_idx in logo.split(F_b, y_true, groups=groups):
                p_tr = pred_b_raw[tr_idx]
                if np.std(p_tr) > 1e-4:
                    sl, bi = np.polyfit(p_tr, y_true[tr_idx], 1)
                else:
                    sl, bi = 1.0, float(np.mean(y_true[tr_idx]))
                oof_true[te_idx] = sl * pred_b_raw[te_idx] + bi
                
            pred_true_clip = np.maximum(oof_true, 0)
            r2_true = r2_score(y_true, pred_true_clip)
            rmse_true = np.sqrt(mean_squared_error(y_true, pred_true_clip))
            
            # Exact 720 index-permutations
            cnt = 0
            for perm in all_perms:
                yp = y_true[list(perm)]
                oof_p = np.zeros(len(yp))
                for tr_idx, te_idx in logo.split(F_b, yp, groups=groups):
                    p_tr = pred_b_raw[tr_idx]
                    if np.std(p_tr) > 1e-4:
                        sl, bi = np.polyfit(p_tr, yp[tr_idx], 1)
                    else:
                        sl, bi = 1.0, float(np.mean(yp[tr_idx]))
                    oof_p[te_idx] = sl * pred_b_raw[te_idx] + bi
                r2_p = r2_score(yp, np.maximum(oof_p, 0))
                if r2_p >= r2_true:
                    cnt += 1
                    
            p_exact = (cnt + 1.0) / (n_perms + 1.0)
            unadjusted_signal = (p_exact < 0.05) and (r2_true > 0)
            
            results.append({
                'configuration': config_name,
                'element': elem,
                'r2_true': r2_true,
                'rmse_true': rmse_true,
                'permutation_beats': cnt,
                'total_index_permutations': n_perms,
                'p_exact_index': p_exact,
                'unadjusted_exploratory_signal': unadjusted_signal,
                'source_training_units': len(X_nasa),
                'source_raw_spectra': len(X_nasa_raw),
                'target_samples_N': len(snames_b)
            })

    df_res = pd.DataFrame(results)
    
    # Compute Bonferroni and Benjamini-Hochberg FDR across all 12 tests
    m_tests = len(df_res)
    df_res['p_bonferroni'] = np.minimum(df_res['p_exact_index'] * m_tests, 1.0)
    
    # BH FDR
    sorted_idx = np.argsort(df_res['p_exact_index'].values)
    ranks = np.arange(1, m_tests + 1)
    p_sorted = df_res['p_exact_index'].values[sorted_idx]
    fdr_sorted = p_sorted * m_tests / ranks
    # enforce monotonicity
    for k in range(m_tests - 2, -1, -1):
        fdr_sorted[k] = min(fdr_sorted[k], fdr_sorted[k+1])
    fdr_sorted = np.minimum(fdr_sorted, 1.0)
    
    df_res['p_bh_fdr'] = 1.0
    for orig_pos, fdr_val in zip(sorted_idx, fdr_sorted):
        df_res.loc[orig_pos, 'p_bh_fdr'] = fdr_val
        
    out_csv = os.path.join(BASE_DIR, 'output', 'phase2_exact_permutation_results.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved exact permutation results to: {out_csv}")
    print(df_res[['configuration', 'element', 'r2_true', 'rmse_true', 'p_exact_index', 'p_bonferroni', 'p_bh_fdr', 'unadjusted_exploratory_signal']].to_string())

if __name__ == '__main__':
    run_exact_permutations()
