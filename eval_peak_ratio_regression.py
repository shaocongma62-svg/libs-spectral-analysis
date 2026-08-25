"""
eval_peak_ratio_regression.py

Comprehensive Physical Evaluation and Implementation of Peak Height Ratio (Internal Standard / Lomakin-Scherbe)
Quantitative LIBS Models under Linear and Non-Linear Plasma Regimes.

Models Implemented and Evaluated under Strict LOSO:
1. Classical Univariate Linear Internal Standard: y = a * (I_analyte / I_IS) + b
2. Lomakin-Scherbe Power-Law: log(y) = b * log(I_analyte / I_IS) + log(a)  ->  y = a * R^b
3. Non-Linear Polynomial Internal Standard: y = a * R^2 + b * R + c
4. Multivariate Multi-Ratio Ridge / SVR (RBF Kernel overcoming self-absorption)
5. Cross-Domain NASA Pre-trained Transfer + Multi-Ratio SBC (T5-Ratio)

Saves:
- output/peak_ratio_regression_results.csv
- output/peak_ratio_calibration_curves.png
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import config
from dataset_loader import load_aligned_dataset_b, load_nasa_calibration_data
from feature_extractor_v2 import NISTFeatureExtractorV2, snip_baseline_physical
from eval_nasa_feasibility import load_nasa_regression_data, aggregate_by_standard, REGRESSION_ELEMENTS as NASA_ELEM

OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

ELEMS = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn']

# Key atomic emission lines for direct univariate ratio testing
PRIMARY_LINES = {
    'Ca': {'analyte': 393.37, 'is_ref': 259.94, 'is_name': 'Fe II 259.94', 'fallback_is': 288.16, 'fallback_name': 'Si I 288.16'},
    'Si': {'analyte': 288.16, 'is_ref': 259.94, 'is_name': 'Fe II 259.94', 'fallback_is': 422.67, 'fallback_name': 'Ca I 422.67'},
    'Fe': {'analyte': 404.58, 'is_ref': 422.67, 'is_name': 'Ca I 422.67', 'fallback_is': 288.16, 'fallback_name': 'Si I 288.16'},
    'Mg': {'analyte': 518.36, 'is_ref': 259.94, 'is_name': 'Fe II 259.94', 'fallback_is': 422.67, 'fallback_name': 'Ca I 422.67'},
    'Al': {'analyte': 394.40, 'is_ref': 259.94, 'is_name': 'Fe II 259.94', 'fallback_is': 422.67, 'fallback_name': 'Ca I 422.67'},
    'Zn': {'analyte': 481.05, 'is_ref': 259.94, 'is_name': 'Fe II 259.94', 'fallback_is': 422.67, 'fallback_name': 'Ca I 422.67'}
}

def extract_net_peak(wvl, spec_net, center_wl, half_win=0.35):
    m = (wvl >= center_wl - half_win) & (wvl <= center_wl + half_win)
    if not np.any(m):
        return 0.0
    return float(np.max(spec_net[m]))

def robust_polyfit_1d(x_tr, y_tr, x_te, deg=1):
    x_tr = np.asarray(x_tr, dtype=np.float64).ravel()
    y_tr = np.asarray(y_tr, dtype=np.float64).ravel()
    x_te = np.asarray(x_te, dtype=np.float64).ravel()
    
    if np.std(x_tr) < 1e-6 or np.std(y_tr) < 1e-6:
        return float(np.mean(y_tr)) * np.ones_like(x_te)
        
    try:
        if deg == 1:
            lr = LinearRegression()
            lr.fit(x_tr.reshape(-1, 1), y_tr)
            return lr.predict(x_te.reshape(-1, 1))
        elif deg == 2:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            x_tr_p = poly.fit_transform(x_tr.reshape(-1, 1))
            x_te_p = poly.transform(x_te.reshape(-1, 1))
            ridge = Ridge(alpha=1.0)
            ridge.fit(x_tr_p, y_tr)
            return ridge.predict(x_te_p)
    except Exception:
        return float(np.mean(y_tr)) * np.ones_like(x_te)
    return float(np.mean(y_tr)) * np.ones_like(x_te)

def robust_powerlaw_fit(r_tr, y_tr, r_te):
    r_tr = np.maximum(np.asarray(r_tr, dtype=np.float64).ravel(), 1e-4)
    y_tr = np.maximum(np.asarray(y_tr, dtype=np.float64).ravel(), 1e-4)
    r_te = np.maximum(np.asarray(r_te, dtype=np.float64).ravel(), 1e-4)
    
    log_r_tr = np.log(r_tr)
    log_y_tr = np.log(y_tr)
    
    if np.std(log_r_tr) < 1e-6:
        return float(np.mean(y_tr)) * np.ones_like(r_te)
        
    lr = LinearRegression()
    lr.fit(log_r_tr.reshape(-1, 1), log_y_tr)
    log_pred = lr.predict(np.log(r_te).reshape(-1, 1))
    return np.exp(log_pred)

def main():
    print("=" * 105)
    print("=== Empirical & Physical Evaluation of Peak Height Ratio (Internal Standard) Models ===")
    print("=" * 105)

    # 1. Load Local Dataset B Target Data
    df_labels = pd.read_csv(os.path.join(BASE_DIR, 'labels', 'unified_wt.csv'))
    _, _, _, target_wvl = load_nasa_calibration_data()
    Xb_all, _, names_b = load_aligned_dataset_b(target_wvl=target_wvl)
    
    mask_b = (target_wvl >= 385.0) & (target_wvl <= 645.0)
    wvl_b = target_wvl[mask_b]
    
    sample_dict_b = {}
    for i, name in enumerate(names_b):
        s_id = name.split('_')[1]
        sample_dict_b.setdefault(s_id, []).append(Xb_all[i])
    snames_b = sorted(list(sample_dict_b.keys()))
    
    # 10-shot average per physical sample
    X_b_raw = np.array([np.mean(sample_dict_b[s], axis=0) for s in snames_b], dtype=np.float32)
    X_b_crop = X_b_raw[:, mask_b]
    
    # Compute SNIP net spectrum for all B samples
    X_b_net = np.zeros_like(X_b_crop)
    for i in range(len(X_b_crop)):
        base = snip_baseline_physical(X_b_crop[i], wvl_b, window_nm=1.5)
        X_b_net[i] = np.maximum(X_b_crop[i] - base, 0.0)

    Y_b = np.zeros((len(snames_b), len(ELEMS)), dtype=np.float32)
    for i, s in enumerate(snames_b):
        row = df_labels[(df_labels['sample_id'] == s) & (df_labels['dataset'] == 'B')]
        Y_b[i] = [row.iloc[0][e] for e in ELEMS]

    # 2. Extract Univariate Peak Ratios for each element
    ratio_dict = {}
    peak_dict = {}
    for elem in ELEMS:
        info = PRIMARY_LINES[elem]
        r_list = []
        p_list = []
        for i in range(len(snames_b)):
            i_analyte = extract_net_peak(wvl_b, X_b_net[i], info['analyte'])
            i_is = extract_net_peak(wvl_b, X_b_net[i], info['is_ref'])
            if i_is < 10.0:
                i_is = extract_net_peak(wvl_b, X_b_net[i], info['fallback_is'])
            r = i_analyte / max(i_is, 1.0)
            r_list.append(r)
            p_list.append(i_analyte)
        ratio_dict[elem] = np.array(r_list, dtype=np.float32)
        peak_dict[elem] = np.array(p_list, dtype=np.float32)

    # 3. Load NASA Source Data for Multivariate Ratio Transfer
    X_nasa_raw, Y_nasa_raw, std_names, wvl_nasa = load_nasa_regression_data()
    X_nasa, Y_nasa, _ = aggregate_by_standard(X_nasa_raw, Y_nasa_raw, std_names)
    X_nasa_crop = X_nasa[:, mask_b]
    
    nasa_map = [NASA_ELEM.index(e) if e in NASA_ELEM else -1 for e in ELEMS]
    Y_nasa_b = np.zeros((len(Y_nasa), len(ELEMS)), dtype=np.float32)
    for j, idx in enumerate(nasa_map):
        if idx >= 0: Y_nasa_b[:, j] = Y_nasa[:, idx]

    # Extract NIST features (which include Lomakin-Scherbe Ratios & Self-Absorption Ratios)
    ext = NISTFeatureExtractorV2(wavelengths=wvl_b, elements=ELEMS)
    F_nasa, _ = ext.transform(X_nasa_crop, correct_drift=False)
    F_b, _ = ext.transform(X_b_crop, correct_drift=False)

    # Extract pure Ratio Columns from NIST feature matrix (column 4 for each line: is_ratio + self-absorption ratio)
    n_lines = 24
    ratio_col_indices = [k * 6 + 4 for k in range(n_lines)] + [n_lines * 6]
    F_nasa_ratios = F_nasa[:, ratio_col_indices]
    F_b_ratios = F_b[:, ratio_col_indices]

    # =========================================================================
    # 4. Strict LOSO Evaluation across 5 Method Paradigms
    # =========================================================================
    logo = LeaveOneGroupOut()
    groups = np.arange(len(snames_b))
    n_samples = len(snames_b)

    results_table = []
    plot_data = {}

    for e_i, elem in enumerate(ELEMS):
        y_true = Y_b[:, e_i]
        R = ratio_dict[elem]
        P = peak_dict[elem]

        oof_raw_peak = np.zeros(n_samples)
        oof_linear_is = np.zeros(n_samples)
        oof_power_is = np.zeros(n_samples)
        oof_poly_is = np.zeros(n_samples)
        oof_ratio_t5 = np.zeros(n_samples)

        for tr_idx, te_idx in logo.split(R, y_true, groups=groups):
            # 1. Raw Peak Height (No Internal Standard)
            oof_raw_peak[te_idx] = robust_polyfit_1d(P[tr_idx], y_true[tr_idx], P[te_idx], deg=1)

            # 2. Linear Internal Standard: y = a * R + b
            oof_linear_is[te_idx] = robust_polyfit_1d(R[tr_idx], y_true[tr_idx], R[te_idx], deg=1)

            # 3. Lomakin-Scherbe Power Law: log(y) = b * log(R) + log(a)
            oof_power_is[te_idx] = robust_powerlaw_fit(R[tr_idx], y_true[tr_idx], R[te_idx])

            # 4. Polynomial (Quadratic Non-Linear) Internal Standard: y = a*R^2 + b*R + c
            oof_poly_is[te_idx] = robust_polyfit_1d(R[tr_idx], y_true[tr_idx], R[te_idx], deg=2)

            # 5. Multivariate Multi-Ratio Transfer (NASA Ridge + SBC on pure Ratio features)
            scaler_r = StandardScaler()
            Fn_r_s = scaler_r.fit_transform(F_nasa_ratios)
            m_r = Ridge(alpha=10.0)
            m_r.fit(Fn_r_s, Y_nasa_b[:, e_i])
            pred_tr = m_r.predict(scaler_r.transform(F_b_ratios[tr_idx]))
            if np.std(pred_tr) > 1e-4:
                sl_t5, bi_t5 = np.polyfit(pred_tr, y_true[tr_idx], 1)
            else:
                sl_t5, bi_t5 = 1.0, float(np.mean(y_true[tr_idx]))
            oof_ratio_t5[te_idx] = sl_t5 * m_r.predict(scaler_r.transform(F_b_ratios[te_idx])) + bi_t5

        # Compute metrics
        def score(pred):
            c = np.maximum(pred, 0)
            return r2_score(y_true, c), np.sqrt(mean_squared_error(y_true, c)), c

        r2_raw, rmse_raw, p_raw = score(oof_raw_peak)
        r2_lin, rmse_lin, p_lin = score(oof_linear_is)
        r2_pow, rmse_pow, p_pow = score(oof_power_is)
        r2_poly, rmse_poly, p_poly = score(oof_poly_is)
        r2_t5r, rmse_t5r, p_t5r = score(oof_ratio_t5)

        results_table.append({
            'element': elem,
            'primary_line_nm': PRIMARY_LINES[elem]['analyte'],
            'is_line_nm': PRIMARY_LINES[elem]['is_name'],
            'Raw_Peak_R2': r2_raw, 'Raw_Peak_RMSE': rmse_raw,
            'Linear_IS_R2': r2_lin, 'Linear_IS_RMSE': rmse_lin,
            'PowerLaw_IS_R2': r2_pow, 'PowerLaw_IS_RMSE': rmse_pow,
            'Poly_NonLin_IS_R2': r2_poly, 'Poly_NonLin_IS_RMSE': rmse_poly,
            'MultiRatio_T5_R2': r2_t5r, 'MultiRatio_T5_RMSE': rmse_t5r
        })

        plot_data[elem] = {
            'y_true': y_true,
            'R': R,
            'pred_linear': p_lin,
            'pred_power': p_pow,
            'pred_t5_ratio': p_t5r
        }

    df_res = pd.DataFrame(results_table)
    out_csv = os.path.join(OUT_DIR, 'peak_ratio_regression_results.csv')
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 125)
    print(f"{'Element':<8} | {'Raw Peak (No IS)':<18} | {'Linear IS Ratio':<18} | {'Power-Law IS (b!=1)':<22} | {'Non-Lin Poly IS':<18} | {'Multi-Ratio T5':<18}")
    print(f"{'':<8} | {'R2 (RMSE wt%)':<18} | {'R2 (RMSE wt%)':<18} | {'R2 (RMSE wt%)':<22} | {'R2 (RMSE wt%)':<18} | {'R2 (RMSE wt%)':<18}")
    print("-" * 125)
    for row in results_table:
        s_raw = f"{row['Raw_Peak_R2']:+.3f} ({row['Raw_Peak_RMSE']:.2f})"
        s_lin = f"{row['Linear_IS_R2']:+.3f} ({row['Linear_IS_RMSE']:.2f})"
        s_pow = f"{row['PowerLaw_IS_R2']:+.3f} ({row['PowerLaw_IS_RMSE']:.2f})"
        s_ply = f"{row['Poly_NonLin_IS_R2']:+.3f} ({row['Poly_NonLin_IS_RMSE']:.2f})"
        s_t5r = f"{row['MultiRatio_T5_R2']:+.3f} ({row['MultiRatio_T5_RMSE']:.2f})"
        print(f"{row['element']:<8} | {s_raw:<18} | {s_lin:<18} | {s_pow:<22} | {s_ply:<18} | {s_t5r:<18}")
    print("=" * 125)
    print(f"Results saved to {out_csv}")

    # =========================================================================
    # 5. Diagnostic Plotting: Calibration Curves & Non-Linear Power-Law Mapping
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, elem in enumerate(ELEMS):
        ax = axes[idx]
        d = plot_data[elem]
        yt = d['y_true']
        R = d['R']
        
        # Scatter plot of Concentration vs Ratio
        ax.scatter(R, yt, color='navy', s=80, zorder=5, label='Actual Samples (N=6)')
        for s_i, sn in enumerate(snames_b):
            ax.annotate(sn, (R[s_i], yt[s_i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

        # Plot calibration curve lines
        r_grid = np.linspace(max(min(R)*0.7, 1e-4), max(R)*1.2, 100)
        
        # Linear fit
        lr_all = LinearRegression().fit(R.reshape(-1, 1), yt)
        y_lin_grid = lr_all.predict(r_grid.reshape(-1, 1))
        r2_lin_val = df_res.loc[df_res["element"]==elem, "Linear_IS_R2"].values[0]
        ax.plot(r_grid, y_lin_grid, color='crimson', linestyle='--', label=f'Linear: LOSO R²={r2_lin_val:+.2f}')
        
        # Power law curve: y = a * R^b
        r_pos = np.maximum(R, 1e-4)
        yt_pos = np.maximum(yt, 1e-4)
        lr_log = LinearRegression().fit(np.log(r_pos).reshape(-1, 1), np.log(yt_pos))
        b_exp = lr_log.coef_[0]
        r_grid_pos = np.maximum(r_grid, 1e-4)
        y_pow_grid = np.exp(lr_log.predict(np.log(r_grid_pos).reshape(-1, 1)))
        r2_pow_val = df_res.loc[df_res["element"]==elem, "PowerLaw_IS_R2"].values[0]
        ax.plot(r_grid, y_pow_grid, color='forestgreen', linestyle='-.', label=f'Power-Law ($R^{{{b_exp:.2f}}}$): LOSO R²={r2_pow_val:+.2f}')

        ax.set_title(f"元素: {elem} (内标线: {PRIMARY_LINES[elem]['is_name']})", fontsize=11, fontweight='bold')
        ax.set_xlabel("峰高比值 R = I_analyte / I_IS", fontsize=9)
        ax.set_ylabel("真实质量百分比 (wt%)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right' if lr_all.coef_[0] > 0 else 'upper right', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'peak_ratio_calibration_curves.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Calibration curves saved to: {fig_path}")

if __name__ == '__main__':
    main()
