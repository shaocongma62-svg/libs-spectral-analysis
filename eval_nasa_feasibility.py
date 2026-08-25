"""NASA ChemCam calibration feasibility benchmark.

Purpose: validate the current NIST-feature + regression approach on the
large, well-labelled ChemCam Earth-calibration dataset before investing
in more local gradient samples.

Design:
- Standard-level aggregation: repeated spectra of the same geostandard are
  averaged into one physical sample.
- GroupKFold by geostandard to avoid same-standard leakage.
- Per-fold feature selection and scaling.
- Models: NIST features + Ridge / SVR, plus raw-spectrum PLS baseline.
- Coverage modes: full 240-906 nm, local-like 240-760 nm, B-like 385-645 nm.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import normalize_name
from model_dual_stream import NISTFeatureExtractor
from wavelet_preprocess import wavelet_denoise


REGRESSION_ELEMENTS = ['Si', 'Ti', 'Al', 'Fe', 'Mn', 'Mg', 'Ca', 'Na', 'K']
COVERAGE_RANGES = {
    'full': (240.0, 906.0),
    '240_760': (240.0, 760.0),
    '385_645': (385.0, 645.0),
}


def load_nasa_regression_data():
    calib = pd.read_csv(os.path.join(config.NASA_DATA_DIR, 'msl_ccam_libs_calib.csv'))
    comp = pd.read_csv(os.path.join(config.NASA_DATA_DIR, 'ccam_calibration_compositions.csv')).dropna(subset=['Target'])
    comp['norm_name'] = comp['Target'].apply(normalize_name)

    X_list, y_list, std_names = [], [], []
    for col in [c for c in calib.columns if c != 'Wvl']:
        target_name = col.split('.')[0]
        match = comp[comp['norm_name'] == normalize_name(target_name)]
        if match.empty:
            # Some calib names are abbreviations of comp names (e.g. BIR1 vs BIR-1a).
            prefix_match = comp[comp['norm_name'].str.startswith(normalize_name(target_name))]
            if len(prefix_match) == 1:
                match = prefix_match
        if match.empty:
            continue
        row = match.iloc[0]
        elem_wt = {}
        for ox in config.OXIDES:
            val = float(row[ox]) if pd.notna(row[ox]) else 0.0
            if val <= 0 or ox not in config.OXIDE_ELEM:
                continue
            for elem, frac in config.OXIDE_ELEM[ox].items():
                if elem in REGRESSION_ELEMENTS:
                    elem_wt[elem] = elem_wt.get(elem, 0.0) + val * frac
        y_vec = np.array([elem_wt.get(e, 0.0) for e in REGRESSION_ELEMENTS], dtype=np.float32)
        X_list.append(calib[col].values.astype(np.float32))
        y_list.append(y_vec)
        std_names.append(target_name)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.float32)
    wavelengths = calib['Wvl'].values.astype(np.float32)
    return X, Y, np.array(std_names), wavelengths


def aggregate_by_standard(X, Y, std_names):
    groups = {}
    for i, name in enumerate(std_names):
        groups.setdefault(name, []).append(i)
    names = sorted(groups.keys())
    X_agg = np.zeros((len(names), X.shape[1]), dtype=np.float32)
    Y_agg = np.zeros((len(names), Y.shape[1]), dtype=np.float32)
    for i, name in enumerate(names):
        idx = groups[name]
        X_agg[i] = np.mean(X[idx], axis=0)
        Y_agg[i] = Y[idx[0]]
    return X_agg, Y_agg, np.array(names)


def snv_normalize(X):
    X_snv = np.zeros_like(X)
    for i in range(len(X)):
        mu = np.mean(X[i])
        std = np.std(X[i]) + 1e-8
        X_snv[i] = (X[i] - mu) / std
    return np.nan_to_num(X_snv, nan=0.0, posinf=0.0, neginf=0.0)


def run_feasibility(coverage_key, seed=42):
    print('\n' + '=' * 85)
    print(f'NASA ChemCam feasibility: coverage={coverage_key} '
          f'{COVERAGE_RANGES[coverage_key][0]} - {COVERAGE_RANGES[coverage_key][1]} nm')
    print('=' * 85)

    X_raw, Y_raw, std_names, wvl = load_nasa_regression_data()
    X, Y, names = aggregate_by_standard(X_raw, Y_raw, std_names)
    print(f'Loaded {len(X_raw)} spectra -> {len(X)} standards, {X.shape[1]} channels.')

    lo, hi = COVERAGE_RANGES[coverage_key]
    mask = (wvl >= lo) & (wvl <= hi)
    wvl_c = wvl[mask]
    X_c = X[:, mask]

    X_snv = snv_normalize(X_c)
    X_den = wavelet_denoise(X_snv, wavelet='db4', level=1, mode='soft')

    extractor = NISTFeatureExtractor(wavelengths=wvl_c, elements=REGRESSION_ELEMENTS)
    valid, blind = extractor.get_coverage_report(X_c)
    F_covered, covered_lines = extractor.transform_covered(X_den)
    print(f'Covered NIST features: {F_covered.shape[1]} from {len(covered_lines)} lines; '
          f'modelable: {sorted(valid)}; blind: {sorted(blind)}')
    modelable = sorted(set(REGRESSION_ELEMENTS) & set(valid))

    gkf = GroupKFold(n_splits=5)
    oof_ridge = np.zeros_like(Y)
    oof_svr = np.zeros_like(Y)
    oof_pls = np.zeros_like(Y)

    for train_idx, test_idx in gkf.split(F_covered, Y, groups=names):
        # NIST-feature models with per-fold feature selection and scaling.
        F_tr, F_te = F_covered[train_idx], F_covered[test_idx]
        fold_var = np.var(F_tr, axis=0) > 1e-6
        if not fold_var.any():
            fold_var = np.ones(F_tr.shape[1], dtype=bool)
        F_tr, F_te = F_tr[:, fold_var], F_te[:, fold_var]

        scaler = StandardScaler()
        F_tr_s = scaler.fit_transform(F_tr)
        F_te_s = scaler.transform(F_te)

        ridge = Ridge(alpha=10.0)
        ridge.fit(F_tr_s, Y[train_idx])
        oof_ridge[test_idx] = np.maximum(ridge.predict(F_te_s), 0.0)

        svr = MultiOutputRegressor(SVR(C=10.0, epsilon=0.1))
        svr.fit(F_tr_s, Y[train_idx])
        oof_svr[test_idx] = np.maximum(svr.predict(F_te_s), 0.0)

        # Raw-spectrum PLS baseline.
        pls = PLSRegression(n_components=min(10, len(train_idx) - 1))
        pls.fit(X_snv[train_idx], Y[train_idx])
        oof_pls[test_idx] = np.maximum(pls.predict(X_snv[test_idx]), 0.0)

    print('\n' + '-' * 85)
    print(f"{'Element':<6} | {'Ridge RMSE':<12} | {'Ridge R^2':<10} | "
          f"{'SVR RMSE':<10} | {'SVR R^2':<8} | {'PLS RMSE':<10} | {'PLS R^2':<8}")
    print('-' * 85)
    rows = []
    for j, elem in enumerate(REGRESSION_ELEMENTS):
        if elem not in modelable:
            print(f"{elem:<6} | {'BLIND':<12} | {'N/A':<10} | "
                  f"{'N/A':<10} | {'N/A':<8} | {'N/A':<10} | {'N/A':<8}")
            rows.append({
                'coverage': coverage_key,
                'element': elem,
                'ridge_rmse': np.nan,
                'ridge_r2': np.nan,
                'svr_rmse': np.nan,
                'svr_r2': np.nan,
                'pls_rmse': np.nan,
                'pls_r2': np.nan,
            })
            continue
        r_rmse = np.sqrt(mean_squared_error(Y[:, j], oof_ridge[:, j]))
        r_r2 = r2_score(Y[:, j], oof_ridge[:, j])
        s_rmse = np.sqrt(mean_squared_error(Y[:, j], oof_svr[:, j]))
        s_r2 = r2_score(Y[:, j], oof_svr[:, j])
        p_rmse = np.sqrt(mean_squared_error(Y[:, j], oof_pls[:, j]))
        p_r2 = r2_score(Y[:, j], oof_pls[:, j])
        print(f"{elem:<6} | {r_rmse:<12.3f} | {r_r2:<10.3f} | "
              f"{s_rmse:<10.3f} | {s_r2:<8.3f} | {p_rmse:<10.3f} | {p_r2:<8.3f}")
        rows.append({
            'coverage': coverage_key,
            'element': elem,
            'ridge_rmse': r_rmse,
            'ridge_r2': r_r2,
            'svr_rmse': s_rmse,
            'svr_r2': s_r2,
            'pls_rmse': p_rmse,
            'pls_r2': p_r2,
        })
    print('-' * 85)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='NASA ChemCam feasibility benchmark.')
    parser.add_argument('--coverage', choices=list(COVERAGE_RANGES.keys()) + ['all'], default='all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=os.path.join('output', 'nasa_feasibility_metrics.csv'))
    args = parser.parse_args()

    coverages = list(COVERAGE_RANGES.keys()) if args.coverage == 'all' else [args.coverage]
    frames = []
    for coverage in coverages:
        frames.append(run_feasibility(coverage, seed=args.seed))

    result = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'\nSaved feasibility metrics to {args.out}')


if __name__ == '__main__':
    main()
