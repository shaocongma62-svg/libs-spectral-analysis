"""Multi-seed ECNN ablation on ChemCam calibration data.

Comparison question:
    Does increasing ensemble size and training length (10 members, 60 epochs)
    improve held-out standard R^2/RMSE relative to the baseline
    (5 members, 40 epochs)?

Protocol:
    - same standard-level train/val/test split logic for every run,
    - split seed varies per run (three seeds),
    - val is used only for early stopping,
    - test metrics are computed once per run.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_calibration_ecnn import (
    REGRESSION_ELEMENTS,
    load_nasa_regression_data,
    standard_split,
    train_member,
)
from wavelet_preprocess import wavelet_denoise


def evaluate_ensemble(models, X_te, Y_te, device):
    preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            preds.append(model(torch.from_numpy(X_te).unsqueeze(1).to(device)).cpu().numpy())
    pred = np.maximum(np.mean(preds, axis=0), 0.0)
    rows = []
    for j, elem in enumerate(REGRESSION_ELEMENTS):
        rows.append({
            'rmse': float(np.sqrt(mean_squared_error(Y_te[:, j], pred[:, j]))),
            'r2': float(r2_score(Y_te[:, j], pred[:, j])),
        })
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-seed ECNN ablation.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 202, 777])
    parser.add_argument('--out-runs', default=os.path.join('output', 'ecnn_tuning_runs.csv'))
    parser.add_argument('--out-summary', default=os.path.join('output', 'ecnn_tuning_summary.csv'))
    args = parser.parse_args()

    methods = [
        {'method': 'baseline', 'members': 5, 'epochs': 40},
        {'method': 'tuned', 'members': 10, 'epochs': 60},
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    X_raw, Y, names, _ = load_nasa_regression_data()
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X = wavelet_denoise(X_raw, wavelet='db4', level=1, mode='soft')

    run_rows = []
    for method_cfg in methods:
        for seed in args.seeds:
            train_mask, val_mask, test_mask, _, _, _ = standard_split(names, seed=seed)
            mean = X[train_mask].mean(axis=0, keepdims=True)
            std = X[train_mask].std(axis=0, keepdims=True) + 1e-8
            X_tr = (X[train_mask] - mean) / std
            X_val = (X[val_mask] - mean) / std
            X_te = (X[test_mask] - mean) / std
            Y_tr, Y_val, Y_te = Y[train_mask], Y[val_mask], Y[test_mask]

            print(f"\n=== {method_cfg['method']}: seed={seed}, "
                  f"members={method_cfg['members']}, epochs={method_cfg['epochs']} ===")
            models = []
            for i in range(method_cfg['members']):
                model, _ = train_member(
                    X_tr, Y_tr, X_val, Y_val, Y.shape[1],
                    method_cfg['epochs'], 1e-3, device, seed=seed + i)
                models.append(model)

            for j, elem in enumerate(REGRESSION_ELEMENTS):
                metrics = evaluate_ensemble(models, X_te, Y_te, device)[j]
                run_rows.append({
                    'method': method_cfg['method'],
                    'seed': seed,
                    'members': method_cfg['members'],
                    'epochs': method_cfg['epochs'],
                    'element': elem,
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                })

    runs = pd.DataFrame(run_rows)
    os.makedirs(os.path.dirname(args.out_runs), exist_ok=True)
    runs.to_csv(args.out_runs, index=False, encoding='utf-8-sig')

    summary = runs.groupby(['method', 'element']).agg(
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        r2_mean=('r2', 'mean'),
        r2_std=('r2', 'std'),
        n_seeds=('seed', 'nunique'),
    ).reset_index()
    summary.to_csv(args.out_summary, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 100)
    print('=== ECNN Ablation Summary (mean +/- std over seeds) ===')
    print('=' * 100)
    pivot_r2 = summary.pivot(index='element', columns='method', values='r2_mean')
    pivot_r2_std = summary.pivot(index='element', columns='method', values='r2_std')
    pivot_rmse = summary.pivot(index='element', columns='method', values='rmse_mean')
    pivot_rmse_std = summary.pivot(index='element', columns='method', values='rmse_std')
    for elem in REGRESSION_ELEMENTS:
        line = f"{elem:<6} R2  "
        for method in [m['method'] for m in methods]:
            line += f" | {method:<8} {pivot_r2.loc[elem, method]:+.3f}±{pivot_r2_std.loc[elem, method]:.3f}"
        line += "  | RMSE"
        for method in [m['method'] for m in methods]:
            line += f" | {method:<8} {pivot_rmse.loc[elem, method]:.2f}±{pivot_rmse_std.loc[elem, method]:.2f}"
        print(line)
    print('=' * 100)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        x = np.arange(len(REGRESSION_ELEMENTS))
        width = 0.35
        colors = {'baseline': '#4C72B0', 'tuned': '#DD8452'}
        for offset, method in enumerate([m['method'] for m in methods]):
            vals = [pivot_r2.loc[e, method] for e in REGRESSION_ELEMENTS]
            errs = [pivot_r2_std.loc[e, method] for e in REGRESSION_ELEMENTS]
            axes[0].bar(x + offset * width - width / 2, vals, width, yerr=errs,
                        label=method, color=colors[method], capsize=3)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(REGRESSION_ELEMENTS)
        axes[0].set_ylabel('Test-standard R²')
        axes[0].set_title('ECNN ensemble size / epochs: R²')
        axes[0].axhline(0, color='gray', lw=0.8)
        axes[0].legend()

        for offset, method in enumerate([m['method'] for m in methods]):
            vals = [pivot_rmse.loc[e, method] for e in REGRESSION_ELEMENTS]
            errs = [pivot_rmse_std.loc[e, method] for e in REGRESSION_ELEMENTS]
            axes[1].bar(x + offset * width - width / 2, vals, width, yerr=errs,
                        label=method, color=colors[method], capsize=3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(REGRESSION_ELEMENTS)
        axes[1].set_ylabel('Test-standard RMSE (wt%)')
        axes[1].set_title('ECNN ensemble size / epochs: RMSE')
        axes[1].legend()

        fig.tight_layout()
        fig_path = os.path.join('output', 'ecnn_tuning_compare.png')
        fig.savefig(fig_path, dpi=150)
        print(f'Saved comparison figure to {fig_path}')
    except Exception as exc:
        print('Figure generation skipped:', exc)

    print(f'Saved run metrics to {args.out_runs}')
    print(f'Saved summary to {args.out_summary}')


if __name__ == '__main__':
    main()
