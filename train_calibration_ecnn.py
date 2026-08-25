"""Leakage-free ECNN calibration benchmark on NASA ChemCam data.

This is a cleaned version of the earlier prototype. It uses a standard-level
train/val/test split: val is used only for early stopping, test is evaluated
exactly once. The first ensemble member's convolutional backbone plus the
preprocessing statistics are saved for later transfer experiments.

The task is quantitative regression of element wt%, matching Yu & Yao (2023).
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from eval_nasa_feasibility import REGRESSION_ELEMENTS, load_nasa_regression_data
from wavelet_preprocess import wavelet_denoise


class ECNNRegressor(nn.Module):
    def __init__(self, input_len, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=25, stride=8, padding=12)
        self.relu = nn.ReLU()
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, input_len)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.fc(self.relu(self.conv(x)))


def standard_split(names, val_frac=0.15, test_frac=0.15, seed=42):
    unique = np.unique(names)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_val = max(int(len(unique) * val_frac), 1)
    n_test = max(int(len(unique) * test_frac), 1)
    val_std = set(unique[:n_val])
    test_std = set(unique[n_val:n_val + n_test])
    train_std = set(unique[n_val + n_test:])
    train_mask = np.array([n in train_std for n in names])
    val_mask = np.array([n in val_std for n in names])
    test_mask = np.array([n in test_std for n in names])
    return train_mask, val_mask, test_mask, train_std, val_std, test_std


def train_member(X_tr, Y_tr, X_val, Y_val, out_dim, epochs, lr, device, seed):
    torch.manual_seed(seed)
    model = ECNNRegressor(X_tr.shape[1], out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr).unsqueeze(1), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val).unsqueeze(1), torch.from_numpy(Y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_losses.append(criterion(model(bx), by).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0:
            print(f'  member={seed} epoch={epoch + 1}/{epochs} val_loss={val_loss:.4f}')

    model.load_state_dict(best_state)
    return model, best_val


def main():
    parser = argparse.ArgumentParser(description='Leakage-free ECNN ChemCam benchmark.')
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=os.path.join('output', 'ecnn_calibration_metrics.csv'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    X_raw, Y, names, wvl = load_nasa_regression_data()
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X = wavelet_denoise(X_raw, wavelet='db4', level=1, mode='soft')

    train_mask, val_mask, test_mask, tr_s, va_s, te_s = standard_split(
        names, seed=args.seed)
    print(f'Standards: train={len(tr_s)}, val={len(va_s)}, test={len(te_s)}; '
          f'spectra: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}')
    assert set(tr_s).isdisjoint(va_s) and set(tr_s).isdisjoint(te_s) and set(va_s).isdisjoint(te_s)

    mean = X[train_mask].mean(axis=0, keepdims=True)
    std = X[train_mask].std(axis=0, keepdims=True) + 1e-8
    X_tr = (X[train_mask] - mean) / std
    X_val = (X[val_mask] - mean) / std
    X_te = (X[test_mask] - mean) / std
    Y_tr, Y_val, Y_te = Y[train_mask], Y[val_mask], Y[test_mask]

    members = []
    for i in range(args.members):
        model, _ = train_member(
            X_tr, Y_tr, X_val, Y_val, Y.shape[1], args.epochs, args.lr,
            device, seed=args.seed + i)
        members.append(model)
        print(f'  Member {i + 1}/{args.members} trained.')

    model.eval()
    preds = []
    with torch.no_grad():
        for model in members:
            model.eval()
            preds.append(model(torch.from_numpy(X_te).unsqueeze(1).to(device)).cpu().numpy())
    Y_pred = np.mean(preds, axis=0)
    Y_pred = np.maximum(Y_pred, 0.0)

    print('\n' + '=' * 80)
    print('=== ECNN Ensemble ChemCam Test-Standard Results (wt%) ===')
    print('=' * 80)
    print(f"{'Element':<6} | {'RMSE':<10} | {'R^2':<8}")
    print('-' * 80)
    rows = []
    for j, elem in enumerate(REGRESSION_ELEMENTS):
        rmse = np.sqrt(mean_squared_error(Y_te[:, j], Y_pred[:, j]))
        r2 = r2_score(Y_te[:, j], Y_pred[:, j])
        print(f"{elem:<6} | {rmse:<10.3f} | {r2:<8.3f}")
        rows.append({'element': elem, 'rmse': rmse, 'r2': r2})
    print('=' * 80)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding='utf-8-sig')

    backbone_path = os.path.join(config.SAVE_DIR, 'nasa_backbone_conv.pt')
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    torch.save({
        'conv_state_dict': members[0].conv.state_dict(),
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
        'wavelengths': wvl.astype(np.float32),
        'elements': REGRESSION_ELEMENTS,
        'seed': args.seed,
    }, backbone_path)
    print(f'Saved backbone to {backbone_path}')
    print(f'Saved metrics to {args.out}')


if __name__ == '__main__':
    main()
