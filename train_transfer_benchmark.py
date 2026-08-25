"""
train_transfer_benchmark.py

Phase 2: Rigorous NASA ChemCam Cross-Domain Calibration Transfer Benchmark.
Transfers pre-trained representation from NASA ChemCam (59 geostandards, 744 points, 385-645nm)
to local industrial dusts (Dataset B, N=6 TR samples) under strict Leave-One-Sample-Out (LOSO).

Methodological Rigor Enhancements:
1. Multi-seed Neural Baselines (T0, T1, T2, T4): 5 random seeds (42..46) reporting Mean ± Std.
2. T5 (NIST-SBC) Dual Evaluation:
   - Primary: Δλ=0.0 (No per-sample drift distortion for pre-calibrated Dataset B)
   - Secondary: With per-sample drift estimation
   - 1000-iteration Permutation Test & 1000-resample Bootstrap 95% CI
3. Transparent output saved to output/phase2_transfer_rigorous.csv.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_aligned_dataset_b, load_nasa_calibration_data
from feature_extractor_v2 import NISTFeatureExtractorV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(config.BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

REGRESSION_ELEMENTS_B = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn']

# ==========================================
# Architectures
# ==========================================
class SlimECNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=8, padding=12),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.head(self.features(x))

class LEVIRegressor(nn.Module):
    def __init__(self, out_dim, pretrained_conv=None):
        super().__init__()
        self.frozen_conv = nn.Conv1d(1, 32, kernel_size=25, stride=8, padding=12)
        if pretrained_conv is not None:
            self.frozen_conv.load_state_dict(pretrained_conv.state_dict())
            
        for p in self.frozen_conv.parameters():
            p.requires_grad = False
            
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=8, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.pool_frozen = nn.AdaptiveAvgPool1d(16)
        
        self.head_frozen = nn.Linear(32 * 16, out_dim)
        self.head_local = nn.Linear(16 * 16, out_dim)
        self.head_fused = nn.Linear((32 + 16) * 16, out_dim)

    def forward(self, x):
        with torch.no_grad():
            f_froz = self.pool_frozen(self.frozen_conv(x)).flatten(1)
        f_loc = self.local_conv(x).flatten(1)
        f_all = torch.cat([f_froz, f_loc], dim=1)
        
        out_froz = self.head_frozen(f_froz)
        out_loc = self.head_local(f_loc)
        out_fused = self.head_fused(f_all)
        
        return (out_froz + out_loc + out_fused) / 3.0

# ==========================================
# Training & Benchmarking
# ==========================================
def train_nasa_source_model(X_nasa, Y_nasa, out_dim, seed=42, epochs=45, lr=1e-3):
    torch.manual_seed(seed)
    model = SlimECNN(out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    
    ds = TensorDataset(torch.from_numpy(X_nasa).unsqueeze(1).float(), torch.from_numpy(Y_nasa).float())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            
    return model

def run_neural_strategy(strategy_name, X_b_norm, Y_b, pretrained_base, n_seeds=5):
    logo = LeaveOneGroupOut()
    groups = np.arange(len(X_b_norm))
    n_samples, n_elems = Y_b.shape
    
    seed_r2s = {e: [] for e in range(n_elems)}
    seed_rmses = {e: [] for e in range(n_elems)}
    
    for s_idx, seed in enumerate(range(42, 42 + n_seeds)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        oof = np.zeros_like(Y_b)
        
        for tr_idx, te_idx in logo.split(X_b_norm, Y_b, groups=groups):
            bx_tr, by_tr = X_b_norm[tr_idx], Y_b[tr_idx]
            bx_te = X_b_norm[te_idx]
            ds_tr = TensorDataset(torch.from_numpy(bx_tr).unsqueeze(1).float(), torch.from_numpy(by_tr).float())
            loader = DataLoader(ds_tr, batch_size=4, shuffle=True)
            
            if strategy_name == 'T0_Scratch':
                m = SlimECNN(n_elems).to(DEVICE)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
                epochs = 50
            elif strategy_name == 'T1_Frozen':
                m = copy.deepcopy(pretrained_base)
                for p in m.features.parameters():
                    p.requires_grad = False
                opt = torch.optim.Adam(m.head.parameters(), lr=2e-3, weight_decay=1e-4)
                epochs = 35
            elif strategy_name == 'T2_S2_Tune':
                m = copy.deepcopy(pretrained_base)
                opt = torch.optim.Adam([
                    {'params': m.features.parameters(), 'lr': 1e-5},
                    {'params': m.head.parameters(), 'lr': 1e-3}
                ], weight_decay=1e-4)
                epochs = 40
            elif strategy_name == 'T4_LEVI':
                m = LEVIRegressor(n_elems, pretrained_conv=pretrained_base.features[0]).to(DEVICE)
                opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-3, weight_decay=1e-4)
                epochs = 40
                
            for _ in range(epochs):
                m.train()
                for x_b, y_b in loader:
                    opt.zero_grad()
                    loss = nn.MSELoss()(m(x_b.to(DEVICE)), y_b.to(DEVICE))
                    loss.backward()
                    opt.step()
                    
            m.eval()
            with torch.no_grad():
                oof[te_idx] = m(torch.from_numpy(bx_te).unsqueeze(1).float().to(DEVICE)).cpu().numpy()
                
        for e_i in range(n_elems):
            y_true = Y_b[:, e_i]
            clipped = np.maximum(oof[:, e_i], 0)
            seed_r2s[e_i].append(r2_score(y_true, clipped))
            seed_rmses[e_i].append(np.sqrt(mean_squared_error(y_true, clipped)))
            
    summary = {}
    for e_i in range(n_elems):
        summary[e_i] = {
            'r2_mean': float(np.mean(seed_r2s[e_i])),
            'r2_std': float(np.std(seed_r2s[e_i])),
            'rmse_mean': float(np.mean(seed_rmses[e_i])),
            'rmse_std': float(np.std(seed_rmses[e_i]))
        }
    return summary

def run_t5_with_statistics(F_nasa, Y_nasa_b, F_b, Y_b):
    logo = LeaveOneGroupOut()
    groups = np.arange(len(Y_b))
    n_samples, n_elems = Y_b.shape
    oof_t5 = np.zeros_like(Y_b)
    
    # 1. Closed-form T5 Predictions
    for tr_idx, te_idx in logo.split(F_b, Y_b, groups=groups):
        by_tr = Y_b[tr_idx]
        for e_i in range(n_elems):
            scaler_feat = StandardScaler()
            F_nasa_s = scaler_feat.fit_transform(F_nasa)
            m = Ridge(alpha=10.0)
            m.fit(F_nasa_s, Y_nasa_b[:, e_i])
            
            F_tr_s = scaler_feat.transform(F_b[tr_idx])
            pred_tr = m.predict(F_tr_s)
            if np.std(pred_tr) > 1e-4:
                slope, bias = np.polyfit(pred_tr, by_tr[:, e_i], 1)
            else:
                slope, bias = 1.0, float(np.mean(by_tr[:, e_i]) - np.mean(pred_tr))
                
            pred_te = m.predict(scaler_feat.transform(F_b[te_idx]))
            oof_t5[te_idx, e_i] = slope * pred_te + bias

    # 2. Permutation Test & Bootstrap CI
    def t5_fit_score(y_target, e_i):
        oof = np.zeros(len(y_target))
        for tr_idx, te_idx in logo.split(F_b, y_target, groups=groups):
            scaler_feat = StandardScaler()
            F_nasa_s = scaler_feat.fit_transform(F_nasa)
            m = Ridge(alpha=10.0)
            m.fit(F_nasa_s, Y_nasa_b[:, e_i])
            pred_tr = m.predict(scaler_feat.transform(F_b[tr_idx]))
            if np.std(pred_tr) > 1e-4:
                slope, bias = np.polyfit(pred_tr, y_target[tr_idx], 1)
            else:
                slope, bias = 1.0, float(np.mean(y_target[tr_idx]))
            oof[te_idx] = slope * m.predict(scaler_feat.transform(F_b[te_idx])) + bias
        return r2_score(y_target, np.maximum(oof, 0))

    t5_stats = {}
    rng = np.random.RandomState(42)
    
    for e_i in range(n_elems):
        y_true = Y_b[:, e_i]
        pred_clipped = np.maximum(oof_t5[:, e_i], 0)
        r2_val = r2_score(y_true, pred_clipped)
        rmse_val = np.sqrt(mean_squared_error(y_true, pred_clipped))
        
        # Permutation Test (1000 perms)
        p_val = 1.0
        if r2_val > 0:
            cnt = 0
            n_perm = 1000
            for _ in range(n_perm):
                y_p = rng.permutation(y_true)
                if t5_fit_score(y_p, e_i) >= r2_val:
                    cnt += 1
            p_val = (cnt + 1.0) / (n_perm + 1.0)
            
        # Bootstrap 95% CI (1000 resamples)
        boot_r2s, boot_rmses = [], []
        for _ in range(1000):
            idx = rng.randint(0, n_samples, n_samples)
            if len(np.unique(y_true[idx])) < 2: continue
            b_r2 = r2_score(y_true[idx], pred_clipped[idx])
            b_rmse = np.sqrt(mean_squared_error(y_true[idx], pred_clipped[idx]))
            boot_r2s.append(b_r2)
            boot_rmses.append(b_rmse)
            
        r2_ci = (np.percentile(boot_r2s, 2.5), np.percentile(boot_r2s, 97.5)) if len(boot_r2s) > 0 else (r2_val, r2_val)
        rmse_ci = (np.percentile(boot_rmses, 2.5), np.percentile(boot_rmses, 97.5)) if len(boot_rmses) > 0 else (rmse_val, rmse_val)
        
        t5_stats[e_i] = {
            'r2': r2_val,
            'r2_ci': r2_ci,
            'rmse': rmse_val,
            'rmse_ci': rmse_ci,
            'p_value': p_val
        }
        
    return t5_stats

def main():
    print("=" * 110)
    print("=== Phase 2: Rigorous Multi-Seed & Dual-Drift Calibration Transfer Benchmark ===")
    print("=" * 110)
    
    # 1. Load NASA Source Data
    from eval_nasa_feasibility import load_nasa_regression_data, aggregate_by_standard, REGRESSION_ELEMENTS as NASA_ELEM
    X_nasa_raw, Y_nasa_raw, std_names, wvl_nasa = load_nasa_regression_data()
    X_nasa, Y_nasa, _ = aggregate_by_standard(X_nasa_raw, Y_nasa_raw, std_names)
    
    mask_b = (wvl_nasa >= 385.0) & (wvl_nasa <= 645.0)
    wvl_b_win = wvl_nasa[mask_b]
    X_nasa_crop = X_nasa[:, mask_b]
    X_nasa_norm = (X_nasa_crop - np.mean(X_nasa_crop, axis=1, keepdims=True)) / (np.std(X_nasa_crop, axis=1, keepdims=True) + 1e-8)
    
    nasa_map = [NASA_ELEM.index(e) if e in NASA_ELEM else -1 for e in REGRESSION_ELEMENTS_B]
    Y_nasa_b = np.zeros((len(Y_nasa), len(REGRESSION_ELEMENTS_B)), dtype=np.float32)
    for j, idx in enumerate(nasa_map):
        if idx >= 0:
            Y_nasa_b[:, j] = Y_nasa[:, idx]

    # 2. Load Local Dataset B Target Data
    df_labels = pd.read_csv(os.path.join(config.BASE_DIR, 'labels', 'unified_wt.csv'))
    _, _, _, target_wvl = load_nasa_calibration_data()
    Xb_all, _, names_b = load_aligned_dataset_b(target_wvl=target_wvl)
    
    sample_dict_b = {}
    for i, name in enumerate(names_b):
        s_id = name.split('_')[1]
        sample_dict_b.setdefault(s_id, []).append(Xb_all[i])
    snames_b = sorted(list(sample_dict_b.keys()))
    
    X_b_full = np.array([np.mean(sample_dict_b[s], axis=0) for s in snames_b], dtype=np.float32)
    X_b_crop = X_b_full[:, mask_b]
    X_b_norm = (X_b_crop - np.mean(X_b_crop, axis=1, keepdims=True)) / (np.std(X_b_crop, axis=1, keepdims=True) + 1e-8)
    
    Y_b = np.zeros((len(snames_b), len(REGRESSION_ELEMENTS_B)), dtype=np.float32)
    for i, s in enumerate(snames_b):
        row = df_labels[(df_labels['sample_id'] == s) & (df_labels['dataset'] == 'B')]
        Y_b[i] = [row.iloc[0][e] for e in REGRESSION_ELEMENTS_B]

    # Extract Features for T5 under both Drift OFF and Drift ON
    extractor_nasa = NISTFeatureExtractorV2(wavelengths=wvl_b_win, elements=REGRESSION_ELEMENTS_B)
    F_nasa_nodrift, _ = extractor_nasa.transform(X_nasa_crop, correct_drift=False)
    F_nasa_drift, _ = extractor_nasa.transform(X_nasa_crop, correct_drift=True)
    
    extractor_b = NISTFeatureExtractorV2(wavelengths=wvl_b_win, elements=REGRESSION_ELEMENTS_B)
    F_b_nodrift, _ = extractor_b.transform(X_b_crop, correct_drift=False)
    F_b_drift, _ = extractor_b.transform(X_b_crop, correct_drift=True)

    # 3. Train Pre-trained Source Model
    print("Pre-training base SlimECNN on NASA ChemCam standards (Seed=42)...")
    pretrained_slim = train_nasa_source_model(X_nasa_norm, Y_nasa_b, out_dim=len(REGRESSION_ELEMENTS_B), seed=42)

    # 4. Run Neural Multi-Seed Evaluations
    print("Running Multi-Seed (5 seeds) Neural Evaluations...")
    res_t0 = run_neural_strategy('T0_Scratch', X_b_norm, Y_b, pretrained_slim, n_seeds=5)
    res_t1 = run_neural_strategy('T1_Frozen', X_b_norm, Y_b, pretrained_slim, n_seeds=5)
    res_t2 = run_neural_strategy('T2_S2_Tune', X_b_norm, Y_b, pretrained_slim, n_seeds=5)
    res_t4 = run_neural_strategy('T4_LEVI', X_b_norm, Y_b, pretrained_slim, n_seeds=5)

    # 5. Run T5 Dual Evaluations (Drift OFF vs Drift ON)
    print("Running Deterministic T5 Transfer (Drift OFF & Drift ON)...")
    res_t5_nodrift = run_t5_with_statistics(F_nasa_nodrift, Y_nasa_b, F_b_nodrift, Y_b)
    res_t5_drift = run_t5_with_statistics(F_nasa_drift, Y_nasa_b, F_b_drift, Y_b)

    # 6. Print Rigorous Synthesis Table
    print(f"\n{'='*130}")
    print(f"{'Element':<8} | {'T0: Scratch (5-seed)':<22} | {'T1: Frozen (5-seed)':<22} | {'T2: S2 Tune (5-seed)':<22} | {'T5: NIST-SBC (Drift OFF)':<26} | {'T5: NIST-SBC (Drift ON)':<26}")
    print(f"{'':<8} | {'R2 (Mean±Std)':<22} | {'R2 (Mean±Std)':<22} | {'R2 (Mean±Std)':<22} | {'R2 [95% CI], p-val':<26} | {'R2 [95% CI], p-val':<26}")
    print("-" * 130)
    
    rows_out = []
    for e_i, elem in enumerate(REGRESSION_ELEMENTS_B):
        t0_str = f"{res_t0[e_i]['r2_mean']:+.2f}±{res_t0[e_i]['r2_std']:.2f}"
        t1_str = f"{res_t1[e_i]['r2_mean']:+.2f}±{res_t1[e_i]['r2_std']:.2f}"
        t2_str = f"{res_t2[e_i]['r2_mean']:+.2f}±{res_t2[e_i]['r2_std']:.2f}"
        
        t5_off_r2 = res_t5_nodrift[e_i]['r2']
        t5_off_ci = res_t5_nodrift[e_i]['r2_ci']
        t5_off_p = res_t5_nodrift[e_i]['p_value']
        t5_off_str = f"{t5_off_r2:+.3f} [{t5_off_ci[0]:+.2f},{t5_off_ci[1]:+.2f}], p={t5_off_p:.3f}"
        
        t5_on_r2 = res_t5_drift[e_i]['r2']
        t5_on_ci = res_t5_drift[e_i]['r2_ci']
        t5_on_p = res_t5_drift[e_i]['p_value']
        t5_on_str = f"{t5_on_r2:+.3f} [{t5_on_ci[0]:+.2f},{t5_on_ci[1]:+.2f}], p={t5_on_p:.3f}"
        
        print(f"{elem:<8} | {t0_str:<22} | {t1_str:<22} | {t2_str:<22} | {t5_off_str:<26} | {t5_on_str:<26}")
        
        rows_out.append({
            'element': elem,
            'T0_r2_mean': res_t0[e_i]['r2_mean'], 'T0_r2_std': res_t0[e_i]['r2_std'],
            'T1_r2_mean': res_t1[e_i]['r2_mean'], 'T1_r2_std': res_t1[e_i]['r2_std'],
            'T2_r2_mean': res_t2[e_i]['r2_mean'], 'T2_r2_std': res_t2[e_i]['r2_std'],
            'T4_r2_mean': res_t4[e_i]['r2_mean'], 'T4_r2_std': res_t4[e_i]['r2_std'],
            'T5_NoDrift_r2': t5_off_r2, 'T5_NoDrift_rmse': res_t5_nodrift[e_i]['rmse'], 'T5_NoDrift_p': t5_off_p,
            'T5_WithDrift_r2': t5_on_r2, 'T5_WithDrift_rmse': res_t5_drift[e_i]['rmse'], 'T5_WithDrift_p': t5_on_p
        })

    out_csv = os.path.join(OUT_DIR, 'phase2_transfer_rigorous.csv')
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\nRigorous dual-drift transfer benchmark saved to {out_csv}")

if __name__ == '__main__':
    main()
