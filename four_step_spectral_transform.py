"""
four_step_spectral_transform.py

Implementation and Evaluation of Self-Developed 4-Step Cross-Domain Physical Transfer Pipeline
(Inspired by Jin et al. 2022 & Jia et al. 2022 MarSCoDe/ChemCam calibration concepts):
- Step 1: Laser Pulse Energy & Radiometric Scaling
- Step 2: Instrument Line Shape (ILS) Gaussian Convolution (0.06nm -> 0.20nm)
- Step 3: SNIP Continuum & Graphite Background Harmonization
- Step 4: Cross-Instrument Transfer (Full Spectrum Ridge vs NIST Feature SBC)

Results are deterministically evaluated and saved to output/four_step_results.csv.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import config
from dataset_loader import load_aligned_dataset_b, load_nasa_calibration_data
from eval_nasa_feasibility import load_nasa_regression_data, aggregate_by_standard, REGRESSION_ELEMENTS as NASA_ELEM
from feature_extractor_v2 import NISTFeatureExtractorV2, snip_baseline_physical

OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

REGRESSION_ELEMENTS_B = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Zn']

class FourStepSpectralTransformer:
    def __init__(self, local_wvl, target_wvl, 
                 local_fwhm=0.06, target_fwhm=0.20,
                 local_laser_energy_mj=38.0, target_laser_energy_mj=14.0):
        self.local_wvl = np.array(local_wvl, dtype=np.float64)
        self.target_wvl = np.array(target_wvl, dtype=np.float64)
        self.local_fwhm = local_fwhm
        self.target_fwhm = target_fwhm
        self.local_laser_energy = local_laser_energy_mj
        self.target_laser_energy = target_laser_energy_mj
        
        # Gaussian convolution width
        delta_fwhm = np.sqrt(max(self.target_fwhm**2 - self.local_fwhm**2, 0.0))
        d_lambda_local = np.median(np.diff(self.local_wvl))
        self.conv_sigma_pixels = (delta_fwhm / 2.35482) / max(d_lambda_local, 1e-4)

    def transform(self, X_local):
        n_samples = len(X_local)
        X_trans_norm = np.zeros((n_samples, len(self.target_wvl)), dtype=np.float32)
        X_trans_net = np.zeros((n_samples, len(self.target_wvl)), dtype=np.float32)
        
        energy_ratio = self.target_laser_energy / max(self.local_laser_energy, 1.0)
        
        for i in range(n_samples):
            # Step 1: Energy Scaling
            s1 = X_local[i] * energy_ratio
            
            # Step 2: High-Resolution ILS Gaussian Broadening
            if self.conv_sigma_pixels > 0.5:
                s2 = gaussian_filter1d(s1, sigma=self.conv_sigma_pixels, mode='reflect')
            else:
                s2 = s1
                
            # Resample to Target Grid
            interp_fn = interp1d(self.local_wvl, s2, kind='linear', bounds_error=False, fill_value=0.0)
            s2_grid = np.maximum(interp_fn(self.target_wvl), 0.0)
            
            # Step 3: Physical SNIP Baseline Subtraction
            base = snip_baseline_physical(s2_grid, self.target_wvl, window_nm=1.5)
            net = np.maximum(s2_grid - base, 0.0)
            
            band_mask = (self.target_wvl >= 385.0) & (self.target_wvl <= 645.0)
            tot_energy = np.sum(net[band_mask])
            norm = net / max(tot_energy, 1e-4)
            
            X_trans_norm[i] = norm
            X_trans_net[i] = net
            
        return X_trans_norm, X_trans_net

def main():
    print("=" * 100)
    print("=== Rigorous Evaluation of 4-Step Physical Transfer Pipeline on Dataset B ===")
    print("=" * 100)
    
    # 1. NASA Source Data
    X_nasa_raw, Y_nasa_raw, std_names, wvl_nasa = load_nasa_regression_data()
    X_nasa, Y_nasa, _ = aggregate_by_standard(X_nasa_raw, Y_nasa_raw, std_names)
    
    mask_band = (wvl_nasa >= 385.0) & (wvl_nasa <= 645.0)
    wvl_crop = wvl_nasa[mask_band]
    X_nasa_crop = X_nasa[:, mask_band]
    
    # NASA preprocessing
    X_nasa_norm = np.zeros_like(X_nasa_crop)
    for i in range(len(X_nasa_crop)):
        base = snip_baseline_physical(X_nasa_crop[i], wvl_crop, window_nm=1.5)
        net = np.maximum(X_nasa_crop[i] - base, 0.0)
        X_nasa_norm[i] = net / max(np.sum(net), 1e-4)

    nasa_map = [NASA_ELEM.index(e) if e in NASA_ELEM else -1 for e in REGRESSION_ELEMENTS_B]
    Y_nasa_b = np.zeros((len(Y_nasa), len(REGRESSION_ELEMENTS_B)), dtype=np.float32)
    for j, idx in enumerate(nasa_map):
        if idx >= 0: Y_nasa_b[:, j] = Y_nasa[:, idx]

    # 2. Local Dataset B Target Data
    df_labels = pd.read_csv(os.path.join(BASE_DIR, 'labels', 'unified_wt.csv'))
    _, _, _, target_wvl = load_nasa_calibration_data()
    Xb_all, _, names_b = load_aligned_dataset_b(target_wvl=target_wvl)
    
    sample_dict_b = {}
    for i, name in enumerate(names_b):
        s_id = name.split('_')[1]
        sample_dict_b.setdefault(s_id, []).append(Xb_all[i])
    snames_b = sorted(list(sample_dict_b.keys()))
    
    X_b_raw = np.array([np.mean(sample_dict_b[s], axis=0) for s in snames_b], dtype=np.float32)
    Y_b = np.zeros((len(snames_b), len(REGRESSION_ELEMENTS_B)), dtype=np.float32)
    for i, s in enumerate(snames_b):
        row = df_labels[(df_labels['sample_id'] == s) & (df_labels['dataset'] == 'B')]
        Y_b[i] = [row.iloc[0][e] for e in REGRESSION_ELEMENTS_B]

    # 3. 4-Step Transform
    transformer = FourStepSpectralTransformer(
        local_wvl=target_wvl,
        target_wvl=wvl_nasa,
        local_fwhm=0.06,
        target_fwhm=0.20,
        local_laser_energy_mj=38.0,
        target_laser_energy_mj=14.0
    )
    X_b_trans_norm, X_b_trans_net = transformer.transform(X_b_raw)
    X_b_norm_crop = X_b_trans_norm[:, mask_band]
    X_b_net_crop = X_b_trans_net[:, mask_band]

    # Mode A: Full-Spectrum Ridge (2765 channels)
    logo = LeaveOneGroupOut()
    groups = np.arange(len(snames_b))
    oof_fullspec = np.zeros_like(Y_b)
    
    for tr_idx, te_idx in logo.split(X_b_norm_crop, Y_b, groups=groups):
        by_tr = Y_b[tr_idx]
        for e_i in range(len(REGRESSION_ELEMENTS_B)):
            scaler = StandardScaler()
            X_nasa_s = scaler.fit_transform(X_nasa_norm)
            m = Ridge(alpha=100.0)
            m.fit(X_nasa_s, Y_nasa_b[:, e_i])
            pred_tr = m.predict(scaler.transform(X_b_norm_crop[tr_idx]))
            if np.std(pred_tr) > 1e-4:
                slope, bias = np.polyfit(pred_tr, by_tr[:, e_i], 1)
            else:
                slope, bias = 1.0, float(np.mean(by_tr[:, e_i]) - np.mean(pred_tr))
            pred_te = m.predict(scaler.transform(X_b_norm_crop[te_idx]))
            oof_fullspec[te_idx, e_i] = slope * pred_te + bias

    # Mode B: 4-Step Transformed + NIST Physical Feature SBC Transfer
    ext_n = NISTFeatureExtractorV2(wavelengths=wvl_crop, elements=REGRESSION_ELEMENTS_B)
    Fn, _ = ext_n.transform(X_nasa_crop, correct_drift=False)
    ext_b = NISTFeatureExtractorV2(wavelengths=wvl_crop, elements=REGRESSION_ELEMENTS_B)
    Fb, _ = ext_b.transform(X_b_net_crop, correct_drift=False)

    oof_nist_sbc = np.zeros_like(Y_b)
    for tr_idx, te_idx in logo.split(Fb, Y_b, groups=groups):
        by_tr = Y_b[tr_idx]
        for e_i in range(len(REGRESSION_ELEMENTS_B)):
            scaler = StandardScaler()
            Fn_s = scaler.fit_transform(Fn)
            m = Ridge(alpha=10.0)
            m.fit(Fn_s, Y_nasa_b[:, e_i])
            pred_tr = m.predict(scaler.transform(Fb[tr_idx]))
            if np.std(pred_tr) > 1e-4:
                slope, bias = np.polyfit(pred_tr, by_tr[:, e_i], 1)
            else:
                slope, bias = 1.0, float(np.mean(by_tr[:, e_i]) - np.mean(pred_tr))
            pred_te = m.predict(scaler.transform(Fb[te_idx]))
            oof_nist_sbc[te_idx, e_i] = slope * pred_te + bias

    # 4. Print & Save Detailed Results
    print(f"\n{'='*105}")
    print(f"{'Element':<8} | {'Mode A: Full-Spectrum Ridge':<35} | {'Mode B: 4-Step + NIST Feature SBC':<35}")
    print(f"{'':<8} | {'RMSE (wt%)':<16} | {'R2 Score':<16} | {'RMSE (wt%)':<16} | {'R2 Score':<16}")
    print("-" * 105)
    
    rows_out = []
    for e_i, elem in enumerate(REGRESSION_ELEMENTS_B):
        y_true = Y_b[:, e_i]
        
        c_a = np.maximum(oof_fullspec[:, e_i], 0)
        r2_a = r2_score(y_true, c_a)
        rmse_a = np.sqrt(mean_squared_error(y_true, c_a))
        
        c_b = np.maximum(oof_nist_sbc[:, e_i], 0)
        r2_b = r2_score(y_true, c_b)
        rmse_b = np.sqrt(mean_squared_error(y_true, c_b))
        
        print(f"{elem:<8} | {rmse_a:<16.3f} | {r2_a:<16.3f} | {rmse_b:<16.3f} | {r2_b:<16.3f}")
        
        rows_out.append({
            'element': elem,
            'ModeA_FullSpec_RMSE': rmse_a,
            'ModeA_FullSpec_R2': r2_a,
            'ModeB_NIST_SBC_RMSE': rmse_b,
            'ModeB_NIST_SBC_R2': r2_b
        })

    out_csv = os.path.join(OUT_DIR, 'four_step_results.csv')
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\nExact 4-Step results successfully written to {out_csv}")

if __name__ == '__main__':
    main()
