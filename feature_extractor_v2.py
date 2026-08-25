"""
feature_extractor_v2.py

Upgraded, Mathematically and Spectroscopically Robust NIST Feature Extractor:
1. Consensus-Gated Anchor Line Drift Estimator:
   - Uses multi-line consensus across multiple elements.
   - Falls back to 0.0 shift if no strong consensus exists.
   - For already calibrated datasets (like Dataset B), drift correction should be set to False (Δλ = 0.0).
2. Physical SNIP Baseline Subtraction:
   - Sets iteration count dynamically based on grid resolution (window_nm = 1.5 nm)
   - Never erodes wide/self-absorbed atomic emission peaks.
3. Fine-Window (±0.35nm) Peak Parameterization (6 features/line):
   - 0: Net Peak Height
   - 1: Local SNR (Net Height / Local Noise)
   - 2: Net Peak Area
   - 3: Relative Area (Peak Area / Total Spectrum Area)
   - 4: Lomakin-Scherbe Ratio (Target / Fe 259.94 or Target / Si 288.16)
   - 5: Local Line Position Shift (delta lambda)
4. Self-Absorption Diagnostic:
   - Ca II 393.37 / Ca I 422.67 ratio (when Ca is present).
   - Removed unmeasured/constant-zero H-alpha feature to prevent spurious zero-padding.
"""

import numpy as np

NIST_EMISSION_LINES = {
    'Fe': [259.94, 274.93, 275.57, 371.99, 404.581, 406.359, 407.174, 438.354, 440.475, 495.760],
    'Ca': [315.89, 393.37, 396.85, 422.673, 445.478, 610.272, 612.222, 616.217, 643.907],
    'Si': [288.16, 390.55, 615.513],
    'Mg': [279.55, 280.27, 285.21, 516.732, 517.268, 518.360],
    'Al': [308.22, 309.27, 394.401, 396.152],
    'Ti': [334.94, 336.12, 337.28, 375.93],
    'Zn': [330.26, 468.014, 472.20, 481.053, 636.235],
    'Mn': [257.61, 259.37, 260.57, 403.08],
    'K':  [766.49, 769.90],
    'Na': [589.00, 589.59],
    'O':  [777.19, 777.42, 777.54],
    'S':  [921.29, 922.81],
    'P':  [213.62, 214.91, 253.56],
    'C':  [193.09, 247.86],
    'Cl': [837.60, 839.19]
}

ANCHOR_LINES = [
    259.94,  # Fe II UV
    279.55,  # Mg II UV
    288.16,  # Si I UV
    309.27,  # Al I UV
    393.37,  # Ca II
    394.40,  # Al I
    403.08,  # Mn I
    404.58,  # Fe I
    422.67,  # Ca I
    481.05,  # Zn I
    518.36,  # Mg I
    589.00,  # Na I
    616.22   # Ca I
]

def snip_baseline_physical(spectrum, wavelengths, window_nm=1.5):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP) baseline.
    Iteration count is dynamically determined from wavelength grid resolution.
    """
    spec = spectrum.copy()
    d_lambda = np.median(np.diff(wavelengths))
    num_iterations = int(max(window_nm / d_lambda, 5))
    
    b = np.log(np.log(np.sqrt(np.maximum(spec, 0) + 1) + 1) + 1)
    
    for p in range(1, num_iterations + 1):
        b_prev = b.copy()
        left = np.pad(b_prev[:-2*p], (2*p, 0), mode='edge')
        right = np.pad(b_prev[2*p:], (0, 2*p), mode='edge')
        avg = (left + right) / 2.0
        b = np.minimum(b_prev, avg)
        
    base = (np.exp(np.exp(b) - 1) - 1)**2 - 1
    return np.maximum(base, 0)

def estimate_wavelength_shift(wvl, spec, anchor_lines=ANCHOR_LINES, search_window=0.8):
    """
    Estimates systematic wavelength drift Δλ (in nm) using consensus-gated voting.
    Falls back to 0.0 if no reliable consensus exists.
    """
    shifts = []
    weights = []
    
    for line in anchor_lines:
        if line < wvl.min() + search_window or line > wvl.max() - search_window:
            continue
        mask = (wvl >= line - search_window) & (wvl <= line + search_window)
        sub_w = wvl[mask]
        sub_s = spec[mask]
        if len(sub_s) < 5:
            continue
            
        peak_idx = np.argmax(sub_s)
        peak_wvl = sub_w[peak_idx]
        local_min = np.min(sub_s)
        peak_height = sub_s[peak_idx] - local_min
        local_noise = np.std(sub_s) + 1e-6
        
        if peak_height > 100.0 and (peak_height / local_noise) > 3.5:
            shift = peak_wvl - line
            if abs(shift) <= 0.5:
                shifts.append(shift)
                weights.append(peak_height)
                
    if len(shifts) == 0:
        return 0.0
        
    if len(shifts) == 1:
        if weights[0] > 500.0:
            return float(np.clip(shifts[0], -0.5, 0.5))
        return 0.0
        
    shifts = np.array(shifts)
    weights = np.array(weights)
    median_shift = float(np.median(shifts))
    
    agree_mask = np.abs(shifts - median_shift) <= 0.15
    if np.sum(agree_mask) >= 2 or np.max(weights[agree_mask]) > 1000.0:
        consensus_shift = float(np.average(shifts[agree_mask], weights=weights[agree_mask]))
        return float(np.clip(consensus_shift, -0.5, 0.5))
        
    return 0.0

class NISTFeatureExtractorV2:
    def __init__(self, wavelengths, elements=None, fine_window=0.35, search_window=0.8):
        self.wavelengths = np.array(wavelengths, dtype=np.float32)
        self.elements = elements if elements is not None else list(NIST_EMISSION_LINES.keys())
        self.fine_window = fine_window
        self.search_window = search_window
        
        self.valid_mask = (self.wavelengths >= 240.0) & (self.wavelengths <= 760.0)
        
        self.registered_lines = []
        for elem in self.elements:
            lines = NIST_EMISSION_LINES.get(elem, [])
            for wl in lines:
                if 240.0 <= wl <= 760.0:
                    self.registered_lines.append((elem, wl))
                    
        self.fe_ref_line = 259.94
        self.si_ref_line = 288.16

    def detect_coverage(self, X_raw):
        """Detect which NIST lines have genuine signal (not zero-padded)."""
        n_samples = len(X_raw)
        coverage = []
        
        for elem, line_wl in self.registered_lines:
            mask = (self.wavelengths >= line_wl - self.fine_window) & (self.wavelengths <= line_wl + self.fine_window) & self.valid_mask
            if not np.any(mask):
                coverage.append((elem, line_wl, False))
                continue
            n_sig = 0
            for i in range(n_samples):
                sub = np.nan_to_num(X_raw[i, mask], nan=0.0)
                if np.max(np.abs(sub)) > 1e-3 and np.std(sub) > 1e-5:
                    n_sig += 1
            has_sig = (n_sig / max(n_samples, 1)) >= 0.5
            coverage.append((elem, line_wl, has_sig))
            
        return coverage

    def transform(self, X_raw, correct_drift=False):
        """
        Extracts robust NIST physical features with dynamic physical SNIP baseline.
        By default correct_drift=False for already wavelength-calibrated datasets.
        """
        n_samples = len(X_raw)
        coverage = self.detect_coverage(X_raw)
        covered_lines = [(e, w) for (e, w, has_sig) in coverage if has_sig]
        
        n_feats_per_line = 6
        n_global_feats = 1  # 1 self-absorption ratio (Ca II / Ca I)
        total_feats = len(covered_lines) * n_feats_per_line + n_global_feats
        
        feat_matrix = np.zeros((n_samples, total_feats), dtype=np.float32)
        
        for i in range(n_samples):
            raw_spec = np.nan_to_num(X_raw[i], nan=0.0)
            total_area = max(np.sum(np.abs(raw_spec)), 1e-6)
            
            # 1. Consensus-gated drift estimation (if requested)
            if correct_drift:
                delta_lambda = estimate_wavelength_shift(self.wavelengths, raw_spec, search_window=self.search_window)
            else:
                delta_lambda = 0.0
                
            wvl_corr = self.wavelengths - delta_lambda
            
            # 2. Dynamic Physical SNIP baseline
            baseline = snip_baseline_physical(raw_spec, self.wavelengths, window_nm=1.5)
            net_spec = np.maximum(raw_spec - baseline, 0.0)
            
            # 3. Internal reference peaks
            fe_mask = (wvl_corr >= self.fe_ref_line - self.fine_window) & (wvl_corr <= self.fe_ref_line + self.fine_window) & self.valid_mask
            si_mask = (wvl_corr >= self.si_ref_line - self.fine_window) & (wvl_corr <= self.si_ref_line + self.fine_window) & self.valid_mask
            fe_ref_val = max(np.max(net_spec[fe_mask]) if np.any(fe_mask) else 1.0, 1.0)
            si_ref_val = max(np.max(net_spec[si_mask]) if np.any(si_mask) else 1.0, 1.0)
            
            # 4. Extract Line-by-Line Features
            feat_idx = 0
            for elem, line_wl in covered_lines:
                win_mask = (wvl_corr >= line_wl - self.fine_window) & (wvl_corr <= line_wl + self.fine_window) & self.valid_mask
                if not np.any(win_mask):
                    feat_idx += n_feats_per_line
                    continue
                    
                sub_w = wvl_corr[win_mask]
                sub_net = net_spec[win_mask]
                
                peak_max = np.max(sub_net)
                peak_wvl = sub_w[np.argmax(sub_net)]
                local_noise = np.std(sub_net) + 1.0
                local_snr = np.clip(peak_max / local_noise, 0.0, 1000.0)
                peak_area = np.sum(sub_net)
                rel_area = peak_area / total_area
                is_ratio = peak_max / fe_ref_val if fe_ref_val > 5.0 else (peak_max / si_ref_val)
                line_shift = peak_wvl - line_wl
                
                feat_matrix[i, feat_idx + 0] = peak_max
                feat_matrix[i, feat_idx + 1] = local_snr
                feat_matrix[i, feat_idx + 2] = peak_area
                feat_matrix[i, feat_idx + 3] = rel_area
                feat_matrix[i, feat_idx + 4] = is_ratio
                feat_matrix[i, feat_idx + 5] = line_shift
                feat_idx += n_feats_per_line
                
            # 5. Global Feature: Ca II 393.37 / Ca I 422.67 Self-Absorption
            ca_ii_mask = (wvl_corr >= 393.37 - self.fine_window) & (wvl_corr <= 393.37 + self.fine_window) & self.valid_mask
            ca_i_mask = (wvl_corr >= 422.67 - self.fine_window) & (wvl_corr <= 422.67 + self.fine_window) & self.valid_mask
            ca_ii_val = np.max(net_spec[ca_ii_mask]) if np.any(ca_ii_mask) else 0.0
            ca_i_val = np.max(net_spec[ca_i_mask]) if np.any(ca_i_mask) else 0.0
            self_abs_ratio = (ca_ii_val / (ca_i_val + 1.0)) if (ca_ii_val > 5.0 and ca_i_val > 5.0) else 0.0
            feat_matrix[i, feat_idx + 0] = self_abs_ratio
            
        return np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0), covered_lines
