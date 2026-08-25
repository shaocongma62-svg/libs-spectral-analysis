import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

NIST_EMISSION_LINES = {
    # Deep UV lines (ChemCam/NASA) + Visible 4-window hardware lines (Local Spectrometer)
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

class NISTFeatureExtractor:
    """Extracts normalized physical NIST emission peak intensities and peak-to-background ratios.
    
    Now includes spectral coverage detection: can distinguish real signal from
    zero-padded interpolation gaps, and only extract features where signal exists.
    """
    def __init__(self, wavelengths, elements=config.ALL_ELEMENTS, window_nm=0.2):
        self.wavelengths = np.array(wavelengths, dtype=np.float32)
        self.elements = elements
        self.window_nm = window_nm
        self.peak_indices = []
        self._coverage_mask = None  # populated by detect_signal_coverage()

        # Restrict wavelength coverage to valid hardware span (240nm - 760nm) to avoid zero-padded tail explosion
        valid_mask = (self.wavelengths >= 240.0) & (self.wavelengths <= 760.0)
        
        for elem in elements:
            lines = NIST_EMISSION_LINES.get(elem, [])
            for line_wl in lines:
                if line_wl >= 240.0 and line_wl <= 760.0:
                    mask = (self.wavelengths >= line_wl - window_nm) & (self.wavelengths <= line_wl + window_nm) & valid_mask
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        self.peak_indices.append((elem, line_wl, indices))

        self.num_features = len(self.peak_indices) * 6
        self.fe_mask = (self.wavelengths >= 259.94 - window_nm) & (self.wavelengths <= 259.94 + window_nm) & valid_mask
        self.si_mask = (self.wavelengths >= 288.16 - window_nm) & (self.wavelengths <= 288.16 + window_nm) & valid_mask

    def detect_signal_coverage(self, X_raw):
        """Detect which registered NIST lines have real spectral signal in X_raw.
        
        Args:
            X_raw: (n_samples, n_channels) raw spectral matrix BEFORE SNV normalization,
                   where zero-padded non-covered regions are 0.0.
            
        Returns:
            coverage: list of (elem, line_wl, has_signal) tuples
        """
        n_samples = len(X_raw)
        coverage = []
        
        for elem, line_wl, idx_arr in self.peak_indices:
            n_with_signal = 0
            for i in range(n_samples):
                spec = np.nan_to_num(X_raw[i], nan=0.0)
                window_max = np.max(np.abs(spec[idx_arr]))
                window_std = np.std(spec[idx_arr])
                # Real spectral signal has non-zero raw intensity and non-zero local variation
                if window_max > 1e-3 and window_std > 1e-5:
                    n_with_signal += 1
            has_signal = (n_with_signal / n_samples) >= 0.5
            coverage.append((elem, line_wl, has_signal))
        
        self._coverage_mask = [c[2] for c in coverage]
        return coverage

    def get_coverage_report(self, X):
        """Print a human-readable coverage diagnostic.
        
        Returns:
            valid_elements: set of element names with >=1 covered NIST line
            blind_elements: set of element names with 0 covered NIST lines
        """
        coverage = self.detect_signal_coverage(X)
        
        elem_status = {}
        for elem, line_wl, has_signal in coverage:
            if elem not in elem_status:
                elem_status[elem] = {'covered': [], 'blind': []}
            if has_signal:
                elem_status[elem]['covered'].append(line_wl)
            else:
                elem_status[elem]['blind'].append(line_wl)
        
        valid_elements = set()
        blind_elements = set()
        
        print("\n=== NIST Emission Line Coverage Diagnostic ===")
        for elem, status in elem_status.items():
            covered = status['covered']
            blind = status['blind']
            if len(covered) > 0:
                valid_elements.add(elem)
                flag = "[OK]" if len(blind) == 0 else "[!!]"
                print(f"  {flag} {elem}: {len(covered)} lines covered {covered}"
                      + (f", {len(blind)} lines blind {blind}" if blind else ""))
            else:
                blind_elements.add(elem)
                print(f"  [XX] {elem}: ALL {len(blind)} lines blind {blind} -- CANNOT model this element")
        print(f"  Summary: {len(valid_elements)} modelable elements, {len(blind_elements)} blind elements")
        print("=" * 50)
        
        return valid_elements, blind_elements

    def transform_covered(self, X):
        """Extract features ONLY for NIST lines with real signal coverage.
        
        Must call detect_signal_coverage(X) first.
        
        Returns:
            features: (n_samples, n_covered_lines * 4) — only net_height, snr, area, relative_area
            covered_line_info: list of (elem, line_wl) for covered lines
        """
        if self._coverage_mask is None:
            self.detect_signal_coverage(X)
        
        covered_indices = [(elem, wl, idx) for (elem, wl, idx), has_sig 
                           in zip(self.peak_indices, self._coverage_mask) if has_sig]
        
        if len(covered_indices) == 0:
            return np.zeros((len(X), 0), dtype=np.float32), []
        
        n_samples = len(X)
        n_feats_per_line = 4  # net_height, snr, area, relative_area (drop fe/si ratio for covered-only)
        features = np.zeros((n_samples, len(covered_indices) * n_feats_per_line), dtype=np.float32)
        
        for i in range(n_samples):
            spec = np.nan_to_num(X[i], nan=0.0)
            total_area = max(np.sum(np.abs(spec)), 1e-6)
            
            feat_idx = 0
            for elem, line_wl, idx_arr in covered_indices:
                window_intensities = spec[idx_arr]
                if len(window_intensities) == 0:
                    feat_idx += n_feats_per_line
                    continue
                
                local_bg = np.min(window_intensities)
                local_noise = np.std(window_intensities) + 1.0
                peak_max = np.max(window_intensities)
                net_height = max(peak_max - local_bg, 0.0)
                local_snr = np.clip(net_height / local_noise, 0.0, 1000.0)
                peak_area = np.sum(np.maximum(window_intensities - local_bg, 0))
                
                features[i, feat_idx] = net_height
                features[i, feat_idx + 1] = local_snr
                features[i, feat_idx + 2] = peak_area
                features[i, feat_idx + 3] = net_height / total_area
                feat_idx += n_feats_per_line
        
        covered_line_info = [(e, w) for e, w, _ in covered_indices]
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), covered_line_info

    def transform(self, X):
        """Original transform — extracts ALL registered NIST line features (backward compatible)."""
        n_samples = len(X)
        features = np.zeros((n_samples, self.num_features), dtype=np.float32)

        for i in range(n_samples):
            spec = np.nan_to_num(X[i], nan=0.0)
            total_area = max(np.sum(np.abs(spec)), 1e-6)
            
            fe_max = np.max(spec[self.fe_mask]) if np.any(self.fe_mask) else 1e-6
            fe_max = fe_max if fe_max > 1e-6 else 1e-6
            
            si_max = np.max(spec[self.si_mask]) if np.any(self.si_mask) else 1e-6
            si_max = si_max if si_max > 1e-6 else 1e-6
            
            feat_idx = 0
            for elem, line_wl, idx_arr in self.peak_indices:
                window_intensities = spec[idx_arr]
                if len(window_intensities) == 0:
                    features[i, feat_idx:feat_idx+6] = 0.0
                    feat_idx += 6
                    continue
                
                local_bg = np.min(window_intensities)
                local_noise = np.std(window_intensities) + 1.0
                
                peak_max = np.max(window_intensities)
                net_height = max(peak_max - local_bg, 0.0)
                local_snr = np.clip(net_height / local_noise, 0.0, 1000.0)
                peak_area = np.sum(np.maximum(window_intensities - local_bg, 0))
                
                features[i, feat_idx] = net_height
                features[i, feat_idx + 1] = local_snr
                features[i, feat_idx + 2] = peak_area
                features[i, feat_idx + 3] = net_height / total_area
                features[i, feat_idx + 4] = net_height / fe_max
                features[i, feat_idx + 5] = net_height / si_max
                feat_idx += 6

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

class GatedFusionUnit(nn.Module):
    """Sigmoid-activated adaptive gating unit to combine physical, NASA-pretrained, and local features."""
    def __init__(self, dim_phys, dim_nasa, dim_local, hidden_dim=64):
        super().__init__()
        total_dim = dim_phys + dim_nasa + dim_local
        self.gate_fc = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, f_phys, f_nasa, f_local):
        concat_f = torch.cat([f_phys, f_nasa, f_local], dim=1)
        weights = self.gate_fc(concat_f)  # (batch, 3)

        w_phys = weights[:, 0:1]
        w_nasa = weights[:, 1:2]
        w_local = weights[:, 2:3]

        f_fused = w_phys * f_phys + w_nasa * f_nasa + w_local * f_local
        return f_fused, weights

class DualStreamECNN(nn.Module):
    """
    Physics-Informed Dual-Stream ECNN:
    - Stream 1 (Physics): MLP processing normalized NIST emission line features.
    - Stream 2 (NASA Backbone): Pre-trained 1D-Conv from Earth standards (Frozen).
    - Stream 3 (Local Backbone): Trainable 1D-Conv for local instrument characteristics.
    - Fusion: Adaptive Gated Fusion Unit.
    """
    def __init__(self, input_channels, num_phys_features, num_classes, pretrained_checkpoint=None, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes

        # Stream 1: Physical Peak MLP
        self.phys_mlp = nn.Sequential(
            nn.Linear(num_phys_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Stream 2: Pre-trained NASA Backbone (Frozen)
        self.nasa_conv = nn.Conv1d(1, 32, kernel_size=25, stride=8, padding=12)
        if pretrained_checkpoint is not None and os.path.exists(pretrained_checkpoint):
            self.load_pretrained_nasa_conv(pretrained_checkpoint)

        for param in self.nasa_conv.parameters():
            param.requires_grad = False  # Freeze NASA backbone

        with torch.no_grad():
            f_dim_nasa = self.nasa_conv(torch.zeros(1, 1, input_channels)).numel()
        self.nasa_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f_dim_nasa, hidden_dim),
            nn.ReLU()
        )

        # Stream 3: Local Trainable Conv Backbone
        self.local_conv = nn.Conv1d(1, 32, kernel_size=25, stride=8, padding=12)
        with torch.no_grad():
            f_dim_local = self.local_conv(torch.zeros(1, 1, input_channels)).numel()
        self.local_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f_dim_local, hidden_dim),
            nn.ReLU()
        )

        # Gated Fusion
        self.fusion = GatedFusionUnit(hidden_dim, hidden_dim, hidden_dim, hidden_dim=64)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def load_pretrained_nasa_conv(self, checkpoint_path):
        """Explicitly load pre-trained Conv weights into frozen NASA stream."""
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if 'conv_state_dict' in ckpt and ckpt['conv_state_dict'] is not None:
                self.nasa_conv.load_state_dict(ckpt['conv_state_dict'])
            elif 'model_states' in ckpt and len(ckpt['model_states']) > 0:
                first_member = ckpt['model_states'][0]
                conv_weights = {k.replace('conv.', ''): v for k, v in first_member.items() if k.startswith('conv.')}
                self.nasa_conv.load_state_dict(conv_weights)
        except Exception as e:
            pass

    def forward(self, x_spec, x_phys):
        f_phys = self.phys_mlp(x_phys)
        with torch.no_grad():
            f_nasa = self.nasa_fc(F.relu(self.nasa_conv(x_spec)))
        f_local = self.local_fc(F.relu(self.local_conv(x_spec)))

        f_fused, gate_weights = self.fusion(f_phys, f_nasa, f_local)
        logits = self.classifier(f_fused)
        return logits, gate_weights

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation (SE) Block for 1D CNN features to adaptively reweight channels."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DualStreamECNNAttention(DualStreamECNN):
    """
    Extends DualStreamECNN with SE Channel Attention blocks 
    after the local conv layers to enhance faint trace element signals 
    and suppress background noise.
    """
    def __init__(self, input_channels, num_phys_features, num_classes, pretrained_checkpoint=None, hidden_dim=64):
        super().__init__(input_channels, num_phys_features, num_classes, pretrained_checkpoint, hidden_dim)
        
        # Add SE Block to Local Conv stream to suppress noise channels
        self.local_se = SEBlock1D(32, reduction=4)
        
    def forward(self, x_spec, x_phys):
        f_phys = self.phys_mlp(x_phys)
        
        with torch.no_grad():
            f_nasa = self.nasa_fc(F.relu(self.nasa_conv(x_spec)))
            
        # Apply attention to local stream before flatten
        f_local_conv = F.relu(self.local_conv(x_spec))
        f_local_conv = self.local_se(f_local_conv)
        f_local = self.local_fc(f_local_conv)
        
        f_fused, gate_weights = self.fusion(f_phys, f_nasa, f_local)
        logits = self.classifier(f_fused)
        return logits, gate_weights

if __name__ == '__main__':
    print("DualStreamECNN module ready with normalized NIST features.")
