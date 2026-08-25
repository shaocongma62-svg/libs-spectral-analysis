import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from wavelet_preprocess import wavelet_denoise

class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1) if X.ndim == 2 else torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def normalize_name(name):
    """Normalize target standard names for string matching."""
    return str(name).upper().replace('-', '').replace('.', '').replace(' ', '')

def oxide_to_element_label_batch(y_oxides, elements=config.ALL_ELEMENTS, threshold=config.DEFAULT_HYPERPARAMS["element_threshold_wt%"]):
    """Convert a batch of oxide weight percentages into multi-label binary matrix."""
    n_samples = len(y_oxides)
    y_binary = np.zeros((n_samples, len(elements)), dtype=np.float32)

    for i in range(n_samples):
        elem_wt = {e: 0.0 for e in elements if e != 'O'}
        any_elem_present = False

        for j, ox in enumerate(config.OXIDES):
            val = y_oxides[i, j]
            if val <= 0 or ox not in config.OXIDE_ELEM:
                continue
            for elem, frac in config.OXIDE_ELEM[ox].items():
                if elem in elem_wt:
                    elem_wt[elem] += val * frac

        for j, elem in enumerate(elements):
            if elem == 'O':
                continue
            if elem_wt.get(elem, 0.0) >= threshold:
                y_binary[i, j] = 1.0
                any_elem_present = True

        if 'O' in elements:
            y_binary[i, elements.index('O')] = 1.0 if any_elem_present else 0.0

    return y_binary

def load_20260422_dataset():
    out_dir = os.path.join(config.BASE_DIR, '20260422_processed')
    X = np.load(os.path.join(out_dir, 'X_stitched.npy'))
    y = np.load(os.path.join(out_dir, 'y_stitched.npy'))
    names = np.load(os.path.join(out_dir, 'names.npy'))
    return X, y, list(names)

def oxide_to_element_label_single(oxide_dict, elements=config.ALL_ELEMENTS, threshold=config.DEFAULT_HYPERPARAMS["element_threshold_wt%"]):
    """Convert a single sample oxide dictionary into multi-label binary vector."""
    elem_wt = {e: 0.0 for e in elements if e != 'O'}
    y_binary = np.zeros(len(elements), dtype=np.float32)
    any_elem_present = False

    for ox_name, val in oxide_dict.items():
        if val <= 0 or ox_name not in config.OXIDE_ELEM:
            continue
        for elem, frac in config.OXIDE_ELEM[ox_name].items():
            if elem in elem_wt:
                elem_wt[elem] += val * frac

    for j, elem in enumerate(elements):
        if elem == 'O':
            continue
        if elem_wt.get(elem, 0.0) >= threshold:
            y_binary[j] = 1.0
            any_elem_present = True

    if 'O' in elements:
        y_binary[elements.index('O')] = 1.0 if any_elem_present else 0.0

    return y_binary

def load_nasa_calibration_data(data_dir=config.NASA_DATA_DIR, apply_wavelet=True):
    """Load Earth calibration standards (spectra and composition labels)."""
    calib_csv = os.path.join(data_dir, 'msl_ccam_libs_calib.csv')
    comp_csv = os.path.join(data_dir, 'ccam_calibration_compositions.csv')

    if not os.path.exists(calib_csv) or not os.path.exists(comp_csv):
        raise FileNotFoundError(f"NASA calibration files missing in {data_dir}")

    calib_df = pd.read_csv(calib_csv)
    comp_df = pd.read_csv(comp_csv).dropna(subset=['Target'])
    comp_df['norm_name'] = comp_df['Target'].apply(normalize_name)

    spec_cols = [c for c in calib_df.columns if c != 'Wvl']
    X_raw, y_ox, names = [], [], []

    for col in spec_cols:
        parts = col.split('.')
        target_name = parts[0]
        match = comp_df[comp_df['norm_name'] == normalize_name(target_name)]
        if match.empty:
            continue
        row = match.iloc[0]
        oxides_val = row[config.OXIDES].fillna(0.0).values.astype(np.float32)

        X_raw.append(calib_df[col].values.astype(np.float32))
        y_ox.append(oxides_val)
        names.append(target_name)

    X = np.array(X_raw, dtype=np.float32)
    y_ox = np.array(y_ox, dtype=np.float32)
    names = np.array(names)
    wavelengths = calib_df['Wvl'].values.astype(np.float32)

    if apply_wavelet:
        X = wavelet_denoise(X, wavelet=config.DEFAULT_HYPERPARAMS["wavelet_name"], level=config.DEFAULT_HYPERPARAMS["wavelet_level"], mode='soft')

    y = oxide_to_element_label_batch(y_ox, elements=config.GEOCHEMICAL_ELEMENTS)

    # Filter non-zero label samples
    mask = y.sum(axis=1) > 0
    return X[mask], y[mask], names[mask], wavelengths

def load_mars_ccs_spectrum(ccs_path, apply_wavelet=True):
    """Load a ChemCam Mars CCS spectrum file."""
    with open(ccs_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('#'):
            try:
                float(line.split(',')[0])
                data_start = i
                break
            except ValueError:
                continue

    df = pd.read_csv(ccs_path, skiprows=data_start, header=None)
    wavelengths = df.iloc[:, 0].values.astype(np.float32)
    shots = df.iloc[:, 1:].values.astype(np.float32).T  # (n_shots, 6144)

    if apply_wavelet:
        shots = wavelet_denoise(shots, wavelet=config.DEFAULT_HYPERPARAMS["wavelet_name"], level=config.DEFAULT_HYPERPARAMS["wavelet_level"], mode='soft')

    return wavelengths, shots

def load_moc_labels(moc_path, target_sclk):
    """Load MOC table and extract predicted oxide compositions for a target by SCLK or name."""
    if not os.path.exists(moc_path):
        return None, None, None

    col_map = {'SiO2': 2, 'TiO2': 6, 'Al2O3': 10, 'FeOT': 14, 'MgO': 18, 'CaO': 22, 'Na2O': 26, 'K2O': 30, 'MnO': 34}

    with open(moc_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in lines[7:]:
        parts = line.split(',')
        if len(parts) < 35:
            continue
        if str(target_sclk).upper() in parts[0].upper():
            oxides = {}
            for ox, ci in col_map.items():
                try:
                    oxides[ox] = float(parts[ci]) if parts[ci].strip() else 0.0
                except (ValueError, IndexError):
                    oxides[ox] = 0.0
            return oxides, parts[0].strip(), parts[1].strip()

    return None, None, None

# Helper functions for pure-NumPy physical spectral transformations
def _subpixel_shift(spec, shift_pixels):
    """Apply sub-pixel 1D wavelength shift via linear interpolation."""
    n_channels = len(spec)
    x_grid = np.arange(n_channels, dtype=np.float32)
    x_shifted = x_grid + shift_pixels
    return np.interp(x_shifted, x_grid, spec, left=spec[0], right=spec[-1])

def _gaussian_broaden(spec, sigma):
    """Apply 1D Gaussian kernel convolution to simulate Stark line broadening."""
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(spec, kernel, mode='same')

def physical_augment_data(X, y, factor=config.DEFAULT_HYPERPARAMS["augmentation_factor"], ratio=config.DEFAULT_HYPERPARAMS["noise_std_ratio"], seed=config.DEFAULT_HYPERPARAMS["seed"]):
    """
    Physics-informed multi-strategy spectrum augmentation:
    - 50% Strategy A: Pure Gaussian noise (Readout / Shot detector noise)
    - 20% Strategy B: Sub-pixel Wavelength Shift + Gaussian noise (Grating thermal/mechanical drift)
    - 30% Strategy C: Stark Broadening + Shift + Gaussian noise (Plasma electron density & laser energy jitter)
    """
    rng = np.random.RandomState(seed)
    n_samples, n_channels = X.shape

    X_aug_list, y_aug_list = [X], [y]

    for _ in range(factor - 1):
        X_batch = np.zeros_like(X)
        for i in range(n_samples):
            spec = X[i].copy()
            r = rng.uniform(0.0, 1.0)
            noise_std = spec.std() * rng.uniform(ratio * 0.5, ratio * 1.5)

            if r < 0.50:
                # Strategy A: Pure Gaussian Noise (50%)
                spec_aug = spec + rng.normal(0, noise_std, n_channels)

            elif r < 0.70:
                # Strategy B: Wavelength Shift + Gaussian (20%)
                shift_pixels = rng.uniform(-2.0, 2.0)
                spec_shifted = _subpixel_shift(spec, shift_pixels)
                spec_aug = spec_shifted + rng.normal(0, noise_std, n_channels)

            else:
                # Strategy C: Stark Broadening + Shift + Gaussian (30%)
                sigma_broadening = rng.uniform(0.5, 1.2)
                spec_broad = _gaussian_broaden(spec, sigma_broadening)
                
                shift_pixels = rng.uniform(-2.0, 2.0)
                spec_shifted = _subpixel_shift(spec_broad, shift_pixels)
                spec_aug = spec_shifted + rng.normal(0, noise_std, n_channels)

            X_batch[i] = spec_aug.astype(np.float32)

        X_aug_list.append(X_batch)
        y_aug_list.append(y)

    return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)

# Backwards compatibility alias
augment_data = physical_augment_data

class AlignedDataset(Dataset):
    """PyTorch Dataset holding dual-stream inputs (spectrum, physical features, labels)."""
    def __init__(self, X_spec, X_phys, y):
        self.X_spec = torch.from_numpy(X_spec).float().unsqueeze(1) if X_spec.ndim == 2 else torch.from_numpy(X_spec).float()
        self.X_phys = torch.from_numpy(X_phys).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X_spec)

    def __getitem__(self, i):
        return self.X_spec[i], self.X_phys[i], self.y[i]

DATASET_A_SUBSTANCE_MAP = {
    "一氧化锰": ['Mn', 'O'], "四氧化三铁": ['Fe', 'O'], "氧化硅": ['Si', 'O'], "氧化钙": ['Ca', 'O'],
    "氧化钛": ['Ti', 'O'], "氧化铝": ['Al', 'O'], "氧化镁": ['Mg', 'O'], "硫化铁": ['Fe', 'S'],
    "硫酸钾": ['K', 'S', 'O'], "磷酸二氢钾": ['K', 'P', 'O'], "碳粉": ['C'], "锌粒": ['Zn'],
    "fe粉仓": ['Fe'], "氯化钾": ['K'], "11号重力": ['Fe', 'Si', 'Ca', 'Al', 'Mg', 'C', 'O'],
    "9号仓布袋": ['Fe', 'Si', 'Ca', 'Al', 'Mg', 'C', 'O'], "窑渣": ['Fe', 'Si', 'Ca', 'Al', 'Mg', 'C', 'O'],
    "脱zn1号仓": ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O'],
    "HG_ARdeng": ['Fe']
}

DATASET_B_SAMPLE_MAP = {
    'TR-01': ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O'],
    'TR-03': ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O'],
    'TR-07': ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O'],
    'TR-10': ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O'],
    'TR-12': ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O'],
    'TR-20': ['Fe', 'Zn', 'Si', 'Ca', 'Mg', 'Al', 'C', 'O']
}

CALIBRATION_COEFFS = {
    "170": {"a": 0.022, "b": 149.8},
    "240": {"a": 0.022, "b": 219.8},
    "260": {"a": 0.022, "b": 239.8},
    "290": {"a": 0.022, "b": 269.9},
    "320": {"a": 0.022, "b": 300.0},
    "350": {"a": 0.022, "b": 330.2},
    "380": {"a": 0.022, "b": 360.2},
    "410": {"a": 0.022, "b": 390.5},
    "440": {"a": 0.022, "b": 420.6},
    "470": {"a": 0.022, "b": 450.8},
    "500": {"a": 0.021, "b": 480.9},
    "530": {"a": 0.021, "b": 511.1},
    "560": {"a": 0.021, "b": 541.3},
    "590": {"a": 0.021, "b": 571.5},
    "620": {"a": 0.020, "b": 601.6},
    "650": {"a": 0.020, "b": 631.6},
    "770": {"a": 0.020, "b": 751.6},
}

def load_aligned_dataset_a(base_folder=config.LOCAL_DATA_DIR_A, target_wvl=None):
    """Load Dataset A and align to NASA target physical wavelength grid via linear interpolation."""
    if target_wvl is None:
        _, _, _, target_wvl = load_nasa_calibration_data()

    if not os.path.exists(base_folder):
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

    tf = []
    for r, d, f in os.walk(base_folder):
        c = sorted([x for x in f if x.endswith('.csv')])
        if len(c) == 13:
            tf = c
            break

    X_list, y_list, names = [], [], []
    for d in sorted(x for x in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, x))):
        dp = os.path.join(base_folder, d)
        for s in os.listdir(dp):
            sp = os.path.join(dp, s)
            if not os.path.isdir(sp): continue
            v = [f for f in os.listdir(sp) if f.endswith('.csv') and f in tf]
            if len(v) < 8: continue

            dfs = {f: pd.read_csv(os.path.join(sp, f)) for f in v}
            first_df = list(dfs.values())[0]
            spec_cols = [c for c in first_df.columns if 'Spectrum' in c]
            sub_name = s.split('-')[0]

            for c in spec_cols:
                wvl_pts, int_pts = [], []
                for t in tf:
                    if t in dfs and c in dfs[t].columns:
                        df_t = dfs[t]
                        seg_key = t.replace('nm.csv', '')
                        if seg_key in CALIBRATION_COEFFS:
                            a = CALIBRATION_COEFFS[seg_key]["a"]
                            b = CALIBRATION_COEFFS[seg_key]["b"]
                            w_arr = (a * np.arange(len(df_t), dtype=np.float32) + b)
                        else:
                            w_col = [col for col in df_t.columns if 'Wavelength' in col or 'wavelength' in col or col == df_t.columns[0]][0]
                            w_arr = df_t[w_col].values.astype(np.float32)
                        
                        n_pts = min(1900, len(w_arr))
                        wvl_pts.extend(w_arr[:n_pts])
                        int_pts.extend(df_t[c].values[:n_pts])

                wvl_pts = np.array(wvl_pts, dtype=np.float32)
                int_pts = np.array(int_pts, dtype=np.float32)
                sort_idx = np.argsort(wvl_pts)
                spec_aligned = np.interp(target_wvl, wvl_pts[sort_idx], int_pts[sort_idx], left=0.0, right=0.0).astype(np.float32)

                X_list.append(spec_aligned)
                names.append(f"A_{d}/{s}_{c}")

                y_vec = np.zeros(len(config.ALL_ELEMENTS), dtype=np.float32)
                for elem in DATASET_A_SUBSTANCE_MAP.get(sub_name, []):
                    if elem in config.ALL_ELEMENTS:
                        y_vec[config.ALL_ELEMENTS.index(elem)] = 1.0
                y_list.append(y_vec)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), names

def load_aligned_dataset_b(base_dir=config.LOCAL_DATA_DIR_B, target_wvl=None):
    """Load Dataset B and align to NASA target physical wavelength grid via linear interpolation."""
    import zipfile
    from xml.etree import ElementTree as ET

    if target_wvl is None:
        _, _, _, target_wvl = load_nasa_calibration_data()

    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('TR')])
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    X_list, y_list, names = [], [], []

    for sd in subdirs:
        if sd not in DATASET_B_SAMPLE_MAP: continue
        xlsx_dir = os.path.join(base_dir, sd, 'xlsx')
        if not os.path.exists(xlsx_dir): xlsx_dir = os.path.join(base_dir, sd)
        target_files = ['405nm_with_wavelength.xlsx', '455nm_with_wavelength.xlsx', '500nm_with_wavelength.xlsx', '625nm_with_wavelength.xlsx']
        band_wvl_int = {}

        for tf in target_files:
            fp = os.path.join(xlsx_dir, tf)
            if not os.path.exists(fp): continue
            z = zipfile.ZipFile(fp)
            tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
            rows = tree.findall('.//x:row', ns)
            wvls, shots_data = [], [[] for _ in range(10)]
            for r in rows[1:]:
                cells = r.findall('x:c', ns)
                if len(cells) >= 12:
                    w_val = float(cells[1].find('x:v', ns).text) if cells[1].find('x:v', ns) is not None else 0.0
                    wvls.append(w_val)
                    for col_idx in range(2, 12):
                        v = cells[col_idx].find('x:v', ns)
                        val = float(v.text) if (v is not None and v.text) else 0.0
                        shots_data[col_idx - 2].append(val)
            band_wvl_int[tf] = (np.array(wvls, dtype=np.float32), np.array(shots_data, dtype=np.float32))

        if len(band_wvl_int) == 4:
            for shot_i in range(10):
                wvl_pts, int_pts = [], []
                for tf in target_files:
                    w_arr, s_arr = band_wvl_int[tf]
                    wvl_pts.extend(w_arr)
                    int_pts.extend(s_arr[shot_i])
                wvl_pts = np.array(wvl_pts, dtype=np.float32)
                int_pts = np.array(int_pts, dtype=np.float32)
                sort_idx = np.argsort(wvl_pts)
                spec_aligned = np.interp(target_wvl, wvl_pts[sort_idx], int_pts[sort_idx], left=0.0, right=0.0).astype(np.float32)

                X_list.append(spec_aligned)
                y_vec = np.zeros(len(config.ALL_ELEMENTS), dtype=np.float32)
                for elem in DATASET_B_SAMPLE_MAP[sd]:
                    if elem in config.ALL_ELEMENTS:
                        y_vec[config.ALL_ELEMENTS.index(elem)] = 1.0
                y_list.append(y_vec)
                names.append(f"B_{sd}_shot{shot_i}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), names

def load_aligned_combined_dataset():
    """Load and merge aligned Dataset A (172 spectra) + Dataset B (60 spectra) -> 232 total spectra."""
    Xa, ya, names_a = load_aligned_dataset_a()
    Xb, yb, names_b = load_aligned_dataset_b()
    X_all = np.concatenate([Xa, Xb], axis=0) if len(Xb) > 0 else Xa
    y_all = np.concatenate([ya, yb], axis=0) if len(yb) > 0 else ya
    names_all = names_a + names_b
    return X_all, y_all, names_all

if __name__ == '__main__':
    print("Testing Physics-Informed Dataset Loader module...")
    X, y, names, wvl = load_nasa_calibration_data()
    print(f"NASA Calibration Loaded: {len(X)} samples, {X.shape[1]} channels, {y.shape[1]} target elements.")

    Xa, ya, names_a = load_aligned_dataset_a()
    Xb, yb, names_b = load_aligned_dataset_b()
    X_ab, y_ab, names_ab = load_aligned_combined_dataset()
    print(f"Dataset A Loaded: {len(Xa)} spectra.")
    print(f"Dataset B Loaded: {len(Xb)} spectra.")
    print(f"Combined A+B Loaded: {len(X_ab)} spectra ({X_ab.shape[1]} channels).")

    X_aug, y_aug = physical_augment_data(X_ab[:10], y_ab[:10], factor=10)
    print(f"Physical Augmentation Test: input 10 spectra -> output {len(X_aug)} augmented spectra.")
    print("Dataset Loader test passed successfully.")
