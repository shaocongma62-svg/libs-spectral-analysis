import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from industrial_preprocess import group_and_average_spectra

# Wavelength calibration coefficients from the user's screenshot
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

# Define a high-resolution universal wavelength grid
GLOBAL_WVL = np.arange(150.0, 760.0, 0.02)

def stitch_sample(sample_dir):
    """
    Reads all segment CSVs in a sample directory,
    stitches them into 10 full spectra.
    Returns: X_shots (10, len(GLOBAL_WVL)), valid (bool)
    """
    files = [f for f in os.listdir(sample_dir) if f.endswith('nm.csv')]
    if not files:
        return None
        
    # We assume 10 shots (Spectrum1 to Spectrum10)
    X_shots = np.zeros((10, len(GLOBAL_WVL)))
    overlap_counts = np.zeros((10, len(GLOBAL_WVL)))
    
    for f in files:
        segment = f.replace('nm.csv', '')
        if segment not in CALIBRATION_COEFFS:
            print(f"  Warning: Unknown segment {segment} in {sample_dir}")
            continue
            
        coeffs = CALIBRATION_COEFFS[segment]
        path = os.path.join(sample_dir, f)
        
        try:
            df = pd.read_csv(path)
            # Remove leading/trailing spaces from column names (e.g., " Spectrum1" -> "Spectrum1")
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            continue
            
        # Calculate true wavelengths for this segment using a*Index + b
        wvl = coeffs['a'] * np.arange(len(df)) + coeffs['b']
        
        # Iterate over the 10 shots
        for i in range(10):
            col_name = f'Spectrum{i+1}'
            if col_name in df.columns:
                intensities = df[col_name].values
                # Interpolate onto global grid
                # Only update where GLOBAL_WVL is within the segment's physical range
                mask = (GLOBAL_WVL >= wvl.min()) & (GLOBAL_WVL <= wvl.max())
                interpolated = np.interp(GLOBAL_WVL[mask], wvl, intensities)
                
                X_shots[i, mask] += interpolated
                overlap_counts[i, mask] += 1
                
    # Average the overlapping regions (e.g. 270-280nm covered by both 260nm and 290nm files)
    valid_mask = overlap_counts > 0
    X_shots[valid_mask] = X_shots[valid_mask] / overlap_counts[valid_mask]
    
    return X_shots, overlap_counts[0] > 0

def build_dataset():
    data_dir = os.path.join(BASE_DIR, '20260422data')
    all_X = []
    all_names = []
    
    print("=== Phase 1: Stitching Segmented Spectra ===")
    
    for date_folder in os.listdir(data_dir):
        date_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(date_path): continue
            
        for sample_folder in os.listdir(date_path):
            sample_path = os.path.join(date_path, sample_folder)
            if not os.path.isdir(sample_path): continue
                
            print(f"Processing {date_folder} / {sample_folder}...")
            res = stitch_sample(sample_path)
            if res is not None:
                X_shots, valid_mask = res
                
                # Check if this sample actually got any valid data stitched
                if np.sum(valid_mask) == 0:
                    print(f"  -> Failed: No valid overlapping data.")
                    continue
                    
                # Append 10 individual shots with the _shot prefix
                for i in range(10):
                    all_X.append(X_shots[i])
                    all_names.append(f"{sample_folder}_shot{i+1}")

    all_X = np.array(all_X)
    
    if len(all_X) == 0:
        print("No data parsed. Exiting.")
        return
        
    print("\n=== Phase 2: Applying Industrial Outlier Rejection ===")
    dummy_y = np.zeros((len(all_X), 1))
    X_avg, _, names_avg = group_and_average_spectra(all_X, dummy_y, all_names, corr_threshold=0.85)
    
    out_dir = os.path.join(BASE_DIR, '20260422_processed')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'X_stitched.npy'), X_avg)
    np.save(os.path.join(out_dir, 'names.npy'), names_avg)
    np.save(os.path.join(out_dir, 'wavelengths.npy'), GLOBAL_WVL)
    
    print(f"\nSuccessfully saved stitched dataset to {out_dir}")
    print(f"Final Physical Samples Found: {len(names_avg)}")
    print(f"Unique Sample Names: {set(names_avg)}")
    
if __name__ == '__main__':
    build_dataset()
