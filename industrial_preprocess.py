import numpy as np
from scipy.stats import pearsonr

def group_and_average_spectra(X, y, names, corr_threshold=0.85):
    """
    Groups spectra by physical sample name, calculates correlation matrix,
    rejects outliers, and averages the remaining spectra.
    """
    unique_groups = {}
    for i, name in enumerate(names):
        # Extract physical sample ID (e.g., "A_123_4.csv" -> "A_123" or "11号重力_shot1" -> "11号重力")
        if "_shot" in name:
            group_id = name.rsplit('_shot', 1)[0]
        elif name.startswith("A_") or name.startswith("B_"):
            group_id = name.rsplit('_', 1)[0]
        else:
            group_id = name
            
        if group_id not in unique_groups:
            unique_groups[group_id] = []
        unique_groups[group_id].append(i)
        
    X_avg_list = []
    y_avg_list = []
    names_avg_list = []
    
    total_spectra = len(X)
    retained_spectra = 0
    
    for group_id, indices in unique_groups.items():
        if len(indices) == 1:
            X_avg_list.append(X[indices[0]])
            y_avg_list.append(y[indices[0]])
            names_avg_list.append(group_id)
            retained_spectra += 1
            continue
            
        group_X = X[indices]
        n_samples = len(group_X)
        
        # Calculate pairwise correlation matrix
        corr_matrix = np.ones((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Only use a subset of the spectrum for correlation to avoid being dominated by flat background
                # e.g., points above the median
                mask = group_X[i] > np.median(group_X[i])
                if np.sum(mask) > 10:
                    r, _ = pearsonr(group_X[i][mask], group_X[j][mask])
                else:
                    r = 0
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
                
        # Average correlation of each spectrum with the rest
        mean_corrs = (np.sum(corr_matrix, axis=1) - 1) / (n_samples - 1)
        
        # Keep spectra above threshold
        valid_indices = np.where(mean_corrs >= corr_threshold)[0]
        
        if len(valid_indices) == 0:
            # If all are bad, just keep the one with the highest correlation to others, or average all
            valid_indices = np.array([np.argmax(mean_corrs)])
            
        valid_X = group_X[valid_indices]
        avg_X = np.mean(valid_X, axis=0)
        
        X_avg_list.append(avg_X)
        y_avg_list.append(y[indices[0]])  # y is identical for the group
        names_avg_list.append(group_id)
        retained_spectra += len(valid_indices)
        
    print(f"[Outlier Rejection] Reduced {total_spectra} raw spectra to {len(X_avg_list)} physical samples.")
    print(f"[Outlier Rejection] Retained {retained_spectra}/{total_spectra} high-quality shots ({(retained_spectra/total_spectra)*100:.1f}%).")
    
    return np.array(X_avg_list), np.array(y_avg_list), names_avg_list

def correct_wavelength_drift(X, wavelengths, anchor_wl=259.94, search_window=1.0):
    """
    Corrects wavelength drift by aligning an anchor peak (e.g., Fe 259.94 nm).
    """
    print(f"[Drift Correction] Aligning spectra to anchor peak {anchor_wl} nm...")
    # Find indices for the search window
    mask = (wavelengths >= anchor_wl - search_window) & (wavelengths <= anchor_wl + search_window)
    window_wls = wavelengths[mask]
    
    if len(window_wls) == 0:
        return X
        
    X_shifted = np.zeros_like(X)
    shifts = []
    
    for i in range(len(X)):
        spec = X[i]
        window_intensities = spec[mask]
        
        # Find the actual peak in this spectrum
        actual_peak_idx = np.argmax(window_intensities)
        actual_peak_wl = window_wls[actual_peak_idx]
        
        # Calculate drift (tau)
        tau = actual_peak_wl - anchor_wl
        shifts.append(tau)
        
        # Shift the spectrum using interpolation
        # If tau > 0, the physical peak shifted to the right. 
        # We need to evaluate the spectrum at (wavelengths - tau) to pull it back left.
        shifted_spec = np.interp(wavelengths, wavelengths - tau, spec)
        X_shifted[i] = shifted_spec
        
    mean_shift = np.mean(shifts)
    std_shift = np.std(shifts)
    print(f"[Drift Correction] Applied mean shift: {mean_shift:.3f} nm (Std: {std_shift:.3f} nm)")
    
    return X_shifted
