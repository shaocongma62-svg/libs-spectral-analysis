import numpy as np
import pywt


def wavelet_denoise(X_raw, wavelet="db4", level=1, mode="soft"):
    X_denoised = np.empty_like(X_raw)
    for i, row in enumerate(X_raw):
        coeffs = pywt.wavedec(row, wavelet, mode="per", level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(row)))
        coeffs_thresh = (
            [coeffs[0]]
            + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
        )
        rec = pywt.waverec(coeffs_thresh, wavelet, mode="per")
        X_denoised[i] = rec[:len(row)]
    X_denoised = X_denoised[:, : X_raw.shape[1]]
    return X_denoised
