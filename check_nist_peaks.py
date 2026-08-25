"""Validate NIST emission-line positions against observed local maxima.

Uses raw (pre-SNV) sample-level spectra so zero-padded regions are not mistaken
for real signal. Saves a CSV report to output/nist_peak_validation.csv.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_aligned_dataset_a, load_aligned_dataset_b, load_nasa_calibration_data
from model_dual_stream import NIST_EMISSION_LINES


def _group_spectra(X, names, key_fn):
    groups = {}
    for i, name in enumerate(names):
        key = key_fn(name)
        groups.setdefault(key, []).append(X[i])
    return {k: np.mean(v, axis=0) for k, v in sorted(groups.items())}


def _group_a(name):
    return name.split("/")[1].split("_")[0].split("-")[0]


def _group_b(name):
    return name.split("_")[1]


def validate_dataset(dataset_name, X, names, wvl, key_fn, window_nm, tol_nm):
    spectra = _group_spectra(X, names, key_fn)
    rows = []
    for elem, lines in NIST_EMISSION_LINES.items():
        for line_nm in lines:
            if line_nm < 240.0 or line_nm > 760.0:
                continue
            idx_mask = (wvl >= line_nm - window_nm) & (wvl <= line_nm + window_nm)
            observed = []
            for spec in spectra.values():
                window = spec[idx_mask]
                if window.size == 0:
                    continue
                peak_idx_local = int(np.argmax(np.abs(window)))
                if window[peak_idx_local] <= 1e-3:
                    continue
                global_idx = np.where(idx_mask)[0][peak_idx_local]
                observed.append((wvl[global_idx], window[peak_idx_local]))

            if not observed:
                status = "BLIND"
                observed_peak = np.nan
                offset = np.nan
                n_signal = 0
                max_intensity = 0.0
            else:
                obs_wvls = np.array([o[0] for o in observed])
                ints = np.array([o[1] for o in observed])
                observed_peak = float(np.median(obs_wvls))
                offset = observed_peak - line_nm
                n_signal = len(observed)
                max_intensity = float(np.max(ints))
                if abs(offset) <= tol_nm:
                    status = "OK"
                else:
                    status = "SHIFT"

            rows.append({
                "dataset": dataset_name,
                "element": elem,
                "line_nm": line_nm,
                "n_signal": n_signal,
                "n_samples": len(spectra),
                "observed_peak_nm": observed_peak,
                "offset_nm": offset,
                "max_intensity": max_intensity,
                "status": status,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate NIST peak positions.")
    parser.add_argument("--dataset", choices=["A", "B", "all"], default="all")
    parser.add_argument("--window", type=float, default=0.3, help="search half-window in nm")
    parser.add_argument("--tol", type=float, default=0.15, help="allowed peak offset in nm")
    parser.add_argument("--out", default=os.path.join("output", "nist_peak_validation.csv"))
    args = parser.parse_args()

    _, _, _, target_wvl = load_nasa_calibration_data()
    frames = []

    if args.dataset in ("A", "all"):
        Xa, _, names_a = load_aligned_dataset_a(target_wvl=target_wvl)
        frames.append(validate_dataset("A", Xa, names_a, target_wvl, _group_a, args.window, args.tol))

    if args.dataset in ("B", "all"):
        Xb, _, names_b = load_aligned_dataset_b(target_wvl=target_wvl)
        frames.append(validate_dataset("B", Xb, names_b, target_wvl, _group_b, args.window, args.tol))

    report = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(report.to_string(index=False))
    print(f"\nSaved peak validation report to {args.out}")


if __name__ == "__main__":
    main()
