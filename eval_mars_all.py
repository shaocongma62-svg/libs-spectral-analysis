import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import (
    load_nasa_calibration_data,
    load_mars_ccs_spectrum,
    load_moc_labels,
    oxide_to_element_label_single
)
from model_dual_stream import NISTFeatureExtractor, DualStreamECNN
from train_dual_stream_ecnn import ensemble_predict_dual_stream, load_earth_to_mars_corr

device = config.DEVICE
print(f"=== Phase 4: Full Multi-Target Mars Zero-Shot Evaluation ===")
print(f"Using compute device: {device}")

def evaluate_all_mars_targets():
    checkpoint_path = os.path.join(config.SAVE_DIR, 'dual_stream_ecnn_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file missing: {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model_states = ckpt['model_states']
    m_cal, s_cal = ckpt['scaler_mean'], ckpt['scaler_std']
    m_phys, s_phys = ckpt['phys_mean'], ckpt['phys_std']
    active_elements = ckpt['active_elements']
    n_active = len(active_elements)

    print(f"Loaded checkpoint with {len(model_states)} ensemble members.")
    print(f"Active geochemical elements ({n_active}): {active_elements}")

    # Load Exact Earth Calibration Wavelengths Grid
    X_cal, y_cal, std_names, wvl = load_nasa_calibration_data()
    extractor = NISTFeatureExtractor(wvl, elements=config.GEOCHEMICAL_ELEMENTS)
    corr_factor = load_earth_to_mars_corr(config.NASA_DATA_DIR, 6144)

    # Find all Mars CCS files and MOC files in NASA data
    ccs_files = sorted(glob.glob(os.path.join(config.NASA_DATA_DIR, "cl5_*ccs_*.csv")))
    moc_files = sorted(glob.glob(os.path.join(config.NASA_DATA_DIR, "moc_*.csv")))

    print(f"Found {len(ccs_files)} Mars CCS spectrum files and {len(moc_files)} MOC tables.")

    target_results = []
    seen_targets = set()

    for ccs_path in ccs_files:
        fn = os.path.basename(ccs_path)
        # Extract SCLK from CCS filename (e.g. cl5_430599902ccs_... -> 430599902)
        sclk = fn.split('_')[1].replace('ccs', '') if '_' in fn else fn[:9]

        # Search across all MOC files for this SCLK
        moc_oxides, ccs_name, target_name = None, None, None
        for moc_path in moc_files:
            oxides, cname, tname = load_moc_labels(moc_path, sclk)
            if oxides is not None:
                moc_oxides, ccs_name, target_name = oxides, cname, tname
                break

        if moc_oxides is None or target_name in seen_targets:
            continue
        seen_targets.add(target_name)

        actual_ml = oxide_to_element_label_single(moc_oxides, elements=config.GEOCHEMICAL_ELEMENTS)
        actual_dict = {config.GEOCHEMICAL_ELEMENTS[i]: actual_ml[i] for i in range(len(config.GEOCHEMICAL_ELEMENTS))}

        wv_m, shots = load_mars_ccs_spectrum(ccs_path)
        shots_corrected = shots * corr_factor
        shots_s = (shots_corrected - m_cal) / s_cal

        shots_phys = extractor.transform(shots_corrected)
        shots_phys_s = (shots_phys - m_phys) / s_phys

        probs, preds, gates = ensemble_predict_dual_stream(
            model_states, shots_s, shots_phys_s, shots_s.shape[1], n_active, checkpoint_path, threshold=0.5
        )
        avg_probs = probs.mean(axis=0)
        avg_gates = gates.mean(axis=0)

        matches = 0
        elem_matches = {}
        for i, elem in enumerate(active_elements):
            act = int(actual_dict.get(elem, 0))
            pred = int(avg_probs[i] > 0.5)
            match = (act == pred)
            elem_matches[elem] = match
            if match:
                matches += 1

        acc = matches / n_active * 100
        target_results.append({
            'target_name': target_name,
            'sclk': sclk,
            'filename': fn,
            'accuracy': acc,
            'matches': matches,
            'n_active': n_active,
            'gate_phys': avg_gates[0],
            'gate_nasa': avg_gates[1],
            'gate_local': avg_gates[2],
            'elem_matches': elem_matches,
            'avg_probs': avg_probs
        })

        print(f"Target '{target_name:<16}' (SCLK {sclk}): Accuracy = {matches}/{n_active} ({acc:5.1f}%) | Gates=(Phys:{avg_gates[0]:.2f}, NASA:{avg_gates[1]:.2f}, Local:{avg_gates[2]:.2f})")

    if not target_results:
        print("No valid MOC-matched Mars targets found!")
        return

    # Aggregate Full Matrix Results
    df_res = pd.DataFrame(target_results)
    total_m = sum(r['matches'] for r in target_results)
    total_e = sum(r['n_active'] for r in target_results)
    overall_acc = total_m / total_e * 100

    print("\n" + "=" * 60)
    print(f"=== Full Mars Multi-Target Zero-Shot Summary ===")
    print(f"Total Unique Targets Evaluated : {len(target_results)}")
    print(f"Overall Element Accuracy       : {total_m}/{total_e} ({overall_acc:.2f}%)")
    print(f"Mean Gate Weights Allocated    : Phys={df_res['gate_phys'].mean():.3f}, NASA={df_res['gate_nasa'].mean():.3f}, Local={df_res['gate_local'].mean():.3f}")
    print("=" * 60)

    # Plot Full Mars Zero-Shot Summary Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    targets = [r['target_name'] for r in target_results]
    accs = [r['accuracy'] for r in target_results]

    bars = ax.bar(targets, accs, color='#3498DB', edgecolor='black', alpha=0.85, width=0.55)
    ax.axhline(y=overall_acc, color='red', ls='--', lw=1.5, label=f'Mean Accuracy ({overall_acc:.1f}%)')
    ax.set_ylabel('Zero-Shot Accuracy (%)', fontsize=12)
    ax.set_xlabel('Mars Target Rocks', fontsize=12)
    ax.set_title('Phase 4: Full Multi-Target Mars Zero-Shot Accuracy (Dual-Stream ECNN)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.set_xticklabels(targets, rotation=35, ha='right', fontsize=10)
    ax.grid(axis='y', ls=':', alpha=0.6)
    ax.legend(fontsize=11)

    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_fig = os.path.join(config.OUTPUT_DIR, 'mars_full_multitarget_zeroshot.png')
    plt.savefig(save_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Phase 4 Evaluation Summary Plot: {save_fig}")

if __name__ == '__main__':
    evaluate_all_mars_targets()
