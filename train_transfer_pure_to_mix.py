import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import config
from dataset_loader import load_20260422_dataset, load_aligned_dataset_a, load_aligned_dataset_b, load_nasa_calibration_data
from model_dual_stream import NISTFeatureExtractor
from wavelet_preprocess import wavelet_denoise

print("=== Phase 5: Zero-Shot Transfer (Pure Chemical Substances -> Mixed Industrial Dust) ===")

PURE_NAMES = ['一氧化锰', '四氧化三铁', '氧化硅', '氧化钙', '氧化钛', '氧化铝', '氧化镁', '碳粉', '锌粒', 'fe粉仓', '氯化钠', '氯化钾', '硫化铁', '硫酸钾', '磷酸二氢钾']

# 1. Load 20260422 Dataset & Filter ONLY Pure Chemical Substances for Training
X_2026, y_2026, names_2026 = load_20260422_dataset()
WVL_high = np.load(os.path.join(BASE_DIR, '20260422_processed', 'wavelengths.npy'))

pure_train_indices = []
for i, name in enumerate(names_2026):
    base_name = name.split('-')[0]
    if base_name in PURE_NAMES or name in PURE_NAMES:
        pure_train_indices.append(i)

X_pure_train = X_2026[pure_train_indices]
y_pure_train = y_2026[pure_train_indices]
names_pure_train = [names_2026[i] for i in pure_train_indices]

print(f"Training Set (Pure Substances Only): {len(names_pure_train)} samples across pure chemical reagents.")
print(f"  Samples: {set([n.split('-')[0] for n in names_pure_train])}")

# 2. Load Dataset A & B Mixed Industrial Dust Samples for Blind Testing
Xa, ya, names_a = load_aligned_dataset_a(target_wvl=WVL_high)
Xb, yb, names_b = load_aligned_dataset_b(target_wvl=WVL_high)

# Filter Dataset A for industrial mixtures ONLY (11号重力, 9号仓布袋)
mix_a_indices = []
for i, name in enumerate(names_a):
    if '11号重力' in name or '9号仓布袋' in name:
        mix_a_indices.append(i)

X_mix_a = Xa[mix_a_indices] if len(mix_a_indices) > 0 else np.array([], dtype=np.float32)
y_mix_a = ya[mix_a_indices] if len(mix_a_indices) > 0 else np.array([], dtype=np.float32)
names_mix_a = [names_a[i] for i in mix_a_indices]

# Combine Dataset A mixtures + Dataset B mixtures
if len(X_mix_a) > 0 and len(Xb) > 0:
    X_test_raw = np.concatenate([X_mix_a, Xb], axis=0)
    y_test = np.concatenate([y_mix_a, yb], axis=0)
    names_test = names_mix_a + names_b
elif len(Xb) > 0:
    X_test_raw = Xb
    y_test = yb
    names_test = names_b
else:
    X_test_raw = X_mix_a
    y_test = y_mix_a
    names_test = names_mix_a

print(f"\nBlind Test Set (Industrial Dust Mixtures Only): {len(names_test)} samples (0% Overlap with Train!).")

# 3. Wavelet Denoising & Pre-Sanitization
X_train_clean = np.nan_to_num(X_pure_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test_clean = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

X_train_denoised = wavelet_denoise(X_train_clean, wavelet='db4', level=1, mode='soft')
X_test_denoised = wavelet_denoise(X_test_clean, wavelet='db4', level=1, mode='soft')

# 4. Feature Extraction
extractor = NISTFeatureExtractor(wavelengths=WVL_high)
F_train_raw = extractor.transform(X_train_denoised)
F_test_raw = extractor.transform(X_test_denoised)

keep_indices = []
for i in range(len(extractor.peak_indices)):
    keep_indices.extend([i*6, i*6+1, i*6+2, i*6+3])

F_train = np.nan_to_num(F_train_raw[:, keep_indices], nan=0.0)
F_test = np.nan_to_num(F_test_raw[:, keep_indices], nan=0.0)

# 5. Evaluate Zero-Shot Transfer
print("\n" + "=" * 65)
print("=== Transfer Results (Strict Pure -> Mix, 0% Overlap) ===")
print("=" * 65)
print(f"{'Element':<10} | {'RF F1-Score':<15} | {'PLS F1-Score':<15} | {'Train Positives'}")
print("-" * 65)

for elem_idx, elem in enumerate(config.ALL_ELEMENTS):
    yt = y_pure_train[:, elem_idx]
    ytest = y_test[:, elem_idx]

    pos_tr = int(np.sum(yt))
    pos_te = int(np.sum(ytest))

    if pos_tr == 0 or pos_tr == len(yt) or pos_te == 0:
        print(f"{elem:<10} | {'N/A (No data)':<15} | {'N/A':<15} | {pos_tr}")
        continue

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(F_train, yt)
    rf_pred = rf.predict(F_test)
    rf_f1 = f1_score(ytest, rf_pred, zero_division=0)

    pls = PLSRegression(n_components=min(5, F_train.shape[0]-1))
    pls.fit(F_train, yt)
    pls_pred_raw = pls.predict(F_test).flatten()
    pls_pred = (pls_pred_raw > 0.5).astype(int)
    pls_f1 = f1_score(ytest, pls_pred, zero_division=0)

    print(f"{elem:<10} | {rf_f1:.3f}           | {pls_f1:.3f}           | {pos_tr}")

print("=" * 65)
