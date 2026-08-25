import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import (
    load_nasa_calibration_data,
    load_mars_ccs_spectrum,
    load_moc_labels,
    oxide_to_element_label_single,
    physical_augment_data
)
from model_dual_stream import NISTFeatureExtractor, DualStreamECNN

device = config.DEVICE
print(f"=== Phase 3: Dual-Stream ECNN Training & Mars Multi-Target Eval ===")
print(f"Using compute device: {device}")

class DualStreamDataset(Dataset):
    def __init__(self, X_spec, X_phys, y):
        self.X_spec = torch.from_numpy(X_spec).float().unsqueeze(1)
        self.X_phys = torch.from_numpy(X_phys).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X_spec)

    def __getitem__(self, i):
        return self.X_spec[i], self.X_phys[i], self.y[i]

def load_earth_to_mars_corr(data_dir=config.NASA_DATA_DIR, n_channels=6144):
    corr_path = os.path.join(data_dir, 'earth_2_mars_corr.csv')
    if os.path.exists(corr_path):
        try:
            df = pd.read_csv(corr_path)
            col = df.columns[-1]
            corr_vector = df[col].values.astype(np.float32)
            if len(corr_vector) == n_channels:
                print(f"Loaded Earth-to-Mars response correction factor ({n_channels} channels).")
                return corr_vector
        except Exception as e:
            print(f"Warning loading corr factor: {e}")

    print("Response correction factor not applied (using identity factor = 1.0).")
    return np.ones(n_channels, dtype=np.float32)

def train_dual_stream_ensemble(Xtr_s, Xtr_phys_s, ytr, Xte_s, Xte_phys_s, yte, active_indices, pretrained_ckpt, n_members=10, epochs=50):
    n_active = len(active_indices)
    input_channels = Xtr_s.shape[1]
    num_phys_features = Xtr_phys_s.shape[1]

    tr_ds = DualStreamDataset(Xtr_s, Xtr_phys_s, ytr[:, active_indices])
    te_ds = DualStreamDataset(Xte_s, Xte_phys_s, yte[:, active_indices])

    val_loader = DataLoader(te_ds, batch_size=config.DEFAULT_HYPERPARAMS["batch_size"], shuffle=False)
    model_states = []

    for mi in range(n_members):
        model = DualStreamECNN(
            input_channels=input_channels,
            num_phys_features=num_phys_features,
            num_classes=n_active,
            pretrained_checkpoint=pretrained_ckpt
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=config.DEFAULT_HYPERPARAMS["learning_rate"], weight_decay=1e-5)
        crit = nn.BCEWithLogitsLoss()
        tr_loader = DataLoader(tr_ds, batch_size=config.DEFAULT_HYPERPARAMS["batch_size"], shuffle=True)

        best_val_loss = float('inf')
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for Xb_spec, Xb_phys, yb in tr_loader:
                Xb_spec, Xb_phys, yb = Xb_spec.to(device), Xb_phys.to(device), yb.to(device)
                opt.zero_grad()
                logits, _ = model(Xb_spec, Xb_phys)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xvb_spec, Xvb_phys, yvb in val_loader:
                    Xvb_spec, Xvb_phys, yvb = Xvb_spec.to(device), Xvb_phys.to(device), yvb.to(device)
                    logits_v, _ = model(Xvb_spec, Xvb_phys)
                    val_loss += crit(logits_v, yvb).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            if ep % 10 == 0:
                print('  Member {} ep {}: train_loss={:.4f} val_loss={:.4f}'.format(mi + 1, ep, train_loss / len(tr_loader), val_loss))

        model_states.append(best_state)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('  Member {}/{} done, best_val_loss={:.4f}'.format(mi + 1, n_members, best_val_loss))

    return model_states

def ensemble_predict_dual_stream(model_states, X_spec, X_phys, input_channels, n_active, pretrained_ckpt, threshold=0.5):
    n_phys = X_phys.shape[1]
    Xt_spec = torch.from_numpy(X_spec).float().unsqueeze(1).to(device)
    Xt_phys = torch.from_numpy(X_phys).float().to(device)

    out = torch.zeros(len(Xt_spec), n_active, device=device)
    all_gates = []

    with torch.no_grad():
        for ms in model_states:
            model = DualStreamECNN(input_channels, n_phys, n_active, pretrained_checkpoint=pretrained_ckpt).to(device)
            model.load_state_dict(ms)
            model.eval()
            logits, gates = model(Xt_spec, Xt_phys)
            out += torch.sigmoid(logits)
            all_gates.append(gates.cpu().numpy())
            del model

    out /= len(model_states)
    probs = out.cpu().numpy()
    preds = (probs > threshold).astype(int)
    avg_gates = np.mean(all_gates, axis=0)
    return probs, preds, avg_gates

def report(yt, yb, names):
    lines = []
    for i, n in enumerate(names):
        tp = ((yt[:, i] == 1) & (yb[:, i] == 1)).sum()
        fp = ((yt[:, i] == 0) & (yb[:, i] == 1)).sum()
        fn = ((yt[:, i] == 1) & (yb[:, i] == 0)).sum()
        s = int((yt[:, i] == 1).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append((n, p, r, f, s))

    print('  Element    precision  recall  f1-score  support')
    print('-' * 50)
    for n, p, r, f, s in lines:
        print('  {:>6}    {:>7.2f}  {:>6.2f}  {:>7.2f}  {:>5}'.format(n, p, r, f, s))
    return {n: {'p': p, 'r': r, 'f': f, 's': s} for n, p, r, f, s in lines}

if __name__ == '__main__':
    pretrained_ckpt = os.path.join(config.SAVE_DIR, 'nasa_backbone_conv.pt')
    if not os.path.exists(pretrained_ckpt):
        pretrained_ckpt = os.path.join(config.SAVE_DIR, 'earth_ecnn_checkpoint.pt')

    print(f"Loading Phase 2 pre-trained Conv weights from: {pretrained_ckpt}")

    # 1. Load Calibration Spectra
    X_cal, y_cal, std_names, wvl = load_nasa_calibration_data()
    print(f"Loaded {len(X_cal)} Earth calibration spectra, {X_cal.shape[1]} channels.")

    # 2. Extract NIST Physical Features
    extractor = NISTFeatureExtractor(wvl, elements=config.GEOCHEMICAL_ELEMENTS)
    X_phys = extractor.transform(X_cal)

    # 3. Standard-level Cross Validation Split
    unique_stds = np.unique(std_names)
    rs = np.random.RandomState(config.DEFAULT_HYPERPARAMS["seed"])
    rs.shuffle(unique_stds)
    split_idx = max(int(len(unique_stds) * 0.8), 1)

    tr_stds = set(unique_stds[:split_idx])
    te_stds = set(unique_stds[split_idx:])

    tr_mask = np.array([s in tr_stds for s in std_names])
    te_mask = np.array([s in te_stds for s in std_names])

    Xtr, Xte = X_cal[tr_mask], X_cal[te_mask]
    ytr, yte = y_cal[tr_mask], y_cal[te_mask]
    Xtr_phys, Xte_phys = X_phys[tr_mask], X_phys[te_mask]

    # Physical Augmentation (Train set only)
    Xtr, ytr = physical_augment_data(Xtr, ytr, factor=10)
    Xtr_phys = extractor.transform(Xtr)

    print(f"Train split: {len(Xtr)} spectra from {len(tr_stds)} standards")
    print(f"Test split:  {len(Xte)} spectra from {len(te_stds)} unseen standards")

    # Standardize Spectral Channels
    m_cal, s_cal = Xtr.mean(0), Xtr.std(0)
    s_cal[s_cal < 1e-8] = 1.0
    Xtr_s = (Xtr - m_cal) / s_cal
    Xte_s = (Xte - m_cal) / s_cal

    # Standardize Physical Features with safety clipping
    m_phys, s_phys = Xtr_phys.mean(0), Xtr_phys.std(0)
    s_phys[s_phys < 1e-3] = 1.0  # Safety clip against near-zero std
    Xtr_phys_s = (Xtr_phys - m_phys) / s_phys
    Xte_phys_s = (Xte_phys - m_phys) / s_phys

    active_indices = [i for i, p in enumerate(ytr.sum(0) > 0) if p]
    active_elements = [config.GEOCHEMICAL_ELEMENTS[i] for i in active_indices]
    n_active = len(active_elements)

    print(f"\nTraining Dual-Stream ECNN with Pre-trained Conv Loaded (10 members, 50 epochs)...")
    model_states = train_dual_stream_ensemble(
        Xtr_s, Xtr_phys_s, ytr, Xte_s, Xte_phys_s, yte,
        active_indices, pretrained_ckpt, n_members=10, epochs=50
    )

    # Save Checkpoint
    checkpoint_path = os.path.join(config.SAVE_DIR, 'dual_stream_ecnn_checkpoint.pt')
    torch.save({
        'model_states': model_states,
        'scaler_mean': m_cal,
        'scaler_std': s_cal,
        'phys_mean': m_phys,
        'phys_std': s_phys,
        'active_elements': active_elements,
        'active_indices': active_indices
    }, checkpoint_path)
    print(f"\nSaved Dual-Stream ECNN checkpoint to: {checkpoint_path}")

    # Evaluate on Calibration Test Set (Standard threshold = 0.5)
    probs_te, preds_te, gates_te = ensemble_predict_dual_stream(
        model_states, Xte_s, Xte_phys_s, Xtr_s.shape[1], n_active, pretrained_ckpt, threshold=0.5
    )
    print("\n================ Calibration Test Set (Dual-Stream ECNN, th=0.5) ================")
    report(yte[:, active_indices], preds_te, active_elements)
    print(f"Average Gate Weights (Phys, NASA, Local): {gates_te.mean(axis=0)}")

    # Step 2: Multi-Target Mars Evaluation with Optical Response Correction
    print("\n" + "=" * 60)
    print("Step 2: Multi-Target Mars Zero-Shot Evaluation (th=0.5)")
    print("=" * 60)

    corr_factor = load_earth_to_mars_corr(config.NASA_DATA_DIR, Xtr_s.shape[1])

    mars_targets = [
        ('moc_0360_0449.csv', 'cl5_430599902ccs_f0140000ccam01373p3.csv', '430599902'),
        ('moc_0000_0089.csv', 'cl5_398645626ccs_f0030004ccam02013p3.csv', '398645626'),
        ('moc_0000_0089.csv', 'cl5_398646038ccs_f0030004ccam03013p3.csv', '398646038'),
        ('moc_0000_0089.csv', 'cl5_398737278ccs_f0030004ccam02014p3.csv', '398737278'),
    ]

    total_correct, total_elements = 0, 0

    for moc_fn, ccs_fn, sclk in mars_targets:
        moc_path = os.path.join(config.NASA_DATA_DIR, moc_fn)
        ccs_path = os.path.join(config.NASA_DATA_DIR, ccs_fn)

        if not os.path.exists(ccs_path):
            continue

        moc_oxides, ccs_name, target_name = load_moc_labels(moc_path, sclk)
        if moc_oxides is None:
            continue

        actual_ml = oxide_to_element_label_single(moc_oxides, elements=config.GEOCHEMICAL_ELEMENTS)
        actual_dict = {config.GEOCHEMICAL_ELEMENTS[i]: actual_ml[i] for i in range(len(config.GEOCHEMICAL_ELEMENTS))}

        wv_m, shots = load_mars_ccs_spectrum(ccs_path)
        shots_corrected = shots * corr_factor
        shots_s = (shots_corrected - m_cal) / s_cal

        shots_phys = extractor.transform(shots_corrected)
        shots_phys_s = (shots_phys - m_phys) / s_phys

        probs_m, preds_m, gates_m = ensemble_predict_dual_stream(
            model_states, shots_s, shots_phys_s, Xtr_s.shape[1], n_active, pretrained_ckpt, threshold=0.5
        )
        avg_probs_m = probs_m.mean(axis=0)

        matches = 0
        for i, elem in enumerate(active_elements):
            act = int(actual_dict.get(elem, 0))
            pred = int(avg_probs_m[i] > 0.5)
            if act == pred:
                matches += 1

        acc = matches / n_active * 100
        total_correct += matches
        total_elements += n_active
        print(f"Mars Target '{target_name}' (SCLK {sclk}): Accuracy = {matches}/{n_active} ({acc:.1f}%) | Avg Gate Weights = {gates_m.mean(axis=0)}")

    if total_elements > 0:
        overall_acc = total_correct / total_elements * 100
        print(f"\n================ Overall Mars Multi-Target Zero-Shot Accuracy (th=0.5) ================")
        print(f"Total Matched: {total_correct}/{total_elements} = {overall_acc:.1f}%")

    print("\n" + "=" * 60)
    print("Phase 3 Execution Completed Successfully.")
    print("=" * 60)
