import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from wavelet_preprocess import wavelet_denoise
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NIST_LINES = {
    "Fe": [259.94, 274.93, 275.57],
    "Ca": [315.89, 393.37, 396.85],
    "Si": [288.16],
    "Mg": [279.55, 280.27, 285.21],
    "Al": [308.22, 309.27],
    "Ti": [334.94, 336.12, 337.28],
    "Zn": [330.26, 472.20, 481.05],
    "Mn": [257.61, 259.37, 260.57],
    "K":  [766.49, 769.90],
    "Na": [589.00, 589.59],
    "O":  [777.19, 777.42, 777.54],
    "S":  [545.38, 564.00],
    "P":  [253.56],
    "C":  [247.86],
    "Cl": [837.60],
}
ELEMENT_LIST = list(NIST_LINES.keys())
NUM_ELEMENTS = len(ELEMENT_LIST)
NUM_SAMPLES = 5000
WAVELET_LEVEL = 1
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chemcam_cache.npz")

def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X)); split = int(len(X) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def compute_classification_report(y_true, y_pred, names):
    lines = []
    for i, n in enumerate(names):
        tp = ((y_true[:,i]==1)&(y_pred[:,i]==1)).sum()
        fp = ((y_true[:,i]==0)&(y_pred[:,i]==1)).sum()
        fn = ((y_true[:,i]==1)&(y_pred[:,i]==0)).sum()
        s = int((y_true[:,i]==1).sum())
        p = tp/(tp+fp) if (tp+fp)>0 else 0.
        r = tp/(tp+fn) if (tp+fn)>0 else 0.
        f = 2*p*r/(p+r) if (p+r)>0 else 0.
        lines.append((n,p,r,f,s))
    print(f"{'':>10}  {'precision':>9} {'recall':>6} {'f1-score':>8} {'support':>7}")
    print("-"*50)
    for n,p,r,f,s in lines: print(f"{n:>10}  {p:>9.2f} {r:>6.2f} {f:>8.2f} {s:>7}")
    return {n:{"precision":p,"recall":r,"f1-score":f,"support":s} for n,p,r,f,s in lines}

def load_nasa_chemcam(num_samples=NUM_SAMPLES):
    if os.path.exists(CACHE_PATH):
        print("Loading cached ChemCam data from local disk...")
        cached = np.load(CACHE_PATH)
        return cached["X"], cached["y"], cached["wavelengths"]
    print("Downloading ChemCam data from HuggingFace (one-time)...")
    dataset = load_dataset("jfang/ChemCam-LIBS-2512", "stat", split="train", streaming=True)
    sample_row = next(iter(dataset))
    wave_cols = [c for c in sample_row.keys() if c.startswith("wave@")]
    wave_cols.sort(key=lambda x: float(x.split("@")[1]))
    wavelengths = np.array([float(c.split("@")[1]) for c in wave_cols])
    X, y = [], []
    it = iter(dataset)
    for i in range(num_samples):
        row = next(it)
        spectrum = np.array([row[c] for c in wave_cols])
        X.append(spectrum)
        label = np.zeros(NUM_ELEMENTS)
        mu, s = spectrum.mean(), spectrum.std()
        thresh = mu + 3 * s
        for idx, elem in enumerate(ELEMENT_LIST):
            for tw in NIST_LINES[elem]:
                if spectrum[np.argmin(np.abs(wavelengths - tw))] > thresh:
                    label[idx] = 1; break
        y.append(label)
        if (i+1) % 1000 == 0: print(f"  Loaded {i+1}/{num_samples}")
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    np.savez_compressed(CACHE_PATH, X=X, y=y, wavelengths=wavelengths)
    print(f"Saved to local cache: {CACHE_PATH}")
    return X, y, wavelengths

class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ── EC-CNN: Lightweight sub-model (no pooling, stride-based) ──
class ECNNSubModel(nn.Module):
    """Single CNN sub-model: lightweight with stride downsampling (Yu & Yao 2023)."""
    def __init__(self, L, C, ckw=25, stride=8, nck=32):
        super().__init__()
        self.conv = nn.Conv1d(1, nck, kernel_size=ckw, stride=stride, padding=ckw//2)
        self.relu = nn.ReLU()
        # Infer flattened dim
        with torch.no_grad():
            n_after_conv = self.conv(torch.zeros(1, 1, L)).shape[2]
            flat_dim = nck * n_after_conv
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, C),
        )

    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.fc(x)


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model')


def save_checkpoint(members, scaler_mean, scaler_std, active_elements, path=MODEL_DIR):
    os.makedirs(path, exist_ok=True)
    for i, m in enumerate(members):
        torch.save(m.state_dict(), os.path.join(path, f'member_{i}.pt'))
    np.savez(os.path.join(path, 'preprocess_params.npz'),
             mean=scaler_mean, std=scaler_std, active_elements=np.array(active_elements, dtype=object))
    import json
    with open(os.path.join(path, 'active_elements.json'), 'w') as f:
        json.dump(active_elements, f)
    print(f'Model saved to {path}')


def load_checkpoint(L, C, path=MODEL_DIR):
    params = np.load(os.path.join(path, 'preprocess_params.npz'), allow_pickle=True)
    active = params['active_elements'].tolist()
    members = []
    for i in range(10):
        model = ECNNSubModel(L, C).to(device)
        model.load_state_dict(torch.load(os.path.join(path, f'member_{i}.pt'), map_location=device))
        members.append(model)
    return members, params['mean'], params['std'], active


def train_ecnn_ensemble(train_dataset, test_dataset, num_classes, n_members=10, epochs=10):
    """Train ECNN ensemble via Bagging (Yu & Yao 2023 method)."""
    n_train = len(train_dataset)
    members = []
    history = {"loss":[], "val_loss":[], "acc":[], "val_acc":[]}
    best_val_acc = 0.0

    for m in range(n_members):
        # Bootstrap sample: random indices with replacement
        rng = np.random.RandomState(42 + m)
        boot_idx = rng.choice(n_train, n_train, replace=True)
        boot_dataset = Subset(train_dataset, boot_idx.tolist())
        boot_loader = DataLoader(boot_dataset, batch_size=64, shuffle=True)

        val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        input_len = train_dataset[0][0].shape[1]
        model = ECNNSubModel(input_len, num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        member_hist = {"loss":[], "val_loss":[], "acc":[], "val_acc":[]}

        for epoch in range(1, epochs + 1):
            model.train()
            tl, tc, tt = 0., 0, 0
            for Xb, yb in boot_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward(); optimizer.step()
                tl += loss.item() * Xb.size(0)
                preds = (torch.sigmoid(model(Xb)) > .5).float()
                tc += (preds == yb).sum().item(); tt += yb.numel()
            tl /= len(boot_dataset); ta = tc / tt

            model.eval()
            vl, vc, vt = 0., 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    loss = criterion(model(Xb), yb)
                    vl += loss.item() * Xb.size(0)
                    preds = (torch.sigmoid(model(Xb)) > .5).float()
                    vc += (preds == yb).sum().item(); vt += yb.numel()
            vl /= len(test_dataset); va = vc / vt
            member_hist["loss"].append(tl); member_hist["val_loss"].append(vl)
            member_hist["acc"].append(ta); member_hist["val_acc"].append(va)

        # Track best aggregated validation accuracy
        if member_hist["val_acc"][-1] > best_val_acc:
            best_val_acc = member_hist["val_acc"][-1]

        members.append(model)

        # Aggregate histories from all members
        for key in history:
            history[key].append(member_hist[key][-1])

        mi = "; ".join(f"{k}: {v[-1]:.4f}" for k, v in member_hist.items())
        print(f"  Member {m+1:2d}/{n_members} | {mi}")

    return members, history

if __name__ == "__main__":
    X_raw, y_multi, wavelengths = load_nasa_chemcam(NUM_SAMPLES)
    print(f"Loaded {len(X_raw)} spectra, {X_raw.shape[1]} channels each.\n")

    print(f"Wavelet denoising (db4, level={WAVELET_LEVEL})...")
    X_raw = wavelet_denoise(X_raw, wavelet="db4", level=WAVELET_LEVEL, mode="soft")
    print("Done.\n")

    counts = y_multi.sum(axis=0)
    print("[NIST auto-label] Element occurrence:")
    for e, c in zip(ELEMENT_LIST, counts): print(f"  {e}: {int(c)} times")

    present = counts > 0
    active_elements = [e for e, p in zip(ELEMENT_LIST, present) if p]
    active_indices = [i for i, p in enumerate(present) if p]
    print(f"\nActive elements: {active_elements}")

    y_filt = y_multi[:, active_indices]
    sample_mask = y_filt.sum(axis=1) > 0
    X_filt, y_filt = X_raw[sample_mask], y_filt[sample_mask]
    print(f"Samples with at least one active element: {len(X_filt)}")

    X_train, X_test, y_train, y_test = manual_train_test_split(X_filt, y_filt)
    m, s = X_train.mean(axis=0), X_train.std(axis=0)
    s[s < 1e-8] = 1.0
    X_train_s = (X_train - m) / s
    X_test_s = (X_test - m) / s

    train_ds = SpectraDataset(X_train_s, y_train)
    test_ds = SpectraDataset(X_test_s, y_test)

    num_active = len(active_elements)
    print(f"\nTraining ECNN ensemble ({num_active} output classes)...")
    members, hist = train_ecnn_ensemble(
        train_dataset=train_ds,
        test_dataset=test_ds,
        num_classes=num_active,
        n_members=10,
        epochs=10,
    )

    # Save model for local data evaluation
    save_checkpoint(members, m, s, active_elements)

    # ── Evaluate ensemble ──
    val_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_probs = []
    all_true = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            # Average predictions across all members
            ensemble_probs = torch.zeros(len(Xb), num_active, device=device)
            for m_i, model in enumerate(members):
                model.eval()
                ensemble_probs += torch.sigmoid(model(Xb))
            ensemble_probs /= len(members)
            all_probs.append(ensemble_probs.cpu().numpy())
            all_true.append(yb.numpy())

    yp = np.concatenate(all_probs); yt = np.concatenate(all_true)
    yb = (yp > .5).astype(int)

    print("\n================ ECNN Classification Report (Ensemble of 10) ================")
    report_dict = compute_classification_report(yt, yb, active_elements)

    # ── Learning curves ──
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["acc"], label=f"Train (mean of {len(members)})")
    plt.plot(hist["val_acc"], label=f"Val (mean of {len(members)})")
    plt.title("Ensemble Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="Train"); plt.plot(hist["val_loss"], label="Test")
    plt.title("Ensemble Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout(); plt.show()

    # ── F1 bar ──
    plt.rcParams["figure.figsize"] = (11, 5.5)
    f1s = [report_dict[e]["f1-score"] for e in active_elements]
    fig, ax = plt.subplots()
    bars = ax.bar(active_elements, f1s, color="#2ECC71", edgecolor="black", alpha=.85, width=.6)
    ax.axhline(y=.8, color="darkgray", ls="--", lw=1.5, label="Expected (0.80)")
    ax.set_ylabel("F1-score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elements", fontsize=12, fontweight="bold")
    ax.set_title("ECNN Ensemble (10 members, Bagging + Stride-based CNN)")
    ax.set_ylim(0, 1.1); ax.grid(axis="y", ls=":", alpha=.6); ax.legend(loc="upper right", fontsize=11)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig("cnn_ecnn_chemcam.png", dpi=300)
    print(f"\nChart saved as 'cnn_ecnn_chemcam.png'")
    plt.show()