import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_nasa_calibration_data, physical_augment_data, SpectralDataset

device = config.DEVICE
print(f"=== Phase 2: NASA Base ECNN Pre-training ===")
print(f"Using compute device: {device}")

class ECNNBaseModel(nn.Module):
    def __init__(self, L, C, ckw=25, stride=8, nck=32):
        super().__init__()
        self.conv = nn.Conv1d(1, nck, ckw, stride, ckw // 2)
        self.relu = nn.ReLU()
        with torch.no_grad():
            f = self.conv(torch.zeros(1, 1, L)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, C)
        )

    def forward(self, x):
        return self.fc(self.relu(self.conv(x)))

def pretrain_nasa_base():
    X_cal, y_cal, std_names, wvl = load_nasa_calibration_data()
    print(f"Loaded {len(X_cal)} Earth calibration spectra ({X_cal.shape[1]} channels).")

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

    Xtr, ytr = physical_augment_data(Xtr, ytr, factor=10)

    print(f"Train split: {len(Xtr)} spectra from {len(tr_stds)} standards")
    print(f"Test split:  {len(Xte)} spectra from {len(te_stds)} unseen standards")

    m_cal, s_cal = Xtr.mean(0), Xtr.std(0)
    s_cal[s_cal < 1e-8] = 1.0
    Xtr_s = (Xtr - m_cal) / s_cal
    Xte_s = (Xte - m_cal) / s_cal

    active_indices = [i for i, p in enumerate(ytr.sum(0) > 0) if p]
    active_elements = [config.GEOCHEMICAL_ELEMENTS[i] for i in active_indices]
    n_active = len(active_elements)

    print(f"Active geochemical elements ({n_active}): {active_elements}")

    n_members = config.DEFAULT_HYPERPARAMS["ensemble_members"]
    epochs = config.DEFAULT_HYPERPARAMS["epochs"]
    batch_size = config.DEFAULT_HYPERPARAMS["batch_size"]

    tr_ds = SpectralDataset(Xtr_s, ytr[:, active_indices])
    te_ds = SpectralDataset(Xte_s, yte[:, active_indices])
    val_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model_states = []
    best_conv_weights = None
    best_overall_val_loss = float('inf')

    print(f"\nPre-training {n_members}-member NASA base ECNN...")
    for mi in range(n_members):
        m = ECNNBaseModel(Xtr_s.shape[1], n_active).to(device)
        opt = optim.Adam(m.parameters(), lr=config.DEFAULT_HYPERPARAMS["learning_rate"], weight_decay=1e-5)
        crit = nn.BCEWithLogitsLoss()
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = None

        for ep in range(1, epochs + 1):
            m.train()
            train_loss = 0.0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                l = crit(m(Xb), yb)
                l.backward()
                opt.step()
                train_loss += l.item()

            m.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xvb, yvb in val_loader:
                    Xvb, yvb = Xvb.to(device), yvb.to(device)
                    val_loss += crit(m(Xvb), yvb).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in m.state_dict().items()}
                if val_loss < best_overall_val_loss:
                    best_overall_val_loss = val_loss
                    best_conv_weights = {k: v.cpu() for k, v in m.conv.state_dict().items()}

            if ep % 10 == 0:
                print(f"  Member {mi+1} ep {ep}: train_loss={train_loss/len(tr_loader):.4f} val_loss={val_loss:.4f}")

        model_states.append(best_state)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Member {mi+1}/{n_members} pre-training complete. Best Val Loss = {best_val_loss:.4f}")

    checkpoint_path = os.path.join(config.SAVE_DIR, 'nasa_backbone_conv.pt')
    torch.save({
        'conv_state_dict': best_conv_weights,
        'model_states': model_states,
        'scaler_mean': m_cal,
        'scaler_std': s_cal,
        'active_elements': active_elements,
        'active_indices': active_indices,
        'wavelengths': wvl
    }, checkpoint_path)

    print(f"\nPhase 2 Completed! Exported base pre-trained Conv weights to: {checkpoint_path}")

if __name__ == '__main__':
    pretrain_nasa_base()
