import os
import torch

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
NASA_DATA_DIR = os.path.join(BASE_DIR, "NASA data")
LOCAL_DATA_DIR_B = os.path.join(BASE_DIR, "2026.07.16wl_corr")
LOCAL_DATA_DIR_A = os.path.join(BASE_DIR, "20260422data")

# Saved models and output directories
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target Elements (full 15-element set enabled by 0-800nm coverage)
ALL_ELEMENTS = ['Fe', 'Ca', 'Si', 'Mg', 'Al', 'Ti', 'Zn', 'Mn', 'K', 'Na', 'O', 'S', 'P', 'C', 'Cl']
GEOCHEMICAL_ELEMENTS = ['Si', 'Ti', 'Al', 'Fe', 'Mn', 'Mg', 'Ca', 'Na', 'K', 'O']

# Chemical Stoichiometry Mapping (Oxide -> Element weight fractions)
OXIDE_ELEM = {
    'SiO2':  {'Si': 0.4674, 'O': 0.5326},
    'TiO2':  {'Ti': 0.5995, 'O': 0.4005},
    'Al2O3': {'Al': 0.5293, 'O': 0.4707},
    'FeOT':  {'Fe': 0.7773, 'O': 0.2227},
    'MnO':   {'Mn': 0.7744, 'O': 0.2256},
    'MgO':   {'Mg': 0.6031, 'O': 0.3969},
    'CaO':   {'Ca': 0.7147, 'O': 0.2853},
    'Na2O':  {'Na': 0.7419, 'O': 0.2581},
    'K2O':   {'K':  0.8302, 'O': 0.1698},
}
OXIDES = list(OXIDE_ELEM.keys())

# Default Hyperparameters
DEFAULT_HYPERPARAMS = {
    "wavelet_level": 1,
    "wavelet_name": "db4",
    "element_threshold_wt%": 1.0,
    "ensemble_members": 10,
    "learning_rate": 1e-3,
    "fine_tune_lr_backbone": 1e-5,
    "fine_tune_lr_head": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "augmentation_factor": 10,
    "noise_std_ratio": 0.05,
    "seed": 42
}
