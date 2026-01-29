# config.py
"""Centralized configuration and hyperparameter management."""
from pathlib import Path
import numpy as np

# --- System & Paths ---
BASE_OUTPUT_DIR = Path("Outputs/Reports/GridSearch_Full_Report_25")
INPUT_DIR = Path("Inputs/")
INPUT_VIDEO = INPUT_DIR / "video1_cropped_adj.tif"
NEURON_GT_CSV = INPUT_DIR / "Neuron and Blood Vessel labeled_CL(Video1_Neuron)_cropped_adj.csv"
VESSEL_GT_CSV = None  # Set to a path if available

# --- Experiment Settings ---
TOP_K_MODELS_TO_REPORT = 16
TRUNCATION_FPPI_LIMIT = 30.0
NUM_WORKERS = 6
RANDOM_SEED = 42

# --- Grid Search Hyperparameters ---
Z_SCORE_SWEEP = np.linspace(-2, 4, 256)
RESTRICTIVE_Z_SWEEP = np.linspace(-2, 4, 512)

# Which pre-processing families to include in the grid search
ENABLED_FILTER_FAMILIES = ["gamma", "kalman_mcc"]   # or ["gamma"] or ["kalman_mcc"]


# Example from your code; expand as needed for full grid search
# GAMMA_PARAMS = {
#     'filter_type': ['gamma'],
#     't_decay': [2**(-5) * i for i in range(33)],
#     's_decay': [2**(-4) * i for i in range(9)],
# }
ARCHITECTURES = ['single', 'two_stage']   # allow both

# Stage-specific config holders (Gamma-only for ST2 for now)
# ST1_PARAMS = {
#     'filter_type': ['gamma'],
#     't_decay': [2**(-3) * i for i in range(16)],  # example subset
#     's_decay': [2**(-2) * i for i in range(2)],
# }
# ST2_PARAMS = {
#     'filter_type': ['gamma'],                    # Gamma-only (your request)
#     't_decay': [2**(-3) * i for i in range(16)],  # small subset
#     's_decay': [2**(-2) * i for i in range(2)],
# }

ST1_PARAMS = {
    'filter_type': ['gamma'],
    't_decay': [2**(-4) * i for i in range(8)],  # example subset
    's_decay': [2**(-3) * i for i in range(2)],
}
ST2_PARAMS = {
    'filter_type': ['gamma'],                    # Gamma-only (your request)
    't_decay': [2**(-4) * i for i in range(8)],  # small subset
    's_decay': [2**(-3) * i for i in range(2)],
}

# Put this right after ST1_PARAMS / ST2_PARAMS
GAMMA_PARAMS = ST1_PARAMS  # back-compat so PREPROCESSOR_GRIDS['gamma'] works

# --- Kalman–MCC grid (as an alternative pre-processor) ---
# KALMAN_PARAMS = {
#     'filter_type': ['kalman_mcc'],
#     'sigma': [2**(-4) * i for i in range(17)],  # tune as needed
#     'mu':    [2**(-4) * i for i in range(17)],
# }

KALMAN_PARAMS = {
    'filter_type': ['kalman_mcc'],
    'sigma': [0.5],  # tune as needed
    'mu':    [0.18],
}

PREPROCESSOR_GRIDS = {
    "gamma": GAMMA_PARAMS,
    "kalman_mcc": KALMAN_PARAMS,
}

# --- Confidence Intervals ---
CI_METHOD = "parametric"   # "parametric" or "bootstrap"
CI_ALPHA  = 0.05           # 95% CI
BOOTSTRAP_N = 200          # used only if CI_METHOD == "bootstrap"

# --- Kalman–MCC (for background-quantification figures only) ---
KALMAN_MCC = {
    "enabled": True,             # set False to skip MCC figures
    "sigma": 1.0,               # correntropy kernel width (normalized units)
    "mu": 0.18,                   # step size for the IRLS update
    "max_frames_for_plots": 100, # cap for speed on long videos
    "num_trace_pixels": 3,        # how many pixels to plot traces for
    "write_tiffs": True,  # write background TIFFs
}

# --- Confidence Intervals ---
CI_METHOD = "parametric"   # "parametric" or "bootstrap"
CI_ALPHA  = 0.05           # 95% CI
BOOTSTRAP_N = 200          # used only if CI_METHOD == "bootstrap"


# --- Shared (non-CFAR) settings ---
SHARED_PARAMS = {
    'distance_tolerance': [6],
    'neighborhood_config': [(3, 1)],
}

# --- Stage-specific CFAR grids (can differ in eps, kernel, gamma radii, etc.) ---
_CFAR_BASE = {
    'type': 'local-separate-gamma',
    'gamma_radial_params': (9, 35),
    'kernel_size': (23, 23),
}

# Example sweeps — tune to taste:
CFAR1_PARAMS = [{**_CFAR_BASE, 'eps': e} for e in [64]]
CFAR2_PARAMS = [{**_CFAR_BASE, 'eps': e} for e in [64, 96, 128]]
