"""Configuration for the Stanford SECL pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data directories ──────────────────────────────────────────────────────────
# Canonical layout:  data/stanford/  data/escl/  data/warwick/
DATA_DIR          = PROJECT_ROOT / "data"
STANFORD_DATA_DIR = DATA_DIR / "stanford"
ESCL_DATA_DIR     = DATA_DIR / "escl"
WARWICK_DATA_DIR  = DATA_DIR / "warwick"

# Legacy alias (backward compat — raw/diagnostic_test symlinked in data/stanford)
RAW_DIAG_DIR = STANFORD_DATA_DIR / "diagnostic_test"

NOMINAL_CAPACITY = 4.85  # Ah  (Stanford SECL LCO 18650)

# Nominal capacities per dataset (Ah) — single source of truth for all phases.
# Used to convert dimensionless MAE → MAE% for cross-dataset comparability.
NOMINAL_CAPACITIES = {
    "stanford": 4.85,   # LCO 18650
    "escl":     2.5,    # Samsung 25R INR18650
    "warwick":  5.0,    # NMC811 INR21700-50E
}

# Cells with valid EIS data
CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]

# QRC
N_QUBITS = 6
N_PCA = 6
DEPTH_RANGE = [1, 2, 3, 4]
RANDOM_STATE = 42
# Finer log-spaced Ridge alpha grid — reviewer: "consider a finer log-spaced
# grid or Bayesian optimization for the readout."
RIDGE_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# Noisy simulation — IBM Marrakesh (Heron r2, 156 qubits)
# Median values extracted from FakeMarrakesh (qiskit-ibm-runtime)
# Kept for backward compat with Phase 6 submission scripts
MARRAKESH_NOISE = {
    "single_qubit_error": 2.3e-4,   # SX median 0.023%
    "two_qubit_error": 3.3e-3,      # CZ median 0.33%
    "measurement_error": 0.0095,    # Readout median 0.95%
    "t1_us": 197.36,                # T1 median (us)
    "t2_us": 118.43,                # T2 median (us)
}

# Noisy simulation — IBM Fez (Heron r2, 156 qubits)
# Calibration values read directly from IBM Quantum Platform dashboard (2026-03-13)
# QPU version 1.3.37  |  CLOPS 320K
BACKEND_NAME = "ibm_fez"
FEZ_NOISE = {
    "single_qubit_error": 2.643e-4,  # SX median 0.02643%
    "two_qubit_error": 2.688e-3,     # CZ median 0.2688%
    "measurement_error": 1.54e-2,    # Readout assignment error median 1.54%
    "t1_us": 142.56,                 # T1 median (us)
    "t2_us": 100.11,                 # T2 median (us)
}
# Alias used by stage2_noisy.py and run_phase_4.py
HERON_R2_NOISE = FEZ_NOISE
SHOTS_LIST = [1024, 4096, 8192]

# Classical models
MODEL_NAMES = ["svr", "xgboost", "linear_pc1", "rff", "esn", "mlp"]


def get_result_paths(phase: int):
    """Create and return (data_dir, plot_dir) for a phase."""
    base = PROJECT_ROOT / "result" / f"phase_{phase}"
    data_dir = base / "data"
    plot_dir = base / "plot"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir

