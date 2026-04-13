"""Configuration for the ESCL Lab Data pipeline (Samsung 25R cells)."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
LAB_DATA_DIR = PROJECT_ROOT / "data" / "escl"
NOMINAL_CAPACITY_MAH = 2500.0  # mAh for Samsung INR18650-25R

# Cell IDs derived from filenames
# Each CA6 file = one experimental session; CA5 = temperature aging
# Note: CA6_210216 excluded — different column format, causes loading errors
CELL_IDS_LAB = [
    "CA6_210201",
    "CA6_210304",
    "CA6_210309",
    "CA6_210315",
    "CA6_210319",
    "CA6_210325",
    "CA6_210330",
    "CA6_210410",
    "CA6_210510",
    "CA6_210517",
    "CA5_AGING",
]

# Map cell IDs to filenames
CELL_FILE_MAP = {
    "CA6_210201": "25R_Cell_EIS_Cycle(210201)_CA6.txt",
    "CA6_210216": "25R_Cell_EIS_Cycle(210216)_CA6.txt",
    "CA6_210304": "25R_Cell_EIS_Cycle(210304)_CA6.txt",
    "CA6_210309": "25R_Cell_EIS_Cycle(210309)_CA6.txt",
    "CA6_210315": "25R_Cell_EIS_Cycle(210315)_CA6.txt",
    "CA6_210319": "25R_Cell_EIS_Cycle(210319)_CA6.txt",
    "CA6_210325": "25R_Cell_EIS_Cycle(210325)_CA6.txt",
    "CA6_210330": "25R_Cell_EIS_Cycle(210330)_CA6.txt",
    "CA6_210410": "25R_Cell_EIS_Cycle(210410)_CA6.txt",
    "CA6_210510": "25R_Cell_EIS_Cycle(210510)_CA6.txt",
    "CA6_210517": "25R_Cell_EIS_Cycle(210517)_CA6.txt",
    "CA5_AGING": "temperature_60_aging2_20200117_CA5.txt",
}

# EIS feature dimensions
N_FREQUENCIES_LAB = 36       # 36 log-spaced frequencies per sweep
N_EIS_RAW_LAB = 72           # 36 Re + 36 Im
N_PCA_LAB = 6                # same as Stanford (-> 6 qubits)

# SOC selection: high-SOC sweeps (Ecell > 3.8V)
HIGH_SOC_VOLTAGE_THRESHOLD = 3.8  # V

# QRC (same as Stanford pipeline)
N_QUBITS = 6
N_PCA = 6
DEPTH_RANGE = [1, 2, 3, 4]
RANDOM_STATE = 42
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Noisy simulation (reuse IBM Marrakesh params)
MARRAKESH_NOISE = {
    "single_qubit_error": 2.3e-4,
    "two_qubit_error": 3.3e-3,
    "measurement_error": 0.0095,
    "t1_us": 197.36,
    "t2_us": 118.43,
}
SHOTS_LIST = [1024, 4096, 8192]

# Classical models
MODEL_NAMES = ["svr", "xgboost", "linear_pc1", "rff", "esn"]


def get_lab_result_paths(phase: int):
    """Create and return (data_dir, plot_dir) for a lab phase."""
    base = PROJECT_ROOT / "result" / f"phase_{phase}"
    data_dir = base / "data"
    plot_dir = base / "plot"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir
