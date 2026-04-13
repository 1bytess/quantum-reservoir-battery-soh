"""Configuration for Phase 4-Lab: QRC on EIS features (noiseless + noisy).

Two stages:
  Stage 1 — Noiseless (statevector): exact quantum simulation
  Stage 2 — Noisy (IBM Heron R2 digital twin): shot-based + noise model
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── QRC architecture ──────────────────────────────────────────────────────
N_QUBITS: int = 6
DEPTH_RANGE: List[int] = [1, 2, 3, 4]
DEFAULT_DEPTH: int = 2
USE_ZZ_CORRELATORS: bool = True
ENCODING_METHOD: str = "arctan"
CLAMP_RANGE: tuple = (-3.0, 3.0)

# ── Enhanced circuit flags ───────────────────────────────────────────────
REUPLOAD: bool = False           # Data re-uploading after each CZ ring
DUAL_AXIS: bool = False          # RY+RZ dual-axis encoding
OBSERVABLE_SET: str = "Z"        # "Z" (6+15=21) or "XYZ" (18+135=153)

# ── Ridge readout ─────────────────────────────────────────────────────────
# Finer log-spaced grid (reviewer §5.2.3): "consider a finer log-spaced grid
# or Bayesian optimization for the readout."
RIDGE_ALPHAS: List[float] = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# ── Noiseless backend ─────────────────────────────────────────────────────
BACKEND_TYPE: str = "statevector"

# ── Noisy backend (IBM Fez digital twin) ──────────────────────────────────
BACKEND_NAME: str = "ibm_fez"
SHOTS_LIST: List[int] = [1024, 4096, 8192]
DEFAULT_SHOTS: int = 8192

# IBM Fez (Heron R2) noise parameters — read from IBM Quantum Platform dashboard
# Calibration date: 2026-03-13  |  QPU version 1.3.37  |  CLOPS 320K
HERON_R2_NOISE = {
    "single_qubit_error": 2.643e-4,  # SX median (0.02643%)
    "two_qubit_error": 2.688e-3,     # CZ median (0.2688%)
    "measurement_error": 1.54e-2,    # Readout assignment error median (1.54%)
    "t1_us": 142.56,                 # T1 median (us)
    "t2_us": 100.11,                 # T2 median (us)
}

# ── Cell config (same as Phase 3-Lab) ─────────────────────────────────────
CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]
TEMP_GROUPS = {
    "25C": ["W3", "W8", "W9", "W10"],
    "unknown": ["V4", "V5"],
}

RANDOM_STATE: int = 42


@dataclass
class Phase4LabPaths:
    """Directory layout for Phase 4-Lab."""

    results_dir: Path = PROJECT_ROOT / "result" / "phase_4"

    @property
    def data_dir(self) -> Path:
        return self.results_dir / "data"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plot"

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def phase2_lab_dir(self) -> Path:
        return PROJECT_ROOT / "result" / "phase_2" / "data"

    @property
    def phase3_lab_dir(self) -> Path:
        return PROJECT_ROOT / "result" / "phase_3" / "data"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
