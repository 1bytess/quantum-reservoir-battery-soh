"""Configuration for Phase 2-Lab: EIS feature engineering."""

from dataclasses import dataclass
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── EIS Feature Dimensions ───────────────────────────────────────────────
N_FREQUENCIES = 19          # frequencies per spectrum
N_EIS_RAW = 38              # Re + Im per frequency = 2 * 19
N_QRC_INPUT = 6             # target dim for QRC (6 qubits)

# ── Aggregation config ───────────────────────────────────────────────────
AGGREGATION_METHOD = "mean"  # block-level: average spectra within block

# ── Dimensionality reduction ─────────────────────────────────────────────
REDUCTION_METHOD = "pca"     # options: "pca", "random_projection"
EXPLAINED_VARIANCE_TARGET = 0.95  # for reporting

# ── Cell IDs (import from phase_1) ────────────────────────────────────
CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]
TEMP_GROUPS = {
    "25C": ["W3", "W8", "W9", "W10"],
    "unknown": ["V4", "V5"],
}

RANDOM_STATE = 42


@dataclass
class Phase2LabPaths:
    """Directory layout for Phase 2-Lab."""

    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "result" / "phase_2"

    @property
    def phase1_lab_dir(self) -> Path:
        return PROJECT_ROOT / "result" / "phase_1"

    @property
    def features_dir(self) -> Path:
        return self.results_dir / "data"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plot"

    def ensure_dirs(self) -> None:
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
