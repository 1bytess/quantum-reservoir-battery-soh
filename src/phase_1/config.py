"""Configuration for Phase 1-Lab: Lab data loading and exploration."""

from dataclasses import dataclass
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Cell identifiers ──────────────────────────────────────────────────────
CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]

TEMP_GROUPS = {
    "25C": ["W3", "W8", "W9", "W10"],
    "unknown": ["V4", "V5"],
}

# ── Battery spec ──────────────────────────────────────────────────────────
NOMINAL_CAPACITY_MAH = 4850.0
BATTERY_TYPE = "LG INR21700-M50T"

# ── EIS constants ─────────────────────────────────────────────────────────
N_FREQUENCIES = 19  # 19 log-spaced points per spectrum
FREQ_MIN_HZ = 0.01  # ~10 mHz (approximate)
FREQ_MAX_HZ = 10_000.0  # ~10 kHz (approximate)

# ── Plot settings ─────────────────────────────────────────────────────────
CELL_COLORS = {
    "W3": "#1f77b4",   # blue
    "W8": "#ff7f0e",   # orange
    "W9": "#2ca02c",   # green
    "W10": "#d62728",  # red
    "V4": "#9467bd",   # purple
    "V5": "#8c564b",   # brown
}
FIGSIZE_SINGLE = (90 / 25.4, 70 / 25.4)  # single column (90mm)
FIGSIZE_WIDE = (180 / 25.4, 70 / 25.4)   # double column (180mm)
DPI = 300
RANDOM_STATE = 42


@dataclass
class Phase1LabPaths:
    """Directory layout for Phase 1-Lab."""

    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "result" / "phase_1"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plot"

    @property
    def extracted_data_dir(self) -> Path:
        return self.results_dir / "data"

    def ensure_dirs(self) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_data_dir.mkdir(parents=True, exist_ok=True)
