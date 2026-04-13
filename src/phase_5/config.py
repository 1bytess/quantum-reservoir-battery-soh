"""Configuration for Phase 5-Lab: Ablation study."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_5_ROOT = PROJECT_ROOT / "result" / "phase_5"

# ── Noise ablation config ─────────────────────────────────────────────────
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
CLASSICAL_MODELS = ["svr", "xgboost", "rff", "esn"]

CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]
TEMP_GROUPS = {
    "25C": ["W3", "W8", "W9", "W10"],
    "unknown": ["V4", "V5"],
}

RANDOM_STATE = 42
N_NOISE_REPEATS = 5  # Average over multiple noise injections

# ── Stochastic resonance sweep (quantum noise channels) ──────────────
SR_NOISE_RATES = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
SR_CHANNELS = ["depolarizing", "amplitude_damping", "phase_damping"]
SR_SHOTS = 8192
SR_REPEATS = 3


@dataclass(frozen=True)
class Phase5LabPaths:
    """Directory layout for Phase 5-Lab."""

    stage_name: str | None = None

    @property
    def results_dir(self) -> Path:
        if self.stage_name is None:
            return PHASE_5_ROOT
        return PHASE_5_ROOT / self.stage_name

    @property
    def data_dir(self) -> Path:
        return self.results_dir / "data"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plot"

    @property
    def phase2_lab_dir(self) -> Path:
        return PROJECT_ROOT / "result" / "phase_2" / "data"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


def get_stage_paths(stage_name: str) -> tuple[Path, Path]:
    """Create and return ``(data_dir, plot_dir)`` for a Phase 5 stage."""
    paths = Phase5LabPaths(stage_name)
    paths.ensure_dirs()
    return paths.data_dir, paths.plots_dir
