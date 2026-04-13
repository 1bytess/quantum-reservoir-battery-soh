"""Configuration for Phase 3-Lab: Classical baselines on EIS."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]
TEMP_GROUPS = {
    "25C": ["W3", "W8", "W9", "W10"],
    "unknown": ["V4", "V5"],
}

# Primary models (paper-reported). MLP demoted to supplementary — reviewer
# flag: a 1-hidden-layer MLP with alpha=0.01 is not a representative baseline.
# Add gp (Gaussian Process) and cnn1d as the two SOTA baselines requested
# by the reviewer (Zhang et al. 2020 NatComm approach and spectral CNN).
MODEL_NAMES = ["ridge", "svr", "xgboost", "linear_pc1", "rff", "esn", "gp", "cnn1d"]

# MLP kept separately so it can be run on demand without polluting primary table
MLP_MODEL_NAMES = ["mlp"]

RANDOM_STATE = 42

# Finer log-spaced Ridge alpha grid (reviewer: "consider finer log-spaced grid")
RIDGE_ALPHAS_FINE = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# RFF n_components matched to QRC reservoir dimensionality for fair kernel
# comparison (reviewer §7.3: "you need a classical random feature baseline that
# matches QRC dimensionality exactly — 21 random features from 6 inputs").
RFF_N_COMPONENTS_MATCHED = 21   # matches QRC Z+ZZ observable dimension
RFF_N_COMPONENTS_FULL   = 100   # kept for supplementary comparison

HYPERPARAM_GRIDS = {
    "ridge": {
        "ridge__alpha": RIDGE_ALPHAS_FINE,
    },
    "svr": {
        "svr__C": [0.1, 1.0, 10.0, 100.0],
        "svr__gamma": ["scale", "auto"],
        "svr__epsilon": [0.01, 0.05, 0.1],
    },
    "xgboost": {
        "xgb__n_estimators": [100, 200, 500],
        "xgb__max_depth": [3, 4, 6],
        "xgb__learning_rate": [0.05, 0.1],
    },
    "esn": {
        "esn__hidden_size": [16, 32, 64],
        "esn__ridge_alpha": [0.1, 1.0, 10.0],
    },
    "linear_pc1": {
        "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    # RFF: n_components fixed at matched dim (21); gamma tuned only
    "rff": {
        "rff__gamma": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        "ridge__alpha": RIDGE_ALPHAS_FINE,
    },
    # GP: kernel length-scale and noise tuned
    "gp": {
        "gp__alpha": [1e-6, 1e-4, 1e-2],
    },
    # 1D-CNN: no sklearn GridSearchCV (trained internally); empty grid placeholder
    "cnn1d": {},
    # MLP: tuned with early stopping and dropout (supplementary)
    "mlp": {
        "mlp__hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16)],
        "mlp__alpha": [0.001, 0.01, 0.1, 1.0],
        "mlp__learning_rate_init": [1e-3, 5e-4],
    },
}

FEW_SHOT_N = [1, 2]
TEMPORAL_TRAIN_FRAC = 0.7


@dataclass
class Phase3LabPaths:
    """Directory layout for Phase 3-Lab."""

    data_dir: Path = PROJECT_ROOT / "result" / "phase_3" / "data"
    results_dir: Path = PROJECT_ROOT / "result" / "phase_3"

    @property
    def phase2_lab_dir(self) -> Path:
        return PROJECT_ROOT / "result" / "phase_2" / "data"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plot"

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)