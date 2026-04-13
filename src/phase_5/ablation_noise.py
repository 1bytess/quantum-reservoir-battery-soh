"""Noise ablation study on classical models with EIS features.

Injects Gaussian noise into EIS features at increasing levels,
then evaluates classical models to compare with QRC's intrinsic
quantum shot noise robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

from .config import (
    NOISE_LEVELS, CLASSICAL_MODELS, CELL_IDS,
    RANDOM_STATE, N_NOISE_REPEATS,
)
from src.phase_3.models import get_model_pipeline
from src.phase_3.data_loader import get_cell_data
from src.phase_3.config import Phase3LabPaths

# ── PCA defaults ──────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 6
PCA_RANDOM_STATE = 42


def _fit_pca_in_fold(
    X_train_72d: np.ndarray,
    X_test_72d: np.ndarray,
    n_components: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training data only, transform both train and test."""
    n_components = min(n_components, X_train_72d.shape[0], X_train_72d.shape[1])
    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    X_train_6d = pca.fit_transform(X_train_72d)
    X_test_6d = pca.transform(X_test_72d)
    return X_train_6d, X_test_6d


def inject_noise(X: np.ndarray, noise_level: float, rng: np.random.RandomState) -> np.ndarray:
    """Add Gaussian noise scaled by feature std.

    noise_level=0.01 means 1% of each feature's standard deviation.
    """
    if noise_level == 0:
        return X.copy()
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero
    noise = rng.randn(*X.shape) * noise_level * std
    return X + noise


def run_noise_ablation(
    cell_data: Dict[str, Dict],
    noise_levels: List[float] = NOISE_LEVELS,
    models: List[str] = CLASSICAL_MODELS,
    n_repeats: int = N_NOISE_REPEATS,
) -> pd.DataFrame:
    """Run noise ablation on classical models with LOCO evaluation.

    PCA is fit INSIDE each fold on training data only (leakage-free).

    For each noise level:
      - Fit PCA on training cells' 72D features (in-fold)
      - Inject noise into training 6D features (test features kept clean)
      - Evaluate using LOCO across all cells
      - Repeat n_repeats times and average

    Returns:
        DataFrame with columns: model, noise_level, repeat, test_cell, mae
    """
    results = []
    rng = np.random.RandomState(RANDOM_STATE)

    for noise_level in noise_levels:
        print(f"\n  Noise level: {noise_level}")

        for repeat in range(n_repeats):
            for model_name in models:
                for test_cell in CELL_IDS:
                    train_cells = [c for c in CELL_IDS if c != test_cell]

                    # Always start from 72D, apply PCA in-fold
                    X_train_72d = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
                    y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
                    X_test_72d = cell_data[test_cell]["X_72d"]
                    y_test = cell_data[test_cell]["y"]

                    X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

                    # Inject noise into training features
                    X_train_noisy = inject_noise(X_train, noise_level, rng)

                    model = get_model_pipeline(model_name)
                    model.fit(X_train_noisy, y_train)
                    y_pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    results.append({
                        "model": model_name,
                        "noise_level": noise_level,
                        "repeat": repeat,
                        "test_cell": test_cell,
                        "mae": mae,
                    })

        # Progress
        n_done = len([r for r in results if r["noise_level"] <= noise_level])
        print(f"    Completed: {n_done} evaluations so far")

    return pd.DataFrame(results)
