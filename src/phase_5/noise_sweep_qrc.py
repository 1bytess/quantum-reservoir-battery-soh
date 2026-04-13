"""QRC noise sweep: inject input feature noise at various levels.

Mirrors the classical noise ablation in ablation_noise.py, but for QRC.
Tests whether QRC exhibits stochastic resonance — improved performance
at moderate noise levels — due to its quantum reservoir's intrinsic
noise tolerance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

from .config import (
    NOISE_LEVELS, CELL_IDS, RANDOM_STATE, N_NOISE_REPEATS,
)
from ..phase_4.qrc_model import QuantumReservoir
from ..phase_4.config import USE_ZZ_CORRELATORS, DEPTH_RANGE

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
    """Add Gaussian noise scaled by feature std."""
    if noise_level == 0:
        return X.copy()
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    noise = rng.randn(*X.shape) * noise_level * std
    return X + noise


def run_noise_sweep_qrc(
    cell_data: Dict[str, Dict],
    noise_levels: List[float] = NOISE_LEVELS,
    depth: int = 3,
    n_repeats: int = N_NOISE_REPEATS,
) -> pd.DataFrame:
    """Sweep input noise levels on QRC with LOCO evaluation.

    For each noise level:
      - Fit PCA in-fold (leakage-free)
      - Inject noise into training features (test features kept clean)
      - Train QRC on noisy features, evaluate on clean test
      - Repeat n_repeats times

    Args:
        cell_data: Per-cell EIS data
        noise_levels: List of noise magnitudes to test
        depth: QRC circuit depth (default: best depth from Phase 4)
        n_repeats: Number of random repetitions per noise level

    Returns:
        DataFrame with columns: noise_level, repeat, test_cell, mae
    """
    results = []
    rng = np.random.RandomState(RANDOM_STATE)

    for noise_level in noise_levels:
        print(f"\n  QRC noise sweep: level={noise_level}")

        for repeat in range(n_repeats):
            for test_cell in CELL_IDS:
                train_cells = [c for c in CELL_IDS if c != test_cell]

                # Start from 72D, PCA in-fold
                X_train_72d = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
                y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
                X_test_72d = cell_data[test_cell]["X_72d"]
                y_test = cell_data[test_cell]["y"]

                X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

                # Inject noise into training features
                X_train_noisy = inject_noise(X_train, noise_level, rng)

                # Train QRC on noisy features
                qrc = QuantumReservoir(
                    depth=depth,
                    use_zz=USE_ZZ_CORRELATORS,
                    use_classical_fallback=True,
                    add_random_rotations=True,
                )
                qrc.fit(X_train_noisy, y_train)
                y_pred = qrc.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                results.append({
                    "model": "qrc",
                    "noise_level": noise_level,
                    "repeat": repeat,
                    "test_cell": test_cell,
                    "depth": depth,
                    "mae": mae,
                })

    return pd.DataFrame(results)
