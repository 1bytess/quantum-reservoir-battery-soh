"""QRC evaluation on EIS features: depth sweep with LOCO and temporal.

Reuses the QuantumReservoir model from phase_4 (quantum circuit is dataset-agnostic).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    DEPTH_RANGE, CELL_IDS,
    USE_ZZ_CORRELATORS, RANDOM_STATE,
    Phase4LabPaths,
)

# Reuse the quantum reservoir from the NASA phase
from .qrc_model import QuantumReservoir

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_qrc_loco(
    cell_data: Dict[str, Dict],
    depth: int,
    use_classical: bool = False,
) -> pd.DataFrame:
    """Run QRC with LOCO evaluation across all cells.

    PCA is fit INSIDE each fold on training data only (leakage-free).

    Args:
        cell_data: {cell_id: {"X_6d": array, "X_72d": array, "y": array, ...}}
        depth: QRC circuit depth
        use_classical: Use classical fallback simulation

    Returns:
        DataFrame with per-fold results
    """
    results = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]

        # Always start from 72D, apply PCA in-fold
        X_train_72d = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        train_groups = np.concatenate([
            np.full(len(cell_data[c]["y"]), c, dtype=object) for c in train_cells
        ])

        X_test_72d = cell_data[test_cell]["X_72d"]
        y_test = cell_data[test_cell]["y"]

        X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

        # Train QRC
        qrc = QuantumReservoir(
            depth=depth,
            use_zz=USE_ZZ_CORRELATORS,
            use_classical_fallback=use_classical,
            add_random_rotations=True,
        )
        qrc.fit(X_train, y_train, groups=train_groups)
        y_pred = qrc.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

        results.append({
            "regime": "loco",
            "depth": depth,
            "test_cell": test_cell,
            "train_cells": "+".join(train_cells),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "naive_mae": naive_mae,
            "beats_naive": metrics["mae"] < naive_mae,
            "reservoir_dim": qrc.get_reservoir_dim(),
        })

    return pd.DataFrame(results)


def run_qrc_temporal(
    cell_data: Dict[str, Dict],
    depth: int,
    use_classical: bool = False,
    train_frac: float = 0.7,
) -> pd.DataFrame:
    """Run QRC with temporal split per cell.

    PCA is fit INSIDE each split on training blocks only (leakage-free).
    """
    results = []

    for cid in CELL_IDS:
        data = cell_data[cid]
        X_72d = data["X_72d"]
        y = data["y"]
        blocks = data["block_ids"]

        n_total = len(y)
        if n_total < 3:
            continue

        n_train = max(2, int(n_total * train_frac))
        sort_idx = np.argsort(blocks)
        X_72d_sorted = X_72d[sort_idx]
        y_sorted = y[sort_idx]

        X_train_72d, X_test_72d = X_72d_sorted[:n_train], X_72d_sorted[n_train:]
        y_train, y_test = y_sorted[:n_train], y_sorted[n_train:]

        if len(y_test) == 0:
            continue

        # Apply PCA in-fold (leakage-free)
        X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

        qrc = QuantumReservoir(
            depth=depth,
            use_zz=USE_ZZ_CORRELATORS,
            use_classical_fallback=use_classical,
        )
        qrc.fit(X_train, y_train)
        y_pred = qrc.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        persist_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train[-1]))

        results.append({
            "regime": "temporal",
            "depth": depth,
            "test_cell": cid,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "persist_mae": persist_mae,
            "beats_persist": metrics["mae"] < persist_mae,
        })

    return pd.DataFrame(results)


def run_depth_sweep(
    cell_data: Dict[str, Dict],
    paths: Phase4LabPaths,
    use_classical: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Run QRC depth sweep with LOCO and temporal evaluations.

    Returns:
        (results_df, predictions_dict)
    """
    all_results = []

    for depth in DEPTH_RANGE:
        print(f"\n  --- Depth {depth} ---")

        # LOCO
        loco_df = run_qrc_loco(cell_data, depth, use_classical)
        all_results.append(loco_df)
        avg_loco = loco_df["mae"].mean()
        print(f"    LOCO:     avg MAE = {avg_loco:.4f}")

        # Temporal
        temp_df = run_qrc_temporal(cell_data, depth, use_classical)
        all_results.append(temp_df)
        avg_temp = temp_df["mae"].mean() if len(temp_df) > 0 else float("nan")
        print(f"    Temporal: avg MAE = {avg_temp:.4f}")

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(paths.data_dir / "qrc_results.csv", index=False)
    print(f"\n  Saved qrc_results.csv ({len(results_df)} rows)")

    return results_df, {}
