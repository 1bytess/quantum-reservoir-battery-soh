"""LOCO evaluation wrapper for TemporalQuantumReservoir (no PCA)."""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import CELL_IDS, Phase4LabPaths
from .temporal_qrc import TemporalQuantumReservoir


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_temporal_qrc_loco(
    cell_data: Dict[str, Dict],
    depth: int = 1,
    use_zz: bool = True,
    washout: int = 3,
) -> pd.DataFrame:
    """Run temporal QRC with LOCO evaluation (72D raw EIS, no PCA).

    Args:
        cell_data: ``{cell_id: {"X_72d": array, "y": array, ...}}``
        depth: CZ-ring layers per frequency step.
        use_zz: Include ZZ correlators.
        washout: Transient steps to discard.

    Returns:
        DataFrame with per-fold results.
    """
    results = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]

        X_train = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test = cell_data[test_cell]["X_72d"]
        y_test = cell_data[test_cell]["y"]

        print(f"    Temporal QRC  {'+'.join(train_cells)} -> {test_cell} "
              f"(depth={depth}, 72D->temporal)")

        tqrc = TemporalQuantumReservoir(
            depth=depth, use_zz=use_zz, washout=washout,
        )
        tqrc.fit(X_train, y_train)
        y_pred = tqrc.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        naive_mae = mean_absolute_error(
            y_test, np.full_like(y_test, y_train.mean()),
        )

        results.append({
            "model": "temporal_qrc",
            "stage": "noiseless",
            "regime": "loco",
            "depth": depth,
            "test_cell": test_cell,
            "train_cells": "+".join(train_cells),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "naive_mae": naive_mae,
            "beats_naive": metrics["mae"] < naive_mae,
            "reservoir_dim": tqrc.get_reservoir_dim(),
        })

        status = "+" if metrics["mae"] < naive_mae else "-"
        print(f"      {status} {test_cell}: MAE={metrics['mae']:.4f} "
              f"(naive={naive_mae:.4f}, dim={tqrc.get_reservoir_dim()})")

    return pd.DataFrame(results)


def run_temporal_qrc(
    cell_data: Dict[str, Dict],
    paths: Phase4LabPaths,
    depths: list = None,
) -> pd.DataFrame:
    """Run temporal QRC evaluation across depths and save results."""
    if depths is None:
        depths = [1, 2]

    print("\n  --- Temporal QRC (72D raw EIS, no PCA) ---")

    all_results = []
    for depth in depths:
        print(f"\n  Depth {depth}:")
        df = run_temporal_qrc_loco(cell_data, depth=depth)
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = paths.data_dir / "temporal_qrc.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved temporal_qrc.csv ({len(results_df)} rows)")

    # Summary
    summary = results_df.groupby("depth")["mae"].agg(["mean", "std"]).round(4)
    print("\n  Temporal QRC Summary (LOCO MAE):")
    print(summary.to_string())

    return results_df
