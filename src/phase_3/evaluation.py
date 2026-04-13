"""Evaluation utilities for Phase 3-Lab: LOCO and temporal evaluation on EIS.

Implements:
1. Leave-One-Cell-Out (LOCO) across all cells
2. Temporal split (within-cell, early → late blocks)

Mandatory baselines:
- Mean predictor (predict train-set mean)
- Linear-in-block-index (extrapolate linear trend)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from .config import (
    CELL_IDS, MODEL_NAMES,
    TEMPORAL_TRAIN_FRAC, Phase3LabPaths, HYPERPARAM_GRIDS,
    RIDGE_ALPHAS_FINE,
)

import sys as _sys
from pathlib import Path as _Path
_src = _Path(__file__).resolve().parents[1]
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))
from config import NOMINAL_CAPACITIES  # noqa: E402  (single source of truth)
from .models import get_model_pipeline, get_param_grid

# ── PCA defaults ──────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 6
PCA_RANDOM_STATE = 42

# NOMINAL_CAPACITIES imported from src/config.py (single source of truth).


def _fit_pca_in_fold(
    X_train_72d: np.ndarray,
    X_test_72d: np.ndarray,
    n_components: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training data only, transform both train and test.

    This prevents data leakage: the PCA basis is derived solely from
    training cells and applied to the held-out test cell.

    Returns:
        (X_train_6d, X_test_6d)
    """
    n_components = min(n_components, X_train_72d.shape[0], X_train_72d.shape[1])
    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    X_train_6d = pca.fit_transform(X_train_72d)
    X_test_6d = pca.transform(X_test_72d)
    return X_train_6d, X_test_6d


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nominal_capacity_ah: Optional[float] = None,
    dataset: Optional[str] = None,
) -> dict:
    """Compute MAE, RMSE, MAPE, R² and cross-dataset comparable metrics.

    Reviewer: "Nature Comms battery papers typically report MAE%, RMSE%, and
    R-squared. Convert all metrics to percentage of nominal capacity for
    cross-dataset comparability."

    Args:
        y_true: True SOH values (fractions, e.g. 0.85).
        y_pred: Predicted SOH values.
        nominal_capacity_ah: Nominal cell capacity in Ah. If provided, computes
            MAE_Ah = MAE × nominal_capacity_ah and MAE_pct = MAE × 100.
        dataset: Dataset key for automatic nominal capacity lookup
            (one of "stanford", "escl", "warwick").

    Returns:
        dict with keys: mae, rmse, r2, mape, mae_norm, mae_pct, rmse_pct,
            mae_ah (if nominal_capacity_ah known).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true > 0) else float("nan")
    soh_range = y_true.max() - y_true.min() if len(y_true) > 1 else 1.0
    mae_norm = mae / soh_range if soh_range > 0 else float("nan")

    # Resolve nominal capacity
    if nominal_capacity_ah is None and dataset is not None:
        nominal_capacity_ah = NOMINAL_CAPACITIES.get(dataset)

    metrics: dict = {
        "mae":      mae,
        "rmse":     rmse,
        "r2":       r2,
        "mape":     mape,
        "mae_norm": mae_norm,
        # Cross-dataset comparable: MAE as % of SOH range (0→1 = 100 pp)
        "mae_pct":  mae * 100.0,
        "rmse_pct": rmse * 100.0,
    }

    if nominal_capacity_ah is not None:
        metrics["mae_ah"]  = mae * nominal_capacity_ah
        metrics["rmse_ah"] = rmse * nominal_capacity_ah

    return metrics


def naive_baseline_mean(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Predict train-set mean for all test samples."""
    return np.full_like(y_test, y_train.mean())


def naive_baseline_linear_block(
    block_train: np.ndarray,
    y_train: np.ndarray,
    block_test: np.ndarray,
) -> np.ndarray:
    """Linear-in-block-index baseline: fit SOH = a*block + b from training."""
    if len(block_train) < 2:
        return np.full(len(block_test), y_train.mean())
    coeffs = np.polyfit(block_train, y_train, 1)
    return np.polyval(coeffs, block_test)


def run_loco_evaluation(
    cell_data: Dict[str, Dict],
    model_name: str,
    use_6d: bool = True,
    tune_hyperparams: bool = True,
    dataset: Optional[str] = None,
    cell_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run Leave-One-Cell-Out evaluation across all cells.

    For each test cell:
      - Train on the other 3 cells
      - Test on the held-out cell

    PCA is now fit INSIDE each fold on training data only (leakage-free).

    Args:
        cell_data: {cell_id: {"X_6d": array, "X_72d": array, "y": array, ...}}
        model_name: Name of model to evaluate
        use_6d: If True, apply in-fold PCA (72D→6D). If False, use raw 72D.
        tune_hyperparams: Enable GridSearchCV tuning
        dataset: Dataset name for MAE% reporting ("stanford", "escl", "warwick")
        cell_ids: Override default CELL_IDS (useful for Warwick 24-cell LOCO)

    Returns:
        DataFrame with columns: model, test_cell, train_cells, mae, rmse, r2,
            mae_pct, rmse_pct, mae_ah (if dataset known), ...
    """
    results = []
    ids_to_use = cell_ids if cell_ids is not None else CELL_IDS

    for test_cell in ids_to_use:
        train_cells = [c for c in ids_to_use if c != test_cell]

        # Always start from 72D raw features (or X_raw if X_72d absent)
        raw_key = "X_72d" if "X_72d" in cell_data[test_cell] else "X_raw"
        X_train_72d = np.vstack([cell_data[c][raw_key] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        train_groups = np.concatenate([
            np.full(len(cell_data[c]["y"]), c, dtype=object) for c in train_cells
        ])

        X_test_72d = cell_data[test_cell][raw_key]
        y_test = cell_data[test_cell]["y"]

        # Apply PCA in-fold (leakage-free) or use raw 72D
        if use_6d:
            X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)
        else:
            X_train, X_test = X_train_72d, X_test_72d

        # Fit model
        model = get_model_pipeline(model_name)
        param_grid = get_param_grid(model_name)

        if tune_hyperparams and param_grid:
            n_group_folds = len(np.unique(train_groups))
            if n_group_folds >= 2:
                grid = GridSearchCV(
                    model, param_grid,
                    cv=LeaveOneGroupOut(),
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train, groups=train_groups)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, dataset=dataset)

        # Compute baselines
        y_naive_mean = naive_baseline_mean(y_train, y_test)
        naive_mae = mean_absolute_error(y_test, y_naive_mean)

        row = {
            "model": model_name,
            "dataset": dataset or "unknown",
            "test_cell": test_cell,
            "train_cells": "+".join(train_cells),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "mae_norm": metrics["mae_norm"],
            "mae_pct": metrics.get("mae_pct", float("nan")),
            "rmse_pct": metrics.get("rmse_pct", float("nan")),
            "naive_mean_mae": naive_mae,
            "beats_naive": metrics["mae"] < naive_mae,
        }
        if "mae_ah" in metrics:
            row["mae_ah"] = metrics["mae_ah"]
            row["rmse_ah"] = metrics["rmse_ah"]
        results.append(row)

    return pd.DataFrame(results)


def run_temporal_evaluation(
    cell_data: Dict[str, Dict],
    model_name: str,
    use_6d: bool = True,
    train_frac: float = TEMPORAL_TRAIN_FRAC,
) -> pd.DataFrame:
    """Within-cell temporal split: train on early blocks, test on late blocks.

    PCA is fit INSIDE each split on training blocks only (leakage-free).

    Args:
        cell_data: {cell_id: {"X_6d", "X_72d", "y", "block_ids", ...}}
        model_name: Model to evaluate
        use_6d: If True, apply in-fold PCA (72D→6D). If False, use raw 72D.
        train_frac: Fraction of blocks for training

    Returns:
        DataFrame with per-cell results
    """
    results = []

    for cid in CELL_IDS:
        data = cell_data[cid]
        X_72d = data["X_72d"]
        y = data["y"]
        blocks = data["block_ids"]

        n_total = len(y)
        if n_total < 3:
            print(f"  WARNING: {cid} only has {n_total} blocks, skipping temporal split")
            continue

        n_train = max(2, int(n_total * train_frac))
        sort_idx = np.argsort(blocks)
        X_72d_sorted = X_72d[sort_idx]
        y_sorted = y[sort_idx]
        blocks_sorted = blocks[sort_idx]

        X_train_72d, X_test_72d = X_72d_sorted[:n_train], X_72d_sorted[n_train:]
        y_train, y_test = y_sorted[:n_train], y_sorted[n_train:]
        blocks_train = blocks_sorted[:n_train]
        blocks_test = blocks_sorted[n_train:]

        if len(y_test) == 0:
            continue

        # Apply PCA in-fold (leakage-free) or use raw 72D
        if use_6d:
            X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)
        else:
            X_train, X_test = X_train_72d, X_test_72d

        # Fit model
        model = get_model_pipeline(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        # Baselines
        y_naive_mean = naive_baseline_mean(y_train, y_test)
        y_naive_linear = naive_baseline_linear_block(blocks_train, y_train, blocks_test)
        persist_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train[-1]))
        linear_mae = mean_absolute_error(y_test, y_naive_linear)

        # A temporal result is valid only when there are enough training blocks
        # for the model to be well-conditioned.  Ridge in particular blows up
        # when n_train < n_features (extrapolation to MAE ~ 1e+25 for V5 which
        # has only 3 blocks → 2 train / 1 test).  We keep the row so the cell
        # count is consistent but flag it so downstream tables can filter it out.
        n_features = X_train.shape[1] if X_train.ndim == 2 else 1
        valid = (len(y_train) >= 3) and (metrics["mae"] <= 1.0)

        results.append({
            "model": model_name,
            "cell": cid,
            "regime": "temporal",
            "n_train": len(y_train),
            "n_test": len(y_test),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "persist_mae": persist_mae,
            "linear_mae": linear_mae,
            "beats_persist": metrics["mae"] < persist_mae,
            "beats_linear": metrics["mae"] < linear_mae,
            "valid": valid,
        })

    return pd.DataFrame(results)


def run_all_evaluations(
    cell_data: Dict[str, Dict],
    paths: Phase3LabPaths,
    tune_hyperparams: bool = True,
    model_names: Optional[List[str]] = None,
    incremental: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """Run all evaluation regimes for all models.

    Args:
        cell_data: Per-cell feature/label dicts.
        paths: Phase3LabPaths with output directories.
        tune_hyperparams: Enable GridSearchCV tuning.
        model_names: Override the default MODEL_NAMES list. Pass e.g.
            ["gp", "cnn1d"] to run only those models without re-running
            everything. If None, all MODEL_NAMES are run.
        incremental: If True and result CSVs already exist, load them and
            skip models whose results are already present. New results are
            appended and the CSVs are re-saved. This lets you add GP/CNN1D
            to an existing run without recomputing ridge/svr/xgboost/etc.

    Returns:
        (results_dict, predictions_dict)
    """
    models_to_run = model_names if model_names is not None else MODEL_NAMES

    # ── Incremental load ───────────────────────────────────────────────────────
    existing_loco: List[pd.DataFrame] = []
    existing_temp: List[pd.DataFrame] = []
    already_done: set = set()

    loco_csv = paths.data_dir / "loco_results.csv"
    temp_csv = paths.data_dir / "temporal_results.csv"

    if incremental and loco_csv.exists() and temp_csv.exists():
        existing_loco = [pd.read_csv(loco_csv)]
        existing_temp = [pd.read_csv(temp_csv)]
        already_done = set(existing_loco[0]["model"].unique())
        print(f"  [incremental] Loaded existing results — skipping: {sorted(already_done)}")
        # Remove already-completed models from the run list
        models_to_run = [m for m in models_to_run if m not in already_done]
        if not models_to_run:
            print("  [incremental] All requested models already in results. Nothing to do.")
            return {
                "loco":     existing_loco[0],
                "temporal": existing_temp[0],
            }, {}

    all_results: Dict[str, List[pd.DataFrame]] = {
        "loco": list(existing_loco),
        "temporal": list(existing_temp),
    }

    for model_name in models_to_run:
        print(f"\n  === {model_name.upper()} ===")

        try:
            # LOCO on 6D
            print(f"  Running LOCO (6D)...")
            loco_df = run_loco_evaluation(
                cell_data, model_name, use_6d=True,
                tune_hyperparams=tune_hyperparams,
            )
            all_results["loco"].append(loco_df)
            for _, row in loco_df.iterrows():
                status = "[OK]" if row["beats_naive"] else "[X]"
                print(f"    {status} {row['test_cell']}: MAE={row['mae']:.4f} "
                      f"(naive={row['naive_mean_mae']:.4f})")

            # Temporal on 6D
            print(f"  Running Temporal (6D)...")
            temp_df = run_temporal_evaluation(
                cell_data, model_name, use_6d=True,
            )
            all_results["temporal"].append(temp_df)
            for _, row in temp_df.iterrows():
                status = "[OK]" if row["beats_persist"] else "[X]"
                print(f"    {status} {row['cell']}: MAE={row['mae']:.4f} "
                      f"(persist={row['persist_mae']:.4f})")

            # Incremental save after each model so a crash doesn't lose work
            if incremental:
                _save_incremental(all_results, paths)

        except Exception as e:
            print(f"  *** FAILED: {model_name} — {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ── QRC temporal evaluation (always run on full suite) ─────────────────────
    # QRC uses a bespoke QuantumReservoir fit and cannot go through the standard
    # get_model_pipeline path.  We run the temporal evaluation here so that
    # "qrc_temporal" appears in temporal_results.csv alongside classical baselines.
    # Skip when doing a targeted --models partial run (model_names is not None)
    # unless "qrc" is explicitly in the requested models.
    if model_names is None or "qrc" in (model_names or []):
        if not (incremental and "qrc_temporal" in already_done):
            print("\n  === QRC_TEMPORAL ===")
            try:
                qrc_temp_df = run_qrc_temporal_evaluation(cell_data)
                qrc_temp_df["model"] = "qrc_temporal"
                all_results["temporal"].append(qrc_temp_df)
                for _, row in qrc_temp_df.iterrows():
                    status = "[OK]" if row["beats_persist"] else "[X]"
                    print(f"    {status} {row['cell']}: MAE={row['mae']:.4f} "
                          f"(persist={row['persist_mae']:.4f})")
                if incremental:
                    _save_incremental(all_results, paths)
            except Exception as e:
                print(f"  *** FAILED: qrc_temporal — {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

    # Also run raw-72D baselines for fair comparison with temporal QRC
    # (only when running the full suite, not on partial --models runs)
    if model_names is None:
        baselines_72d = [
            ("ridge", "ridge_72d"),
            ("svr", "svr_72d"),
            ("xgboost", "xgboost_72d"),
            ("rff", "rff_72d"),
            ("esn", "esn_72d"),
        ]
        for base_model_name, output_model_name in baselines_72d:
            if incremental and output_model_name in already_done:
                continue
            print(f"\n  === {output_model_name.upper()} (baseline) ===")
            try:
                loco_72d = run_loco_evaluation(
                    cell_data, base_model_name, use_6d=False,
                    tune_hyperparams=tune_hyperparams,
                )
                loco_72d["model"] = output_model_name
                all_results["loco"].append(loco_72d)
                for _, row in loco_72d.iterrows():
                    status = "[OK]" if row["beats_naive"] else "[X]"
                    print(f"    {status} {row['test_cell']}: MAE={row['mae']:.4f}")

                temp_72d = run_temporal_evaluation(
                    cell_data, base_model_name, use_6d=False,
                )
                temp_72d["model"] = output_model_name
                all_results["temporal"].append(temp_72d)
            except Exception as e:
                print(f"  *** FAILED: {output_model_name} — {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

    # Combine and save
    combined: Dict[str, pd.DataFrame] = {}
    for key in all_results:
        if all_results[key]:
            combined[key] = pd.concat(all_results[key], ignore_index=True)
        else:
            combined[key] = pd.DataFrame()

    combined["loco"].to_csv(loco_csv, index=False)
    combined["temporal"].to_csv(temp_csv, index=False)
    print(f"\n  Saved results to {paths.data_dir}")

    return combined, {}


def _save_incremental(
    all_results: Dict[str, List[pd.DataFrame]],
    paths: Phase3LabPaths,
) -> None:
    """Write intermediate combined CSVs after each model completes."""
    for key, csv_name in [("loco", "loco_results.csv"), ("temporal", "temporal_results.csv")]:
        if all_results[key]:
            pd.concat(all_results[key], ignore_index=True).to_csv(
                paths.data_dir / csv_name, index=False
            )


def run_qrc_temporal_evaluation(
    cell_data: Dict[str, Dict],
    train_frac: float = TEMPORAL_TRAIN_FRAC,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Within-cell temporal split for QRC — mirrors run_temporal_evaluation.

    Lazy-imports QuantumReservoir from phase_4 so that phase_3 has no hard
    dependency on Qiskit at import time.  PCA is fit inside each temporal
    split on training blocks only (leakage-free).

    N_QUBITS is fixed at 6 in the QRC circuit; the in-fold PCA already
    produces 6 components by default, so no padding/truncation is needed
    as long as N_PCA_COMPONENTS == 6 (the default).

    Args:
        cell_data:  {cell_id: {"X_72d": array, "y": array, "block_ids": array}}
        train_frac: Fraction of blocks for training (default TEMPORAL_TRAIN_FRAC).
        dataset:    Dataset key for MAE% reporting ("stanford", "escl", "warwick").

    Returns:
        DataFrame with per-cell temporal results for model="qrc_temporal".
    """
    # Lazy import — keeps phase_3 importable even without Qiskit installed
    try:
        from phase_4.qrc_model import QuantumReservoir  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "phase_4.qrc_model could not be imported.  Ensure Qiskit is installed "
            "and the src/ directory is on sys.path before calling "
            "run_qrc_temporal_evaluation()."
        ) from exc

    results = []

    for cid in CELL_IDS:
        data = cell_data[cid]
        X_72d   = data["X_72d"]
        y       = data["y"]
        blocks  = data["block_ids"]

        n_total = len(y)
        if n_total < 3:
            print(f"  WARNING: {cid} only has {n_total} blocks, skipping QRC temporal split")
            continue

        n_train = max(2, int(n_total * train_frac))
        sort_idx = np.argsort(blocks)
        X_sorted = X_72d[sort_idx]
        y_sorted = y[sort_idx]
        blocks_sorted = blocks[sort_idx]

        X_train_72d, X_test_72d = X_sorted[:n_train],  X_sorted[n_train:]
        y_train,     y_test     = y_sorted[:n_train],   y_sorted[n_train:]
        blocks_train = blocks_sorted[:n_train]
        blocks_test  = blocks_sorted[n_train:]

        if len(y_test) == 0:
            continue

        # In-fold PCA (leakage-free): fit on training blocks only
        X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

        # QRC temporal fit — group labels not needed for temporal (single cell)
        qrc = QuantumReservoir(
            depth=1,
            use_zz=True,
            observable_set="Z",
            add_random_rotations=True,
        )
        qrc.fit(X_train, y_train)
        y_pred = qrc.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, dataset=dataset)

        # Baselines
        y_naive_linear = naive_baseline_linear_block(blocks_train, y_train, blocks_test)
        persist_mae    = mean_absolute_error(y_test, np.full_like(y_test, y_train[-1]))
        linear_mae     = mean_absolute_error(y_test, y_naive_linear)

        valid = (len(y_train) >= 3) and (metrics["mae"] <= 1.0)
        row = {
            "model":         "qrc_temporal",
            "cell":          cid,
            "regime":        "temporal",
            "n_train":       len(y_train),
            "n_test":        len(y_test),
            "mae":           metrics["mae"],
            "rmse":          metrics["rmse"],
            "r2":            metrics["r2"],
            "mape":          metrics["mape"],
            "mae_pct":       metrics.get("mae_pct", float("nan")),
            "rmse_pct":      metrics.get("rmse_pct", float("nan")),
            "persist_mae":   persist_mae,
            "linear_mae":    linear_mae,
            "beats_persist": metrics["mae"] < persist_mae,
            "beats_linear":  metrics["mae"] < linear_mae,
            "valid":         valid,
        }
        if "mae_ah" in metrics:
            row["mae_ah"]  = metrics["mae_ah"]
            row["rmse_ah"] = metrics["rmse_ah"]
        results.append(row)

    return pd.DataFrame(results)
