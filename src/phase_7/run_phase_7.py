"""Phase 7: Lab Data Validation — single-cell temporal QRC evaluation.

Validates the QRC pipeline on ESCL lab data (Samsung INR18650-25R) using
temporal splits only.  Train on early cycles, predict later cycles — the
real-world BMS use case.

Usage:
    cd src
    python run_phase_7.py                   # Full run (QRC + classical)
    python run_phase_7.py --skip-classical  # QRC only (faster dev runs)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config_lab import (
    CELL_IDS_LAB,
    DEPTH_RANGE,
    N_PCA_LAB as N_PCA,
    RANDOM_STATE,
    RIDGE_ALPHAS,
    get_lab_result_paths,
)
from data_loader_lab import load_lab_data

# Import QRC + classical models from existing phases
from phase_4.qrc_model import QuantumReservoir
from phase_3.models import get_model_pipeline

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "model",
    "cell",
    "depth",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "r2",
    "persist_mae",
    "beats_persist",
    "soh_range_train",
    "soh_range_test",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TeeLogger:
    """Tee stdout to console and a log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def _save_figure(fig: plt.Figure, plot_dir, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = plot_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_pca_in_fold(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_components = min(N_PCA, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def _temporal_split(
    cell_data: dict, train_frac: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by block_ids and split 70/30."""
    X = cell_data["X_raw"]
    y = cell_data["y"]
    blocks = cell_data["block_ids"]

    order = np.argsort(blocks)
    X_sorted = X[order]
    y_sorted = y[order]

    n_total = len(y_sorted)
    n_train = int(np.floor(train_frac * n_total))
    n_train = max(2, min(n_train, n_total - 1))

    return (
        X_sorted[:n_train],
        y_sorted[:n_train],
        X_sorted[n_train:],
        y_sorted[n_train:],
    )


# ---------------------------------------------------------------------------
# Stage 1: QRC temporal evaluation
# ---------------------------------------------------------------------------

def run_qrc_temporal(
    cell_data: Dict[str, dict],
) -> pd.DataFrame:
    """Run noiseless QRC at each depth on every lab cell (temporal split)."""
    print("\n" + "=" * 70)
    print("Stage 1: Noiseless QRC temporal evaluation")
    print("=" * 70)

    rows: List[Dict] = []

    for depth in DEPTH_RANGE:
        print(f"\n  Depth {depth}")
        for cell_id, cdata in sorted(cell_data.items()):
            if len(cdata["y"]) < 4:
                print(f"    {cell_id}: skipped (< 4 samples)")
                continue

            X_train_raw, y_train, X_test_raw, y_test = _temporal_split(cdata)
            X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)

            qrc = QuantumReservoir(
                depth=depth,
                use_zz=True,
                use_classical_fallback=False,
                add_random_rotations=True,
                observable_set="Z",
            )
            qrc.fit(X_train, y_train)
            y_pred = qrc.predict(X_test)
            m = _metrics(y_test, y_pred)
            persist_mae = mean_absolute_error(
                y_test, np.full_like(y_test, y_train[-1])
            )

            rows.append({
                "model": f"qrc_d{depth}",
                "cell": cell_id,
                "depth": depth,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "persist_mae": float(persist_mae),
                "beats_persist": bool(m["mae"] < persist_mae),
                "soh_range_train": f"{100*y_train.min():.1f}-{100*y_train.max():.1f}",
                "soh_range_test": f"{100*y_test.min():.1f}-{100*y_test.max():.1f}",
            })
            print(
                f"    {cell_id}: MAE={m['mae']:.4f}  "
                f"persist={persist_mae:.4f}  "
                f"{'✓' if m['mae'] < persist_mae else '✗'}"
            )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)

    # Summary
    for depth in DEPTH_RANGE:
        sub = df[df["depth"] == depth]
        if not sub.empty:
            print(f"\n  Depth {depth} mean MAE: {sub['mae'].mean():.4f}")

    return df


# ---------------------------------------------------------------------------
# Stage 2: Classical baselines (temporal)
# ---------------------------------------------------------------------------

CLASSICAL_MODELS = ["esn", "xgboost", "svr", "linear_pc1", "rff", "mlp"]


def run_classical_temporal(
    cell_data: Dict[str, dict],
) -> pd.DataFrame:
    """Run classical baselines on every lab cell (temporal split)."""
    print("\n" + "=" * 70)
    print("Stage 2: Classical baselines (temporal)")
    print("=" * 70)

    rows: List[Dict] = []

    for model_name in CLASSICAL_MODELS:
        print(f"\n  Model: {model_name}")
        for cell_id, cdata in sorted(cell_data.items()):
            if len(cdata["y"]) < 4:
                print(f"    {cell_id}: skipped (< 4 samples)")
                continue

            X_train_raw, y_train, X_test_raw, y_test = _temporal_split(cdata)
            X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)

            try:
                model = get_model_pipeline(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                m = _metrics(y_test, y_pred)
            except Exception as exc:
                print(f"    {cell_id}: FAILED ({type(exc).__name__}: {exc})")
                continue

            persist_mae = mean_absolute_error(
                y_test, np.full_like(y_test, y_train[-1])
            )

            rows.append({
                "model": model_name,
                "cell": cell_id,
                "depth": np.nan,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "persist_mae": float(persist_mae),
                "beats_persist": bool(m["mae"] < persist_mae),
                "soh_range_train": f"{100*y_train.min():.1f}-{100*y_train.max():.1f}",
                "soh_range_test": f"{100*y_test.min():.1f}-{100*y_test.max():.1f}",
            })
            print(
                f"    {cell_id}: MAE={m['mae']:.4f}  "
                f"persist={persist_mae:.4f}  "
                f"{'✓' if m['mae'] < persist_mae else '✗'}"
            )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)

    # Summary
    for mn in CLASSICAL_MODELS:
        sub = df[df["model"] == mn]
        if not sub.empty:
            print(f"\n  {mn} mean MAE: {sub['mae'].mean():.4f}")

    return df


# ---------------------------------------------------------------------------
# Stage 3: Comparison + Plots
# ---------------------------------------------------------------------------

def build_comparison(
    qrc_df: pd.DataFrame, classical_df: pd.DataFrame
) -> pd.DataFrame:
    """Build summary comparison table (mean MAE across cells for each model)."""
    rows = []

    # Best QRC depth
    if not qrc_df.empty:
        by_depth = qrc_df.groupby("depth")["mae"].mean()
        best_depth = int(by_depth.idxmin())
        best_sub = qrc_df[qrc_df["depth"] == best_depth]
        rows.append({
            "model": f"qrc_d{best_depth} (best)",
            "mae_mean": float(by_depth.loc[best_depth]),
            "rmse_mean": float(best_sub["rmse"].mean()),
            "r2_mean": float(best_sub["r2"].mean()),
            "n_cells": int(best_sub["cell"].nunique()),
        })
        # All depths
        for d in sorted(qrc_df["depth"].dropna().unique()):
            sub = qrc_df[qrc_df["depth"] == d]
            rows.append({
                "model": f"qrc_d{int(d)}",
                "mae_mean": float(sub["mae"].mean()),
                "rmse_mean": float(sub["rmse"].mean()),
                "r2_mean": float(sub["r2"].mean()),
                "n_cells": int(sub["cell"].nunique()),
            })

    # Classical
    if not classical_df.empty:
        for mn in classical_df["model"].unique():
            sub = classical_df[classical_df["model"] == mn]
            rows.append({
                "model": mn,
                "mae_mean": float(sub["mae"].mean()),
                "rmse_mean": float(sub["rmse"].mean()),
                "r2_mean": float(sub["r2"].mean()),
                "n_cells": int(sub["cell"].nunique()),
            })

    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp = comp.sort_values("mae_mean").reset_index(drop=True)
    return comp


def plot_soh_prediction_curve(
    cell_data: Dict[str, dict],
    qrc_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    plot_dir,
) -> None:
    """Plot actual vs predicted SOH over cycle for CA5_AGING (primary cell)."""
    # Pick the cell with the most test samples
    target_cell = "CA5_AGING"
    if target_cell not in cell_data:
        # Fallback to largest cell
        target_cell = max(cell_data, key=lambda c: len(cell_data[c]["y"]))

    cdata = cell_data[target_cell]
    X_train_raw, y_train, X_test_raw, y_test = _temporal_split(cdata)
    blocks = np.sort(cdata["block_ids"])
    n_train = len(y_train)
    train_cycles = blocks[:n_train]
    test_cycles = blocks[n_train:]

    # Re-run best QRC to get predictions for plotting
    if not qrc_df.empty:
        best_depth = int(
            qrc_df.groupby("depth")["mae"].mean().idxmin()
        )
    else:
        best_depth = 1

    X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)

    qrc = QuantumReservoir(
        depth=best_depth,
        use_zz=True,
        use_classical_fallback=False,
        add_random_rotations=True,
    )
    qrc.fit(X_train_pca, y_train)
    y_pred_qrc = qrc.predict(X_test_pca)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Actual SOH
    ax.plot(
        np.concatenate([train_cycles, test_cycles]),
        np.concatenate([y_train, y_test]) * 100,
        "k-o", markersize=3, linewidth=1.2, label="Actual SOH", zorder=3,
    )

    # QRC prediction (test region only)
    ax.plot(
        test_cycles, y_pred_qrc * 100,
        "s-", markersize=4, linewidth=1.5,
        label=f"QRC (depth={best_depth})", color="#2196F3", zorder=4,
    )

    # Classical comparison on same cell (best model)
    if not classical_df.empty:
        cell_classical = classical_df[classical_df["cell"] == target_cell]
        if not cell_classical.empty:
            best_cl = cell_classical.loc[cell_classical["mae"].idxmin()]
            cl_name = best_cl["model"]
            # Re-run to get predictions
            try:
                model = get_model_pipeline(cl_name)
                model.fit(X_train_pca, y_train)
                y_pred_cl = model.predict(X_test_pca)
                ax.plot(
                    test_cycles, y_pred_cl * 100,
                    "^--", markersize=4, linewidth=1.2,
                    label=f"{cl_name}", color="#FF9800", zorder=4,
                )
            except Exception:
                pass

    # Train/test boundary
    ax.axvline(
        x=train_cycles[-1] + 0.5, color="gray", linestyle="--",
        alpha=0.7, label="Train/Test split",
    )
    ax.axvspan(
        test_cycles[0] - 0.5, test_cycles[-1] + 0.5,
        alpha=0.08, color="blue", label="Test region",
    )

    ax.set_xlabel("Cycle Index", fontsize=12)
    ax.set_ylabel("SOH (%)", fontsize=12)
    ax.set_title(f"SOH Prediction — {target_cell} (Temporal Split)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Close box
    for spine in ax.spines.values():
        spine.set_visible(True)

    _save_figure(fig, plot_dir, "soh_prediction_curve")


def plot_model_comparison(
    comp_df: pd.DataFrame, plot_dir
) -> None:
    """Bar chart: mean temporal MAE across cells for each model."""
    if comp_df.empty:
        print("Skipping model_comparison plot: no data")
        return

    # Filter out the "(best)" duplicate row if present
    plot_df = comp_df[~comp_df["model"].str.contains("best", case=False)]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    colors = []
    for m in plot_df["model"]:
        if m.startswith("qrc"):
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    bars = ax.bar(x, plot_df["mae_mean"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean Temporal MAE", fontsize=12)
    ax.set_title("QRC vs Classical Baselines — Lab Data Temporal", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)

    _save_figure(fig, plot_dir, "model_comparison")


def plot_depth_sensitivity(
    qrc_df: pd.DataFrame, plot_dir
) -> None:
    """QRC MAE vs depth for temporal evaluation."""
    if qrc_df.empty:
        print("Skipping depth_sensitivity plot: no QRC data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Mean across all cells
    by_depth = qrc_df.groupby("depth")["mae"].agg(["mean", "std"]).sort_index()
    ax.errorbar(
        by_depth.index, by_depth["mean"], yerr=by_depth["std"],
        marker="o", linewidth=2.0, capsize=4, label="All cells (mean ± std)",
        color="#2196F3",
    )

    # Per-cell lines (faded)
    for cell_id in qrc_df["cell"].unique():
        sub = qrc_df[qrc_df["cell"] == cell_id].sort_values("depth")
        ax.plot(
            sub["depth"], sub["mae"],
            marker=".", linewidth=0.8, alpha=0.35, label=None,
        )

    ax.set_xlabel("QRC Depth", fontsize=12)
    ax.set_ylabel("Temporal MAE", fontsize=12)
    ax.set_title("QRC Depth Sensitivity — Lab Data Temporal", fontsize=13)
    ax.set_xticks(DEPTH_RANGE)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)

    _save_figure(fig, plot_dir, "depth_sensitivity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_classical: bool = True) -> None:
    data_dir, plot_dir = get_lab_result_paths(7)
    log_path = data_dir / "phase_7_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 7: Lab Data Validation (Single-Cell Temporal)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")

        # Load ESCL lab data
        cell_data = load_lab_data()
        print(f"\nCells loaded: {list(cell_data.keys())}")
        for cid in sorted(cell_data.keys()):
            n = cell_data[cid]["X_raw"].shape[0]
            y = cell_data[cid]["y"]
            print(f"  {cid}: {n} samples, SOH={100*y.min():.1f}-{100*y.max():.1f}%")

        # Stage 1: QRC temporal
        qrc_df = run_qrc_temporal(cell_data)
        qrc_path = data_dir / "qrc_temporal.csv"
        qrc_df.to_csv(qrc_path, index=False)
        print(f"\nSaved: {qrc_path} ({len(qrc_df)} rows)")

        # Stage 2: Classical baselines
        if run_classical:
            classical_df = run_classical_temporal(cell_data)
        else:
            classical_df = pd.DataFrame(columns=RESULT_COLUMNS)
            print("\nStage 2 skipped by user flag.")

        cl_path = data_dir / "classical_temporal.csv"
        classical_df.to_csv(cl_path, index=False)
        print(f"Saved: {cl_path} ({len(classical_df)} rows)")

        # Stage 3: Comparison + plots
        print("\n" + "=" * 70)
        print("Stage 3: Comparison & Plots")
        print("=" * 70)

        comp_df = build_comparison(qrc_df, classical_df)
        comp_path = data_dir / "comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\nSaved: {comp_path} ({len(comp_df)} rows)")

        if not comp_df.empty:
            print("\nModel comparison (sorted by mean temporal MAE):")
            for _, row in comp_df.iterrows():
                print(
                f"  {row['model']:25s}  "
                f"MAE={row['mae_mean']:.4f}  "
                f"RMSE={row['rmse_mean']:.4f}  "
                f"R²={row['r2_mean']:.4f}"
            )

        plot_soh_prediction_curve(cell_data, qrc_df, classical_df, plot_dir)
        plot_model_comparison(comp_df, plot_dir)
        plot_depth_sensitivity(qrc_df, plot_dir)

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        if not qrc_df.empty:
            best_depth = int(qrc_df.groupby("depth")["mae"].mean().idxmin())
            best_mae = qrc_df.groupby("depth")["mae"].mean().loc[best_depth]
            print(f"  Best QRC depth: {best_depth} (mean temporal MAE={best_mae:.4f})")

            # CA5_AGING specific
            ca5 = qrc_df[
                (qrc_df["cell"] == "CA5_AGING") & (qrc_df["depth"] == best_depth)
            ]
            if not ca5.empty:
                print(f"  CA5_AGING @ depth={best_depth}: "
                      f"MAE={ca5.iloc[0]['mae']:.4f}, "
                      f"R²={ca5.iloc[0]['r2']:.4f}")

        if not classical_df.empty:
            best_cl = classical_df.groupby("model")["mae"].mean().idxmin()
            best_cl_mae = classical_df.groupby("model")["mae"].mean().loc[best_cl]
            print(f"  Best classical: {best_cl} (mean temporal MAE={best_cl_mae:.4f})")

        print(f"\nCompleted: {datetime.now().isoformat()}")

    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 7: Lab Data Validation (single-cell temporal QRC)."
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip Stage 2 classical baselines (faster dev runs).",
    )
    args = parser.parse_args()
    main(run_classical=not args.skip_classical)
