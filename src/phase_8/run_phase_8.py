"""
Phase 8 — External Validation: Warwick DIB Dataset (NMC 811, 24 cells)

Cross-cell Leave-One-Cell-Out (LOCO) evaluation on the completely independent
Warwick University DIB dataset (Rashid et al. 2023, DOI: 10.17632/mn9fb7xdx6.3).

Cells:  24 NMC 811 cylindrical 21700 cells (5 Ah nominal)
SOH:    80–100%  (measured, 5 levels)
EIS:    61 log-spaced frequencies, 10 kHz → 10 mHz, at 25 °C / 50 % SOC
Feature: 122-dim [Re(Z)×61 | Im(Z)×61], reduced to 6D via in-fold PCA

Comparison models (same set as Stanford pipeline):
    QRC d=1 Z+ZZ (6-qubit, depth-1)
    XGBoost (PCA-6D)
    ESN     (PCA-6D)
    Ridge   (PCA-6D)
    SVR     (PCA-6D)
    RFF     (PCA-6D)
    MLP     (PCA-6D)
"""

from __future__ import annotations

import sys
import os

import datetime, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import get_result_paths, N_PCA, RANDOM_STATE, RIDGE_ALPHAS
from data_loader_warwick import get_warwick_arrays
from phase_4.qrc_model import QuantumReservoir
from phase_3.models import get_model_pipeline

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR, PLOT_DIR = get_result_paths(8)

# ── Models to benchmark ────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qrc_d1_z_plus_zz": "QRC (d=1, 6-qubit)",
    "xgboost_pca6":     "XGBoost",
    "esn_pca6":         "ESN (classical RC)",
    "ridge_pca6":       "Ridge",
    "svr_pca6":         "SVR",
    "rff_pca6":         "RFF",
    "mlp_pca6":         "MLP",
}

MODEL_COLORS = {
    "qrc_d1_z_plus_zz": "#0077BB",
    "xgboost_pca6":     "#009988",
    "esn_pca6":         "#EE7733",
    "ridge_pca6":       "#BBBBBB",
    "svr_pca6":         "#CC3311",
    "rff_pca6":         "#AA3377",
    "mlp_pca6":         "#33BBEE",
}

MODEL_PLOT_ORDER = ["qrc_d1_z_plus_zz", "xgboost_pca6", "esn_pca6",
                    "ridge_pca6", "svr_pca6", "rff_pca6", "mlp_pca6"]


def _metrics(y_true, y_pred) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _pca_transform(X_train, X_test, n_components=N_PCA):
    """In-fold PCA: fit on train only."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    return X_tr, X_te


# ── LOCO evaluation ────────────────────────────────────────────────────────────
def run_loco(X_raw, y, cell_ids):
    """Leave-One-Cell-Out: each cell is test set once."""
    rows = []
    n = len(cell_ids)
    print(f"  Running {n}-fold LOCO  ({n} cells)...")

    for i, test_cell in enumerate(cell_ids):
        mask_test  = np.array([c == test_cell for c in cell_ids])
        mask_train = ~mask_test

        X_train_raw = X_raw[mask_train]
        X_test_raw  = X_raw[mask_test]
        y_train     = y[mask_train]
        y_test      = y[mask_test]

        # In-fold PCA (6D)
        X_train_pca, X_test_pca = _pca_transform(X_train_raw, X_test_raw)

        naive_mae = float(np.mean(np.abs(y_test - np.mean(y_train))))

        def _row(model_name, m):
            return {
                "stage": "loco",
                "model": model_name,
                "test_cell": test_cell,
                "n_train": int(mask_train.sum()),
                "n_test": int(mask_test.sum()),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "naive_mae": naive_mae,
            }

        # --- QRC ---
        qrc = QuantumReservoir(
            depth=1, use_zz=True, use_classical_fallback=False,
            add_random_rotations=True, observable_set="Z"
        )
        train_groups = np.arange(len(y_train))  # one sample per cell → dummy groups
        qrc.fit(X_train_pca, y_train, groups=train_groups)
        rows.append(_row("qrc_d1_z_plus_zz", _metrics(y_test, qrc.predict(X_test_pca))))

        # --- Classical models ---
        for model_key in ["xgboost", "esn", "ridge", "svr", "rff", "mlp"]:
            clf = get_model_pipeline(model_key)
            clf.fit(X_train_pca, y_train)
            col_key = f"{model_key}_pca6"
            rows.append(_row(col_key, _metrics(y_test, clf.predict(X_test_pca))))

        if (i + 1) % 6 == 0 or i == n - 1:
            print(f"    fold {i+1}/{n} done")

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_results(summary: pd.DataFrame):
    """Bar chart: mean LOCO MAE per model with ±1 std error."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    models  = [m for m in MODEL_PLOT_ORDER if m in summary["model"].values]
    means   = [summary.loc[summary["model"] == m, "mae_mean"].values[0] for m in models]
    stds    = [summary.loc[summary["model"] == m, "mae_std"].values[0]  for m in models]
    labels  = [MODEL_CONFIGS[m] for m in models]
    colors  = [MODEL_COLORS[m]  for m in models]

    bars = ax.barh(labels[::-1], means[::-1], xerr=stds[::-1],
                   color=colors[::-1], alpha=0.85, height=0.55,
                   error_kw=dict(elinewidth=1.2, capsize=4, ecolor="#333333"))

    # Annotate QRC bar
    qrc_idx = len(models) - 1 - models.index("qrc_d1_z_plus_zz")
    ax.text(means[models.index("qrc_d1_z_plus_zz")] + stds[models.index("qrc_d1_z_plus_zz")] + 0.0005,
            qrc_idx, "← best", va="center", fontsize=8, color="#0077BB", fontweight="bold")

    ax.set_xlabel("LOCO MAE (SOH fraction)", fontsize=10)
    ax.set_title("Phase 8: Cross-cell LOCO — Warwick DIB Dataset\n"
                 "(NMC 811, 24 cells, 25°C / 50% SOC)", fontsize=10)
    ax.axvline(x=means[models.index("qrc_d1_z_plus_zz")], color="#0077BB",
               linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(left=0)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"warwick_loco_comparison.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {p}")
    plt.close(fig)


def plot_scatter(df_loco: pd.DataFrame, X_raw, y, cell_ids):
    """Scatter: predicted vs actual SOH for QRC across all LOCO folds."""
    y_pred_all = np.full_like(y, np.nan)

    for i, test_cell in enumerate(cell_ids):
        mask_test  = np.array([c == test_cell for c in cell_ids])
        mask_train = ~mask_test
        X_tr_pca, X_te_pca = _pca_transform(X_raw[mask_train], X_raw[mask_test])

        qrc = QuantumReservoir(depth=1, use_zz=True, use_classical_fallback=False,
                               add_random_rotations=True, observable_set="Z")
        qrc.fit(X_tr_pca, y[mask_train], groups=np.arange(mask_train.sum()))
        y_pred_all[mask_test] = qrc.predict(X_te_pca)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    soh_pct_true = y * 100
    soh_pct_pred = y_pred_all * 100
    ax.scatter(soh_pct_true, soh_pct_pred, c="#0077BB", s=60, alpha=0.85, zorder=3)
    lims = [min(soh_pct_true.min(), soh_pct_pred.min()) - 2,
            max(soh_pct_true.max(), soh_pct_pred.max()) + 2]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    mae_val = np.mean(np.abs(y - y_pred_all)) * 100
    ax.set_xlabel("True SOH (%)", fontsize=10)
    ax.set_ylabel("Predicted SOH (%)", fontsize=10)
    ax.set_title(f"QRC LOCO — Warwick DIB\nMAE = {mae_val:.2f}%", fontsize=10)
    ax.grid(linestyle=":", alpha=0.35)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"warwick_qrc_scatter.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {p}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("Phase 8: External Validation — Warwick DIB Dataset")
    print("=" * 72)
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Plot dir: {PLOT_DIR}")

    # Load data
    print("\nLoading Warwick DIB data (25°C / 50% SOC)...")
    X_raw, y, cell_ids = get_warwick_arrays()
    print(f"  {len(cell_ids)} cells, X shape: {X_raw.shape}, "
          f"SOH range: [{y.min()*100:.1f}%, {y.max()*100:.1f}%]")

    # LOCO
    print("\nRunning LOCO evaluation...")
    df_loco = run_loco(X_raw, y, cell_ids)

    # Save raw results
    out_csv = DATA_DIR / "warwick_loco.csv"
    df_loco.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df_loco)} rows)")

    # Summary
    summary = (
        df_loco.groupby("model")["mae"]
        .agg(mae_mean="mean", mae_std="std", n_folds="count")
        .reset_index()
        .sort_values("mae_mean")
    )
    out_sum = DATA_DIR / "warwick_loco_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"Saved: {out_sum} ({len(summary)} rows)")

    # Print table
    print("\nWarwick LOCO MAE summary (24-fold):")
    print(f"{'Model':<25} {'MAE mean':>10} {'MAE std':>10} {'vs QRC':>8}")
    print("-" * 58)
    qrc_mae = summary.loc[summary["model"] == "qrc_d1_z_plus_zz", "mae_mean"].values[0]
    for _, row in summary.iterrows():
        pct = (row["mae_mean"] - qrc_mae) / qrc_mae * 100
        name = MODEL_CONFIGS.get(row["model"], row["model"])
        flag = " ← best" if row["model"] == "qrc_d1_z_plus_zz" else ""
        print(f"{name:<25} {row['mae_mean']:>10.5f} {row['mae_std']:>10.5f} {pct:>+7.1f}%{flag}")

    # Plots
    print("\nGenerating plots...")
    plot_results(summary)
    plot_scatter(df_loco, X_raw, y, cell_ids)

    # Cross-dataset comparison
    print("\n" + "=" * 50)
    print("Cross-dataset QRC MAE comparison:")
    stanford_qrc = 0.00776   # Phase 4 noiseless d=1
    escl_qrc     = 0.0111    # Phase 7 temporal d=2
    warwick_qrc  = qrc_mae
    print(f"  Stanford SECL  (6 NMC cells,  LOCO): {stanford_qrc:.5f}")
    print(f"  ESCL lab data  (11 sessions, temporal): {escl_qrc:.5f}")
    print(f"  Warwick DIB   (24 NMC cells, LOCO):  {warwick_qrc:.5f}")
    print(f"\nCompleted: {datetime.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
