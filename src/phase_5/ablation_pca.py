"""PCA ablation: compare in-fold PCA vs global PCA vs raw features.
   Also sweeps N_PCA components to justify the choice of 6.

Reviewer §5.1.5:
    "You chose N_PCA=6 but never justify it. Add an ablation over
    N_PCA ∈ {4, 6, 8} for both classical and QRC. If QRC is robust to
    N_PCA but XGBoost is sensitive, that supports the reservoir claim."

Two ablation axes are now combined here:
  Axis A — PCA mode (in_fold vs global vs raw_72d):
      Quantifies leakage impact on model performance.

  Axis B — N_PCA component count (4, 6, 8, 10, 12):
      Shows model sensitivity to dimensionality reduction depth.
      Critical for justifying N_PCA=6 in the paper.
      QRC is expected to be more robust (reservoir decouples encoding from readout).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_3.models import get_model_pipeline
from phase_3.evaluation import compute_metrics
from phase_4.qrc_model import QuantumReservoir
import phase_4.config as phase4_config
import phase_4.qrc_model as phase4_qrc_model
from contextlib import contextmanager
from config import RANDOM_STATE

# ── defaults — shared with phase_5 config ────────────────────────────────────
try:
    from .config import CELL_IDS
except ImportError:
    CELL_IDS = ["W3", "W8", "W9", "W10", "V4", "V5"]

PCA_RANDOM_STATE = RANDOM_STATE
N_PCA_DEFAULT    = 6

# Axis B sweep — reviewer wants [4, 6, 8]; extended to full range for thorough ablation
N_PCA_SWEEP      = [2, 4, 6, 8, 10, 12]

ABLATION_MODELS  = ["ridge", "svr", "xgboost", "qrc"]
CLASSICAL_MODELS = ["ridge", "svr", "xgboost"]


# ── Helpers ────────────────────────────────────────────────────────────────────

@contextmanager
def _patched_seed(seed: int):
    old_qrc = phase4_qrc_model.RANDOM_STATE
    old_cfg = phase4_config.RANDOM_STATE
    phase4_qrc_model.RANDOM_STATE = seed
    phase4_config.RANDOM_STATE    = seed
    try:
        yield
    finally:
        phase4_qrc_model.RANDOM_STATE = old_qrc
        phase4_config.RANDOM_STATE    = old_cfg


def _fit_pca_in_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = N_PCA_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training data only, transform both train and test (leakage-free)."""
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _raw_key(cell_data: Dict[str, Dict], cell: str) -> str:
    """Return the correct raw feature key (X_72d or X_raw)."""
    return "X_72d" if "X_72d" in cell_data[cell] else "X_raw"


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_dir / stem}.png")


# ── Axis A: PCA mode ablation ─────────────────────────────────────────────────

def run_pca_mode_ablation(
    cell_data: Dict[str, Dict],
    models: List[str] = CLASSICAL_MODELS,
    cell_ids: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compare three PCA modes across LOCO evaluation.

    Modes:
      - "in_fold":  PCA fit on training cells only (leakage-free, correct)
      - "global":   PCA fit on all cells (pre-computed X_6d, leaky)
      - "raw":      No PCA — raw feature vector used directly

    Args:
        cell_data: Per-cell EIS data with X_6d, X_raw/X_72d, y keys.
        models:    Model names to ablate (classical only — QRC always uses in-fold).
        cell_ids:  Override default CELL_IDS.
        output_dir: Where to save results CSV + summary print.

    Returns:
        DataFrame with per-cell MAE for each mode and leakage delta.
    """
    ids = cell_ids or CELL_IDS
    results = []

    for model_name in models:
        for test_cell in ids:
            train_cells = [c for c in ids if c != test_cell]
            rk = _raw_key(cell_data, test_cell)

            X_train_raw = np.vstack([cell_data[c][rk] for c in train_cells])
            y_train     = np.concatenate([cell_data[c]["y"] for c in train_cells])
            X_test_raw  = cell_data[test_cell][rk]
            y_test      = cell_data[test_cell]["y"]

            # Mode 1: In-fold PCA (correct, leakage-free)
            X_tr_if, X_te_if = _fit_pca_in_fold(X_train_raw, X_test_raw)
            m = get_model_pipeline(model_name)
            m.fit(X_tr_if, y_train)
            mae_infold = mean_absolute_error(y_test, m.predict(X_te_if))

            # Mode 2: Global PCA (leaky — uses pre-computed X_6d if available)
            if "X_6d" in cell_data[test_cell]:
                X_train_gl = np.vstack([cell_data[c]["X_6d"] for c in train_cells])
                X_test_gl  = cell_data[test_cell]["X_6d"]
                m = get_model_pipeline(model_name)
                m.fit(X_train_gl, y_train)
                mae_global = mean_absolute_error(y_test, m.predict(X_test_gl))
            else:
                mae_global = float("nan")

            # Mode 3: Raw (no PCA)
            m = get_model_pipeline(model_name)
            m.fit(X_train_raw, y_train)
            mae_raw = mean_absolute_error(y_test, m.predict(X_test_raw))

            results.append({
                "model":           model_name,
                "test_cell":       test_cell,
                "mae_in_fold_pca": mae_infold,
                "mae_global_pca":  mae_global,
                "mae_raw":         mae_raw,
                "leakage_delta":   mae_infold - mae_global,   # positive = in-fold harder (expected)
                "pca_benefit":     mae_raw - mae_infold,      # positive = PCA helps
            })

    df = pd.DataFrame(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "ablation_pca_modes.csv", index=False)
        print(f"\n  Saved: {output_dir / 'ablation_pca_modes.csv'}")

        print("\n  ┌──────────────────────────────────────────────────────┐")
        print("  │  PCA Mode Ablation Summary (macro MAE ± std)        │")
        print("  └──────────────────────────────────────────────────────┘")
        for mn in models:
            sub = df[df["model"] == mn]
            print(f"\n  {mn.upper()}:")
            print(f"    In-fold PCA:  {sub['mae_in_fold_pca'].mean():.4f} ± {sub['mae_in_fold_pca'].std():.4f}")
            if not sub["mae_global_pca"].isna().all():
                print(f"    Global PCA:   {sub['mae_global_pca'].mean():.4f} ± {sub['mae_global_pca'].std():.4f}")
                print(f"    Leakage Δ:    {sub['leakage_delta'].mean():+.4f}  (positive = in-fold harder)")
            print(f"    Raw (no PCA): {sub['mae_raw'].mean():.4f} ± {sub['mae_raw'].std():.4f}")
            print(f"    PCA benefit:  {sub['pca_benefit'].mean():+.4f}  (positive = PCA helps)")

    return df


# ── Axis B: N_PCA component count sweep ──────────────────────────────────────

def run_npca_sweep(
    cell_data: Dict[str, Dict],
    models: List[str] = ABLATION_MODELS,
    n_pca_values: List[int] = N_PCA_SWEEP,
    cell_ids: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    qrc_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Sweep N_PCA ∈ N_PCA_SWEEP for both classical models and QRC.

    For each (model, n_pca, test_cell):
      - Fit in-fold PCA with n_pca components on training cells
      - Fit model on PCA-projected training features
      - Evaluate on held-out test cell

    Key reviewer concern: "If QRC is robust to N_PCA but XGBoost is sensitive,
    that demonstrates the reservoir adds invariance to the compression level."

    Args:
        cell_data:    Per-cell EIS data.
        models:       Models to sweep (include "qrc" for quantum comparison).
        n_pca_values: List of N_PCA values to test.
        cell_ids:     Override default CELL_IDS.
        output_dir:   Where to save results.
        qrc_seed:     RNG seed for QRC circuit parameter initialisation.

    Returns:
        DataFrame with columns: model, n_pca, test_cell, mae, mae_pct.
    """
    ids = cell_ids or CELL_IDS
    rows = []

    for n_pca in n_pca_values:
        print(f"\n  N_PCA={n_pca}:")
        for model_name in models:
            maes = []
            for test_cell in ids:
                train_cells = [c for c in ids if c != test_cell]
                rk = _raw_key(cell_data, test_cell)

                X_train_raw = np.vstack([cell_data[c][rk] for c in train_cells])
                y_train     = np.concatenate([cell_data[c]["y"] for c in train_cells])
                X_test_raw  = cell_data[test_cell][rk]
                y_test      = cell_data[test_cell]["y"]

                X_train_pca, X_test_pca = _fit_pca_in_fold(
                    X_train_raw, X_test_raw, n_components=n_pca,
                )
                actual_n_pca = X_train_pca.shape[1]  # may be capped by n_train

                try:
                    if model_name == "qrc":
                        with _patched_seed(qrc_seed):
                            qrc = QuantumReservoir(
                                depth=1, use_zz=True, observable_set="Z",
                                add_random_rotations=True,
                            )
                            # QRC circuit width is fixed at N_QUBITS (6) regardless of
                            # the ablation N_PCA value. Alignment strategy:
                            #   N_PCA < N_QUBITS: zero-pad the extra dimensions
                            #     (neutral initialisation — unused angles stay 0).
                            #   N_PCA > N_QUBITS: truncate to the first N_QUBITS PCs
                            #     (drops the least-variance components beyond circuit width).
                            # Both branches preserve the in-fold PCA basis; no leakage.
                            n_qubits = phase4_config.N_QUBITS
                            if actual_n_pca < n_qubits:
                                pad = np.zeros((X_train_pca.shape[0], n_qubits - actual_n_pca))
                                X_train_pca = np.hstack([X_train_pca, pad])
                                pad_te = np.zeros((X_test_pca.shape[0], n_qubits - actual_n_pca))
                                X_test_pca = np.hstack([X_test_pca, pad_te])
                            elif actual_n_pca > n_qubits:
                                X_train_pca = X_train_pca[:, :n_qubits]
                                X_test_pca  = X_test_pca[:, :n_qubits]

                            qrc.fit(X_train_pca, y_train)
                            y_pred = qrc.predict(X_test_pca)
                    else:
                        m = get_model_pipeline(model_name)
                        m.fit(X_train_pca, y_train)
                        y_pred = m.predict(X_test_pca)

                    mae = mean_absolute_error(y_test, y_pred)
                except Exception as e:
                    print(f"    WARN: {model_name} n_pca={n_pca} cell={test_cell}: {e}")
                    mae = float("nan")

                maes.append(mae)
                rows.append({
                    "model":      model_name,
                    "n_pca":      n_pca,
                    "actual_dim": actual_n_pca,
                    "test_cell":  test_cell,
                    "mae":        mae,
                    "mae_pct":    mae * 100.0,
                })

            macro = np.nanmean(maes)
            print(f"    {model_name:12s}: macro MAE={macro:.4f} ({macro*100:.2f}%)")

    df = pd.DataFrame(rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "ablation_npca_sweep.csv", index=False)
        print(f"\n  Saved: {output_dir / 'ablation_npca_sweep.csv'}")

    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "qrc":      "#1f77b4",
    "xgboost":  "#ff7f0e",
    "ridge":    "#2ca02c",
    "svr":      "#d62728",
    "esn":      "#9467bd",
    "gp":       "#8c564b",
}


def plot_npca_sensitivity(
    results: pd.DataFrame,
    plot_dir: Path,
    default_n_pca: int = N_PCA_DEFAULT,
) -> None:
    """Line plot of macro MAE vs N_PCA, one line per model.

    Highlights N_PCA=6 (paper choice) with a vertical dashed line.
    QRC's line is expected to be flatter (more robust to compression level).
    """
    summary = (
        results.groupby(["model", "n_pca"])["mae_pct"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, grp in summary.groupby("model"):
        color = COLORS.get(model_name, None)
        lw    = 2.5 if model_name == "qrc" else 1.5
        ls    = "-"  if model_name == "qrc" else "--"
        ax.plot(grp["n_pca"], grp["mean"], label=model_name,
                color=color, linewidth=lw, linestyle=ls, marker="o", markersize=5)
        ax.fill_between(
            grp["n_pca"],
            grp["mean"] - grp["std"],
            grp["mean"] + grp["std"],
            alpha=0.15, color=color,
        )

    ax.axvline(default_n_pca, color="gray", linestyle=":", linewidth=1.5,
               label=f"Paper choice (N_PCA={default_n_pca})")
    ax.set_xlabel("Number of PCA components (N_PCA)")
    ax.set_ylabel("Macro MAE (%)")
    ax.set_title("Sensitivity to N_PCA: QRC vs classical baselines\n"
                 "(flatter = more robust to compression level)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(sorted(results["n_pca"].unique()))
    fig.tight_layout()
    _save_figure(fig, plot_dir, "ablation_npca_sensitivity")


def plot_pca_mode_comparison(results_mode: pd.DataFrame, plot_dir: Path) -> None:
    """Grouped bar chart: in-fold vs global vs raw for each model."""
    models = results_mode["model"].unique()
    x      = np.arange(len(models))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2), 5))
    for i, (col, label, color) in enumerate([
        ("mae_in_fold_pca", "In-fold PCA (correct)", "#1f77b4"),
        ("mae_global_pca",  "Global PCA (leaky)",    "#ff7f0e"),
        ("mae_raw",         "Raw (no PCA)",           "#2ca02c"),
    ]):
        vals = [results_mode[results_mode["model"] == m][col].mean() * 100
                for m in models]
        errs = [results_mode[results_mode["model"] == m][col].std() * 100
                for m in models]
        ax.bar(x + (i - 1) * width, vals, width * 0.9, yerr=errs, capsize=4,
               label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("MAE (%)")
    ax.set_title("PCA mode ablation: leakage impact per model")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, plot_dir, "ablation_pca_mode_comparison")


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_full_pca_ablation(
    cell_data: Dict[str, Dict],
    output_dir: Path,
    plot_dir: Path,
    models: List[str] = ABLATION_MODELS,
    n_pca_values: List[int] = N_PCA_SWEEP,
    cell_ids: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run both PCA ablation axes and generate all plots.

    Returns:
        {"mode_ablation": df_mode, "npca_sweep": df_sweep}
    """
    print("\n=== PCA Ablation: Axis A — Mode (in-fold vs global vs raw) ===")
    df_mode = run_pca_mode_ablation(
        cell_data,
        models=[m for m in models if m != "qrc"],
        cell_ids=cell_ids,
        output_dir=output_dir,
    )

    print("\n=== PCA Ablation: Axis B — N_PCA sweep ===")
    df_sweep = run_npca_sweep(
        cell_data,
        models=models,
        n_pca_values=n_pca_values,
        cell_ids=cell_ids,
        output_dir=output_dir,
    )

    print("\n=== Generating PCA ablation plots ===")
    plot_pca_mode_comparison(df_mode, plot_dir)
    plot_npca_sensitivity(df_sweep, plot_dir)

    return {"mode_ablation": df_mode, "npca_sweep": df_sweep}
