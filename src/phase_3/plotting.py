"""Plotting for Phase 3-Lab — Tier-1, dedicated model colours.

Reviewer §5.3 requested figures (new in this revision):
  - plot_parity():         Predicted vs actual SOH with 45° line and ±2% band
  - plot_per_cell_trajectories(): Per-cell SOH trajectories (true + predicted)
  - plot_nyquist_with_predictions(): EIS Nyquist coloured by predicted SOH
  - plot_metric_summary_table(): LaTeX-formatted metric table (MAE, RMSE, R²)

All four plots are added as standalone functions callable from any phase runner.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Phase3LabPaths, MODEL_NAMES
from ..plot_constants import (
    tier1_rc, save_fig, model_color, model_label, DPI,
)


def plot_loco_comparison(paths: Phase3LabPaths = None) -> None:
    """Grouped bar: LOCO MAE per model × test cell."""
    if paths is None:
        paths = Phase3LabPaths()

    df = pd.read_csv(paths.data_dir / "loco_results.csv")
    df["mae_capped"] = df["mae"].clip(upper=0.15)

    models = df["model"].unique()
    cells = df["test_cell"].unique()
    n_m = len(models)

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        width = 0.75 / n_m
        x = np.arange(len(cells))

        for i, m in enumerate(models):
            mdf = df[df["model"] == m]
            vals = [
                mdf.loc[mdf["test_cell"] == c, "mae_capped"].values[0]
                if c in mdf["test_cell"].values else 0
                for c in cells
            ]
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   color=model_color(m), edgecolor="black", linewidth=0.3,
                   label=model_label(m))

            # Mark clipped (catastrophic) bars
            for j, c in enumerate(cells):
                real = mdf.loc[mdf["test_cell"] == c, "mae"].values
                if len(real) > 0 and real[0] > 0.15:
                    ax.text(x[j] + offset, 0.152, "×",
                            ha="center", va="bottom", fontsize=5,
                            color="red", fontweight="bold")

        ax.set_xlabel("Test Cell")
        ax.set_ylabel("MAE (SOH)")
        ax.set_ylim(top=0.16)
        ax.set_title("LOCO Evaluation", fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cells)
        ax.legend(ncol=2, loc="upper left")

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "loco_comparison")


def plot_temporal_comparison(paths: Phase3LabPaths = None) -> None:
    """Grouped bar: temporal MAE per model × cell."""
    if paths is None:
        paths = Phase3LabPaths()

    df = pd.read_csv(paths.data_dir / "temporal_results.csv")
    # Keep only valid rows (n_train>=3 and mae<=1.0); fall back to mae<=1.0 for older CSVs.
    if "valid" in df.columns:
        df = df[df["valid"]]
    else:
        df = df[df["mae"] <= 1.0]
    models = df["model"].unique()
    cells = df["cell"].unique()
    n_m = len(models)

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        width = 0.75 / n_m
        x = np.arange(len(cells))

        for i, m in enumerate(models):
            mdf = df[df["model"] == m]
            vals = [
                mdf.loc[mdf["cell"] == c, "mae"].values[0]
                if c in mdf["cell"].values else 0
                for c in cells
            ]
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   color=model_color(m), edgecolor="black", linewidth=0.3,
                   label=model_label(m))

        # Persistence baseline
        persist = df.groupby("cell")["persist_mae"].first()
        for j, c in enumerate(cells):
            if c in persist.index:
                ax.plot([x[j] - 0.4, x[j] + 0.4], [persist[c], persist[c]],
                        "--", color="0.4", lw=0.6)

        ax.set_xlabel("Cell")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("Temporal Split", fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cells)
        ax.legend(ncol=2, loc="upper left")

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "temporal_comparison")


def plot_summary_table(paths: Phase3LabPaths = None) -> None:
    if paths is None:
        paths = Phase3LabPaths()

    loco = pd.read_csv(paths.data_dir / "loco_results.csv")
    temp = pd.read_csv(paths.data_dir / "temporal_results.csv")
    if "valid" in temp.columns:
        temp = temp[temp["valid"]]
    else:
        temp = temp[temp["mae"] <= 1.0]

    rows = []
    for m in loco["model"].unique():
        loco_mae = loco[loco["model"] == m]["mae"].mean()
        temp_mae = (temp[temp["model"] == m]["mae"].mean()
                    if m in temp["model"].values else float("nan"))
        rows.append({"model": m, "loco_mae": round(loco_mae, 4),
                      "temporal_mae": round(temp_mae, 4)})

    df = pd.DataFrame(rows)
    df.to_csv(paths.data_dir / "summary_table.csv", index=False)
    print(f"  Saved summary_table.csv")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer-requested plots (new in revision)
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_label_str: str,
    plot_dir: Path,
    stem: str = "parity",
    error_band_pct: float = 2.0,
    dataset_label: str = "",
) -> None:
    """Predicted vs actual SOH parity plot.

    Reviewer §5.3.2:
        "A parity plot (predicted vs actual SOH, with the 45° line) is the
        standard figure for regression tasks in battery papers. Its absence
        is a major gap."

    Features:
      - Scatter of predicted vs actual SOH (%)
      - 45° perfect-prediction line
      - Shaded ±error_band_pct% error band
      - R² and RMSE annotations in the corner

    Args:
        y_true:          True SOH fractions (0–1 scale).
        y_pred:          Predicted SOH fractions.
        model_label_str: Model name for title/legend.
        plot_dir:        Output directory.
        stem:            Output filename stem.
        error_band_pct:  Half-width of shaded error band in SOH percentage points.
        dataset_label:   Optional dataset name for subplot title.
    """
    from sklearn.metrics import r2_score, mean_squared_error

    y_true_pct = y_true * 100.0
    y_pred_pct = y_pred * 100.0
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100.0

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        lo = min(y_true_pct.min(), y_pred_pct.min()) - 1.0
        hi = max(y_true_pct.max(), y_pred_pct.max()) + 1.0
        lims = [lo, hi]

        # ±N% error band
        ax.fill_between(lims,
                        [l - error_band_pct for l in lims],
                        [l + error_band_pct for l in lims],
                        alpha=0.12, color="steelblue",
                        label=f"±{error_band_pct:.0f}% band")

        # 45° line
        ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect prediction")

        # Scatter
        ax.scatter(y_true_pct, y_pred_pct, s=18, alpha=0.75,
                   edgecolors="k", linewidths=0.3,
                   color=model_color(model_label_str),
                   label=model_label_str, zorder=4)

        # Annotation
        ax.text(0.05, 0.93,
                f"$R^2$={r2:.3f}\nRMSE={rmse:.2f}%",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True SOH (%)")
        ax.set_ylabel("Predicted SOH (%)")
        title = f"Parity — {model_label_str}"
        if dataset_label:
            title += f"\n({dataset_label})"
        ax.set_title(title, fontweight="bold", pad=3)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_aspect("equal")
        fig.tight_layout()
        save_fig(fig, plot_dir, stem)


def plot_per_cell_trajectories(
    cell_data: Dict[str, Dict],
    predictions: Dict[str, Dict[str, np.ndarray]],
    plot_dir: Path,
    stem: str = "per_cell_trajectories",
    cell_ids: Optional[List[str]] = None,
    dataset_label: str = "",
) -> None:
    """Per-cell SOH degradation trajectories with model predictions.

    Reviewer §5.3.3:
        "Show per-cell SOH prediction trajectories. This is standard in
        battery degradation papers and much more informative than a single
        aggregate MAE number."

    Args:
        cell_data:    {cell_id: {"y": array, "block_ids": array, ...}}
        predictions:  {model_name: {cell_id: y_pred array}}
        plot_dir:     Output directory.
        stem:         Filename stem.
        cell_ids:     Ordered list of cell IDs to plot.
        dataset_label: Optional dataset name for figure title.
    """
    ids   = cell_ids or sorted(cell_data.keys())
    n     = len(ids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.5, nrows * 2.2),
                                 sharey=True)
        axes_flat = np.array(axes).flatten()

        for ax_idx, cell in enumerate(ids):
            ax = axes_flat[ax_idx]
            data = cell_data[cell]
            y    = data["y"]

            # x-axis: block_ids if available, else sequential
            x = data.get("block_ids", np.arange(len(y)))

            # True SOH
            ax.plot(x, y * 100, "k-o", linewidth=1.2, markersize=3,
                    label="True SOH", zorder=5)

            # Model predictions
            for model_name, cell_preds in predictions.items():
                if cell in cell_preds:
                    y_pred = cell_preds[cell]
                    ax.plot(x, y_pred * 100, "--",
                            color=model_color(model_name),
                            linewidth=1.0, markersize=2,
                            label=model_label(model_name), zorder=4)

            ax.set_title(cell, fontweight="bold", fontsize=8, pad=2)
            ax.set_xlabel("Block" if "block_ids" in data else "Index", fontsize=7)
            if ax_idx % ncols == 0:
                ax.set_ylabel("SOH (%)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)

        # Turn off unused axes
        for ax_idx in range(len(ids), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        # Shared legend below figure
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=min(len(handles), 4), fontsize=7,
                       bbox_to_anchor=(0.5, -0.02))

        title = "Per-cell SOH Trajectories"
        if dataset_label:
            title += f" — {dataset_label}"
        fig.suptitle(title, fontweight="bold", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        save_fig(fig, plot_dir, stem)


def plot_nyquist_with_predictions(
    cell_data: Dict[str, Dict],
    y_pred: Dict[str, np.ndarray],
    plot_dir: Path,
    stem: str = "nyquist_with_predictions",
    model_label_str: str = "QRC",
    dataset_label: str = "",
) -> None:
    """Nyquist plots coloured by predicted SOH (per cell or per block).

    Reviewer §5.3.1:
        "Overlay QRC predictions on the Nyquist plot: colour each EIS spectrum
        by its predicted SOH value. This visualises whether the model has
        learned a sensible manifold in impedance space."

    Args:
        cell_data:     {cell_id: {"X_raw"/"X_72d": array, "y": array,
                                   "freq": array (Hz), ...}}
                       X_raw is [Re(Z)×F | Im(Z)×F] concatenated.
        y_pred:        {cell_id: ndarray (n_samples,)} — model predictions.
        plot_dir:      Output directory.
        stem:          Filename stem.
        model_label_str: Model name for colourbar label.
        dataset_label: Optional dataset name.
    """
    cell_ids = sorted(cell_data.keys())

    # Determine n_freq from first cell
    sample_data = next(iter(cell_data.values()))
    raw_key = "X_72d" if "X_72d" in sample_data else "X_raw"
    n_raw   = sample_data[raw_key].shape[1]
    n_freq  = n_raw // 2        # [Re×F | Im×F] → F

    # Gather all SOH values for colour normalisation
    all_soh = np.concatenate([
        y_pred.get(c, cell_data[c]["y"]) for c in cell_ids
    ])
    norm  = Normalize(vmin=all_soh.min() * 100, vmax=all_soh.max() * 100)
    cmap  = cm.get_cmap("RdYlGn")   # red=degraded, green=healthy

    ncols = min(3, len(cell_ids))
    nrows = int(np.ceil(len(cell_ids) / ncols))

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.5, nrows * 2.2))
        axes_flat = np.array(axes).flatten()

        for ax_idx, cell in enumerate(cell_ids):
            ax   = axes_flat[ax_idx]
            data = cell_data[cell]
            X    = data[raw_key]   # (n_samples, 2*F)
            soh  = y_pred.get(cell, data["y"])  # predicted SOH fractions

            # Plot each EIS spectrum as a coloured Nyquist curve
            for i in range(len(X)):
                re = X[i, :n_freq]
                im = -X[i, n_freq:]   # convention: Im(Z) positive in impedance space
                color = cmap(norm(float(soh[i]) * 100))
                ax.plot(re, im, "-", linewidth=0.7, alpha=0.8, color=color)

            ax.set_xlabel("Re(Z) [Ω]", fontsize=7)
            if ax_idx % ncols == 0:
                ax.set_ylabel("−Im(Z) [Ω]", fontsize=7)
            ax.set_title(cell, fontweight="bold", fontsize=8, pad=2)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.2)

        # Turn off unused axes
        for ax_idx in range(len(cell_ids), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        # Colourbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes_flat[:len(cell_ids)],
                            orientation="vertical", pad=0.02, shrink=0.8)
        cbar.set_label(f"Predicted SOH % ({model_label_str})", fontsize=8)

        title = "Nyquist Plot — EIS coloured by predicted SOH"
        if dataset_label:
            title += f" ({dataset_label})"
        fig.suptitle(title, fontweight="bold", fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        save_fig(fig, plot_dir, stem)


def plot_metric_summary_table(
    results_loco: pd.DataFrame,
    plot_dir: Path,
    stem: str = "metric_summary_table",
    metric_cols: Optional[List[str]] = None,
    dataset_label: str = "",
) -> None:
    """Render a LaTeX-style metric table as a matplotlib figure.

    Reviewer §5.3.4:
        "Your table only shows MAE. Include RMSE and R² for Nature Comms
        standard, and show MAE% (percentage of nominal capacity)."

    Args:
        results_loco:  LOCO results DataFrame (must have 'model', 'mae',
                       'rmse', 'r2', 'mae_pct' columns).
        plot_dir:      Output directory.
        stem:          Filename stem.
        metric_cols:   Columns to include in table (default: mae, rmse, r2, mae_pct).
        dataset_label: Optional dataset name for table title.
    """
    if metric_cols is None:
        metric_cols = ["mae", "rmse", "r2", "mae_pct"]

    # Build summary (macro mean ± std per model)
    agg = (
        results_loco.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )

    # Flatten multi-level columns
    col_headers = []
    cell_data_table = []
    for col in metric_cols:
        mu  = agg[col]["mean"]
        std = agg[col]["std"]
        col_headers.append(col.upper().replace("_", "\n"))
        cell_data_table.append([f"{m:.4f}\n±{s:.4f}" for m, s in zip(mu, std)])

    row_labels = agg.index.tolist()
    n_rows, n_cols = len(row_labels), len(metric_cols)

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(max(5, n_cols * 1.4), max(2, n_rows * 0.55 + 0.8)))
        ax.axis("off")

        tbl = ax.table(
            cellText=list(zip(*cell_data_table)),
            rowLabels=row_labels,
            colLabels=col_headers,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.6)

        # Style header row
        for j in range(n_cols):
            tbl[(0, j)].set_facecolor("#dce8f5")
            tbl[(0, j)].set_text_props(fontweight="bold")

        # Highlight best (min MAE, min RMSE, max R²)
        best_mae_row  = agg["mae"]["mean"].values.argmin()   if "mae"  in metric_cols else -1
        best_r2_row   = agg["r2"]["mean"].values.argmax()    if "r2"   in metric_cols else -1
        mae_col_idx   = metric_cols.index("mae")             if "mae"  in metric_cols else -1
        r2_col_idx    = metric_cols.index("r2")              if "r2"   in metric_cols else -1
        if mae_col_idx >= 0 and best_mae_row >= 0:
            tbl[(best_mae_row + 1, mae_col_idx)].set_facecolor("#c8e6c9")
        if r2_col_idx >= 0 and best_r2_row >= 0:
            tbl[(best_r2_row + 1, r2_col_idx)].set_facecolor("#c8e6c9")

        title = "LOCO Metric Summary (MAE, RMSE, R², MAE%)"
        if dataset_label:
            title += f" — {dataset_label}"
        ax.set_title(title, fontweight="bold", fontsize=9, pad=6)
        fig.tight_layout()
        save_fig(fig, plot_dir, stem)


def main(paths: Phase3LabPaths = None) -> None:
    if paths is None:
        paths = Phase3LabPaths()
    paths.ensure_dirs()
    plot_loco_comparison(paths)
    plot_temporal_comparison(paths)
    plot_summary_table(paths)

    # Load results and render new reviewer-requested summary table
    try:
        loco_df = pd.read_csv(paths.data_dir / "loco_results.csv")
        available_metrics = [
            c for c in ["mae", "rmse", "r2", "mae_pct", "rmse_pct"]
            if c in loco_df.columns
        ]
        if available_metrics:
            plot_metric_summary_table(
                loco_df, paths.plots_dir,
                metric_cols=available_metrics,
            )
    except FileNotFoundError:
        pass
