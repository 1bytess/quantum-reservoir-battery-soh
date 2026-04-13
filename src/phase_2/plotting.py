"""Plotting for Phase 2 — Tier-1 single-column, viridis palette."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

from .config import CELL_IDS, N_QRC_INPUT
from ..plot_constants import tier1_rc, save_fig, DPI

_CMAP = plt.get_cmap("viridis")
CELL_COLORS_V = {cid: _CMAP(i / 3) for i, cid in enumerate(CELL_IDS)}


def plot_explained_variance(reducer, output_dir: Path) -> None:
    """PCA explained variance — bar + cumulative line."""
    if not hasattr(reducer, "explained_variance_ratio_"):
        print("  Skipping PCA variance plot (not PCA)")
        return

    var = reducer.explained_variance_ratio_
    cum_var = np.cumsum(var)
    cols = _CMAP(np.linspace(0.2, 0.8, len(var)))

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        ax.bar(range(1, len(var) + 1), var, color=cols, edgecolor="black",
               linewidth=0.4, alpha=0.85, label="Individual")
        ax.plot(range(1, len(cum_var) + 1), cum_var, "o-",
                color=_CMAP(0.9), markersize=3, label="Cumulative")
        ax.axhline(0.95, color="0.5", ls="--", lw=0.6, label="95%")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance")
        ax.legend(loc="center right")
        ax.set_xticks(range(1, len(var) + 1))
        fig.tight_layout()
        save_fig(fig, output_dir, "pca_explained_variance")


def plot_features_vs_soh(features_6d: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: each PC vs SOH, 2x3 grid."""
    pc_cols = [c for c in features_6d.columns if c.startswith("pc")]

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.8))
        axes = axes.ravel()

        for i, col in enumerate(pc_cols[:6]):
            ax = axes[i]
            for cid in CELL_IDS:
                mask = features_6d["cell_id"] == cid
                ax.scatter(
                    features_6d.loc[mask, col],
                    features_6d.loc[mask, "soh_pct"],
                    color=CELL_COLORS_V[cid],
                    label=cid if i == 0 else None,
                    s=12, alpha=0.85, edgecolors="white", linewidth=0.3,
                )
            ax.set_xlabel(col.upper())
            if i % 3 == 0:
                ax.set_ylabel("SOH [%]")
            else:
                ax.tick_params(labelleft=False)

        for j in range(len(pc_cols), 6):
            axes[j].set_visible(False)

        fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout(h_pad=0.5, w_pad=0.3)
        save_fig(fig, output_dir, "features_vs_soh")


def plot_feature_correlation(features_72d: pd.DataFrame, output_dir: Path) -> None:
    """EIS feature-SOH correlation bar chart."""
    feat_cols = [c for c in features_72d.columns
                 if c.startswith("re_f") or c.startswith("im_f")]
    corr = features_72d[feat_cols + ["soh_pct"]].corr()["soh_pct"].drop("soh_pct")

    n = len(corr)
    colors = [_CMAP(0.2) if c.startswith("re_") else _CMAP(0.7) for c in corr.index]

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 1.8))
        ax.bar(range(n), corr.values, color=colors, width=1.0, linewidth=0)
        ax.set_xlabel("EIS Feature Index")
        ax.set_ylabel("Pearson r (SOH)")
        ax.axhline(0, color="black", lw=0.4)
        ax.legend(
            handles=[Patch(color=_CMAP(0.2), label="Re(Z)"),
                     Patch(color=_CMAP(0.7), label="Im(Z)")],
        )
        fig.tight_layout()
        save_fig(fig, output_dir, "eis_soh_correlation")
