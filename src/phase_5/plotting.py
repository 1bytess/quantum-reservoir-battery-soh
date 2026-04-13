"""Plotting for Phase 5: Noise ablation + stochastic resonance."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Phase5LabPaths
from ..plot_constants import tier1_rc, save_fig, DPI, MODEL_COLORS

DEFAULT_STAGE = "stage_1"


def plot_noise_sweep(paths: Phase5LabPaths = None) -> None:
    """Plot MAE vs noise level for each classical model."""
    if paths is None:
        paths = Phase5LabPaths(DEFAULT_STAGE)

    df = pd.read_csv(paths.data_dir / "noise_ablation.csv")
    models = df["model"].unique()
    colors = plt.cm.Set1(np.linspace(0, 0.5, len(models)))

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        for i, model in enumerate(models):
            model_df = df[df["model"] == model]
            summary = model_df.groupby("noise_level")["mae"].agg(["mean", "std"]).reset_index()

            ax.errorbar(
                summary["noise_level"], summary["mean"], yerr=summary["std"],
                fmt="o-", label=model, capsize=2,
                color=colors[i], linewidth=0.9, markersize=3, elinewidth=0.5,
            )

        ax.set_xlabel("Noise Level (fraction of std)")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("Classical Noise Sensitivity", fontweight="bold", pad=3)
        ax.legend(loc="best")

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "noise_sweep")


def plot_noise_vs_qrc(paths: Phase5LabPaths = None) -> None:
    """Overlay QRC performance on noise sweep."""
    if paths is None:
        paths = Phase5LabPaths(DEFAULT_STAGE)

    noise_path = paths.data_dir / "noise_ablation.csv"
    qrc_path = paths.results_dir.parent / "phase_4" / "data" / "qrc_results.csv"

    if not noise_path.exists() or not qrc_path.exists():
        print("  Missing data files, skipping noise_vs_qrc plot")
        return

    noise_df = pd.read_csv(noise_path)
    qrc_df = pd.read_csv(qrc_path)

    loco_qrc = qrc_df[qrc_df["regime"] == "loco"]
    if loco_qrc.empty:
        return
    qrc_best_mae = loco_qrc.groupby("depth")["mae"].mean().min()

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        for model in noise_df["model"].unique():
            model_df = noise_df[noise_df["model"] == model]
            summary = model_df.groupby("noise_level")["mae"].agg(["mean"]).reset_index()
            ax.plot(summary["noise_level"], summary["mean"], "o-", label=model, markersize=3)

        ax.axhline(qrc_best_mae, color="red", linestyle="--", linewidth=0.8,
                   label="QRC (best)", alpha=0.8)

        ax.set_xlabel("Classical Noise Level (fraction of std)")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("Noise Sensitivity: Classical vs QRC", fontweight="bold", pad=3)
        ax.legend(loc="best")

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "noise_vs_qrc")


# =========================================================================
# Stochastic resonance plot
# =========================================================================

def plot_stochastic_resonance(paths: Phase5LabPaths = None) -> None:
    """1x3 panel: one per noise channel, U-curve with error bars.

    X-axis: symlog noise rate.  Dashed line: noiseless baseline (rate=0).
    """
    if paths is None:
        paths = Phase5LabPaths(DEFAULT_STAGE)

    sr_p = paths.data_dir / "stochastic_resonance.csv"
    if not sr_p.exists():
        print("  No stochastic_resonance.csv, skipping")
        return

    df = pd.read_csv(sr_p)
    channels = df["channel"].unique()
    n_panels = len(channels)

    channel_colors = {
        "depolarizing": MODEL_COLORS.get("qrc_noiseless", "#D55E00"),
        "amplitude_damping": MODEL_COLORS.get("temporal_qrc", "#332288"),
        "phase_damping": MODEL_COLORS.get("qrc_expanded", "#AA4499"),
    }

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 2.2), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for idx, channel in enumerate(channels):
            ax = axes[idx]
            ch_df = df[df["channel"] == channel]

            # Noiseless baseline (rate == 0)
            baseline_df = ch_df[ch_df["rate"] == 0.0]
            if not baseline_df.empty:
                baseline_mae = baseline_df["mae"].mean()
                ax.axhline(baseline_mae, color="black", ls="--", lw=0.5,
                           alpha=0.6, label="Noiseless")

            # Non-zero rates
            nonzero = ch_df[ch_df["rate"] > 0]
            if nonzero.empty:
                continue

            summary = nonzero.groupby("rate")["mae"].agg(
                ["mean", "std"]).reset_index()

            color = channel_colors.get(channel, "#999999")
            ax.errorbar(
                summary["rate"], summary["mean"], yerr=summary["std"],
                fmt="o-", capsize=2, color=color,
                linewidth=0.9, markersize=3, elinewidth=0.5,
            )
            ax.set_xscale("symlog", linthresh=1e-4)
            ax.set_xlabel("Error rate")
            if idx == 0:
                ax.set_ylabel("MAE (SOH)")

            title = channel.replace("_", " ").title()
            ax.set_title(title, fontweight="bold", pad=3, fontsize=6.5)
            ax.legend(loc="best", fontsize=5)

        fig.tight_layout(w_pad=0.4)
        save_fig(fig, paths.plots_dir, "stochastic_resonance")
