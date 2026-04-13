"""Plotting utilities for phase 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .config import CELL_COLORS, CELL_IDS
from ..plot_constants import save_fig, tier1_rc


AGING_CMAP = plt.get_cmap("viridis_r")


def _aging_color(index: int, total: int):
    return AGING_CMAP(index / max(total - 1, 1))


def _cell_annotation(ax, cell_id: str, capacity_df: pd.DataFrame | None = None) -> None:
    temp = "?"
    if capacity_df is not None:
        cell_cap = capacity_df[capacity_df["cell_id"] == cell_id]
        if not cell_cap.empty:
            temp = f"{cell_cap['temperature_C'].iloc[0]:.0f}"
    ax.annotate(
        f"{cell_id} ({temp}C)",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=6,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85),
    )


def _aging_colorbar(fig, axes) -> None:
    sm = ScalarMappable(cmap=AGING_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=np.asarray(axes).ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.12,
        aspect=30,
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Fresh", "Aged"])
    cbar.ax.tick_params(labelsize=6)


def plot_nyquist_per_cell(
    eis_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(5.2, 3.4),
            gridspec_kw={"hspace": 0.35, "wspace": 0.35},
        )
        axes = np.asarray(axes)

        for idx, cell_id in enumerate(CELL_IDS):
            ax = axes[idx // 3, idx % 3]
            cell_eis = eis_df[eis_df["cell_id"] == cell_id]
            blocks = sorted(cell_eis["block_id"].unique())

            for block_index, block_id in enumerate(blocks):
                block_df = cell_eis[cell_eis["block_id"] == block_id]
                avg = block_df.groupby("frequency_Hz")[["re_Z_ohm", "im_Z_ohm"]]
                avg = avg.mean().sort_index(ascending=False)
                ax.plot(
                    avg["re_Z_ohm"] * 1000.0,
                    -avg["im_Z_ohm"] * 1000.0,
                    "-",
                    color=_aging_color(block_index, len(blocks)),
                    linewidth=0.6,
                    alpha=0.85,
                )

            _cell_annotation(ax, cell_id, capacity_df)
            if idx // 3 == 1:
                ax.set_xlabel("Re(Z) [mOhm]")
            else:
                ax.tick_params(labelbottom=False)
            if idx % 3 == 0:
                ax.set_ylabel("-Im(Z) [mOhm]")
            else:
                ax.tick_params(labelleft=False)

        fig.subplots_adjust(bottom=0.14)
        _aging_colorbar(fig, axes)
        save_fig(fig, output_dir, "nyquist_per_cell")


def plot_capacity_fade(
    capacity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))

        for cell_id in CELL_IDS:
            cell_df = capacity_df[capacity_df["cell_id"] == cell_id].sort_values("block_id")
            ax.plot(
                cell_df["block_id"],
                cell_df["soh_pct"],
                "o-",
                color=CELL_COLORS[cell_id],
                label=cell_id,
            )

        ax.set_xlabel("Block Index")
        ax.set_ylabel("SOH [%]")
        ax.set_ylim(bottom=87)
        ax.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        save_fig(fig, output_dir, "capacity_fade")


def plot_eis_evolution(
    eis_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(len(CELL_IDS), 2, figsize=(3.5, 7.6))

        for row, cell_id in enumerate(CELL_IDS):
            cell_eis = eis_df[eis_df["cell_id"] == cell_id]
            blocks = sorted(cell_eis["block_id"].unique())

            ax_re = axes[row, 0]
            ax_im = axes[row, 1]

            for block_index, block_id in enumerate(blocks):
                block_df = cell_eis[cell_eis["block_id"] == block_id]
                avg = block_df.groupby("frequency_Hz")[["re_Z_ohm", "im_Z_ohm"]]
                avg = avg.mean().sort_index()
                color = _aging_color(block_index, len(blocks))
                ax_re.semilogx(avg.index, avg["re_Z_ohm"] * 1000.0, color=color, linewidth=0.5, alpha=0.85)
                ax_im.semilogx(avg.index, -avg["im_Z_ohm"] * 1000.0, color=color, linewidth=0.5, alpha=0.85)

            _cell_annotation(ax_im, cell_id, capacity_df)

            if row == 0:
                ax_re.set_title("Re(Z)")
                ax_im.set_title("-Im(Z)")
            if row < len(CELL_IDS) - 1:
                ax_re.tick_params(labelbottom=False)
                ax_im.tick_params(labelbottom=False)
            else:
                ax_re.set_xlabel("Frequency [Hz]")
                ax_im.set_xlabel("Frequency [Hz]")

            ax_re.set_ylabel("mOhm")

        fig.tight_layout(h_pad=0.3, w_pad=0.4)
        fig.subplots_adjust(bottom=0.12)
        _aging_colorbar(fig, axes)
        save_fig(fig, output_dir, "eis_evolution")


def plot_cycling_summary(
    cycling_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(2, 3, figsize=(5.2, 2.8))
        axes = np.asarray(axes)

        for idx, cell_id in enumerate(CELL_IDS):
            ax = axes[idx // 3, idx % 3]
            cell_df = cycling_df[cycling_df["cell_id"] == cell_id].copy()
            cell_df = cell_df[cell_df["discharge_capacity_mAh"] > 0].reset_index(drop=True)

            ax.plot(
                cell_df.index,
                cell_df["discharge_capacity_mAh"],
                ".",
                color=CELL_COLORS[cell_id],
                markersize=1.0,
                alpha=0.6,
            )

            if cell_df.empty:
                ax.set_title(cell_id, fontsize=7, fontweight="bold")
            else:
                temp_c = cell_df["temperature_C"].iloc[0]
                ax.set_title(f"{cell_id} ({temp_c:.0f}C)", fontsize=7, fontweight="bold")

            if idx // 3 == 1:
                ax.set_xlabel("Cycle Index")
            else:
                ax.tick_params(labelbottom=False)
            if idx % 3 == 0:
                ax.set_ylabel("Disch. [mAh]")
            else:
                ax.tick_params(labelleft=False)

        fig.tight_layout(h_pad=0.4, w_pad=0.3)
        save_fig(fig, output_dir, "cycling_summary")


def plot_r0_evolution(
    eis_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))

        for cell_id in CELL_IDS:
            cell_eis = eis_df[eis_df["cell_id"] == cell_id]
            blocks = sorted(cell_eis["block_id"].unique())
            r0_values = []

            for block_id in blocks:
                block_df = cell_eis[cell_eis["block_id"] == block_id]
                max_freq = block_df["frequency_Hz"].max()
                r0 = block_df[block_df["frequency_Hz"] == max_freq]["re_Z_ohm"].mean()
                r0_values.append(r0 * 1000.0)

            ax.plot(blocks, r0_values, "o-", color=CELL_COLORS[cell_id], label=cell_id)

        ax.set_xlabel("Block Index")
        ax.set_ylabel("R0 [mOhm]")
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        save_fig(fig, output_dir, "r0_evolution")
