"""Phase 1: Stanford SECL data exploration and plotting.

Usage:
    python -m src.phase_1.run_phase_1_stanford
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from config import CELL_IDS, get_result_paths
from data_loader import load_stanford_data


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
    """Save figure as both PNG and PDF."""
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


def _plot_soh_curves(cell_data, plot_dir) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cell_id in CELL_IDS:
        blocks = cell_data[cell_id]["block_ids"]
        soh = 100.0 * cell_data[cell_id]["y"]
        ax.plot(blocks, soh, marker="o", linewidth=1.8, label=cell_id)

    ax.set_xlabel("Diagnostic Number")
    ax.set_ylabel("SOH (%)")
    ax.set_title("SOH Degradation Curves by Cell")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2)
    _save_figure(fig, plot_dir, "soh_degradation_curves")


def _plot_nyquist_per_cell(cell_data, plot_dir) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    global_blocks = np.concatenate([cell_data[c]["block_ids"] for c in CELL_IDS])
    norm = plt.Normalize(vmin=global_blocks.min(), vmax=global_blocks.max())

    for idx, cell_id in enumerate(CELL_IDS):
        ax = axes[idx]
        X = cell_data[cell_id]["X_raw"]
        blocks = cell_data[cell_id]["block_ids"]

        for i in range(X.shape[0]):
            re_part = X[i, :19]
            im_part = X[i, 19:]
            color = cm.viridis(norm(blocks[i]))
            ax.plot(re_part, -im_part, color=color, linewidth=1.0, alpha=0.9)

        ax.set_title(f"{cell_id} (n={X.shape[0]})")
        ax.set_xlabel("Re(Z) [Ohm]")
        ax.set_ylabel("-Im(Z) [Ohm]")
        ax.grid(True, linestyle="--", alpha=0.3)

    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.9)
    cbar.set_label("Diagnostic Number")
    fig.suptitle("Nyquist Plots per Cell (Color = Diagnostic Number)", y=1.02)
    _save_figure(fig, plot_dir, "nyquist_per_cell")


def _plot_nyquist_overlay(cell_data, plot_dir) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(CELL_IDS)))

    for color, cell_id in zip(colors, CELL_IDS):
        X = cell_data[cell_id]["X_raw"]
        for i in range(X.shape[0]):
            re_part = X[i, :19]
            im_part = X[i, 19:]
            ax.plot(re_part, -im_part, color=color, alpha=0.20, linewidth=0.8)

        mean_re = X[:, :19].mean(axis=0)
        mean_im = X[:, 19:].mean(axis=0)
        ax.plot(
            mean_re,
            -mean_im,
            color=color,
            linewidth=2.2,
            label=f"{cell_id} mean",
        )

    ax.set_xlabel("Re(Z) [Ohm]")
    ax.set_ylabel("-Im(Z) [Ohm]")
    ax.set_title("Nyquist Overlay Across Cells")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2)
    _save_figure(fig, plot_dir, "nyquist_overlay_all_cells")


def main() -> None:
    data_dir, plot_dir = get_result_paths(1)
    log_path = data_dir / "phase_1_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 1: Data Exploration (Stanford SECL)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")

        cell_data = load_stanford_data()

        print("\nSummary table")
        print("-" * 70)
        print(f"{'Cell':<6}{'N points':>10}{'SOH min (%)':>14}{'SOH max (%)':>14}")
        print("-" * 70)
        total_points = 0
        for cell_id in CELL_IDS:
            y = cell_data[cell_id]["y"]
            n_points = y.shape[0]
            total_points += n_points
            print(f"{cell_id:<6}{n_points:>10}{100.0*y.min():>14.2f}{100.0*y.max():>14.2f}")
        print("-" * 70)
        print(f"{'Total':<6}{total_points:>10}")

        print("\nGenerating plots...")
        _plot_soh_curves(cell_data, plot_dir)
        _plot_nyquist_per_cell(cell_data, plot_dir)
        _plot_nyquist_overlay(cell_data, plot_dir)

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
