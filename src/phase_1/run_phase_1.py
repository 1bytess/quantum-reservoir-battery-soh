"""Orchestration script for phase 1."""

from __future__ import annotations

import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

from .config import CELL_IDS, Phase1LabPaths
from .data_loader import load_all_data, summarize_data
from .plotting import (
    plot_capacity_fade,
    plot_cycling_summary,
    plot_eis_evolution,
    plot_nyquist_per_cell,
    plot_r0_evolution,
)


class TeeLogger:
    """Tee stdout to both console and log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(
            message.encode(self.terminal.encoding or "utf-8", errors="replace").decode(
                self.terminal.encoding or "utf-8",
                errors="replace",
            )
        )
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main() -> None:
    paths = Phase1LabPaths()
    paths.ensure_dirs()

    log_path = paths.extracted_data_dir / "phase_1_log.txt"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    print("Phase 1: Data Loading and Visualization")
    print("=" * 60)

    data = load_all_data(paths.data_dir)
    summary = summarize_data(data)

    print("\n[1/3] Loaded data")
    print(summary)

    with open(paths.extracted_data_dir / "data_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(summary)

    eis = data["eis"]
    cap = data["capacity"]

    print("\n[2/3] Running data quality checks...")

    for cell_id in CELL_IDS:
        cell_eis = eis[eis["cell_id"] == cell_id]
        freqs_per_spectrum = cell_eis.groupby("spectrum_id")["frequency_Hz"].count()
        n_unique = freqs_per_spectrum.unique()
        if len(n_unique) == 1:
            print(
                f"  {cell_id}: all {cell_eis['spectrum_id'].nunique()} spectra have "
                f"{n_unique[0]} frequency points"
            )
        else:
            print(f"  {cell_id}: WARNING inconsistent frequency counts {n_unique}")

    for cell_id in CELL_IDS:
        cell_cap = cap[cap["cell_id"] == cell_id].sort_values("block_id")
        soh_values = cell_cap["soh_pct"].values
        non_monotonic = sum(
            1 for idx in range(1, len(soh_values)) if soh_values[idx] > soh_values[idx - 1]
        )
        if non_monotonic > 0:
            print(f"  {cell_id}: WARNING {non_monotonic} non-monotonic SOH transitions")
        else:
            print(f"  {cell_id}: SOH monotonically decreasing")

    for cell_id in CELL_IDS:
        eis_blocks = set(eis[eis["cell_id"] == cell_id]["block_id"].unique())
        cap_blocks = set(cap[cap["cell_id"] == cell_id]["block_id"].unique())
        if eis_blocks == cap_blocks:
            print(f"  {cell_id}: EIS blocks match capacity blocks")
            continue
        missing_eis = cap_blocks - eis_blocks
        missing_cap = eis_blocks - cap_blocks
        if missing_eis:
            print(f"  {cell_id}: WARNING blocks in capacity but not EIS: {missing_eis}")
        if missing_cap:
            print(f"  {cell_id}: INFO blocks in EIS but not capacity: {missing_cap}")

    print("\n[3/3] Generating plots...")
    plot_nyquist_per_cell(eis, cap, paths.plots_dir)
    plot_capacity_fade(cap, paths.plots_dir)
    plot_eis_evolution(eis, cap, paths.plots_dir)
    plot_cycling_summary(data["cycling_summary"], paths.plots_dir)
    plot_r0_evolution(eis, cap, paths.plots_dir)

    print("\n" + "=" * 60)
    print("Phase 1 complete")
    print("=" * 60)
    print(f"  Plots: {paths.plots_dir}")
    print(f"  Data:  {paths.extracted_data_dir}")
    print(f"  Finished at: {datetime.now().isoformat()}")

    sys.stdout = tee.terminal
    tee.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
