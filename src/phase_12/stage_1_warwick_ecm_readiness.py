"""Phase 12 Stage 1: Audit Warwick EIS readiness for ECM-style baselines."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_12.config import get_stage_paths
from phase_12.ecm_features import build_readiness_table, load_warwick_impedance_records


class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def _save_figure(fig: plt.Figure, plot_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / stem}.png")


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_1")
    logger = TeeLogger(data_dir / "stage_1_log.txt")
    old_stdout = sys.stdout
    sys.stdout = logger
    try:
        print(f"Started: {datetime.now().isoformat()}")
        records = load_warwick_impedance_records()
        readiness = build_readiness_table(records)
        readiness.to_csv(data_dir / "warwick_ecm_readiness.csv", index=False)

        ref_freq = pd.DataFrame({"freq_hz": records[0]["freq"]})
        ref_freq.to_csv(data_dir / "warwick_frequency_grid.csv", index=False)

        n_cells = len(records)
        n_freq = int(readiness["n_freq"].iloc[0]) if not readiness.empty else 0
        same_grid = bool(readiness["freq_matches_reference"].all()) if not readiness.empty else False
        descending = bool(readiness["freq_descending"].all()) if not readiness.empty else False

        summary = [
            "# Warwick ECM readiness audit",
            "",
            f"- Cells audited: {n_cells}",
            f"- Frequency points per cell: {n_freq}",
            f"- Shared frequency grid across cells: {'yes' if same_grid else 'no'}",
            f"- Frequency order descending for all cells: {'yes' if descending else 'no'}",
            f"- Median ohmic intercept (Ohm): {readiness['r_ohm_ohm'].median():.5f}",
            f"- Median low-frequency real impedance (Ohm): {readiness['r_lowfreq_ohm'].median():.5f}",
            f"- Median apparent resistance span (Ohm): {readiness['delta_r_ohm'].median():.5f}",
            "",
            "Interpretation: Warwick is structurally suitable for an ECM-inspired cross-cell baseline.",
            "This phase uses the shared 25degC / 50SOC EIS condition as the starting point.",
        ]
        (data_dir / "stage_1_summary.md").write_text("\n".join(summary), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(7, 5))
        for rec in records:
            ax.plot(rec["re_z"], -rec["im_z"], alpha=0.55, linewidth=1.0)
        ax.set_xlabel("Re(Z) [Ohm]")
        ax.set_ylabel("-Im(Z) [Ohm]")
        ax.set_title("Warwick 25degC / 50SOC Nyquist overlay")
        ax.grid(True, alpha=0.3)
        _save_figure(fig, plot_dir, "warwick_nyquist_overlay")

        print(f"Audited {n_cells} cells with {n_freq} frequency points each.")
        print(f"Shared frequency grid: {same_grid}")
        print(f"Descending frequency order: {descending}")
    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    main()
