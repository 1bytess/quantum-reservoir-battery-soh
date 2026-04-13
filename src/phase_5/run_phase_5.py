"""Orchestration script for Phase 5-Lab: Ablation Study.

Usage:
    # Classical noise ablation (original)
    python -m src.phase_5.run_phase_5

    # Stochastic resonance sweep (quantum noise channels)
    python -m src.phase_5.run_phase_5 --stochastic-resonance
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.phase_5.config import Phase5LabPaths
from src.phase_3.data_loader import get_cell_data
from src.phase_3.config import Phase3LabPaths

STAGE_NAME = "stage_1"


class TeeLogger:
    """Tee stdout to both console and log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message.encode(
            self.terminal.encoding or "utf-8", errors="replace"
        ).decode(self.terminal.encoding or "utf-8", errors="replace"))
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def main(run_stochastic_resonance: bool = False):
    """Run Phase 5-Lab ablation study."""
    paths = Phase5LabPaths(STAGE_NAME)
    paths.ensure_dirs()

    # Setup logging
    log_path = paths.data_dir / "stage_1_log.txt"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    print("Phase 5 Stage 1: Ablation Study (Noise Sensitivity)")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading features...")
    phase3_paths = Phase3LabPaths()
    cell_data = get_cell_data(phase3_paths)

    for cid, data in cell_data.items():
        print(f"  {cid}: {data['X_6d'].shape[0]} blocks")

    # Run noise ablation (classical)
    from src.phase_5.ablation_noise import run_noise_ablation
    print("\n[2/3] Running noise ablation...")
    results = run_noise_ablation(cell_data)
    results.to_csv(paths.data_dir / "noise_ablation.csv", index=False)
    print(f"\n  Saved noise_ablation.csv ({len(results)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("NOISE ABLATION SUMMARY")
    print("=" * 60)
    summary = results.groupby(["model", "noise_level"])["mae"].mean().unstack()
    print(summary.round(4))

    # Stochastic resonance sweep (quantum noise channels)
    sr_df = None
    if run_stochastic_resonance:
        print("\n[2b/3] Running stochastic resonance sweep...")
        from src.phase_5.stochastic_resonance import run_stochastic_resonance as _run_sr
        sr_df = _run_sr(cell_data, paths)

    # Generate plots
    print("\n[3/3] Generating plots...")
    from src.phase_5.plotting import (
        plot_noise_sweep, plot_noise_vs_qrc, plot_stochastic_resonance,
    )
    plot_noise_sweep(paths)
    plot_noise_vs_qrc(paths)
    if sr_df is not None:
        plot_stochastic_resonance(paths)

    print("\n" + "=" * 60)
    print("Phase 5 Stage 1 Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Results: {paths.data_dir}")
    print(f"  Plots:   {paths.plots_dir}")
    print(f"  Finished at: {datetime.now().isoformat()}")

    sys.stdout = tee.terminal
    tee.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5-Lab: Ablation Study")
    parser.add_argument("--stochastic-resonance", action="store_true",
                        help="Run quantum noise channel sweep (depolarizing, "
                             "amplitude damping, phase damping)")
    args = parser.parse_args()
    main(run_stochastic_resonance=args.stochastic_resonance)
