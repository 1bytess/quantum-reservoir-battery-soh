"""Orchestration script for Phase 3-Lab: Classical Baselines on EIS.

Usage:
    # Full run (all models from MODEL_NAMES):
    python -m src.phase_3.run_phase_3

    # Add only GP and CNN1D to an existing run (incremental, no recompute):
    python -m src.phase_3.run_phase_3 --models gp cnn1d --incremental

    # Dry-run a single model from scratch:
    python -m src.phase_3.run_phase_3 --models ridge
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.phase_3.config import Phase3LabPaths, MODEL_NAMES
from src.phase_3.data_loader import get_cell_data
from src.phase_3.evaluation import run_all_evaluations
from src.phase_3.plotting import main as generate_plots


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3-Lab: Classical Baselines on EIS Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Full run:                python -m src.phase_3.run_phase_3\n"
            "  Add GP+CNN1D only:       python -m src.phase_3.run_phase_3 --models gp cnn1d --incremental\n"
            "  Single model from scratch: python -m src.phase_3.run_phase_3 --models ridge\n"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help=(
            "Space-separated list of model names to run "
            f"(choices: {', '.join(MODEL_NAMES)}). "
            "If omitted, all models in MODEL_NAMES are run."
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help=(
            "Load existing loco_results.csv / temporal_results.csv and skip "
            "models already present. Use with --models to append new models "
            "without recomputing everything."
        ),
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        default=False,
        help="Disable GridSearchCV hyperparameter tuning (faster, for debugging).",
    )
    return parser.parse_args()


def main():
    """Run Phase 3-Lab classical baselines evaluation."""
    args = parse_args()

    paths = Phase3LabPaths()
    paths.ensure_dirs()

    # Setup logging
    log_path = paths.data_dir / "phase_3_log.txt"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    print("Phase 3-Lab: Classical Baselines on EIS Features")
    print("=" * 60)

    # Echo CLI options
    if args.models:
        print(f"  --models:      {', '.join(args.models)}")
    else:
        print(f"  --models:      (all) {', '.join(MODEL_NAMES)}")
    print(f"  --incremental: {args.incremental}")
    print(f"  --no-tune:     {args.no_tune}")

    print(f"\nPaths:")
    print(f"  Phase 2-Lab features: {paths.phase2_lab_dir}")
    print(f"  Results: {paths.data_dir}")

    # Load data
    print("\n[1/3] Loading features from Phase 2-Lab...")
    cell_data = get_cell_data(paths)

    for cid, data in cell_data.items():
        print(f"  {cid}: {data['X_6d'].shape[0]} blocks, "
              f"6D features shape={data['X_6d'].shape}, "
              f"72D shape={data['X_72d'].shape}")

    # Run evaluations
    models_label = ", ".join(args.models) if args.models else f"{', '.join(MODEL_NAMES)} + *_72d baselines"
    print("\n[2/3] Running model evaluations...")
    print(f"  Models: {models_label}")
    print(f"  Regimes: LOCO (leave-one-cell-out, all 4 cells), Temporal split")

    all_results, _ = run_all_evaluations(
        cell_data,
        paths,
        tune_hyperparams=not args.no_tune,
        model_names=args.models,       # None → full suite
        incremental=args.incremental,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for regime in ["loco", "temporal"]:
        df = all_results.get(regime)
        if df is None or df.empty:
            print(f"\n--- {regime.upper()} --- (no results)")
            continue
        print(f"\n--- {regime.upper()} ---")
        agg = df.groupby("model")["mae"].agg(["mean", "std"]).round(4)
        print(agg)

    # Generate plots (only meaningful on full results)
    if not args.models or len(args.models) == len(MODEL_NAMES):
        print("\n[3/3] Generating plots...")
        generate_plots(paths)
    else:
        print("\n[3/3] Skipping plot generation (partial --models run).")
        print("      Re-run without --models to regenerate plots over all results.")

    print("\n" + "=" * 60)
    print("Phase 3-Lab Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Results: {paths.data_dir}")
    print(f"  Plots:   {paths.plots_dir}")
    print(f"  Finished at: {datetime.now().isoformat()}")

    sys.stdout = tee.terminal
    tee.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
