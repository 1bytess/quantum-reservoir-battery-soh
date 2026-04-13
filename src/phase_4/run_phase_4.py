"""Orchestration script for Phase 4-Lab: QRC on EIS (Noiseless + Noisy).

Two stages:
  Stage 1 — Noiseless (statevector): exact quantum simulation
  Stage 2 — Noisy (IBM Heron R2 digital twin): shot-based + noise model

Usage:
    # Run both stages
    python -m src.phase_4.run_phase_4

    # Noiseless only
    python -m src.phase_4.run_phase_4 --noiseless-only

    # Noisy only (requires qiskit-aer)
    python -m src.phase_4.run_phase_4 --noisy-only

    # Try real IBM backend for noise model
    python -m src.phase_4.run_phase_4 --ibm
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.phase_4.config import Phase4LabPaths, DEPTH_RANGE, SHOTS_LIST
from src.phase_3.data_loader import get_cell_data
from src.phase_3.config import Phase3LabPaths


class TeeLogger:
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


def main(
    run_noiseless: bool = True,
    run_noisy: bool = True,
    use_ibm_noise: bool = False,
    run_temporal_qrc: bool = False,
    run_observable_ablation: bool = False,
):
    """Run Phase 4-Lab: QRC evaluation on EIS features."""
    paths = Phase4LabPaths()
    paths.ensure_dirs()

    log_path = paths.data_dir / "phase_4_log.txt"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    print("Phase 4-Lab: QRC on EIS Features")
    print("  Stage 1: Noiseless (statevector)")
    print("  Stage 2: Noisy (IBM Heron R2 digital twin)")
    print("=" * 60)

    # Check Qiskit availability
    try:
        from .circuit import QISKIT_AVAILABLE
        print(f"\n  Qiskit available: {QISKIT_AVAILABLE}")
    except Exception:
        QISKIT_AVAILABLE = False
        print("\n  Qiskit NOT available")

    try:
        from qiskit_aer import AerSimulator
        AER_AVAILABLE = True
        print(f"  Qiskit Aer available: True")
    except ImportError:
        AER_AVAILABLE = False
        print(f"  Qiskit Aer available: False")

    if not QISKIT_AVAILABLE and run_noiseless:
        print("\n  WARNING: Qiskit not available. Stage 1 will use classical fallback.")

    if not AER_AVAILABLE and run_noisy:
        print("\n  WARNING: Qiskit Aer not available. Skipping Stage 2 (noisy).")
        run_noisy = False

    # Load data
    print("\n[1/4] Loading features from Phase 2-Lab...")
    phase3_paths = Phase3LabPaths()
    cell_data = get_cell_data(phase3_paths)

    for cid, data in cell_data.items():
        print(f"  {cid}: {data['X_6d'].shape[0]} blocks, shape={data['X_6d'].shape}")

    # Stage 1: Noiseless
    noiseless_df = None
    if run_noiseless:
        print(f"\n[2/4] Running Stage 1: Noiseless (depths: {DEPTH_RANGE})...")
        from src.phase_4.stage1_noiseless import run_stage1
        noiseless_df = run_stage1(cell_data, paths)

    # Stage 2: Noisy
    noisy_df = None
    if run_noisy:
        print(f"\n[3/4] Running Stage 2: Noisy IBM Heron R2 Twin "
              f"(depths: {DEPTH_RANGE}, shots: {SHOTS_LIST})...")
        from src.phase_4.stage2_noisy import run_stage2
        noisy_df = run_stage2(
            cell_data, paths,
            use_ibm_noise=use_ibm_noise,
        )

    # Stage 3: Temporal QRC (optional)
    temporal_df = None
    if run_temporal_qrc:
        print(f"\n[3b/4] Running Temporal QRC (72D raw EIS, no PCA)...")
        from src.phase_4.temporal_qrc_eval import run_temporal_qrc as _run_tqrc
        temporal_df = _run_tqrc(cell_data, paths)

    # Stage 4: Observable Ablation (optional)
    obs_ablation_df = None
    if run_observable_ablation:
        print(f"\n[3c/4] Running Observable Ablation...")
        from src.phase_4.observable_ablation import run_observable_ablation as _run_oa
        obs_ablation_df = _run_oa(cell_data, paths)

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if noiseless_df is not None:
        print("\n--- STAGE 1: NOISELESS ---")
        for regime in ["loco", "temporal"]:
            regime_df = noiseless_df[noiseless_df["regime"] == regime]
            if regime_df.empty:
                continue
            summary = regime_df.groupby("depth")["mae"].agg(["mean", "std"]).round(4)
            print(f"\n  {regime.upper()}:")
            print(summary.to_string(index=True))
            best_depth = summary["mean"].idxmin()
            print(f"  → Best: depth={best_depth}, MAE={summary.loc[best_depth, 'mean']:.4f}")

    if noisy_df is not None:
        print("\n--- STAGE 2: NOISY (IBM HERON R2) ---")
        loco_noisy = noisy_df[noisy_df["regime"] == "loco"]
        if not loco_noisy.empty:
            summary = loco_noisy.groupby(["depth", "shots"])["mae"].agg(
                ["mean", "std"]
            ).round(4)
            print(summary.to_string(index=True))

    # Noiseless vs Noisy degradation
    if noiseless_df is not None and noisy_df is not None:
        print("\n--- NOISE DEGRADATION ---")
        nl_loco = noiseless_df[noiseless_df["regime"] == "loco"]
        max_shots = noisy_df["shots"].max()
        ny_loco = noisy_df[
            (noisy_df["regime"] == "loco") & (noisy_df["shots"] == max_shots)
        ]

        nl_avg = nl_loco.groupby("depth")["mae"].mean()
        ny_avg = ny_loco.groupby("depth")["mae"].mean()

        for d in sorted(set(nl_avg.index) & set(ny_avg.index)):
            deg = (ny_avg[d] - nl_avg[d]) / nl_avg[d] * 100
            print(f"  Depth {d}: {nl_avg[d]:.4f} → {ny_avg[d]:.4f} ({deg:+.1f}%)")

    # Compare with Phase 3-Lab baselines
    p3_path = paths.phase3_lab_dir / "loco_results.csv"
    if p3_path.exists():
        import pandas as pd
        p3 = pd.read_csv(p3_path)
        print("\n--- COMPARISON WITH PHASE 3-LAB BASELINES ---")
        for model in p3["model"].unique():
            model_mae = p3[p3["model"] == model]["mae"].mean()
            print(f"  {model}: MAE = {model_mae:.4f}")

        if noiseless_df is not None:
            nl_loco = noiseless_df[noiseless_df["regime"] == "loco"]
            best_qrc = nl_loco.groupby("depth")["mae"].mean().min()
            print(f"  QRC noiseless (best): MAE = {best_qrc:.4f}")

        if noisy_df is not None:
            ny_loco = noisy_df[
                (noisy_df["regime"] == "loco") &
                (noisy_df["shots"] == noisy_df["shots"].max())
            ]
            if not ny_loco.empty:
                best_noisy = ny_loco.groupby("depth")["mae"].mean().min()
                print(f"  QRC noisy (best):    MAE = {best_noisy:.4f}")

    # Generate plots
    print("\n[4/4] Generating plots...")
    from src.phase_4.plotting import (
        plot_noiseless_depth_sweep,
        plot_noisy_depth_sweep,
        plot_noiseless_vs_noisy,
        plot_qrc_vs_classical,
        plot_grand_comparison,
        plot_temporal_vs_static_depth,
        plot_observable_ablation,
        plot_noiseless_noisy_scatter,
    )

    if noiseless_df is not None:
        plot_noiseless_depth_sweep(paths)
    if noisy_df is not None:
        plot_noisy_depth_sweep(paths)
    if noiseless_df is not None and noisy_df is not None:
        plot_noiseless_vs_noisy(paths)
    plot_qrc_vs_classical(paths)

    # Tier-1 plots
    plot_grand_comparison(paths)
    if temporal_df is not None:
        plot_temporal_vs_static_depth(paths)
    if obs_ablation_df is not None:
        plot_observable_ablation(paths)
    if noiseless_df is not None:
        plot_noiseless_noisy_scatter(paths)

    print("\n" + "=" * 60)
    print("Phase 4-Lab Complete!")
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
    parser = argparse.ArgumentParser(description="Phase 4-Lab: QRC on EIS")
    parser.add_argument("--noiseless-only", action="store_true",
                        help="Run only Stage 1 (noiseless)")
    parser.add_argument("--noisy-only", action="store_true",
                        help="Run only Stage 2 (noisy)")
    parser.add_argument("--ibm", action="store_true",
                        help="Try to load noise model from real IBM backend")
    parser.add_argument("--temporal-qrc", action="store_true",
                        help="Run temporal QRC on 72D raw EIS")
    parser.add_argument("--observable-ablation", action="store_true",
                        help="Run observable set ablation (Z/ZZ/XYZ)")
    args = parser.parse_args()

    run_nl = not args.noisy_only
    run_ny = not args.noiseless_only
    main(
        run_noiseless=run_nl,
        run_noisy=run_ny,
        use_ibm_noise=args.ibm,
        run_temporal_qrc=args.temporal_qrc,
        run_observable_ablation=args.observable_ablation,
    )
