"""Orchestration script for Phase 2-Lab: EIS Feature Engineering.

Usage:
    python -m src.phase_2.run_phase_2
"""

import sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")

from .config import Phase2LabPaths, CELL_IDS, N_EIS_RAW, N_QRC_INPUT
from ..phase_1.data_loader import load_eis, load_capacity
from .feature_engineering import build_feature_table
from .plotting import (
    plot_explained_variance,
    plot_features_vs_soh,
    plot_feature_correlation,
)


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


def main() -> None:
    """Run Phase 2-Lab EIS feature engineering."""
    paths = Phase2LabPaths()
    paths.ensure_dirs()

    # Setup logging
    log_path = paths.features_dir / "phase_2_log.txt"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    print("Phase 2-Lab: EIS Feature Engineering")
    print("=" * 60)

    # ── Load raw data ─────────────────────────────────────────
    print("\n[1/4] Loading raw EIS and capacity data...")
    eis_df = load_eis(data_dir=paths.data_dir)
    cap_df = load_capacity(data_dir=paths.data_dir)
    print(f"  EIS rows: {len(eis_df):,}")
    print(f"  Capacity rows: {len(cap_df)}")

    # ── Build feature tables ──────────────────────────────────
    print("\n[2/4] Building feature tables...")
    features_72d, features_6d, reducer = build_feature_table(eis_df, cap_df)

    # Sanity checks
    n_samples = len(features_72d)
    feature_cols = [c for c in features_72d.columns
                    if c.startswith("re_f") or c.startswith("im_f")]
    n_features = len(feature_cols)

    print(f"\n  Sanity checks:")
    print(f"  [OK] Samples: {n_samples} (expected ~34)")
    print(f"  [OK] Raw features: {n_features} (expected {N_EIS_RAW})")

    pc_cols = [c for c in features_6d.columns if c.startswith("pc")]
    print(f"  [OK] Reduced features: {len(pc_cols)} (expected {N_QRC_INPUT})")

    assert n_features == N_EIS_RAW, f"Expected {N_EIS_RAW} features, got {n_features}"
    assert len(pc_cols) == N_QRC_INPUT, f"Expected {N_QRC_INPUT} PCs, got {len(pc_cols)}"

    # ── Save features ─────────────────────────────────────────
    print("\n[3/4] Saving feature tables...")
    f72_path = paths.features_dir / "features_eis_72d.csv"
    f6_path = paths.features_dir / "features_eis_6d.csv"
    features_72d.to_csv(f72_path, index=False)
    features_6d.to_csv(f6_path, index=False)
    print(f"  Saved: {f72_path}")
    print(f"  Saved: {f6_path}")

    # Save PCA components if applicable
    if hasattr(reducer, "components_"):
        import numpy as np
        np.save(
            paths.features_dir / "pca_components.npy",
            reducer.components_,
        )
        print(f"  Saved: pca_components.npy")

    # ── Generate plots ────────────────────────────────────────
    print("\n[4/4] Generating plots...")
    plot_explained_variance(reducer, paths.plots_dir)
    plot_features_vs_soh(features_6d, paths.plots_dir)
    plot_feature_correlation(features_72d, paths.plots_dir)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2-Lab Complete!")
    print("=" * 60)
    print(f"  Features (72D): {f72_path}")
    print(f"  Features (6D):  {f6_path}")
    print(f"  Plots:          {paths.plots_dir}")

    # Per-cell summary
    print("\n  Per-cell block counts:")
    for cid in CELL_IDS:
        n = len(features_6d[features_6d["cell_id"] == cid])
        print(f"    {cid}: {n} blocks")

    soh = features_6d["soh_pct"]
    print(f"\n  SOH range: {soh.min():.2f}% — {soh.max():.2f}%")
    print(f"  SOH std:   {soh.std():.2f}%")
    print(f"  Finished at: {datetime.now().isoformat()}")

    sys.stdout = tee.terminal
    tee.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
