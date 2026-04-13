"""
Compute manuscript statistics not currently saved to CSV:
1. Cohen's d for Warwick QRC vs XGBoost (from phase_11 fold-level data)
2. Few-shot 9-18 cell regime averages (from phase_5/stage_5 data)

Outputs saved to src/manuscript_support/data/
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_warwick_cohens_d():
    """Compute Cohen's d for paired fold-level MAE from phase_11 data."""
    csv_path = PROJECT_ROOT / "result" / "phase_11" / "stage_2" / "data" / "per_fold_qrc_vs_xgb.csv"
    df = pd.read_csv(csv_path)

    qrc = df["mae_qrc_pct"].values
    xgb = df["mae_xgb_pct"].values
    diff = qrc - xgb  # QRC minus XGBoost (negative = QRC better)

    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff

    # Also compute from raw values for cross-check
    mean_qrc = np.mean(qrc)
    mean_xgb = np.mean(xgb)
    pooled_std = np.sqrt((np.std(qrc, ddof=1)**2 + np.std(xgb, ddof=1)**2) / 2)
    cohens_d_pooled = (mean_qrc - mean_xgb) / pooled_std

    results = pd.DataFrame([
        {"metric": "n_folds", "value": n},
        {"metric": "mean_mae_qrc_pct", "value": round(mean_qrc, 4)},
        {"metric": "mean_mae_xgb_pct", "value": round(mean_xgb, 4)},
        {"metric": "mean_diff_pct", "value": round(mean_diff, 4)},
        {"metric": "std_diff_pct", "value": round(std_diff, 4)},
        {"metric": "cohens_d_paired", "value": round(cohens_d, 4)},
        {"metric": "cohens_d_pooled", "value": round(cohens_d_pooled, 4)},
    ])

    out_path = OUTPUT_DIR / "warwick_cohens_d.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(results.to_string(index=False))
    print(f"\nCohen's d (paired): {cohens_d:.4f}")
    print(f"Cohen's d (pooled): {cohens_d_pooled:.4f}")
    print(f"Draft claims: -0.46")
    return cohens_d


def compute_fewshot_regime_averages():
    """Compute mean MAE for each model in the 9-18 cell training regime."""
    csv_path = PROJECT_ROOT / "result" / "phase_5" / "stage_5" / "data" / "warwick_fewshot_results.csv"
    df = pd.read_csv(csv_path)

    # Filter to n_train in {9, 12, 15, 18}
    regime = df[df["n_train"].isin([9, 12, 15, 18])]

    # Mean MAE per model across all repeats in the regime
    summary = regime.groupby("model").agg(
        mean_mae=("mae", "mean"),
        mean_mae_pct=("mae_pct", "mean"),
        std_mae_pct=("mae_pct", "std"),
        n_repeats=("mae", "count"),
    ).reset_index()

    summary = summary.sort_values("mean_mae_pct")
    summary["mean_mae_pct"] = summary["mean_mae_pct"].round(4)
    summary["std_mae_pct"] = summary["std_mae_pct"].round(4)
    summary["mean_mae"] = summary["mean_mae"].round(6)

    out_path = OUTPUT_DIR / "warwick_fewshot_9_18_regime.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(summary.to_string(index=False))

    # Also save per-n_train breakdown
    per_n = regime.groupby(["model", "n_train"]).agg(
        mean_mae_pct=("mae_pct", "mean"),
        std_mae_pct=("mae_pct", "std"),
        n_repeats=("mae", "count"),
    ).reset_index()
    per_n["mean_mae_pct"] = per_n["mean_mae_pct"].round(4)
    per_n["std_mae_pct"] = per_n["std_mae_pct"].round(4)

    per_n_path = OUTPUT_DIR / "warwick_fewshot_per_ntrain.csv"
    per_n.to_csv(per_n_path, index=False)
    print(f"\nSaved: {per_n_path}")
    print(per_n.to_string(index=False))

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("1. COHEN'S D FROM WARWICK FOLD-LEVEL DATA")
    print("=" * 60)
    compute_warwick_cohens_d()

    print("\n" + "=" * 60)
    print("2. FEW-SHOT 9-18 REGIME AVERAGES")
    print("=" * 60)
    compute_fewshot_regime_averages()
