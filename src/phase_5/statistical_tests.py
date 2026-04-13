"""Statistical tests for QRC vs baseline comparisons.

Implements:
 - Bootstrap resampling (1000 iterations) on per-fold MAE
 - Paired Wilcoxon signed-rank test (QRC vs each baseline)
 - Cohen's d effect sizes
 - 95% confidence intervals for mean MAE differences
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from scipy import stats

from .config import RANDOM_STATE

N_BOOTSTRAP = 1000
ALPHA = 0.05  # for 95% CI


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d between two arrays (paired)."""
    diff = a - b
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
    return d


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    means = np.zeros(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(data[idx])
    ci_lower = np.percentile(means, 100 * alpha / 2)
    ci_upper = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(data), ci_lower, ci_upper


def run_statistical_tests(
    results_csv: Path,
    qrc_model: str = "qrc",
    output_dir: Path = None,
) -> pd.DataFrame:
    """Run statistical tests comparing QRC to all baselines.

    Args:
        results_csv: Path to combined results CSV with columns
                     [model, test_cell, mae, ...]
        qrc_model: Name of the QRC model in the CSV
        output_dir: Directory to save outputs

    Returns:
        DataFrame with statistical test results
    """
    df = pd.read_csv(results_csv)

    # Get per-fold MAE for each model
    models = [m for m in df["model"].unique() if m != qrc_model]

    if qrc_model not in df["model"].unique():
        print(f"  WARNING: '{qrc_model}' not in results. Available: {df['model'].unique()}")
        return pd.DataFrame()

    qrc_folds = df[df["model"] == qrc_model].sort_values("test_cell")
    qrc_mae = qrc_folds["mae"].values

    results = []

    for model_name in models:
        baseline_folds = df[df["model"] == model_name].sort_values("test_cell")
        baseline_mae = baseline_folds["mae"].values

        # Ensure aligned folds
        n = min(len(qrc_mae), len(baseline_mae))
        if n < 2:
            print(f"  WARNING: Not enough folds for {model_name} (n={n}), skipping")
            continue

        q = qrc_mae[:n]
        b = baseline_mae[:n]

        # Paired Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = stats.wilcoxon(q, b)
        except ValueError:
            # Happens when all differences are zero
            w_stat, w_pvalue = 0.0, 1.0

        # Cohen's d
        d = cohens_d(q, b)

        # Bootstrap CI for mean MAE difference (QRC - baseline)
        diff = q - b
        mean_diff, ci_lower, ci_upper = bootstrap_ci(diff)

        # Bootstrap CI for QRC MAE
        qrc_mean, qrc_ci_lo, qrc_ci_hi = bootstrap_ci(q)

        # Bootstrap CI for baseline MAE
        bl_mean, bl_ci_lo, bl_ci_hi = bootstrap_ci(b)

        results.append({
            "baseline": model_name,
            "n_folds": n,
            "qrc_mae_mean": qrc_mean,
            "qrc_mae_ci_lo": qrc_ci_lo,
            "qrc_mae_ci_hi": qrc_ci_hi,
            "baseline_mae_mean": bl_mean,
            "baseline_mae_ci_lo": bl_ci_lo,
            "baseline_mae_ci_hi": bl_ci_hi,
            "mae_diff_mean": mean_diff,
            "mae_diff_ci_lo": ci_lower,
            "mae_diff_ci_hi": ci_upper,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_pvalue,
            "significant": w_pvalue < ALPHA,
            "cohens_d": d,
            "qrc_wins": np.sum(q < b),
            "ties": np.sum(q == b),
            "qrc_losses": np.sum(q > b),
        })

    results_df = pd.DataFrame(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "statistical_tests.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n  Saved {out_path}")

        # Print summary
        print("\n  ┌──────────────────────────────────────────────────────┐")
        print("  │  Statistical Test Summary (QRC vs Baselines)        │")
        print("  └──────────────────────────────────────────────────────┘")
        for _, row in results_df.iterrows():
            sig = "★" if row["significant"] else " "
            print(f"  {sig} vs {row['baseline']:15s}: "
                  f"Δ={row['mae_diff_mean']:+.4f} "
                  f"[{row['mae_diff_ci_lo']:+.4f}, {row['mae_diff_ci_hi']:+.4f}] "
                  f"p={row['wilcoxon_p']:.4f} d={row['cohens_d']:+.3f}")

    return results_df
