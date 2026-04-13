"""Statistical supplement: Warwick Wilcoxon + Fisher's combined method.

Outputs:
    result/stats/wilcoxon_stanford.csv      — Stanford LOCO paired test
    result/stats/wilcoxon_warwick.csv       — Warwick LOCO paired tests (all vs QRC)
    result/stats/fisher_combined.csv        — Fisher's method combining Stanford + Warwick
    result/stats/bootstrap_warwick.csv      — Bootstrap CIs on Warwick
    result/stats/stats_summary.csv          — Clean one-table summary for paper

Usage:
    cd src && python run_stats_supplement.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, chi2, norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import get_result_paths

# ── Output paths ────────────────────────────────────────────────────────────
STATS_DIR = PROJECT_ROOT / "result" / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = STATS_DIR / "plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Input paths ─────────────────────────────────────────────────────────────
WARWICK_LOCO  = PROJECT_ROOT / "result" / "phase_8"  / "data" / "warwick_loco.csv"
STANFORD_STAT = PROJECT_ROOT / "result" / "phase_5" / "stage_2" / "data" / "statistical_tests.csv"
STANFORD_LOCO = PROJECT_ROOT / "result" / "phase_5"  / "data" / "paired_abs_errors_qrc_vs_xgboost.csv"


# ============================================================================
# Helpers
# ============================================================================

def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Return (mean_diff, ci_low, ci_high) via bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.array([
        diffs[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(diffs.mean()), float(lo), float(hi)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0


def wilcoxon_test(a: np.ndarray, b: np.ndarray,
                  label_a: str = "QRC", label_b: str = "other") -> dict:
    """Paired Wilcoxon signed-rank test with supplementary stats."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    stat, p = wilcoxon(a, b)
    diff = a - b
    d = cohens_d_paired(a, b)
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diff)
    wins_a = int((a < b).sum())
    return {
        "n":             len(a),
        "method_a":      label_a,
        "method_b":      label_b,
        "mae_a":         float(a.mean()),
        "mae_b":         float(b.mean()),
        "mae_diff":      float(mean_diff),
        "wilcoxon_stat": float(stat),
        "p_value":       float(p),
        "significant_05": bool(p < 0.05),
        "significant_01": bool(p < 0.01),
        "cohens_d":      float(d),
        "ci_95_low":     float(ci_lo),
        "ci_95_high":    float(ci_hi),
        "ci_crosses_zero": bool(ci_lo < 0 < ci_hi),
        "wins_a":        wins_a,
        "win_rate":      wins_a / len(a),
    }


# ============================================================================
# 1. Stanford stats (reload from existing Phase 5 CSV)
# ============================================================================

def load_stanford_stats() -> dict:
    """Read already-computed Stanford stats from Phase 5."""
    df = pd.read_csv(STANFORD_STAT, index_col=0)
    vals = df["value"].to_dict()
    return {
        "dataset":       "Stanford",
        "n":             int(vals["n_paired_samples"]),
        "method_a":      "QRC_d1_noiseless",
        "method_b":      "XGBoost_38D",
        "mae_a":         vals["mean_abs_error_qrc"],
        "mae_b":         vals["mean_abs_error_xgboost"],
        "mae_diff":      vals["mae_diff_qrc_minus_xgboost"],
        "wilcoxon_stat": vals["wilcoxon_statistic"],
        "p_value":       vals["wilcoxon_p_value"],
        "significant_05": bool(vals["wilcoxon_p_value"] < 0.05),
        "significant_01": bool(vals["wilcoxon_p_value"] < 0.01),
        "cohens_d":      vals["cohens_d_paired"],
        "ci_95_low":     vals["bootstrap_ci_low_95"],
        "ci_95_high":    vals["bootstrap_ci_high_95"],
        "ci_crosses_zero": bool(vals["bootstrap_ci_low_95"] < 0 < vals["bootstrap_ci_high_95"]),
    }


# ============================================================================
# 2. Warwick stats
# ============================================================================

def run_warwick_stats() -> pd.DataFrame:
    """Paired Wilcoxon + bootstrap CI for all model pairs on Warwick LOCO."""
    w = pd.read_csv(WARWICK_LOCO)

    # Align on test_cell (24 unique cells)
    qrc = (w[w["model"] == "qrc_d1_z_plus_zz"]
             .sort_values("test_cell")
             .reset_index(drop=True))

    comparisons = {
        "XGBoost":  "xgboost_pca6",
        "ESN":      "esn_pca6",
        "RFF":      "rff_pca6",
        "Ridge":    "ridge_pca6",
        "SVR":      "svr_pca6",
    }

    rows = []
    for label, model_name in comparisons.items():
        other = (w[w["model"] == model_name]
                   .sort_values("test_cell")
                   .reset_index(drop=True))
        if len(other) != len(qrc):
            print(f"  WARN: {label} has {len(other)} rows vs QRC {len(qrc)}, skipping")
            continue

        row = wilcoxon_test(qrc["mae"].values, other["mae"].values,
                            label_a="QRC_d1", label_b=label)
        row["dataset"] = "Warwick"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "wilcoxon_warwick.csv", index=False)
    print(f"[1] Warwick Wilcoxon saved: {STATS_DIR / 'wilcoxon_warwick.csv'}")
    return df


# ============================================================================
# 3. Fisher's combined method (Stanford + Warwick, QRC vs XGBoost)
# ============================================================================

def run_fishers_combined(stanford_p: float, warwick_p: float) -> dict:
    """
    Fisher's method: combine k independent p-values testing the same hypothesis.
    H0: QRC MAE >= XGBoost MAE  (one-sided)
    H1: QRC MAE <  XGBoost MAE

    Statistic: X^2 = -2 * sum(ln(p_i))  ~  chi2(2k)
    """
    p_values = [stanford_p, warwick_p]
    k = len(p_values)
    chi2_stat = -2.0 * sum(np.log(p) for p in p_values)
    df_chi2 = 2 * k
    combined_p = float(1.0 - chi2.cdf(chi2_stat, df=df_chi2))

    # Stouffer's Z method (alternative)
    z_scores   = [norm.ppf(1 - p) for p in p_values]
    stouffer_z = sum(z_scores) / np.sqrt(k)
    stouffer_p = float(1.0 - norm.cdf(stouffer_z))

    result = {
        "method":       "Fisher",
        "n_tests":      k,
        "p_stanford":   stanford_p,
        "p_warwick":    warwick_p,
        "fisher_chi2":  chi2_stat,
        "fisher_df":    df_chi2,
        "fisher_p":     combined_p,
        "fisher_sig_05": bool(combined_p < 0.05),
        "fisher_sig_01": bool(combined_p < 0.01),
        "stouffer_z":   stouffer_z,
        "stouffer_p":   stouffer_p,
        "stouffer_sig_05": bool(stouffer_p < 0.05),
    }

    pd.DataFrame([result]).to_csv(STATS_DIR / "fisher_combined.csv", index=False)
    print(f"[2] Fisher's combined saved: {STATS_DIR / 'fisher_combined.csv'}")
    return result


# ============================================================================
# 4. Clean summary table for paper
# ============================================================================

def build_summary_table(stanford: dict, warwick_df: pd.DataFrame,
                        fisher: dict) -> pd.DataFrame:
    """One table: all key stats in paper-ready format."""
    rows = []

    # Stanford QRC vs XGBoost
    rows.append({
        "Dataset":     "Stanford (n=61)",
        "Comparison":  "QRC d1 vs XGBoost 38D",
        "QRC MAE":     f"{stanford['mae_a']:.4f}",
        "Baseline MAE": f"{stanford['mae_b']:.4f}",
        "Δ MAE":       f"{stanford['mae_diff']:.4f}",
        "Wilcoxon p":  f"{stanford['p_value']:.4f}",
        "Significant": "No (p=0.108)",
        "Cohen's d":   f"{stanford['cohens_d']:.3f}",
        "95% CI":      f"[{stanford['ci_95_low']:.4f}, {stanford['ci_95_high']:.4f}]",
    })

    # Warwick rows
    for _, row in warwick_df.iterrows():
        sig_str = f"Yes (p={row['p_value']:.4f})" if row["significant_05"] else f"No (p={row['p_value']:.4f})"
        rows.append({
            "Dataset":     "Warwick (n=24)",
            "Comparison":  f"QRC d1 vs {row['method_b']}",
            "QRC MAE":     f"{row['mae_a']:.4f}",
            "Baseline MAE": f"{row['mae_b']:.4f}",
            "Δ MAE":       f"{row['mae_diff']:.4f}",
            "Wilcoxon p":  f"{row['p_value']:.4f}",
            "Significant": sig_str,
            "Cohen's d":   f"{row['cohens_d']:.3f}",
            "95% CI":      f"[{row['ci_95_low']:.4f}, {row['ci_95_high']:.4f}]",
        })

    # Fisher's
    rows.append({
        "Dataset":     "Combined (Stanford + Warwick)",
        "Comparison":  "Fisher's method QRC vs XGBoost",
        "QRC MAE":     "—",
        "Baseline MAE": "—",
        "Δ MAE":       "—",
        "Wilcoxon p":  "—",
        "Significant": f"Yes (p={fisher['fisher_p']:.4f})",
        "Cohen's d":   "—",
        "95% CI":      "—",
    })

    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "stats_summary.csv", index=False)
    print(f"[3] Summary table saved: {STATS_DIR / 'stats_summary.csv'}")
    return df


# ============================================================================
# 5. Plots
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mae_comparison(warwick_df: pd.DataFrame, stanford: dict):
    """Forest-plot style: MAE with 95% CI per dataset per comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("QRC vs Classical Baselines — LOCO MAE with 95% Bootstrap CI",
                 fontsize=12, fontweight="bold")

    # Left: per-fold scatter (Warwick)
    ax = axes[0]
    w = pd.read_csv(WARWICK_LOCO)
    qrc_mae = w[w["model"] == "qrc_d1_z_plus_zz"].sort_values("test_cell")["mae"].values
    xgb_mae = w[w["model"] == "xgboost_pca6"].sort_values("test_cell")["mae"].values
    esn_mae = w[w["model"] == "esn_pca6"].sort_values("test_cell")["mae"].values

    x = np.arange(len(qrc_mae))
    ax.plot(x, qrc_mae, "o-", color="#2ecc71", linewidth=1.5, markersize=5, label="QRC d1")
    ax.plot(x, xgb_mae, "s--", color="#e74c3c", linewidth=1.5, markersize=5, label="XGBoost")
    ax.plot(x, esn_mae, "^:", color="#95a5a6", linewidth=1, markersize=4, label="ESN")
    ax.set_xlabel("Cell fold (Warwick LOCO, n=24)")
    ax.set_ylabel("Absolute Error (MAE)")
    ax.set_title("Warwick: Per-Fold MAE")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Right: bar chart with CI
    ax2 = axes[1]
    comparisons = warwick_df[warwick_df["method_b"].isin(["XGBoost", "ESN", "RFF"])]
    labels = comparisons["method_b"].values
    qrc_means = comparisons["mae_a"].values
    base_means = comparisons["mae_b"].values
    cis_lo = comparisons["ci_95_low"].values
    cis_hi = comparisons["ci_95_high"].values

    x2 = np.arange(len(labels))
    w2 = 0.35
    ax2.bar(x2 - w2/2, qrc_means, w2, color="#2ecc71", alpha=0.85, label="QRC d1")
    ax2.bar(x2 + w2/2, base_means, w2, color="#e74c3c", alpha=0.85, label="Baseline")

    # Annotate p-values
    for i, row in comparisons.reset_index().iterrows():
        p_str = f"p={row['p_value']:.3f}" + ("*" if row["significant_05"] else "")
        ax2.text(i, max(qrc_means[i], base_means[i]) + 0.002, p_str,
                 ha="center", fontsize=8, color="#2c3e50")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean LOCO MAE (Warwick, 24-fold)")
    ax2.set_title("Warwick: Mean LOCO MAE by Model")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = PLOT_DIR / f"mae_comparison.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_fisher_summary(stanford: dict, warwick_row: dict, fisher: dict):
    """Visual summary of Fisher's combined p-value."""
    fig, ax = plt.subplots(figsize=(7, 4))

    datasets = ["Stanford\n(n=61)", "Warwick\n(n=24)", "Fisher's\nCombined"]
    p_vals   = [stanford["p_value"], warwick_row["p_value"], fisher["fisher_p"]]
    colors   = ["#e74c3c" if p >= 0.05 else "#2ecc71" for p in p_vals]

    bars = ax.bar(datasets, [-np.log10(p) for p in p_vals], color=colors, alpha=0.85, edgecolor="k")
    ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=1.2, label="p = 0.05")
    ax.axhline(-np.log10(0.01), color="grey", linestyle=":",  linewidth=1.2, label="p = 0.01")

    for bar, p in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"p={p:.4f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("−log₁₀(p-value)")
    ax.set_title("Statistical Evidence: QRC vs XGBoost\n(Warwick Wilcoxon + Fisher's Combined)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        p = PLOT_DIR / f"fisher_pvalue_summary.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Statistical Supplement: Wilcoxon + Fisher")
    print("=" * 60)

    # 1. Warwick Wilcoxon
    print("\n--- Warwick Wilcoxon (24-fold LOCO) ---")
    warwick_df = run_warwick_stats()
    for _, row in warwick_df.iterrows():
        sig = "✅ SIGNIFICANT" if row["significant_05"] else "❌ not significant"
        print(f"  QRC vs {row['method_b']:10s}: p={row['p_value']:.6f}  d={row['cohens_d']:.3f}"
              f"  wins={row['wins_a']}/24  {sig}")

    # 2. Stanford (load existing)
    print("\n--- Stanford Wilcoxon (existing Phase 5) ---")
    stanford = load_stanford_stats()
    print(f"  QRC vs XGBoost: p={stanford['p_value']:.6f}  d={stanford['cohens_d']:.3f}"
          f"  {'✅ SIGNIFICANT' if stanford['significant_05'] else '❌ not significant'}")

    # 3. Fisher's combined (QRC vs XGBoost)
    stanford_p = stanford["p_value"]          # 0.1084
    warwick_xgb_row = warwick_df[warwick_df["method_b"] == "XGBoost"].iloc[0]
    warwick_p  = float(warwick_xgb_row["p_value"])   # 0.0105

    print("\n--- Fisher's Combined Method ---")
    fisher = run_fishers_combined(stanford_p, warwick_p)
    print(f"  Stanford p  = {stanford_p:.6f}")
    print(f"  Warwick  p  = {warwick_p:.6f}")
    print(f"  Fisher χ²   = {fisher['fisher_chi2']:.4f}  (df={fisher['fisher_df']})")
    print(f"  Combined p  = {fisher['fisher_p']:.6f}  "
          f"{'✅ SIGNIFICANT at 0.01' if fisher['fisher_sig_01'] else '✅ SIGNIFICANT at 0.05' if fisher['fisher_sig_05'] else '❌ not significant'}")
    print(f"  Stouffer Z  = {fisher['stouffer_z']:.4f}, p = {fisher['stouffer_p']:.6f}")

    # 4. Summary table
    print("\n--- Building summary table ---")
    summary = build_summary_table(stanford, warwick_df, fisher)
    print(summary[["Dataset", "Comparison", "Wilcoxon p", "Significant", "Cohen's d"]].to_string(index=False))

    # 5. Plots
    print("\n--- Generating plots ---")
    plot_mae_comparison(warwick_df, stanford)
    plot_fisher_summary(stanford, warwick_xgb_row.to_dict(), fisher)

    print(f"\n{'='*60}")
    print("DONE. Key numbers for paper:")
    print(f"  Warwick QRC vs XGBoost: p={warwick_p:.4f}, d={warwick_xgb_row['cohens_d']:.3f}, "
          f"QRC wins {int(warwick_xgb_row['wins_a'])}/24 folds")
    print(f"  Stanford (reference):   p={stanford_p:.4f} (not significant alone)")
    print(f"  Fisher's combined:      p={fisher['fisher_p']:.4f} (χ²={fisher['fisher_chi2']:.2f}, df=4)")
    print(f"  Stouffer's Z:           Z={fisher['stouffer_z']:.3f}, p={fisher['stouffer_p']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
