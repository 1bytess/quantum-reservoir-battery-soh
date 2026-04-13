"""Phase 11 Stage 2b: Statistical tests for nested Warwick LOCO results.

Loads nested LOCO predictions from:
    result/phase_8/stage_2/data/nested_warwick_loco_predictions.csv

Computes:
  - Primary paired Wilcoxon test: QRC vs XGBoost
  - Cell-level bootstrap CI for mean(QRC MAE) - mean(baseline MAE)
  - Optional paired stats table for all nested baselines

Outputs:
  result/phase_11/stage_2b_nested_warwick/data/nested_primary_wilcoxon.csv
  result/phase_11/stage_2b_nested_warwick/data/nested_all_pairwise_stats.csv
  result/phase_11/stage_2b_nested_warwick/data/nested_per_fold_qrc_vs_xgb.csv
  result/phase_11/stage_2b_nested_warwick/data/nested_stats_summary.md
  result/phase_11/stage_2b_nested_warwick/plot/nested_primary_comparison_forest.png/pdf
  result/phase_11/stage_2b_nested_warwick/plot/nested_bootstrap_ci_forest.png/pdf
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_11.config import PROJECT_ROOT, RANDOM_STATE, N_BOOTSTRAP, get_stage_paths


NESTED_PREDICTIONS_CSV = (
    PROJECT_ROOT / "result" / "phase_8" / "stage_2" / "data" / "nested_warwick_loco_predictions.csv"
)
DISPLAY_NAMES = {
    "qrc": "QRC",
    "xgboost": "XGBoost",
    "ridge": "Ridge",
    "svr": "SVR",
    "rff": "RFF",
    "esn": "ESN",
}
QRC_MODEL = "qrc"
PRIMARY_BASELINE = "xgboost"
BASELINES = ["xgboost", "ridge", "svr", "rff", "esn"]


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


def _load_nested_per_fold() -> pd.DataFrame:
    if not NESTED_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Nested prediction CSV not found: {NESTED_PREDICTIONS_CSV}\n"
            "Run: python -m src.phase_8.stage_2_nested_warwick_cv"
        )

    df = pd.read_csv(NESTED_PREDICTIONS_CSV)
    required = {"outer_fold", "test_cell", "model", "abs_error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in nested predictions CSV: {sorted(missing)}")

    pivot = (
        df.pivot_table(
            index=["outer_fold", "test_cell"],
            columns="model",
            values="abs_error",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["outer_fold", "test_cell"])
        .reset_index(drop=True)
    )
    return pivot


def _wilcoxon_paired(qrc_maes: np.ndarray, baseline_maes: np.ndarray) -> dict:
    stat_two, p_two = wilcoxon(qrc_maes, baseline_maes, alternative="two-sided", zero_method="wilcox")
    stat_less, p_less = wilcoxon(qrc_maes, baseline_maes, alternative="less", zero_method="wilcox")
    diffs = qrc_maes - baseline_maes
    return {
        "n_folds": int(len(qrc_maes)),
        "mean_qrc_mae": float(qrc_maes.mean()),
        "mean_baseline_mae": float(baseline_maes.mean()),
        "mean_diff_qrc_minus_baseline": float(diffs.mean()),
        "median_diff_qrc_minus_baseline": float(np.median(diffs)),
        "qrc_better_folds": int(np.sum(qrc_maes < baseline_maes)),
        "baseline_better_folds": int(np.sum(qrc_maes > baseline_maes)),
        "tied_folds": int(np.sum(np.isclose(qrc_maes, baseline_maes))),
        "wilcoxon_stat_twosided": float(stat_two),
        "p_value_twosided": float(p_two),
        "wilcoxon_stat_less": float(stat_less),
        "p_value_less": float(p_less),
        "significant_alpha05_twosided": bool(p_two < 0.05),
        "significant_alpha05_onesided": bool(p_less < 0.05),
    }


def _bootstrap_ci(
    qrc_maes: np.ndarray,
    baseline_maes: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float, float, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(qrc_maes)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = qrc_maes[idx].mean() - baseline_maes[idx].mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(ci_low), float(ci_high), diffs


def _plot_primary_forest(
    fold_table: pd.DataFrame,
    ci_low: float,
    ci_high: float,
    p_value: float,
    plot_dir: Path,
) -> None:
    diffs = fold_table["diff_qrc_minus_xgb"].to_numpy()
    cells = fold_table["test_cell"].tolist()
    mean_diff = float(diffs.mean())

    fig, ax = plt.subplots(figsize=(10, max(5, len(cells) * 0.38)))
    for i, (cell, diff) in enumerate(zip(cells, diffs)):
        color = "tab:blue" if diff < 0 else "tab:red"
        ax.barh(i, diff * 100, color=color, alpha=0.75)
        ax.text(diff * 100 + 0.01, i, f"{diff*100:+.2f}%", va="center", fontsize=7)

    ax.axvline(mean_diff * 100, color="black", linewidth=1.5, linestyle="--",
               label=f"Mean diff = {mean_diff*100:+.3f}%")
    ax.axvspan(ci_low * 100, ci_high * 100, alpha=0.12, color="steelblue",
               label=f"95% CI [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%]")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells, fontsize=7)
    ax.set_xlabel("MAE difference (QRC - XGBoost, %) | Negative = QRC better")
    ax.set_title(
        "Nested Warwick LOCO: QRC vs XGBoost\n"
        f"Wilcoxon p = {p_value:.4f} (two-sided), K = {len(cells)} paired folds"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, plot_dir, "nested_primary_comparison_forest")


def _plot_ci_forest(ci_table: pd.DataFrame, plot_dir: Path) -> None:
    df = ci_table.copy().sort_values("mean_diff")
    labels = [DISPLAY_NAMES.get(model, model) for model in df["baseline_model"]]
    means = df["mean_diff"].to_numpy()
    lows = df["ci_low_95"].to_numpy()
    highs = df["ci_high_95"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, max(4, len(labels) * 0.9)))
    y = np.arange(len(labels))

    for i, (mean, low, high) in enumerate(zip(means, lows, highs)):
        color = "tab:blue" if high < 0 else ("tab:red" if low > 0 else "tab:gray")
        ax.plot([low, high], [i, i], color=color, linewidth=2.5)
        ax.scatter([mean], [i], color=color, zorder=5, s=50)
        ax.text(high + abs(high - low) * 0.06 + 0.0001, i, f"[{low:+.4f}, {high:+.4f}]",
                va="center", fontsize=8)

    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean MAE difference (QRC - baseline)")
    ax.set_title(
        "Nested Warwick LOCO: 95% bootstrap CIs\n"
        f"Cell-level resampling, n = {N_BOOTSTRAP:,}"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, plot_dir, "nested_bootstrap_ci_forest")


def _write_summary(
    primary_row: pd.Series,
    ci_table: pd.DataFrame,
    data_dir: Path,
) -> None:
    improvement = (
        (primary_row["mean_baseline_mae"] - primary_row["mean_qrc_mae"])
        / primary_row["mean_baseline_mae"]
        * 100.0
    )
    xgb_ci_low = primary_row["bootstrap_ci_low_95"] * 100.0
    xgb_ci_high = primary_row["bootstrap_ci_high_95"] * 100.0

    lines = [
        "# Nested Warwick Statistical Summary",
        "",
        "## Primary comparison",
        f"- QRC mean MAE: {primary_row['mean_qrc_mae']*100:.3f}%",
        f"- XGBoost mean MAE: {primary_row['mean_baseline_mae']*100:.3f}%",
        f"- Relative improvement: {improvement:.1f}%",
        f"- Wilcoxon p (two-sided): {primary_row['p_value_twosided']:.6f}",
        f"- Wilcoxon p (one-sided, QRC < XGBoost): {primary_row['p_value_less']:.6f}",
        f"- 95% bootstrap CI for QRC - XGBoost: [{xgb_ci_low:+.3f}%, {xgb_ci_high:+.3f}%]",
        f"- QRC better folds: {int(primary_row['qrc_better_folds'])}/{int(primary_row['n_folds'])}",
        "",
        "## All baselines",
    ]

    for row in ci_table.itertuples(index=False):
        label = DISPLAY_NAMES.get(row.baseline_model, row.baseline_model)
        lines.append(
            f"- {label}: diff={row.mean_diff*100:+.3f}%, "
            f"CI=[{row.ci_low_95*100:+.3f}%, {row.ci_high_95*100:+.3f}%], "
            f"p={row.p_value_twosided:.6f}"
        )

    path = data_dir / "nested_stats_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_2b_nested_warwick")
    log_path = data_dir / "stage_2b_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Phase 11 Stage 2b: Nested Warwick Statistical Tests")
        print("=" * 72)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Input: {NESTED_PREDICTIONS_CSV}")
        print(f"Output dir: {data_dir}")
        print(f"Bootstrap resamples: {N_BOOTSTRAP:,}")

        pivot = _load_nested_per_fold()
        missing_models = [model for model in [QRC_MODEL] + BASELINES if model not in pivot.columns]
        if missing_models:
            raise ValueError(f"Missing nested model columns: {missing_models}")

        print(f"\nLoaded {len(pivot)} paired folds")
        print(f"Available models: {[col for col in pivot.columns if col not in ['outer_fold', 'test_cell']]}")

        qrc_maes = pivot[QRC_MODEL].to_numpy()
        xgb_maes = pivot[PRIMARY_BASELINE].to_numpy()

        primary_stats = _wilcoxon_paired(qrc_maes, xgb_maes)
        boot_mean, ci_low, ci_high, boot_diffs = _bootstrap_ci(qrc_maes, xgb_maes)

        primary_row = {
            "dataset": "warwick_dib_nested_loco",
            "comparison": "qrc_vs_xgboost",
            **primary_stats,
            "bootstrap_mean_diff": boot_mean,
            "bootstrap_ci_low_95": ci_low,
            "bootstrap_ci_high_95": ci_high,
            "ci_entirely_below_zero": bool(ci_high < 0),
            "input_predictions_csv": str(NESTED_PREDICTIONS_CSV),
        }
        primary_df = pd.DataFrame([primary_row])
        primary_df.to_csv(data_dir / "nested_primary_wilcoxon.csv", index=False)
        pd.DataFrame({"bootstrap_diff": boot_diffs}).to_csv(
            data_dir / "nested_bootstrap_diffs_xgboost.csv", index=False
        )

        fold_table = pd.DataFrame({
            "outer_fold": pivot["outer_fold"],
            "test_cell": pivot["test_cell"],
            "mae_qrc": qrc_maes,
            "mae_xgboost": xgb_maes,
            "mae_qrc_pct": qrc_maes * 100.0,
            "mae_xgboost_pct": xgb_maes * 100.0,
            "diff_qrc_minus_xgb": qrc_maes - xgb_maes,
            "diff_qrc_minus_xgb_pct": (qrc_maes - xgb_maes) * 100.0,
            "qrc_better": qrc_maes < xgb_maes,
        })
        fold_table.to_csv(data_dir / "nested_per_fold_qrc_vs_xgb.csv", index=False)

        print("\nPrimary comparison: QRC vs XGBoost")
        print(f"  QRC mean MAE      = {qrc_maes.mean()*100:.3f}%")
        print(f"  XGBoost mean MAE  = {xgb_maes.mean()*100:.3f}%")
        print(f"  Wilcoxon p (2s)   = {primary_stats['p_value_twosided']:.6f}")
        print(f"  Wilcoxon p (1s)   = {primary_stats['p_value_less']:.6f}")
        print(f"  95% bootstrap CI  = [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%]")
        print(f"  QRC better folds  = {primary_stats['qrc_better_folds']}/{primary_stats['n_folds']}")

        rows: List[dict] = []
        for baseline in BASELINES:
            baseline_maes = pivot[baseline].to_numpy()
            stats = _wilcoxon_paired(qrc_maes, baseline_maes)
            mean_diff, base_ci_low, base_ci_high, diffs = _bootstrap_ci(qrc_maes, baseline_maes)
            rows.append({
                "baseline_model": baseline,
                "baseline_display": DISPLAY_NAMES.get(baseline, baseline),
                **stats,
                "mean_diff": mean_diff,
                "ci_low_95": base_ci_low,
                "ci_high_95": base_ci_high,
                "ci_entirely_below_zero": bool(base_ci_high < 0),
            })
            pd.DataFrame({"bootstrap_diff": diffs}).to_csv(
                data_dir / f"nested_bootstrap_diffs_{baseline}.csv", index=False
            )

        all_stats_df = pd.DataFrame(rows).sort_values("mean_diff").reset_index(drop=True)
        all_stats_df.to_csv(data_dir / "nested_all_pairwise_stats.csv", index=False)

        print("\nAll pairwise comparisons:")
        display_df = all_stats_df[[
            "baseline_model", "mean_qrc_mae", "mean_baseline_mae",
            "mean_diff", "ci_low_95", "ci_high_95", "p_value_twosided",
        ]].copy()
        print(display_df.to_string(index=False))

        _plot_primary_forest(
            fold_table=fold_table,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=primary_stats["p_value_twosided"],
            plot_dir=plot_dir,
        )
        _plot_ci_forest(all_stats_df, plot_dir)
        _write_summary(primary_df.iloc[0], all_stats_df, data_dir)

        print(f"\nSaved: {data_dir / 'nested_primary_wilcoxon.csv'}")
        print(f"Saved: {data_dir / 'nested_all_pairwise_stats.csv'}")
        print(f"Saved: {data_dir / 'nested_per_fold_qrc_vs_xgb.csv'}")
        print(f"Completed: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
