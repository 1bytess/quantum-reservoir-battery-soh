"""Phase 5 Stage 2b: Statistical Significance Testing on Warwick DIB (24 cells).

Reviewer §1.2 & §5.1.6:
    "Statistical tests were done on the Stanford dataset (61 samples, 6 cells).
    Your strongest results are actually on Warwick (24 cells). Redo the full
    statistical significance analysis on Warwick where QRC shows 38% improvement
    over the best classical model."

This script runs Warwick LOCO (24-cell leave-one-cell-out) with:
  - QRC (noiseless, depth=1, Z+ZZ observables) — 20 random seeds
  - XGBoost, Ridge, ESN as classical comparisons
  - Wilcoxon signed-rank test on paired per-cell MAEs
  - Bootstrap 95% CI on cell-level MAE difference
  - Cohen's d effect size

Warwick has one sample per cell (24 cells × 1 EIS spectrum each at 25°C/50SOC),
so the paired test is cell-level (N=24 paired differences).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import RANDOM_STATE
from phase_5.config import get_stage_paths
import phase_3.models as phase3_models
import phase_4.config as phase4_config
import phase_4.qrc_model as phase4_qrc_model
from phase_3.models import get_model_pipeline
from phase_4.qrc_model import QuantumReservoir
from data_loader_warwick import load_warwick_data

# ── Constants ──────────────────────────────────────────────────────────────────
N_PCA = 6
WARWICK_NOMINAL_CAP_AH = 5.0   # NMC811 INR21700-50E
WARWICK_TEMP = "25degC"
WARWICK_SOC  = "50SOC"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_result_paths() -> Tuple[Path, Path]:
    return get_stage_paths("stage_2b_warwick")


def _save_figure(fig: plt.Figure, plot_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = plot_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_dir / stem}.png")


def _fit_pca_in_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = N_PCA,
) -> Tuple[np.ndarray, np.ndarray]:
    """In-fold PCA — leakage-free."""
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _load_warwick_cell_data() -> Tuple[Dict[str, Dict], List[str]]:
    """Load Warwick 24-cell data and return (cell_data, cell_ids)."""
    raw = load_warwick_data(temp=WARWICK_TEMP, soc=WARWICK_SOC)
    # raw[cell_id] = {"X_raw": (1, 122), "y": (1,), ...}
    cell_ids = sorted(raw.keys())
    return raw, cell_ids


@contextmanager
def _patched_qrc_seed(seed: int):
    old_qrc  = phase4_qrc_model.RANDOM_STATE
    old_cfg  = phase4_config.RANDOM_STATE
    phase4_qrc_model.RANDOM_STATE = seed
    phase4_config.RANDOM_STATE    = seed
    try:
        yield
    finally:
        phase4_qrc_model.RANDOM_STATE = old_qrc
        phase4_config.RANDOM_STATE    = old_cfg


def _safe_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std  = diff.std(ddof=1)
    return float("nan") if np.isclose(std, 0.0) else float(diff.mean() / std)


# ── QRC LOCO on Warwick ────────────────────────────────────────────────────────

def _run_qrc_loco_warwick(
    cell_data: Dict[str, Dict],
    cell_ids: List[str],
    seed: int,
) -> pd.DataFrame:
    """Run one QRC LOCO pass on Warwick with a specific seed."""
    rows: List[Dict] = []
    with _patched_qrc_seed(seed):
        for test_cell in cell_ids:
            train_cells = [c for c in cell_ids if c != test_cell]

            X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
            y_train     = np.concatenate([cell_data[c]["y"] for c in train_cells])
            X_test_raw  = cell_data[test_cell]["X_raw"]
            y_test      = cell_data[test_cell]["y"]

            X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)
            train_groups = np.array([c for c in train_cells for _ in range(len(cell_data[c]["y"]))])

            qrc = QuantumReservoir(
                depth=1,
                use_zz=True,
                observable_set="Z",
                add_random_rotations=True,
            )
            qrc.fit(X_train_pca, y_train, groups=train_groups)
            y_pred    = qrc.predict(X_test_pca)
            abs_error = np.abs(y_test - y_pred)

            for i in range(len(y_test)):
                rows.append({
                    "model":     "qrc_noiseless_d1",
                    "seed":      seed,
                    "test_cell": test_cell,
                    "sample_idx": i,
                    "y_true":    float(y_test[i]),
                    "y_pred":    float(y_pred[i]),
                    "abs_error": float(abs_error[i]),
                    "mae_pct":   float(abs_error[i]) * 100.0,
                    "mae_ah":    float(abs_error[i]) * WARWICK_NOMINAL_CAP_AH,
                })
    return pd.DataFrame(rows)


def _run_qrc_multiseed_warwick(
    cell_data: Dict[str, Dict],
    cell_ids: List[str],
    seeds: Iterable[int],
) -> pd.DataFrame:
    frames = []
    seeds_list = list(seeds)
    for idx, seed in enumerate(seeds_list, 1):
        print(f"  QRC seed {seed} ({idx}/{len(seeds_list)})...")
        frame = _run_qrc_loco_warwick(cell_data, cell_ids, seed)
        macro = frame.groupby("test_cell")["abs_error"].mean().mean()
        print(f"    seed macro_mae={macro:.6f}  ({macro*100:.3f}%)")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ── Classical LOCO on Warwick ──────────────────────────────────────────────────

def _run_classical_loco_warwick(
    cell_data: Dict[str, Dict],
    cell_ids: List[str],
    model_name: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]
        X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
        y_train     = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_raw  = cell_data[test_cell]["X_raw"]
        y_test      = cell_data[test_cell]["y"]

        X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)
        model = get_model_pipeline(model_name)
        model.fit(X_train_pca, y_train)
        y_pred    = model.predict(X_test_pca)
        abs_error = np.abs(y_test - y_pred)

        for i in range(len(y_test)):
            rows.append({
                "model":     model_name,
                "seed":      RANDOM_STATE,
                "test_cell": test_cell,
                "sample_idx": i,
                "y_true":    float(y_test[i]),
                "y_pred":    float(y_pred[i]),
                "abs_error": float(abs_error[i]),
                "mae_pct":   float(abs_error[i]) * 100.0,
                "mae_ah":    float(abs_error[i]) * WARWICK_NOMINAL_CAP_AH,
            })
    return pd.DataFrame(rows)


# ── Statistical tests ─────────────────────────────────────────────────────────

def _bootstrap_cell_level_ci(
    qrc_cell_mae: np.ndarray,
    classical_cell_mae: np.ndarray,
    cell_ids: List[str],
    n_resamples: int = 10_000,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float, float, np.ndarray]:
    """Cell-level bootstrap on MAE differences.

    Resamples cells (not samples) to respect the correlation structure of
    the LOCO evaluation.
    """
    rng  = np.random.RandomState(seed)
    n    = len(cell_ids)
    diff = qrc_cell_mae - classical_cell_mae   # positive = QRC worse

    boot_diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx             = rng.randint(0, n, size=n)
        boot_diffs[b]   = diff[idx].mean()

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return float(diff.mean()), float(ci_low), float(ci_high), boot_diffs


def _run_significance_tests(
    qrc_cell_mae: np.ndarray,
    classical_cell_mae: np.ndarray,
    cell_ids: List[str],
    classical_name: str,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    """Full significance test suite: Wilcoxon, bootstrap CI, Cohen d."""
    wc = wilcoxon(qrc_cell_mae, classical_cell_mae, alternative="two-sided",
                  zero_method="wilcox")
    d  = _cohens_d_paired(qrc_cell_mae, classical_cell_mae)
    boot_mean, ci_low, ci_high, _ = _bootstrap_cell_level_ci(
        qrc_cell_mae, classical_cell_mae, cell_ids, n_resamples=n_resamples,
    )
    mean_diff = float((qrc_cell_mae - classical_cell_mae).mean())

    rows = [
        {"metric": "n_cells",                       "value": len(cell_ids)},
        {"metric": "mean_mae_qrc",                  "value": float(qrc_cell_mae.mean())},
        {"metric": f"mean_mae_{classical_name}",    "value": float(classical_cell_mae.mean())},
        {"metric": "mae_pct_qrc",                   "value": float(qrc_cell_mae.mean()) * 100},
        {"metric": f"mae_pct_{classical_name}",     "value": float(classical_cell_mae.mean()) * 100},
        {"metric": "mae_diff_qrc_minus_classical",  "value": mean_diff},
        {"metric": "pct_improvement",               "value": -mean_diff / classical_cell_mae.mean() * 100},
        {"metric": "wilcoxon_statistic",            "value": float(wc.statistic)},
        {"metric": "wilcoxon_p_value",              "value": float(wc.pvalue)},
        {"metric": "bootstrap_mean_diff",           "value": boot_mean},
        {"metric": "bootstrap_ci_low_95",           "value": ci_low},
        {"metric": "bootstrap_ci_high_95",          "value": ci_high},
        {"metric": "cohens_d_paired",               "value": d},
        {"metric": "significant_at_0.05",           "value": float(wc.pvalue < 0.05)},
    ]
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_paired_cell_mae(
    cell_ids: List[str],
    qrc_cell_mae: np.ndarray,
    classical_maes: Dict[str, np.ndarray],
    plot_dir: Path,
) -> None:
    """Paired bar chart of per-cell MAE (QRC vs classical) on Warwick."""
    n_cells    = len(cell_ids)
    n_models   = 1 + len(classical_maes)
    bar_width  = 0.8 / n_models
    x          = np.arange(n_cells)
    colors     = ["#1f77b4"] + ["#aec7e8", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(max(12, n_cells), 5))
    for i, (name, mae) in enumerate([("QRC (depth=1)", qrc_cell_mae)] +
                                     list(classical_maes.items())):
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, mae * 100, width=bar_width * 0.9,
               label=name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(cell_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (% of SOH)")
    ax.set_title("Per-cell LOCO MAE on Warwick DIB (24 cells, 25°C/50SOC)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, plot_dir, "warwick_per_cell_mae")


def _plot_parity(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    model_label: str,
    plot_dir: Path,
    stem: str = "warwick_parity",
) -> None:
    """Predicted vs actual parity plot with 45° line and ±2% error band."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(all_y_true * 100, all_y_pred * 100, s=30, alpha=0.7,
               edgecolors="k", linewidths=0.4, label=model_label)

    lims = [min(all_y_true.min(), all_y_pred.min()) * 100 - 1,
            max(all_y_true.max(), all_y_pred.max()) * 100 + 1]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect")
    ax.fill_between(lims, [l - 2 for l in lims], [l + 2 for l in lims],
                    alpha=0.12, color="gray", label="±2% band")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    ax.set_title(f"Parity plot — {model_label} (Warwick LOCO)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_figure(fig, plot_dir, stem)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_seeds: int = 20, n_bootstrap: int = 10_000) -> None:
    data_dir, plot_dir = _get_result_paths()
    log_path = data_dir / "stage_2b_warwick_log.txt"

    orig_stdout = sys.stdout
    log_f = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def write(self, m):  orig_stdout.write(m); log_f.write(m); log_f.flush()
        def flush(self):     orig_stdout.flush(); log_f.flush()

    sys.stdout = _Tee()
    try:
        print("=" * 70)
        print("Phase 5b-Warwick: Statistical Significance (Warwick 24 cells)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"n_seeds={n_seeds}, n_bootstrap={n_bootstrap}")

        # ── Load data ──
        print("\nLoading Warwick DIB data (25°C / 50% SOC)...")
        cell_data, cell_ids = _load_warwick_cell_data()
        print(f"  Loaded {len(cell_ids)} cells: {cell_ids[:5]} ...")

        # ── QRC multi-seed ──
        print(f"\nRunning QRC LOCO ({n_seeds} seeds)...")
        seeds = [RANDOM_STATE + i for i in range(n_seeds)]
        qrc_samples = _run_qrc_multiseed_warwick(cell_data, cell_ids, seeds)
        qrc_samples.to_csv(data_dir / "warwick_qrc_multiseed.csv", index=False)

        # Aggregate: per-cell MAE averaged over seeds
        qrc_cell_mae_df = (
            qrc_samples.groupby(["seed", "test_cell"])["abs_error"].mean()
            .reset_index(name="cell_mae")
        )
        qrc_mean_cell_mae = (
            qrc_cell_mae_df.groupby("test_cell")["cell_mae"].mean()
        )
        # Seed=RANDOM_STATE for paired tests
        qrc_seed0 = qrc_samples[qrc_samples["seed"] == RANDOM_STATE]
        qrc_cell_mae_seed0 = (
            qrc_seed0.groupby("test_cell")["abs_error"].mean()
        )

        # ── Classical ──
        print("\nRunning classical LOCO on Warwick...")
        classical_results: Dict[str, pd.DataFrame] = {}
        for cls_name in ["xgboost", "ridge", "esn"]:
            print(f"  {cls_name}...")
            df = _run_classical_loco_warwick(cell_data, cell_ids, cls_name)
            classical_results[cls_name] = df
            macro = df.groupby("test_cell")["abs_error"].mean().mean()
            print(f"    macro_mae={macro:.6f}  ({macro*100:.3f}%)")
            df.to_csv(data_dir / f"warwick_{cls_name}_loco.csv", index=False)

        classical_cell_maes = {
            name: df.groupby("test_cell")["abs_error"].mean()
            for name, df in classical_results.items()
        }

        # ── Statistical tests (QRC vs each classical) ──
        # Three comparisons (QRC vs XGBoost, Ridge, ESN).
        #
        # Correction strategy: Holm-Bonferroni (step-down, Holm 1979).
        # Holm-Bonferroni is uniformly more powerful than Bonferroni while
        # still controlling family-wise error rate (FWER) at α=0.05.
        #
        # Procedure:
        #   1. Sort raw p-values in ascending order: p_(1) ≤ p_(2) ≤ p_(3)
        #   2. Compare p_(k) against threshold α / (m − k + 1), m=3:
        #        rank 1 → α/3 = 0.0167  (same as Bonferroni)
        #        rank 2 → α/2 = 0.025
        #        rank 3 → α/1 = 0.050
        #   3. Reject H_(k) iff ALL p_(j) ≤ α/(m−j+1) for j ≤ k
        #      (i.e., stop at first non-significant result).
        #   4. Holm-adjusted p-value: p_holm_(k) = min(1, max_{j≤k} (m−j+1)·p_(j))
        #
        # For context: Bonferroni adjusts ALL comparisons by ×3 regardless of
        # rank, making it more conservative for comparisons 2 and 3.
        # A QRC vs XGBoost raw p=0.032 fails Bonferroni (0.096) but can survive
        # Holm if the other two comparisons are strongly significant first.
        print("\nRunning significance tests (Holm-Bonferroni FWER correction)...")
        n_comparisons = len(classical_cell_maes)

        # --- Pass 1: run all Wilcoxon tests, collect raw p-values and intermediate results ---
        comparison_data = []  # (cls_name, shared_cells, qrc_arr, cls_arr, stats_df, raw_p)
        for cls_name, cls_cell_mae in classical_cell_maes.items():
            shared_cells = sorted(set(qrc_cell_mae_seed0.index) & set(cls_cell_mae.index))
            qrc_arr = qrc_cell_mae_seed0[shared_cells].values
            cls_arr = cls_cell_mae[shared_cells].values

            stats_df = _run_significance_tests(
                qrc_arr, cls_arr, shared_cells, cls_name, n_resamples=n_bootstrap,
            )
            stats_df["comparison"] = f"QRC_vs_{cls_name}"
            raw_p = float(stats_df.loc[stats_df["metric"] == "wilcoxon_p_value", "value"].values[0])
            comparison_data.append((cls_name, shared_cells, qrc_arr, cls_arr, stats_df, raw_p))

        # --- Pass 2: apply Holm-Bonferroni jointly across all raw p-values ---
        raw_ps = np.array([raw_p for *_, raw_p in comparison_data])
        # Holm-adjusted p-values (vectorised):
        #   p_holm_(k) = min(1, max_{j=1..k}( (m-j+1) * p_(j) ))
        #   where indices are over the rank-sorted order.
        sort_idx     = np.argsort(raw_ps)          # ascending rank order
        holm_adj     = np.empty(n_comparisons)
        running_max  = 0.0
        for rank, orig_idx in enumerate(sort_idx):
            adjusted   = (n_comparisons - rank) * raw_ps[orig_idx]
            running_max = max(running_max, adjusted)
            holm_adj[orig_idx] = min(1.0, running_max)
        holm_significant = holm_adj < 0.05

        # Also keep simple Bonferroni for reference (not used for primary inference)
        bonf_adj = np.minimum(raw_ps * n_comparisons, 1.0)

        # --- Pass 3: append correction rows and print ---
        all_stats = []
        for i, (cls_name, shared_cells, qrc_arr, cls_arr, stats_df, raw_p) in enumerate(comparison_data):
            rank_of_this = int(np.where(sort_idx == i)[0][0]) + 1   # 1-based
            correction_rows = pd.DataFrame([
                {"metric": "n_comparisons",              "value": float(n_comparisons)},
                {"metric": "wilcoxon_p_bonferroni",      "value": float(bonf_adj[i])},
                {"metric": "wilcoxon_p_holm",            "value": float(holm_adj[i])},
                {"metric": "holm_rank",                  "value": float(rank_of_this)},
                {"metric": "holm_threshold_at_rank",     "value": 0.05 / (n_comparisons - rank_of_this + 1)},
                {"metric": "significant_bonferroni_0.05","value": float(bonf_adj[i] < 0.05)},
                {"metric": "significant_holm_0.05",      "value": float(holm_significant[i])},
            ])
            correction_rows["comparison"] = f"QRC_vs_{cls_name}"
            stats_df = pd.concat([stats_df, correction_rows], ignore_index=True)
            all_stats.append(stats_df)

            print(f"\n  QRC vs {cls_name} (N_cells={len(shared_cells)}):")
            for _, row in stats_df.iterrows():
                print(f"    {row['metric']:42s}: {row['value']:.6f}")

        print(f"\n  [FWER] Holm-Bonferroni correction over {n_comparisons} comparisons:")
        for i, (cls_name, *_, raw_p) in enumerate(comparison_data):
            rank_of_this = int(np.where(sort_idx == i)[0][0]) + 1
            threshold    = 0.05 / (n_comparisons - rank_of_this + 1)
            sig_marker   = "[SIGNIFICANT]" if holm_significant[i] else "[not sig]"
            print(f"    QRC vs {cls_name:10s}  raw_p={raw_p:.4f}  "
                  f"Holm_p={holm_adj[i]:.4f}  "
                  f"threshold={threshold:.4f}  {sig_marker}")
        print(f"  [REF]  Bonferroni alpha = 0.05 / {n_comparisons} = {0.05/n_comparisons:.4f}")

        stats_combined = pd.concat(all_stats, ignore_index=True)
        stats_combined.to_csv(data_dir / "warwick_significance_tests.csv", index=False)

        # ── Summary table ──
        print("\nCell-level MAE summary (Warwick, seed=42):")
        summary_rows = []
        for cell in cell_ids:
            row = {"cell": cell, "qrc_mae": qrc_cell_mae_seed0.get(cell, float("nan"))}
            for cls_name, cls_cell_mae in classical_cell_maes.items():
                row[f"{cls_name}_mae"] = cls_cell_mae.get(cell, float("nan"))
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(data_dir / "warwick_per_cell_mae_table.csv", index=False)
        print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        # ── Plots ──
        print("\nGenerating plots...")
        classical_mae_arrays = {
            name: classical_cell_maes[name].reindex(cell_ids).values
            for name in classical_results
        }
        qrc_arr_all = qrc_cell_mae_seed0.reindex(cell_ids).values
        _plot_paired_cell_mae(cell_ids, qrc_arr_all, classical_mae_arrays, plot_dir)

        # Parity plot for QRC
        _plot_parity(
            qrc_seed0["y_true"].values,
            qrc_seed0["y_pred"].values,
            "QRC (depth=1, noiseless)",
            plot_dir,
            stem="warwick_qrc_parity",
        )
        # Parity plot for best classical (XGBoost)
        xgb_df = classical_results["xgboost"]
        _plot_parity(
            xgb_df["y_true"].values,
            xgb_df["y_pred"].values,
            "XGBoost",
            plot_dir,
            stem="warwick_xgb_parity",
        )

        print(f"\nCompleted: {datetime.now().isoformat()}")
        print(f"Results in: {data_dir}")
        print(f"Plots in:   {plot_dir}")
        print(f"Log:        {log_path}")
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5b-Warwick: Statistical significance on 24-cell LOCO."
    )
    parser.add_argument("--n-seeds",     type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    main(n_seeds=args.n_seeds, n_bootstrap=args.n_bootstrap)
