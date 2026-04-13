"""Phase 5 Stage 5: Few-shot learning curves on Warwick DIB (24 cells).

Reviewer §5.1.4:
    "Few-shot learning curves on Warwick: You have 24 cells. Show QRC
    performance when training on 3, 6, 12, 18 cells. If QRC plateaus
    earlier (needs fewer training cells), that's the small-data story."

This script runs repeated random LOCO-like splits on Warwick where:
  - Training set size n_train ∈ {3, 6, 9, 12, 15, 18, 21}
  - Test set = remaining cells (all held out simultaneously)
  - Repeated K times with different random splits to get mean±std curves
  - Models: QRC (depth=1), XGBoost, Ridge, ESN, GP

The key finding to look for:
  - QRC achieves its performance plateau with fewer training cells
  - Classical models (especially XGBoost) need more cells to converge
  - This supports the "small-data advantage" narrative (reviewer §2.1)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import RANDOM_STATE
from phase_5.config import get_stage_paths
from phase_3.models import get_model_pipeline
from phase_4.qrc_model import QuantumReservoir
from data_loader_warwick import load_warwick_data
import phase_4.config as phase4_config
import phase_4.qrc_model as phase4_qrc_model
from contextlib import contextmanager

# ── Constants ──────────────────────────────────────────────────────────────────
N_PCA              = 6
WARWICK_TEMP       = "25degC"
WARWICK_SOC        = "50SOC"
WARWICK_NOMINAL_AH = 5.0
# Training cell counts to sweep
N_TRAIN_CELLS      = [3, 6, 9, 12, 15, 18, 21]
# Repetitions per n_train (random cell splits)
N_REPEATS          = 30
# Models to compare
MODELS             = ["qrc", "xgboost", "ridge", "esn", "gp"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_result_paths() -> Tuple[Path, Path]:
    return get_stage_paths("stage_5")


def _save_figure(fig: plt.Figure, plot_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / stem}.png")


def _fit_pca_in_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = N_PCA,
) -> Tuple[np.ndarray, np.ndarray]:
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


@contextmanager
def _patched_seed(seed: int):
    old_qrc = phase4_qrc_model.RANDOM_STATE
    old_cfg = phase4_config.RANDOM_STATE
    phase4_qrc_model.RANDOM_STATE = seed
    phase4_config.RANDOM_STATE    = seed
    try:
        yield
    finally:
        phase4_qrc_model.RANDOM_STATE = old_qrc
        phase4_config.RANDOM_STATE    = old_cfg


def _eval_single_split(
    cell_data: Dict[str, dict],
    train_cells: List[str],
    test_cells: List[str],
    model_name: str,
    seed: int,
) -> float:
    """Evaluate one (train_cells, test_cells) split, return macro MAE."""
    X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
    y_train     = np.concatenate([cell_data[c]["y"] for c in train_cells])
    X_test_raw  = np.vstack([cell_data[c]["X_raw"] for c in test_cells])
    y_test      = np.concatenate([cell_data[c]["y"] for c in test_cells])

    X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)

    if model_name == "qrc":
        with _patched_seed(seed):
            qrc = QuantumReservoir(depth=1, use_zz=True, observable_set="Z",
                                   add_random_rotations=True)
            qrc.fit(X_train_pca, y_train)
            y_pred = qrc.predict(X_test_pca)
    else:
        model = get_model_pipeline(model_name)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

    return mean_absolute_error(y_test, y_pred)


def run_fewshot_curves(
    cell_data: Dict[str, dict],
    cell_ids: List[str],
    models: List[str] = MODELS,
    n_train_cells: List[int] = N_TRAIN_CELLS,
    n_repeats: int = N_REPEATS,
    base_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run few-shot learning curve sweep.

    For each (model, n_train, repeat), randomly select n_train training cells,
    test on remaining cells, record MAE.

    Returns:
        DataFrame with columns: model, n_train, repeat, mae, mae_pct
    """
    rng   = np.random.RandomState(base_seed)
    rows: List[dict] = []
    total = len(models) * len(n_train_cells) * n_repeats
    done  = 0

    for model_name in models:
        for n_train in n_train_cells:
            if n_train >= len(cell_ids):
                continue
            for rep in range(n_repeats):
                seed = int(rng.randint(0, 2**31 - 1))
                # Random cell split
                shuffled = list(rng.permutation(cell_ids))
                train_cells = shuffled[:n_train]
                test_cells  = shuffled[n_train:]

                try:
                    mae = _eval_single_split(
                        cell_data, train_cells, test_cells, model_name, seed,
                    )
                except Exception as e:
                    print(f"  WARN: {model_name} n_train={n_train} rep={rep}: {e}")
                    mae = float("nan")

                rows.append({
                    "model":   model_name,
                    "n_train": n_train,
                    "repeat":  rep,
                    "mae":     mae,
                    "mae_pct": mae * 100.0,
                    "mae_ah":  mae * WARWICK_NOMINAL_AH,
                })
                done += 1
                if done % 50 == 0:
                    print(f"  Progress: {done}/{total}")

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "qrc":      "#1f77b4",
    "xgboost":  "#ff7f0e",
    "ridge":    "#2ca02c",
    "esn":      "#9467bd",
    "gp":       "#8c564b",
    "cnn1d":    "#e377c2",
    "mlp":      "#7f7f7f",
}


def plot_learning_curves(
    results: pd.DataFrame,
    plot_dir: Path,
    metric: str = "mae_pct",
    ylabel: str = "MAE (%)",
) -> None:
    """Plot mean±std learning curves per model."""
    summary = (
        results.groupby(["model", "n_train"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, grp in summary.groupby("model"):
        color = COLORS.get(model_name, None)
        lw    = 2.5 if model_name == "qrc" else 1.5
        ls    = "-"  if model_name == "qrc" else "--"
        ax.plot(grp["n_train"], grp["mean"], label=model_name,
                color=color, linewidth=lw, linestyle=ls, marker="o", markersize=4)
        ax.fill_between(
            grp["n_train"],
            grp["mean"] - grp["std"],
            grp["mean"] + grp["std"],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Number of training cells")
    ax.set_ylabel(ylabel)
    ax.set_title("Few-shot learning curves on Warwick DIB (24 cells)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(sorted(results["n_train"].unique()))
    fig.tight_layout()
    _save_figure(fig, plot_dir, "warwick_fewshot_learning_curves")


def plot_plateau_analysis(results: pd.DataFrame, plot_dir: Path) -> None:
    """Show relative performance (model MAE / QRC MAE) vs n_train.

    Values >1 mean the model is worse than QRC; if classical lines stay >1
    at low n_train but converge to 1 at high n_train, QRC has a data-efficiency
    advantage.
    """
    summary = (
        results.groupby(["model", "n_train"])["mae"]
        .mean()
        .reset_index()
    )
    qrc_mean = summary[summary["model"] == "qrc"].set_index("n_train")["mae"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, grp in summary.groupby("model"):
        if model_name == "qrc":
            continue
        ratio = grp.set_index("n_train")["mae"] / qrc_mean
        color = COLORS.get(model_name, None)
        ax.plot(ratio.index, ratio.values, label=model_name,
                color=color, linewidth=1.5, linestyle="--", marker="s", markersize=4)

    ax.axhline(1.0, color="steelblue", linewidth=2.0, linestyle="-", label="QRC (=1)")
    ax.set_xlabel("Number of training cells")
    ax.set_ylabel("MAE ratio (model / QRC)")
    ax.set_title("Data-efficiency advantage of QRC vs classical\n"
                 "(ratio > 1 = model worse than QRC)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(sorted(results["n_train"].unique()))
    fig.tight_layout()
    _save_figure(fig, plot_dir, "warwick_fewshot_plateau_analysis")


# ── Plateau analysis ───────────────────────────────────────────────────────────

def _find_plateau_cells(
    results: pd.DataFrame,
    within_pct: float = 5.0,
    metric: str = "mae",
) -> pd.DataFrame:
    """Find the minimum n_train at which each model reaches its plateau MAE.

    A model is considered to have plateaued at n_train=k when its mean MAE
    at k is within ``within_pct``% of the model's own minimum mean MAE across
    all n_train values.  This threshold is applied multiplicatively:

        plateau_threshold = min_mae * (1 + within_pct / 100)

    The plateau n_train is the smallest k where mean_mae(k) <= plateau_threshold.
    If a model never falls below the threshold, plateau_n is set to the maximum
    observed n_train (i.e., the model has not plateaued yet).

    Args:
        results:     DataFrame from ``run_fewshot_curves`` with columns
                     ["model", "n_train", "repeat", metric].
        within_pct:  Convergence tolerance in percent (default 5%).
        metric:      Column to use (default "mae").

    Returns:
        DataFrame with columns:
            model, plateau_n, min_mae, plateau_threshold, n_train_values,
            mae_at_plateau, pct_above_min
    """
    summary = (
        results.groupby(["model", "n_train"])[metric]
        .mean()
        .reset_index(name="mean_mae")
    )
    rows: List[dict] = []
    for model_name, grp in summary.groupby("model"):
        grp = grp.sort_values("n_train")
        min_mae       = float(grp["mean_mae"].min())
        threshold     = min_mae * (1.0 + within_pct / 100.0)
        plateau_mask  = grp["mean_mae"] <= threshold
        n_train_vals  = sorted(grp["n_train"].tolist())

        if plateau_mask.any():
            plateau_row   = grp[plateau_mask].iloc[0]
            plateau_n     = int(plateau_row["n_train"])
            mae_at_plateau = float(plateau_row["mean_mae"])
        else:
            # Model never converges — use last n_train
            plateau_n      = n_train_vals[-1]
            mae_at_plateau = float(grp.loc[grp["n_train"] == plateau_n, "mean_mae"].values[0])

        pct_above = (mae_at_plateau - min_mae) / min_mae * 100 if min_mae > 0 else float("nan")
        rows.append({
            "model":             model_name,
            "plateau_n":         plateau_n,
            "min_mae":           round(min_mae, 6),
            "plateau_threshold": round(threshold, 6),
            "mae_at_plateau":    round(mae_at_plateau, 6),
            "pct_above_min":     round(pct_above, 2),
            "n_train_values":    str(n_train_vals),
        })
    return pd.DataFrame(rows).sort_values("plateau_n")


def print_plateau_summary(plateau_df: pd.DataFrame, within_pct: float = 5.0) -> None:
    """Print a human-readable plateau analysis comparison."""
    print(f"\n  === Plateau Analysis (within {within_pct}% of converged MAE) ===")
    print(f"  {'Model':12s}  {'Plateau n':10s}  {'Min MAE':9s}  {'MAE@plateau':11s}  {'% above min':11s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*9}  {'-'*11}  {'-'*11}")
    for _, row in plateau_df.iterrows():
        marker = " <-- QRC" if row["model"] == "qrc" else ""
        print(f"  {row['model']:12s}  {row['plateau_n']:10d}  {row['min_mae']:9.4f}  "
              f"{row['mae_at_plateau']:11.4f}  {row['pct_above_min']:9.2f}%{marker}")

    # Direct QRC vs best-classical comparison
    if "qrc" in plateau_df["model"].values:
        qrc_n = int(plateau_df.loc[plateau_df["model"] == "qrc", "plateau_n"].values[0])
        classical_rows = plateau_df[plateau_df["model"] != "qrc"]
        if not classical_rows.empty:
            best_cls  = classical_rows.loc[classical_rows["plateau_n"].idxmin()]
            speedup   = best_cls["plateau_n"] / qrc_n if qrc_n > 0 else float("nan")
            print(f"\n  [KEY] QRC plateau_n = {qrc_n} cells")
            print(f"  [KEY] Best classical ({best_cls['model']}) plateau_n = {best_cls['plateau_n']} cells")
            if speedup > 1:
                print(f"  [KEY] QRC needs {speedup:.1f}x fewer cells to plateau "
                      f"({qrc_n} vs {best_cls['plateau_n']})")
            else:
                print(f"  [NOTE] QRC does not plateau earlier than the best classical model in this sweep.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    n_repeats: int = N_REPEATS,
    models: List[str] = MODELS,
) -> None:
    data_dir, plot_dir = _get_result_paths()
    print("=" * 70)
    print("Phase 5-Stage 5: Few-shot Learning Curves on Warwick DIB")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"n_train_cells = {N_TRAIN_CELLS}")
    print(f"n_repeats     = {n_repeats}")
    print(f"models        = {models}")

    print("\nLoading Warwick data...")
    raw = load_warwick_data(temp=WARWICK_TEMP, soc=WARWICK_SOC)
    cell_ids = sorted(raw.keys())
    print(f"  {len(cell_ids)} cells loaded")

    print("\nRunning few-shot sweep...")
    results = run_fewshot_curves(
        raw, cell_ids,
        models=models,
        n_train_cells=N_TRAIN_CELLS,
        n_repeats=n_repeats,
    )
    results.to_csv(data_dir / "warwick_fewshot_results.csv", index=False)
    print(f"Saved results: {data_dir / 'warwick_fewshot_results.csv'}")

    # Summary table
    summary = (
        results.groupby(["model", "n_train"])["mae_pct"]
        .agg(["mean", "std"])
        .round(3)
        .reset_index()
    )
    print("\nFew-shot summary (MAE%):")
    print(summary.to_string(index=False))

    # ── Quantitative plateau analysis ──────────────────────────────────────────
    plateau_df = _find_plateau_cells(results, within_pct=5.0)
    plateau_df.to_csv(data_dir / "warwick_fewshot_plateau.csv", index=False)
    print_plateau_summary(plateau_df, within_pct=5.0)
    print(f"\n  Plateau table saved: {data_dir / 'warwick_fewshot_plateau.csv'}")

    print("\nGenerating plots...")
    plot_learning_curves(results, plot_dir)
    plot_plateau_analysis(results, plot_dir)

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Results: {data_dir}")
    print(f"Plots:   {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot learning curves on Warwick DIB."
    )
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    main(n_repeats=args.n_repeats, models=args.models)
