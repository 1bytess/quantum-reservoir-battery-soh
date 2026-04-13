"""Phase 10: Cross-dataset transfer learning experiment.

Reviewer §5.1.3:
    "Transfer learning experiment: Train QRC on Stanford, predict on Warwick
    (or vice versa). If the quantum kernel transfers better than classical
    kernels, that's a major finding. Phase 9 already hints at this with
    cross-dataset kernel consistency."

Experimental design:
  Four zero-shot transfer directions are tested:
    A. Stanford → Warwick   (train on 61 Stanford EIS-SOH pairs, test all 24 Warwick cells)
    B. Warwick  → Stanford  (train on 24 Warwick cells, test 6 Stanford cells LOCO)
    C. Stanford → ESCL      (train on 61 Stanford pairs, test ESCL temporal sequences)
    D. Warwick  → ESCL      (train on 24 Warwick cells, test ESCL)

  For each direction:
    - PCA fit on source dataset, projected to 6D, then transferred to target
      (NOTE: this introduces cross-dataset PCA projection — document explicitly)
    - Also tested: separate in-dataset PCA (controls for projection leakage)
    - Models: QRC (depth=1), XGBoost, Ridge, GP
    - Metric: MAE, MAE%, R² on target dataset

  Key hypothesis (reviewer §9.3):
    "If the quantum kernel transfers better than classical kernels, this
    supports the claim that QRC learns a more universal EIS feature space."

Dataset chemistry notes:
    Stanford  — LCO 18650,       25°C, 6 cells, 61 EIS-SOH pairs
    Warwick   — NMC811 21700,    25°C, 24 cells, 1 EIS/cell
    ESCL      — Samsung 25R NMC, RT,   1 cell, 11 temporal EIS snapshots
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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import PROJECT_ROOT, RANDOM_STATE, CELL_IDS as STANFORD_CELL_IDS, NOMINAL_CAPACITIES
from data_loader import load_stanford_data
from data_loader_warwick import load_warwick_data
from phase_3.models import get_model_pipeline
from phase_4.qrc_model import QuantumReservoir
import phase_4.config as phase4_config
import phase_4.qrc_model as phase4_qrc_model
from contextlib import contextmanager

# ── Constants ──────────────────────────────────────────────────────────────────
N_PCA          = 6
NOMINAL_CAP    = NOMINAL_CAPACITIES  # alias — single source of truth in src/config.py
MODELS         = ["qrc", "xgboost", "ridge", "gp"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_result_paths() -> Tuple[Path, Path]:
    base     = PROJECT_ROOT / "result" / "phase_10"
    data_dir = base / "data"
    plot_dir = base / "plot"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir


def _save_figure(fig, plot_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / stem}.png")


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, nominal_ah: float) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {
        "mae":     mae,
        "mae_pct": mae * 100.0,
        "mae_ah":  mae * nominal_ah,
        "r2":      r2,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_stanford() -> Tuple[np.ndarray, np.ndarray]:
    """Load Stanford as flat (X_raw, y) arrays."""
    cell_data = load_stanford_data()
    X = np.vstack([cell_data[c]["X_raw"] for c in STANFORD_CELL_IDS])
    y = np.concatenate([cell_data[c]["y"] for c in STANFORD_CELL_IDS])
    return X, y


def _load_warwick_flat() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Warwick (25°C/50SOC) as flat arrays."""
    raw      = load_warwick_data(temp="25degC", soc="50SOC")
    cell_ids = sorted(raw.keys())
    X = np.vstack([raw[c]["X_raw"] for c in cell_ids])
    y = np.concatenate([raw[c]["y"]   for c in cell_ids])
    return X, y, cell_ids


def _load_escl_flat() -> Tuple[np.ndarray, np.ndarray]:
    """Load ESCL lab data as flat arrays (temporal sequences of one cell)."""
    from data_loader_lab import load_lab_data
    cell_data = load_lab_data()
    X = np.vstack([v["X_raw"] for v in cell_data.values()])
    y = np.concatenate([v["y"]   for v in cell_data.values()])
    return X, y


# ── PCA projection strategies ─────────────────────────────────────────────────

def _fit_source_pca(
    X_source: np.ndarray,
    n_components: int = N_PCA,
) -> PCA:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(X_source)
    return pca


def _project(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Project X using a pre-fit PCA. Handles shape mismatches via zero-padding."""
    n_features_pca = pca.components_.shape[1]
    n_features_X   = X.shape[1]
    if n_features_X != n_features_pca:
        # Datasets differ in EIS frequency count — align by truncating/padding
        if n_features_X < n_features_pca:
            pad = np.zeros((X.shape[0], n_features_pca - n_features_X))
            X   = np.hstack([X, pad])
        else:
            X = X[:, :n_features_pca]
    return pca.transform(X)


# ── Transfer evaluation ────────────────────────────────────────────────────────

def _eval_transfer(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    model_name: str,
    source_name: str,
    target_name: str,
    pca_strategy: str = "source",   # "source" | "target" | "separate"
    seed: int = RANDOM_STATE,
) -> dict:
    """Train on source, predict on target.

    pca_strategy:
        "source"   — PCA fit on source only (cross-dataset projection).
        "target"   — PCA fit on target only (no leakage, but unusual).
        "separate" — Independent in-distribution PCA per dataset.
                     Source PCA used for training, target PCA for test.
                     This is the most methodologically sound approach.
    """
    if pca_strategy == "source":
        pca         = _fit_source_pca(X_source)
        X_train_pca = pca.transform(X_source)
        X_test_pca  = _project(pca, X_target)
    elif pca_strategy == "target":
        pca         = _fit_source_pca(X_target)
        X_train_pca = _project(pca, X_source)
        X_test_pca  = pca.transform(X_target)
    elif pca_strategy == "separate":
        pca_src     = _fit_source_pca(X_source)
        pca_tgt     = _fit_source_pca(X_target)
        X_train_pca = pca_src.transform(X_source)
        X_test_pca  = pca_tgt.transform(X_target)
    else:
        raise ValueError(f"Unknown pca_strategy: {pca_strategy!r}")

    if model_name == "qrc":
        with _patched_seed(seed):
            model = QuantumReservoir(
                depth=1, use_zz=True, observable_set="Z", add_random_rotations=True,
            )
            model.fit(X_train_pca, y_source)
            y_pred = model.predict(X_test_pca)
    else:
        model = get_model_pipeline(model_name)
        model.fit(X_train_pca, y_source)
        y_pred = model.predict(X_test_pca)

    m = _metrics(y_target, y_pred, NOMINAL_CAP.get(target_name, 1.0))
    m.update({
        "model":        model_name,
        "source":       source_name,
        "target":       target_name,
        "pca_strategy": pca_strategy,
        "n_train":      len(y_source),
        "n_test":       len(y_target),
        "seed":         seed,
    })
    return m


def run_all_transfers(
    models: List[str] = MODELS,
    n_seeds: int = 5,
) -> pd.DataFrame:
    """Run all four transfer directions × all models × multiple seeds.

    Returns:
        DataFrame with one row per (direction, model, seed, pca_strategy).
    """
    rows = []

    print("Loading datasets...")
    X_stan, y_stan              = _load_stanford()
    X_warw, y_warw, warw_ids    = _load_warwick_flat()
    print(f"  Stanford:  X={X_stan.shape}, y={y_stan.shape}")
    print(f"  Warwick:   X={X_warw.shape}, y={y_warw.shape}")

    escl_available = True
    try:
        X_escl, y_escl = _load_escl_flat()
        print(f"  ESCL Lab:  X={X_escl.shape}, y={y_escl.shape}")
    except Exception as e:
        print(f"  ESCL Lab:  UNAVAILABLE ({e})")
        escl_available = False

    directions = [
        ("stanford", "warwick",  X_stan, y_stan, X_warw, y_warw),
        ("warwick",  "stanford", X_warw, y_warw, X_stan, y_stan),
    ]
    if escl_available:
        directions += [
            ("stanford", "escl", X_stan, y_stan, X_escl, y_escl),
            ("warwick",  "escl", X_warw, y_warw, X_escl, y_escl),
        ]

    seeds = [RANDOM_STATE + i for i in range(n_seeds)]

    for src_name, tgt_name, X_src, y_src, X_tgt, y_tgt in directions:
        print(f"\nTransfer: {src_name} → {tgt_name}")
        for model_name in models:
            for pca_strat in ("source", "separate"):
                for seed in seeds:
                    try:
                        row = _eval_transfer(
                            X_src, y_src, X_tgt, y_tgt,
                            model_name=model_name,
                            source_name=src_name,
                            target_name=tgt_name,
                            pca_strategy=pca_strat,
                            seed=seed,
                        )
                        rows.append(row)
                        print(f"  {model_name:10s} pca={pca_strat:8s} seed={seed}: "
                              f"MAE={row['mae']:.4f} ({row['mae_pct']:.2f}%) R²={row['r2']:.3f}")
                    except Exception as e:
                        print(f"  WARN: {model_name} {pca_strat} seed={seed}: {e}")

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "qrc":      "#1f77b4",
    "xgboost":  "#ff7f0e",
    "ridge":    "#2ca02c",
    "gp":       "#8c564b",
    "esn":      "#9467bd",
}


def plot_transfer_summary(results: pd.DataFrame, plot_dir: Path) -> None:
    """Grouped bar chart: MAE% per transfer direction and model."""
    pca_strat = "separate"   # cleanest methodological comparison
    df = results[results["pca_strategy"] == pca_strat].copy()
    directions = df["source"].str.cat(df["target"], sep="→").unique()

    fig, ax = plt.subplots(figsize=(max(10, len(directions) * 3), 5))
    n_models  = df["model"].nunique()
    bar_width = 0.8 / n_models
    x_base    = np.arange(len(directions))

    model_order = [m for m in MODELS if m in df["model"].unique()]
    for i, model_name in enumerate(model_order):
        maes = []
        stds = []
        for d in directions:
            src, tgt = d.split("→")
            sub = df[(df["model"] == model_name) & (df["source"] == src) & (df["target"] == tgt)]
            maes.append(sub["mae_pct"].mean() if len(sub) > 0 else float("nan"))
            stds.append(sub["mae_pct"].std()  if len(sub) > 1 else 0.0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x_base + offset, maes, width=bar_width * 0.9, yerr=stds, capsize=3,
               label=model_name, color=COLORS.get(model_name))

    ax.set_xticks(x_base)
    ax.set_xticklabels(directions, rotation=20, ha="right")
    ax.set_ylabel("Transfer MAE (%)")
    ax.set_title("Zero-shot transfer across datasets (separate in-dataset PCA)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, plot_dir, "transfer_learning_summary")


def plot_transfer_parity(
    results: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Compare source-PCA vs separate-PCA strategies to show leakage impact."""
    directions = results.groupby(["source", "target"]).size().index.tolist()
    n_dir = len(directions)
    fig, axes = plt.subplots(1, n_dir, figsize=(4 * n_dir, 4), sharey=True)
    if n_dir == 1:
        axes = [axes]

    for ax, (src, tgt) in zip(axes, directions):
        sub = results[(results["source"] == src) & (results["target"] == tgt)]
        for model_name, grp in sub.groupby("model"):
            src_pca_mae  = grp[grp["pca_strategy"] == "source"]["mae_pct"].mean()
            sep_pca_mae  = grp[grp["pca_strategy"] == "separate"]["mae_pct"].mean()
            ax.scatter([src_pca_mae], [sep_pca_mae], label=model_name,
                       color=COLORS.get(model_name), s=60, zorder=5)

        lim = max(sub["mae_pct"].max() * 1.1, 5)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Source PCA → target (MAE%)")
        ax.set_ylabel("Separate PCA (MAE%)")
        ax.set_title(f"{src}→{tgt}")
        ax.legend(fontsize=7)

    fig.suptitle("Transfer PCA strategy comparison\n(below diagonal = separate PCA better)")
    fig.tight_layout()
    _save_figure(fig, plot_dir, "transfer_pca_strategy_comparison")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_seeds: int = 5, models: List[str] = MODELS) -> None:
    data_dir, plot_dir = _get_result_paths()
    print("=" * 70)
    print("Phase 10: Cross-Dataset Transfer Learning")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    results = run_all_transfers(models=models, n_seeds=n_seeds)
    results.to_csv(data_dir / "transfer_learning_results.csv", index=False)
    print(f"\nSaved: {data_dir / 'transfer_learning_results.csv'}")

    # Summary table
    summary = (
        results[results["pca_strategy"] == "separate"]
        .groupby(["source", "target", "model"])[["mae_pct", "r2"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["mae_pct_mean", "mae_pct_std", "r2_mean", "r2_std"]
    print("\nTransfer summary (separate PCA):")
    print(summary.to_string())
    summary.to_csv(data_dir / "transfer_summary.csv")

    print("\nGenerating plots...")
    plot_transfer_summary(results, plot_dir)
    plot_transfer_parity(results, plot_dir)

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-dataset transfer learning.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    main(n_seeds=args.n_seeds, models=args.models)
