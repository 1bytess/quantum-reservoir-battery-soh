"""
Phase 9 -- Reservoir Feature Importance Analysis

Addresses reviewer SS2.5: "Which of the 21 reservoir observables actually
drive predictive performance? Are ZZ-correlators essential?"

Five complementary analyses on the 21 quantum reservoir features
(6 Z-singles + 15 ZZ-correlators) obtained from a depth-1, Z+ZZ circuit:

    A. Pearson & Spearman correlation with SOH
    B. Mutual information (sklearn mutual_info_regression)
    C. Ridge readout coefficient magnitudes  |w_i|
    D. Greedy forward feature selection     MAE vs number of selected features
    E. Group comparison                     Z-singles (6D) vs ZZ-correlators (15D)
                                            vs combined (21D) via LOCO Ridge MAE

All five analyses use the Stanford SECL dataset (61 samples, 6 cells)
as the primary benchmark.  Warwick DIB (24 samples, 24 cells) is used
in Analysis E to test generalisability.

Output: 2x3 dashboard PNG/PDF + per-analysis CSV tables saved to
        result/phase_9/data/  and  result/phase_9/plot/.

Usage::

    # From project root (src/ on sys.path):
    python -m src.phase_9.reservoir_feature_importance

    # Or from the src/ directory directly:
    python -m phase_9.reservoir_feature_importance
"""

from __future__ import annotations

import datetime
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -- Project imports (same style as run_phase_9.py) ---------------------------
from config import get_result_paths, N_PCA, RANDOM_STATE
from data_loader import load_stanford_data
from data_loader_warwick import get_warwick_arrays
from phase_4.qrc_model import QuantumReservoir

# -- Output paths (reuse phase-9 result directories) --------------------------
DATA_DIR, PLOT_DIR = get_result_paths(9)

# -- Observable labels (depth-1, Z+ZZ, N_QUBITS=6) ---------------------------
N_QUBITS = 6
_Z_LABELS  = [f"Z_{i}" for i in range(N_QUBITS)]
_ZZ_LABELS = [f"ZZ_{i}{j}" for i, j in combinations(range(N_QUBITS), 2)]
FEATURE_LABELS: list[str] = _Z_LABELS + _ZZ_LABELS   # 6 + 15 = 21

# Colour coding: blue = Z-singles, orange = ZZ-correlators
COLOURS: list[str] = (
    ["#0077BB"] * N_QUBITS
    + ["#EE7733"] * len(_ZZ_LABELS)
)

# Ridge regularisation used for analyses C, D, E (fixed for comparability)
RIDGE_ALPHA: float = 1.0


# =============================================================================
# Data helpers
# =============================================================================

def _pca_preprocess(X_raw: np.ndarray, n_components: int = N_PCA) -> np.ndarray:
    """Global StandardScaler -> PCA-N compression (matches run_phase_9.py)."""
    X_sc = StandardScaler().fit_transform(X_raw)
    return PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X_sc)


def _qrc_features(X_pca: np.ndarray) -> np.ndarray:
    """Compute 21-dim QRC reservoir features (fixed circuit, depth=1, Z+ZZ).

    The reservoir is fit with a dummy target so the random rotations are
    seeded deterministically via RANDOM_STATE -- only the readout is discarded.
    """
    qrc = QuantumReservoir(
        depth=1,
        use_zz=True,
        use_classical_fallback=False,
        add_random_rotations=True,
        observable_set="Z",
    )
    # Dummy target (all zeros): only the random rotations (seeded by RANDOM_STATE)
    # and reservoir circuit matter here; the Ridge readout is discarded immediately
    # after fit(). This is intentional — we need fit() only to initialize scaler_
    # and random_rotations_ so that _compute_features() works.
    qrc.fit(X_pca, np.zeros(len(X_pca)))
    return qrc._compute_features(X_pca)   # (N, 21)


def _loco_mae(
    X:      np.ndarray,
    y:      np.ndarray,
    groups: np.ndarray,
    alpha:  float = RIDGE_ALPHA,
) -> float:
    """Leave-One-Cell-Out cross-validated MAE using Ridge(alpha)."""
    logo   = LeaveOneGroupOut()
    scores = cross_val_score(
        Ridge(alpha=alpha), X, y,
        cv=logo, groups=groups,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return float(-scores.mean())


# =============================================================================
# Analysis A -- Pearson & Spearman correlation with SOH
# =============================================================================

def analysis_a_correlation(X_res: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compute Pearson r and Spearman rho for each of the 21 reservoir features."""
    rows = []
    for i, label in enumerate(FEATURE_LABELS):
        pc, pp = pearsonr(X_res[:, i], y)
        sc, sp = spearmanr(X_res[:, i], y)
        rows.append({
            "feature":      label,
            "group":        "Z-single" if i < N_QUBITS else "ZZ-pair",
            "pearson_r":    pc,
            "pearson_p":    pp,
            "spearman_r":   sc,
            "spearman_p":   sp,
            "abs_pearson":  abs(pc),
            "abs_spearman": abs(sc),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Analysis B -- Mutual information
# =============================================================================

def analysis_b_mutual_info(X_res: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Mutual information between each reservoir feature and SOH (in nats)."""
    mi = mutual_info_regression(X_res, y, random_state=RANDOM_STATE)
    return pd.DataFrame({
        "feature":     FEATURE_LABELS,
        "group":       ["Z-single"] * N_QUBITS + ["ZZ-pair"] * len(_ZZ_LABELS),
        "mutual_info": mi,
    })


# =============================================================================
# Analysis C -- Ridge readout coefficient magnitudes
# =============================================================================

def analysis_c_ridge_coef(
    X_res:  np.ndarray,
    y:      np.ndarray,
    groups: np.ndarray,
    alpha:  float = RIDGE_ALPHA,
) -> pd.DataFrame:
    """Fit Ridge on full dataset; return absolute coefficient per feature.

    Using all data (not LOCO) keeps coefficients stable enough for
    feature-importance comparison.  The fixed alpha ensures magnitudes are
    directly comparable across features.
    """
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_res, y)
    return pd.DataFrame({
        "feature":  FEATURE_LABELS,
        "group":    ["Z-single"] * N_QUBITS + ["ZZ-pair"] * len(_ZZ_LABELS),
        "abs_coef": np.abs(ridge.coef_),
    })


# =============================================================================
# Analysis D -- Greedy forward feature selection
# =============================================================================

def analysis_d_greedy_selection(
    X_res:  np.ndarray,
    y:      np.ndarray,
    groups: np.ndarray,
    alpha:  float = RIDGE_ALPHA,
) -> pd.DataFrame:
    """Greedy forward selection: add the feature that most reduces LOCO MAE.

    Returns a DataFrame with one row per step (n_features 0..21), showing
    which feature was added and the resulting LOCO MAE.

    Step 0 is the Leave-One-Cell-Out baseline (predicting the within-fold
    training mean), giving an upper-bound MAE for reference.
    """
    n_total   = X_res.shape[1]
    remaining = list(range(n_total))
    selected: list[int] = []
    rows: list[dict]    = []

    # LOCO baseline: predict training-fold mean for every test sample
    logo = LeaveOneGroupOut()
    baseline_preds: list[float] = []
    baseline_true:  list[float] = []
    for train_idx, test_idx in logo.split(X_res, y, groups):
        baseline_preds.extend([y[train_idx].mean()] * len(test_idx))
        baseline_true.extend(y[test_idx].tolist())
    mae_baseline = float(np.mean(np.abs(
        np.array(baseline_preds) - np.array(baseline_true)
    )))
    rows.append({
        "n_features":       0,
        "selected_feature": "baseline (LOCO mean)",
        "feature_index":    -1,
        "loco_mae":         mae_baseline,
    })

    for step in range(n_total):
        best_mae = np.inf
        best_idx = -1
        for idx in remaining:
            trial = selected + [idx]
            mae   = _loco_mae(X_res[:, trial], y, groups, alpha=alpha)
            if mae < best_mae:
                best_mae = mae
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        rows.append({
            "n_features":       step + 1,
            "selected_feature": FEATURE_LABELS[best_idx],
            "feature_index":    best_idx,
            "loco_mae":         best_mae,
        })
        print(
            f"    Step {step+1:2d}: +{FEATURE_LABELS[best_idx]:<10s}  "
            f"LOCO MAE = {best_mae:.4f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# Analysis E -- Group comparison: Z-singles vs ZZ-correlators vs combined
# =============================================================================

def analysis_e_group_comparison(
    X_res:  np.ndarray,
    y:      np.ndarray,
    groups: np.ndarray,
    alpha:  float = RIDGE_ALPHA,
    tag:    str   = "Stanford",
) -> pd.DataFrame:
    """LOCO Ridge MAE for three feature subsets.

    Subsets
    -------
    Z-singles (6D)       : indices 0..5   -- single-qubit expectation values
    ZZ-correlators (15D) : indices 6..20  -- two-qubit correlators
    Combined (21D)       : all 21 features
    """
    subsets = {
        "Z-singles\n(6D)":        list(range(N_QUBITS)),
        "ZZ-correlators\n(15D)":  list(range(N_QUBITS, N_QUBITS + len(_ZZ_LABELS))),
        "Combined\n(21D)":        list(range(N_QUBITS + len(_ZZ_LABELS))),
    }
    rows = []
    for label, idx in subsets.items():
        mae = _loco_mae(X_res[:, idx], y, groups, alpha=alpha)
        rows.append({
            "dataset":    tag,
            "subset":     label,
            "n_features": len(idx),
            "loco_mae":   mae,
        })
        clean = label.replace("\n", " ")
        print(f"    {clean:<28s}: LOCO MAE = {mae:.4f}")
    return pd.DataFrame(rows)


# =============================================================================
# Dashboard: 2x3 plot
# =============================================================================

def _plot_group_bars(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    """Grouped bar chart for one dataset's group-comparison panel."""
    subsets    = df["subset"].tolist()
    maes       = df["loco_mae"].tolist()
    bar_colors = ["#0077BB", "#EE7733", "#44BB99"]
    bars = ax.bar(
        range(len(subsets)), maes,
        color=bar_colors[:len(subsets)], alpha=0.85, width=0.55,
    )
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{mae:.4f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels(subsets, fontsize=8)
    ax.set_ylabel("LOCO Ridge MAE", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(bottom=0)


def plot_dashboard(
    df_corr:       pd.DataFrame,
    df_mi:         pd.DataFrame,
    df_coef:       pd.DataFrame,
    df_greedy:     pd.DataFrame,
    df_group_stan: pd.DataFrame,
    df_group_war:  pd.DataFrame,
) -> plt.Figure:
    """Assemble the 2x3 feature-importance dashboard.

    Layout
    ------
    Row 0  | (A) Pearson & Spearman |r| | (B) Mutual information | (C) |Ridge coef|
    Row 1  | (D) Greedy forward selection | (E1) Group Stanford   | (E2) Group Warwick
    """
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_a  = fig.add_subplot(gs[0, 0])
    ax_b  = fig.add_subplot(gs[0, 1])
    ax_c  = fig.add_subplot(gs[0, 2])
    ax_d  = fig.add_subplot(gs[1, 0])
    ax_e1 = fig.add_subplot(gs[1, 1])
    ax_e2 = fig.add_subplot(gs[1, 2])

    n_feat  = len(FEATURE_LABELS)
    x       = np.arange(n_feat)
    width   = 0.40
    tick_fs = 6
    lbl_fs  = 8

    # ---- Panel A: Pearson & Spearman |r| ------------------------------------
    ax_a.bar(x - width / 2, df_corr["abs_pearson"],  width=width,
             color=COLOURS, alpha=0.90, label="Pearson |r|")
    ax_a.bar(x + width / 2, df_corr["abs_spearman"], width=width,
             color=COLOURS, alpha=0.45, label="Spearman |rho|")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(FEATURE_LABELS, rotation=90, fontsize=tick_fs)
    ax_a.set_ylabel("|Correlation with SOH|", fontsize=lbl_fs)
    ax_a.set_title("(A) Pearson & Spearman |r|", fontsize=9, fontweight="bold")
    ax_a.legend(fontsize=7)
    ax_a.grid(axis="y", linestyle=":", alpha=0.4)
    ax_a.set_ylim(bottom=0)
    ax_a.axvline(N_QUBITS - 0.5, color="grey", linestyle="--", linewidth=0.8)
    ymax_a = ax_a.get_ylim()[1]
    ax_a.text(1.5,          ymax_a * 0.90, "Z",  fontsize=7, color="#0077BB")
    ax_a.text(N_QUBITS + 1, ymax_a * 0.90, "ZZ", fontsize=7, color="#EE7733")

    # ---- Panel B: Mutual information ----------------------------------------
    ax_b.bar(x, df_mi["mutual_info"], color=COLOURS, alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(FEATURE_LABELS, rotation=90, fontsize=tick_fs)
    ax_b.set_ylabel("Mutual information (nats)", fontsize=lbl_fs)
    ax_b.set_title("(B) Mutual Information", fontsize=9, fontweight="bold")
    ax_b.grid(axis="y", linestyle=":", alpha=0.4)
    ax_b.set_ylim(bottom=0)
    ax_b.axvline(N_QUBITS - 0.5, color="grey", linestyle="--", linewidth=0.8)

    # ---- Panel C: |Ridge coefficient| ---------------------------------------
    ax_c.bar(x, df_coef["abs_coef"], color=COLOURS, alpha=0.85)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(FEATURE_LABELS, rotation=90, fontsize=tick_fs)
    ax_c.set_ylabel("|Ridge coefficient|", fontsize=lbl_fs)
    ax_c.set_title(f"(C) Ridge |w|  (alpha = {RIDGE_ALPHA})",
                   fontsize=9, fontweight="bold")
    ax_c.grid(axis="y", linestyle=":", alpha=0.4)
    ax_c.set_ylim(bottom=0)
    ax_c.axvline(N_QUBITS - 0.5, color="grey", linestyle="--", linewidth=0.8)
    legend_elems = [
        Patch(facecolor="#0077BB", label="Z-singles  (i = 0-5)"),
        Patch(facecolor="#EE7733", label="ZZ-pairs   (i = 6-20)"),
    ]
    ax_c.legend(handles=legend_elems, fontsize=7, loc="upper right")

    # ---- Panel D: Greedy forward selection ----------------------------------
    sub          = df_greedy[df_greedy["n_features"] > 0].copy()
    mae_baseline = float(
        df_greedy.loc[df_greedy["n_features"] == 0, "loco_mae"].iloc[0]
    )

    ax_d.plot(sub["n_features"], sub["loco_mae"],
              marker="o", markersize=4, linewidth=1.5,
              color="#0077BB", zorder=3, label="Greedy MAE")
    ax_d.axhline(mae_baseline, color="grey", linestyle=":",
                 linewidth=1.0, label=f"Baseline  {mae_baseline:.3f}")
    ax_d.axvline(N_QUBITS, color="#EE7733", linestyle="--",
                 linewidth=0.9, label=f"n = {N_QUBITS}  (Z-singles only)")

    # Mark plateau: first step where marginal gain < 1% of current MAE
    maes_arr = sub["loco_mae"].values
    plateau_step = next(
        (i for i in range(1, len(maes_arr))
         if abs(maes_arr[i - 1] - maes_arr[i]) < 0.01 * maes_arr[i - 1]),
        len(maes_arr) - 1,
    )
    plateau_n = int(sub["n_features"].iloc[plateau_step])
    ax_d.axvline(plateau_n, color="#CC3311", linestyle="-.",
                 linewidth=0.9, label=f"Plateau  n = {plateau_n}")

    ax_d.set_xlabel("Number of features selected", fontsize=lbl_fs)
    ax_d.set_ylabel("LOCO Ridge MAE", fontsize=lbl_fs)
    ax_d.set_title("(D) Greedy Forward Selection", fontsize=9, fontweight="bold")
    ax_d.legend(fontsize=7)
    ax_d.grid(linestyle=":", alpha=0.4)
    ax_d.set_xlim(left=1)
    ax_d.set_ylim(bottom=0)

    # ---- Panels E1 & E2: Group comparison -----------------------------------
    _plot_group_bars(ax_e1, df_group_stan, "(E1) Group Comparison -- Stanford")
    _plot_group_bars(ax_e2, df_group_war,  "(E2) Group Comparison -- Warwick")

    fig.suptitle(
        "Phase 9 -- Reservoir Feature Importance: 21 Quantum Observables\n"
        "(Z-singles: <Z_i>   |   ZZ-correlators: <Z_i Z_j>)",
        fontsize=11, y=1.02, fontweight="bold",
    )
    return fig


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Phase 9: Reservoir Feature Importance Analysis")
    print("=" * 72)
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print(
        f"Reservoir observables: {len(FEATURE_LABELS)} features "
        f"({N_QUBITS} Z-singles + {len(_ZZ_LABELS)} ZZ-correlators)"
    )

    # -- Load & preprocess Stanford SECL ---------------------------------------
    print("\n[9.FI.0] Loading Stanford SECL dataset...")
    stan_raw      = load_stanford_data()
    cell_ids_stan = ["W3", "W8", "W9", "W10", "V4", "V5"]
    X_stan_raw    = np.vstack([stan_raw[c]["X_raw"] for c in cell_ids_stan])
    y_stan        = np.concatenate([stan_raw[c]["y"]  for c in cell_ids_stan])
    groups_stan   = np.concatenate([
        np.full(len(stan_raw[c]["y"]), i)
        for i, c in enumerate(cell_ids_stan)
    ])
    print(
        f"  Stanford: {X_stan_raw.shape[0]} samples, {len(cell_ids_stan)} cells, "
        f"raw dim = {X_stan_raw.shape[1]}"
    )
    X_stan_pca = _pca_preprocess(X_stan_raw)
    X_stan_res = _qrc_features(X_stan_pca)
    print(f"  QRC reservoir features: {X_stan_res.shape}")

    # -- Load & preprocess Warwick DIB -----------------------------------------
    print("\n[9.FI.0] Loading Warwick DIB dataset...")
    X_war_raw, y_war, cell_ids_war = get_warwick_arrays()
    # 1 EIS spectrum per cell -> each cell is its own LOCO group
    groups_war = np.arange(len(y_war))
    print(
        f"  Warwick: {X_war_raw.shape[0]} samples, {len(cell_ids_war)} cells, "
        f"raw dim = {X_war_raw.shape[1]}"
    )
    X_war_pca = _pca_preprocess(X_war_raw)
    X_war_res = _qrc_features(X_war_pca)
    print(f"  QRC reservoir features: {X_war_res.shape}")

    # -- Analysis A: correlation -----------------------------------------------
    print("\n[9.FI.A] Pearson & Spearman correlation with SOH (Stanford)...")
    df_corr = analysis_a_correlation(X_stan_res, y_stan)
    df_corr.to_csv(DATA_DIR / "feat_importance_correlation.csv", index=False)
    top5 = (
        df_corr.nlargest(5, "abs_pearson")
        [["feature", "pearson_r", "spearman_r"]]
        .to_string(index=False)
    )
    print(f"  Top-5 by |Pearson r|:\n{top5}")

    # -- Analysis B: mutual information ----------------------------------------
    print("\n[9.FI.B] Mutual information (Stanford)...")
    df_mi = analysis_b_mutual_info(X_stan_res, y_stan)
    df_mi.to_csv(DATA_DIR / "feat_importance_mutual_info.csv", index=False)
    top5_mi = (
        df_mi.nlargest(5, "mutual_info")
        [["feature", "mutual_info"]]
        .to_string(index=False)
    )
    print(f"  Top-5 by MI:\n{top5_mi}")

    # -- Analysis C: Ridge coefficient magnitudes ------------------------------
    print("\n[9.FI.C] Ridge |coefficient| magnitudes (Stanford)...")
    df_coef = analysis_c_ridge_coef(X_stan_res, y_stan, groups_stan)
    df_coef.to_csv(DATA_DIR / "feat_importance_ridge_coef.csv", index=False)
    top5_coef = (
        df_coef.nlargest(5, "abs_coef")
        [["feature", "abs_coef"]]
        .to_string(index=False)
    )
    print(f"  Top-5 by |Ridge coef|:\n{top5_coef}")

    # -- Analysis D: greedy forward selection ----------------------------------
    print("\n[9.FI.D] Greedy forward feature selection (Stanford, LOCO MAE)...")
    df_greedy = analysis_d_greedy_selection(X_stan_res, y_stan, groups_stan)
    df_greedy.to_csv(DATA_DIR / "feat_importance_greedy_selection.csv", index=False)
    valid    = df_greedy[df_greedy["n_features"] > 0]
    best_row = valid.loc[valid["loco_mae"].idxmin()]
    print(
        f"  Best LOCO MAE = {best_row['loco_mae']:.4f} "
        f"at n = {int(best_row['n_features'])} features"
    )

    # -- Analysis E: group comparison ------------------------------------------
    print("\n[9.FI.E] Group comparison -- Stanford...")
    df_group_stan = analysis_e_group_comparison(
        X_stan_res, y_stan, groups_stan, tag="Stanford"
    )
    df_group_stan.to_csv(DATA_DIR / "feat_importance_group_stanford.csv", index=False)

    print("\n[9.FI.E] Group comparison -- Warwick...")
    df_group_war = analysis_e_group_comparison(
        X_war_res, y_war, groups_war, tag="Warwick"
    )
    df_group_war.to_csv(DATA_DIR / "feat_importance_group_warwick.csv", index=False)

    # -- 2x3 Dashboard plot ----------------------------------------------------
    print("\n[9.FI] Generating 2x3 dashboard plot...")
    fig = plot_dashboard(
        df_corr, df_mi, df_coef, df_greedy, df_group_stan, df_group_war
    )
    for fmt in ("png", "pdf"):
        out = PLOT_DIR / f"reservoir_feature_importance.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)

    # -- Composite importance ranking ------------------------------------------
    print("\n[9.FI] Computing composite importance ranking...")
    summary = (
        df_corr[["feature", "group", "abs_pearson", "abs_spearman"]]
        .merge(df_mi  [["feature", "mutual_info"]], on="feature")
        .merge(df_coef[["feature", "abs_coef"]],   on="feature")
    )
    # Normalise each score to [0, 1] and average for a single composite rank
    for col in ("abs_pearson", "abs_spearman", "mutual_info", "abs_coef"):
        rng = summary[col].max() - summary[col].min()
        summary[f"{col}_norm"] = (
            (summary[col] - summary[col].min()) / (rng + 1e-12)
        )
    norm_cols = [c for c in summary.columns if c.endswith("_norm")]
    summary["composite_score"] = summary[norm_cols].mean(axis=1)
    summary.sort_values("composite_score", ascending=False, inplace=True)
    summary.to_csv(DATA_DIR / "feat_importance_summary.csv", index=False)

    print("\nTop-10 reservoir observables by composite importance score:")
    print(
        summary[["feature", "group", "abs_pearson",
                 "mutual_info", "abs_coef", "composite_score"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print(f"Completed: {datetime.datetime.now().isoformat()}")
    print(f"Data saved to:  {DATA_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
