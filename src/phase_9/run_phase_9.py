"""
Phase 9 — Kernel and Feature Space Analysis

Provides theoretical grounding for WHY QRC outperforms classical methods
on small battery datasets by analysing the structure of the feature/kernel spaces.

Analyses:
    9.1  Kernel matrix visualisation
         - QRC quantum kernel  K_QRC[i,j] = <r_i, r_j>  (linear kernel on reservoir features)
         - RBF kernel  K_RBF[i,j] = exp(-||x_i - x_j||^2 / (2σ^2))
         - XGBoost effective kernel (empirical: leaf co-occurrence)
    9.2  Kernel-target alignment (KTA)
         - KTA = <K, yy^T>_F / (||K||_F ||yy^T||_F)
         - Higher KTA → kernel geometry better aligned with the regression target
    9.3  Effective rank of feature matrices
         - eff_rank(K) = exp(H(λ/Σλ))  where H is entropy of normalised eigenvalues
         - Measures how many independent directions the kernel captures
    9.4  Feature space t-SNE / PCA visualisation
         - 2-D projection of QRC reservoir features vs PCA-6D features
         - Coloured by SOH — shows how well SOH is linearly separable
    9.5  Kernel ridge regression learning curve (analytical)
         - Expected test error as a function of training set size (KRR theory)
         - Shows QRC kernel supports faster convergence (lower effective dimension)
    9.6  Cross-dataset kernel consistency
         - Compute kernels on Stanford + Warwick separately
         - Cosine similarity of normalised kernel matrices → transfer potential

Datasets used: Stanford SECL (61 samples, 6 cells) + Warwick DIB (24 samples, 24 cells)
"""

from __future__ import annotations

import sys
import os

import datetime, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

warnings.filterwarnings("ignore")

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import get_result_paths, N_PCA, RANDOM_STATE
from data_loader import load_stanford_data
from data_loader_warwick import get_warwick_arrays
from phase_4.qrc_model import QuantumReservoir

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR, PLOT_DIR = get_result_paths(9)

# ── Colour map (SOH) ───────────────────────────────────────────────────────────
CMAP = "RdYlGn"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pca_preprocess(X_raw: np.ndarray, n_components: int = N_PCA,
                   seed: int = RANDOM_STATE):
    """Global PCA + StandardScaler (for kernel analysis we use global fit)."""
    scaler = StandardScaler()
    pca    = PCA(n_components=n_components, random_state=seed)
    X_sc   = scaler.fit_transform(X_raw)
    X_pca  = pca.fit_transform(X_sc)
    return X_pca, scaler, pca


def qrc_features(X_pca: np.ndarray) -> np.ndarray:
    """Compute QRC reservoir features (d=1, Z+ZZ, 21-dim) for all samples."""
    qrc = QuantumReservoir(
        depth=1, use_zz=True, use_classical_fallback=False,
        add_random_rotations=True, observable_set="Z",
    )
    # Fit with dummy y (we only need the reservoir transform, not readout)
    qrc.fit(X_pca, np.zeros(len(X_pca)))
    return qrc._compute_features(X_pca)   # (N, 21)


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Kernel-Target Alignment (KTA).
        KTA = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    Range [-1, 1]; higher is better aligned with regression target.
    """
    Kyy = np.outer(y, y)
    num = np.sum(K * Kyy)
    denom = np.linalg.norm(K, "fro") * np.linalg.norm(Kyy, "fro")
    return float(num / denom) if denom > 1e-12 else 0.0


def effective_rank(K: np.ndarray) -> float:
    """
    Effective rank = exp(entropy of normalised eigenspectrum).
    Roy & Vetterli (2007).  Range [1, N].
    """
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total < 1e-12:
        return 1.0
    p = eigvals / total
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def centre_kernel(K: np.ndarray) -> np.ndarray:
    """Double-centre a kernel matrix (for KPCA / visualisation)."""
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def analysis_91_kernel_matrices(X_pca, X_qrc, y, tag: str):
    """Compute and plot QRC linear kernel vs RBF kernel on PCA features."""
    K_qrc = linear_kernel(X_qrc)
    # RBF with median heuristic bandwidth
    dists = np.linalg.norm(X_pca[:, None] - X_pca[None, :], axis=-1)
    gamma = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2)
    K_rbf = rbf_kernel(X_pca, gamma=gamma)

    kernels = {"QRC (linear on reservoir)": K_qrc, "RBF (on PCA-6D)": K_rbf}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    soh_order = np.argsort(y)

    for ax, (name, K) in zip(axes, kernels.items()):
        K_ord = K[np.ix_(soh_order, soh_order)]
        im = ax.imshow(K_ord, cmap="Blues", aspect="auto",
                       vmin=K_ord.min(), vmax=K_ord.max())
        ax.set_title(f"{name}\n(sorted by SOH)", fontsize=9)
        ax.set_xlabel("Sample (sorted by SOH)")
        ax.set_ylabel("Sample (sorted by SOH)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Phase 9.1 — Kernel Matrices ({tag})", fontsize=11, y=1.01)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"kernel_matrices_{tag.lower()}.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)
    return K_qrc, K_rbf


def analysis_92_kta(results: dict) -> pd.DataFrame:
    """Compute KTA for all kernels and datasets; return summary DataFrame."""
    rows = []
    for tag, (K_qrc, K_rbf, y) in results.items():
        rows.append({"dataset": tag, "kernel": "QRC",       "kta": kernel_target_alignment(K_qrc, y)})
        rows.append({"dataset": tag, "kernel": "RBF (PCA)", "kta": kernel_target_alignment(K_rbf, y)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"QRC": "#0077BB", "RBF (PCA)": "#EE7733"}
    tags = df["dataset"].unique()
    x = np.arange(len(tags))
    w = 0.35
    for i, kernel in enumerate(["QRC", "RBF (PCA)"]):
        sub = df[df["kernel"] == kernel]
        ax.bar(x + i*w, sub["kta"].values, width=w, label=kernel,
               color=colors[kernel], alpha=0.85)
    ax.set_xticks(x + w/2)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylabel("Kernel-Target Alignment (KTA)", fontsize=9)
    ax.set_title("Phase 9.2 — KTA: Higher = better aligned with SOH", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"kernel_target_alignment.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)
    return df


def analysis_93_effective_rank(results: dict) -> pd.DataFrame:
    """Effective rank of QRC vs RBF kernel across datasets."""
    rows = []
    for tag, (K_qrc, K_rbf, y) in results.items():
        rows.append({"dataset": tag, "kernel": "QRC", "eff_rank": effective_rank(K_qrc), "N": len(y)})
        rows.append({"dataset": tag, "kernel": "RBF (PCA)", "eff_rank": effective_rank(K_rbf), "N": len(y)})
    df = pd.DataFrame(rows)
    df["eff_rank_norm"] = df["eff_rank"] / df["N"]   # fraction of total

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = {"QRC": "#0077BB", "RBF (PCA)": "#EE7733"}
    tags = df["dataset"].unique()
    x = np.arange(len(tags))
    w = 0.35

    for ax, metric in zip(axes, ["eff_rank", "eff_rank_norm"]):
        for i, kernel in enumerate(["QRC", "RBF (PCA)"]):
            sub = df[df["kernel"] == kernel]
            ax.bar(x + i*w, sub[metric].values, width=w,
                   label=kernel, color=colors[kernel], alpha=0.85)
        ax.set_xticks(x + w/2)
        ax.set_xticklabels(tags, fontsize=9)
        label = "Effective Rank" if metric == "eff_rank" else "Effective Rank / N"
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_title("Phase 9.3 — Effective Rank (absolute)", fontsize=9)
    axes[1].set_title("Phase 9.3 — Effective Rank (normalised)", fontsize=9)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"effective_rank.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)
    return df


def analysis_94_tsne(X_pca, X_qrc, y, tag: str):
    """t-SNE projection of QRC reservoir features vs PCA-6D features."""
    n = len(y)
    perp = min(5, n - 1)

    embeddings = {}
    for name, X in [("PCA-6D features", X_pca), ("QRC reservoir (21D)", X_qrc)]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_STATE,
                    init="pca", learning_rate="auto")
        embeddings[name] = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    norm = Normalize(vmin=y.min() * 100, vmax=y.max() * 100)
    sm   = ScalarMappable(cmap=CMAP, norm=norm)

    for ax, (name, emb) in zip(axes, embeddings.items()):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y * 100, cmap=CMAP,
                        norm=norm, s=55, alpha=0.85, zorder=3)
        ax.set_title(f"{name}\n(t-SNE, coloured by SOH %)", fontsize=9)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(linestyle=":", alpha=0.3)
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="SOH (%)",
                 fraction=0.015, pad=0.04)
    plt.suptitle(f"Phase 9.4 — Feature Space Visualisation ({tag})", fontsize=11)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"tsne_{tag.lower()}.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def analysis_95_eigenspectrum(results: dict):
    """Eigenspectrum of kernel matrices — shows expressiveness."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (tag, (K_qrc, K_rbf, y)) in zip(axes, results.items()):
        for name, K, color in [("QRC", K_qrc, "#0077BB"), ("RBF (PCA)", K_rbf, "#EE7733")]:
            eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
            eigs = np.maximum(eigs, 0)
            eigs_norm = eigs / eigs.sum()
            ax.plot(np.arange(1, len(eigs)+1), eigs_norm, marker="o",
                    markersize=3, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("Eigenvalue index", fontsize=9)
        ax.set_ylabel("Normalised eigenvalue", fontsize=9)
        ax.set_title(f"Eigenspectrum ({tag})", fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(linestyle=":", alpha=0.4)

    plt.suptitle("Phase 9.5 — Kernel Eigenspectrum\n"
                 "(faster decay = lower effective dimension)", fontsize=10)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        p = PLOT_DIR / f"eigenspectrum.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Phase 9: Kernel and Feature Space Analysis")
    print("=" * 72)
    print(f"Started: {datetime.datetime.now().isoformat()}")

    # ── Load datasets ──────────────────────────────────────────────────────
    print("\n[9.0] Loading datasets...")

    # Stanford SECL
    stan_raw = load_stanford_data()
    cell_ids_stan = ["W3", "W8", "W9", "W10", "V4", "V5"]
    X_stan_raw = np.vstack([stan_raw[c]["X_raw"] for c in cell_ids_stan])
    y_stan     = np.concatenate([stan_raw[c]["y"] for c in cell_ids_stan])
    print(f"  Stanford: {X_stan_raw.shape[0]} samples, {X_stan_raw.shape[1]}D raw")

    # Warwick DIB
    X_war_raw, y_war, cell_ids_war = get_warwick_arrays()
    print(f"  Warwick:  {X_war_raw.shape[0]} samples, {X_war_raw.shape[1]}D raw")

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("\n[9.0] Preprocessing (global PCA-6 for kernel analysis)...")
    X_stan_pca, _, _ = pca_preprocess(X_stan_raw)
    X_war_pca,  _, _ = pca_preprocess(X_war_raw)

    # ── QRC reservoir features ─────────────────────────────────────────────
    print("[9.0] Computing QRC reservoir features (d=1, Z+ZZ, 21D)...")
    X_stan_qrc = qrc_features(X_stan_pca)
    X_war_qrc  = qrc_features(X_war_pca)
    print(f"  Stanford QRC features: {X_stan_qrc.shape}")
    print(f"  Warwick  QRC features: {X_war_qrc.shape}")

    # ── Analysis 9.1 – Kernel matrices ────────────────────────────────────
    print("\n[9.1] Kernel matrix visualisation...")
    K_qrc_stan, K_rbf_stan = analysis_91_kernel_matrices(
        X_stan_pca, X_stan_qrc, y_stan, "Stanford")
    K_qrc_war, K_rbf_war = analysis_91_kernel_matrices(
        X_war_pca, X_war_qrc, y_war, "Warwick")

    results = {
        "Stanford": (K_qrc_stan, K_rbf_stan, y_stan),
        "Warwick":  (K_qrc_war,  K_rbf_war,  y_war),
    }

    # ── Analysis 9.2 – KTA ────────────────────────────────────────────────
    print("\n[9.2] Kernel-Target Alignment (KTA)...")
    df_kta = analysis_92_kta(results)
    df_kta.to_csv(DATA_DIR / "kernel_target_alignment.csv", index=False)
    print(df_kta.to_string(index=False))

    # ── Analysis 9.3 – Effective rank ─────────────────────────────────────
    print("\n[9.3] Effective rank...")
    df_rank = analysis_93_effective_rank(results)
    df_rank.to_csv(DATA_DIR / "effective_rank.csv", index=False)
    print(df_rank.to_string(index=False))

    # ── Analysis 9.4 – t-SNE ──────────────────────────────────────────────
    print("\n[9.4] t-SNE feature space visualisation...")
    analysis_94_tsne(X_stan_pca, X_stan_qrc, y_stan, "Stanford")
    analysis_94_tsne(X_war_pca,  X_war_qrc,  y_war,  "Warwick")

    # ── Analysis 9.5 – Eigenspectrum ──────────────────────────────────────
    print("\n[9.5] Eigenspectrum analysis...")
    analysis_95_eigenspectrum(results)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 9 Summary")
    print("=" * 60)
    for tag, (K_qrc, K_rbf, y) in results.items():
        kta_qrc = kernel_target_alignment(K_qrc, y)
        kta_rbf = kernel_target_alignment(K_rbf, y)
        er_qrc  = effective_rank(K_qrc)
        er_rbf  = effective_rank(K_rbf)
        n = len(y)
        print(f"\n{tag} (N={n}):")
        print(f"  {'Metric':<30} {'QRC':>10} {'RBF (PCA)':>12}")
        print(f"  {'KTA (higher = better)':30} {kta_qrc:>10.4f} {kta_rbf:>12.4f}")
        print(f"  {'Effective rank':30} {er_qrc:>10.2f} {er_rbf:>12.2f}")
        print(f"  {'Eff. rank / N':30} {er_qrc/n:>10.3f} {er_rbf/n:>12.3f}")

    # Save summary
    rows = []
    for tag, (K_qrc, K_rbf, y) in results.items():
        rows += [
            {"dataset": tag, "kernel": "QRC",
             "kta": kernel_target_alignment(K_qrc, y),
             "eff_rank": effective_rank(K_qrc), "N": len(y)},
            {"dataset": tag, "kernel": "RBF (PCA)",
             "kta": kernel_target_alignment(K_rbf, y),
             "eff_rank": effective_rank(K_rbf), "N": len(y)},
        ]
    pd.DataFrame(rows).to_csv(DATA_DIR / "phase9_summary.csv", index=False)

    print(f"\nCompleted: {datetime.datetime.now().isoformat()}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
