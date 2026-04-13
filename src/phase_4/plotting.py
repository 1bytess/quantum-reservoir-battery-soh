"""Plotting for Phase 4-Lab — Tier-1, dedicated model colours."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Phase4LabPaths, DEPTH_RANGE
from ..plot_constants import (
    tier1_rc, save_fig, model_color, model_label, DPI, MODEL_COLORS,
    PAPER_MODELS,
)

# Depth-specific viridis shades for depth sweep curves (aging-like progression)
_DEPTH_CMAP = plt.get_cmap("viridis")


def plot_noiseless_depth_sweep(paths: Phase4LabPaths = None) -> None:
    """Noiseless MAE vs depth — 2-panel (LOCO + temporal)."""
    if paths is None:
        paths = Phase4LabPaths()

    df = pd.read_csv(paths.data_dir / "qrc_noiseless.csv")

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8), sharey=True)

        for i, regime in enumerate(["loco", "temporal"]):
            ax = axes[i]
            rdf = df[df["regime"] == regime]
            if rdf.empty:
                continue

            s = rdf.groupby("depth")["mae"].agg(["mean", "std"]).reset_index()
            # Use dedicated QRC noiseless colour
            ax.errorbar(
                s["depth"], s["mean"], yerr=s["std"],
                fmt="o-", capsize=2, color=MODEL_COLORS["qrc_noiseless"],
                linewidth=0.9, markersize=3, elinewidth=0.5,
            )
            title = "LOCO" if i == 0 else "Temporal"
            ax.set_title(title, fontweight="bold", pad=3)
            ax.set_xlabel("Depth")
            if i == 0:
                ax.set_ylabel("MAE (SOH)")
            ax.set_xticks(DEPTH_RANGE)

            # Best annotation
            best = s.loc[s["mean"].idxmin()]
            ax.annotate(
                f"d={int(best['depth'])}\n{best['mean']:.4f}",
                xy=(best["depth"], best["mean"]),
                xytext=(5, 8), textcoords="offset points", fontsize=5.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="0.6", alpha=0.85),
            )

        fig.tight_layout(w_pad=0.4)
        save_fig(fig, paths.plots_dir, "noiseless_depth_sweep")


def plot_noisy_depth_sweep(paths: Phase4LabPaths = None) -> None:
    """Noisy MAE vs depth, one curve per shot count — viridis for shot sweep."""
    if paths is None:
        paths = Phase4LabPaths()

    p = paths.data_dir / "qrc_noisy.csv"
    if not p.exists():
        print("  No noisy results, skipping")
        return

    df = pd.read_csv(p)
    loco = df[df["regime"] == "loco"]
    shots_list = sorted(loco["shots"].unique())
    n = len(shots_list)
    # Shot sweep = parameter progression → viridis is appropriate here
    colors = [_DEPTH_CMAP(i / max(n - 1, 1)) for i in range(n)]

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))

        for i, shots in enumerate(shots_list):
            s = loco[loco["shots"] == shots]
            g = s.groupby("depth")["mae"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                g["depth"], g["mean"], yerr=g["std"],
                fmt="s--", capsize=2, label=f"{shots} shots",
                color=colors[i], linewidth=0.9, markersize=3,
                elinewidth=0.5,
            )

        ax.set_xlabel("Depth")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("Noisy QRC (Heron R2) — LOCO", fontweight="bold", pad=3)
        ax.set_xticks(DEPTH_RANGE)
        ax.legend(loc="best")

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "noisy_depth_sweep")


def plot_noiseless_vs_noisy(paths: Phase4LabPaths = None) -> None:
    """Noiseless vs noisy: MAE bars + degradation %."""
    if paths is None:
        paths = Phase4LabPaths()

    nl_p = paths.data_dir / "qrc_noiseless.csv"
    ny_p = paths.data_dir / "qrc_noisy.csv"
    if not nl_p.exists() or not ny_p.exists():
        print("  Missing data for comparison, skipping")
        return

    nl = pd.read_csv(nl_p)
    ny = pd.read_csv(ny_p)

    nl_loco = nl[nl["regime"] == "loco"]
    max_shots = ny["shots"].max()
    ny_loco = ny[(ny["regime"] == "loco") & (ny["shots"] == max_shots)]

    nl_s = nl_loco.groupby("depth")["mae"].mean()
    ny_s = ny_loco.groupby("depth")["mae"].mean()
    depths = sorted(set(nl_s.index) & set(ny_s.index))
    x = np.arange(len(depths))
    w = 0.35

    c_nl = MODEL_COLORS["qrc_noiseless"]
    c_ny = MODEL_COLORS["qrc_noisy"]

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))

        ax = axes[0]
        ax.bar(x - w / 2, [nl_s[d] for d in depths], w,
               color=c_nl, edgecolor="black", lw=0.3, label="Noiseless")
        ax.bar(x + w / 2, [ny_s[d] for d in depths], w,
               color=c_ny, edgecolor="black", lw=0.3,
               label=f"Noisy ({int(max_shots)})")
        ax.set_xlabel("Depth")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("MAE", fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(depths)
        ax.legend(loc="best")

        ax2 = axes[1]
        deg = [(ny_s[d] - nl_s[d]) / nl_s[d] * 100 if nl_s[d] > 0 else 0
               for d in depths]
        bar_colors = [c_ny if v > 0 else c_nl for v in deg]
        bars = ax2.bar(x, deg, color=bar_colors, edgecolor="black", lw=0.3)
        ax2.set_xlabel("Depth")
        ax2.set_ylabel("Δ MAE (%)")
        ax2.set_title("Degradation", fontweight="bold", pad=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(depths)
        ax2.axhline(0, color="black", lw=0.4)

        for bar, val in zip(bars, deg):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.3 if val >= 0 else -0.8),
                     f"{val:+.1f}", ha="center", fontsize=5.5)

        fig.tight_layout(w_pad=0.5)
        save_fig(fig, paths.plots_dir, "noiseless_vs_noisy")


def plot_qrc_vs_classical(paths: Phase4LabPaths = None) -> None:
    """All-models LOCO comparison — dedicated model colours."""
    if paths is None:
        paths = Phase4LabPaths()

    nl_p = paths.data_dir / "qrc_noiseless.csv"
    p3_p = paths.phase3_lab_dir / "loco_results.csv"
    if not nl_p.exists() or not p3_p.exists():
        print("  Missing data for QRC vs classical")
        return

    qrc = pd.read_csv(nl_p)
    p3 = pd.read_csv(p3_p)

    loco_qrc = qrc[qrc["regime"] == "loco"]
    bd = loco_qrc.groupby("depth")["mae"].mean().idxmin()
    qrc_mae = loco_qrc[loco_qrc["depth"] == bd]["mae"].mean()

    noisy_mae = None
    ny_p = paths.data_dir / "qrc_noisy.csv"
    if ny_p.exists():
        ny = pd.read_csv(ny_p)
        ny_l = ny[(ny["regime"] == "loco") & (ny["shots"] == ny["shots"].max())]
        if not ny_l.empty:
            noisy_mae = ny_l[ny_l["depth"] == bd]["mae"].mean()

    model_maes = p3[p3["model"].isin(PAPER_MODELS)].groupby("model")["mae"].mean()
    keys = list(model_maes.index) + [f"qrc_d{bd}"]
    names = [model_label(k) for k in model_maes.index] + [f"QRC d={bd}"]
    vals = list(model_maes.values) + [qrc_mae]
    if noisy_mae is not None:
        names.append(f"QRC d={bd}\n(noisy)")
        keys.append(f"qrc_d{bd}_noisy")
        vals.append(noisy_mae)

    n = len(vals)
    colors = [model_color(k) for k in keys]

    cap = 0.15
    vals_capped = [min(v, cap) for v in vals]
    clipped = [v > cap for v in vals]

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        x = np.arange(n)
        bars = ax.bar(x, vals_capped, color=colors, edgecolor="black", lw=0.3)

        for i, (bar, clip) in enumerate(zip(bars, clipped)):
            if clip:
                ax.text(bar.get_x() + bar.get_width() / 2, cap + 0.002,
                        f"\u00d7\n{vals[i]:.3f}", ha="center", va="bottom",
                        fontsize=4.5, color="red")

        ax.set_ylim(top=0.16)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("LOCO: All Models", fontweight="bold", pad=3)

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "qrc_vs_classical")


# =========================================================================
# Tier-1 Applied Energy plots
# =========================================================================

def plot_grand_comparison(paths: Phase4LabPaths = None) -> None:
    """Horizontal bar chart of ALL models sorted by MAE (3.5 x 2.5 in)."""
    if paths is None:
        paths = Phase4LabPaths()

    rows = []

    # Phase 3-Lab classical baselines (paper subset only)
    p3_p = paths.phase3_lab_dir / "loco_results.csv"
    if p3_p.exists():
        p3 = pd.read_csv(p3_p)
        p3_paper = p3[p3["model"].isin(PAPER_MODELS)]
        for model, grp in p3_paper.groupby("model"):
            rows.append({"model": model, "key": model,
                         "mae": grp["mae"].mean()})

    # Static QRC noiseless
    nl_p = paths.data_dir / "qrc_noiseless.csv"
    if nl_p.exists():
        nl = pd.read_csv(nl_p)
        loco = nl[nl["regime"] == "loco"]
        if not loco.empty:
            bd = loco.groupby("depth")["mae"].mean().idxmin()
            rows.append({"model": f"QRC d={bd}", "key": f"qrc_d{bd}",
                         "mae": loco[loco["depth"] == bd]["mae"].mean()})

    # Temporal QRC
    tq_p = paths.data_dir / "temporal_qrc.csv"
    if tq_p.exists():
        tq = pd.read_csv(tq_p)
        if not tq.empty:
            best_mae = tq.groupby("depth")["mae"].mean().min()
            rows.append({"model": "Temporal QRC", "key": "temporal_qrc",
                         "mae": best_mae})

    # Observable ablation — XYZ expanded
    oa_p = paths.data_dir / "observable_ablation.csv"
    if oa_p.exists():
        oa = pd.read_csv(oa_p)
        xyz = oa[oa["observable_config"] == "XYZ"]
        if not xyz.empty:
            rows.append({"model": "QRC (XYZ)", "key": "qrc_expanded",
                         "mae": xyz["mae"].mean()})

    if not rows:
        print("  No data for grand comparison, skipping")
        return

    comp = pd.DataFrame(rows).sort_values("mae", ascending=False)
    colors = [model_color(k) for k in comp["key"]]
    labels = [model_label(k) for k in comp["key"]]

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        y = np.arange(len(comp))
        ax.barh(y, comp["mae"].values, color=colors, edgecolor="black", lw=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("MAE (SOH)")
        ax.set_title("LOCO: Grand Comparison", fontweight="bold", pad=3)
        ax.invert_yaxis()

        max_val = comp["mae"].max()
        ax.set_xlim(right=max_val * 1.15)

        for i, val in enumerate(comp["mae"].values):
            ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=5.5)

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "grand_comparison")


def plot_temporal_vs_static_depth(paths: Phase4LabPaths = None) -> None:
    """Two curves (static vs temporal QRC) on same axes vs depth."""
    if paths is None:
        paths = Phase4LabPaths()

    nl_p = paths.data_dir / "qrc_noiseless.csv"
    tq_p = paths.data_dir / "temporal_qrc.csv"
    if not nl_p.exists() or not tq_p.exists():
        print("  Missing data for temporal vs static, skipping")
        return

    nl = pd.read_csv(nl_p)
    tq = pd.read_csv(tq_p)

    static = nl[nl["regime"] == "loco"].groupby("depth")["mae"].agg(
        ["mean", "std"]).reset_index()
    temporal = tq.groupby("depth")["mae"].agg(
        ["mean", "std"]).reset_index()

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))

        ax.errorbar(
            static["depth"], static["mean"], yerr=static["std"],
            fmt="o-", capsize=2, label="Static QRC",
            color=MODEL_COLORS["qrc_noiseless"],
            linewidth=0.9, markersize=3, elinewidth=0.5,
        )
        ax.errorbar(
            temporal["depth"], temporal["mean"], yerr=temporal["std"],
            fmt="s-", capsize=2, label="Temporal QRC",
            color=MODEL_COLORS["temporal_qrc"],
            linewidth=0.9, markersize=3, elinewidth=0.5,
        )
        ax.set_xlabel("Depth")
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("Static vs Temporal QRC", fontweight="bold", pad=3)
        ax.legend(loc="best")

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "temporal_vs_static_depth")


def plot_observable_ablation(paths: Phase4LabPaths = None) -> None:
    """3-bar grouped chart: Z-only, Z+ZZ, XYZ."""
    if paths is None:
        paths = Phase4LabPaths()

    oa_p = paths.data_dir / "observable_ablation.csv"
    if not oa_p.exists():
        print("  No observable ablation data, skipping")
        return

    df = pd.read_csv(oa_p)
    summary = df.groupby("observable_config")["mae"].agg(
        ["mean", "std"]).reset_index()

    # Fixed order
    order = ["Z_only", "Z_ZZ", "XYZ"]
    summary = summary.set_index("observable_config").reindex(order).reset_index()
    dim_labels = ["6", "21", "153"]
    bar_colors = [MODEL_COLORS["qrc_noiseless"]] * 2 + [MODEL_COLORS["qrc_expanded"]]

    with plt.rc_context(tier1_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        x = np.arange(len(order))
        bars = ax.bar(
            x, summary["mean"], yerr=summary["std"],
            color=bar_colors, edgecolor="black", lw=0.3,
            capsize=2, error_kw={"elinewidth": 0.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{o}\n(dim={d})" for o, d in zip(order, dim_labels)])
        ax.set_ylabel("MAE (SOH)")
        ax.set_title("Observable Ablation", fontweight="bold", pad=3)

        max_bar = summary["mean"].max()
        ax.set_ylim(top=max_bar * 1.15)

        for bar, val in zip(bars, summary["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=5.5)

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "observable_ablation")


def plot_noiseless_noisy_scatter(paths: Phase4LabPaths = None) -> None:
    """Per-cell summary panel for the best noiseless QRC depth."""
    if paths is None:
        paths = Phase4LabPaths()

    nl_p = paths.data_dir / "qrc_noiseless.csv"
    if not nl_p.exists():
        print("  No noiseless data for scatter, skipping")
        return

    from .config import CELL_IDS

    df = pd.read_csv(nl_p)
    loco = df[df["regime"] == "loco"]
    if loco.empty:
        return

    best_depth = loco.groupby("depth")["mae"].mean().idxmin()
    best = loco[loco["depth"] == best_depth]
    n_cells = len(CELL_IDS)
    ncols = min(3, n_cells)
    nrows = int(np.ceil(n_cells / ncols))

    with plt.rc_context(tier1_rc()):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2, 3.5))
        axes = np.atleast_1d(axes).ravel()

        for idx, cell in enumerate(CELL_IDS):
            ax = axes[idx]
            row = best[best["test_cell"] == cell]
            if row.empty:
                ax.set_title(cell)
                continue

            mae_val = row["mae"].values[0]
            ax.set_title(f"{cell}  MAE={mae_val:.4f}", fontsize=6)

            ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=0.4, alpha=0.5)
            ax.text(
                0.53,
                0.92,
                f"d={int(best_depth)}",
                fontsize=5.5,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85),
            )
            ax.set_xlabel("True SOH")
            ax.set_ylabel("Pred SOH")
            ax.set_aspect("equal")

        for idx in range(n_cells, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        save_fig(fig, paths.plots_dir, "noiseless_noisy_scatter")
