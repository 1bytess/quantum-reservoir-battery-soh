"""Phase 7b: ESCL cohort split and limited-history follow-up.

This runner separates the ESCL analysis into:
1. CA6 clean temporal evaluation (direct-label session files)
2. CA6 limited-history curves (train on first N sweeps, test on later sweeps)
3. CA5 high-temperature stress case (proxy-labeled, reported separately)

Usage:
    conda run -n escl-quantum --no-capture-output python -m src.phase_7.run_phase_7b
    conda run -n escl-quantum --no-capture-output python -m src.phase_7.run_phase_7b --skip-ca5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config_lab import DEPTH_RANGE, N_PCA_LAB as N_PCA, RANDOM_STATE, get_lab_result_paths
from data_loader_lab import load_lab_data
from phase_3.models import get_model_pipeline
from phase_4.qrc_model import QuantumReservoir


CA6_MAIN_MODELS = ["esn", "svr", "linear_pc1", "rff", "xgboost"]
CA6_LIMITED_HISTORY_MODELS = ["qrc_d1", "esn", "linear_pc1", "persistence"]
CA5_STRESS_MODELS = ["esn", "svr", "linear_pc1", "rff", "xgboost", "persistence"]
LIMITED_HISTORY_TRAIN_SIZES = [4, 8, 12, 16]

RESULT_COLUMNS = [
    "cohort",
    "label_source",
    "train_scheme",
    "model",
    "cell",
    "depth",
    "n_total",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "r2",
    "persist_mae",
    "mae_gap_vs_persist",
    "beats_persist",
    "soh_range_train",
    "soh_range_test",
]

SUMMARY_COLUMNS = [
    "model",
    "mae_mean",
    "mae_median",
    "mae_std",
    "rmse_mean",
    "r2_mean",
    "persist_mae_mean",
    "mae_gap_vs_persist_mean",
    "beats_persist_count",
    "win_rate",
    "n_evals",
]

LIMITED_HISTORY_SUMMARY_COLUMNS = ["n_train"] + SUMMARY_COLUMNS


class TeeLogger:
    """Tee stdout to console and a log file."""

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
        path = plot_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    plt.close(fig)


def _fit_pca_in_fold(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_components = min(N_PCA, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def _sort_cell_data(cell_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = cell_data["X_raw"]
    y = cell_data["y"]
    blocks = cell_data["block_ids"]
    order = np.argsort(blocks)
    return X[order], y[order], blocks[order]


def _split_temporal_fraction(
    cell_data: dict, train_frac: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_sorted, y_sorted, blocks_sorted = _sort_cell_data(cell_data)
    n_total = len(y_sorted)
    n_train = int(np.floor(train_frac * n_total))
    n_train = max(2, min(n_train, n_total - 1))
    return (
        X_sorted[:n_train],
        y_sorted[:n_train],
        X_sorted[n_train:],
        y_sorted[n_train:],
        blocks_sorted,
    )


def _split_limited_history(
    cell_data: dict, n_train: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    X_sorted, y_sorted, blocks_sorted = _sort_cell_data(cell_data)
    n_total = len(y_sorted)
    if n_total < n_train + 3:
        return None
    return (
        X_sorted[:n_train],
        y_sorted[:n_train],
        X_sorted[n_train:],
        y_sorted[n_train:],
        blocks_sorted,
    )


def _make_result_row(
    cohort: str,
    label_source: str,
    train_scheme: str,
    model: str,
    cell: str,
    n_total: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    metrics: Dict[str, float],
    persist_metrics: Dict[str, float],
    depth: float = np.nan,
) -> Dict[str, object]:
    is_persistence = model == "persistence"
    return {
        "cohort": cohort,
        "label_source": label_source,
        "train_scheme": train_scheme,
        "model": model,
        "cell": cell,
        "depth": depth,
        "n_total": n_total,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "persist_mae": persist_metrics["mae"],
        "mae_gap_vs_persist": metrics["mae"] - persist_metrics["mae"],
        "beats_persist": bool((metrics["mae"] < persist_metrics["mae"]) and not is_persistence),
        "soh_range_train": f"{100 * y_train.min():.1f}-{100 * y_train.max():.1f}",
        "soh_range_test": f"{100 * y_test.min():.1f}-{100 * y_test.max():.1f}",
    }


def _evaluate_qrc(
    depth: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    qrc = QuantumReservoir(
        depth=depth,
        use_zz=True,
        use_classical_fallback=False,
        add_random_rotations=True,
        observable_set="Z",
    )
    qrc.fit(X_train, y_train)
    y_pred = qrc.predict(X_test)
    return _metrics(y_test, y_pred)


def _evaluate_classical(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    model = get_model_pipeline(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _metrics(y_test, y_pred)


def _evaluate_persistence(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = np.full(y_test.shape, y_train[-1], dtype=float)
    return _metrics(y_test, y_pred)


def _summarize_results(
    results_df: pd.DataFrame, include_n_train: bool = False
) -> pd.DataFrame:
    if results_df.empty:
        columns = LIMITED_HISTORY_SUMMARY_COLUMNS if include_n_train else SUMMARY_COLUMNS
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []

    if include_n_train:
        grouped = results_df.groupby(["n_train", "model"], sort=False)
    else:
        grouped = results_df.groupby("model", sort=False)

    for group_key, sub in grouped:
        if include_n_train:
            n_train, model = group_key
            row: Dict[str, object] = {"n_train": int(n_train), "model": model}
        else:
            row = {"model": group_key}

        row.update({
            "mae_mean": float(sub["mae"].mean()),
            "mae_median": float(sub["mae"].median()),
            "mae_std": float(sub["mae"].std(ddof=0)) if len(sub) > 1 else 0.0,
            "rmse_mean": float(sub["rmse"].mean()),
            "r2_mean": float(sub["r2"].mean()) if not sub["r2"].isna().all() else float("nan"),
            "persist_mae_mean": float(sub["persist_mae"].mean()),
            "mae_gap_vs_persist_mean": float(sub["mae_gap_vs_persist"].mean()),
            "beats_persist_count": int(sub["beats_persist"].sum()),
            "win_rate": float(sub["beats_persist"].mean()),
            "n_evals": int(len(sub)),
        })
        rows.append(row)

    summary = pd.DataFrame(rows)
    sort_cols = ["n_train", "mae_mean", "model"] if include_n_train else ["mae_mean", "model"]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)
    columns = LIMITED_HISTORY_SUMMARY_COLUMNS if include_n_train else SUMMARY_COLUMNS
    return summary[columns]


def _print_cell_summary(title: str, cell_data: Dict[str, dict]) -> None:
    print(f"\n{title}")
    for cell_id in sorted(cell_data):
        y = cell_data[cell_id]["y"]
        n = len(y)
        print(f"  {cell_id}: n={n}, SOH={100 * y.min():.1f}-{100 * y.max():.1f}%")


def run_ca6_clean_temporal(cell_data: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run CA6-only clean temporal benchmarks."""
    print("\n" + "=" * 72)
    print("Stage 1: CA6 clean temporal benchmarks")
    print("=" * 72)
    print("Direct-label CA6 session files only. CA5 proxy labels are excluded here.")

    qrc_rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []

    for cell_id, cdata in sorted(cell_data.items()):
        if len(cdata["y"]) < 4:
            print(f"  {cell_id}: skipped (< 4 samples)")
            continue

        X_train_raw, y_train, X_test_raw, y_test, _ = _split_temporal_fraction(cdata)
        X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)
        persist_metrics = _evaluate_persistence(y_train, y_test)
        n_total = len(cdata["y"])

        print(
            f"\n  {cell_id}: n_total={n_total}, n_train={len(y_train)}, "
            f"n_test={len(y_test)}, persist_mae={persist_metrics['mae']:.4f}"
        )

        for depth in DEPTH_RANGE:
            try:
                metrics = _evaluate_qrc(depth, X_train, y_train, X_test, y_test)
                qrc_rows.append(_make_result_row(
                    cohort="CA6_direct_label",
                    label_source="capacity_derived",
                    train_scheme="temporal_70_30",
                    model=f"qrc_d{depth}",
                    cell=cell_id,
                    depth=float(depth),
                    n_total=n_total,
                    y_train=y_train,
                    y_test=y_test,
                    metrics=metrics,
                    persist_metrics=persist_metrics,
                ))
                print(
                    f"    qrc_d{depth}: MAE={metrics['mae']:.4f}  "
                    f"gap={metrics['mae'] - persist_metrics['mae']:+.4f}"
                )
            except Exception as exc:
                print(f"    qrc_d{depth}: FAILED ({type(exc).__name__}: {exc})")

        baseline_rows.append(_make_result_row(
            cohort="CA6_direct_label",
            label_source="capacity_derived",
            train_scheme="temporal_70_30",
            model="persistence",
            cell=cell_id,
            depth=np.nan,
            n_total=n_total,
            y_train=y_train,
            y_test=y_test,
            metrics=persist_metrics,
            persist_metrics=persist_metrics,
        ))

        for model_name in CA6_MAIN_MODELS:
            try:
                metrics = _evaluate_classical(model_name, X_train, y_train, X_test, y_test)
                baseline_rows.append(_make_result_row(
                    cohort="CA6_direct_label",
                    label_source="capacity_derived",
                    train_scheme="temporal_70_30",
                    model=model_name,
                    cell=cell_id,
                    depth=np.nan,
                    n_total=n_total,
                    y_train=y_train,
                    y_test=y_test,
                    metrics=metrics,
                    persist_metrics=persist_metrics,
                ))
                print(
                    f"    {model_name:10s} MAE={metrics['mae']:.4f}  "
                    f"gap={metrics['mae'] - persist_metrics['mae']:+.4f}"
                )
            except Exception as exc:
                print(f"    {model_name:10s} FAILED ({type(exc).__name__}: {exc})")

    qrc_df = pd.DataFrame(qrc_rows, columns=RESULT_COLUMNS)
    baseline_df = pd.DataFrame(baseline_rows, columns=RESULT_COLUMNS)
    summary_df = _summarize_results(pd.concat([qrc_df, baseline_df], ignore_index=True))
    return qrc_df, baseline_df, summary_df


def run_ca6_limited_history(cell_data: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run limited-history curves on the CA6-only cohort."""
    print("\n" + "=" * 72)
    print("Stage 2: CA6 limited-history curves")
    print("=" * 72)
    print("Train on the first N sweeps of each CA6 session, test on the later sweeps.")

    rows: List[Dict[str, object]] = []

    for n_train in LIMITED_HISTORY_TRAIN_SIZES:
        print(f"\n  n_train={n_train}")
        eligible = 0

        for cell_id, cdata in sorted(cell_data.items()):
            split = _split_limited_history(cdata, n_train=n_train)
            if split is None:
                print(f"    {cell_id}: skipped (need at least {n_train + 3} sweeps)")
                continue

            eligible += 1
            X_train_raw, y_train, X_test_raw, y_test, _ = split
            X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)
            persist_metrics = _evaluate_persistence(y_train, y_test)
            n_total = len(cdata["y"])

            print(
                f"    {cell_id}: n_total={n_total}, n_test={len(y_test)}, "
                f"persist_mae={persist_metrics['mae']:.4f}"
            )

            for model_name in CA6_LIMITED_HISTORY_MODELS:
                try:
                    if model_name == "persistence":
                        metrics = persist_metrics
                        depth = np.nan
                    elif model_name.startswith("qrc_d"):
                        depth = float(int(model_name.split("d", maxsplit=1)[1]))
                        metrics = _evaluate_qrc(int(depth), X_train, y_train, X_test, y_test)
                    else:
                        depth = np.nan
                        metrics = _evaluate_classical(model_name, X_train, y_train, X_test, y_test)

                    rows.append(_make_result_row(
                        cohort="CA6_direct_label",
                        label_source="capacity_derived",
                        train_scheme="limited_history_first_n",
                        model=model_name,
                        cell=cell_id,
                        depth=depth,
                        n_total=n_total,
                        y_train=y_train,
                        y_test=y_test,
                        metrics=metrics,
                        persist_metrics=persist_metrics,
                    ))
                    print(
                        f"      {model_name:11s} MAE={metrics['mae']:.4f}  "
                        f"gap={metrics['mae'] - persist_metrics['mae']:+.4f}"
                    )
                except Exception as exc:
                    print(f"      {model_name:11s} FAILED ({type(exc).__name__}: {exc})")

        print(f"    Eligible sessions: {eligible}")

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary_df = _summarize_results(results_df, include_n_train=True)
    return results_df, summary_df


def run_ca5_stress_case(cell_data: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the CA5 high-temperature stress case separately."""
    print("\n" + "=" * 72)
    print("Stage 3: CA5 high-temperature stress case")
    print("=" * 72)
    print("CA5 is proxy-labeled and should not be mixed into the CA6 headline metric.")

    if "CA5_AGING" not in cell_data:
        print("  CA5_AGING not available. Skipping stress case.")
        empty = pd.DataFrame(columns=RESULT_COLUMNS)
        return empty, pd.DataFrame(columns=SUMMARY_COLUMNS)

    cdata = cell_data["CA5_AGING"]
    if len(cdata["y"]) < 4:
        print("  CA5_AGING has < 4 samples. Skipping stress case.")
        empty = pd.DataFrame(columns=RESULT_COLUMNS)
        return empty, pd.DataFrame(columns=SUMMARY_COLUMNS)

    X_train_raw, y_train, X_test_raw, y_test, _ = _split_temporal_fraction(cdata)
    X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)
    persist_metrics = _evaluate_persistence(y_train, y_test)
    n_total = len(cdata["y"])

    rows: List[Dict[str, object]] = []
    print(
        f"  CA5_AGING: n_total={n_total}, n_train={len(y_train)}, "
        f"n_test={len(y_test)}, persist_mae={persist_metrics['mae']:.4f}"
    )

    for depth in DEPTH_RANGE:
        try:
            metrics = _evaluate_qrc(depth, X_train, y_train, X_test, y_test)
            rows.append(_make_result_row(
                cohort="CA5_high_temp_proxy",
                label_source="estimated_proxy",
                train_scheme="temporal_70_30",
                model=f"qrc_d{depth}",
                cell="CA5_AGING",
                depth=float(depth),
                n_total=n_total,
                y_train=y_train,
                y_test=y_test,
                metrics=metrics,
                persist_metrics=persist_metrics,
            ))
            print(
                f"    qrc_d{depth}: MAE={metrics['mae']:.4f}  "
                f"gap={metrics['mae'] - persist_metrics['mae']:+.4f}"
            )
        except Exception as exc:
            print(f"    qrc_d{depth}: FAILED ({type(exc).__name__}: {exc})")

    for model_name in CA5_STRESS_MODELS:
        try:
            if model_name == "persistence":
                metrics = persist_metrics
            else:
                metrics = _evaluate_classical(model_name, X_train, y_train, X_test, y_test)
            rows.append(_make_result_row(
                cohort="CA5_high_temp_proxy",
                label_source="estimated_proxy",
                train_scheme="temporal_70_30",
                model=model_name,
                cell="CA5_AGING",
                depth=np.nan,
                n_total=n_total,
                y_train=y_train,
                y_test=y_test,
                metrics=metrics,
                persist_metrics=persist_metrics,
            ))
            print(
                f"    {model_name:10s} MAE={metrics['mae']:.4f}  "
                f"gap={metrics['mae'] - persist_metrics['mae']:+.4f}"
            )
        except Exception as exc:
            print(f"    {model_name:10s} FAILED ({type(exc).__name__}: {exc})")

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary_df = _summarize_results(results_df)
    return results_df, summary_df


def plot_ca6_model_comparison(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    if summary_df.empty:
        print("Skipping CA6 model comparison plot: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))

    colors = []
    for model in summary_df["model"]:
        if model.startswith("qrc"):
            colors.append("#1f77b4")
        elif model == "persistence":
            colors.append("#7f7f7f")
        else:
            colors.append("#ff7f0e")

    ax.bar(x, summary_df["mae_mean"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=35, ha="right")
    ax.set_ylabel("Mean MAE")
    ax.set_title("CA6 clean temporal comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_figure(fig, plot_dir, "ca6_clean_model_comparison")


def plot_ca6_depth_sensitivity(qrc_df: pd.DataFrame, plot_dir: Path) -> None:
    if qrc_df.empty:
        print("Skipping CA6 depth plot: no data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    qrc_df = qrc_df.copy()
    qrc_df["depth"] = qrc_df["depth"].astype(int)
    by_depth = qrc_df.groupby("depth")["mae"].agg(["mean", "std"]).sort_index()

    ax.errorbar(
        by_depth.index,
        by_depth["mean"],
        yerr=by_depth["std"].fillna(0.0),
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="#1f77b4",
        label="CA6 sessions",
    )

    for cell_id in sorted(qrc_df["cell"].unique()):
        sub = qrc_df[qrc_df["cell"] == cell_id].sort_values("depth")
        ax.plot(sub["depth"], sub["mae"], marker=".", linewidth=0.8, alpha=0.35, color="#7aa6d1")

    ax.set_xlabel("QRC depth")
    ax.set_ylabel("MAE")
    ax.set_title("CA6 clean temporal QRC depth sensitivity")
    ax.set_xticks(DEPTH_RANGE)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, plot_dir, "ca6_qrc_depth_sensitivity")


def plot_limited_history(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    if summary_df.empty:
        print("Skipping limited-history plots: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "qrc_d1": "#1f77b4",
        "esn": "#ff7f0e",
        "linear_pc1": "#2ca02c",
        "persistence": "#7f7f7f",
    }
    for model_name in CA6_LIMITED_HISTORY_MODELS:
        sub = summary_df[summary_df["model"] == model_name].sort_values("n_train")
        if sub.empty:
            continue
        ax.plot(
            sub["n_train"],
            sub["mae_mean"],
            marker="o",
            linewidth=2.0,
            color=colors.get(model_name, "#444444"),
            label=model_name,
        )

    ax.set_xlabel("Training sweeps kept from session start")
    ax.set_ylabel("Mean MAE")
    ax.set_title("CA6 limited-history learning curves")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, plot_dir, "ca6_limited_history_learning_curve")

    fig, ax = plt.subplots(figsize=(9, 5))
    learned_models = [m for m in CA6_LIMITED_HISTORY_MODELS if m != "persistence"]
    for model_name in learned_models:
        sub = summary_df[summary_df["model"] == model_name].sort_values("n_train")
        if sub.empty:
            continue
        ax.plot(
            sub["n_train"],
            sub["mae_gap_vs_persist_mean"],
            marker="o",
            linewidth=2.0,
            color=colors.get(model_name, "#444444"),
            label=model_name,
        )

    ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Training sweeps kept from session start")
    ax.set_ylabel("Mean MAE gap vs persistence")
    ax.set_title("CA6 limited-history gap to persistence")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, plot_dir, "ca6_limited_history_persistence_gap")


def plot_ca5_stress_case(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    if summary_df.empty:
        print("Skipping CA5 plot: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_df))
    colors = []
    for model in summary_df["model"]:
        if model.startswith("qrc"):
            colors.append("#1f77b4")
        elif model == "persistence":
            colors.append("#7f7f7f")
        else:
            colors.append("#ff7f0e")

    ax.bar(x, summary_df["mae_mean"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=35, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("CA5 high-temperature stress case")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_figure(fig, plot_dir, "ca5_stress_case_comparison")


def _print_topline_summary(
    ca6_summary: pd.DataFrame,
    limited_history_summary: pd.DataFrame,
    ca5_summary: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)

    if not ca6_summary.empty:
        non_persist = ca6_summary[ca6_summary["model"] != "persistence"]
        if not non_persist.empty:
            best = non_persist.iloc[0]
            print(
                f"  CA6 clean best learned model: {best['model']} "
                f"(mean MAE={best['mae_mean']:.4f}, win_rate={best['win_rate']:.2f})"
            )
        persist = ca6_summary[ca6_summary["model"] == "persistence"]
        if not persist.empty:
            print(f"  CA6 persistence mean MAE: {persist.iloc[0]['mae_mean']:.4f}")

    if not limited_history_summary.empty:
        best_rows = []
        for n_train in sorted(limited_history_summary["n_train"].unique()):
            sub = limited_history_summary[
                (limited_history_summary["n_train"] == n_train)
                & (limited_history_summary["model"] != "persistence")
            ]
            if not sub.empty:
                best_rows.append(sub.iloc[0])
        if best_rows:
            print("  Limited-history best learned model by train size:")
            for row in best_rows:
                print(
                    f"    n_train={int(row['n_train']):2d}: {row['model']} "
                    f"(mean MAE={row['mae_mean']:.4f}, gap={row['mae_gap_vs_persist_mean']:+.4f})"
                )

    if not ca5_summary.empty:
        best = ca5_summary.iloc[0]
        print(
            f"  CA5 stress-case best model: {best['model']} "
            f"(MAE={best['mae_mean']:.4f})"
        )


def main(skip_ca5: bool = False) -> None:
    data_dir, plot_dir = get_lab_result_paths("7b")
    log_path = data_dir / "phase_7b_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Phase 7b: ESCL cohort split and limited-history follow-up")
        print("=" * 72)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")

        ca6_data = load_lab_data(include_ca5=False)
        _print_cell_summary("Loaded CA6 direct-label sessions", ca6_data)

        ca6_qrc_df, ca6_baseline_df, ca6_summary_df = run_ca6_clean_temporal(ca6_data)
        ca6_qrc_path = data_dir / "ca6_qrc_temporal.csv"
        ca6_baseline_path = data_dir / "ca6_baseline_temporal.csv"
        ca6_summary_path = data_dir / "ca6_clean_summary.csv"
        ca6_qrc_df.to_csv(ca6_qrc_path, index=False)
        ca6_baseline_df.to_csv(ca6_baseline_path, index=False)
        ca6_summary_df.to_csv(ca6_summary_path, index=False)
        print(f"\nSaved: {ca6_qrc_path} ({len(ca6_qrc_df)} rows)")
        print(f"Saved: {ca6_baseline_path} ({len(ca6_baseline_df)} rows)")
        print(f"Saved: {ca6_summary_path} ({len(ca6_summary_df)} rows)")

        limited_history_df, limited_history_summary_df = run_ca6_limited_history(ca6_data)
        limited_history_path = data_dir / "ca6_limited_history_results.csv"
        limited_history_summary_path = data_dir / "ca6_limited_history_summary.csv"
        limited_history_df.to_csv(limited_history_path, index=False)
        limited_history_summary_df.to_csv(limited_history_summary_path, index=False)
        print(f"\nSaved: {limited_history_path} ({len(limited_history_df)} rows)")
        print(f"Saved: {limited_history_summary_path} ({len(limited_history_summary_df)} rows)")

        if skip_ca5:
            print("\nStage 3 skipped by user flag.")
            ca5_results_df = pd.DataFrame(columns=RESULT_COLUMNS)
            ca5_summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        else:
            ca5_data = load_lab_data(cell_ids=["CA5_AGING"])
            _print_cell_summary("Loaded CA5 stress-case cohort", ca5_data)
            ca5_results_df, ca5_summary_df = run_ca5_stress_case(ca5_data)

        ca5_results_path = data_dir / "ca5_stress_case.csv"
        ca5_summary_path = data_dir / "ca5_stress_summary.csv"
        ca5_results_df.to_csv(ca5_results_path, index=False)
        ca5_summary_df.to_csv(ca5_summary_path, index=False)
        print(f"Saved: {ca5_results_path} ({len(ca5_results_df)} rows)")
        print(f"Saved: {ca5_summary_path} ({len(ca5_summary_df)} rows)")

        print("\n" + "=" * 72)
        print("Stage 4: Plots")
        print("=" * 72)
        plot_ca6_model_comparison(ca6_summary_df, plot_dir)
        plot_ca6_depth_sensitivity(ca6_qrc_df, plot_dir)
        plot_limited_history(limited_history_summary_df, plot_dir)
        plot_ca5_stress_case(ca5_summary_df, plot_dir)

        _print_topline_summary(ca6_summary_df, limited_history_summary_df, ca5_summary_df)
        print(f"\nCompleted: {datetime.now().isoformat()}")

    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 7b: ESCL cohort split and limited-history follow-up."
    )
    parser.add_argument(
        "--skip-ca5",
        action="store_true",
        help="Skip the separate CA5 high-temperature stress case.",
    )
    args = parser.parse_args()
    main(skip_ca5=args.skip_ca5)
