"""Phase 5 Stage 2: Statistical Significance Testing for QRC vs classical baselines."""

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

# Fix sys.path for imports from src/
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CELL_IDS, N_PCA, PROJECT_ROOT, RANDOM_STATE, get_result_paths
from data_loader import load_stanford_data
from phase_5.config import get_stage_paths

sys.path.insert(0, str(PROJECT_ROOT / "src"))
import phase_3.models as phase3_models
import phase_4.config as phase4_config
import phase_4.qrc_model as phase4_qrc_model
from phase_3.models import get_model_pipeline
from phase_4.qrc_model import QuantumReservoir


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
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


def _fit_pca_in_fold(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_components = min(N_PCA, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def _safe_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


@contextmanager
def _patched_qrc_seed(seed: int):
    """Patch phase_4 seed constants used inside QuantumReservoir."""
    old_qrc_seed = phase4_qrc_model.RANDOM_STATE
    old_cfg_seed = phase4_config.RANDOM_STATE
    phase4_qrc_model.RANDOM_STATE = seed
    phase4_config.RANDOM_STATE = seed
    try:
        yield
    finally:
        phase4_qrc_model.RANDOM_STATE = old_qrc_seed
        phase4_config.RANDOM_STATE = old_cfg_seed


def _load_existing_results() -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    phase4_data_dir, _ = get_result_paths(4)
    phase3_data_dir, _ = get_result_paths(3)

    qrc_csv_path = phase4_data_dir / "qrc_noiseless.csv"
    loco_csv_path = phase3_data_dir / "loco_results.csv"

    if not qrc_csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {qrc_csv_path}")
    if not loco_csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {loco_csv_path}")

    qrc_results = pd.read_csv(qrc_csv_path)
    loco_results = pd.read_csv(loco_csv_path)
    return qrc_results, loco_results, qrc_csv_path, loco_csv_path


def _extract_fold_mae_tables(
    qrc_results: pd.DataFrame,
    loco_results: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qrc_df = qrc_results.copy()
    if "depth" in qrc_df.columns:
        qrc_df["depth"] = pd.to_numeric(qrc_df["depth"], errors="coerce")

    qrc_fold = qrc_df[
        (qrc_df.get("stage", "") == "noiseless")
        & (qrc_df.get("regime", "") == "loco")
        & (qrc_df["depth"] == 1)
    ][["test_cell", "mae", "n_test"]].drop_duplicates(subset=["test_cell"])

    xgb_fold = loco_results[
        (loco_results.get("regime", "") == "loco")
        & (loco_results.get("model", "") == "xgboost")
        & (loco_results.get("feature_space", "") == "pca6")
    ][["test_cell", "mae", "n_test"]].drop_duplicates(subset=["test_cell"])

    esn_fold = loco_results[
        (loco_results.get("regime", "") == "loco")
        & (loco_results.get("model", "") == "esn")
        & (loco_results.get("feature_space", "") == "pca6")
    ][["test_cell", "mae", "n_test"]].drop_duplicates(subset=["test_cell"])

    if qrc_fold.empty:
        raise ValueError("Could not find noiseless LOCO depth=1 QRC rows in qrc_noiseless.csv")
    if xgb_fold.empty:
        raise ValueError("Could not find LOCO xgboost pca6 rows in loco_results.csv")
    if esn_fold.empty:
        raise ValueError("Could not find LOCO esn pca6 rows in loco_results.csv")

    return qrc_fold, xgb_fold, esn_fold


def _run_qrc_loco_seed(cell_data: Dict[str, Dict[str, np.ndarray]], seed: int) -> pd.DataFrame:
    rows: List[Dict] = []
    with _patched_qrc_seed(seed):
        for test_cell in CELL_IDS:
            train_cells = [c for c in CELL_IDS if c != test_cell]
            X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
            y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
            X_test_raw = cell_data[test_cell]["X_raw"]
            y_test = cell_data[test_cell]["y"]
            block_ids_test = cell_data[test_cell]["block_ids"]

            X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)
            train_groups = np.concatenate(
                [np.full(cell_data[c]["y"].shape[0], c, dtype=object) for c in train_cells]
            )

            qrc = QuantumReservoir(
                depth=1,
                use_zz=True,
                observable_set="Z",
                add_random_rotations=True,
            )
            qrc.fit(X_train_pca, y_train, groups=train_groups)
            y_pred = qrc.predict(X_test_pca)
            abs_error = np.abs(y_test - y_pred)

            for sample_idx in range(len(y_test)):
                rows.append(
                    {
                        "model": "qrc_noiseless_d1",
                        "seed": seed,
                        "test_cell": test_cell,
                        "sample_idx": int(sample_idx),
                        "block_id": int(block_ids_test[sample_idx]),
                        "y_true": float(y_test[sample_idx]),
                        "y_pred": float(y_pred[sample_idx]),
                        "abs_error": float(abs_error[sample_idx]),
                    }
                )

    return pd.DataFrame(rows)


def _run_classical_loco_model(
    cell_data: Dict[str, Dict[str, np.ndarray]],
    model_name: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    phase3_models.RANDOM_STATE = RANDOM_STATE

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]
        X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_raw = cell_data[test_cell]["X_raw"]
        y_test = cell_data[test_cell]["y"]
        block_ids_test = cell_data[test_cell]["block_ids"]

        X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)
        model = get_model_pipeline(model_name)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        abs_error = np.abs(y_test - y_pred)

        for sample_idx in range(len(y_test)):
            rows.append(
                {
                    "model": model_name,
                    "seed": RANDOM_STATE,
                    "test_cell": test_cell,
                    "sample_idx": int(sample_idx),
                    "block_id": int(block_ids_test[sample_idx]),
                    "y_true": float(y_test[sample_idx]),
                    "y_pred": float(y_pred[sample_idx]),
                    "abs_error": float(abs_error[sample_idx]),
                }
            )

    return pd.DataFrame(rows)


def _run_qrc_multiseed(
    cell_data: Dict[str, Dict[str, np.ndarray]],
    seeds: Iterable[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seeds_list = list(seeds)
    for idx, seed in enumerate(seeds_list, start=1):
        print(f"Running QRC seed {seed} ({idx}/{len(seeds_list)})")
        frame = _run_qrc_loco_seed(cell_data, seed)
        frames.append(frame)
        seed_macro = frame.groupby("test_cell")["abs_error"].mean().mean()
        print(f"  seed macro_mae={seed_macro:.6f}")
    return pd.concat(frames, ignore_index=True)


def _compute_qrc_seed_metrics(qrc_samples: pd.DataFrame) -> pd.DataFrame:
    per_cell = (
        qrc_samples.groupby(["seed", "test_cell"])["abs_error"]
        .mean()
        .reset_index(name="cell_mae")
    )
    macro = per_cell.groupby("seed")["cell_mae"].mean().reset_index(name="macro_mae")
    micro = qrc_samples.groupby("seed")["abs_error"].mean().reset_index(name="micro_mae")
    out = macro.merge(micro, on="seed", how="inner").sort_values("seed").reset_index(drop=True)
    return out


def _merge_paired_errors(qrc_samples_seed: pd.DataFrame, xgb_samples: pd.DataFrame) -> pd.DataFrame:
    left = qrc_samples_seed[
        ["test_cell", "sample_idx", "block_id", "y_true", "abs_error"]
    ].rename(columns={"abs_error": "abs_error_qrc", "y_true": "y_true_qrc"})
    right = xgb_samples[
        ["test_cell", "sample_idx", "block_id", "y_true", "abs_error"]
    ].rename(columns={"abs_error": "abs_error_xgboost", "y_true": "y_true_xgboost"})

    merged = left.merge(
        right,
        on=["test_cell", "sample_idx", "block_id"],
        how="inner",
    ).sort_values(["test_cell", "sample_idx"]).reset_index(drop=True)

    if len(merged) == 0:
        raise ValueError("No paired per-sample rows after merge for QRC vs XGBoost")

    if not np.allclose(merged["y_true_qrc"].to_numpy(), merged["y_true_xgboost"].to_numpy()):
        raise ValueError("Mismatch in y_true values after pairing QRC and XGBoost samples")

    return merged


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std = diff.std(ddof=1)
    if np.isclose(std, 0.0):
        return float("nan")
    return float(diff.mean() / std)


def _bootstrap_mae_diff_cell_level(
    paired_errors: pd.DataFrame,
    n_resamples: int = 10000,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float, float, np.ndarray]:
    rng = np.random.RandomState(seed)

    qrc_err = paired_errors["abs_error_qrc"].to_numpy()
    xgb_err = paired_errors["abs_error_xgboost"].to_numpy()
    cell_series = paired_errors["test_cell"].to_numpy()

    cell_to_pos = {cell: np.where(cell_series == cell)[0] for cell in CELL_IDS}
    for cell, pos in cell_to_pos.items():
        if len(pos) == 0:
            raise ValueError(f"No paired samples for cell {cell}")

    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled_cells = rng.choice(CELL_IDS, size=len(CELL_IDS), replace=True)
        sampled_pos = np.concatenate([cell_to_pos[c] for c in sampled_cells])
        diffs[i] = qrc_err[sampled_pos].mean() - xgb_err[sampled_pos].mean()

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(ci_low), float(ci_high), diffs


def _build_per_cell_table(
    qrc_fold: pd.DataFrame,
    xgb_fold: pd.DataFrame,
    esn_fold: pd.DataFrame,
    qrc_seed42_samples: pd.DataFrame,
    xgb_samples: pd.DataFrame,
    esn_samples: pd.DataFrame,
) -> pd.DataFrame:
    qrc_map = qrc_fold.set_index("test_cell")["mae"].to_dict()
    xgb_map = xgb_fold.set_index("test_cell")["mae"].to_dict()
    esn_map = esn_fold.set_index("test_cell")["mae"].to_dict()
    n_test_map = qrc_fold.set_index("test_cell")["n_test"].astype(int).to_dict()

    qrc_sample_map = qrc_seed42_samples.groupby("test_cell")["abs_error"].mean().to_dict()
    xgb_sample_map = xgb_samples.groupby("test_cell")["abs_error"].mean().to_dict()
    esn_sample_map = esn_samples.groupby("test_cell")["abs_error"].mean().to_dict()

    rows: List[Dict] = []
    for cell in CELL_IDS:
        qrc_mae = float(qrc_map.get(cell, qrc_sample_map[cell]))
        xgb_mae = float(xgb_map.get(cell, xgb_sample_map[cell]))
        esn_mae = float(esn_map.get(cell, esn_sample_map[cell]))
        n_test = int(n_test_map.get(cell, len(qrc_seed42_samples[qrc_seed42_samples["test_cell"] == cell])))

        rows.append(
            {
                "test_cell": cell,
                "n_test": n_test,
                "qrc_mae": qrc_mae,
                "xgboost_mae": xgb_mae,
                "esn_mae": esn_mae,
                "qrc_minus_xgboost": qrc_mae - xgb_mae,
                "qrc_minus_esn": qrc_mae - esn_mae,
            }
        )

    return pd.DataFrame(rows)


def _build_macro_micro_table(
    per_cell_table: pd.DataFrame,
    qrc_seed42_samples: pd.DataFrame,
    xgb_samples: pd.DataFrame,
    esn_samples: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "method": "qrc_noiseless_d1_seed42",
            "macro_mae": float(per_cell_table["qrc_mae"].mean()),
            "micro_mae": float(qrc_seed42_samples["abs_error"].mean()),
            "n_folds": int(per_cell_table.shape[0]),
            "n_samples": int(qrc_seed42_samples.shape[0]),
        },
        {
            "method": "xgboost_pca6_seed42",
            "macro_mae": float(per_cell_table["xgboost_mae"].mean()),
            "micro_mae": float(xgb_samples["abs_error"].mean()),
            "n_folds": int(per_cell_table.shape[0]),
            "n_samples": int(xgb_samples.shape[0]),
        },
        {
            "method": "esn_pca6_seed42",
            "macro_mae": float(per_cell_table["esn_mae"].mean()),
            "micro_mae": float(esn_samples["abs_error"].mean()),
            "n_folds": int(per_cell_table.shape[0]),
            "n_samples": int(esn_samples.shape[0]),
        },
    ]
    return pd.DataFrame(rows)


def _build_bar_summary(loco_results: pd.DataFrame, qrc_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    classical = (
        loco_results[loco_results["regime"] == "loco"]
        .groupby("model")["mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "model": "method",
                "mean": "mae_mean",
                "std": "mae_std",
                "count": "n",
            }
        )
    )

    qrc_row = pd.DataFrame(
        [
            {
                "method": "qrc_noiseless_d1_20seed",
                "mae_mean": float(qrc_seed_metrics["macro_mae"].mean()),
                "mae_std": _safe_std(qrc_seed_metrics["macro_mae"]),
                "n": int(qrc_seed_metrics.shape[0]),
            }
        ]
    )

    out = pd.concat([classical, qrc_row], ignore_index=True)
    out = out.sort_values("mae_mean", ascending=True).reset_index(drop=True)
    return out


def _plot_bar_with_error(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(summary_df))
    y = summary_df["mae_mean"].to_numpy()
    yerr = summary_df["mae_std"].fillna(0.0).to_numpy()
    labels = summary_df["method"].tolist()

    colors = [
        "tab:blue" if method == "qrc_noiseless_d1_20seed" else "tab:gray"
        for method in labels
    ]

    ax.bar(x, y, yerr=yerr, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("QRC (20-seed mean+std) vs Classical Baselines (LOCO)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_figure(fig, plot_dir, "qrc_multiseed_vs_classical")


def main(n_seeds: int = 20, n_bootstrap: int = 10000) -> None:
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if n_bootstrap < 1000:
        raise ValueError("n_bootstrap must be >= 1000")

    data_dir, plot_dir = get_stage_paths("stage_2")
    log_path = data_dir / "stage_2_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 5 Stage 2: Statistical Significance Testing (QRC vs Classical)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")
        print(f"n_seeds={n_seeds}, n_bootstrap={n_bootstrap}")

        qrc_results, loco_results, qrc_csv_path, loco_csv_path = _load_existing_results()
        print("\nLoaded existing results:")
        print(f"  {qrc_csv_path} rows={len(qrc_results)}")
        print(f"  {loco_csv_path} rows={len(loco_results)}")

        qrc_fold, xgb_fold, esn_fold = _extract_fold_mae_tables(qrc_results, loco_results)
        print("\nSelected fold-level baselines:")
        print(f"  QRC noiseless depth=1 LOCO rows={len(qrc_fold)}")
        print(f"  XGBoost pca6 LOCO rows={len(xgb_fold)}")
        print(f"  ESN pca6 LOCO rows={len(esn_fold)}")

        print("\nPer-sample errors are not present in the phase CSVs; rerunning predictions.")
        cell_data = load_stanford_data()

        seeds = [RANDOM_STATE + i for i in range(n_seeds)]
        qrc_samples = _run_qrc_multiseed(cell_data, seeds)
        qrc_seed_metrics = _compute_qrc_seed_metrics(qrc_samples)

        xgb_samples = _run_classical_loco_model(cell_data, "xgboost")
        esn_samples = _run_classical_loco_model(cell_data, "esn")

        qrc_samples_path = data_dir / "qrc_multiseed_per_sample.csv"
        xgb_samples_path = data_dir / "xgboost_per_sample.csv"
        esn_samples_path = data_dir / "esn_per_sample.csv"
        qrc_seed_metrics_path = data_dir / "qrc_seed_metrics.csv"

        qrc_samples.to_csv(qrc_samples_path, index=False)
        xgb_samples.to_csv(xgb_samples_path, index=False)
        esn_samples.to_csv(esn_samples_path, index=False)
        qrc_seed_metrics.to_csv(qrc_seed_metrics_path, index=False)

        qrc_seed42_samples = qrc_samples[qrc_samples["seed"] == RANDOM_STATE].copy()
        if qrc_seed42_samples.empty:
            raise ValueError(f"Seed {RANDOM_STATE} not present in qrc_multiseed_per_sample")

        paired = _merge_paired_errors(qrc_seed42_samples, xgb_samples)

        per_cell_table = _build_per_cell_table(
            qrc_fold=qrc_fold,
            xgb_fold=xgb_fold,
            esn_fold=esn_fold,
            qrc_seed42_samples=qrc_seed42_samples,
            xgb_samples=xgb_samples,
            esn_samples=esn_samples,
        )
        macro_micro_table = _build_macro_micro_table(
            per_cell_table=per_cell_table,
            qrc_seed42_samples=qrc_seed42_samples,
            xgb_samples=xgb_samples,
            esn_samples=esn_samples,
        )

        q_err = paired["abs_error_qrc"].to_numpy()
        x_err = paired["abs_error_xgboost"].to_numpy()

        wilcoxon_result = wilcoxon(q_err, x_err, alternative="two-sided", zero_method="wilcox")
        mean_diff = float(q_err.mean() - x_err.mean())
        d_value = _cohens_d_paired(q_err, x_err)
        boot_mean, ci_low, ci_high, boot_diffs = _bootstrap_mae_diff_cell_level(
            paired_errors=paired,
            n_resamples=n_bootstrap,
            seed=RANDOM_STATE,
        )

        stats_table = pd.DataFrame(
            [
                {"metric": "n_paired_samples", "value": float(len(paired))},
                {"metric": "mean_abs_error_qrc", "value": float(q_err.mean())},
                {"metric": "mean_abs_error_xgboost", "value": float(x_err.mean())},
                {"metric": "mae_diff_qrc_minus_xgboost", "value": mean_diff},
                {"metric": "wilcoxon_statistic", "value": float(wilcoxon_result.statistic)},
                {"metric": "wilcoxon_p_value", "value": float(wilcoxon_result.pvalue)},
                {"metric": "bootstrap_mean_diff", "value": boot_mean},
                {"metric": "bootstrap_ci_low_95", "value": ci_low},
                {"metric": "bootstrap_ci_high_95", "value": ci_high},
                {"metric": "cohens_d_paired", "value": d_value},
            ]
        )

        qrc_seed_summary = pd.DataFrame(
            [
                {
                    "n_seeds": int(n_seeds),
                    "macro_mae_mean": float(qrc_seed_metrics["macro_mae"].mean()),
                    "macro_mae_std": _safe_std(qrc_seed_metrics["macro_mae"]),
                    "micro_mae_mean": float(qrc_seed_metrics["micro_mae"].mean()),
                    "micro_mae_std": _safe_std(qrc_seed_metrics["micro_mae"]),
                }
            ]
        )

        per_cell_path = data_dir / "per_cell_mae_qrc_xgboost_esn.csv"
        macro_micro_path = data_dir / "macro_micro_mae.csv"
        stats_path = data_dir / "statistical_tests.csv"
        seed_summary_path = data_dir / "qrc_multiseed_summary.csv"
        paired_path = data_dir / "paired_abs_errors_qrc_vs_xgboost.csv"
        bootstrap_path = data_dir / "bootstrap_diffs_cell_level.csv"

        per_cell_table.to_csv(per_cell_path, index=False)
        macro_micro_table.to_csv(macro_micro_path, index=False)
        stats_table.to_csv(stats_path, index=False)
        qrc_seed_summary.to_csv(seed_summary_path, index=False)
        paired.to_csv(paired_path, index=False)
        pd.DataFrame({"mae_diff_qrc_minus_xgboost": boot_diffs}).to_csv(bootstrap_path, index=False)

        bar_summary = _build_bar_summary(loco_results, qrc_seed_metrics)
        bar_summary_path = data_dir / "bar_summary_qrc_vs_classical.csv"
        bar_summary.to_csv(bar_summary_path, index=False)
        _plot_bar_with_error(bar_summary, plot_dir)

        print("\nPer-cell MAE table:")
        print(per_cell_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

        print("\nMacro/Micro MAE table:")
        print(macro_micro_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

        print("\nQRC multi-seed summary:")
        print(
            "  macro_mae mean+/-std = "
            f"{qrc_seed_summary.iloc[0]['macro_mae_mean']:.6f}+/-{qrc_seed_summary.iloc[0]['macro_mae_std']:.6f}"
        )
        print(
            "  micro_mae mean+/-std = "
            f"{qrc_seed_summary.iloc[0]['micro_mae_mean']:.6f}+/-{qrc_seed_summary.iloc[0]['micro_mae_std']:.6f}"
        )

        print("\nSignificance tests (QRC vs XGBoost, paired per-sample abs errors):")
        print(f"  Wilcoxon statistic={wilcoxon_result.statistic:.6f}, p_value={wilcoxon_result.pvalue:.6e}")
        print(f"  MAE diff (QRC-XGBoost)={mean_diff:.6f}")
        print(f"  Bootstrap 95% CI diff=[{ci_low:.6f}, {ci_high:.6f}]")
        print(f"  Cohen d (paired)={d_value:.6f}")

        print("\nSaved outputs:")
        print(f"  {qrc_samples_path}")
        print(f"  {xgb_samples_path}")
        print(f"  {esn_samples_path}")
        print(f"  {qrc_seed_metrics_path}")
        print(f"  {per_cell_path}")
        print(f"  {macro_micro_path}")
        print(f"  {stats_path}")
        print(f"  {seed_summary_path}")
        print(f"  {paired_path}")
        print(f"  {bootstrap_path}")
        print(f"  {bar_summary_path}")

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 Stage 2 statistical significance testing.")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of QRC seeds (default: 20).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for CI (default: 10000).",
    )
    args = parser.parse_args()
    main(n_seeds=args.n_seeds, n_bootstrap=args.n_bootstrap)
