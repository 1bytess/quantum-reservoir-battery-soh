"""Phase 5 Stage 4: Nested LOCO Cross-Validation"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Required pattern for local imports from __TEMP.
# Fix sys.path for imports from src
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

from config import CELL_IDS, N_PCA, RANDOM_STATE, PROJECT_ROOT  # noqa: F401
from phase_5.config import get_stage_paths

# Required pattern for imports from src.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase_3.models import get_model_pipeline
from phase_4.qrc_model import QuantumReservoir

from data_loader import load_stanford_data


QRC_DEPTHS = [1, 2, 3, 4]
QRC_OBSERVABLE_GRID = [
    ("Z_only", False, "Z"),
    ("Z+ZZ", True, "Z"),
]
CLASSICAL_MODELS = ["xgboost", "esn", "mlp"]


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
    """Fit PCA on train split only and transform train/test."""
    n_components = min(N_PCA, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _stack_cells(
    cell_data: Dict[str, Dict[str, np.ndarray]],
    cells: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([cell_data[c]["X_raw"] for c in cells])
    y = np.concatenate([cell_data[c]["y"] for c in cells])
    groups = np.concatenate(
        [np.full(cell_data[c]["y"].shape[0], c, dtype=object) for c in cells]
    )
    return X, y, groups


def _run_inner_loop(
    cell_data: Dict[str, Dict[str, np.ndarray]],
    outer_train_cells: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run inner LOCO folds for QRC grid and classical models."""
    qrc_rows: List[Dict] = []
    classical_rows: List[Dict] = []

    for val_cell in outer_train_cells:
        inner_train_cells = [c for c in outer_train_cells if c != val_cell]

        X_train_raw, y_train, train_groups = _stack_cells(cell_data, inner_train_cells)
        X_val_raw = cell_data[val_cell]["X_raw"]
        y_val = cell_data[val_cell]["y"]

        # Required: PCA in-fold for every inner split.
        X_train, X_val = _fit_pca_in_fold(X_train_raw, X_val_raw)

        for depth in QRC_DEPTHS:
            for observable_label, use_zz, observable_set in QRC_OBSERVABLE_GRID:
                qrc = QuantumReservoir(
                    depth=depth,
                    use_zz=use_zz,
                    use_classical_fallback=False,
                    add_random_rotations=True,
                    observable_set=observable_set,
                )
                qrc.fit(X_train, y_train, groups=train_groups)
                y_pred = qrc.predict(X_val)
                val_mae = float(mean_absolute_error(y_val, y_pred))

                qrc_rows.append(
                    {
                        "val_cell": val_cell,
                        "inner_train_cells": "+".join(inner_train_cells),
                        "depth": int(depth),
                        "observable": observable_label,
                        "val_mae": val_mae,
                    }
                )

        for model_name in CLASSICAL_MODELS:
            model = get_model_pipeline(model_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, y_pred))

            classical_rows.append(
                {
                    "val_cell": val_cell,
                    "inner_train_cells": "+".join(inner_train_cells),
                    "model": model_name,
                    "val_mae": val_mae,
                }
            )

    return pd.DataFrame(qrc_rows), pd.DataFrame(classical_rows)


def _select_best_qrc_hparams(inner_qrc_df: pd.DataFrame) -> Tuple[int, str, float]:
    """Pick best QRC hyperparameters by mean inner validation MAE."""
    if inner_qrc_df.empty:
        raise ValueError("Inner QRC results are empty; cannot select hyperparameters")

    summary = (
        inner_qrc_df.groupby(["depth", "observable"], as_index=False)["val_mae"]
        .mean()
        .rename(columns={"val_mae": "mean_inner_val_mae"})
    )

    obs_rank = {"Z_only": 0, "Z+ZZ": 1}
    summary["obs_rank"] = summary["observable"].map(obs_rank).fillna(99)
    summary = summary.sort_values(
        ["mean_inner_val_mae", "depth", "obs_rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    best = summary.iloc[0]
    return int(best["depth"]), str(best["observable"]), float(best["mean_inner_val_mae"])


def _mean_inner_classical_mae(inner_classical_df: pd.DataFrame, model_name: str) -> float:
    if inner_classical_df.empty:
        return float("nan")
    sub = inner_classical_df[inner_classical_df["model"] == model_name]
    if sub.empty:
        return float("nan")
    return float(sub["val_mae"].mean())


def _plot_nested_loco_bars(overall_mae: Dict[str, float], plot_dir: Path) -> None:
    methods = list(overall_mae.keys())
    values = [overall_mae[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(methods))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Nested LOCO MAE")
    ax.set_title("Nested LOCO MAE Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom")

    _save_figure(fig, plot_dir, "nested_loco_mae_comparison")


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_4")
    log_path = data_dir / "stage_4_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 5 Stage 4: Nested LOCO for QRC Hyperparameter Selection")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")
        print(f"Cells: {CELL_IDS}")
        print(f"QRC depths: {QRC_DEPTHS}")
        print("QRC observables: ['Z_only', 'Z+ZZ']")
        print(f"Classical models: {CLASSICAL_MODELS}")

        cell_data = load_stanford_data()

        fold_rows: List[Dict] = []
        sample_rows: List[Dict] = []
        inner_qrc_all: List[pd.DataFrame] = []
        inner_classical_all: List[pd.DataFrame] = []

        print("\nRunning nested LOCO...")
        for outer_idx, test_cell in enumerate(CELL_IDS, start=1):
            print("\n" + "-" * 70)
            print(f"Outer fold {outer_idx}/{len(CELL_IDS)} | test_cell={test_cell}")

            outer_train_cells = [c for c in CELL_IDS if c != test_cell]
            inner_qrc_df, inner_classical_df = _run_inner_loop(cell_data, outer_train_cells)

            inner_qrc_df.insert(0, "test_cell", test_cell)
            inner_classical_df.insert(0, "test_cell", test_cell)
            inner_qrc_all.append(inner_qrc_df)
            inner_classical_all.append(inner_classical_df)

            selected_depth, selected_observable, best_inner_mae = _select_best_qrc_hparams(inner_qrc_df)
            selected_use_zz = selected_observable == "Z+ZZ"

            inner_classical_maes = {}
            for cm in CLASSICAL_MODELS:
                inner_classical_maes[cm] = _mean_inner_classical_mae(inner_classical_df, cm)

            print(
                "Selected QRC hyperparameters: "
                f"depth={selected_depth}, observable={selected_observable}, "
                f"mean_inner_MAE={best_inner_mae:.4f}"
            )
            inner_str = ", ".join(f"{cm}={inner_classical_maes[cm]:.4f}" for cm in CLASSICAL_MODELS)
            print(f"Inner mean MAE (classical): {inner_str}")

            X_outer_train_raw, y_outer_train, outer_groups = _stack_cells(cell_data, outer_train_cells)
            X_outer_test_raw = cell_data[test_cell]["X_raw"]
            y_outer_test = cell_data[test_cell]["y"]
            block_ids = cell_data[test_cell]["block_ids"]

            # Required: PCA in-fold for every outer split.
            X_outer_train, X_outer_test = _fit_pca_in_fold(X_outer_train_raw, X_outer_test_raw)

            qrc = QuantumReservoir(
                depth=selected_depth,
                use_zz=selected_use_zz,
                use_classical_fallback=False,
                add_random_rotations=True,
                observable_set="Z",
            )
            qrc.fit(X_outer_train, y_outer_train, groups=outer_groups)
            qrc_pred = qrc.predict(X_outer_test)
            qrc_mae = float(mean_absolute_error(y_outer_test, qrc_pred))

            # Evaluate all classical models on outer fold
            classical_preds = {}
            classical_maes = {}
            for cm in CLASSICAL_MODELS:
                clf = get_model_pipeline(cm)
                clf.fit(X_outer_train, y_outer_train)
                pred = clf.predict(X_outer_test)
                classical_preds[cm] = pred
                classical_maes[cm] = float(mean_absolute_error(y_outer_test, pred))

            fold_row = {
                "test_cell": test_cell,
                "selected_depth": int(selected_depth),
                "selected_obs": selected_observable,
                "mean_inner_qrc_mae": best_inner_mae,
                "QRC_MAE": qrc_mae,
                "n_test_samples": int(len(y_outer_test)),
            }
            for cm in CLASSICAL_MODELS:
                fold_row[f"mean_inner_{cm}_mae"] = inner_classical_maes[cm]
                fold_row[f"{cm.upper()}_MAE"] = classical_maes[cm]
            fold_rows.append(fold_row)

            for idx in range(len(y_outer_test)):
                sample_row = {
                    "test_cell": test_cell,
                    "sample_index": int(idx),
                    "block_id": int(block_ids[idx]),
                    "y_true": float(y_outer_test[idx]),
                    "qrc_pred": float(qrc_pred[idx]),
                    "selected_depth": int(selected_depth),
                    "selected_obs": selected_observable,
                    "QRC_MAE": qrc_mae,
                }
                for cm in CLASSICAL_MODELS:
                    sample_row[f"{cm}_pred"] = float(classical_preds[cm][idx])
                    sample_row[f"{cm.upper()}_MAE"] = classical_maes[cm]
                sample_rows.append(sample_row)

            outer_str = ", ".join(f"{cm.upper()}={classical_maes[cm]:.4f}" for cm in CLASSICAL_MODELS)
            print(f"Outer test MAE: QRC={qrc_mae:.4f}, {outer_str}")

        fold_df = pd.DataFrame(fold_rows)
        sample_df = pd.DataFrame(sample_rows)
        inner_qrc_full = pd.concat(inner_qrc_all, ignore_index=True) if inner_qrc_all else pd.DataFrame()
        inner_classical_full = (
            pd.concat(inner_classical_all, ignore_index=True) if inner_classical_all else pd.DataFrame()
        )

        overall_maes = {}
        print("\nPer-fold table:")
        if fold_df.empty:
            print("No fold results produced.")
            overall_maes["QRC"] = float("nan")
            for cm in CLASSICAL_MODELS:
                overall_maes[cm.upper()] = float("nan")
        else:
            mae_cols = ["QRC_MAE"] + [f"{cm.upper()}_MAE" for cm in CLASSICAL_MODELS]
            display_cols = ["test_cell", "selected_depth", "selected_obs"] + mae_cols
            print(fold_df[display_cols].to_string(index=False))
            overall_maes["QRC"] = float(fold_df["QRC_MAE"].mean())
            for cm in CLASSICAL_MODELS:
                overall_maes[cm.upper()] = float(fold_df[f"{cm.upper()}_MAE"].mean())

        print("\nOverall nested LOCO MAE:")
        for name, val in overall_maes.items():
            print(f"  {name:10s}: {val:.4f}")

        fold_df_with_type = fold_df.copy()
        fold_df_with_type.insert(0, "row_type", "fold_summary")

        sample_df_with_type = sample_df.copy()
        sample_df_with_type.insert(0, "row_type", "sample_prediction")

        results_df = pd.concat([fold_df_with_type, sample_df_with_type], ignore_index=True, sort=False)
        out_path = data_dir / "nested_loco_results.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(results_df)} rows)")

        inner_qrc_path = data_dir / "nested_loco_inner_qrc.csv"
        inner_classical_path = data_dir / "nested_loco_inner_classical.csv"
        inner_qrc_full.to_csv(inner_qrc_path, index=False)
        inner_classical_full.to_csv(inner_classical_path, index=False)
        print(f"Saved: {inner_qrc_path} ({len(inner_qrc_full)} rows)")
        print(f"Saved: {inner_classical_path} ({len(inner_classical_full)} rows)")

        overall_rows = [{"model": name, "nested_loco_mae": val} for name, val in overall_maes.items()]
        overall_df = pd.DataFrame(overall_rows)
        overall_path = data_dir / "nested_loco_overall_mae.csv"
        overall_df.to_csv(overall_path, index=False)
        print(f"Saved: {overall_path}")

        if not fold_df.empty:
            _plot_nested_loco_bars(overall_maes, plot_dir)

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
