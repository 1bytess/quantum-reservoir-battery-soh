"""Phase 8 Stage 2: Nested LOCO CV on Warwick for fair hyperparameter search.

Runs nested leave-one-cell-out evaluation on the Warwick DIB dataset:
  - Outer loop: 24-fold LOCO (1 held-out cell)
  - Inner loop: LeaveOneOut on the 23 outer-train samples

Classical baselines use GridSearchCV with the shared Phase 3 grids:
  ridge, svr, xgboost, esn, rff

QRC uses manual depth selection in the inner loop:
  depths 1-4 with Z+ZZ observables only

Outputs
-------
result/phase_8/stage_2/data/nested_warwick_loco_predictions.csv
result/phase_8/stage_2/data/nested_warwick_loco_summary.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    _SRC_DIR = Path(__file__).resolve().parents[1]
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from config import N_PCA, PROJECT_ROOT, RANDOM_STATE
    from data_loader_warwick import get_warwick_arrays
    from phase_3.config import HYPERPARAM_GRIDS
    from phase_3.models import get_model_pipeline
    from phase_4.circuit import QISKIT_AVAILABLE
    from phase_4.qrc_model import QuantumReservoir
else:
    from ..config import N_PCA, PROJECT_ROOT, RANDOM_STATE
    from ..data_loader_warwick import get_warwick_arrays
    from ..phase_3.config import HYPERPARAM_GRIDS
    from ..phase_3.models import get_model_pipeline
    from ..phase_4.circuit import QISKIT_AVAILABLE
    from ..phase_4.qrc_model import QuantumReservoir


warnings.filterwarnings("ignore")

CLASSICAL_MODELS = ["ridge", "svr", "xgboost", "esn", "rff"]
QRC_DEPTHS = [1, 2, 3, 4]
QRC_OBSERVABLE = "Z+ZZ"
WARWICK_NOMINAL_CAPACITY_AH = 5.0


def _get_stage_data_dir() -> Path:
    data_dir = PROJECT_ROOT / "result" / "phase_8" / "stage_2" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _fit_pca_in_fold(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    n_components: int = N_PCA,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler+PCA on the outer-train split only."""
    n_components = min(n_components, X_train_raw.shape[0], X_train_raw.shape[1])
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca


def _json_dumps(value: dict[str, Any]) -> str:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    cleaned = {str(k): _convert(v) for k, v in value.items()}
    return json.dumps(cleaned, sort_keys=True)


def _build_classical_search(model_name: str) -> GridSearchCV:
    estimator = get_model_pipeline(model_name)
    if model_name == "xgboost":
        estimator.set_params(xgb__n_jobs=1)

    return GridSearchCV(
        estimator=estimator,
        param_grid=HYPERPARAM_GRIDS[model_name],
        cv=LeaveOneOut(),
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )


def _run_qrc_depth_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, Any]:
    inner_cv = LeaveOneOut()
    depth_rows: list[dict[str, Any]] = []

    for depth in QRC_DEPTHS:
        fold_maes: list[float] = []
        fold_best_alphas: list[float] = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            X_inner_val = X_train[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            qrc = QuantumReservoir(
                depth=depth,
                use_zz=True,
                ridge_alpha=None,
                use_classical_fallback=False,
                add_random_rotations=True,
                observable_set="Z",
            )
            qrc.fit(
                X_inner_train,
                y_inner_train,
                groups=np.arange(len(y_inner_train)),
            )
            y_inner_pred = qrc.predict(X_inner_val)
            fold_maes.append(float(mean_absolute_error(y_inner_val, y_inner_pred)))
            fold_best_alphas.append(float(getattr(qrc, "best_alpha_", np.nan)))

        depth_rows.append(
            {
                "depth": depth,
                "mean_inner_mae": float(np.mean(fold_maes)),
                "std_inner_mae": float(np.std(fold_maes, ddof=0)),
                "mean_inner_alpha": float(np.nanmean(fold_best_alphas)),
            }
        )

    depth_df = pd.DataFrame(depth_rows).sort_values(
        ["mean_inner_mae", "depth"], ascending=[True, True]
    )
    best_depth = int(depth_df.iloc[0]["depth"])
    best_inner_mae = float(depth_df.iloc[0]["mean_inner_mae"])

    final_qrc = QuantumReservoir(
        depth=best_depth,
        use_zz=True,
        ridge_alpha=None,
        use_classical_fallback=False,
        add_random_rotations=True,
        observable_set="Z",
    )
    final_qrc.fit(X_train, y_train, groups=np.arange(len(y_train)))
    y_pred = float(final_qrc.predict(X_test)[0])

    return {
        "model": "qrc",
        "y_pred": y_pred,
        "inner_mae": best_inner_mae,
        "selected_depth": best_depth,
        "selected_ridge_alpha": float(getattr(final_qrc, "best_alpha_", np.nan)),
        "selected_params": _json_dumps(
            {
                "depth": best_depth,
                "observable": QRC_OBSERVABLE,
                "ridge_alpha": float(getattr(final_qrc, "best_alpha_", np.nan)),
            }
        ),
        "search_trace": _json_dumps(
            {
                f"depth_{int(row.depth)}": {
                    "mean_inner_mae": float(row.mean_inner_mae),
                    "std_inner_mae": float(row.std_inner_mae),
                    "mean_inner_alpha": float(row.mean_inner_alpha),
                }
                for row in depth_df.itertuples(index=False)
            }
        ),
    }


def _run_classical_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, Any]:
    search = _build_classical_search(model_name)
    search.fit(X_train, y_train)
    y_pred = float(search.predict(X_test)[0])

    return {
        "model": model_name,
        "y_pred": y_pred,
        "inner_mae": float(-search.best_score_),
        "selected_depth": np.nan,
        "selected_ridge_alpha": np.nan,
        "selected_params": _json_dumps(search.best_params_),
        "search_trace": "",
    }


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        predictions.groupby("model", as_index=False)["abs_error"]
        .agg(mae_mean="mean", mae_std="std", n_folds="count")
        .sort_values(["mae_mean", "model"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary["mae_pct_mean"] = summary["mae_mean"] * 100.0
    summary["mae_pct_std"] = summary["mae_std"] * 100.0
    summary["mae_ah_mean"] = summary["mae_mean"] * WARWICK_NOMINAL_CAPACITY_AH
    summary["mae_ah_std"] = summary["mae_std"] * WARWICK_NOMINAL_CAPACITY_AH
    return summary


def run_nested_warwick_loco() -> tuple[pd.DataFrame, pd.DataFrame]:
    X_raw, y, cell_ids = get_warwick_arrays()
    predictions: list[dict[str, Any]] = []

    print(f"Loaded Warwick data: {len(cell_ids)} cells, X shape={X_raw.shape}")
    print(f"Using QRC backend: {'qiskit-statevector' if QISKIT_AVAILABLE else 'classical-fallback'}")
    print(f"Running outer LOCO folds: {len(cell_ids)}")

    for outer_fold, test_cell in enumerate(cell_ids, start=1):
        test_mask = np.array([cell_id == test_cell for cell_id in cell_ids], dtype=bool)
        train_mask = ~test_mask

        X_train_raw = X_raw[train_mask]
        y_train = y[train_mask]
        X_test_raw = X_raw[test_mask]
        y_test = y[test_mask]

        X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)
        y_true = float(y_test[0])

        print(f"[Outer {outer_fold:02d}/{len(cell_ids)}] hold out {test_cell}")

        fold_results = [_run_qrc_depth_search(X_train_pca, y_train, X_test_pca)]
        for model_name in CLASSICAL_MODELS:
            fold_results.append(_run_classical_search(model_name, X_train_pca, y_train, X_test_pca))

        for result in fold_results:
            abs_error = abs(y_true - result["y_pred"])
            predictions.append(
                {
                    "outer_fold": outer_fold,
                    "test_cell": test_cell,
                    "model": result["model"],
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "pca_components": int(X_train_pca.shape[1]),
                    "y_true": y_true,
                    "y_pred": float(result["y_pred"]),
                    "abs_error": float(abs_error),
                    "abs_error_pct": float(abs_error * 100.0),
                    "abs_error_ah": float(abs_error * WARWICK_NOMINAL_CAPACITY_AH),
                    "inner_mae": float(result["inner_mae"]),
                    "selected_depth": result["selected_depth"],
                    "selected_ridge_alpha": result["selected_ridge_alpha"],
                    "selected_params": result["selected_params"],
                    "search_trace": result["search_trace"],
                }
            )

        fold_df = pd.DataFrame(predictions)
        latest = (
            fold_df[fold_df["outer_fold"] == outer_fold]
            .sort_values(["abs_error", "model"])
            [["model", "abs_error", "inner_mae", "selected_params"]]
        )
        print(latest.to_string(index=False))

    predictions_df = pd.DataFrame(predictions).sort_values(
        ["outer_fold", "model"], ascending=[True, True]
    )
    summary_df = _summarize_predictions(predictions_df)
    return predictions_df, summary_df


def main() -> None:
    data_dir = _get_stage_data_dir()
    predictions_path = data_dir / "nested_warwick_loco_predictions.csv"
    summary_path = data_dir / "nested_warwick_loco_summary.csv"

    print("=" * 72)
    print("Phase 8 Stage 2: Nested Warwick LOCO CV")
    print("=" * 72)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output dir: {data_dir}")
    print(f"Classical models: {CLASSICAL_MODELS}")
    print(f"QRC depths: {QRC_DEPTHS} | observable={QRC_OBSERVABLE}")

    predictions_df, summary_df = run_nested_warwick_loco()

    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nNested Warwick LOCO summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {summary_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
