"""Phase 12 Stage 3: Nested LOCO ridge baseline on fitted ECM parameters."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_12.config import ECM_CANDIDATE_MODELS, ECM_PARAMETER_FEATURES, RIDGE_ALPHAS, get_stage_paths


class TeeLogger:
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
        fig.savefig(plot_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / stem}.png")


def _select_alpha_nested_loo(X_train: np.ndarray, y_train: np.ndarray) -> tuple[float, float]:
    loo = LeaveOneOut()
    best_alpha = RIDGE_ALPHAS[0]
    best_mae = float("inf")

    for alpha in RIDGE_ALPHAS:
        fold_errors = []
        for inner_train_idx, inner_val_idx in loo.split(X_train):
            scaler = StandardScaler()
            X_inner_train = scaler.fit_transform(X_train[inner_train_idx])
            X_inner_val = scaler.transform(X_train[inner_val_idx])

            model = Ridge(alpha=alpha)
            model.fit(X_inner_train, y_train[inner_train_idx])
            pred = model.predict(X_inner_val)
            fold_errors.append(abs(float(pred[0]) - float(y_train[inner_val_idx][0])))

        mean_mae = float(np.mean(fold_errors))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_alpha = alpha

    return float(best_alpha), float(best_mae)


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_3")
    logger = TeeLogger(data_dir / "stage_3_log.txt")
    old_stdout = sys.stdout
    sys.stdout = logger
    try:
        print(f"Started: {datetime.now().isoformat()}")
        feature_path = Path(data_dir.parent.parent / "stage_2" / "data" / "warwick_ecm_selected_parameters.csv")
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing fitted-parameter table: {feature_path}\n"
                "Run stage 2 first: python -m src.phase_12.run_phase_12 --stages 2"
            )

        df = pd.read_csv(feature_path).sort_values("cell_id").reset_index(drop=True)
        model_dummies = pd.get_dummies(df["selected_model"], prefix="model")
        for model_name in ECM_CANDIDATE_MODELS:
            col = f"model_{model_name}"
            if col not in model_dummies.columns:
                model_dummies[col] = 0
        model_dummies = model_dummies[[f"model_{name}" for name in ECM_CANDIDATE_MODELS]]

        feature_df = pd.concat([df[ECM_PARAMETER_FEATURES].fillna(0.0), model_dummies], axis=1)
        feature_cols = feature_df.columns.tolist()

        X = feature_df.to_numpy(dtype=float)
        y = df["soh_frac"].to_numpy(dtype=float)
        cell_ids = df["cell_id"].tolist()

        outer_loo = LeaveOneOut()
        rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_loo.split(X), start=1):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            best_alpha, inner_mae = _select_alpha_nested_loo(X_train, y_train)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=best_alpha)
            model.fit(X_train_scaled, y_train)
            y_pred = float(model.predict(X_test_scaled)[0])
            abs_error_pct = abs(y_pred - float(y_test[0])) * 100.0

            rows.append(
                {
                    "outer_fold": fold_idx,
                    "test_cell": cell_ids[test_idx[0]],
                    "y_true_pct": float(y_test[0] * 100.0),
                    "y_pred_pct": float(y_pred * 100.0),
                    "abs_error_pct": abs_error_pct,
                    "selected_alpha": best_alpha,
                    "inner_cv_mae_pct": inner_mae * 100.0,
                }
            )

        pred_df = pd.DataFrame(rows).sort_values("outer_fold").reset_index(drop=True)
        pred_df.to_csv(data_dir / "warwick_ecm_parameter_loco.csv", index=False)

        mae_pct = float(pred_df["abs_error_pct"].mean())
        rmse_pct = float(np.sqrt(np.mean((pred_df["y_pred_pct"] - pred_df["y_true_pct"]) ** 2)))
        r2 = float(r2_score(pred_df["y_true_pct"], pred_df["y_pred_pct"]))
        summary = pd.DataFrame(
            [
                {
                    "model": "ecm_parameter_ridge_nested_loo",
                    "n_cells": int(len(pred_df)),
                    "mean_mae_pct": mae_pct,
                    "std_abs_error_pct": float(pred_df["abs_error_pct"].std(ddof=1)),
                    "rmse_pct": rmse_pct,
                    "r2": r2,
                    "median_selected_alpha": float(pred_df["selected_alpha"].median()),
                    "n_features": int(len(feature_cols)),
                }
            ]
        )
        summary.to_csv(data_dir / "warwick_ecm_parameter_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(pred_df["y_true_pct"], pred_df["y_pred_pct"], s=45, alpha=0.8)
        bounds = [
            min(pred_df["y_true_pct"].min(), pred_df["y_pred_pct"].min()),
            max(pred_df["y_true_pct"].max(), pred_df["y_pred_pct"].max()),
        ]
        ax.plot(bounds, bounds, linestyle="--", linewidth=1.0, color="black")
        ax.set_xlabel("True SOH [%]")
        ax.set_ylabel("Predicted SOH [%]")
        ax.set_title("Warwick fitted-ECM LOCO predictions")
        ax.grid(True, alpha=0.3)
        _save_figure(fig, plot_dir, "warwick_ecm_parameter_predictions")

        print(f"Completed nested LOCO across {len(pred_df)} Warwick cells.")
        print(f"Mean MAE: {mae_pct:.3f}%")
        print(f"RMSE: {rmse_pct:.3f}%")
        print(f"R2: {r2:.3f}")
    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    main()
