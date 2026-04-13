"""
MLP evaluation under the same nested LOCO protocol as QRC (Phase 5 Stage 4).

Runs on both Stanford (6-cell nested LOCO) and Warwick (24-cell LOCO).
MLP hyperparameters are selected via inner LOCO CV on Stanford.
On Warwick, MLP uses GridSearchCV within each fold (matching phase_8 protocol).

Output: src/manuscript_support/data/mlp_nested_loco_stanford.csv
        src/manuscript_support/data/mlp_loco_warwick.csv
        src/manuscript_support/data/mlp_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from config import CELL_IDS, N_PCA, RANDOM_STATE
from data_loader import load_stanford_data
from data_loader_warwick import load_warwick_data
from phase_3.models import get_model_pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# MLP hyperparameter grid (from phase_3/config.py)
MLP_PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16)],
    "mlp__alpha": [0.001, 0.01, 0.1, 1.0],
    "mlp__learning_rate_init": [1e-3, 5e-4],
}

# Run multiple seeds to check stability
SEEDS = [42, 43, 44, 45, 46]


def _fit_pca_in_fold(
    X_train_raw: np.ndarray, X_test_raw: np.ndarray, n_components: int = N_PCA
):
    """PCA fit on train only, transform both."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)
    return X_train, X_test


def _stack_cells(cell_data, cells):
    """Stack multiple cells into arrays."""
    X = np.vstack([cell_data[c]["X_raw"] for c in cells])
    y = np.concatenate([cell_data[c]["y"] for c in cells])
    return X, y


# ------------------------------------------------------------------ #
#  Stanford: Nested LOCO (matching stage_4 protocol)                  #
# ------------------------------------------------------------------ #

def run_stanford_nested_loco():
    """
    Nested LOCO on Stanford (6 cells):
    - Outer loop: hold out 1 cell for test
    - Inner loop: hold out 1 of remaining 5 for validation
      -> GridSearchCV on MLP hyperparams across inner folds
    - Retrain with best hyperparams on outer 5 cells, evaluate on test cell
    """
    print("=" * 60)
    print("STANFORD: Nested LOCO for MLP")
    print("=" * 60)

    cell_data = load_stanford_data()
    fold_rows = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for outer_idx, test_cell in enumerate(CELL_IDS, 1):
            outer_train_cells = [c for c in CELL_IDS if c != test_cell]

            # Inner loop: select best MLP hyperparams
            inner_results = []
            for val_cell in outer_train_cells:
                inner_train_cells = [c for c in outer_train_cells if c != val_cell]
                X_train_raw, y_train = _stack_cells(cell_data, inner_train_cells)
                X_val_raw = cell_data[val_cell]["X_raw"]
                y_val = cell_data[val_cell]["y"]

                X_train, X_val = _fit_pca_in_fold(X_train_raw, X_val_raw)

                # Evaluate each hyperparameter combo
                for hidden in MLP_PARAM_GRID["mlp__hidden_layer_sizes"]:
                    for alpha in MLP_PARAM_GRID["mlp__alpha"]:
                        for lr in MLP_PARAM_GRID["mlp__learning_rate_init"]:
                            model = get_model_pipeline("mlp")
                            # Override hyperparams
                            model.named_steps["mlp"].set_params(
                                hidden_layer_sizes=hidden,
                                alpha=alpha,
                                learning_rate_init=lr,
                                random_state=seed,
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            val_mae = float(mean_absolute_error(y_val, y_pred))
                            inner_results.append({
                                "val_cell": val_cell,
                                "hidden": str(hidden),
                                "alpha": alpha,
                                "lr": lr,
                                "val_mae": val_mae,
                            })

            inner_df = pd.DataFrame(inner_results)
            # Select best hyperparams by mean inner val MAE
            summary = (
                inner_df.groupby(["hidden", "alpha", "lr"], as_index=False)["val_mae"]
                .mean()
                .sort_values("val_mae")
            )
            best = summary.iloc[0]
            best_hidden = eval(best["hidden"])  # str -> tuple
            best_alpha = best["alpha"]
            best_lr = best["lr"]
            best_inner_mae = best["val_mae"]

            print(
                f"  Fold {outer_idx}/6 test={test_cell} | "
                f"Best: hidden={best_hidden}, alpha={best_alpha}, lr={best_lr}, "
                f"inner_MAE={best_inner_mae:.4f}"
            )

            # Outer evaluation with selected hyperparams
            X_outer_train_raw, y_outer_train = _stack_cells(cell_data, outer_train_cells)
            X_outer_test_raw = cell_data[test_cell]["X_raw"]
            y_outer_test = cell_data[test_cell]["y"]

            X_outer_train, X_outer_test = _fit_pca_in_fold(
                X_outer_train_raw, X_outer_test_raw
            )

            model = get_model_pipeline("mlp")
            model.named_steps["mlp"].set_params(
                hidden_layer_sizes=best_hidden,
                alpha=best_alpha,
                learning_rate_init=best_lr,
                random_state=seed,
            )
            model.fit(X_outer_train, y_outer_train)
            y_pred = model.predict(X_outer_test)
            outer_mae = float(mean_absolute_error(y_outer_test, y_pred))

            fold_rows.append({
                "seed": seed,
                "test_cell": test_cell,
                "selected_hidden": str(best_hidden),
                "selected_alpha": best_alpha,
                "selected_lr": best_lr,
                "mean_inner_mae": best_inner_mae,
                "outer_mae": outer_mae,
                "outer_mae_pct": outer_mae * 100,
                "n_test": len(y_outer_test),
            })

    df = pd.DataFrame(fold_rows)
    out_path = OUTPUT_DIR / "mlp_nested_loco_stanford.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary per seed
    seed_summary = df.groupby("seed").agg(
        mean_mae=("outer_mae", "mean"),
        mean_mae_pct=("outer_mae_pct", "mean"),
        std_mae_pct=("outer_mae_pct", "std"),
    ).reset_index()
    print("\nPer-seed summary:")
    print(seed_summary.to_string(index=False))

    # Overall
    overall_mae = df["outer_mae"].mean()
    overall_mae_pct = df["outer_mae_pct"].mean()
    overall_std = df.groupby("seed")["outer_mae_pct"].mean().std()
    print(f"\nOverall nested LOCO MAE: {overall_mae_pct:.3f}% (+/- {overall_std:.3f}% across seeds)")

    return df


# ------------------------------------------------------------------ #
#  Warwick: Standard LOCO (matching phase_8 protocol)                 #
# ------------------------------------------------------------------ #

def run_warwick_loco():
    """
    Standard LOCO on Warwick (24 cells, 1 sample each).
    MLP with leave-one-out inner CV within each LOCO training set.
    """
    print("\n" + "=" * 60)
    print("WARWICK: LOCO for MLP")
    print("=" * 60)

    cell_data = load_warwick_data()
    cell_ids = sorted(cell_data.keys())
    print(f"Loaded {len(cell_ids)} Warwick cells")

    fold_rows = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for idx, test_cell in enumerate(cell_ids, 1):
            train_cells = [c for c in cell_ids if c != test_cell]

            X_train_raw, y_train = _stack_cells(cell_data, train_cells)
            X_test_raw = cell_data[test_cell]["X_raw"]
            y_test = cell_data[test_cell]["y"]

            X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)

            # GridSearchCV within training fold
            model = get_model_pipeline("mlp")
            model.named_steps["mlp"].set_params(random_state=seed)

            grid = GridSearchCV(
                model,
                MLP_PARAM_GRID,
                cv=LeaveOneOut(),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred))

            fold_rows.append({
                "seed": seed,
                "test_cell": test_cell,
                "mae": mae,
                "mae_pct": mae * 100,
                "y_true": float(y_test[0]),
                "y_pred": float(y_pred[0]),
                "best_params": str(grid.best_params_),
            })

            if idx % 6 == 0:
                print(f"  Completed {idx}/{len(cell_ids)} folds")

    df = pd.DataFrame(fold_rows)
    out_path = OUTPUT_DIR / "mlp_loco_warwick.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary per seed
    seed_summary = df.groupby("seed").agg(
        mean_mae_pct=("mae_pct", "mean"),
        std_mae_pct=("mae_pct", "std"),
    ).reset_index()
    print("\nPer-seed summary:")
    print(seed_summary.to_string(index=False))

    overall_mae_pct = df.groupby("seed")["mae_pct"].mean().mean()
    overall_std = df.groupby("seed")["mae_pct"].mean().std()
    print(f"\nOverall LOCO MAE: {overall_mae_pct:.3f}% (+/- {overall_std:.3f}% across seeds)")

    return df


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    stanford_df = run_stanford_nested_loco()
    warwick_df = run_warwick_loco()

    # Combined summary
    stanford_mean = stanford_df.groupby("seed")["outer_mae_pct"].mean()
    warwick_mean = warwick_df.groupby("seed")["mae_pct"].mean()

    summary = pd.DataFrame({
        "dataset": ["stanford_nested_loco", "stanford_nested_loco",
                     "warwick_loco", "warwick_loco"],
        "metric": ["mean_mae_pct", "std_across_seeds",
                    "mean_mae_pct", "std_across_seeds"],
        "value": [
            stanford_mean.mean(),
            stanford_mean.std(),
            warwick_mean.mean(),
            warwick_mean.std(),
        ],
    })

    # Add comparison row with existing QRC/XGBoost numbers
    comparison = pd.DataFrame([
        {"dataset": "stanford_nested_loco", "metric": "qrc_mae_pct", "value": 0.85},
        {"dataset": "stanford_nested_loco", "metric": "xgboost_mae_pct", "value": 1.09},
        {"dataset": "warwick_loco", "metric": "qrc_mae_pct", "value": 0.93},
        {"dataset": "warwick_loco", "metric": "xgboost_mae_pct", "value": 1.51},
    ])
    summary = pd.concat([summary, comparison], ignore_index=True)

    out_path = OUTPUT_DIR / "mlp_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
