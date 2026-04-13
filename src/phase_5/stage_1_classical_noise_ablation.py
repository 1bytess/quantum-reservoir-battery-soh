"""Phase 5 Stage 1: Classical Noise Ablation"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Fix sys.path for imports from src/
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CELL_IDS, MODEL_NAMES, N_PCA, PROJECT_ROOT, RANDOM_STATE
from data_loader import load_stanford_data
from phase_5.config import get_stage_paths

# Reuse existing classical models from src/phase_3.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase_3.models import get_model_pipeline


NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]


class TeeLogger:
    """Tee stdout to console and a log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def _save_figure(fig: plt.Figure, plot_dir, stem: str) -> None:
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
    return pca.fit_transform(X_train), pca.transform(X_test)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def _run_noise_ablation(cell_data: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: List[Dict] = []

    for noise_level in NOISE_LEVELS:
        print(f"\nNoise level: {noise_level}")

        for fold_idx, test_cell in enumerate(CELL_IDS):
            train_cells = [c for c in CELL_IDS if c != test_cell]
            X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
            y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
            X_test_raw = cell_data[test_cell]["X_raw"]
            y_test = cell_data[test_cell]["y"]

            seed = RANDOM_STATE + int(round(noise_level * 1_000_000)) + fold_idx * 10_000
            rng = np.random.RandomState(seed)
            if noise_level > 0.0:
                X_train_noisy = X_train_raw + rng.normal(0.0, noise_level, size=X_train_raw.shape)
                X_test_noisy = X_test_raw + rng.normal(0.0, noise_level, size=X_test_raw.shape)
            else:
                X_train_noisy = X_train_raw.copy()
                X_test_noisy = X_test_raw.copy()

            # Noise is injected before PCA. PCA is fit on train fold only.
            X_train_eval, X_test_eval = _fit_pca_in_fold(X_train_noisy, X_test_noisy)
            naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

            for model_name in MODEL_NAMES:
                try:
                    model = get_model_pipeline(model_name)
                    model.fit(X_train_eval, y_train)
                    y_pred = model.predict(X_test_eval)
                    m = _metrics(y_test, y_pred)
                    rows.append(
                        {
                            "noise_level": noise_level,
                            "model": model_name,
                            "test_cell": test_cell,
                            "n_train": int(len(y_train)),
                            "n_test": int(len(y_test)),
                            "mae": m["mae"],
                            "rmse": m["rmse"],
                            "r2": m["r2"],
                            "naive_mae": float(naive_mae),
                            "beats_naive": bool(m["mae"] < naive_mae),
                        }
                    )
                except Exception as exc:
                    print(
                        f"  Model failed: model={model_name}, "
                        f"cell={test_cell}, noise={noise_level}, "
                        f"error={type(exc).__name__}: {exc}"
                    )

        level_rows = [r for r in rows if np.isclose(r["noise_level"], noise_level)]
        if level_rows:
            tmp_df = pd.DataFrame(level_rows)
            level_summary = tmp_df.groupby("model")["mae"].mean().sort_values()
            print("  Mean MAE by model:")
            for model_name, mae in level_summary.items():
                print(f"    {model_name}: {mae:.4f}")

    return pd.DataFrame(rows)


def _plot_noise_sweep(results_df: pd.DataFrame, plot_dir) -> None:
    if results_df.empty:
        print("Skipping noise_sweep plot: no data")
        return

    summary = (
        results_df.groupby(["model", "noise_level"])["mae"]
        .mean()
        .reset_index()
        .sort_values(["model", "noise_level"])
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model_name]
        ax.plot(sub["noise_level"], sub["mae"], marker="o", linewidth=1.8, label=model_name)

    ax.set_xlabel("Noise level (Gaussian std)")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("Classical Model Noise Sensitivity")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2)
    _save_figure(fig, plot_dir, "noise_sweep")


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_1")
    log_path = data_dir / "stage_1_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 5: Noise Sensitivity Ablation")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")
        print(f"Noise levels: {NOISE_LEVELS}")

        cell_data = load_stanford_data()
        results_df = _run_noise_ablation(cell_data)

        out_path = data_dir / "noise_ablation.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(results_df)} rows)")

        _plot_noise_sweep(results_df, plot_dir)

        if not results_df.empty:
            summary = (
                results_df.groupby(["model", "noise_level"])["mae"]
                .mean()
                .reset_index()
                .sort_values(["noise_level", "mae"])
            )
            print("\nBest model by noise level:")
            for nl in NOISE_LEVELS:
                sub = summary[np.isclose(summary["noise_level"], nl)]
                if sub.empty:
                    continue
                best = sub.iloc[0]
                print(f"  noise={nl:>6}: {best['model']} (MAE={best['mae']:.4f})")

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
