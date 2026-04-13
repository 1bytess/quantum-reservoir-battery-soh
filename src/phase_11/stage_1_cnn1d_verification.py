"""Phase 11 Stage 1: CNN1D Baseline Verification.

Reviewer concern:
    "CNN1D is actually SVR fallback — not a real DL baseline. Reviewers who
    check will notice."

This stage:
  1. Detects whether PyTorch is installed (TORCH_AVAILABLE flag).
  2. Runs CNN1D from scratch under LOCO-CV on the Stanford dataset,
     explicitly reporting which backend (PyTorch or SVR fallback) was used
     for each fold.
  3. Compares CNN1D results against the stored phase_3 LOCO results to
     confirm whether previously reported CNN1D numbers used real PyTorch.
  4. If PyTorch IS available, re-reports the true DL baseline MAE and
     whether QRC still beats it.
  5. If PyTorch is NOT available, flags this clearly for the user to
     install PyTorch before submission.

Outputs (result/phase_11/stage_1/):
  data/cnn1d_verification_results.csv   — per-fold CNN1D MAE + backend flag
  data/cnn1d_verification_summary.csv   — macro/micro MAE, backend used
  data/cnn1d_vs_qrc_comparison.csv      — CNN1D vs QRC head-to-head
  data/stage_1_log.txt                  — full run log
  plot/cnn1d_vs_baselines.png/pdf       — bar chart CNN1D vs key baselines
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CELL_IDS, N_PCA, PROJECT_ROOT, RANDOM_STATE, get_result_paths
from data_loader import load_stanford_data
from phase_3.models import CNN1DRegressor, get_model_pipeline
from phase_11.config import get_stage_paths, STANFORD_NOMINAL_AH

# Detect PyTorch availability
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


def _fit_pca_in_fold(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_components = min(N_PCA, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_train), pca.transform(X_test)


def _run_cnn1d_loco(
    cell_data: Dict[str, dict],
) -> pd.DataFrame:
    """Run CNN1D under LOCO-CV; record per-fold backend and MAE."""
    rows: List[dict] = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]
        X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_raw = cell_data[test_cell]["X_raw"]
        y_test = cell_data[test_cell]["y"]

        X_train_pca, X_test_pca = _fit_pca_in_fold(X_train_raw, X_test_raw)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = get_model_pipeline("cnn1d")
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)

        # Check if fallback was triggered
        cnn1d_step = model.named_steps.get("cnn1d")
        used_fallback = hasattr(cnn1d_step, "fallback_")
        backend = "SVR_fallback" if used_fallback else "PyTorch_CNN1D"
        fallback_warned = any(
            issubclass(w.category, RuntimeWarning)
            and "falls back to SVR" in str(w.message)
            for w in caught
        )

        mae = float(np.mean(np.abs(y_test - y_pred)))
        mae_pct = mae * 100.0
        mae_ah = mae * STANFORD_NOMINAL_AH

        rows.append({
            "test_cell": test_cell,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "mae": mae,
            "mae_pct": mae_pct,
            "mae_ah": mae_ah,
            "backend": backend,
            "torch_available": TORCH_AVAILABLE,
            "fallback_warned": fallback_warned,
        })
        print(
            f"  {test_cell}: MAE={mae:.4f} ({mae_pct:.3f}%) "
            f"[backend={backend}]"
        )

    return pd.DataFrame(rows)


def _load_phase3_cnn1d(phase3_data_dir: Path) -> pd.DataFrame | None:
    loco_csv = phase3_data_dir / "loco_results.csv"
    if not loco_csv.exists():
        return None
    df = pd.read_csv(loco_csv)
    mask = (
        (df.get("regime", pd.Series(dtype=str)) == "loco")
        & (df.get("model", pd.Series(dtype=str)) == "cnn1d")
        & (df.get("feature_space", pd.Series(dtype=str)) == "pca6")
    )
    sub = df[mask]
    return sub if not sub.empty else None


def _load_phase3_qrc(phase4_data_dir: Path) -> pd.DataFrame | None:
    qrc_csv = phase4_data_dir / "qrc_noiseless.csv"
    if not qrc_csv.exists():
        return None
    df = pd.read_csv(qrc_csv)
    mask = (
        (df.get("stage", pd.Series(dtype=str)) == "noiseless")
        & (df.get("regime", pd.Series(dtype=str)) == "loco")
        & (pd.to_numeric(df.get("depth", pd.Series(dtype=float)), errors="coerce") == 1)
    )
    sub = df[mask][["test_cell", "mae"]].drop_duplicates(subset=["test_cell"])
    return sub if not sub.empty else None


def _plot_cnn1d_vs_baselines(
    cnn1d_mae: float,
    baseline_mae: dict,
    backend: str,
    plot_dir: Path,
) -> None:
    labels = list(baseline_mae.keys()) + [f"CNN1D\n({backend})"]
    values = list(baseline_mae.values()) + [cnn1d_mae]

    colors = []
    for lbl in labels:
        if "CNN1D" in lbl:
            colors.append("tab:pink")
        elif "QRC" in lbl:
            colors.append("tab:blue")
        else:
            colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Macro MAE (LOCO-CV, Stanford)")
    ax.set_title(
        f"CNN1D Baseline Verification\n"
        f"Backend: {backend} | PyTorch available: {TORCH_AVAILABLE}"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    _save_figure(fig, plot_dir, "cnn1d_vs_baselines")


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_1")
    log_path = data_dir / "stage_1_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 11 Stage 1: CNN1D Baseline Verification")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"PyTorch available: {TORCH_AVAILABLE}")
        if not TORCH_AVAILABLE:
            print(
                "\n*** WARNING: PyTorch NOT installed. ***\n"
                "CNN1D will use SVR fallback — this means previously reported\n"
                "CNN1D results may be SVR, not a real deep-learning baseline.\n"
                "Install PyTorch before submission:\n"
                "  conda install pytorch torchvision -c pytorch\n"
                "or: pip install torch\n"
                "Then re-run this stage to get true CNN1D numbers.\n"
            )

        print("\nLoading Stanford data...")
        cell_data = load_stanford_data()
        for cid, d in cell_data.items():
            print(f"  {cid}: {d['X_raw'].shape[0]} blocks, {d['X_raw'].shape[1]} raw features")

        print("\nRunning CNN1D LOCO-CV...")
        cnn1d_results = _run_cnn1d_loco(cell_data)

        # Macro MAE across folds
        macro_mae = float(cnn1d_results["mae"].mean())
        macro_mae_pct = macro_mae * 100.0
        backend_used = cnn1d_results["backend"].iloc[0]
        all_same_backend = (cnn1d_results["backend"] == backend_used).all()

        print(f"\nCNN1D LOCO Macro MAE = {macro_mae:.6f} ({macro_mae_pct:.4f}%)")
        print(f"Backend used: {backend_used} (consistent={all_same_backend})")

        cnn1d_results.to_csv(data_dir / "cnn1d_verification_results.csv", index=False)
        print(f"Saved: {data_dir / 'cnn1d_verification_results.csv'}")

        # Load phase_3 reference results for comparison
        phase3_data_dir, _ = get_result_paths(3)
        phase4_data_dir, _ = get_result_paths(4)

        phase3_cnn1d = _load_phase3_cnn1d(phase3_data_dir)
        phase3_qrc = _load_phase3_qrc(phase4_data_dir)

        # Compare with stored phase_3 CNN1D results
        prev_cnn1d_macro: float | None = None
        if phase3_cnn1d is not None:
            prev_cnn1d_macro = float(phase3_cnn1d["mae"].mean())
            print(f"\nStored phase_3 CNN1D macro MAE = {prev_cnn1d_macro:.6f}")
            mae_diff = abs(macro_mae - prev_cnn1d_macro)
            print(f"Difference vs fresh re-run: {mae_diff:.6f}")
            if mae_diff < 1e-5:
                print("[MATCH] Results are identical — same backend confirmed.")
            else:
                print("[DIVERGE] Results differ — backend may have changed.")
        else:
            print("\nNo stored phase_3 CNN1D results found (skipping comparison).")

        # Load QRC for head-to-head
        qrc_macro: float | None = None
        if phase3_qrc is not None:
            qrc_macro = float(phase3_qrc["mae"].mean())
            print(f"\nQRC (depth=1, noiseless) LOCO macro MAE = {qrc_macro:.6f}")
            if qrc_macro < macro_mae:
                pct_improvement = (macro_mae - qrc_macro) / macro_mae * 100
                print(
                    f"[QRC WINS] QRC is {pct_improvement:.1f}% better than CNN1D "
                    f"({backend_used}) — strong result for submission."
                )
            else:
                pct_gap = (qrc_macro - macro_mae) / macro_mae * 100
                print(
                    f"[CNN1D WINS] CNN1D ({backend_used}) is {pct_gap:.1f}% better "
                    f"than QRC — this weakens the baseline comparison narrative."
                )

        # Build comparison table
        cmp_rows = []
        cmp_rows.append({
            "model": f"cnn1d_{backend_used}",
            "macro_mae": macro_mae,
            "macro_mae_pct": macro_mae_pct,
            "backend": backend_used,
            "torch_available": TORCH_AVAILABLE,
        })
        if prev_cnn1d_macro is not None:
            cmp_rows.append({
                "model": "cnn1d_phase3_stored",
                "macro_mae": prev_cnn1d_macro,
                "macro_mae_pct": prev_cnn1d_macro * 100.0,
                "backend": "stored_phase3",
                "torch_available": "unknown",
            })
        if qrc_macro is not None:
            cmp_rows.append({
                "model": "qrc_noiseless_d1",
                "macro_mae": qrc_macro,
                "macro_mae_pct": qrc_macro * 100.0,
                "backend": "qiskit_statevector",
                "torch_available": "N/A",
            })

        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df.to_csv(data_dir / "cnn1d_vs_qrc_comparison.csv", index=False)
        print(f"\nSaved: {data_dir / 'cnn1d_vs_qrc_comparison.csv'}")

        # Summary record
        summary = pd.DataFrame([{
            "torch_available": TORCH_AVAILABLE,
            "backend_used": backend_used,
            "all_folds_same_backend": bool(all_same_backend),
            "cnn1d_macro_mae": macro_mae,
            "cnn1d_macro_mae_pct": macro_mae_pct,
            "phase3_stored_macro_mae": prev_cnn1d_macro if prev_cnn1d_macro is not None else float("nan"),
            "qrc_d1_macro_mae": qrc_macro if qrc_macro is not None else float("nan"),
            "qrc_beats_cnn1d": bool(qrc_macro is not None and qrc_macro < macro_mae),
            "reviewer_action_needed": not TORCH_AVAILABLE,
        }])
        summary.to_csv(data_dir / "cnn1d_verification_summary.csv", index=False)
        print(f"Saved: {data_dir / 'cnn1d_verification_summary.csv'}")

        # Plot
        baseline_mae_map: dict = {}
        loco_csv = phase3_data_dir / "loco_results.csv"
        if loco_csv.exists():
            loco_df = pd.read_csv(loco_csv)
            for model_name, label in [
                ("xgboost", "XGBoost"), ("ridge", "Ridge"), ("svr", "SVR"),
                ("gp", "GP"), ("esn", "ESN"),
            ]:
                sub = loco_df[
                    (loco_df.get("regime", "") == "loco")
                    & (loco_df.get("model", "") == model_name)
                    & (loco_df.get("feature_space", "") == "pca6")
                ]
                if not sub.empty:
                    baseline_mae_map[label] = float(sub["mae"].mean())
        if qrc_macro is not None:
            baseline_mae_map["QRC (d=1)"] = qrc_macro

        _plot_cnn1d_vs_baselines(macro_mae, baseline_mae_map, backend_used, plot_dir)

        print("\n" + "=" * 70)
        print("REVIEWER ACTION ITEMS FROM STAGE 1:")
        print("=" * 70)
        if not TORCH_AVAILABLE:
            print("  ❌ PyTorch NOT installed — CNN1D used SVR fallback.")
            print("     Install PyTorch and re-run before submission.")
            print("     Command: conda install pytorch -c pytorch")
        else:
            print("  ✅ PyTorch available — CNN1D used real deep-learning backend.")
            if qrc_macro is not None and qrc_macro < macro_mae:
                print(
                    f"  ✅ QRC ({qrc_macro:.4f}) beats PyTorch CNN1D ({macro_mae:.4f})."
                )
                print("     This is a strong result. Report CNN1D as real DL baseline.")
            else:
                print(
                    f"  ⚠  CNN1D ({macro_mae:.4f}) beats or ties QRC ({qrc_macro})."
                )
                print("     Consider whether to include CNN1D as primary or supplementary.")

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
