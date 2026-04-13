"""Phase 5 Stage 3: Extended QRC Analysis"""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Required pattern: ensure local __TEMP imports resolve first.
# Fix sys.path for imports from src
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    CELL_IDS,
    N_PCA,
    RANDOM_STATE,
    PROJECT_ROOT,
    DEPTH_RANGE,
    MARRAKESH_NOISE,
)
from data_loader import load_stanford_data
from phase_5.config import get_stage_paths

# Required pattern: import project src modules.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase_4.qrc_model import QuantumReservoir
from phase_4.stage2_noisy import NoisyQuantumReservoir
from phase_3.models import get_model_pipeline

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


LEARNING_CURVE_CELL_COUNTS = [1, 2, 3, 4, 5]
NOISE_SHOTS_SWEEP = [256, 512, 1024, 2048, 4096, 8192, 16384]

QRC_LOCO_COLUMNS = [
    "stage",
    "model",
    "depth",
    "shots",
    "observable_config",
    "test_cell",
    "train_cells",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "r2",
    "naive_mae",
    "beats_baseline",
    "reservoir_dim",
]

LEARNING_CURVE_COLUMNS = [
    "stage",
    "model",
    "n_train_cells",
    "train_cells",
    "test_cell",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "r2",
    "naive_mae",
]

HEAD_TO_HEAD_COLUMNS = [
    "stage",
    "model",
    "depth",
    "test_cell",
    "train_cells",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "r2",
    "naive_mae",
]

MODEL_DISPLAY_NAMES = {
    "qrc_d1_z_plus_zz": "QRC (d=1, 6-qubit)",
    "xgboost_pca6": "XGBoost",
    "esn_pca6": "ESN (classical RC)",
    "rff_pca6": "RFF (random features)",
}

# Colour palette: QRC = blue, ESN = orange, XGBoost = green, RFF = purple
MODEL_COLORS = {
    "qrc_d1_z_plus_zz": "#0077BB",
    "esn_pca6":          "#EE7733",
    "xgboost_pca6":      "#009988",
    "rff_pca6":          "#AA3377",
}

MODEL_LINESTYLES = {
    "qrc_d1_z_plus_zz": "-",
    "esn_pca6":          "--",
    "xgboost_pca6":      "-.",
    "rff_pca6":          ":",
}

# Order in which to plot (QRC first so it's on top)
MODEL_PLOT_ORDER = ["qrc_d1_z_plus_zz", "esn_pca6", "xgboost_pca6", "rff_pca6"]


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
    return pca.fit_transform(X_train), pca.transform(X_test)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def get_marrakesh_noise_model() -> NoiseModel:
    """Synthetic noise model matching IBM Marrakesh (Heron r2, 156Q).

    Median values extracted from FakeMarrakesh (qiskit-ibm-runtime):
      - SX error:      0.023% depolarizing
      - CZ error:      0.33%  depolarizing
      - Readout error: 0.95%  bitflip
      - T1 ~ 197 us, T2 ~ 118 us (approximated as depolarizing)
    """
    noise_model = NoiseModel()

    sq_error = depolarizing_error(MARRAKESH_NOISE["single_qubit_error"], 1)
    noise_model.add_all_qubit_quantum_error(
        sq_error, ["rx", "ry", "rz", "x", "y", "z", "h", "sx"]
    )

    tq_error = depolarizing_error(MARRAKESH_NOISE["two_qubit_error"], 2)
    noise_model.add_all_qubit_quantum_error(tq_error, ["cx", "cz", "ecr"])

    p_meas = MARRAKESH_NOISE["measurement_error"]
    readout_error = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def _build_qrc_model(
    *,
    noisy: bool,
    depth: int,
    noise_model: Optional[NoiseModel] = None,
    shots: Optional[int] = None,
    use_zz: bool = True,
    observable_set: str = "Z",
):
    if noisy:
        if shots is None:
            raise ValueError("shots must be provided for noisy model")
        return NoisyQuantumReservoir(
            depth=depth,
            use_zz=use_zz,
            noise_model=noise_model,
            shots=shots,
        )

    return QuantumReservoir(
        depth=depth,
        use_zz=use_zz,
        use_classical_fallback=False,
        add_random_rotations=True,
        observable_set=observable_set,
    )


def _run_qrc_loco(
    cell_data: Dict[str, Dict[str, np.ndarray]],
    *,
    stage: str,
    depth: int,
    noisy: bool,
    noise_model: Optional[NoiseModel] = None,
    shots: Optional[int] = None,
    observable_config: str = "Z+ZZ",
    use_zz: bool = True,
    observable_set: str = "Z",
    model_name: str = "qrc",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]
        X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_raw = cell_data[test_cell]["X_raw"]
        y_test = cell_data[test_cell]["y"]

        X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)
        train_groups = np.concatenate(
            [np.full(cell_data[c]["y"].shape[0], c, dtype=object) for c in train_cells]
        )

        qrc = _build_qrc_model(
            noisy=noisy,
            depth=depth,
            noise_model=noise_model,
            shots=shots,
            use_zz=use_zz,
            observable_set=observable_set,
        )
        qrc.fit(X_train, y_train, groups=train_groups)
        y_pred = qrc.predict(X_test)
        m = _metrics(y_test, y_pred)
        naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

        rows.append(
            {
                "stage": stage,
                "model": model_name,
                "depth": depth,
                "shots": shots if shots is not None else np.nan,
                "observable_config": observable_config,
                "test_cell": test_cell,
                "train_cells": "+".join(train_cells),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "naive_mae": float(naive_mae),
                "beats_baseline": bool(m["mae"] < naive_mae),
                "reservoir_dim": int(qrc.get_reservoir_dim()),
            }
        )

    return pd.DataFrame(rows, columns=QRC_LOCO_COLUMNS)


def _run_stage1_noiseless_observable_depth(
    cell_data: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\nStage 1: Noiseless LOCO depth sweep with Z_only vs Z+ZZ")
    configs = [("Z_only", False, "Z"), ("Z+ZZ", True, "Z")]

    frames: List[pd.DataFrame] = []
    for depth in DEPTH_RANGE:
        print(f"  Depth {depth}")
        for label, use_zz, observable_set in configs:
            df = _run_qrc_loco(
                cell_data,
                stage="noiseless_observable_depth",
                depth=depth,
                noisy=False,
                observable_config=label,
                use_zz=use_zz,
                observable_set=observable_set,
                model_name="qrc",
            )
            frames.append(df)
            print(f"    {label:6s} mean LOCO MAE: {df['mae'].mean():.4f}")

    detail_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QRC_LOCO_COLUMNS)

    if detail_df.empty:
        return detail_df, pd.DataFrame(columns=["depth", "mae_z_only", "mae_z_plus_zz", "delta_z_only_minus_z_plus_zz"])

    summary = (
        detail_df.groupby(["depth", "observable_config"])["mae"]
        .mean()
        .reset_index()
        .pivot(index="depth", columns="observable_config", values="mae")
        .reset_index()
        .sort_values("depth")
    )

    if "Z_only" not in summary.columns:
        summary["Z_only"] = np.nan
    if "Z+ZZ" not in summary.columns:
        summary["Z+ZZ"] = np.nan

    summary = summary.rename(columns={"Z_only": "mae_z_only", "Z+ZZ": "mae_z_plus_zz"})
    summary["delta_z_only_minus_z_plus_zz"] = summary["mae_z_only"] - summary["mae_z_plus_zz"]
    return detail_df, summary


def _run_stage2_learning_curve(
    cell_data: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Learning curve: MAE vs number of training cells for QRC, ESN, XGBoost, RFF.

    k=1 is the extreme data-scarcity regime (single-cell training), k=5 is LOCO.
    ESN is the direct classical reservoir analog — the most important baseline for
    demonstrating quantum advantage.
    """
    print("\nStage 2: Learning curve by number of training cells")
    print(f"  Models: QRC (d=1), ESN (classical RC), XGBoost, RFF")
    print(f"  k range: {LEARNING_CURVE_CELL_COUNTS}")

    rows: List[Dict[str, object]] = []

    for k in LEARNING_CURVE_CELL_COUNTS:
        combos = list(combinations(CELL_IDS, k))
        print(f"  Train cells={k}, combinations={len(combos)}")

        for train_combo in combos:
            train_cells = list(train_combo)
            test_cells = [c for c in CELL_IDS if c not in train_cells]

            X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
            y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
            train_groups = np.concatenate(
                [np.full(cell_data[c]["y"].shape[0], c, dtype=object) for c in train_cells]
            )

            for test_cell in test_cells:
                X_test_raw = cell_data[test_cell]["X_raw"]
                y_test = cell_data[test_cell]["y"]

                X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)
                naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

                def _row(model_name, m):
                    return {
                        "stage": "learning_curve",
                        "model": model_name,
                        "n_train_cells": k,
                        "train_cells": "+".join(train_cells),
                        "test_cell": test_cell,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                        "r2": m["r2"],
                        "naive_mae": float(naive_mae),
                    }

                # --- QRC (quantum reservoir) ---
                qrc = QuantumReservoir(
                    depth=1,
                    use_zz=True,
                    use_classical_fallback=False,
                    add_random_rotations=True,
                    observable_set="Z",
                )
                qrc.fit(X_train, y_train, groups=train_groups)
                rows.append(_row("qrc_d1_z_plus_zz", _metrics(y_test, qrc.predict(X_test))))

                # --- ESN (classical reservoir computing — direct analog) ---
                esn = get_model_pipeline("esn")
                esn.fit(X_train, y_train)
                rows.append(_row("esn_pca6", _metrics(y_test, esn.predict(X_test))))

                # --- XGBoost (best classical non-reservoir baseline) ---
                xgb = get_model_pipeline("xgboost")
                xgb.fit(X_train, y_train)
                rows.append(_row("xgboost_pca6", _metrics(y_test, xgb.predict(X_test))))

                # --- RFF (classical random feature method — structural analog to QRC) ---
                rff = get_model_pipeline("rff")
                rff.fit(X_train, y_train)
                rows.append(_row("rff_pca6", _metrics(y_test, rff.predict(X_test))))

        if rows:
            tmp = pd.DataFrame(rows)
            sub = tmp[tmp["n_train_cells"] == k]
            if not sub.empty:
                summary = sub.groupby("model")["mae"].mean().sort_values()
                for model_name, mae in summary.items():
                    print(f"    {model_name}: mean MAE={mae:.4f}")

    detail_df = pd.DataFrame(rows, columns=LEARNING_CURVE_COLUMNS)

    if detail_df.empty:
        return detail_df, pd.DataFrame(columns=["model", "n_train_cells", "mae_mean", "mae_std", "n_folds"])

    summary_df = (
        detail_df.groupby(["model", "n_train_cells"])["mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mae_mean", "std": "mae_std", "count": "n_folds"})
        .sort_values(["model", "n_train_cells"])
    )
    return detail_df, summary_df


def _run_stage3_noise_regularizer(
    cell_data: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\nStage 3: Noise-as-regularizer ablation (depth=1, LOCO, shot sweep)")
    print(f"  Shots: {NOISE_SHOTS_SWEEP}")

    noise_model = get_marrakesh_noise_model()

    frames: List[pd.DataFrame] = []
    for shots in NOISE_SHOTS_SWEEP:
        df = _run_qrc_loco(
            cell_data,
            stage="noise_regularizer",
            depth=1,
            noisy=True,
            noise_model=noise_model,
            shots=shots,
            observable_config="Z+ZZ",
            use_zz=True,
            observable_set="Z",
            model_name="qrc_noisy_d1_z_plus_zz",
        )
        frames.append(df)
        print(f"  Shots {shots:5d}: mean LOCO MAE={df['mae'].mean():.4f}")

    detail_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QRC_LOCO_COLUMNS)

    if detail_df.empty:
        return detail_df, pd.DataFrame(columns=["shots", "mae_mean", "mae_std", "n_folds"])

    summary_df = (
        detail_df.groupby("shots")["mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mae_mean", "std": "mae_std", "count": "n_folds"})
        .sort_values("shots")
    )
    return detail_df, summary_df


def _run_stage4_qrc_vs_esn(
    cell_data: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\nStage 4: QRC vs ESN head-to-head on identical LOCO folds")

    rows: List[Dict[str, object]] = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]
        X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_raw = cell_data[test_cell]["X_raw"]
        y_test = cell_data[test_cell]["y"]

        X_train, X_test = _fit_pca_in_fold(X_train_raw, X_test_raw)
        train_groups = np.concatenate(
            [np.full(cell_data[c]["y"].shape[0], c, dtype=object) for c in train_cells]
        )
        naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

        qrc = QuantumReservoir(
            depth=1,
            use_zz=True,
            use_classical_fallback=False,
            add_random_rotations=True,
            observable_set="Z",
        )
        qrc.fit(X_train, y_train, groups=train_groups)
        y_pred_qrc = qrc.predict(X_test)
        qrc_m = _metrics(y_test, y_pred_qrc)

        rows.append(
            {
                "stage": "qrc_vs_esn",
                "model": "qrc_d1_z_plus_zz",
                "depth": 1,
                "test_cell": test_cell,
                "train_cells": "+".join(train_cells),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mae": qrc_m["mae"],
                "rmse": qrc_m["rmse"],
                "r2": qrc_m["r2"],
                "naive_mae": float(naive_mae),
            }
        )

        esn = get_model_pipeline("esn")
        esn.fit(X_train, y_train)
        y_pred_esn = esn.predict(X_test)
        esn_m = _metrics(y_test, y_pred_esn)

        rows.append(
            {
                "stage": "qrc_vs_esn",
                "model": "esn_pca6",
                "depth": np.nan,
                "test_cell": test_cell,
                "train_cells": "+".join(train_cells),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mae": esn_m["mae"],
                "rmse": esn_m["rmse"],
                "r2": esn_m["r2"],
                "naive_mae": float(naive_mae),
            }
        )

        print(
            f"  {test_cell}: "
            f"QRC MAE={qrc_m['mae']:.4f}, "
            f"ESN MAE={esn_m['mae']:.4f}, "
            f"delta(QRC-ESN)={qrc_m['mae'] - esn_m['mae']:+.4f}"
        )

    detail_df = pd.DataFrame(rows, columns=HEAD_TO_HEAD_COLUMNS)

    if detail_df.empty:
        return detail_df, pd.DataFrame(columns=["test_cell", "mae_qrc", "mae_esn", "delta_qrc_minus_esn"])

    per_cell_df = (
        detail_df.pivot(index="test_cell", columns="model", values="mae")
        .reset_index()
        .rename(columns={"qrc_d1_z_plus_zz": "mae_qrc", "esn_pca6": "mae_esn"})
    )

    if "mae_qrc" not in per_cell_df.columns:
        per_cell_df["mae_qrc"] = np.nan
    if "mae_esn" not in per_cell_df.columns:
        per_cell_df["mae_esn"] = np.nan

    per_cell_df["delta_qrc_minus_esn"] = per_cell_df["mae_qrc"] - per_cell_df["mae_esn"]
    per_cell_df["test_cell"] = pd.Categorical(per_cell_df["test_cell"], categories=CELL_IDS, ordered=True)
    per_cell_df = per_cell_df.sort_values("test_cell").reset_index(drop=True)
    per_cell_df["test_cell"] = per_cell_df["test_cell"].astype(str)

    return detail_df, per_cell_df


def _plot_stage1_observable_depth(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    if summary_df.empty:
        print("Skipping stage1 plot: no data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["depth"], summary_df["mae_z_only"], marker="o", linewidth=2.0, label="Z_only")
    ax.plot(summary_df["depth"], summary_df["mae_z_plus_zz"], marker="s", linewidth=2.0, label="Z+ZZ")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("Noiseless LOCO: Z_only vs Z+ZZ")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, plot_dir, "noiseless_z_only_vs_z_plus_zz")


def _plot_stage2_learning_curve(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    """Publication-quality learning curve with error bands.

    Shows mean MAE ± 1 std across all leave-one-out combinations for each
    number of training cells.  QRC, ESN (classical reservoir), XGBoost, and
    RFF are compared so reviewers can see the quantum advantage is not simply
    a reservoir-computing effect.
    """
    if summary_df.empty:
        print("Skipping stage2 plot: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    k_vals = sorted(summary_df["n_train_cells"].unique())
    x_ticks = k_vals

    for model_name in MODEL_PLOT_ORDER:
        sub = summary_df[summary_df["model"] == model_name].sort_values("n_train_cells")
        if sub.empty:
            continue
        xs = sub["n_train_cells"].values
        ys = sub["mae_mean"].values
        yerr = sub["mae_std"].values

        color = MODEL_COLORS.get(model_name, "grey")
        ls = MODEL_LINESTYLES.get(model_name, "-")
        lw = 2.5 if model_name == "qrc_d1_z_plus_zz" else 1.8
        zorder = 5 if model_name == "qrc_d1_z_plus_zz" else 3

        ax.plot(
            xs, ys,
            marker="o",
            linewidth=lw,
            linestyle=ls,
            color=color,
            zorder=zorder,
            label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
        )
        ax.fill_between(
            xs,
            ys - yerr,
            ys + yerr,
            alpha=0.12,
            color=color,
            zorder=zorder - 1,
        )

    ax.set_xlabel("Number of training cells", fontsize=12)
    ax.set_ylabel("Mean MAE (SOH fraction)", fontsize=12)
    ax.set_title("Data Efficiency: QRC vs Classical Baselines", fontsize=13, pad=10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(k) for k in x_ticks])
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)

    # Annotate QRC advantage at max k vs ESN/XGBoost
    max_k = max(k_vals)
    qrc_max = summary_df[(summary_df["model"] == "qrc_d1_z_plus_zz") & (summary_df["n_train_cells"] == max_k)]
    xgb_max = summary_df[(summary_df["model"] == "xgboost_pca6") & (summary_df["n_train_cells"] == max_k)]
    esn_max = summary_df[(summary_df["model"] == "esn_pca6") & (summary_df["n_train_cells"] == max_k)]

    if not qrc_max.empty and not xgb_max.empty:
        qrc_val = float(qrc_max["mae_mean"].iloc[0])
        xgb_val = float(xgb_max["mae_mean"].iloc[0])
        pct_vs_xgb = 100.0 * (xgb_val - qrc_val) / xgb_val
        ax.annotate(
            f"QRC leads XGBoost\nby {pct_vs_xgb:.0f}% at k={max_k}",
            xy=(max_k, qrc_val),
            xytext=(max_k - 0.6, qrc_val + 0.0035),
            fontsize=9,
            color=MODEL_COLORS["qrc_d1_z_plus_zz"],
            arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["qrc_d1_z_plus_zz"], lw=1.2),
        )

    if not qrc_max.empty and not esn_max.empty:
        esn_val = float(esn_max["mae_mean"].iloc[0])
        qrc_val = float(qrc_max["mae_mean"].iloc[0])
        pct_vs_esn = 100.0 * (esn_val - qrc_val) / esn_val
        ax.annotate(
            f"QRC leads ESN\n(classical RC) by {pct_vs_esn:.0f}%",
            xy=(max_k, esn_val),
            xytext=(max_k - 1.4, esn_val - 0.003),
            fontsize=9,
            color=MODEL_COLORS["esn_pca6"],
            arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["esn_pca6"], lw=1.2),
        )

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, plot_dir, "learning_curve_qrc_vs_xgboost")


def _plot_stage3_shot_sweep(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    if summary_df.empty:
        print("Skipping stage3 plot: no data")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(summary_df["shots"], summary_df["mae_mean"], marker="o", linewidth=2.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NOISE_SHOTS_SWEEP)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Shots (log2 scale)")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("Noisy QRC Depth=1: MAE vs Shots")
    ax.grid(True, linestyle="--", alpha=0.3)
    _save_figure(fig, plot_dir, "noisy_shot_sweep_depth1")


def _plot_stage4_head_to_head(per_cell_df: pd.DataFrame, plot_dir: Path) -> None:
    if per_cell_df.empty:
        print("Skipping stage4 plot: no data")
        return

    x = np.arange(len(per_cell_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, per_cell_df["mae_qrc"], width=width, label="QRC d1 Z+ZZ")
    ax.bar(x + width / 2, per_cell_df["mae_esn"], width=width, label="ESN PCA6")
    ax.set_xticks(x)
    ax.set_xticklabels(per_cell_df["test_cell"])
    ax.set_xlabel("Held-out test cell")
    ax.set_ylabel("LOCO MAE")
    ax.set_title("QRC vs ESN per-cell LOCO MAE")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, plot_dir, "qrc_vs_esn_per_cell")


def _print_stage4_table(per_cell_df: pd.DataFrame) -> None:
    if per_cell_df.empty:
        print("No per-cell comparison rows to print.")
        return

    print("\nPer-cell QRC vs ESN LOCO MAE")
    print("test_cell | mae_qrc | mae_esn | delta_qrc_minus_esn")
    for _, row in per_cell_df.iterrows():
        print(
            f"{row['test_cell']:>8s} | "
            f"{row['mae_qrc']:.4f} | "
            f"{row['mae_esn']:.4f} | "
            f"{row['delta_qrc_minus_esn']:+.4f}"
        )


def _summarize_best(summary_df: pd.DataFrame, value_col: str, label_col: str) -> Optional[Tuple[object, float]]:
    if summary_df.empty:
        return None
    idx = summary_df[value_col].idxmin()
    row = summary_df.loc[idx]
    return row[label_col], float(row[value_col])


def _safe_to_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")


def main(run_noisy: bool = True) -> None:
    data_dir, plot_dir = get_stage_paths("stage_3")
    log_path = data_dir / "stage_3_log.txt"

    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Phase 5 Stage 3: Extended QRC Analyses")
        print("=" * 72)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Data dir: {data_dir}")
        print(f"Plot dir: {plot_dir}")
        print(f"CELL_IDS: {CELL_IDS}")

        cell_data = load_stanford_data()

        # 1) Noiseless Z_only re-run across all depths and direct comparison.
        stage1_detail, stage1_summary = _run_stage1_noiseless_observable_depth(cell_data)
        _safe_to_csv(stage1_detail, data_dir / "noiseless_loco_observable_depth.csv")
        _safe_to_csv(stage1_summary, data_dir / "noiseless_loco_observable_depth_summary.csv")
        _plot_stage1_observable_depth(stage1_summary, plot_dir)

        # 2) Learning curve over training-cell counts.
        stage2_detail, stage2_summary = _run_stage2_learning_curve(cell_data)
        _safe_to_csv(stage2_detail, data_dir / "learning_curve_loco.csv")
        _safe_to_csv(stage2_summary, data_dir / "learning_curve_loco_summary.csv")
        _plot_stage2_learning_curve(stage2_summary, plot_dir)

        # 3) Noise-as-regularizer shot sweep on noisy QRC depth=1.
        if run_noisy:
            stage3_detail, stage3_summary = _run_stage3_noise_regularizer(cell_data)
        else:
            stage3_detail = pd.DataFrame(columns=QRC_LOCO_COLUMNS)
            stage3_summary = pd.DataFrame(columns=["shots", "mae_mean", "mae_std", "n_folds"])
            print("\nStage 3 skipped by user flag.")

        _safe_to_csv(stage3_detail, data_dir / "noisy_shot_sweep_loco.csv")
        _safe_to_csv(stage3_summary, data_dir / "noisy_shot_sweep_summary.csv")
        _plot_stage3_shot_sweep(stage3_summary, plot_dir)

        # 4) QRC vs ESN on identical LOCO folds.
        stage4_detail, stage4_per_cell = _run_stage4_qrc_vs_esn(cell_data)
        _safe_to_csv(stage4_detail, data_dir / "qrc_vs_esn_loco.csv")
        _safe_to_csv(stage4_per_cell, data_dir / "qrc_vs_esn_per_cell.csv")
        _plot_stage4_head_to_head(stage4_per_cell, plot_dir)
        _print_stage4_table(stage4_per_cell)

        print("\nKey summary")
        if not stage1_summary.empty:
            for _, row in stage1_summary.iterrows():
                print(
                    f"  depth={int(row['depth'])}: "
                    f"Z_only={row['mae_z_only']:.4f}, "
                    f"Z+ZZ={row['mae_z_plus_zz']:.4f}, "
                    f"delta={row['delta_z_only_minus_z_plus_zz']:+.4f}"
                )

        if not stage2_summary.empty:
            best_qrc = _summarize_best(
                stage2_summary[stage2_summary["model"] == "qrc_d1_z_plus_zz"],
                "mae_mean",
                "n_train_cells",
            )
            best_xgb = _summarize_best(
                stage2_summary[stage2_summary["model"] == "xgboost_pca6"],
                "mae_mean",
                "n_train_cells",
            )
            if best_qrc is not None:
                print(f"  Learning curve best QRC: k={int(best_qrc[0])}, mean MAE={best_qrc[1]:.4f}")
            if best_xgb is not None:
                print(f"  Learning curve best XGBoost: k={int(best_xgb[0])}, mean MAE={best_xgb[1]:.4f}")

        if not stage3_summary.empty:
            best_shot = _summarize_best(stage3_summary, "mae_mean", "shots")
            if best_shot is not None:
                print(f"  Best noisy shots: {int(best_shot[0])}, mean LOCO MAE={best_shot[1]:.4f}")

        if not stage4_per_cell.empty:
            qrc_mean = stage4_per_cell["mae_qrc"].mean()
            esn_mean = stage4_per_cell["mae_esn"].mean()
            print(f"  Head-to-head mean MAE: QRC={qrc_mean:.4f}, ESN={esn_mean:.4f}")

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 Stage 3 QRC analyses.")
    parser.add_argument(
        "--skip-noisy",
        action="store_true",
        help="Skip Stage 3 noisy shot sweep.",
    )
    args = parser.parse_args()
    main(run_noisy=not args.skip_noisy)
