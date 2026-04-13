"""Shared helpers for the Warwick-primary Phase 6 pipeline.

This module centralizes:
- canonical Stage 4 result paths
- run-label generation
- foldwise reservoir evaluation
- unified Warwick comparison assembly

The intent is to keep Warwick as the primary hardware pipeline with a
single clean provenance chain:
    prepare manifest -> run shadow refs -> run hardware -> analyze
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import RIDGE_ALPHAS
from phase_6.paths import PHASE_6_ROOT


STAGE_NAME = "stage_4_warwick_primary"
DEFAULT_BACKEND = "ibm_marrakesh"
DEFAULT_PREPROCESS_MODE = "foldwise"
DEFAULT_SCOPE = "primary"
DEFAULT_SHOTS = 3072
DEFAULT_MAX_CIRCUITS_PER_BATCH = 96


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def build_run_label(
    backend: str,
    preprocess_mode: str,
    scope: str,
    shots: int,
) -> str:
    return "__".join(
        [
            _slug(backend),
            _slug(preprocess_mode),
            _slug(scope),
            f"s{int(shots)}",
        ]
    )


@dataclass(frozen=True)
class WarwickPrimaryPaths:
    run_label: str

    @property
    def stage_dir(self) -> Path:
        return PHASE_6_ROOT / STAGE_NAME

    @property
    def stage_data_dir(self) -> Path:
        return self.stage_dir / "data"

    @property
    def stage_plot_dir(self) -> Path:
        return self.stage_dir / "plot"

    @property
    def stage_hardware_dir(self) -> Path:
        return self.stage_dir / "hardware"

    @property
    def data_dir(self) -> Path:
        return self.stage_data_dir / self.run_label

    @property
    def plot_dir(self) -> Path:
        return self.stage_plot_dir / self.run_label

    @property
    def manifest_dir(self) -> Path:
        return self.stage_hardware_dir / "manifest" / self.run_label

    @property
    def checkpoint_dir(self) -> Path:
        return self.stage_hardware_dir / "checkpoint" / self.run_label

    @property
    def manifest_path(self) -> Path:
        return self.manifest_dir / "manifest.json"

    @property
    def feature_records_path(self) -> Path:
        return self.data_dir / "warwick_feature_records.csv"

    @property
    def batch_summary_path(self) -> Path:
        return self.data_dir / "warwick_prepare_summary.csv"

    @property
    def prepare_log_path(self) -> Path:
        return self.data_dir / "stage_4_warwick_primary_prepare_log.txt"

    @property
    def shadow_log_path(self) -> Path:
        return self.data_dir / "stage_4_warwick_primary_shadow_log.txt"

    @property
    def hardware_log_path(self) -> Path:
        return self.data_dir / "stage_4_warwick_primary_hardware_log.txt"

    def ensure_dirs(self, include_hardware: bool = False) -> None:
        self.stage_data_dir.mkdir(parents=True, exist_ok=True)
        self.stage_plot_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        if include_hardware:
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_run_paths(run_label: str, include_hardware: bool = False) -> WarwickPrimaryPaths:
    paths = WarwickPrimaryPaths(run_label=run_label)
    paths.ensure_dirs(include_hardware=include_hardware)
    return paths


def load_manifest(paths: WarwickPrimaryPaths) -> dict[str, Any]:
    if not paths.manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {paths.manifest_path}\n"
            "Run python -m src.phase_6.prepare_warwick_hardware first."
        )
    with open(paths.manifest_path, encoding="utf-8") as f:
        return json.load(f)


def load_feature_records(manifest: dict[str, Any]) -> pd.DataFrame:
    rel_path = manifest.get("feature_records_csv")
    if not rel_path:
        raise KeyError("Manifest missing feature_records_csv")
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Feature record CSV not found: {path}")
    return pd.read_csv(path)


def load_feature_matrix(feature_records: pd.DataFrame) -> np.ndarray:
    pc_cols = sorted(
        [col for col in feature_records.columns if col.startswith("pc")],
        key=lambda name: int(name[2:]),
    )
    if not pc_cols:
        raise ValueError("Feature records do not contain PCA columns (pc1..pcN).")
    return feature_records[pc_cols].to_numpy(dtype=float)


def unique_manifest_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    configs: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for batch_cfg in manifest["batches"].values():
        key = (
            int(batch_cfg["depth"]),
            int(batch_cfg["seed"]),
            int(batch_cfg["shots"]),
            str(batch_cfg["experiment"]),
        )
        if key not in configs:
            configs[key] = {
                "depth": int(batch_cfg["depth"]),
                "seed": int(batch_cfg["seed"]),
                "shots": int(batch_cfg["shots"]),
                "experiment": str(batch_cfg["experiment"]),
                "random_rotations": np.asarray(batch_cfg["random_rotations"], dtype=float),
            }
    return [configs[key] for key in sorted(configs)]


def evaluate_reservoir_runs(
    run_records: list[dict[str, Any]],
    feature_records: pd.DataFrame,
    manifest: dict[str, Any],
    run_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocess_mode = str(manifest.get("preprocess_mode", ""))
    if preprocess_mode != "foldwise":
        raise NotImplementedError(
            "Warwick-primary evaluation currently supports foldwise manifests only."
        )

    n_features = int(manifest["n_feature_rows"])
    rows: list[dict[str, Any]] = []

    for run in run_records:
        reservoir = np.asarray(run["reservoir"], dtype=float)
        if reservoir.shape[0] != n_features:
            raise ValueError(
                f"Reservoir row mismatch for {run['mode']} {run['backend']} "
                f"d={run['depth']} s={run['seed']}: {reservoir.shape[0]} vs {n_features}"
            )

        valid_mask = ~np.isnan(reservoir[:, 0])
        if not np.all(valid_mask):
            missing = np.where(~valid_mask)[0].tolist()
            raise RuntimeError(
                f"Missing feature rows for {run['mode']} {run['backend']} "
                f"d={run['depth']} s={run['seed']}: {missing[:10]}"
            )

        for outer_fold in sorted(feature_records["outer_fold"].unique()):
            fold_df = feature_records[feature_records["outer_fold"] == outer_fold].copy()
            train_df = fold_df[fold_df["role"] == "train"].copy()
            test_df = fold_df[fold_df["role"] == "test"].copy()
            if len(test_df) != 1:
                raise ValueError(
                    f"Expected one test row for outer_fold={outer_fold}, found {len(test_df)}"
                )

            train_ids = train_df["feature_id"].astype(int).to_numpy()
            test_id = int(test_df["feature_id"].iloc[0])
            X_train = reservoir[train_ids]
            y_train = train_df["y_true"].to_numpy(dtype=float)
            X_test = reservoir[[test_id]]
            y_test = test_df["y_true"].to_numpy(dtype=float)

            search = GridSearchCV(
                estimator=Ridge(),
                param_grid={"alpha": RIDGE_ALPHAS},
                cv=LeaveOneOut(),
                scoring="neg_mean_absolute_error",
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            abs_error = float(abs(y_test[0] - y_pred[0]))

            rows.append(
                {
                    "run_label": run_label,
                    "mode": str(run["mode"]),
                    "backend": str(run["backend"]),
                    "depth": int(run["depth"]),
                    "seed": int(run["seed"]),
                    "shots": int(run["shots"]),
                    "experiment": str(run["experiment"]),
                    "outer_fold": int(outer_fold),
                    "test_cell": test_df["test_cell"].iloc[0],
                    "y_true": float(y_test[0]),
                    "y_pred": float(y_pred[0]),
                    "abs_error": abs_error,
                    "abs_error_pct": float(abs_error * 100.0),
                    "abs_error_ah": float(abs_error * manifest["nominal_capacity_ah"]),
                    "best_alpha": float(search.best_params_["alpha"]),
                    "n_train": int(len(train_ids)),
                    "n_test": 1,
                    "job_ids": str(run.get("job_ids", "")),
                    "qpu_seconds_total": float(run.get("qpu_seconds_total", 0.0)),
                }
            )

    results_df = pd.DataFrame(rows)
    summary_df = (
        results_df.groupby(
            ["run_label", "mode", "backend", "depth", "seed", "shots", "experiment"],
            as_index=False,
        )["abs_error"]
        .agg(mae_mean="mean", mae_std="std", n_folds="count")
    )
    summary_df["mae_pct_mean"] = summary_df["mae_mean"] * 100.0
    summary_df["mae_pct_std"] = summary_df["mae_std"] * 100.0
    return results_df, summary_df


def load_latest_unified_warwick_reference() -> tuple[Path | None, pd.DataFrame | None]:
    unified_dir = PROJECT_ROOT / "result" / "manuscript_support" / "unified_loco" / "data"
    if not unified_dir.exists():
        return None, None

    candidates = sorted(
        unified_dir.glob("unified_loco__*__summary.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        subset = df[(df["dataset"] == "warwick") & (df["feature_variant"] == "native_pca6")]
        if not subset.empty:
            return path, subset.copy()
    return None, None


def build_warwick_comparison_rows(
    summary_df: pd.DataFrame,
    source_tag: str,
) -> tuple[pd.DataFrame | None, Path | None]:
    unified_path, unified_df = load_latest_unified_warwick_reference()
    if unified_df is None:
        return None, None

    best_row = summary_df.sort_values("mae_mean").iloc[0]
    comp_rows = unified_df[["model", "mae_mean", "mae_pct_mean"]].copy()
    comp_rows = pd.concat(
        [
            comp_rows,
            pd.DataFrame(
                [
                    {
                        "model": (
                            f"{source_tag}_{best_row['mode']}_{best_row['backend']}"
                            f"_d{int(best_row['depth'])}_s{int(best_row['seed'])}"
                        ),
                        "mae_mean": best_row["mae_mean"],
                        "mae_pct_mean": best_row["mae_pct_mean"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    comp_rows["source"] = [
        *(["unified_warwick"] * len(unified_df)),
        source_tag,
    ]
    return comp_rows, unified_path


def write_current_run_metadata(
    paths: WarwickPrimaryPaths,
    manifest: dict[str, Any],
    *,
    updated_by: str,
) -> Path:
    payload = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "updated_by": updated_by,
        "run_label": paths.run_label,
        "stage_name": STAGE_NAME,
        "backend": manifest.get("backend"),
        "preprocess_mode": manifest.get("preprocess_mode"),
        "scope": manifest.get("scope"),
        "shots": manifest.get("shots_default"),
        "manifest_path": str(paths.manifest_path.relative_to(PROJECT_ROOT)),
    }
    meta_path = paths.stage_data_dir / "current_primary_run.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return meta_path
