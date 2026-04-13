"""Phase 6 Warwick-primary preparation (offline only).

Builds a run-ready manifest for Warwick hardware experiments without submitting
to IBM. This is the primary Phase 6 hardware path and writes into a clean,
run-labeled Stage 4 root so different backends or preprocessing choices never
overwrite each other.

Default target:
- dataset: Warwick DIB (25degC / 50SOC)
- preprocessing: foldwise train-only StandardScaler(raw) -> PCA(6) -> StandardScaler(PCA)
- scope: primary = depth 1, seed 42
- backend tag: ibm_marrakesh
- shots: 3072

Usage:
    python -m src.phase_6.prepare_warwick_hardware
    python -m src.phase_6.prepare_warwick_hardware --preprocess-mode global
    python -m src.phase_6.prepare_warwick_hardware --scope full
    python -m src.phase_6.prepare_warwick_hardware --backend ibm_fez
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import N_PCA, N_QUBITS, RANDOM_STATE
from data_loader_warwick import (
    TARGET_SOC,
    TARGET_TEMP,
    WARWICK_NOMINAL_CAP_AH,
    load_warwick_data,
)
from phase_4.circuit import build_qrc_circuit, encode_features
from phase_4.config import CLAMP_RANGE, ENCODING_METHOD
from phase_6.warwick_primary_common import (
    DEFAULT_BACKEND,
    DEFAULT_MAX_CIRCUITS_PER_BATCH,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_SCOPE,
    DEFAULT_SHOTS,
    STAGE_NAME,
    build_run_label,
    get_run_paths,
    write_current_run_metadata,
)

try:
    from qiskit.qasm2 import dumps as qasm2_dumps
except ImportError:
    from qiskit import qasm2

    qasm2_dumps = qasm2.dumps


QPU_EST_BASE_SEC = 2.0
QPU_EST_PER_SHOT_CIRCUIT = 0.00035

SCOPE_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "primary": [
        {"depth": 1, "seed": 42, "experiment": "primary"},
    ],
    "depth_sweep": [
        {"depth": 1, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 2, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 3, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 4, "seed": 42, "experiment": "depth_sweep"},
    ],
    "seed_sweep": [
        {"depth": 1, "seed": 42, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 43, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 44, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 45, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 46, "experiment": "seed_sweep"},
    ],
    "full": [
        {"depth": 1, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 2, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 3, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 4, "seed": 42, "experiment": "depth_sweep"},
        {"depth": 1, "seed": 43, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 44, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 45, "experiment": "seed_sweep"},
        {"depth": 1, "seed": 46, "experiment": "seed_sweep"},
    ],
}


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
def _estimate_qpu_seconds(n_circuits: int, shots: int) -> float:
    return QPU_EST_BASE_SEC + QPU_EST_PER_SHOT_CIRCUIT * (n_circuits * shots)


def _fit_projection_in_fold(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    n_components: int = N_PCA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_components = min(n_components, X_train_raw.shape[0], X_train_raw.shape[1])
    scaler_raw = StandardScaler()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    scaler_pca = StandardScaler()

    X_train_scaled_raw = scaler_raw.fit_transform(X_train_raw)
    X_test_scaled_raw = scaler_raw.transform(X_test_raw)
    X_train_pca = pca.fit_transform(X_train_scaled_raw)
    X_test_pca = pca.transform(X_test_scaled_raw)
    X_train_qrc = scaler_pca.fit_transform(X_train_pca)
    X_test_qrc = scaler_pca.transform(X_test_pca)

    meta = {
        "n_components": int(n_components),
        "pipeline": "raw_scaler_then_pca_then_post_pca_scaler",
        "raw_scaler_mean": scaler_raw.mean_.tolist(),
        "raw_scaler_scale": scaler_raw.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "post_pca_scaler_mean": scaler_pca.mean_.tolist(),
        "post_pca_scaler_scale": scaler_pca.scale_.tolist(),
    }
    return X_train_qrc, X_test_qrc, meta


def _load_warwick_bundle() -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    data = load_warwick_data(temp=TARGET_TEMP, soc=TARGET_SOC)
    cell_ids = sorted(data.keys())
    return data, cell_ids


def _build_global_feature_table(
    cell_data: dict[str, dict[str, np.ndarray]],
    cell_ids: list[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    X_raw = np.vstack([cell_data[cell_id]["X_raw"] for cell_id in cell_ids])
    y = np.concatenate([cell_data[cell_id]["y"] for cell_id in cell_ids])

    scaler_raw = StandardScaler()
    pca = PCA(n_components=min(N_PCA, X_raw.shape[0], X_raw.shape[1]), random_state=RANDOM_STATE)
    scaler_pca = StandardScaler()
    X_scaled_raw = scaler_raw.fit_transform(X_raw)
    X_pca = pca.fit_transform(X_scaled_raw)
    X_qrc = scaler_pca.fit_transform(X_pca)
    X_qrc = np.clip(X_qrc, CLAMP_RANGE[0], CLAMP_RANGE[1])

    rows: list[dict[str, Any]] = []
    for feature_id, cell_id in enumerate(cell_ids):
        rows.append({
            "feature_id": feature_id,
            "preprocess_mode": "global",
            "outer_fold": -1,
            "test_cell": "",
            "role": "global",
            "source_cell": cell_id,
            "source_index": int(feature_id),
            "is_test_sample": False,
            "y_true": float(y[feature_id]),
            **{f"pc{i+1}": float(X_qrc[feature_id, i]) for i in range(X_qrc.shape[1])},
        })

    meta = {
        "mode": "global",
        "pipeline": "raw_scaler_then_pca_then_post_pca_scaler",
        "n_feature_rows": int(len(rows)),
        "n_components": int(X_qrc.shape[1]),
        "raw_scaler_mean": scaler_raw.mean_.tolist(),
        "raw_scaler_scale": scaler_raw.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "post_pca_scaler_mean": scaler_pca.mean_.tolist(),
        "post_pca_scaler_scale": scaler_pca.scale_.tolist(),
    }
    return X_qrc, pd.DataFrame(rows), meta


def _build_foldwise_feature_table(
    cell_data: dict[str, dict[str, np.ndarray]],
    cell_ids: list[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    feature_rows: list[np.ndarray] = []
    record_rows: list[dict[str, Any]] = []
    fold_transforms: list[dict[str, Any]] = []
    feature_id = 0

    for outer_fold, test_cell in enumerate(cell_ids, start=1):
        train_cells = [cell_id for cell_id in cell_ids if cell_id != test_cell]
        X_train_raw = np.vstack([cell_data[cell_id]["X_raw"] for cell_id in train_cells])
        X_test_raw = cell_data[test_cell]["X_raw"]

        X_train_pca, X_test_pca, meta = _fit_projection_in_fold(X_train_raw, X_test_raw)
        X_train_pca = np.clip(X_train_pca, CLAMP_RANGE[0], CLAMP_RANGE[1])
        X_test_pca = np.clip(X_test_pca, CLAMP_RANGE[0], CLAMP_RANGE[1])

        fold_transforms.append({
            "outer_fold": int(outer_fold),
            "test_cell": test_cell,
            **meta,
        })

        for local_idx, train_cell in enumerate(train_cells):
            feature_rows.append(X_train_pca[local_idx])
            record_rows.append({
                "feature_id": feature_id,
                "preprocess_mode": "foldwise",
                "outer_fold": int(outer_fold),
                "test_cell": test_cell,
                "role": "train",
                "source_cell": train_cell,
                "source_index": int(cell_ids.index(train_cell)),
                "is_test_sample": False,
                "y_true": float(cell_data[train_cell]["y"][0]),
                **{f"pc{i+1}": float(X_train_pca[local_idx, i]) for i in range(X_train_pca.shape[1])},
            })
            feature_id += 1

        feature_rows.append(X_test_pca[0])
        record_rows.append({
            "feature_id": feature_id,
            "preprocess_mode": "foldwise",
            "outer_fold": int(outer_fold),
            "test_cell": test_cell,
            "role": "test",
            "source_cell": test_cell,
            "source_index": int(cell_ids.index(test_cell)),
            "is_test_sample": True,
            "y_true": float(cell_data[test_cell]["y"][0]),
            **{f"pc{i+1}": float(X_test_pca[0, i]) for i in range(X_test_pca.shape[1])},
        })
        feature_id += 1

    X_features = np.vstack(feature_rows)
    meta = {
        "mode": "foldwise",
        "n_feature_rows": int(len(record_rows)),
        "n_outer_folds": int(len(cell_ids)),
        "fold_transforms": fold_transforms,
    }
    return X_features, pd.DataFrame(record_rows), meta


def _build_feature_table(
    preprocess_mode: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, dict[str, np.ndarray]], list[str]]:
    cell_data, cell_ids = _load_warwick_bundle()
    if preprocess_mode == "global":
        X_features, feature_records, preprocess_meta = _build_global_feature_table(cell_data, cell_ids)
    else:
        X_features, feature_records, preprocess_meta = _build_foldwise_feature_table(cell_data, cell_ids)
    return X_features, feature_records, preprocess_meta, cell_data, cell_ids


def _build_budget_options(
    n_cells: int,
    shots: int,
    max_circuits_per_batch: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_config = {
        "global": n_cells,
        "foldwise": n_cells * n_cells,
    }
    for preprocess_mode, circuits_per_config in per_config.items():
        for scope, configs in SCOPE_CONFIGS.items():
            total_circuits = circuits_per_config * len(configs)
            circuits_per_batches = _chunk_indices(
                list(range(circuits_per_config)),
                max_circuits_per_batch,
            )
            est_per_config_sec = sum(
                _estimate_qpu_seconds(len(batch), shots)
                for batch in circuits_per_batches
            )
            est_total_qpu_sec = est_per_config_sec * len(configs)
            rows.append({
                "preprocess_mode": preprocess_mode,
                "scope": scope,
                "n_configs": int(len(configs)),
                "circuits_per_config": int(circuits_per_config),
                "total_circuits": int(total_circuits),
                "max_circuits_per_batch": int(max_circuits_per_batch),
                "estimated_batches": int(len(configs) * len(circuits_per_batches)),
                "estimated_total_qpu_sec": float(est_total_qpu_sec),
                "estimated_total_qpu_min": float(est_total_qpu_sec / 60.0),
                "shots": int(shots),
            })
    return pd.DataFrame(rows)


def _chunk_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def _build_manifest(
    feature_records: pd.DataFrame,
    X_features: np.ndarray,
    preprocess_meta: dict[str, Any],
    cell_data: dict[str, dict[str, np.ndarray]],
    cell_ids: list[str],
    preprocess_mode: str,
    scope: str,
    shots: int,
    backend: str,
    max_circuits_per_batch: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    configs = SCOPE_CONFIGS[scope]
    angles_all = encode_features(X_features, method=ENCODING_METHOD)
    feature_ids = feature_records["feature_id"].astype(int).tolist()

    manifest: dict[str, Any] = {
        "created": datetime.now().isoformat(),
        "dataset": "warwick",
        "temp": TARGET_TEMP,
        "soc": TARGET_SOC,
        "nominal_capacity_ah": WARWICK_NOMINAL_CAP_AH,
        "feature_variant": "native_pca6",
        "preprocess_mode": preprocess_mode,
        "scope": scope,
        "stage_name": STAGE_NAME,
        "offline_only": True,
        "n_source_samples": int(len(cell_ids)),
        "n_feature_rows": int(len(feature_ids)),
        "n_qubits": N_QUBITS,
        "backend": backend,
        "shots_default": int(shots),
        "max_circuits_per_batch": int(max_circuits_per_batch),
        "source_cell_ids": cell_ids,
        "source_soh": [float(cell_data[cell_id]["y"][0]) for cell_id in cell_ids],
        "frequency_hz": cell_data[cell_ids[0]]["freq"].astype(float).tolist(),
        "preprocessing": preprocess_meta,
        "qpu_estimator": {
            "base_seconds": QPU_EST_BASE_SEC,
            "seconds_per_shot_circuit": QPU_EST_PER_SHOT_CIRCUIT,
        },
        "notes": [
            "Default primary scope uses depth 1, seed 42, because stored nested Warwick LOCO selected depth 1 consistently.",
            "Foldwise preprocessing matches the Warwick benchmark path: StandardScaler(raw) -> PCA(6) -> StandardScaler(PCA), fit on train only per outer fold.",
            "Run src.phase_6.run_warwick_shadow before paid submission so noiseless and Marrakesh digital-twin references are current.",
            "Run src.phase_6.run_warwick_hardware for paid execution using the same run label.",
        ],
        "batches": {},
    }

    batch_rows: list[dict[str, Any]] = []
    batch_id = 1
    for cfg in configs:
        depth = int(cfg["depth"])
        seed = int(cfg["seed"])
        experiment = str(cfg["experiment"])
        rng = np.random.RandomState(seed)
        random_rotations = rng.uniform(0, 2 * np.pi, (depth, N_QUBITS, 3))
        for chunk_idx, chunk_ids in enumerate(_chunk_indices(feature_ids, max_circuits_per_batch), start=1):
            qasm_list: list[str] = []
            for feature_id in chunk_ids:
                qc = build_qrc_circuit(
                    angles_all[feature_id],
                    depth=depth,
                    random_rotations=random_rotations,
                )
                qc.measure_all()
                qasm_list.append(qasm2_dumps(qc))

            est_qpu_sec = _estimate_qpu_seconds(len(chunk_ids), shots)
            manifest["batches"][str(batch_id)] = {
                "depth": depth,
                "seed": seed,
                "shots": int(shots),
                "experiment": experiment,
                "sample_indices": [int(idx) for idx in chunk_ids],
                "random_rotations": random_rotations.tolist(),
                "n_circuits": int(len(chunk_ids)),
                "qasm": qasm_list,
            }
            batch_rows.append({
                "batch_id": int(batch_id),
                "depth": depth,
                "seed": seed,
                "experiment": experiment,
                "chunk_index": int(chunk_idx),
                "n_circuits": int(len(chunk_ids)),
                "shots": int(shots),
                "estimated_qpu_sec": float(est_qpu_sec),
                "estimated_qpu_min": float(est_qpu_sec / 60.0),
            })
            batch_id += 1

    return manifest, pd.DataFrame(batch_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Warwick hardware manifests offline.")
    parser.add_argument(
        "--preprocess-mode",
        choices=("foldwise", "global"),
        default=DEFAULT_PREPROCESS_MODE,
        help="foldwise matches the Warwick benchmark preprocessing; global is cheaper but approximate.",
    )
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPE_CONFIGS.keys()),
        default=DEFAULT_SCOPE,
        help="primary is the safest first paid Warwick target.",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
    parser.add_argument("--max-circuits-per-batch", type=int, default=DEFAULT_MAX_CIRCUITS_PER_BATCH)
    args = parser.parse_args()

    run_label = build_run_label(
        backend=args.backend,
        preprocess_mode=args.preprocess_mode,
        scope=args.scope,
        shots=args.shots,
    )
    paths = get_run_paths(run_label, include_hardware=True)
    data_dir = paths.data_dir
    log_path = paths.prepare_log_path
    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Phase 6 Stage 4: Warwick Primary Preparation (offline only)")
        print("=" * 72)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Run label: {run_label}")
        print(f"Preprocess mode: {args.preprocess_mode}")
        print(f"Scope: {args.scope}")
        print(f"Shots: {args.shots}")
        print(f"Backend tag: {args.backend}")
        print(f"Max circuits / batch: {args.max_circuits_per_batch}")

        X_features, feature_records, preprocess_meta, cell_data, cell_ids = _build_feature_table(
            args.preprocess_mode
        )
        print(f"\nLoaded Warwick cells: {len(cell_ids)}")
        print(f"Feature rows prepared: {len(feature_records)}")
        print(f"Raw feature dim: {next(iter(cell_data.values()))['X_raw'].shape[1]}")
        print(f"PCA feature dim: {X_features.shape[1]}")

        budget_df = _build_budget_options(
            n_cells=len(cell_ids),
            shots=args.shots,
            max_circuits_per_batch=args.max_circuits_per_batch,
        )
        budget_path = data_dir / "budget_options.csv"
        budget_df.to_csv(budget_path, index=False)
        print(f"\nSaved budget table: {budget_path}")
        chosen_budget = budget_df[
            (budget_df["preprocess_mode"] == args.preprocess_mode)
            & (budget_df["scope"] == args.scope)
        ].iloc[0]
        print(chosen_budget.to_string())

        manifest, batch_df = _build_manifest(
            feature_records=feature_records,
            X_features=X_features,
            preprocess_meta=preprocess_meta,
            cell_data=cell_data,
            cell_ids=cell_ids,
            preprocess_mode=args.preprocess_mode,
            scope=args.scope,
            shots=args.shots,
            backend=args.backend,
            max_circuits_per_batch=args.max_circuits_per_batch,
        )
        manifest["run_label"] = run_label

        feature_records.to_csv(paths.feature_records_path, index=False)
        batch_df.to_csv(paths.batch_summary_path, index=False)
        manifest["feature_records_csv"] = str(paths.feature_records_path.relative_to(PROJECT_ROOT))
        manifest["batch_summary_csv"] = str(paths.batch_summary_path.relative_to(PROJECT_ROOT))

        with open(paths.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        meta_path = write_current_run_metadata(paths, manifest, updated_by="prepare_warwick_hardware")

        total_circuits = int(batch_df["n_circuits"].sum())
        total_qpu_sec = float(batch_df["estimated_qpu_sec"].sum())
        print(f"\nSaved feature records: {paths.feature_records_path}")
        print(f"Saved batch summary:  {paths.batch_summary_path}")
        print(f"Saved manifest:       {paths.manifest_path}")
        print(f"Saved stage pointer:  {meta_path}")
        print(f"Total circuits:       {total_circuits}")
        print(f"Estimated QPU time:   {total_qpu_sec:.1f}s ({total_qpu_sec/60.0:.1f} min)")
        print("\nPer-batch estimate:")
        print(batch_df.to_string(index=False))

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
