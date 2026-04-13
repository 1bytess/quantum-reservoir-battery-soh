"""Phase 6: IBM Quantum Hardware Validation (Stanford SECL pipeline).

Usage:
    cd src
    python run_phase_6.py --prepare                  # Build circuits offline (no QPU)
    python run_phase_6.py --run-account 1             # Run ACC1 batches (QPU $$$)
    python run_phase_6.py --run-account 2             # Run ACC2 batches (QPU $$$)
    python run_phase_6.py --run-account 3             # Run ACC3 batches (QPU $$$)
    python run_phase_6.py --run-batch 1               # Run single batch (QPU $$$)
    python run_phase_6.py --analyze                   # Process results offline
    python run_phase_6.py --validate-global-scaler    # Confirm global vs in-fold PCA

Global preprocessing strategy:
    A global PCA(6) + StandardScaler is fit on all 61 samples so each sample
    always gets the same circuit angles regardless of LOCO fold.  This means
    only 61 circuits per (depth, seed) config instead of 61 x 6 folds.
    Ridge readout is still trained fold-wise (LOCO) with group-aware inner CV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from config import (
    CELL_IDS,
    N_PCA,
    N_QUBITS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RIDGE_ALPHAS,
)
from data_loader import load_stanford_data
from phase_6.env_utils import load_phase6_env
from phase_6.paths import get_stage_paths

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase_4.circuit import build_qrc_circuit, encode_features
from phase_4.config import CLAMP_RANGE, ENCODING_METHOD, USE_ZZ_CORRELATORS

try:
    from qiskit.qasm2 import dumps as qasm2_dumps, loads as qasm2_loads
except ImportError:
    from qiskit import qasm2
    qasm2_dumps = qasm2.dumps
    qasm2_loads = qasm2.loads

# Load .env.local first, then .env as fallback
load_phase6_env(PROJECT_ROOT)


# ============================================================================
# Config
# ============================================================================

BACKEND_NAME = "ibm_marrakesh"
OPTIMIZATION_LEVEL = 1
STAGE_NAME = "stage_1"

BATCH_CONFIGS: Dict[int, dict] = {
    # Account 1: depth sweep (seed=42, 4096 shots)
    1: {"depth": 1, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    2: {"depth": 2, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    3: {"depth": 3, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    # Account 2: depth=4 + first two seed sweeps
    4: {"depth": 4, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    5: {"depth": 1, "shots": 4096, "seed": 43, "experiment": "seed_sweep"},
    6: {"depth": 1, "shots": 4096, "seed": 44, "experiment": "seed_sweep"},
    # Account 3: remaining seed sweeps
    7: {"depth": 1, "shots": 4096, "seed": 45, "experiment": "seed_sweep"},
    8: {"depth": 1, "shots": 4096, "seed": 46, "experiment": "seed_sweep"},
}

ACCOUNT_ASSIGNMENTS = {
    1: [1, 2, 3],       # ACC1 (~4.5 min QPU)
    2: [4, 5, 6],       # ACC2 (~4.5 min QPU)
    3: [7, 8],           # ACC3 (~3.0 min QPU)
}


# ============================================================================
# Paths
# ============================================================================

def _get_phase6_paths():
    """Return (data_dir, plot_dir, hardware_dir) for Phase 6."""
    paths = get_stage_paths(STAGE_NAME, include_hardware=True)
    return paths.data_dir, paths.plot_dir, paths.manifest_dir, paths.checkpoint_dir


# ============================================================================
# TeeLogger
# ============================================================================

class TeeLogger:
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


# ============================================================================
# IBM service
# ============================================================================

def _get_service(account: int):
    """Create QiskitRuntimeService from .env credentials."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    api_key = os.environ[f"IBM_ACC{account}_API"]
    crn = os.environ[f"IBM_ACC{account}_CRN"]
    return QiskitRuntimeService(channel="ibm_cloud", token=api_key, instance=crn)


def _account_for_batch(batch_id: int) -> int:
    for acct, batches in ACCOUNT_ASSIGNMENTS.items():
        if batch_id in batches:
            return acct
    raise ValueError(f"Batch {batch_id} not assigned to any account")


# ============================================================================
# Counts -> expectation values
# ============================================================================

def counts_to_expectations(
    counts: Dict[str, int],
    n_qubits: int = N_QUBITS,
    use_zz: bool = USE_ZZ_CORRELATORS,
) -> np.ndarray:
    """Convert measurement counts to Z / ZZ expectation values."""
    total_shots = sum(counts.values())
    expectations: List[float] = []

    padded: Dict[str, int] = {}
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        padded[bs.zfill(n_qubits)] = padded.get(bs.zfill(n_qubits), 0) + count

    for i in range(n_qubits):
        exp_val = 0.0
        for bitstring, count in padded.items():
            bit = int(bitstring[-(i + 1)])
            exp_val += (1 - 2 * bit) * count / total_shots
        expectations.append(exp_val)

    if use_zz:
        for i, j in combinations(range(n_qubits), 2):
            exp_val = 0.0
            for bitstring, count in padded.items():
                bit_i = int(bitstring[-(i + 1)])
                bit_j = int(bitstring[-(j + 1)])
                parity = (1 - 2 * bit_i) * (1 - 2 * bit_j)
                exp_val += parity * count / total_shots
            expectations.append(exp_val)

    return np.array(expectations)


# ============================================================================
# Load & preprocess (global PCA + scaler)
# ============================================================================

def _load_and_preprocess():
    """Load Stanford data, apply global PCA(6) + StandardScaler.

    Returns:
        X_scaled: (61, 6) globally scaled PCA features
        cell_ids: (61,) cell labels
        soh: (61,) SOH values
        scaler_params: dict with mean/scale for reproducibility
    """
    cell_data = load_stanford_data()

    X_raw_all = []
    cell_id_all = []
    soh_all = []
    for cid in CELL_IDS:
        n = cell_data[cid]["X_raw"].shape[0]
        X_raw_all.append(cell_data[cid]["X_raw"])
        cell_id_all.extend([cid] * n)
        soh_all.extend(cell_data[cid]["y"].tolist())

    X_raw = np.vstack(X_raw_all)  # (61, 38)
    cell_ids = np.array(cell_id_all)
    soh = np.array(soh_all)

    # Global PCA: 38D -> 6D
    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    X_6d = pca.fit_transform(X_raw)

    # Global StandardScaler
    scaler = StandardScaler().fit(X_6d)
    X_scaled = np.clip(scaler.transform(X_6d), CLAMP_RANGE[0], CLAMP_RANGE[1])

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    print(f"  Loaded {len(soh)} samples, PCA 38D -> {N_PCA}D")
    print(f"  PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"  Cells: {dict(zip(*np.unique(cell_ids, return_counts=True)))}")

    return X_scaled, cell_ids, soh, scaler_params


def _load_unified_stanford_summary() -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
    """Return the newest unified Stanford summary with qrc/xgboost/ridge, if present."""
    unified_dir = (
        PROJECT_ROOT / "result" / "manuscript_support" / "unified_loco" / "data"
    )
    if not unified_dir.exists():
        return None, None

    required_cols = {"dataset", "feature_variant", "model", "mae_mean"}
    required_models = {"qrc", "xgboost", "ridge"}
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

        if not required_cols.issubset(df.columns):
            continue

        subset = df[
            (df["dataset"] == "stanford") &
            (df["feature_variant"] == "native_pca6")
        ].copy()
        if required_models.issubset(set(subset["model"].astype(str))):
            return path, subset

    return None, None


# ============================================================================
# --prepare: build circuits offline
# ============================================================================

def prepare_circuits():
    """Build all circuits and save manifest + QASM to disk."""
    _, _, manifest_dir, _ = _get_phase6_paths()

    print("\n[prepare] Loading features ...")
    X_scaled, cell_ids, soh, scaler_params = _load_and_preprocess()
    n_samples = len(soh)

    angles_all = encode_features(X_scaled, method=ENCODING_METHOD)

    manifest = {
        "created": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_qubits": N_QUBITS,
        "cell_ids": cell_ids.tolist(),
        "soh": soh.tolist(),
        "scaler": scaler_params,
        "backend": BACKEND_NAME,
        "batches": {},
    }

    total_circuits = 0
    for batch_id, cfg in BATCH_CONFIGS.items():
        depth = cfg["depth"]
        seed = cfg["seed"]
        shots = cfg["shots"]

        rng = np.random.RandomState(seed)
        random_rotations = rng.uniform(0, 2 * np.pi, (depth, N_QUBITS, 3))

        qasm_list = []
        for idx in range(n_samples):
            qc = build_qrc_circuit(
                angles_all[idx],
                depth=depth,
                random_rotations=random_rotations,
            )
            qc.measure_all()
            qasm_list.append(qasm2_dumps(qc))

        manifest["batches"][str(batch_id)] = {
            "depth": depth,
            "seed": seed,
            "shots": shots,
            "experiment": cfg["experiment"],
            "sample_indices": list(range(n_samples)),
            "random_rotations": random_rotations.tolist(),
            "n_circuits": len(qasm_list),
            "qasm": qasm_list,
        }
        total_circuits += len(qasm_list)
        acct = _account_for_batch(batch_id)
        est_qpu = 2 + 0.00035 * (len(qasm_list) * shots)
        print(f"  Batch {batch_id} (ACC{acct}): d={depth} seed={seed} "
              f"shots={shots} circuits={len(qasm_list)} est_qpu={est_qpu:.0f}s")

    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[prepare] Saved manifest: {manifest_path}")
    print(f"  Total circuits: {total_circuits}")

    # Print QPU budget summary
    for acct, batch_ids in ACCOUNT_ASSIGNMENTS.items():
        total_est = sum(
            2 + 0.00035 * (61 * BATCH_CONFIGS[bid]["shots"])
            for bid in batch_ids
        )
        print(f"  ACC{acct}: batches {batch_ids} -> est {total_est:.0f}s ({total_est/60:.1f} min)")

    return manifest_path


# ============================================================================
# --run-batch / --run-account: submit to IBM hardware
# ============================================================================

def run_batch(batch_id: int, backend_name: str = BACKEND_NAME):
    """Execute a single batch on IBM Quantum hardware."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Batch, SamplerV2

    _, _, manifest_dir, checkpoint_dir = _get_phase6_paths()

    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}\nRun --prepare first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    batch_key = str(batch_id)
    if batch_key not in manifest["batches"]:
        raise ValueError(f"Batch {batch_id} not in manifest.")

    batch_cfg = manifest["batches"][batch_key]
    shots = batch_cfg["shots"]
    n_circuits = batch_cfg["n_circuits"]

    account = _account_for_batch(batch_id)
    print(f"\n[run_batch {batch_id}] Connecting to {backend_name} (ACC{account}) ...")
    service = _get_service(account)
    backend = service.backend(backend_name)
    print(f"  Backend: {backend.name}")

    # Rebuild circuits from QASM
    print(f"  Rebuilding {n_circuits} circuits from QASM ...")
    circuits = [qasm2_loads(q) for q in batch_cfg["qasm"]]

    # Transpile
    print(f"  Transpiling (optimization_level={OPTIMIZATION_LEVEL}) ...")
    pm = generate_preset_pass_manager(
        optimization_level=OPTIMIZATION_LEVEL,
        backend=backend,
    )
    isa_circuits = pm.run(circuits)

    est_qpu = 2 + 0.00035 * (n_circuits * shots)
    print(f"  Submitting {n_circuits} circuits ({shots} shots each) ...")
    print(f"  Estimated QPU time: {est_qpu:.0f}s ({est_qpu/60:.1f} min)")

    # Checkpoint skeleton (saved before waiting)
    checkpoint = {
        "batch_id": batch_id,
        "account": account,
        "backend": backend_name,
        "timestamp_start": datetime.now().isoformat(),
        "config": BATCH_CONFIGS[batch_id],
        "n_circuits": len(isa_circuits),
        "status": "submitted",
    }

    with Batch(backend=backend) as batch_ctx:
        sampler = SamplerV2(mode=batch_ctx)
        job = sampler.run(isa_circuits, shots=shots)
        checkpoint["job_id"] = job.job_id()

        # Save checkpoint immediately with job_id
        ckpt_path = checkpoint_dir / f"batch{batch_id}.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  Job submitted: {job.job_id()}")
        print(f"  Checkpoint saved: {ckpt_path}")

        # Wait for results
        print(f"  Waiting for results ...")
        t0 = time.time()
        result = job.result()
        wall_time = time.time() - t0

    # Extract counts
    sample_indices = batch_cfg["sample_indices"]
    depth = batch_cfg["depth"]
    seed = batch_cfg["seed"]
    all_counts = {}

    for i, pub_result in enumerate(result):
        creg = pub_result.data
        raw_counts = None
        for attr_name in dir(creg):
            attr = getattr(creg, attr_name)
            if hasattr(attr, "get_counts"):
                raw_counts = attr.get_counts()
                break
        if raw_counts is None:
            raw_counts = creg.meas.get_counts()

        sample_idx = sample_indices[i]
        key = f"d{depth}_s{seed}_sample{sample_idx}"
        all_counts[key] = {str(k): int(v) for k, v in raw_counts.items()}

    # Get QPU usage
    qpu_seconds = None
    try:
        usage = job.usage()
        qpu_seconds = getattr(usage, "quantum_seconds", None)
    except Exception:
        pass

    # Update checkpoint
    checkpoint["timestamp_end"] = datetime.now().isoformat()
    checkpoint["wall_time_sec"] = round(wall_time, 1)
    checkpoint["qpu_seconds"] = qpu_seconds
    checkpoint["counts"] = all_counts
    checkpoint["status"] = "completed"

    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n[run_batch {batch_id}] Complete!")
    print(f"  Wall time: {wall_time:.1f}s")
    if qpu_seconds is not None:
        print(f"  QPU time:  {qpu_seconds:.1f}s")
    print(f"  Counts saved: {len(all_counts)} entries")
    print(f"  Checkpoint: {ckpt_path}")

    return ckpt_path


def run_account(account: int, backend_name: str = BACKEND_NAME):
    """Run all batches assigned to an account sequentially."""
    batch_ids = ACCOUNT_ASSIGNMENTS[account]
    print(f"\n{'='*60}")
    print(f"Running Account {account}: batches {batch_ids}")
    print(f"{'='*60}")

    total_qpu = 0.0
    total_wall = 0.0

    for bid in batch_ids:
        ckpt_path = run_batch(bid, backend_name)

        with open(ckpt_path) as f:
            data = json.load(f)
        if data.get("qpu_seconds"):
            total_qpu += data["qpu_seconds"]
        if data.get("wall_time_sec"):
            total_wall += data["wall_time_sec"]

    print(f"\n{'='*60}")
    print(f"Account {account} complete!")
    print(f"  Total QPU time: {total_qpu:.1f}s ({total_qpu/60:.1f} min)")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"  Batches completed: {len(batch_ids)}")
    print(f"{'='*60}")


# ============================================================================
# --analyze: process counts into SOH predictions
# ============================================================================

def analyze_results():
    """Load checkpoint files, build reservoir features, evaluate LOCO + temporal."""
    data_dir, plot_dir, manifest_dir, checkpoint_dir = _get_phase6_paths()

    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    cell_ids = np.array(manifest["cell_ids"])
    soh = np.array(manifest["soh"])
    n_samples = manifest["n_samples"]

    ckpt_files = sorted(checkpoint_dir.glob("batch*.json"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {checkpoint_dir}")

    print(f"\n[analyze] Found {len(ckpt_files)} checkpoint files")

    # Parse all counts, grouped by (depth, seed)
    run_data: Dict[str, Dict] = {}
    for ckpt_path in ckpt_files:
        with open(ckpt_path) as f:
            ckpt = json.load(f)

        if ckpt.get("status") != "completed" or "counts" not in ckpt:
            print(f"  Skipping {ckpt_path.name} (status={ckpt.get('status')})")
            continue

        cfg = ckpt["config"]
        depth, seed, shots = cfg["depth"], cfg["seed"], cfg["shots"]
        run_key = f"d{depth}_s{seed}"

        if run_key not in run_data:
            run_data[run_key] = {
                "depth": depth,
                "seed": seed,
                "shots": shots,
                "experiment": cfg["experiment"],
                "expectations": {},
                "batch_id": ckpt["batch_id"],
                "job_id": ckpt.get("job_id", "unknown"),
                "qpu_seconds": ckpt.get("qpu_seconds"),
                "wall_time_sec": ckpt.get("wall_time_sec"),
            }

        for count_key, counts in ckpt["counts"].items():
            sample_idx = int(count_key.split("_sample")[1])
            exp_vals = counts_to_expectations(counts, N_QUBITS, USE_ZZ_CORRELATORS)
            run_data[run_key]["expectations"][sample_idx] = exp_vals

    print(f"  Parsed {len(run_data)} (depth, seed) configurations")

    # Evaluate
    exp_dim = N_QUBITS + (N_QUBITS * (N_QUBITS - 1) // 2 if USE_ZZ_CORRELATORS else 0)
    all_results = []

    for run_key, rd in sorted(run_data.items()):
        depth = rd["depth"]
        seed = rd["seed"]
        shots = rd["shots"]

        # Build feature matrix
        X_reservoir = np.full((n_samples, exp_dim), np.nan)
        for sample_idx, exp_vals in rd["expectations"].items():
            X_reservoir[sample_idx] = exp_vals

        valid_mask = ~np.isnan(X_reservoir[:, 0])
        n_valid = valid_mask.sum()
        print(f"  {run_key}: {n_valid}/{n_samples} samples")

        if n_valid < n_samples:
            print(f"    WARNING: Missing {n_samples - n_valid} samples")

        # --- LOCO evaluation ---
        for test_cell in CELL_IDS:
            train_cells = [c for c in CELL_IDS if c != test_cell]
            train_mask = np.isin(cell_ids, train_cells) & valid_mask
            test_mask = (cell_ids == test_cell) & valid_mask

            X_train = X_reservoir[train_mask]
            y_train = soh[train_mask]
            X_test = X_reservoir[test_mask]
            y_test = soh[test_mask]
            train_groups = cell_ids[train_mask]

            if len(y_train) == 0 or len(y_test) == 0:
                continue

            n_groups = len(np.unique(train_groups))
            if n_groups >= 2:
                cv = LeaveOneGroupOut()
                fit_kwargs = {"groups": train_groups}
            else:
                cv = min(3, len(y_train))
                fit_kwargs = {}

            grid = GridSearchCV(
                Ridge(), {"alpha": RIDGE_ALPHAS},
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train, **fit_kwargs)
            y_pred = grid.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")
            naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

            all_results.append({
                "stage": "hardware",
                "regime": "loco",
                "depth": depth,
                "seed": seed,
                "shots": shots,
                "experiment": rd["experiment"],
                "test_cell": test_cell,
                "train_cells": "+".join(train_cells),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "naive_mae": naive_mae,
                "beats_naive": bool(mae < naive_mae),
                "best_alpha": grid.best_params_["alpha"],
                "job_id": rd.get("job_id", ""),
                "qpu_seconds": rd.get("qpu_seconds"),
            })

        # --- Temporal evaluation ---
        for cid in CELL_IDS:
            cell_mask = (cell_ids == cid) & valid_mask
            indices = np.where(cell_mask)[0]
            if len(indices) < 3:
                continue

            X_cell = X_reservoir[indices]
            y_cell = soh[indices]
            n_total = len(y_cell)
            n_train = max(2, int(n_total * 0.7))
            n_train = min(n_train, n_total - 1)

            X_tr, X_te = X_cell[:n_train], X_cell[n_train:]
            y_tr, y_te = y_cell[:n_train], y_cell[n_train:]

            if len(y_te) == 0:
                continue

            grid = GridSearchCV(
                Ridge(), {"alpha": RIDGE_ALPHAS},
                cv=min(3, len(y_tr)),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X_tr, y_tr)
            y_pred = grid.predict(X_te)

            mae = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2 = r2_score(y_te, y_pred) if len(y_te) > 1 else float("nan")
            persist_mae = mean_absolute_error(y_te, np.full_like(y_te, y_tr[-1]))

            all_results.append({
                "stage": "hardware",
                "regime": "temporal",
                "depth": depth,
                "seed": seed,
                "shots": shots,
                "experiment": rd["experiment"],
                "test_cell": cid,
                "train_cells": "",
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "naive_mae": float("nan"),
                "beats_naive": float("nan"),
                "persist_mae": persist_mae,
                "beats_persist": bool(mae < persist_mae),
                "best_alpha": grid.best_params_["alpha"],
                "job_id": rd.get("job_id", ""),
                "qpu_seconds": rd.get("qpu_seconds"),
            })

    results_df = pd.DataFrame(all_results)
    csv_path = data_dir / "qrc_hardware.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[analyze] Saved {len(results_df)} rows to {csv_path}")

    # Summary tables
    if not results_df.empty:
        loco = results_df[results_df["regime"] == "loco"]
        if not loco.empty:
            print("\n  LOCO summary (by depth, seed):")
            summary = loco.groupby(["depth", "seed"])["mae"].agg(["mean", "std", "count"])
            print(summary.round(4).to_string())

            # Per-cell table for best config
            best_idx = loco.groupby(["depth", "seed"])["mae"].mean().idxmin()
            best_depth, best_seed = best_idx
            best_rows = loco[(loco["depth"] == best_depth) & (loco["seed"] == best_seed)]
            print(f"\n  Best hardware config: depth={best_depth}, seed={best_seed}")
            print(f"  Per-cell MAE:")
            for _, row in best_rows.iterrows():
                beat = "YES" if row["beats_naive"] else "no"
                print(f"    {row['test_cell']}: MAE={row['mae']:.4f} "
                      f"(naive={row['naive_mae']:.4f}, beats={beat})")

        temporal = results_df[results_df["regime"] == "temporal"]
        if not temporal.empty:
            print("\n  Temporal summary (by depth, seed):")
            t_summary = temporal.groupby(["depth", "seed"])["mae"].agg(["mean", "count"])
            print(t_summary.round(4).to_string())

    # Hardware vs simulator comparison
    _build_hardware_comparison(results_df, data_dir)

    # Plots
    _plot_hardware_depth_sweep(results_df, plot_dir)
    _plot_hardware_vs_simulator(results_df, data_dir, plot_dir)
    _plot_hardware_seed_variance(results_df, plot_dir)

    return results_df


def _build_hardware_comparison(hw_df: pd.DataFrame, data_dir: Path):
    """Compare hardware QRC results against simulator + classical baselines."""
    rows = []

    # Hardware best (LOCO)
    loco = hw_df[hw_df["regime"] == "loco"]
    if not loco.empty:
        by_config = loco.groupby(["depth", "seed"])["mae"].mean()
        best_idx = by_config.idxmin()
        rows.append({
            "method": f"qrc_hardware_d{best_idx[0]}_s{best_idx[1]}",
            "mae_mean": float(by_config.loc[best_idx]),
            "source": "phase6_hardware",
        })

        # All seed average at depth=1
        d1_seeds = loco[loco["depth"] == 1]
        if not d1_seeds.empty:
            d1_mean = d1_seeds.groupby("seed")["mae"].mean()
            rows.append({
                "method": "qrc_hardware_d1_mean_seeds",
                "mae_mean": float(d1_mean.mean()),
                "source": "phase6_hardware",
            })

    # Prefer standardized unified baselines; fall back to legacy phase_4 outputs.
    unified_path, unified_df = _load_unified_stanford_summary()
    if unified_df is not None:
        for _, row in unified_df.iterrows():
            rows.append({
                "method": f"{row['model']}_unified_stanford",
                "mae_mean": float(row["mae_mean"]),
                "source": f"unified_loco:{unified_path.name}",
            })
    else:
        sim_path = PROJECT_ROOT / "result" / "phase_4" / "data" / "qrc_vs_classical.csv"
        if sim_path.exists():
            sim_df = pd.read_csv(sim_path)
            for _, row in sim_df.iterrows():
                rows.append({
                    "method": row["method"],
                    "mae_mean": row["mae_mean"],
                    "source": row["source"],
                })

    if rows:
        comp_df = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
        comp_path = data_dir / "hardware_vs_all.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\n[analyze] Hardware comparison saved: {comp_path}")
        print(comp_df.to_string(index=False))


# ============================================================================
# Plots
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_fig(fig, plot_dir, stem):
    for ext in ["png", "pdf"]:
        path = plot_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    plt.close(fig)


def _plot_hardware_depth_sweep(df: pd.DataFrame, plot_dir: Path):
    loco = df[(df["regime"] == "loco") & (df["experiment"] == "depth_sweep")]
    if loco.empty:
        return
    summary = loco.groupby("depth")["mae"].agg(["mean", "std"]).sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
                marker="o", linewidth=2, capsize=5, label="Hardware LOCO")

    # Overlay simulator results
    sim_path = PROJECT_ROOT / "result" / "phase_4" / "data" / "qrc_noiseless.csv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path)
        sim_loco = sim[sim["regime"] == "loco"].groupby("depth")["mae"].mean().sort_index()
        ax.plot(sim_loco.index, sim_loco.values, marker="s", linewidth=2,
                linestyle="--", label="Simulator (noiseless)")

    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("Hardware vs Simulator Depth Sweep")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    _save_fig(fig, plot_dir, "hardware_depth_sweep")


def _plot_hardware_vs_simulator(df: pd.DataFrame, data_dir: Path, plot_dir: Path):
    comp_path = data_dir / "hardware_vs_all.csv"
    if not comp_path.exists():
        return
    comp = pd.read_csv(comp_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for src in comp["source"]:
        if "hardware" in src:
            colors.append("#e74c3c")
        elif "qrc" in str(comp[comp["source"] == src]["method"].iloc[0]):
            colors.append("#3498db")
        else:
            colors.append("#95a5a6")

    x = np.arange(len(comp))
    ax.bar(x, comp["mae_mean"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["method"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("Hardware QRC vs Simulator vs Classical")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_fig(fig, plot_dir, "hardware_vs_all")


def _plot_hardware_seed_variance(df: pd.DataFrame, plot_dir: Path):
    seed_rows = df[(df["regime"] == "loco") & (df["depth"] == 1)]
    if seed_rows.empty or seed_rows["seed"].nunique() < 2:
        return
    per_seed = seed_rows.groupby("seed")["mae"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(per_seed)), per_seed.values, tick_label=[f"seed={s}" for s in per_seed.index])
    ax.axhline(per_seed.mean(), color="red", linestyle="--", label=f"mean={per_seed.mean():.4f}")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("Hardware QRC Seed Variance (depth=1)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_fig(fig, plot_dir, "hardware_seed_variance")


# ============================================================================
# --validate-global-scaler
# ============================================================================

def validate_global_scaler():
    """Compare global PCA+scaler vs in-fold PCA+scaler on noiseless QRC."""
    from phase_4.circuit import compute_reservoir_features

    print("\n[validate_global_scaler] Loading data ...")
    cell_data = load_stanford_data()

    X_raw_all = []
    cell_id_all = []
    soh_all = []
    for cid in CELL_IDS:
        n = cell_data[cid]["X_raw"].shape[0]
        X_raw_all.append(cell_data[cid]["X_raw"])
        cell_id_all.extend([cid] * n)
        soh_all.extend(cell_data[cid]["y"].tolist())

    X_raw = np.vstack(X_raw_all)
    cell_ids = np.array(cell_id_all)
    soh = np.array(soh_all)

    depth = 1  # Use best depth
    rng = np.random.RandomState(RANDOM_STATE)
    random_rotations = rng.uniform(0, 2 * np.pi, (depth, N_QUBITS, 3))

    # --- Global PCA+scaler approach (Phase 6 strategy) ---
    pca_global = PCA(n_components=N_PCA, random_state=RANDOM_STATE).fit(X_raw)
    X_6d_global = pca_global.transform(X_raw)
    scaler_global = StandardScaler().fit(X_6d_global)
    X_scaled_global = np.clip(scaler_global.transform(X_6d_global), CLAMP_RANGE[0], CLAMP_RANGE[1])

    reservoir_global = compute_reservoir_features(
        X_scaled_global, depth=depth, use_zz=USE_ZZ_CORRELATORS,
        random_rotations=random_rotations,
    )

    global_maes = []
    for test_cell in CELL_IDS:
        train_mask = cell_ids != test_cell
        test_mask = cell_ids == test_cell
        train_groups = cell_ids[train_mask]
        grid = GridSearchCV(
            Ridge(), {"alpha": RIDGE_ALPHAS},
            cv=LeaveOneGroupOut(),
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        grid.fit(
            reservoir_global[train_mask],
            soh[train_mask],
            groups=train_groups,
        )
        y_pred = grid.predict(reservoir_global[test_mask])
        global_maes.append(mean_absolute_error(soh[test_mask], y_pred))

    mae_global = np.mean(global_maes)

    # --- In-fold PCA+scaler approach (Phase 4 strategy) ---
    infold_maes = []
    for test_cell in CELL_IDS:
        train_mask = cell_ids != test_cell
        test_mask = cell_ids == test_cell
        train_groups = cell_ids[train_mask]

        pca_fold = PCA(n_components=N_PCA, random_state=RANDOM_STATE).fit(X_raw[train_mask])
        X_tr_6d = pca_fold.transform(X_raw[train_mask])
        X_te_6d = pca_fold.transform(X_raw[test_mask])

        scaler_fold = StandardScaler().fit(X_tr_6d)
        X_tr_scaled = np.clip(scaler_fold.transform(X_tr_6d), CLAMP_RANGE[0], CLAMP_RANGE[1])
        X_te_scaled = np.clip(scaler_fold.transform(X_te_6d), CLAMP_RANGE[0], CLAMP_RANGE[1])

        rng2 = np.random.RandomState(RANDOM_STATE)
        rr2 = rng2.uniform(0, 2 * np.pi, (depth, N_QUBITS, 3))

        res_tr = compute_reservoir_features(
            X_tr_scaled, depth=depth, use_zz=USE_ZZ_CORRELATORS, random_rotations=rr2,
        )
        res_te = compute_reservoir_features(
            X_te_scaled, depth=depth, use_zz=USE_ZZ_CORRELATORS, random_rotations=rr2,
        )

        grid = GridSearchCV(
            Ridge(), {"alpha": RIDGE_ALPHAS},
            cv=LeaveOneGroupOut(),
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        grid.fit(res_tr, soh[train_mask], groups=train_groups)
        y_pred = grid.predict(res_te)
        infold_maes.append(mean_absolute_error(soh[test_mask], y_pred))

    mae_infold = np.mean(infold_maes)
    pct_diff = abs(mae_global - mae_infold) / mae_infold * 100

    print(f"\n  Global PCA+scaler MAE:  {mae_global:.4f}")
    print(f"  In-fold PCA+scaler MAE: {mae_infold:.4f}")
    print(f"  Difference:             {pct_diff:.1f}%")
    if pct_diff < 10.0:
        print(f"  PASS: Global scaler is within 10% of in-fold scaler")
    else:
        print(f"  WARNING: Global scaler differs by {pct_diff:.1f}%")

    # ---- Save per-cell breakdown to CSV (Reviewer $2.4) ----
    data_dir, plot_dir, _, _ = _get_phase6_paths()

    rows = []
    for i, cid in enumerate(CELL_IDS):
        rows.append({
            "cell": cid,
            "mae_global_pca": global_maes[i],
            "mae_infold_pca": infold_maes[i],
            "delta": global_maes[i] - infold_maes[i],
            "delta_pct": (global_maes[i] - infold_maes[i]) / infold_maes[i] * 100,
        })
    rows.append({
        "cell": "MEAN",
        "mae_global_pca": mae_global,
        "mae_infold_pca": mae_infold,
        "delta": mae_global - mae_infold,
        "delta_pct": pct_diff if mae_global >= mae_infold else -pct_diff,
    })

    leakage_df = pd.DataFrame(rows)
    csv_path = data_dir / "pca_leakage_quantification.csv"
    leakage_df.to_csv(csv_path, index=False)
    print(f"\n  Per-cell leakage breakdown saved: {csv_path}")
    print(leakage_df.to_string(index=False))

    # ---- Comparison bar plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CELL_IDS))
    w = 0.35
    ax.bar(x - w / 2, global_maes, w, label="Global PCA (hardware strategy)", color="#e74c3c", alpha=0.85)
    ax.bar(x + w / 2, infold_maes, w, label="In-fold PCA (leakage-free)", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_IDS)
    ax.set_xlabel("Test Cell")
    ax.set_ylabel("LOCO MAE")
    ax.set_title(f"PCA Leakage Impact: Global vs In-fold (delta={pct_diff:.1f}%)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_fig(fig, plot_dir, "pca_leakage_comparison")

    return {"mae_global": mae_global, "mae_infold": mae_infold, "pct_diff": pct_diff,
            "per_cell": leakage_df}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 6: IBM Quantum Hardware Validation")
    parser.add_argument("--prepare", action="store_true", help="Build circuits offline")
    parser.add_argument("--run-account", type=int, metavar="N", help="Run all batches for account N")
    parser.add_argument("--run-batch", type=int, metavar="N", help="Run single batch N")
    parser.add_argument("--analyze", action="store_true", help="Process results offline")
    parser.add_argument("--validate-global-scaler", action="store_true")
    parser.add_argument("--backend", type=str, default=BACKEND_NAME)
    args = parser.parse_args()

    if not any([args.prepare, args.run_account is not None, args.run_batch is not None,
                args.analyze, args.validate_global_scaler]):
        parser.print_help()
        return

    data_dir, _, _, _ = _get_phase6_paths()
    log_path = data_dir / "stage_1_log.txt"
    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 6: IBM Quantum Hardware Validation (Stanford SECL)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Backend: {args.backend}")

        if args.prepare:
            print("\n[MODE] Preparing circuits (offline) ...")
            prepare_circuits()

        if args.validate_global_scaler:
            validate_global_scaler()

        if args.run_account is not None:
            if args.run_account not in ACCOUNT_ASSIGNMENTS:
                print(f"ERROR: Account {args.run_account} not found. "
                      f"Available: {sorted(ACCOUNT_ASSIGNMENTS.keys())}")
            else:
                run_account(args.run_account, args.backend)

        if args.run_batch is not None:
            if args.run_batch not in BATCH_CONFIGS:
                print(f"ERROR: Batch {args.run_batch} not found. "
                      f"Available: {sorted(BATCH_CONFIGS.keys())}")
            else:
                run_batch(args.run_batch, args.backend)

        if args.analyze:
            print("\n[MODE] Analyzing results (offline) ...")
            analyze_results()

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
