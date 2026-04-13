"""Phase 6b: IBM Fez Hardware Validation — Stanford SECL.

Mirrors Phase 6 (ibm_marrakesh) but targets ibm_fez, writes to result/phase_6/stage_2/,
and adds --compare mode to produce side-by-side fez vs marrakesh tables + plots.

Usage:
    cd src
    python run_phase_6b.py --prepare                  # Build circuits offline (no QPU)
    python run_phase_6b.py --run-account 1             # Run ACC1 batches on ibm_fez
    python run_phase_6b.py --run-account 2             # Run ACC2 batches on ibm_fez
    python run_phase_6b.py --run-batch 1               # Run single batch (QPU $$$)
    python run_phase_6b.py --analyze                   # Process ibm_fez results offline
    python run_phase_6b.py --compare                   # Fez vs Marrakesh comparison

Budget:  ~9 min QPU across 2 accounts (ACC1 + ACC2, no ACC3 needed)
         Batch 1-3 → ACC1 (~4.5 min)
         Batch 4-6 → ACC2 (~4.5 min)

ibm_fez specs (2026-03-13 dashboard):
    CZ error  2.688e-3   (better than marrakesh 3.3e-3, -19%)
    SX error  2.643e-4   (slightly worse, +15%)
    Readout   1.54e-2    (worse than marrakesh 0.95e-2, +62%)
    T1        142.56 µs  (shorter)
    T2        100.11 µs  (shorter)
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from config import (
    CELL_IDS,
    FEZ_NOISE,
    MARRAKESH_NOISE,
    N_PCA,
    N_QUBITS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RIDGE_ALPHAS,
)
from data_loader import load_stanford_data
from phase_6.env_utils import load_phase6_env
from phase_6.paths import Phase6StagePaths, get_stage_paths

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
# Config — ibm_fez specific
# ============================================================================

BACKEND_NAME   = "ibm_fez"           # target device
PHASE_LABEL    = "stage_2"           # output folder
COMPARE_STAGE  = "stage_1"           # marrakesh results to compare against
OPTIMIZATION_LEVEL = 1

# 6 batches across 2 accounts → ~9 min total QPU
# Each batch: 61 circuits × 4096 shots ≈ 89s QPU + overhead ≈ 1.5 min
BATCH_CONFIGS: Dict[int, dict] = {
    # ACC1: depth sweep d1–d3 (seed=42, 4096 shots)
    1: {"depth": 1, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    2: {"depth": 2, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    3: {"depth": 3, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    # ACC2: depth d4 + seed sweep d1
    4: {"depth": 4, "shots": 4096, "seed": 42, "experiment": "depth_sweep"},
    5: {"depth": 1, "shots": 4096, "seed": 43, "experiment": "seed_sweep"},
    6: {"depth": 1, "shots": 4096, "seed": 44, "experiment": "seed_sweep"},
}

ACCOUNT_ASSIGNMENTS = {
    1: [1, 2, 3],   # ACC1 → ~4.5 min QPU
    2: [4, 5, 6],   # ACC2 → ~4.5 min QPU
}


# ============================================================================
# Paths
# ============================================================================

def _get_paths():
    """Return (data_dir, plot_dir, manifest_dir, checkpoint_dir) for stage 2."""
    paths = get_stage_paths(PHASE_LABEL, include_hardware=True)
    return paths.data_dir, paths.plot_dir, paths.manifest_dir, paths.checkpoint_dir


def _get_marrakesh_paths():
    """Paths to Phase 6 (ibm_marrakesh) results for comparison."""
    base = Phase6StagePaths(COMPARE_STAGE)
    return (
        base.data_dir / "qrc_hardware.csv",
        base.manifest_dir / "manifest.json",
    )


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
    crn     = os.environ[f"IBM_ACC{account}_CRN"]
    return QiskitRuntimeService(channel="ibm_cloud", token=api_key, instance=crn)


def _account_for_batch(batch_id: int) -> int:
    for acct, batches in ACCOUNT_ASSIGNMENTS.items():
        if batch_id in batches:
            return acct
    raise ValueError(f"Batch {batch_id} not assigned to any account")


# ============================================================================
# Counts → expectation values
# ============================================================================

def counts_to_expectations(
    counts: Dict[str, int],
    n_qubits: int = N_QUBITS,
    use_zz: bool = USE_ZZ_CORRELATORS,
) -> np.ndarray:
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
# Load & preprocess (global PCA + scaler — identical to Phase 6)
# ============================================================================

def _load_and_preprocess():
    """Load Stanford data, apply global PCA(6) + StandardScaler.
    Uses the SAME global scaler as Phase 6 for direct circuit-level comparability.
    """
    cell_data = load_stanford_data()

    X_raw_all, cell_id_all, soh_all = [], [], []
    for cid in CELL_IDS:
        n = cell_data[cid]["X_raw"].shape[0]
        X_raw_all.append(cell_data[cid]["X_raw"])
        cell_id_all.extend([cid] * n)
        soh_all.extend(cell_data[cid]["y"].tolist())

    X_raw    = np.vstack(X_raw_all)
    cell_ids = np.array(cell_id_all)
    soh      = np.array(soh_all)

    pca     = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    X_6d    = pca.fit_transform(X_raw)
    scaler  = StandardScaler().fit(X_6d)
    X_scaled = np.clip(scaler.transform(X_6d), CLAMP_RANGE[0], CLAMP_RANGE[1])

    scaler_params = {
        "mean":                      scaler.mean_.tolist(),
        "scale":                     scaler.scale_.tolist(),
        "pca_components":            pca.components_.tolist(),
        "pca_mean":                  pca.mean_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    print(f"  Loaded {len(soh)} samples, PCA 38D -> {N_PCA}D")
    print(f"  PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"  Cells: {dict(zip(*np.unique(cell_ids, return_counts=True)))}")

    return X_scaled, cell_ids, soh, scaler_params


def _load_unified_stanford_summary() -> tuple[Optional[Path], Optional[pd.DataFrame]]:
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
    """Build all circuits and save manifest + QASM to disk (no QPU required)."""
    _, _, manifest_dir, _ = _get_paths()

    print("\n[prepare] Loading features ...")
    X_scaled, cell_ids, soh, scaler_params = _load_and_preprocess()
    n_samples = len(soh)
    angles_all = encode_features(X_scaled, method=ENCODING_METHOD)

    manifest = {
        "created":   datetime.now().isoformat(),
        "phase":     PHASE_LABEL,
        "backend":   BACKEND_NAME,
        "n_samples": n_samples,
        "n_qubits":  N_QUBITS,
        "cell_ids":  cell_ids.tolist(),
        "soh":       soh.tolist(),
        "scaler":    scaler_params,
        "fez_noise_params": FEZ_NOISE,
        "batches":   {},
    }

    total_circuits = 0
    for batch_id, cfg in BATCH_CONFIGS.items():
        depth = cfg["depth"]
        seed  = cfg["seed"]
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

        est_qpu = 2 + 0.00035 * (len(qasm_list) * shots)
        acct    = _account_for_batch(batch_id)

        manifest["batches"][str(batch_id)] = {
            "depth":          depth,
            "seed":           seed,
            "shots":          shots,
            "experiment":     cfg["experiment"],
            "sample_indices": list(range(n_samples)),
            "random_rotations": random_rotations.tolist(),
            "n_circuits":     len(qasm_list),
            "qasm":           qasm_list,
        }
        total_circuits += len(qasm_list)
        print(f"  Batch {batch_id} (ACC{acct}): d={depth} seed={seed} "
              f"shots={shots} circuits={len(qasm_list)} est_qpu={est_qpu:.0f}s")

    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[prepare] Saved manifest: {manifest_path}")
    print(f"  Total circuits: {total_circuits}")

    # Budget summary
    print(f"\n  QPU budget estimate:")
    for acct, batch_ids in ACCOUNT_ASSIGNMENTS.items():
        total_est = sum(
            2 + 0.00035 * (61 * BATCH_CONFIGS[bid]["shots"])
            for bid in batch_ids
        )
        print(f"  ACC{acct}: batches {batch_ids} -> est {total_est:.0f}s ({total_est/60:.1f} min)")

    grand_total = sum(
        2 + 0.00035 * (61 * cfg["shots"])
        for cfg in BATCH_CONFIGS.values()
    )
    print(f"  TOTAL: {grand_total:.0f}s ({grand_total/60:.1f} min) across both accounts")
    return manifest_path


# ============================================================================
# --run-batch / --run-account: submit to ibm_fez
# ============================================================================

def run_batch(batch_id: int, backend_name: str = BACKEND_NAME):
    """Execute a single batch on IBM Quantum (ibm_fez)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Batch, SamplerV2

    _, _, manifest_dir, checkpoint_dir = _get_paths()
    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}\nRun --prepare first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    batch_key = str(batch_id)
    if batch_key not in manifest["batches"]:
        raise ValueError(f"Batch {batch_id} not in manifest.")

    batch_cfg = manifest["batches"][batch_key]
    shots      = batch_cfg["shots"]
    n_circuits = batch_cfg["n_circuits"]

    account = _account_for_batch(batch_id)
    print(f"\n[run_batch {batch_id}] Connecting to {backend_name} (ACC{account}) ...")
    service = _get_service(account)
    backend = service.backend(backend_name)
    print(f"  Backend: {backend.name}")

    print(f"  Rebuilding {n_circuits} circuits from QASM ...")
    circuits = [qasm2_loads(q) for q in batch_cfg["qasm"]]

    print(f"  Transpiling (optimization_level={OPTIMIZATION_LEVEL}) ...")
    pm = generate_preset_pass_manager(
        optimization_level=OPTIMIZATION_LEVEL,
        backend=backend,
    )
    isa_circuits = pm.run(circuits)

    est_qpu = 2 + 0.00035 * (n_circuits * shots)
    print(f"  Submitting {n_circuits} circuits ({shots} shots each) ...")
    print(f"  Estimated QPU time: {est_qpu:.0f}s ({est_qpu/60:.1f} min)")

    checkpoint = {
        "batch_id":        batch_id,
        "account":         account,
        "backend":         backend_name,
        "timestamp_start": datetime.now().isoformat(),
        "config":          BATCH_CONFIGS[batch_id],
        "n_circuits":      len(isa_circuits),
        "status":          "submitted",
    }

    with Batch(backend=backend) as batch_ctx:
        sampler = SamplerV2(mode=batch_ctx)
        job     = sampler.run(isa_circuits, shots=shots)
        checkpoint["job_id"] = job.job_id()

        ckpt_path = checkpoint_dir / f"batch{batch_id}.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  Job submitted: {job.job_id()}")
        print(f"  Checkpoint saved: {ckpt_path}")

        print(f"  Waiting for results ...")
        t0     = time.time()
        result = job.result()
        wall_time = time.time() - t0

    sample_indices = batch_cfg["sample_indices"]
    depth = batch_cfg["depth"]
    seed  = batch_cfg["seed"]
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
        key        = f"d{depth}_s{seed}_sample{sample_idx}"
        all_counts[key] = {str(k): int(v) for k, v in raw_counts.items()}

    qpu_seconds = None
    try:
        usage = job.usage()
        qpu_seconds = getattr(usage, "quantum_seconds", None)
    except Exception:
        pass

    checkpoint["timestamp_end"] = datetime.now().isoformat()
    checkpoint["wall_time_sec"] = round(wall_time, 1)
    checkpoint["qpu_seconds"]   = qpu_seconds
    checkpoint["counts"]        = all_counts
    checkpoint["status"]        = "completed"

    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n[run_batch {batch_id}] Complete!")
    print(f"  Wall time: {wall_time:.1f}s")
    if qpu_seconds is not None:
        print(f"  QPU time:  {qpu_seconds:.1f}s")
    print(f"  Counts saved: {len(all_counts)} entries")
    return ckpt_path


def run_account(account: int, backend_name: str = BACKEND_NAME):
    """Run all batches for an account sequentially."""
    batch_ids = ACCOUNT_ASSIGNMENTS[account]
    print(f"\n{'='*60}")
    print(f"Running Account {account}: batches {batch_ids} on {backend_name}")
    print(f"{'='*60}")

    total_qpu = 0.0
    total_wall = 0.0
    for bid in batch_ids:
        ckpt_path = run_batch(bid, backend_name)
        with open(ckpt_path) as f:
            data = json.load(f)
        if data.get("qpu_seconds"):
            total_qpu  += data["qpu_seconds"]
        if data.get("wall_time_sec"):
            total_wall += data["wall_time_sec"]

    print(f"\n{'='*60}")
    print(f"Account {account} complete!")
    print(f"  Total QPU time:  {total_qpu:.1f}s ({total_qpu/60:.1f} min)")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"{'='*60}")


# ============================================================================
# --analyze: process counts → SOH predictions
# ============================================================================

def analyze_results():
    """Load checkpoint files, evaluate LOCO, save qrc_hardware_fez.csv."""
    data_dir, plot_dir, manifest_dir, checkpoint_dir = _get_paths()

    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    cell_ids  = np.array(manifest["cell_ids"])
    soh       = np.array(manifest["soh"])
    n_samples = manifest["n_samples"]

    ckpt_files = sorted(checkpoint_dir.glob("batch*.json"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {checkpoint_dir}")

    print(f"\n[analyze] Found {len(ckpt_files)} checkpoint files")

    run_data: Dict[str, Dict] = {}
    for ckpt_path in ckpt_files:
        with open(ckpt_path) as f:
            ckpt = json.load(f)

        if ckpt.get("status") != "completed" or "counts" not in ckpt:
            print(f"  Skipping {ckpt_path.name} (status={ckpt.get('status')})")
            continue

        cfg   = ckpt["config"]
        depth = cfg["depth"]
        seed  = cfg["seed"]
        shots = cfg["shots"]
        run_key = f"d{depth}_s{seed}"

        if run_key not in run_data:
            run_data[run_key] = {
                "depth":        depth,
                "seed":         seed,
                "shots":        shots,
                "experiment":   cfg["experiment"],
                "expectations": {},
                "batch_id":     ckpt["batch_id"],
                "job_id":       ckpt.get("job_id", "unknown"),
                "qpu_seconds":  ckpt.get("qpu_seconds"),
                "wall_time_sec": ckpt.get("wall_time_sec"),
            }

        for count_key, counts in ckpt["counts"].items():
            sample_idx = int(count_key.split("_sample")[1])
            exp_vals   = counts_to_expectations(counts, N_QUBITS, USE_ZZ_CORRELATORS)
            run_data[run_key]["expectations"][sample_idx] = exp_vals

    print(f"  Parsed {len(run_data)} (depth, seed) configurations")

    exp_dim  = N_QUBITS + (N_QUBITS * (N_QUBITS - 1) // 2 if USE_ZZ_CORRELATORS else 0)
    all_results = []

    for run_key, rd in sorted(run_data.items()):
        depth = rd["depth"]
        seed  = rd["seed"]
        shots = rd["shots"]

        X_reservoir = np.full((n_samples, exp_dim), np.nan)
        for sample_idx, exp_vals in rd["expectations"].items():
            X_reservoir[sample_idx] = exp_vals

        valid_mask = ~np.isnan(X_reservoir[:, 0])
        n_valid    = valid_mask.sum()
        print(f"  {run_key}: {n_valid}/{n_samples} samples")

        for test_cell in CELL_IDS:
            train_cells = [c for c in CELL_IDS if c != test_cell]
            train_mask  = np.isin(cell_ids, train_cells) & valid_mask
            test_mask   = (cell_ids == test_cell) & valid_mask

            X_train = X_reservoir[train_mask]
            y_train = soh[train_mask]
            X_test  = X_reservoir[test_mask]
            y_test  = soh[test_mask]
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

            mae      = mean_absolute_error(y_test, y_pred)
            rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
            r2       = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")
            naive_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

            all_results.append({
                "backend":     BACKEND_NAME,
                "stage":       "hardware",
                "regime":      "loco",
                "depth":       depth,
                "seed":        seed,
                "shots":       shots,
                "experiment":  rd["experiment"],
                "test_cell":   test_cell,
                "train_cells": "+".join(train_cells),
                "n_train":     int(len(y_train)),
                "n_test":      int(len(y_test)),
                "mae":         mae,
                "rmse":        rmse,
                "r2":          r2,
                "naive_mae":   naive_mae,
                "beats_naive": bool(mae < naive_mae),
                "best_alpha":  grid.best_params_["alpha"],
                "job_id":      rd.get("job_id", ""),
                "qpu_seconds": rd.get("qpu_seconds"),
            })

    results_df = pd.DataFrame(all_results)
    csv_path   = data_dir / "qrc_hardware_fez.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[analyze] Saved {len(results_df)} rows to {csv_path}")

    # Summary
    if not results_df.empty:
        loco = results_df[results_df["regime"] == "loco"]
        if not loco.empty:
            print("\n  LOCO summary (by depth, seed):")
            summary = loco.groupby(["depth", "seed"])["mae"].agg(["mean", "std", "count"])
            print(summary.round(5).to_string())

            best_idx          = loco.groupby(["depth", "seed"])["mae"].mean().idxmin()
            best_depth, best_seed = best_idx
            best_rows = loco[(loco["depth"] == best_depth) & (loco["seed"] == best_seed)]
            print(f"\n  Best config: depth={best_depth}, seed={best_seed}")
            for _, row in best_rows.iterrows():
                beat = "YES" if row["beats_naive"] else "no"
                print(f"    {row['test_cell']}: MAE={row['mae']:.5f} "
                      f"(naive={row['naive_mae']:.5f}, beats={beat})")

    _plot_fez_results(results_df, plot_dir)
    return results_df


# ============================================================================
# --compare: fez vs marrakesh side-by-side
# ============================================================================

def compare_backends():
    """Load phase_6 (marrakesh) and phase_6b (fez), print + plot comparison."""
    data_dir, plot_dir, _, _ = _get_paths()

    fez_path = data_dir / "qrc_hardware_fez.csv"
    mar_path, _ = _get_marrakesh_paths()
    sim_path = PROJECT_ROOT / "result" / "phase_4" / "data" / "qrc_vs_classical.csv"

    if not fez_path.exists():
        print(f"ERROR: ibm_fez results not found at {fez_path}")
        print("  Run --analyze first after hardware runs complete.")
        return

    fez_df = pd.read_csv(fez_path)
    fez_df["backend"] = "ibm_fez"

    if not mar_path.exists():
        print(f"WARNING: ibm_marrakesh results not found at {mar_path}")
        print("  Showing ibm_fez results only.")
        mar_df = pd.DataFrame()
    else:
        mar_df = pd.read_csv(mar_path)
        mar_df["backend"] = "ibm_marrakesh"

    # --- Combine LOCO results ---
    frames = []
    for df in [fez_df, mar_df]:
        if df.empty:
            continue
        loco = df[df["regime"] == "loco"] if "regime" in df.columns else df
        if not loco.empty:
            frames.append(loco)

    if not frames:
        print("ERROR: No LOCO results to compare.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Per-backend summary (mean LOCO MAE by depth)
    print("\n" + "=" * 65)
    print("  ibm_fez vs ibm_marrakesh — LOCO MAE by depth")
    print("=" * 65)
    summary = (
        combined.groupby(["backend", "depth"])["mae"]
        .agg(["mean", "std"])
        .round(5)
        .reset_index()
    )
    print(summary.to_string(index=False))

    # Best-config comparison
    print("\n  Best configuration per backend (lowest mean LOCO MAE):")
    for backend_name, grp in combined.groupby("backend"):
        by_config = grp.groupby(["depth", "seed"])["mae"].mean()
        best_idx  = by_config.idxmin()
        best_mae  = by_config.loc[best_idx]
        print(f"  {backend_name:20s}: depth={best_idx[0]}, seed={best_idx[1]}, "
              f"LOCO MAE={best_mae:.5f}")

    # Prefer standardized unified baselines for Stanford context.
    unified_path, unified_df = _load_unified_stanford_summary()
    if unified_df is not None:
        print(f"\n  Unified Stanford baselines ({unified_path.name}):")
        for model in ["qrc", "xgboost", "ridge"]:
            row = unified_df[unified_df["model"] == model]
            if not row.empty:
                print(f"  {model:24s} MAE={row['mae_mean'].iloc[0]:.5f}")
    else:
        if sim_path.exists():
            sim_df = pd.read_csv(sim_path)
            noiseless = sim_df[sim_df["method"].str.contains("noiseless", na=False)]
            noisy_fez = sim_df[sim_df["method"].str.contains("fez|noisy", na=False)]
            if not noiseless.empty:
                print(f"\n  QRC simulator (noiseless):    MAE={noiseless['mae_mean'].iloc[0]:.5f}")
            if not noisy_fez.empty:
                print(f"  QRC digital twin (ibm_fez):   MAE={noisy_fez['mae_mean'].iloc[0]:.5f}")

    # Relative degradation table
    print("\n  Hardware vs Digital Twin (d=1, seed=42):")
    print(f"  {'Backend':22s} {'HW MAE':>10s} {'Sim MAE':>10s} {'Δ MAE':>10s}")
    print(f"  {'-'*55}")

    backends_info = {
        "ibm_fez": {
            "noise_key": "readout_error",
            "readout":   FEZ_NOISE["measurement_error"],
        },
        "ibm_marrakesh": {
            "readout":   MARRAKESH_NOISE["measurement_error"],
        },
    }

    noisy_fez_mae   = None
    noisy_mar_mae   = None
    if sim_path.exists():
        sim_df = pd.read_csv(sim_path)
        fez_rows = sim_df[sim_df["method"].str.contains("fez", na=False)]
        mar_rows = sim_df[sim_df["method"].str.contains("marrakesh|noisy", na=False)]
        if not fez_rows.empty:
            noisy_fez_mae = fez_rows["mae_mean"].iloc[0]
        if not mar_rows.empty:
            noisy_mar_mae = mar_rows["mae_mean"].iloc[0]

    for backend_name, grp in combined.groupby("backend"):
        d1 = grp[(grp["depth"] == 1) & (grp["seed"] == 42)]
        if d1.empty:
            continue
        hw_mae = d1["mae"].mean()
        if backend_name == "ibm_fez" and noisy_fez_mae:
            sim_mae = noisy_fez_mae
        elif backend_name == "ibm_marrakesh" and noisy_mar_mae:
            sim_mae = noisy_mar_mae
        else:
            sim_mae = float("nan")
        delta = hw_mae - sim_mae
        print(f"  {backend_name:22s} {hw_mae:10.5f} {sim_mae:10.5f} {delta:+10.5f}")

    # Save combined CSV
    comp_csv = data_dir / "fez_vs_marrakesh.csv"
    combined.to_csv(comp_csv, index=False)
    print(f"\n  Combined results saved: {comp_csv}")

    # Plots
    _plot_backend_comparison(combined, noisy_fez_mae, noisy_mar_mae, plot_dir)


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
        print(f"  Saved: {path}")
    plt.close(fig)


def _plot_fez_results(df: pd.DataFrame, plot_dir: Path):
    """Depth sweep plot for ibm_fez LOCO results."""
    loco = df[(df["regime"] == "loco") & (df["experiment"] == "depth_sweep")]
    if loco.empty:
        return
    summary = loco.groupby("depth")["mae"].agg(["mean", "std"]).sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
                marker="o", linewidth=2, capsize=5, color="#e74c3c", label="ibm_fez LOCO MAE")

    # Overlay noiseless simulator
    sim_path = PROJECT_ROOT / "result" / "phase_4" / "data" / "qrc_noiseless.csv"
    if sim_path.exists():
        sim  = pd.read_csv(sim_path)
        sim_loco = sim[sim["regime"] == "loco"].groupby("depth")["mae"].mean().sort_index()
        ax.plot(sim_loco.index, sim_loco.values, marker="s", linewidth=2,
                linestyle="--", color="#3498db", label="Simulator (noiseless)")

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("ibm_fez Hardware — Depth Sweep (Stanford SECL)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    _save_fig(fig, plot_dir, "fez_depth_sweep")


def _plot_backend_comparison(combined: pd.DataFrame,
                              noisy_fez_mae: Optional[float],
                              noisy_mar_mae: Optional[float],
                              plot_dir: Path):
    """Side-by-side bar: fez vs marrakesh, hardware vs digital twin."""
    backends = ["ibm_fez", "ibm_marrakesh"]
    colors   = {"ibm_fez": "#e74c3c", "ibm_marrakesh": "#e67e22"}

    # LOCO MAE by depth, per backend
    fig, ax = plt.subplots(figsize=(9, 5))
    depths = sorted(combined["depth"].unique())
    x      = np.arange(len(depths))
    width  = 0.35

    for i, backend in enumerate(backends):
        grp = combined[combined["backend"] == backend]
        if grp.empty:
            continue
        by_depth = grp.groupby("depth")["mae"].agg(["mean", "std"]).reindex(depths)
        ax.bar(x + i * width, by_depth["mean"], width,
               yerr=by_depth["std"], capsize=4,
               label=backend, color=colors[backend], alpha=0.85)

    # Digital twin reference lines
    if noisy_fez_mae is not None:
        ax.axhline(noisy_fez_mae, color=colors["ibm_fez"], linestyle=":",
                   linewidth=1.5, label="ibm_fez digital twin (d=1)")
    if noisy_mar_mae is not None:
        ax.axhline(noisy_mar_mae, color=colors["ibm_marrakesh"], linestyle=":",
                   linewidth=1.5, label="ibm_marrakesh digital twin (d=1)")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"d={d}" for d in depths])
    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("ibm_fez vs ibm_marrakesh — Hardware LOCO MAE\n(Stanford SECL, 6-fold LOCO)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save_fig(fig, plot_dir, "fez_vs_marrakesh_depth")

    # Seed variance at d=1
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for backend in backends:
        d1 = combined[(combined["backend"] == backend) & (combined["depth"] == 1)]
        if d1.empty:
            continue
        per_seed = d1.groupby("seed")["mae"].mean().sort_index()
        ax2.plot(per_seed.index, per_seed.values, marker="o",
                 linewidth=2, label=backend, color=colors[backend])

    ax2.set_xlabel("Random Seed")
    ax2.set_ylabel("Mean LOCO MAE")
    ax2.set_title("Seed Variance at d=1: ibm_fez vs ibm_marrakesh")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3)
    _save_fig(fig2, plot_dir, "fez_vs_marrakesh_seeds")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6b: ibm_fez Hardware Validation (Stanford SECL)"
    )
    parser.add_argument("--prepare",      action="store_true", help="Build circuits offline")
    parser.add_argument("--run-account",  type=int, metavar="N",
                        help="Run all batches for account N (1 or 2)")
    parser.add_argument("--run-batch",    type=int, metavar="N", help="Run single batch N")
    parser.add_argument("--analyze",      action="store_true",
                        help="Process ibm_fez checkpoint results offline")
    parser.add_argument("--compare",      action="store_true",
                        help="Compare ibm_fez vs ibm_marrakesh (phase_6) results")
    parser.add_argument("--backend",      type=str, default=BACKEND_NAME,
                        help=f"Override backend name (default: {BACKEND_NAME})")
    args = parser.parse_args()

    if not any([args.prepare, args.run_account is not None, args.run_batch is not None,
                args.analyze, args.compare]):
        parser.print_help()
        return

    data_dir, _, _, _ = _get_paths()
    log_path = data_dir / "stage_2_log.txt"
    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("Phase 6b: ibm_fez Hardware Validation (Stanford SECL)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Backend: {args.backend}")
        print(f"ibm_fez specs: CZ={FEZ_NOISE['two_qubit_error']:.3e}, "
              f"SX={FEZ_NOISE['single_qubit_error']:.3e}, "
              f"Readout={FEZ_NOISE['measurement_error']:.3e}, "
              f"T1={FEZ_NOISE['t1_us']} µs, T2={FEZ_NOISE['t2_us']} µs")

        if args.prepare:
            print("\n[MODE] Preparing circuits (offline) ...")
            prepare_circuits()

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
            print("\n[MODE] Analyzing ibm_fez results (offline) ...")
            analyze_results()

        if args.compare:
            print("\n[MODE] Comparing ibm_fez vs ibm_marrakesh ...")
            compare_backends()

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
