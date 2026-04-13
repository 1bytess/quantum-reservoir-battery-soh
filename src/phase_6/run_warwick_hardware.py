"""Phase 6 Warwick-primary hardware execution and analysis.

This script consumes a manifest prepared by
``src.phase_6.prepare_warwick_hardware`` and keeps run provenance clean by
operating on an explicit run label.

Recommended sequence:
    python -m src.phase_6.prepare_warwick_hardware --backend ibm_marrakesh
    python -m src.phase_6.run_warwick_shadow --run-label <label>
    python -m src.phase_6.run_warwick_hardware --run-batches 1 --account 1 --run-label <label>
    python -m src.phase_6.run_warwick_hardware --run-batches 2 --account 2 --run-label <label>
    python -m src.phase_6.run_warwick_hardware --run-batches 3,4,5,6 --account 3 --run-label <label>
    python -m src.phase_6.run_warwick_hardware --analyze --run-label <label>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase_6.env_utils import load_phase6_env
from phase_6.prepare_warwick_hardware import TeeLogger, _estimate_qpu_seconds
from phase_6.warwick_primary_common import (
    DEFAULT_BACKEND,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_SCOPE,
    DEFAULT_SHOTS,
    build_run_label,
    build_warwick_comparison_rows,
    evaluate_reservoir_runs,
    get_run_paths,
    load_manifest,
    load_feature_records,
    write_current_run_metadata,
)

load_phase6_env(PROJECT_ROOT)

try:
    from qiskit.qasm2 import loads as qasm2_loads
except ImportError:
    from qiskit import qasm2

    qasm2_loads = qasm2.loads


def _get_service(account: int):
    import os
    from qiskit_ibm_runtime import QiskitRuntimeService

    api_key = os.environ[f"IBM_ACC{account}_API"]
    crn = os.environ[f"IBM_ACC{account}_CRN"]
    return QiskitRuntimeService(channel="ibm_cloud", token=api_key, instance=crn)


def _parse_counts(pub_result: Any) -> dict[str, int]:
    creg = pub_result.data
    raw_counts = None
    for attr_name in dir(creg):
        attr = getattr(creg, attr_name)
        if hasattr(attr, "get_counts"):
            raw_counts = attr.get_counts()
            break
    if raw_counts is None:
        raw_counts = creg.meas.get_counts()
    return {str(k): int(v) for k, v in raw_counts.items()}


def counts_to_expectations(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    total = sum(counts.values())
    padded: dict[str, int] = {}
    for bs, cnt in counts.items():
        key = bs.replace(" ", "").zfill(n_qubits)
        padded[key] = padded.get(key, 0) + cnt

    exps: list[float] = []
    for i in range(n_qubits):
        ev = sum((1 - 2 * int(bs[-(i + 1)])) * c / total for bs, c in padded.items())
        exps.append(ev)

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            ev = sum(
                (1 - 2 * int(bs[-(i + 1)])) * (1 - 2 * int(bs[-(j + 1)])) * c / total
                for bs, c in padded.items()
            )
            exps.append(ev)
    return np.array(exps, dtype=float)


def _default_account_assignments(batch_ids: list[int]) -> dict[int, list[int]]:
    half = int(np.ceil(len(batch_ids) / 2))
    return {
        1: batch_ids[:half],
        2: batch_ids[half:],
    }


def _parse_batch_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one batch id.")
    return [int(value) for value in values]


def run_batch(
    *,
    run_label: str,
    batch_id: int,
    account: int,
    backend_name: str | None = None,
) -> Path:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Batch, SamplerV2

    paths = get_run_paths(run_label, include_hardware=True)
    manifest = load_manifest(paths)
    batches = manifest["batches"]
    batch_key = str(batch_id)
    if batch_key not in batches:
        raise ValueError(f"Batch {batch_id} not found in manifest {paths.manifest_path}.")

    batch_cfg = batches[batch_key]
    backend_to_use = backend_name or str(manifest.get("backend") or DEFAULT_BACKEND)
    shots = int(batch_cfg["shots"])
    n_circuits = int(batch_cfg["n_circuits"])

    print(f"\n[run_batch {batch_id}] Run label: {run_label}")
    print(f"  Connecting to {backend_to_use} (ACC{account}) ...")
    service = _get_service(account)
    backend = service.backend(backend_to_use)
    print(f"  Backend: {backend.name}")

    circuits = [qasm2_loads(qasm) for qasm in batch_cfg["qasm"]]
    print(f"  Rebuilt {len(circuits)} circuits from QASM")

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuits = pm.run(circuits)

    est_qpu = _estimate_qpu_seconds(n_circuits, shots)
    print(f"  Estimated QPU time: {est_qpu:.1f}s ({est_qpu/60.0:.1f} min)")

    checkpoint = {
        "run_label": run_label,
        "batch_id": int(batch_id),
        "account": int(account),
        "backend": backend_to_use,
        "timestamp_start": datetime.now().isoformat(),
        "config": {
            "depth": int(batch_cfg["depth"]),
            "seed": int(batch_cfg["seed"]),
            "shots": shots,
            "experiment": str(batch_cfg["experiment"]),
            "chunk_index": batch_cfg.get("chunk_index"),
        },
        "n_circuits": int(len(isa_circuits)),
        "status": "submitted",
    }

    ckpt_path = paths.checkpoint_dir / f"batch{batch_id}.json"
    with Batch(backend=backend) as batch_ctx:
        sampler = SamplerV2(mode=batch_ctx)
        job = sampler.run(isa_circuits, shots=shots)
        checkpoint["job_id"] = job.job_id()
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  Job submitted: {job.job_id()}")
        print(f"  Checkpoint saved: {ckpt_path}")

        print("  Waiting for results ...")
        t0 = time.time()
        result = job.result()
        wall_time = time.time() - t0

    all_counts: dict[str, dict[str, int]] = {}
    for idx, pub_result in enumerate(result):
        feature_id = int(batch_cfg["sample_indices"][idx])
        key = f"d{batch_cfg['depth']}_s{batch_cfg['seed']}_feature{feature_id}"
        all_counts[key] = _parse_counts(pub_result)

    qpu_seconds = None
    try:
        usage = job.usage()
        qpu_seconds = getattr(usage, "quantum_seconds", None)
    except Exception:
        pass

    checkpoint["timestamp_end"] = datetime.now().isoformat()
    checkpoint["wall_time_sec"] = round(wall_time, 1)
    checkpoint["qpu_seconds"] = qpu_seconds
    checkpoint["counts"] = all_counts
    checkpoint["status"] = "completed"
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"  Complete. Counts saved: {len(all_counts)}")
    return ckpt_path


def run_batches(
    *,
    run_label: str,
    batch_ids: list[int],
    account: int,
    backend_name: str | None = None,
) -> None:
    total_qpu = 0.0
    total_wall = 0.0
    for batch_id in batch_ids:
        ckpt_path = run_batch(
            run_label=run_label,
            batch_id=batch_id,
            account=account,
            backend_name=backend_name,
        )
        with open(ckpt_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        total_qpu += float(checkpoint.get("qpu_seconds") or 0.0)
        total_wall += float(checkpoint.get("wall_time_sec") or 0.0)

    print(f"\nAccount {account} complete for run {run_label}")
    print(f"  Total QPU time:  {total_qpu:.1f}s ({total_qpu/60.0:.1f} min)")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/60.0:.1f} min)")


def run_account(
    *,
    run_label: str,
    account: int,
    backend_name: str | None = None,
) -> None:
    paths = get_run_paths(run_label, include_hardware=True)
    manifest = load_manifest(paths)
    batch_ids = sorted(int(k) for k in manifest["batches"].keys())
    assignments = _default_account_assignments(batch_ids)
    assigned = assignments.get(account, [])
    if not assigned:
        print(f"No default batches assigned to account {account}.")
        return

    print(f"\nLegacy two-account split for run {run_label}: ACC{account} -> {assigned}")
    run_batches(
        run_label=run_label,
        batch_ids=assigned,
        account=account,
        backend_name=backend_name,
    )


def analyze_results(*, run_label: str) -> Any:
    paths = get_run_paths(run_label, include_hardware=True)
    manifest = load_manifest(paths)
    feature_records = load_feature_records(manifest)
    n_features = int(manifest["n_feature_rows"])
    n_qubits = int(manifest["n_qubits"])
    exp_dim = n_qubits + n_qubits * (n_qubits - 1) // 2

    ckpt_files = sorted(paths.checkpoint_dir.glob("batch*.json"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {paths.checkpoint_dir}")

    run_records: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for ckpt_path in ckpt_files:
        with open(ckpt_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        if checkpoint.get("status") != "completed" or "counts" not in checkpoint:
            continue

        cfg = checkpoint["config"]
        run_key = f"d{cfg['depth']}_s{cfg['seed']}"
        if run_key not in by_key:
            by_key[run_key] = {
                "mode": "hardware",
                "depth": int(cfg["depth"]),
                "seed": int(cfg["seed"]),
                "shots": int(cfg["shots"]),
                "experiment": str(cfg["experiment"]),
                "backend": str(checkpoint["backend"]),
                "reservoir": np.full((n_features, exp_dim), np.nan),
                "qpu_seconds_total": 0.0,
                "job_ids": [],
            }

        if checkpoint.get("qpu_seconds") is not None:
            by_key[run_key]["qpu_seconds_total"] += float(checkpoint["qpu_seconds"])
        if checkpoint.get("job_id"):
            by_key[run_key]["job_ids"].append(str(checkpoint["job_id"]))

        for count_key, counts in checkpoint["counts"].items():
            feature_id = int(count_key.split("_feature")[1])
            by_key[run_key]["reservoir"][feature_id] = counts_to_expectations(counts, n_qubits)

    for run in by_key.values():
        run["job_ids"] = ",".join(run["job_ids"])
        run_records.append(run)

    results_df, summary_df = evaluate_reservoir_runs(
        run_records=run_records,
        feature_records=feature_records,
        manifest=manifest,
        run_label=run_label,
    )

    results_path = paths.data_dir / "qrc_hardware_warwick.csv"
    summary_path = paths.data_dir / "qrc_hardware_warwick_summary.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    canonical_results = paths.stage_data_dir / "qrc_hardware_warwick.csv"
    canonical_summary = paths.stage_data_dir / "qrc_hardware_warwick_summary.csv"
    results_df.to_csv(canonical_results, index=False)
    summary_df.to_csv(canonical_summary, index=False)

    comp_rows, unified_path = build_warwick_comparison_rows(summary_df, source_tag="hardware_warwick")
    if comp_rows is not None and unified_path is not None:
        comp_path = paths.data_dir / "hardware_vs_unified_warwick.csv"
        canonical_comp = paths.stage_data_dir / "hardware_vs_unified_warwick.csv"
        comp_rows.to_csv(comp_path, index=False)
        comp_rows.to_csv(canonical_comp, index=False)
        print(f"Saved comparison:     {comp_path}")
        print(f"Unified reference:    {unified_path.name}")

    meta_path = write_current_run_metadata(paths, manifest, updated_by="run_warwick_hardware")
    print(f"Saved per-fold results: {results_path}")
    print(f"Saved summary:          {summary_path}")
    print(f"Saved stage pointer:    {meta_path}")
    print(summary_df.to_string(index=False))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Warwick-primary hardware batches from a prepared manifest.")
    parser.add_argument("--prepare", action="store_true", help="Print reminder to use prepare_warwick_hardware.")
    parser.add_argument("--run-label", type=str, help="Explicit run label from prepare_warwick_hardware.")
    parser.add_argument("--run-account", type=int, metavar="N", help="Legacy two-account convenience split for account N.")
    parser.add_argument("--run-batch", type=int, metavar="N", help="Run a single batch by id; requires --account.")
    parser.add_argument("--run-batches", type=str, help="Comma-separated batch ids to run on one account; requires --account.")
    parser.add_argument("--account", type=int, metavar="N", help="IBM account number for --run-batch or --run-batches.")
    parser.add_argument("--analyze", action="store_true", help="Analyze completed Warwick checkpoints.")
    parser.add_argument("--backend", type=str, default=None, help="Override backend; defaults to manifest backend.")
    parser.add_argument("--preprocess-mode", type=str, default=DEFAULT_PREPROCESS_MODE)
    parser.add_argument("--scope", type=str, default=DEFAULT_SCOPE)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    args = parser.parse_args()

    if not any(
        [
            args.prepare,
            args.run_account is not None,
            args.run_batch is not None,
            args.run_batches,
            args.analyze,
        ]
    ):
        parser.print_help()
        return

    run_label = args.run_label or build_run_label(
        backend=args.backend or DEFAULT_BACKEND,
        preprocess_mode=args.preprocess_mode,
        scope=args.scope,
        shots=args.shots,
    )
    paths = get_run_paths(run_label, include_hardware=True)
    log_path = paths.hardware_log_path
    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Phase 6 Stage 4: Warwick Primary Hardware Execution")
        print("=" * 72)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Run label: {run_label}")
        print(f"Backend override: {args.backend or '<manifest>'}")

        if args.prepare:
            print(
                "Use python -m src.phase_6.prepare_warwick_hardware "
                "--backend <backend> --shots <shots> before submission."
            )

        if args.run_account is not None:
            run_account(
                run_label=run_label,
                account=args.run_account,
                backend_name=args.backend,
            )

        if args.run_batch is not None:
            if args.account is None:
                raise ValueError("--account is required with --run-batch.")
            run_batches(
                run_label=run_label,
                batch_ids=[args.run_batch],
                account=args.account,
                backend_name=args.backend,
            )

        if args.run_batches:
            if args.account is None:
                raise ValueError("--account is required with --run-batches.")
            run_batches(
                run_label=run_label,
                batch_ids=_parse_batch_list(args.run_batches),
                account=args.account,
                backend_name=args.backend,
            )

        if args.analyze:
            analyze_results(run_label=run_label)

        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
