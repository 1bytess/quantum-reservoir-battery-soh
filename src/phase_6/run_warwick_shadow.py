"""Phase 6 Warwick-primary shadow references (offline only).

Consumes a prepared Warwick-primary manifest and builds offline references
before any paid hardware submission:
- a hardware-matched noiseless shadow
- a backend-specific digital twin

Typical use:
    python -m src.phase_6.prepare_warwick_hardware --backend ibm_marrakesh
    python -m src.phase_6.run_warwick_shadow --run-label ibm_marrakesh__foldwise__primary__s3072
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import FEZ_NOISE, MARRAKESH_NOISE, N_QUBITS
from phase_4.circuit import build_qrc_circuit, compute_reservoir_features, encode_features
from phase_4.config import ENCODING_METHOD, USE_ZZ_CORRELATORS
from phase_6.prepare_warwick_hardware import TeeLogger
from phase_6.warwick_primary_common import (
    DEFAULT_BACKEND,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_SCOPE,
    DEFAULT_SHOTS,
    build_run_label,
    build_warwick_comparison_rows,
    evaluate_reservoir_runs,
    get_run_paths,
    load_feature_matrix,
    load_feature_records,
    load_manifest,
    unique_manifest_configs,
    write_current_run_metadata,
)


try:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False


def _noise_spec_for_backend(backend_name: str) -> dict[str, float]:
    backend = backend_name.lower()
    if "fez" in backend:
        return FEZ_NOISE
    if "marrakesh" in backend:
        return MARRAKESH_NOISE
    raise ValueError(f"Unsupported digital-twin backend: {backend_name}")


def _build_noise_model(noise_spec: dict[str, float]) -> NoiseModel:
    noise_model = NoiseModel()
    sq_error = depolarizing_error(noise_spec["single_qubit_error"], 1)
    tq_error = depolarizing_error(noise_spec["two_qubit_error"], 2)
    noise_model.add_all_qubit_quantum_error(sq_error, ["rx", "ry", "rz", "x", "y", "z", "h"])
    noise_model.add_all_qubit_quantum_error(tq_error, ["cx", "cz", "ecr"])

    p_meas = noise_spec["measurement_error"]
    readout_error = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def _counts_to_expectations(counts: dict[str, int], n_qubits: int = N_QUBITS) -> np.ndarray:
    total = sum(counts.values())
    padded: dict[str, int] = {}
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        padded[bs.zfill(n_qubits)] = padded.get(bs.zfill(n_qubits), 0) + count

    expectations: list[float] = []
    for i in range(n_qubits):
        exp_val = 0.0
        for bitstring, count in padded.items():
            bit = int(bitstring[-(i + 1)])
            exp_val += (1 - 2 * bit) * count / total
        expectations.append(exp_val)

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            exp_val = 0.0
            for bitstring, count in padded.items():
                bit_i = int(bitstring[-(i + 1)])
                bit_j = int(bitstring[-(j + 1)])
                parity = (1 - 2 * bit_i) * (1 - 2 * bit_j)
                exp_val += parity * count / total
            expectations.append(exp_val)
    return np.array(expectations, dtype=float)


def _compute_digital_twin_reservoir(
    X_features: np.ndarray,
    *,
    depth: int,
    random_rotations: np.ndarray,
    shots: int,
    backend_name: str,
) -> np.ndarray:
    if not QISKIT_AER_AVAILABLE:
        raise ImportError("qiskit-aer is required for Warwick digital-twin simulation.")

    noise_spec = _noise_spec_for_backend(backend_name)
    noise_model = _build_noise_model(noise_spec)
    simulator = AerSimulator(noise_model=noise_model)
    angles_all = encode_features(X_features, method=ENCODING_METHOD)

    reservoir_rows: list[np.ndarray] = []
    for idx, angles in enumerate(angles_all, start=1):
        qc = build_qrc_circuit(
            angles,
            depth=depth,
            random_rotations=random_rotations,
        )
        qc.measure_all()
        qc_transpiled = transpile(qc, simulator, optimization_level=1)
        result = simulator.run(qc_transpiled, shots=shots).result()
        counts = result.get_counts()
        reservoir_rows.append(_counts_to_expectations(counts))
        if idx % 64 == 0 or idx == len(angles_all):
            print(f"    Digital twin {backend_name}: {idx}/{len(angles_all)} circuits")

    return np.asarray(reservoir_rows, dtype=float)


def _build_run_records(
    *,
    X_features: np.ndarray,
    manifest: dict[str, Any],
    modes: list[str],
    twin_backend: str,
) -> list[dict[str, Any]]:
    run_records: list[dict[str, Any]] = []
    for cfg in unique_manifest_configs(manifest):
        depth = int(cfg["depth"])
        seed = int(cfg["seed"])
        shots = int(cfg["shots"])
        experiment = str(cfg["experiment"])
        random_rotations = np.asarray(cfg["random_rotations"], dtype=float)

        print(f"\nConfig: depth={depth}, seed={seed}, shots={shots}, experiment={experiment}")
        if "noiseless" in modes:
            print("  Building noiseless shadow ...")
            reservoir = compute_reservoir_features(
                X_features,
                depth=depth,
                use_zz=USE_ZZ_CORRELATORS,
                random_rotations=random_rotations,
            )
            run_records.append(
                {
                    "mode": "shadow_noiseless",
                    "backend": "statevector",
                    "depth": depth,
                    "seed": seed,
                    "shots": shots,
                    "experiment": experiment,
                    "reservoir": reservoir,
                    "qpu_seconds_total": 0.0,
                    "job_ids": "",
                }
            )

        if "digital_twin" in modes:
            print(f"  Building digital twin ({twin_backend}) ...")
            reservoir = _compute_digital_twin_reservoir(
                X_features,
                depth=depth,
                random_rotations=random_rotations,
                shots=shots,
                backend_name=twin_backend,
            )
            run_records.append(
                {
                    "mode": "shadow_digital_twin",
                    "backend": twin_backend,
                    "depth": depth,
                    "seed": seed,
                    "shots": shots,
                    "experiment": experiment,
                    "reservoir": reservoir,
                    "qpu_seconds_total": 0.0,
                    "job_ids": "",
                }
            )

    return run_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Warwick-primary shadow references from a prepared manifest.")
    parser.add_argument("--run-label", type=str, help="Explicit run label from prepare_warwick_hardware.")
    parser.add_argument(
        "--mode",
        choices=("noiseless", "digital_twin", "both"),
        default="both",
        help="Which shadow references to build.",
    )
    parser.add_argument(
        "--digital-backend",
        choices=("ibm_marrakesh", "ibm_fez"),
        default=DEFAULT_BACKEND,
        help="Noise profile for the digital twin.",
    )
    parser.add_argument("--preprocess-mode", type=str, default=DEFAULT_PREPROCESS_MODE)
    parser.add_argument("--scope", type=str, default=DEFAULT_SCOPE)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    args = parser.parse_args()

    run_label = args.run_label or build_run_label(
        backend=args.digital_backend,
        preprocess_mode=args.preprocess_mode,
        scope=args.scope,
        shots=args.shots,
    )
    paths = get_run_paths(run_label, include_hardware=True)
    log_path = paths.shadow_log_path
    tee = TeeLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Phase 6 Stage 4: Warwick Primary Shadow References")
        print("=" * 72)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Run label: {run_label}")
        print(f"Digital twin backend: {args.digital_backend}")

        manifest = load_manifest(paths)
        feature_records = load_feature_records(manifest)
        X_features = load_feature_matrix(feature_records)
        modes = ["noiseless", "digital_twin"] if args.mode == "both" else [args.mode]

        print(f"Feature rows: {len(feature_records)}")
        print(f"PCA feature dim: {X_features.shape[1]}")

        run_records = _build_run_records(
            X_features=X_features,
            manifest=manifest,
            modes=modes,
            twin_backend=args.digital_backend,
        )

        results_df, summary_df = evaluate_reservoir_runs(
            run_records=run_records,
            feature_records=feature_records,
            manifest=manifest,
            run_label=run_label,
        )

        results_path = paths.data_dir / "qrc_shadow_warwick.csv"
        summary_path = paths.data_dir / "qrc_shadow_warwick_summary.csv"
        results_df.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        canonical_results = paths.stage_data_dir / "qrc_shadow_warwick.csv"
        canonical_summary = paths.stage_data_dir / "qrc_shadow_warwick_summary.csv"
        results_df.to_csv(canonical_results, index=False)
        summary_df.to_csv(canonical_summary, index=False)

        comp_rows, unified_path = build_warwick_comparison_rows(summary_df, source_tag="shadow_warwick")
        if comp_rows is not None and unified_path is not None:
            comp_path = paths.data_dir / "shadow_vs_unified_warwick.csv"
            canonical_comp = paths.stage_data_dir / "shadow_vs_unified_warwick.csv"
            comp_rows.to_csv(comp_path, index=False)
            comp_rows.to_csv(canonical_comp, index=False)
            print(f"Saved comparison:     {comp_path}")
            print(f"Unified reference:    {unified_path.name}")

        meta_path = write_current_run_metadata(paths, manifest, updated_by="run_warwick_shadow")
        print(f"Saved per-fold results: {results_path}")
        print(f"Saved summary:          {summary_path}")
        print(f"Saved stage pointer:    {meta_path}")
        print(summary_df.to_string(index=False))
        print(f"\nCompleted: {datetime.now().isoformat()}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
