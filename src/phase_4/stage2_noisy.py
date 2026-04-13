"""Stage 2 — Noisy QRC evaluation: IBM Heron R2 Digital Twin.

Uses Qiskit Aer simulator with a noise model matching IBM's ibm_fez
(Heron R2 processor). Evaluates how quantum noise affects SOH prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    DEPTH_RANGE, CELL_IDS,
    USE_ZZ_CORRELATORS, RANDOM_STATE,
    HERON_R2_NOISE, SHOTS_LIST, DEFAULT_SHOTS, BACKEND_NAME,
    Phase4LabPaths, N_QUBITS,
)

# ── Qiskit availability check ────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# Reuse quantum reservoir and circuit
from .qrc_model import QuantumReservoir
from .circuit import encode_features, build_qrc_circuit

# ── PCA defaults ──────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 6
PCA_RANDOM_STATE = 42


def _fit_pca_in_fold(
    X_train_72d: np.ndarray,
    X_test_72d: np.ndarray,
    n_components: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training data only, transform both train and test."""
    n_components = min(n_components, X_train_72d.shape[0], X_train_72d.shape[1])
    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    X_train_6d = pca.fit_transform(X_train_72d)
    X_test_6d = pca.transform(X_test_72d)
    return X_train_6d, X_test_6d


# =============================================================================
# Noise Model Creation
# =============================================================================

def get_heron_r2_noise_model() -> "NoiseModel":
    """Create a synthetic noise model matching IBM Heron R2 (ibm_fez).

    Parameters from IBM Quantum documentation:
      - Single-qubit gate error: 0.02% depolarizing
      - Two-qubit gate error:    0.5% depolarizing
      - Measurement error:       1% bitflip
      - T1 ~ 300 μs, T2 ~ 200 μs (not modeled as depolarizing approximation)
    """
    if not QISKIT_AER_AVAILABLE:
        raise ImportError(
            "Qiskit Aer required for noisy simulation. "
            "Install with: pip install qiskit-aer"
        )

    noise_model = NoiseModel()

    # Single-qubit depolarizing error
    sq_error = depolarizing_error(HERON_R2_NOISE["single_qubit_error"], 1)
    noise_model.add_all_qubit_quantum_error(
        sq_error, ['rx', 'ry', 'rz', 'x', 'y', 'z', 'h']
    )

    # Two-qubit depolarizing error
    tq_error = depolarizing_error(HERON_R2_NOISE["two_qubit_error"], 2)
    noise_model.add_all_qubit_quantum_error(
        tq_error, ['cx', 'cz', 'ecr']
    )

    # Measurement (readout) error
    p_meas = HERON_R2_NOISE["measurement_error"]
    readout_error = ReadoutError(
        [[1 - p_meas, p_meas], [p_meas, 1 - p_meas]]
    )
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def get_ibm_noise_model(backend_name: str = BACKEND_NAME) -> "NoiseModel":
    """Try to load noise model from actual IBM backend, fall back to synthetic."""
    if not IBM_AVAILABLE:
        print(f"  IBM Runtime not available → using synthetic Heron R2 noise model")
        return get_heron_r2_noise_model()

    try:
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend_name)
        noise_model = NoiseModel.from_backend(backend)
        print(f"  Loaded noise model from real {backend_name} backend")
        return noise_model
    except Exception as e:
        print(f"  Could not connect to {backend_name}: {e}")
        print(f"  → Using synthetic Heron R2 noise model")
        return get_heron_r2_noise_model()


# =============================================================================
# Noisy Quantum Reservoir
# =============================================================================

class NoisyQuantumReservoir(QuantumReservoir):
    """QRC with shot-based noisy simulation using Qiskit Aer.

    Extends QuantumReservoir to use AerSimulator with noise model
    instead of exact statevector simulation.
    """

    def __init__(
        self,
        depth: int = 2,
        use_zz: bool = USE_ZZ_CORRELATORS,
        ridge_alpha: float = None,
        noise_model: "NoiseModel" = None,
        shots: int = DEFAULT_SHOTS,
    ):
        super().__init__(
            depth=depth,
            use_zz=use_zz,
            ridge_alpha=ridge_alpha,
            use_classical_fallback=False,
            add_random_rotations=True,
        )
        self.noise_model = noise_model
        self.shots = shots

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """Compute reservoir features with noise simulation."""
        from itertools import combinations

        # Standardize input (using fitted scaler)
        X_scaled = self.scaler_.transform(X)
        X_scaled = np.clip(X_scaled, -3.0, 3.0)

        # Encode as rotation angles
        angles = encode_features(X_scaled)

        n_samples = X.shape[0]
        reservoir_list = []

        # Create noisy simulator
        if self.noise_model is not None:
            simulator = AerSimulator(noise_model=self.noise_model)
        else:
            simulator = AerSimulator()

        for i in range(n_samples):
            # Build circuit
            qc = build_qrc_circuit(angles[i], self.depth, self.random_rotations_)

            # Add measurements
            qc_meas = qc.copy()
            qc_meas.measure_all()

            # Transpile for noisy simulator
            qc_transpiled = transpile(qc_meas, simulator, optimization_level=1)

            # Run
            job = simulator.run(qc_transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            # Extract expectation values from measurement statistics
            exp_vals = self._counts_to_expectations(
                counts, N_QUBITS, self.use_zz
            )
            reservoir_list.append(exp_vals)

        return np.array(reservoir_list)

    @staticmethod
    def _counts_to_expectations(
        counts: Dict,
        n_qubits: int,
        use_zz: bool,
    ) -> np.ndarray:
        """Convert measurement counts -> expectation values <Z_i>, <Z_iZ_j>."""
        from itertools import combinations

        total_shots = sum(counts.values())
        expectations = []

        # Zero-pad bitstrings to n_qubits (transpiler may shorten them)
        padded_counts = {}
        for bitstring, count in counts.items():
            bs = bitstring.replace(" ", "")
            padded_counts[bs.zfill(n_qubits)] = count

        # Single-qubit <Z_i>
        for i in range(n_qubits):
            exp_val = 0.0
            for bitstring, count in padded_counts.items():
                # Qiskit uses little-endian bit ordering
                bit = int(bitstring[-(i + 1)])
                exp_val += (1 - 2 * bit) * count / total_shots
            expectations.append(exp_val)

        # Two-qubit correlators <Z_iZ_j>
        if use_zz:
            for i, j in combinations(range(n_qubits), 2):
                exp_val = 0.0
                for bitstring, count in padded_counts.items():
                    bit_i = int(bitstring[-(i + 1)])
                    bit_j = int(bitstring[-(j + 1)])
                    parity = (1 - 2 * bit_i) * (1 - 2 * bit_j)
                    exp_val += parity * count / total_shots
                expectations.append(exp_val)

        return np.array(expectations)


# =============================================================================
# Noisy Evaluation Functions
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_noisy_loco(
    cell_data: Dict[str, Dict],
    depth: int,
    noise_model: "NoiseModel",
    shots: int = DEFAULT_SHOTS,
) -> pd.DataFrame:
    """Run noisy QRC with LOCO evaluation across all cells.

    PCA is fit INSIDE each fold on training data only (leakage-free).
    """
    results = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]

        # Always start from 72D, apply PCA in-fold
        X_train_72d = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        train_groups = np.concatenate([
            np.full(len(cell_data[c]["y"]), c, dtype=object) for c in train_cells
        ])
        X_test_72d = cell_data[test_cell]["X_72d"]
        y_test = cell_data[test_cell]["y"]

        X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

        qrc = NoisyQuantumReservoir(
            depth=depth,
            use_zz=USE_ZZ_CORRELATORS,
            noise_model=noise_model,
            shots=shots,
        )
        qrc.fit(X_train, y_train, groups=train_groups)
        y_pred = qrc.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        naive_mae = mean_absolute_error(
            y_test, np.full_like(y_test, y_train.mean())
        )

        results.append({
            "stage": "noisy",
            "regime": "loco",
            "depth": depth,
            "shots": shots,
            "test_cell": test_cell,
            "train_cells": "+".join(train_cells),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "naive_mae": naive_mae,
            "beats_naive": metrics["mae"] < naive_mae,
        })

    return pd.DataFrame(results)


def run_noisy_temporal(
    cell_data: Dict[str, Dict],
    depth: int,
    noise_model: "NoiseModel",
    shots: int = DEFAULT_SHOTS,
    train_frac: float = 0.7,
) -> pd.DataFrame:
    """Run noisy QRC with temporal split evaluation.

    PCA is fit INSIDE each split on training blocks only (leakage-free).
    """
    results = []

    for cid in CELL_IDS:
        data = cell_data[cid]
        X_72d = data["X_72d"]
        y = data["y"]
        blocks = data["block_ids"]

        n_total = len(y)
        if n_total < 3:
            continue

        n_train = max(2, int(n_total * train_frac))
        sort_idx = np.argsort(blocks)
        X_72d_sorted, y_sorted = X_72d[sort_idx], y[sort_idx]

        X_train_72d, X_test_72d = X_72d_sorted[:n_train], X_72d_sorted[n_train:]
        y_train, y_test = y_sorted[:n_train], y_sorted[n_train:]

        if len(y_test) == 0:
            continue

        # Apply PCA in-fold (leakage-free)
        X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)

        qrc = NoisyQuantumReservoir(
            depth=depth,
            use_zz=USE_ZZ_CORRELATORS,
            noise_model=noise_model,
            shots=shots,
        )
        qrc.fit(X_train, y_train)
        y_pred = qrc.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        persist_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train[-1]))

        results.append({
            "stage": "noisy",
            "regime": "temporal",
            "depth": depth,
            "shots": shots,
            "test_cell": cid,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "persist_mae": persist_mae,
            "beats_persist": metrics["mae"] < persist_mae,
        })

    return pd.DataFrame(results)


def run_stage2(
    cell_data: Dict[str, Dict],
    paths: Phase4LabPaths,
    use_ibm_noise: bool = False,
    depths: list = None,
    shots_list: list = None,
) -> pd.DataFrame:
    """Run Stage 2: noisy (IBM Heron R2 digital twin) depth sweep.

    Args:
        cell_data: Per-cell EIS features
        paths: Output paths
        use_ibm_noise: Try real IBM backend noise model
        depths: Circuit depths to test (default: DEPTH_RANGE)
        shots_list: Shot counts to sweep (default: SHOTS_LIST)

    Returns:
        Combined DataFrame of all noisy results.
    """
    if depths is None:
        depths = DEPTH_RANGE
    if shots_list is None:
        shots_list = SHOTS_LIST

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║  Stage 2: Noisy (IBM Heron R2 Twin)     ║")
    print("  ╚══════════════════════════════════════════╝")

    # Get noise model
    if use_ibm_noise:
        noise_model = get_ibm_noise_model()
    else:
        noise_model = get_heron_r2_noise_model()
        print(f"  Using synthetic IBM Heron R2 noise model")
        print(f"    1-qubit error: {HERON_R2_NOISE['single_qubit_error']}")
        print(f"    2-qubit error: {HERON_R2_NOISE['two_qubit_error']}")
        print(f"    Readout error: {HERON_R2_NOISE['measurement_error']}")

    all_results = []

    for depth in depths:
        for shots in shots_list:
            print(f"\n  --- Depth {depth}, Shots {shots} ---")

            # LOCO
            loco_df = run_noisy_loco(cell_data, depth, noise_model, shots)
            all_results.append(loco_df)
            avg_loco = loco_df["mae"].mean()
            print(f"    LOCO:     avg MAE = {avg_loco:.4f}")

            # Temporal
            temp_df = run_noisy_temporal(cell_data, depth, noise_model, shots)
            all_results.append(temp_df)
            if len(temp_df) > 0:
                avg_temp = temp_df["mae"].mean()
                print(f"    Temporal: avg MAE = {avg_temp:.4f}")

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(paths.data_dir / "qrc_noisy.csv", index=False)
    print(f"\n  Saved qrc_noisy.csv ({len(results_df)} rows)")

    return results_df
