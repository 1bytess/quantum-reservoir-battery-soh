"""Stochastic resonance sweep: controlled quantum noise channels.

Sweeps depolarizing, amplitude-damping, and phase-damping error channels
at multiple rates.  Each configuration is evaluated with LOCO on EIS
features using a shot-based noisy QRC.
"""

import numpy as np
import pandas as pd
from typing import Dict
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    Phase5LabPaths, SR_NOISE_RATES, SR_CHANNELS, SR_SHOTS, SR_REPEATS,
    CELL_IDS, RANDOM_STATE,
)

DEFAULT_STAGE = "stage_1"

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..phase_4.circuit import encode_features, build_qrc_circuit
from ..phase_4.config import (
    N_QUBITS, USE_ZZ_CORRELATORS, CLAMP_RANGE,
)

# ── PCA defaults (same as phase_4) ───────────────────────────────────
N_PCA_COMPONENTS = 6
PCA_RANDOM_STATE = 42


# =========================================================================
# Noise model construction
# =========================================================================

def build_noise_model(channel: str, rate: float) -> "NoiseModel":
    """Build a single-channel NoiseModel at the given error rate.

    Args:
        channel: One of ``"depolarizing"``, ``"amplitude_damping"``,
                 ``"phase_damping"``.
        rate: Error probability per gate (0 = noiseless).

    Returns:
        A Qiskit ``NoiseModel`` with the requested error on all gates.
    """
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        amplitude_damping_error,
        phase_damping_error,
    )

    noise_model = NoiseModel()
    if rate <= 0:
        return noise_model  # empty = noiseless

    if channel == "depolarizing":
        err_1q = depolarizing_error(rate, 1)
        err_2q = depolarizing_error(rate, 2)
    elif channel == "amplitude_damping":
        err_1q = amplitude_damping_error(rate)
        err_2q = err_1q.tensor(err_1q)
    elif channel == "phase_damping":
        err_1q = phase_damping_error(rate)
        err_2q = err_1q.tensor(err_1q)
    else:
        raise ValueError(f"Unknown noise channel: {channel}")

    noise_model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "x", "y", "z", "h"],
    )
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr"])
    return noise_model


# =========================================================================
# Shot-based expectation values (reuses pattern from stage2_noisy)
# =========================================================================

def _counts_to_expectations(
    counts: Dict[str, int],
    n_qubits: int,
    use_zz: bool,
) -> np.ndarray:
    """Convert measurement counts to Z / ZZ expectation values."""
    total_shots = sum(counts.values())
    expectations = []

    # Zero-pad bitstrings to n_qubits (transpiler may shorten them)
    padded_counts = {}
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        padded_counts[bs.zfill(n_qubits)] = count

    for i in range(n_qubits):
        exp_val = 0.0
        for bitstring, count in padded_counts.items():
            bit = int(bitstring[-(i + 1)])
            exp_val += (1 - 2 * bit) * count / total_shots
        expectations.append(exp_val)

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


def _noisy_reservoir_features(
    X_scaled: np.ndarray,
    depth: int,
    noise_model: "NoiseModel",
    shots: int,
    use_zz: bool,
    random_rotations: np.ndarray,
) -> np.ndarray:
    """Compute reservoir features via shot-based noisy simulation."""
    angles = encode_features(X_scaled)
    n_samples = X_scaled.shape[0]
    simulator = AerSimulator(noise_model=noise_model)
    reservoir_list = []

    for i in range(n_samples):
        qc = build_qrc_circuit(angles[i], depth=depth, random_rotations=random_rotations)
        qc_meas = qc.copy()
        qc_meas.measure_all()
        qc_t = transpile(qc_meas, simulator, optimization_level=1)
        job = simulator.run(qc_t, shots=shots)
        counts = job.result().get_counts()
        exp_vals = _counts_to_expectations(counts, N_QUBITS, use_zz)
        reservoir_list.append(exp_vals)

    return np.array(reservoir_list)


# =========================================================================
# LOCO evaluation helper
# =========================================================================

def _fit_pca_in_fold(X_train_72d, X_test_72d, n_components=N_PCA_COMPONENTS):
    n_components = min(n_components, X_train_72d.shape[0], X_train_72d.shape[1])
    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    return pca.fit_transform(X_train_72d), pca.transform(X_test_72d)


def _compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _loco_noisy_eval(
    cell_data: Dict[str, Dict],
    depth: int,
    noise_model: "NoiseModel",
    shots: int,
    use_zz: bool = USE_ZZ_CORRELATORS,
) -> pd.DataFrame:
    """Run one LOCO pass with a noisy QRC (Ridge readout)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    rng = np.random.RandomState(RANDOM_STATE)
    random_rotations = rng.uniform(0, 2 * np.pi, (depth, N_QUBITS, 3))

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]

        X_train_72d = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_72d = cell_data[test_cell]["X_72d"]
        y_test = cell_data[test_cell]["y"]

        X_train_6d, X_test_6d = _fit_pca_in_fold(X_train_72d, X_test_72d)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_6d)
        X_te = scaler.transform(X_test_6d)
        X_tr = np.clip(X_tr, -3.0, 3.0)
        X_te = np.clip(X_te, -3.0, 3.0)

        R_train = _noisy_reservoir_features(
            X_tr, depth, noise_model, shots, use_zz, random_rotations,
        )
        R_test = _noisy_reservoir_features(
            X_te, depth, noise_model, shots, use_zz, random_rotations,
        )

        grid = GridSearchCV(
            Ridge(), {"alpha": alphas},
            cv=min(3, len(y_train)),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid.fit(R_train, y_train)
        y_pred = grid.predict(R_test)
        metrics = _compute_metrics(y_test, y_pred)

        results.append({
            "test_cell": test_cell,
            **metrics,
        })

    return pd.DataFrame(results)


# =========================================================================
# Main sweep
# =========================================================================

def run_stochastic_resonance(
    cell_data: Dict[str, Dict],
    paths: Phase5LabPaths = None,
    depth: int = 2,
) -> pd.DataFrame:
    """Sweep quantum noise channels and rates, full LOCO each time.

    Returns:
        DataFrame with columns: channel, rate, repeat, test_cell, mae, ...
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit + Aer required for stochastic resonance sweep")

    if paths is None:
        paths = Phase5LabPaths(DEFAULT_STAGE)
    paths.ensure_dirs()

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║  Stochastic Resonance Sweep              ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"  Channels: {SR_CHANNELS}")
    print(f"  Rates:    {SR_NOISE_RATES}")
    print(f"  Shots:    {SR_SHOTS},  Repeats: {SR_REPEATS}")

    all_rows = []

    for channel in SR_CHANNELS:
        for rate in SR_NOISE_RATES:
            for rep in range(SR_REPEATS):
                tag = f"{channel} rate={rate:.4f} rep={rep+1}/{SR_REPEATS}"
                print(f"\n    {tag}")

                nm = build_noise_model(channel, rate)
                fold_df = _loco_noisy_eval(
                    cell_data, depth=depth, noise_model=nm, shots=SR_SHOTS,
                )

                for _, row in fold_df.iterrows():
                    all_rows.append({
                        "channel": channel,
                        "rate": rate,
                        "repeat": rep,
                        "depth": depth,
                        "shots": SR_SHOTS,
                        "test_cell": row["test_cell"],
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "r2": row["r2"],
                    })

                avg_mae = fold_df["mae"].mean()
                print(f"      avg MAE = {avg_mae:.4f}")

    results_df = pd.DataFrame(all_rows)
    out_path = paths.data_dir / "stochastic_resonance.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved stochastic_resonance.csv ({len(results_df)} rows)")

    # Summary
    summary = (
        results_df.groupby(["channel", "rate"])["mae"]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\n  Stochastic Resonance Summary:")
    print(summary.to_string())

    return results_df
