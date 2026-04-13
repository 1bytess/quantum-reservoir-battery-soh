"""Quantum circuit construction for QRC.

Implements:
1. Angle encoding (arctan with clamping), optional dual-axis (RY+RZ)
2. Fixed entangling ansatz (CZ ring), optional data re-uploading
3. Observable extraction (⟨Z_i⟩, optional ⟨Z_iZ_j⟩, optional XYZ multi-basis)
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not installed. QRC will use classical simulation fallback.")

from .config import N_QUBITS, CLAMP_RANGE, ENCODING_METHOD, USE_ZZ_CORRELATORS


def encode_features(
    features: np.ndarray,
    method: str = ENCODING_METHOD,
    dual_axis: bool = False,
) -> np.ndarray:
    """
    Encode classical features as rotation angles.

    Args:
        features: (n_samples, n_features) array of standardized features
        method: "arctan" or "linear"
        dual_axis: If True, return (theta_ry, phi_rz) tuple for dual-axis
                   encoding.  phi uses a quarter-phase offset (π/4).

    Returns:
        If dual_axis is False:
            angles: (n_samples, n_features) RY rotation angles
        If dual_axis is True:
            (theta, phi) tuple — each (n_samples, n_features)
    """
    # Clamp to prevent extreme angles
    features_clamped = np.clip(features, CLAMP_RANGE[0], CLAMP_RANGE[1])

    if method == "arctan":
        # arctan maps R → (-π/2, π/2), then shift to [0, π]
        angles = np.arctan(features_clamped) + np.pi / 2
    elif method == "linear":
        # Linear scaling to [0, 2π]
        # Standardized features typically in [-3, 3], map to [0, 2π]
        angles = (features_clamped - CLAMP_RANGE[0]) / (CLAMP_RANGE[1] - CLAMP_RANGE[0]) * 2 * np.pi
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    if dual_axis:
        phi = angles + np.pi / 4  # quarter-phase offset for RZ axis
        return angles, phi
    return angles


def build_qrc_circuit(
    angles: np.ndarray,
    depth: int = 2,
    random_rotations: Optional[np.ndarray] = None,
    reupload: bool = False,
    phi_angles: Optional[np.ndarray] = None,
) -> "QuantumCircuit":
    """
    Build QRC circuit for a single sample.

    Architecture:
    - Layer 0: Initial RY encoding (+ optional RZ from phi_angles)
    - Layers 1..depth: [Random Rotations] + CZ ring
      + optional data re-upload (RY, and RZ if phi_angles given)

    Args:
        angles: (n_qubits,) array of RY rotation angles
        depth: Number of entangling layers
        random_rotations: Optional (depth, n_qubits, 3) array of fixed random angles
        reupload: If True, re-apply RY(angles) after each CZ ring layer
        phi_angles: Optional (n_qubits,) array of RZ rotation angles (dual-axis)

    Returns:
        QuantumCircuit ready for simulation
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit required for circuit construction")

    n_qubits = len(angles)
    qc = QuantumCircuit(n_qubits)

    # --- helper: encode RY (+ optional RZ) on all qubits ---
    def _apply_encoding():
        for i in range(n_qubits):
            qc.ry(angles[i], i)
        if phi_angles is not None:
            for i in range(n_qubits):
                qc.rz(phi_angles[i], i)

    # Depth 0: pure encoding (no entanglement)
    if depth == 0:
        _apply_encoding()
        return qc

    # Initial encoding
    _apply_encoding()

    # Entangling layers
    for layer in range(depth):
        # Apply random rotations to break symmetry (if provided)
        if random_rotations is not None:
            if layer < random_rotations.shape[0]:
                rot_layer = random_rotations[layer]
                for i in range(n_qubits):
                    qc.rx(rot_layer[i, 0], i)
                    qc.ry(rot_layer[i, 1], i)
                    qc.rz(rot_layer[i, 2], i)

        # CZ ring entanglement
        for i in range(n_qubits):
            qc.cz(i, (i + 1) % n_qubits)

        # Data re-uploading (optional)
        if reupload:
            _apply_encoding()

    return qc


def compute_expectation_values(
    qc: "QuantumCircuit",
    use_zz: bool = USE_ZZ_CORRELATORS,
    backend_type: str = "statevector",
    observable_set: str = "Z",
) -> np.ndarray:
    """
    Compute expectation values from the circuit statevector.

    Args:
        qc: Quantum circuit
        use_zz: Whether to include two-body correlators (for ``observable_set="Z"``)
        backend_type: "statevector" for exact, "qasm" for shot-based
        observable_set:
            ``"Z"``  — Z singles (6) + optional ZZ pairs (15) = 6 or 21
            ``"XYZ"`` — X/Y/Z singles (18) + all two-body Pauli pairs (135) = 153

    Returns:
        Array of expectation values
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit required for expectation values")

    n_qubits = qc.num_qubits
    sv = Statevector(qc)

    def _ev(pauli_str: str) -> float:
        op = SparsePauliOp.from_list([(pauli_str, 1.0)])
        return sv.expectation_value(op).real

    def _single_pauli_str(basis: str, qubit: int) -> str:
        chars = ["I"] * n_qubits
        # Qiskit Pauli strings are ordered right-to-left (qubit 0 is the
        # rightmost character). Index reversal maps logical qubit index
        # (0 = first qubit added to the circuit) to the correct string position.
        # So observable Z_0 in the paper corresponds to chars[n_qubits-1].
        chars[n_qubits - qubit - 1] = basis
        return "".join(chars)

    def _two_pauli_str(b1: str, q1: int, b2: str, q2: int) -> str:
        chars = ["I"] * n_qubits
        chars[n_qubits - q1 - 1] = b1  # same right-to-left Qiskit convention
        chars[n_qubits - q2 - 1] = b2
        return "".join(chars)

    expectations = []

    if observable_set == "XYZ":
        # --- Full multi-basis: X/Y/Z singles + all Pauli pairs ----------
        bases = ["X", "Y", "Z"]
        # Singles: 3 × n_qubits = 18
        for basis in bases:
            for i in range(n_qubits):
                expectations.append(_ev(_single_pauli_str(basis, i)))
        # Pairs: 9 combos × C(n_qubits,2) = 9×15 = 135
        for b1 in bases:
            for b2 in bases:
                for i, j in combinations(range(n_qubits), 2):
                    expectations.append(_ev(_two_pauli_str(b1, i, b2, j)))
    else:
        # --- Z-only (original behaviour) --------------------------------
        for i in range(n_qubits):
            expectations.append(_ev(_single_pauli_str("Z", i)))
        if use_zz:
            for i, j in combinations(range(n_qubits), 2):
                expectations.append(_ev(_two_pauli_str("Z", i, "Z", j)))

    return np.array(expectations)


def compute_reservoir_features(
    features: np.ndarray,
    depth: int = 2,
    use_zz: bool = USE_ZZ_CORRELATORS,
    random_rotations: Optional[np.ndarray] = None,
    verbose: bool = False,
    reupload: bool = False,
    dual_axis: bool = False,
    observable_set: str = "Z",
) -> np.ndarray:
    """
    Compute QRC reservoir features for a batch of samples.

    Args:
        features: (n_samples, 6) standardized input features
        depth: Circuit depth
        use_zz: Include two-body correlators (Z-only mode)
        random_rotations: Optional (depth, n_qubits, 3) array
        verbose: Print timing information
        reupload: Re-apply encoding after each entangling layer
        dual_axis: Use RY+RZ dual-axis encoding
        observable_set: ``"Z"`` or ``"XYZ"``

    Returns:
        reservoir_features: (n_samples, n_observables) array
    """
    import time

    n_samples = features.shape[0]
    start_time = time.time()

    # Encode all features
    if dual_axis:
        theta_all, phi_all = encode_features(features, dual_axis=True)
    else:
        theta_all = encode_features(features)
        phi_all = None
    encode_time = time.time() - start_time

    # Compute reservoir features for each sample
    reservoir_list = []
    circuit_times = []

    for i in range(n_samples):
        sample_start = time.time()
        phi_i = phi_all[i] if phi_all is not None else None
        qc = build_qrc_circuit(
            theta_all[i],
            depth=depth,
            random_rotations=random_rotations,
            reupload=reupload,
            phi_angles=phi_i,
        )
        exp_vals = compute_expectation_values(
            qc, use_zz=use_zz, observable_set=observable_set,
        )
        reservoir_list.append(exp_vals)
        circuit_times.append(time.time() - sample_start)

    total_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*50}")
        print(f"QRC TIMING REPORT")
        print(f"{'='*50}")
        print(f"Samples:          {n_samples}")
        print(f"Depth:            {depth}")
        print(f"Use ZZ:           {use_zz}")
        print(f"Reupload:         {reupload}")
        print(f"Dual axis:        {dual_axis}")
        print(f"Observable set:   {observable_set}")
        print(f"Encode time:      {encode_time:.4f}s")
        print(f"Per-sample avg:   {np.mean(circuit_times)*1000:.2f}ms")
        print(f"Per-sample std:   {np.std(circuit_times)*1000:.2f}ms")
        print(f"Total time:       {total_time:.2f}s")
        print(f"Throughput:       {n_samples/total_time:.1f} samples/sec")
        print(f"{'='*50}")

        hw_factor = 30
        hw_estimate = total_time * hw_factor
        print(f"\nHARDWARE ESTIMATION (×{hw_factor} factor):")
        print(f"  Estimated time:  {hw_estimate:.1f}s ({hw_estimate/60:.1f}min)")
        print(f"  With 8k shots:   {hw_estimate * 8192/1024:.1f}s ({hw_estimate * 8192/1024/60:.1f}min)")

    return np.array(reservoir_list)


# =============================================================================
# Classical Fallback (for development without Qiskit)
# =============================================================================

def compute_reservoir_features_classical(
    features: np.ndarray,
    depth: int = 2,
    use_zz: bool = False,
    random_rotations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Classical approximation of QRC reservoir features (for testing).
    
    Uses tanh nonlinearity to mimic quantum circuit behavior.
    """
    n_samples, n_features = features.shape
    
    # Encode
    angles = encode_features(features)
    
    # Simulate reservoir dynamics with classical nonlinearity
    h = np.sin(angles)  # Initial state
    
    for layer in range(depth):
        # Mix with random rotations (simulated as random scaling/shift)
        if random_rotations is not None and layer < random_rotations.shape[0]:
            # Use mean of rotations to shift phase
            shift = np.mean(random_rotations[layer], axis=1)
            h = h + shift[None, :]

        # "Entanglement": mix features
        h_mixed = np.roll(h, 1, axis=1) + h
        # Nonlinearity + re-encoding
        h = np.tanh(h_mixed) * np.cos(angles)
    
    reservoir = h
    
    # Add correlators if needed
    if use_zz:
        correlators = []
        for i, j in combinations(range(n_features), 2):
            correlators.append(reservoir[:, i] * reservoir[:, j])
        correlators = np.array(correlators).T
        reservoir = np.hstack([reservoir, correlators])
    
    return reservoir
