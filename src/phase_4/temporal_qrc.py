"""Temporal Quantum Reservoir Computing on raw 72D EIS features.

Input: 72D raw EIS vector (36 Re + 36 Im) — NO PCA compression.
Reshape to (36, 2): [Re(Z_f), Im(Z_f)] per frequency.
At each frequency step, encode 2D -> 6 qubits, evolve, measure.
Concatenate all snapshot observables (after washout) -> Ridge readout.
"""

import numpy as np
from typing import Optional
from itertools import combinations
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .config import (
    N_QUBITS, CLAMP_RANGE, ENCODING_METHOD, USE_ZZ_CORRELATORS, RANDOM_STATE,
)

# ── Temporal QRC defaults ────────────────────────────────────────────────
N_FREQ_STEPS = 36          # 72D -> 36 frequency steps x 2 (Re, Im)
WASHOUT_STEPS = 3           # discard first 3 transient steps
RIDGE_ALPHAS_TEMPORAL = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _encode_2d_to_angles(re_val: float, im_val: float) -> np.ndarray:
    """Map 2 real values to 6 qubit RY angles.

    Qubits 0-2 encode Re-derived angles, qubits 3-5 encode Im-derived.
    Uses arctan encoding with linear, quadratic, and cross terms.
    """
    re_c = np.clip(re_val, CLAMP_RANGE[0], CLAMP_RANGE[1])
    im_c = np.clip(im_val, CLAMP_RANGE[0], CLAMP_RANGE[1])

    # Re-derived: linear, squared, cubic
    a0 = np.arctan(re_c) + np.pi / 2
    a1 = np.arctan(re_c ** 2 / 3.0) + np.pi / 2
    a2 = np.arctan(re_c * im_c / 3.0) + np.pi / 2
    # Im-derived: linear, squared, cubic
    a3 = np.arctan(im_c) + np.pi / 2
    a4 = np.arctan(im_c ** 2 / 3.0) + np.pi / 2
    a5 = np.arctan((re_c + im_c) / 3.0) + np.pi / 2

    return np.array([a0, a1, a2, a3, a4, a5])


def _measure_observables(sv: "Statevector", n_qubits: int, use_zz: bool) -> np.ndarray:
    """Extract Z (and optionally ZZ) expectation values from statevector."""
    expectations = []

    for i in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[n_qubits - i - 1] = "Z"
        op = SparsePauliOp.from_list([("".join(chars), 1.0)])
        expectations.append(sv.expectation_value(op).real)

    if use_zz:
        for i, j in combinations(range(n_qubits), 2):
            chars = ["I"] * n_qubits
            chars[n_qubits - i - 1] = "Z"
            chars[n_qubits - j - 1] = "Z"
            op = SparsePauliOp.from_list([("".join(chars), 1.0)])
            expectations.append(sv.expectation_value(op).real)

    return np.array(expectations)


class TemporalQuantumReservoir(BaseEstimator, RegressorMixin):
    """Temporal QRC operating on raw 72D EIS (no PCA).

    At each of the 36 frequency steps the 2D input is encoded into
    6 qubits, followed by a CZ-ring entangling layer.  One cumulative
    circuit is built; the statevector is snapshotted at each step.
    After discarding the first ``washout`` transient steps, all snapshot
    observables are concatenated into a single feature vector and
    fed to a Ridge readout.

    Parameters
    ----------
    depth : int
        CZ-ring layers per frequency step (default 1).
    use_zz : bool
        Include ZZ two-body correlators (21 obs per step, else 6).
    washout : int
        Number of initial frequency steps to discard.
    ridge_alpha : float or None
        Fixed regularisation; ``None`` triggers CV selection.
    """

    def __init__(
        self,
        depth: int = 1,
        use_zz: bool = USE_ZZ_CORRELATORS,
        washout: int = WASHOUT_STEPS,
        ridge_alpha: Optional[float] = None,
    ):
        self.depth = depth
        self.use_zz = use_zz
        self.washout = washout
        self.ridge_alpha = ridge_alpha

    # ── reservoir feature computation ────────────────────────────────────

    def _compute_temporal_features(self, X_72d: np.ndarray) -> np.ndarray:
        """Build temporal reservoir features for a batch of 72D samples."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for TemporalQuantumReservoir")

        n_qubits = N_QUBITS
        n_samples = X_72d.shape[0]
        obs_dim = n_qubits + (n_qubits * (n_qubits - 1) // 2 if self.use_zz else 0)
        active_steps = N_FREQ_STEPS - self.washout
        total_dim = obs_dim * active_steps

        all_features = np.zeros((n_samples, total_dim))

        for s in range(n_samples):
            x = X_72d[s]
            # Reshape to (36, 2): column 0 = Re, column 1 = Im
            pairs = x.reshape(N_FREQ_STEPS, 2)

            qc = QuantumCircuit(n_qubits)
            snapshot_obs = []

            for t in range(N_FREQ_STEPS):
                re_val, im_val = pairs[t]
                angles = _encode_2d_to_angles(
                    self.scaler_.transform([[re_val, im_val]])[0, 0],
                    self.scaler_.transform([[re_val, im_val]])[0, 1],
                ) if hasattr(self, '_pair_scaler') else _encode_2d_to_angles(re_val, im_val)

                # Encode
                for i in range(n_qubits):
                    qc.ry(float(angles[i]), i)

                # Entangling layers
                for _ in range(self.depth):
                    for i in range(n_qubits):
                        qc.cz(i, (i + 1) % n_qubits)

                # Snapshot
                sv = Statevector(qc)
                obs = _measure_observables(sv, n_qubits, self.use_zz)
                snapshot_obs.append(obs)

            # Discard washout, concatenate
            kept = snapshot_obs[self.washout:]
            all_features[s] = np.concatenate(kept)

        return all_features

    # ── sklearn interface ────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Fit temporal QRC.

        Args:
            X: (n_samples, 72) raw EIS features
            y: (n_samples,) SOH labels
            groups: Optional group labels for nested CV
        """
        assert X.shape[1] == 72, f"Expected 72D input, got {X.shape[1]}D"

        # Standardise the 72D input
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Compute temporal reservoir features
        reservoir = self._compute_temporal_features(X_scaled)

        # Ridge readout with CV alpha selection
        alphas = RIDGE_ALPHAS_TEMPORAL
        if self.ridge_alpha is not None:
            self.readout_ = Ridge(alpha=self.ridge_alpha)
            self.readout_.fit(reservoir, y)
            self.best_alpha_ = self.ridge_alpha
        else:
            grid = GridSearchCV(
                Ridge(), {"alpha": alphas},
                cv=min(3, len(y)),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(reservoir, y)
            self.readout_ = grid.best_estimator_
            self.best_alpha_ = grid.best_params_["alpha"]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict SOH from 72D EIS."""
        X_scaled = self.scaler_.transform(X)
        reservoir = self._compute_temporal_features(X_scaled)
        return self.readout_.predict(reservoir)

    def get_reservoir_dim(self) -> int:
        """Dimensionality of concatenated temporal features."""
        obs_dim = N_QUBITS + (N_QUBITS * (N_QUBITS - 1) // 2 if self.use_zz else 0)
        return obs_dim * (N_FREQ_STEPS - self.washout)
