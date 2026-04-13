"""Quantum Reservoir Computing model (sklearn-compatible).

Implements QRC with:
1. Angle encoding
2. Fixed entangling circuit
3. Ridge readout with nested CV for alpha selection
"""

import warnings
import numpy as np
import sklearn
from typing import Optional, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# GridSearchCV.fit() correctly forwards the `groups` parameter only in
# sklearn >= 1.2. Older versions silently ignore it, which would break the
# nested LOCO-CV alpha selection path.
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
if _sklearn_version < (1, 2):
    warnings.warn(
        f"scikit-learn {sklearn.__version__} detected. Version >= 1.2 is "
        "required for GridSearchCV to forward `groups` to the CV splitter. "
        "LOCO alpha selection may be silently incorrect. "
        "Please upgrade: pip install 'scikit-learn>=1.2'.",
        UserWarning,
        stacklevel=1,
    )

from .config import (
    RIDGE_ALPHAS, RANDOM_STATE, USE_ZZ_CORRELATORS, DEFAULT_DEPTH, N_QUBITS,
    REUPLOAD, DUAL_AXIS, OBSERVABLE_SET,
)

try:
    from .circuit import compute_reservoir_features, QISKIT_AVAILABLE
except ImportError:
    QISKIT_AVAILABLE = False
    from .circuit import compute_reservoir_features_classical as compute_reservoir_features


class QuantumReservoir(BaseEstimator, RegressorMixin):
    """
    Quantum Reservoir Computing model for SOH prediction.

    Parameters:
        depth: Number of entangling layers (1-4)
        use_zz: Include two-body correlators ⟨Z_iZ_j⟩
        ridge_alpha: Ridge regularization (None = use CV)
        use_classical_fallback: Use classical simulation if True
        add_random_rotations: Add random fixed single-qubit rotations to break symmetry
        reupload: Data re-uploading after each CZ ring
        dual_axis: RY+RZ dual-axis encoding
        observable_set: ``"Z"`` or ``"XYZ"``
    """

    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        use_zz: bool = USE_ZZ_CORRELATORS,
        ridge_alpha: Optional[float] = None,
        use_classical_fallback: bool = False,
        add_random_rotations: bool = True,
        verbose: bool = False,
        reupload: bool = REUPLOAD,
        dual_axis: bool = DUAL_AXIS,
        observable_set: str = OBSERVABLE_SET,
    ):
        self.depth = depth
        self.use_zz = use_zz
        self.ridge_alpha = ridge_alpha
        self.use_classical_fallback = use_classical_fallback
        self.add_random_rotations = add_random_rotations
        self.verbose = verbose
        self.reupload = reupload
        self.dual_axis = dual_axis
        self.observable_set = observable_set

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """Compute reservoir features with standardization."""
        X_scaled = self.scaler_.transform(X)
        X_scaled = np.clip(X_scaled, -3.0, 3.0)

        if self.use_classical_fallback or not QISKIT_AVAILABLE:
            from .circuit import compute_reservoir_features_classical
            return compute_reservoir_features_classical(
                X_scaled, self.depth, self.use_zz, self.random_rotations_
            )
        else:
            return compute_reservoir_features(
                X_scaled, self.depth, self.use_zz, self.random_rotations_,
                verbose=self.verbose,
                reupload=self.reupload,
                dual_axis=self.dual_axis,
                observable_set=self.observable_set,
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """
        Fit QRC model.
        
        Args:
            X: (n_samples, 6) input features
            y: (n_samples,) SOH labels
            groups: Optional group labels for nested CV
        """
        # Initialize random rotations if enabled
        if self.add_random_rotations and self.depth > 0:
            rng = np.random.RandomState(RANDOM_STATE)
            # Shape: (depth, n_qubits, 3) for Rx, Ry, Rz
            self.random_rotations_ = rng.uniform(0, 2 * np.pi, (self.depth, N_QUBITS, 3))
        else:
            self.random_rotations_ = None

        # Fit scaler on training data
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        
        # Compute reservoir features
        reservoir_features = self._compute_features(X)
        
        # Ridge readout with alpha selection
        if self.ridge_alpha is None:
            # Nested CV for alpha selection
            if groups is not None and len(np.unique(groups)) >= 3:
                cv = LeaveOneGroupOut()
                grid = GridSearchCV(
                    Ridge(),
                    {"alpha": RIDGE_ALPHAS},
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )
                grid.fit(reservoir_features, y, groups=groups)
                self.readout_ = grid.best_estimator_
                self.best_alpha_ = grid.best_params_["alpha"]
            else:
                cv_folds = min(3, len(y))
                if cv_folds >= 2:
                    grid = GridSearchCV(
                        Ridge(),
                        {"alpha": RIDGE_ALPHAS},
                        cv=cv_folds,
                        scoring="neg_mean_absolute_error",
                        n_jobs=1,
                    )
                    grid.fit(reservoir_features, y)
                    self.readout_ = grid.best_estimator_
                    self.best_alpha_ = grid.best_params_["alpha"]
                else:
                    self.best_alpha_ = RIDGE_ALPHAS[0]
                    self.readout_ = Ridge(alpha=self.best_alpha_)
                    self.readout_.fit(reservoir_features, y)
        else:
            self.readout_ = Ridge(alpha=self.ridge_alpha)
            self.readout_.fit(reservoir_features, y)
            self.best_alpha_ = self.ridge_alpha
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict SOH values."""
        reservoir_features = self._compute_features(X)
        return self.readout_.predict(reservoir_features)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        random_state: int = RANDOM_STATE,
    ) -> dict:
        """Bootstrap prediction intervals for SOH estimates.

        Reviewer §3.4: "Battery SOH estimation in real BMS requires confidence
        intervals. Your Ridge readout provides point predictions only. Nature
        Communications battery papers increasingly include uncertainty."

        Implements paired bootstrap: resample (X_train, y_train) with replacement,
        refit the Ridge readout (reservoir features are fixed — only readout varies),
        and collect predictions across bootstrap replicates.

        Args:
            X: Input features (n_samples, n_features).
            n_bootstrap: Number of bootstrap replicates.
            ci: Confidence interval coverage (default 0.95 → 95% CI).
            random_state: RNG seed for reproducibility.

        Returns:
            dict with keys:
                y_pred:     Point prediction (mean of bootstrap replicates).
                y_lower:    Lower CI bound.
                y_upper:    Upper CI bound.
                y_std:      Standard deviation of bootstrap predictions.
                boot_preds: Array (n_bootstrap, n_samples) — all replicates.

        Note: Requires fit_and_store() instead of fit(). fit_and_store()
        caches the training reservoir features needed for bootstrap.
        """
        if not hasattr(self, "X_train_reservoir_") or not hasattr(self, "y_train_"):
            raise RuntimeError(
                "predict_with_uncertainty() requires cached training data. "
                "Use fit_and_store(X, y) instead of fit(X, y)."
            )

        rng = np.random.RandomState(random_state)
        X_res = self._compute_features(X)
        n_train = len(self.y_train_)
        boot_preds = np.empty((n_bootstrap, len(X)), dtype=float)

        for b in range(n_bootstrap):
            idx = rng.randint(0, n_train, size=n_train)
            X_b = self.X_train_reservoir_[idx]
            y_b = self.y_train_[idx]
            # Refit only the linear readout (cheap)
            readout_b = Ridge(alpha=self.best_alpha_)
            readout_b.fit(X_b, y_b)
            boot_preds[b] = readout_b.predict(X_res)

        alpha = (1.0 - ci) / 2.0
        return {
            "y_pred":     boot_preds.mean(axis=0),
            "y_lower":    np.percentile(boot_preds, 100 * alpha, axis=0),
            "y_upper":    np.percentile(boot_preds, 100 * (1 - alpha), axis=0),
            "y_std":      boot_preds.std(axis=0),
            "boot_preds": boot_preds,
        }

    def fit_and_store(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ):
        """Fit the model and cache training reservoir features for bootstrap.

        Use this instead of fit() when you need predict_with_uncertainty().
        """
        self.fit(X, y, groups=groups)
        # Cache reservoir features of training data (fixed after fit)
        self.X_train_reservoir_ = self._compute_features(X)
        self.y_train_ = y.copy()
        return self

    def get_reservoir_dim(self) -> int:
        """Get dimensionality of reservoir features."""
        n = N_QUBITS
        n_pairs = n * (n - 1) // 2
        if self.observable_set == "XYZ":
            return 3 * n + 9 * n_pairs  # 18 + 135 = 153
        elif self.use_zz:
            return n + n_pairs           # 6 + 15 = 21
        else:
            return n                      # 6


class QRCPipeline(BaseEstimator, RegressorMixin):
    """
    Complete QRC pipeline with preprocessing.
    
    Ensures fair comparison with Phase 3 classical models.
    """
    
    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        use_zz: bool = USE_ZZ_CORRELATORS, # Use default from config
        use_classical_fallback: bool = False,
        add_random_rotations: bool = True,
    ):
        self.depth = depth
        self.use_zz = use_zz
        self.use_classical_fallback = use_classical_fallback
        self.add_random_rotations = add_random_rotations
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Fit QRC pipeline."""
        self.qrc_ = QuantumReservoir(
            depth=self.depth,
            use_zz=self.use_zz,
            ridge_alpha=None,  # Use CV
            use_classical_fallback=self.use_classical_fallback,
            add_random_rotations=self.add_random_rotations,
        )
        self.qrc_.fit(X, y, groups)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict SOH values."""
        return self.qrc_.predict(X)
