"""Model factories for Phase 3 classical baselines on EIS features.

All models are wrapped in Pipeline with StandardScaler for leakage-free
evaluation.

Primary baselines (paper-reported):
  Ridge, SVR, XGBoost, Linear-PC1, RFF (n=21, matched to QRC dim), ESN,
  GP (Zhang et al. 2020 NatComm approach), 1D-CNN (spectral baseline).

Supplementary:
  MLP (properly tuned with early stopping — demoted from primary after
  reviewer flag that a single hidden-layer MLP without tuning is unrepresentative).

RFF change (§7.3 reviewer): n_components fixed at 21 to match QRC reservoir
dimensionality (Z+ZZ observables on 6 qubits = 21 features). This isolates
the quantum contribution rather than comparing against a higher-dimensional
classical kernel.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from .config import RANDOM_STATE, RFF_N_COMPONENTS_MATCHED

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception as exc:
    warnings.warn(
        f"PyTorch unavailable or broken ({type(exc).__name__}: {exc}). "
        "CNN1DRegressor will fall back to SVR(rbf).",
        RuntimeWarning,
    )
    TORCH_AVAILABLE = False


class SimpleESNRegressor(BaseEstimator, RegressorMixin):
    """Echo State Network with random fixed weights and Ridge readout.

    Treats each sample's features as a length-T sequence of 1D inputs
    (T = n_features). Uses LSTM-style gating with random fixed weights.
    Final hidden state is passed to a trained Ridge readout.

    Note on feature ordering: EIS impedance features (or their PCA projections)
    are not temporally ordered within a single measurement — they are spectral
    (frequency-domain) values. The ESN processes them as an arbitrary sequence
    of scalar inputs rather than a true time series. The recurrent dynamics act
    as a nonlinear feature expansion (analogous to a random kernel), not a
    sequence model. This is intentional and consistent with how ESNs are used
    as classical reservoir computing baselines in the literature.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        ridge_alpha: float = 1.0,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_size = hidden_size
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        h = self.hidden_size

        scale_i = np.sqrt(2.0 / (n_features + h))
        scale_h = np.sqrt(2.0 / (h + h))

        self.W_i_ = rng.randn(4 * h, 1) * scale_i
        self.W_h_ = rng.randn(4 * h, h) * scale_h
        self.b_ = np.zeros(4 * h)
        self.b_[h:2 * h] = 1.0

        H = self._forward(X)

        from sklearn.linear_model import Ridge as RidgeReg
        self.readout_ = RidgeReg(alpha=self.ridge_alpha)
        self.readout_.fit(H, y)
        return self

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run the fixed recurrent dynamics and return final hidden states."""
        n_samples, n_features = X.shape
        h = self.hidden_size
        all_h = np.zeros((n_samples, h))

        for s in range(n_samples):
            h_t = np.zeros(h)
            c_t = np.zeros(h)

            for t in range(n_features):
                x_t = X[s, t:t + 1]
                gates = self.W_i_ @ x_t + self.W_h_ @ h_t + self.b_
                i_gate = self._sigmoid(gates[:h])
                f_gate = self._sigmoid(gates[h:2 * h])
                g_gate = np.tanh(gates[2 * h:3 * h])
                o_gate = self._sigmoid(gates[3 * h:4 * h])

                c_t = f_gate * c_t + i_gate * g_gate
                h_t = o_gate * np.tanh(c_t)

            all_h[s] = h_t
        return all_h

    def predict(self, X):
        H = self._forward(X)
        return self.readout_.predict(H)


class GaussianProcessRegressorWrapper(BaseEstimator, RegressorMixin):
    """GP regressor with Matérn-5/2 + WhiteKernel.

    Replicates the Zhang et al. (2020, Nat. Commun.) approach of applying a
    GP directly to impedance features without a fixed parametric form. The
    GP posterior provides natural uncertainty quantification.

    The kernel is: Matérn(nu=2.5) + WhiteKernel() so the model learns both
    the length-scale of the impedance feature space and the noise level.
    """

    def __init__(self, alpha: float = 1e-6, random_state: int = RANDOM_STATE):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=self.alpha)
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,           # numerical stability; noise handled by kernel
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp_.fit(X, y)
        return self

    def predict(self, X, return_std: bool = False):
        return self.gp_.predict(X, return_std=return_std)

    def predict_with_std(self, X):
        """Return (y_pred, y_std) for uncertainty quantification."""
        return self.gp_.predict(X, return_std=True)


class CNN1DRegressor(BaseEstimator, RegressorMixin):
    """1D-CNN treating EIS spectrum as a 1D signal.

    Architecture: Conv1D(16, k=3) → ReLU → GlobalAvgPool → Dense(1)
    This is the natural deep learning baseline for spectral data. The model
    treats the EIS feature vector as a 1D time-series where adjacent frequency
    bins are spatially correlated.

    Falls back to RBF-SVR if PyTorch is unavailable.
    """

    def __init__(
        self,
        n_filters: int = 16,
        kernel_size: int = 3,
        lr: float = 1e-3,
        n_epochs: int = 300,
        patience: int = 30,
        random_state: int = RANDOM_STATE,
    ):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

    def _build_model(self, n_features: int):
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv1d(1, self.n_filters, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.n_filters, 1),
        )
        return model

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            warnings.warn(
                "PyTorch not available; CNN1DRegressor falls back to SVR(rbf).",
                RuntimeWarning,
            )
            self.fallback_ = SVR(kernel="rbf", C=10.0, gamma="scale")
            self.fallback_.fit(X, y)
            return self

        import torch

        torch.manual_seed(self.random_state)
        n_samples, n_features = X.shape
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, F)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

        self.model_ = self._build_model(n_features)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        best_loss = float("inf")
        patience_count = 0
        self.best_state_ = None

        self.model_.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            pred = self.model_(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            val_loss = loss.item()
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                self.best_state_ = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        if self.best_state_ is not None:
            self.model_.load_state_dict(self.best_state_)
        self.model_.eval()
        return self

    def predict(self, X):
        if not TORCH_AVAILABLE or hasattr(self, "fallback_"):
            return self.fallback_.predict(X)

        import torch
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            return self.model_(X_t).squeeze(1).numpy()


def get_model_pipeline(model_name: str) -> Pipeline:
    """Get model wrapped in Pipeline with StandardScaler.

    RFF uses n_components=RFF_N_COMPONENTS_MATCHED (21) by default so that
    it has the same output dimensionality as the QRC Z+ZZ reservoir. This
    isolates the quantum contribution in the kernel comparison (reviewer §7.3).
    Pass model_name='rff_full' to get the original 100-component version for
    supplementary tables.
    """
    if model_name == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0)),
        ])

    if model_name == 'ridge_72d':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0)),
        ])

    if model_name == 'svr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)),
        ])

    if model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError('XGBoost not installed. pip install xgboost')
        return Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ])

    if model_name == 'esn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('esn', SimpleESNRegressor(
                hidden_size=32,
                ridge_alpha=1.0,
                random_state=RANDOM_STATE,
            )),
        ])

    if model_name == 'linear_pc1':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('select_pc1', FunctionTransformer(
                func=lambda X: X[:, 0:1], validate=False
            )),
            ('ridge', Ridge(alpha=1.0)),
        ])

    if model_name == 'rff':
        # Matched dimensionality: 21 components = QRC Z+ZZ reservoir (reviewer §7.3)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rff', RBFSampler(
                gamma=1.0,
                n_components=RFF_N_COMPONENTS_MATCHED,
                random_state=RANDOM_STATE,
            )),
            ('ridge', Ridge(alpha=1.0)),
        ])

    if model_name == 'rff_full':
        # Original 100-component RFF for supplementary comparison
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rff', RBFSampler(
                gamma=1.0,
                n_components=100,
                random_state=RANDOM_STATE,
            )),
            ('ridge', Ridge(alpha=1.0)),
        ])

    if model_name == 'gp':
        # Gaussian Process on EIS features — Zhang et al. (2020, Nat. Commun.) approach.
        # GP provides both point predictions and natural uncertainty quantification
        # via the posterior standard deviation.
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gp', GaussianProcessRegressorWrapper(
                alpha=1e-6,
                random_state=RANDOM_STATE,
            )),
        ])

    if model_name == 'cnn1d':
        # 1D-CNN treating EIS spectrum as a 1D signal (spectral deep-learning baseline).
        # Falls back to SVR(rbf) if PyTorch is not installed.
        return Pipeline([
            ('scaler', StandardScaler()),
            ('cnn1d', CNN1DRegressor(
                n_filters=16,
                kernel_size=3,
                lr=1e-3,
                n_epochs=300,
                patience=30,
                random_state=RANDOM_STATE,
            )),
        ])

    if model_name == 'mlp':
        # MLP: properly tuned with adam solver and early stopping.
        # Demoted to supplementary — reviewer flag: "tune it properly with
        # early stopping and dropout (so the comparison is fair) or remove it."
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                learning_rate_init=1e-3,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=RANDOM_STATE,
            )),
        ])

    raise ValueError(f'Unknown model: {model_name}')


def get_param_grid(model_name: str) -> dict:
    """Get hyperparameter grid for a model."""
    from .config import HYPERPARAM_GRIDS
    return HYPERPARAM_GRIDS.get(model_name, {})
