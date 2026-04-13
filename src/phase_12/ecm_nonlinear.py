"""Nonlinear equivalent-circuit fitting helpers for Phase 12."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import least_squares


ECM_MODEL_NAMES = ["L_R_Rc", "L_R_Rcpe", "L_R_Rcpe_w"]
ECM_PARAMETER_COLUMNS = [
    "L_h",
    "r0_ohm",
    "r1_ohm",
    "c1_f",
    "q1",
    "alpha1",
    "warburg_sigma",
]


def _z_parallel_rc(omega: np.ndarray, r1: float, c1: float) -> np.ndarray:
    admittance = (1.0 / r1) + 1j * omega * c1
    return 1.0 / admittance


def _z_parallel_rcpe(omega: np.ndarray, r1: float, q1: float, alpha1: float) -> np.ndarray:
    admittance = (1.0 / r1) + q1 * (1j * omega) ** alpha1
    return 1.0 / admittance


def _z_warburg(omega: np.ndarray, sigma: float) -> np.ndarray:
    return sigma / np.sqrt(1j * omega)


def model_impedance(freq_hz: np.ndarray, model_name: str, params: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * freq_hz

    if model_name == "L_R_Rc":
        l_h, r0, r1, c1 = params
        return 1j * omega * l_h + r0 + _z_parallel_rc(omega, r1, c1)

    if model_name == "L_R_Rcpe":
        l_h, r0, r1, q1, alpha1 = params
        return 1j * omega * l_h + r0 + _z_parallel_rcpe(omega, r1, q1, alpha1)

    if model_name == "L_R_Rcpe_w":
        l_h, r0, r1, q1, alpha1, sigma = params
        return 1j * omega * l_h + r0 + _z_parallel_rcpe(omega, r1, q1, alpha1) + _z_warburg(omega, sigma)

    raise ValueError(f"Unknown ECM model: {model_name}")


def _bounds_for_model(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "L_R_Rc":
        lower = np.array([0.0, 1e-6, 1e-6, 1e-6], dtype=float)
        upper = np.array([1e-3, 1.0, 1.0, 1e4], dtype=float)
        return lower, upper

    if model_name == "L_R_Rcpe":
        lower = np.array([0.0, 1e-6, 1e-6, 1e-6, 0.3], dtype=float)
        upper = np.array([1e-3, 1.0, 1.0, 1e4, 1.0], dtype=float)
        return lower, upper

    if model_name == "L_R_Rcpe_w":
        lower = np.array([0.0, 1e-6, 1e-6, 1e-6, 0.3, 0.0], dtype=float)
        upper = np.array([1e-3, 1.0, 1.0, 1e4, 1.0, 1.0], dtype=float)
        return lower, upper

    raise ValueError(f"Unknown ECM model: {model_name}")


def _initial_guess_bank(freq_hz: np.ndarray, re_z: np.ndarray, im_z: np.ndarray, model_name: str) -> list[np.ndarray]:
    omega = 2.0 * np.pi * freq_hz
    r0_est = float(np.clip(re_z[0], 1e-5, 0.5))
    r1_est = float(np.clip(max(re_z[-1] - re_z[0], 5e-4), 1e-5, 0.5))
    l_est = float(np.clip(max(im_z[0], 0.0) / omega[0], 1e-10, 1e-4))

    peak_idx = int(np.argmax(np.abs(im_z)))
    tau_est = float(np.clip(1.0 / (2.0 * np.pi * freq_hz[peak_idx]), 1e-8, 1e3))
    c_est = float(np.clip(tau_est / max(r1_est, 1e-6), 1e-6, 1e4))
    q_est = c_est
    sigma_est = float(np.clip(np.abs(im_z[-1]) * np.sqrt(omega[-1]), 1e-6, 0.5))

    if model_name == "L_R_Rc":
        return [
            np.array([l_est, r0_est, r1_est, c_est], dtype=float),
            np.array([max(l_est * 3.0, 1e-9), r0_est * 0.9, r1_est * 1.5, c_est * 0.5], dtype=float),
            np.array([1e-8, r0_est * 1.1, max(r1_est * 0.6, 1e-4), c_est * 2.0], dtype=float),
        ]

    if model_name == "L_R_Rcpe":
        return [
            np.array([l_est, r0_est, r1_est, q_est, 0.95], dtype=float),
            np.array([max(l_est * 3.0, 1e-9), r0_est * 0.9, r1_est * 1.5, q_est * 0.5, 0.8], dtype=float),
            np.array([1e-8, r0_est * 1.1, max(r1_est * 0.6, 1e-4), q_est * 2.0, 0.65], dtype=float),
        ]

    if model_name == "L_R_Rcpe_w":
        return [
            np.array([l_est, r0_est, r1_est, q_est, 0.95, sigma_est], dtype=float),
            np.array([max(l_est * 3.0, 1e-9), r0_est * 0.9, r1_est * 1.5, q_est * 0.5, 0.8, sigma_est * 2.0], dtype=float),
            np.array([1e-8, r0_est * 1.1, max(r1_est * 0.6, 1e-4), q_est * 2.0, 0.65, sigma_est * 0.5], dtype=float),
        ]

    raise ValueError(f"Unknown ECM model: {model_name}")


def fit_ecm_model(freq_hz: np.ndarray, re_z: np.ndarray, im_z: np.ndarray, model_name: str) -> dict:
    lower, upper = _bounds_for_model(model_name)
    starts = _initial_guess_bank(freq_hz, re_z, im_z, model_name)
    n_obs = int(2 * len(freq_hz))
    best = None

    def residuals(params: np.ndarray) -> np.ndarray:
        z_fit = model_impedance(freq_hz, model_name, params)
        return np.concatenate([z_fit.real - re_z, z_fit.imag - im_z])

    for x0 in starts:
        try:
            result = least_squares(
                residuals,
                x0=x0,
                bounds=(lower, upper),
                method="trf",
                max_nfev=8000,
            )
        except Exception:
            continue

        sse = float(np.sum(result.fun ** 2))
        if not np.isfinite(sse):
            continue

        z_fit = model_impedance(freq_hz, model_name, result.x)
        k = len(result.x)
        rmse = float(np.sqrt(sse / n_obs))
        aic = float(n_obs * np.log((sse / n_obs) + 1e-24) + 2 * k)
        bic = float(n_obs * np.log((sse / n_obs) + 1e-24) + k * np.log(n_obs))
        candidate = {
            "model_name": model_name,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "sse": sse,
            "rmse_ohm": rmse,
            "aic": aic,
            "bic": bic,
            "params": result.x.copy(),
            "z_fit": z_fit,
        }
        if best is None or candidate["aic"] < best["aic"]:
            best = candidate

    if best is None:
        raise RuntimeError(f"All fits failed for model {model_name}")

    return best


def params_to_feature_row(model_name: str, params: np.ndarray) -> dict[str, float]:
    row = {name: np.nan for name in ECM_PARAMETER_COLUMNS}

    if model_name == "L_R_Rc":
        row.update(
            {
                "L_h": float(params[0]),
                "r0_ohm": float(params[1]),
                "r1_ohm": float(params[2]),
                "c1_f": float(params[3]),
            }
        )
        return row

    if model_name == "L_R_Rcpe":
        row.update(
            {
                "L_h": float(params[0]),
                "r0_ohm": float(params[1]),
                "r1_ohm": float(params[2]),
                "q1": float(params[3]),
                "alpha1": float(params[4]),
            }
        )
        return row

    if model_name == "L_R_Rcpe_w":
        row.update(
            {
                "L_h": float(params[0]),
                "r0_ohm": float(params[1]),
                "r1_ohm": float(params[2]),
                "q1": float(params[3]),
                "alpha1": float(params[4]),
                "warburg_sigma": float(params[5]),
            }
        )
        return row

    raise ValueError(f"Unknown ECM model: {model_name}")
