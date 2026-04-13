"""Shared helpers for Phase 12 ECM-inspired feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data_loader_warwick import load_warwick_data
from phase_12.config import ECM_PROXY_FEATURES, WARWICK_SOC, WARWICK_TEMP


def load_warwick_impedance_records(
    temp: str = WARWICK_TEMP,
    soc: str = WARWICK_SOC,
) -> list[dict[str, Any]]:
    """Load Warwick cell records with split impedance arrays."""
    raw = load_warwick_data(temp=temp, soc=soc)
    records: list[dict[str, Any]] = []

    for cell_id in sorted(raw):
        rec = raw[cell_id]
        freq = np.asarray(rec["freq"], dtype=float)
        x_raw = np.asarray(rec["X_raw"][0], dtype=float)
        n_freq = len(freq)
        re_z = x_raw[:n_freq]
        im_z = x_raw[n_freq:]

        # Normalize ordering once so downstream stages can assume descending freq.
        order = np.argsort(freq)[::-1]
        freq = freq[order]
        re_z = re_z[order]
        im_z = im_z[order]

        records.append(
            {
                "cell_id": cell_id,
                "soh_frac": float(rec["y"][0]),
                "soh_pct": float(rec["y"][0] * 100.0),
                "freq": freq,
                "re_z": re_z,
                "im_z": im_z,
            }
        )

    return records


def build_readiness_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize ECM-readiness diagnostics for each Warwick cell."""
    if not records:
        return pd.DataFrame()

    ref_freq = records[0]["freq"]
    rows = []
    for rec in records:
        freq = rec["freq"]
        re_z = rec["re_z"]
        im_z = rec["im_z"]
        rows.append(
            {
                "cell_id": rec["cell_id"],
                "soh_pct": rec["soh_pct"],
                "n_freq": int(len(freq)),
                "freq_descending": bool(np.all(np.diff(freq) < 0)),
                "freq_matches_reference": bool(np.allclose(freq, ref_freq)),
                "re_all_finite": bool(np.isfinite(re_z).all()),
                "im_all_finite": bool(np.isfinite(im_z).all()),
                "re_nonnegative_fraction": float(np.mean(re_z >= 0.0)),
                "r_ohm_ohm": float(re_z[0]),
                "r_lowfreq_ohm": float(re_z[-1]),
                "delta_r_ohm": float(re_z[-1] - re_z[0]),
                "im_abs_peak_ohm": float(np.max(np.abs(im_z))),
                "imag_sign_changes": int(np.sum(np.diff(np.signbit(im_z)) != 0)),
            }
        )

    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def extract_ecm_proxy_features(
    freq: np.ndarray,
    re_z: np.ndarray,
    im_z: np.ndarray,
) -> dict[str, float]:
    """Extract compact ECM-inspired features from one EIS spectrum.

    This is not full nonlinear equivalent-circuit fitting. It is a compact
    physics-informed summary intended to scaffold a reviewer-facing ECM phase.
    """
    abs_im = np.abs(im_z)
    peak_idx = int(np.argmax(abs_im))
    peak_freq_hz = float(freq[peak_idx])
    tau_peak_s = float(1.0 / (2.0 * np.pi * peak_freq_hz))
    r_ohm = float(re_z[0])
    r_lowfreq = float(re_z[-1])
    delta_r = float(r_lowfreq - r_ohm)

    if delta_r > 1e-12:
        c_pseudo_f = float(tau_peak_s / delta_r)
    else:
        c_pseudo_f = float("nan")

    logf_asc = np.log10(freq[::-1])
    abs_im_asc = abs_im[::-1]
    area_abs_im = float(np.trapz(abs_im_asc, x=logf_asc))

    low_k = min(5, len(freq))
    x_low = np.log10(freq[-low_k:][::-1])
    y_low = re_z[-low_k:][::-1]
    if len(x_low) >= 2 and np.ptp(x_low) > 0:
        lowfreq_re_slope = float(np.polyfit(x_low, y_low, deg=1)[0])
    else:
        lowfreq_re_slope = 0.0

    return {
        "r_ohm_ohm": r_ohm,
        "r_lowfreq_ohm": r_lowfreq,
        "delta_r_ohm": delta_r,
        "im_abs_peak_ohm": float(abs_im[peak_idx]),
        "peak_freq_hz": peak_freq_hz,
        "tau_peak_s": tau_peak_s,
        "c_pseudo_f": c_pseudo_f,
        "area_abs_im_ohm_loghz": area_abs_im,
        "lowfreq_re_slope": lowfreq_re_slope,
    }


def build_feature_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract ECM-inspired features for all Warwick cells."""
    rows = []
    for rec in records:
        features = extract_ecm_proxy_features(rec["freq"], rec["re_z"], rec["im_z"])
        row = {
            "cell_id": rec["cell_id"],
            "soh_frac": rec["soh_frac"],
            "soh_pct": rec["soh_pct"],
        }
        row.update(features)
        rows.append(row)

    cols = ["cell_id", "soh_frac", "soh_pct", *ECM_PROXY_FEATURES]
    return pd.DataFrame(rows, columns=cols).sort_values("cell_id").reset_index(drop=True)
