"""
Data loader for the Warwick DIB dataset.

Reference:
    Rashid et al. (2023). "Dataset for rapid state of health estimation of
    lithium batteries using EIS and machine learning: Training and validation."
    Data in Brief, 48, 109157.
    DOI: 10.17632/mn9fb7xdx6.3  (Mendeley Data, CC0 1.0)

Dataset summary:
    - 25 NMC 811 cylindrical 21700 cells (5 Ah nominal, INR21700-50E)
    - 5 SOH target levels: 100%, 95%, 90%, 85%, 80%  (5 cells each)
    - EIS measured at 3 temperatures × 5 SOC conditions = 15 spectra / cell
    - 61 log-spaced frequencies: 10 kHz → 10 mHz
    - .mat files: columns = [freq_Hz, Re(Z_Ohm), Im(Z_Ohm)]
    - Actual measured SOH encoded in filename (e.g. "9505" → 95.05%)

This loader extracts the condition:  25 °C, 50 % SOC  (24 files)
and returns a dict compatible with the Stanford pipeline:
    {cell_id: {"X_raw": (1, 122), "y": (1,), "freq": (61,)}}

Feature vector = [Re(Z) × 61, Im(Z) × 61] = 122-dimensional
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .config import WARWICK_DATA_DIR
except ImportError:
    from config import WARWICK_DATA_DIR

# ── Constants ──────────────────────────────────────────────────────────────────
EIS_DIR = WARWICK_DATA_DIR / ".matfiles" / "EIS_Test"

# Target condition (matches Stanford at ~50 % SOC, room temperature)
TARGET_TEMP = "25degC"
TARGET_SOC  = "50SOC"

# Nominal capacity of NMC 811 INR21700-50E cell (mAh → Ah)
WARWICK_NOMINAL_CAP_AH = 5.0


def _parse_filename(fname: str) -> dict | None:
    """Extract cell_id, soh_nominal, temp, soc, soh_measured from filename."""
    # e.g.  Cell02_95SOH_25degC_50SOC_9505.mat
    m = re.match(
        r"(Cell\d+)_(\d+)SOH_([\ddeg]+C)_(\d+SOC)_(\d+)\.mat",
        fname,
    )
    if m is None:
        return None
    cell_id     = m.group(1)
    soh_nominal = int(m.group(2))         # 80, 85, 90, 95, 100
    temp        = m.group(3)              # "25degC"
    soc         = m.group(4)             # "50SOC"
    soh_code    = m.group(5)             # "9505" → 95.05 %
    # Decode: 4-digit "9505" → 95.05%, 5-digit "10000" → 100.00%
    if len(soh_code) == 5:
        soh_measured = int(soh_code[:3]) + int(soh_code[3:]) * 0.01
    else:
        soh_measured = int(soh_code[:2]) + int(soh_code[2:]) * 0.01
    return dict(
        cell_id=cell_id,
        soh_nominal=soh_nominal,
        temp=temp,
        soc=soc,
        soh_measured=soh_measured,
    )


def load_warwick_data(
    temp: str = TARGET_TEMP,
    soc: str  = TARGET_SOC,
    eis_dir: Path | None = None,
) -> dict[str, dict]:
    """
    Load Warwick DIB EIS data for a given temperature/SOC condition.

    Returns
    -------
    data : dict
        {cell_id: {
            "X_raw":  ndarray (1, 122)  — [Re(Z)×61 | Im(Z)×61],
            "y":      ndarray (1,)      — SOH fraction [0, 1],
            "freq":   ndarray (61,)     — frequency in Hz,
            "soh_nominal": int          — target SOH % (80/85/90/95/100),
        }}
    """
    if eis_dir is None:
        eis_dir = EIS_DIR

    eis_dir = Path(eis_dir)
    if not eis_dir.exists():
        raise FileNotFoundError(f"EIS directory not found: {eis_dir}")

    data: dict[str, dict] = {}

    for mat_file in sorted(eis_dir.glob("*.mat")):
        info = _parse_filename(mat_file.name)
        if info is None:
            continue
        if info["temp"] != temp or info["soc"] != soc:
            continue

        mat = loadmat(str(mat_file))
        arr = mat["data"]          # shape (61, 3): [freq, Re(Z), Im(Z)]

        freq  = arr[:, 0].astype(np.float64)    # (61,)
        re_z  = arr[:, 1].astype(np.float64)    # (61,)
        im_z  = arr[:, 2].astype(np.float64)    # (61,)

        # Feature vector: concatenate Re and Im  → 122-dim
        X_raw = np.concatenate([re_z, im_z])[np.newaxis, :]  # (1, 122)

        # SOH as fraction
        soh_frac = info["soh_measured"] / 100.0

        cell_id = info["cell_id"]
        data[cell_id] = {
            "X_raw":       X_raw,
            "y":           np.array([soh_frac]),
            "freq":        freq,
            "soh_nominal": info["soh_nominal"],
        }

    if not data:
        raise RuntimeError(
            f"No EIS files found for condition {temp}/{soc} in {eis_dir}"
        )

    return data


def get_warwick_arrays(
    temp: str = TARGET_TEMP,
    soc: str  = TARGET_SOC,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convenience wrapper returning flat arrays ready for LOCO.

    Returns
    -------
    X : ndarray (N, 122)   — raw EIS features
    y : ndarray (N,)       — SOH fractions
    cell_ids : list[str]   — one entry per sample (= one per cell here)
    """
    data = load_warwick_data(temp=temp, soc=soc)
    cell_ids = sorted(data.keys())

    X = np.vstack([data[c]["X_raw"] for c in cell_ids])   # (N, 122)
    y = np.concatenate([data[c]["y"] for c in cell_ids])  # (N,)

    return X, y, cell_ids


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, ids = get_warwick_arrays()
    print(f"Loaded {len(ids)} cells")
    print(f"X shape : {X.shape}   (N_cells × 122-dim EIS features)")
    print(f"y range : {y.min():.4f} – {y.max():.4f}  (SOH fraction)")
    print(f"SOH values (%):")
    for c, soh in zip(ids, y * 100):
        print(f"  {c}: {soh:.2f}%")
