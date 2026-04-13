"""Load Stanford SECL diagnostic .mat files into per-cell arrays."""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .config import CELL_IDS, NOMINAL_CAPACITY, RAW_DIAG_DIR
except ImportError:
    from config import CELL_IDS, NOMINAL_CAPACITY, RAW_DIAG_DIR


def _to_str(x) -> str:
    """Extract a clean scalar string from MATLAB-loaded nested objects."""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        return str(x.flatten()[0]).strip()
    return str(x).strip()


def _is_valid_eis_entry(entry) -> bool:
    """True when an EIS object entry is a non-missing (19, 3) matrix."""
    if not isinstance(entry, np.ndarray):
        return False
    if entry.shape != (19, 3):
        return False
    if np.isnan(entry).any():
        return False
    return True


def _is_valid_cap_entry(entry) -> bool:
    """True when a capacity entry contains a real discharge curve."""
    if not isinstance(entry, np.ndarray):
        return False
    if entry.size <= 1:
        return False
    if np.isnan(entry).all():
        return False
    return True


def load_stanford_data(
    data_dir: Path | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load Stanford SECL EIS+capacity data into per-cell records.

    Returns:
        Dict keyed by cell ID:
        {
            "W8": {
                "X_raw": np.ndarray (n_diag, 38),
                "y": np.ndarray (n_diag,),
                "block_ids": np.ndarray (n_diag,),
                "freq": np.ndarray (19,),
            },
            ...
        }
    """
    diag_dir = RAW_DIAG_DIR if data_dir is None else Path(data_dir) / "diagnostic_test"
    eis_path = diag_dir / "EIS_test.mat"
    cap_path = diag_dir / "capacity_test.mat"

    if not eis_path.exists():
        raise FileNotFoundError(f"Missing EIS file: {eis_path}")
    if not cap_path.exists():
        raise FileNotFoundError(f"Missing capacity file: {cap_path}")

    eis = loadmat(eis_path)
    cap = loadmat(cap_path)

    eis_labels = [_to_str(x) for x in eis["col_cell_label"][0]]
    cap_labels = [_to_str(x) for x in cap["col_cell_label"][0]]
    diag_numbers = np.asarray(eis["row_diag_number"]).astype(int).flatten()

    out: Dict[str, Dict[str, np.ndarray]] = {}

    for cell_id in CELL_IDS:
        if cell_id not in eis_labels:
            raise KeyError(f"Cell {cell_id} not found in EIS_test.mat labels")
        if cell_id not in cap_labels:
            raise KeyError(f"Cell {cell_id} not found in capacity_test.mat labels")

        eis_col = eis_labels.index(cell_id)
        cap_col = cap_labels.index(cell_id)

        features = []
        targets = []
        block_ids = []
        freq_ref = None

        for diag_idx in range(len(diag_numbers)):
            re_entry = eis["re_z"][diag_idx, eis_col]
            im_entry = eis["im_z"][diag_idx, eis_col]
            fr_entry = eis["freq"][diag_idx, eis_col]
            cap_entry = cap["cap"][diag_idx, cap_col]

            valid_eis = (
                _is_valid_eis_entry(re_entry)
                and _is_valid_eis_entry(im_entry)
                and _is_valid_eis_entry(fr_entry)
            )
            valid_cap = _is_valid_cap_entry(cap_entry)

            if not (valid_eis and valid_cap):
                continue

            # 50% SOC is column index 1.
            re_50 = re_entry[:, 1].astype(float)
            im_50 = im_entry[:, 1].astype(float)
            feature_vec = np.concatenate([re_50, im_50], axis=0)

            cap_curve = np.asarray(cap_entry).astype(float).flatten()
            soh = float(np.nanmax(cap_curve) / NOMINAL_CAPACITY)

            features.append(feature_vec)
            targets.append(soh)
            block_ids.append(int(diag_numbers[diag_idx]))

            if freq_ref is None:
                freq_ref = fr_entry[:, 1].astype(float)

        if not features:
            raise ValueError(f"No matched EIS+capacity samples found for cell {cell_id}")

        X_raw = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        blocks = np.asarray(block_ids, dtype=int)

        sort_idx = np.argsort(blocks)
        out[cell_id] = {
            "X_raw": X_raw[sort_idx],
            "y": y[sort_idx],
            "block_ids": blocks[sort_idx],
            "freq": np.asarray(freq_ref, dtype=float),
        }

    return out


if __name__ == "__main__":
    data = load_stanford_data()
    total = sum(v["X_raw"].shape[0] for v in data.values())
    print("Loaded Stanford SECL data")
    for cid in CELL_IDS:
        n = data[cid]["X_raw"].shape[0]
        y = data[cid]["y"]
        print(f"{cid}: n={n}, SOH range={100.0 * y.min():.1f}-{100.0 * y.max():.1f}%")
    print(f"Total matched samples: {total}")
