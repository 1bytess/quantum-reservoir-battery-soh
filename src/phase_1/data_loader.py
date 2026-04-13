"""Tabular loaders for phase 1 through phase 4.

The current phase pipeline expects combined CSV tables. After the path
cleanup, the repo stores the Stanford source data under ``data/stanford``.
This module now supports both layouts:

1. Preprocessed tables under ``data/`` or ``data/stanford/``
2. Raw Stanford MAT files under ``data/stanford/diagnostic_test/``
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import NOMINAL_CAPACITY
from ..data_loader import load_stanford_data
from .config import BATTERY_TYPE, CELL_IDS, Phase1LabPaths


PREPROCESSED_FILENAMES = {
    "eis": "all_eis.csv",
    "capacity": "all_capacity.csv",
    "cycling_summary": "all_cycling_summary.csv",
    "metadata": "metadata.json",
}


def _default_temperature_map() -> dict[str, float]:
    return {cell_id: 25.0 for cell_id in CELL_IDS}


def _resolve_dataset_dir(data_dir: Optional[Path] = None) -> Tuple[Path, str]:
    """Resolve the concrete dataset directory and source type."""
    base_dir = Phase1LabPaths().data_dir if data_dir is None else Path(data_dir)
    candidates = [base_dir]
    if base_dir.name != "stanford":
        candidates.append(base_dir / "stanford")

    for candidate in candidates:
        if all((candidate / filename).exists() for filename in PREPROCESSED_FILENAMES.values()):
            return candidate, "preprocessed"

    for candidate in candidates:
        diag_dir = candidate / "diagnostic_test"
        if (diag_dir / "EIS_test.mat").exists() and (diag_dir / "capacity_test.mat").exists():
            return candidate, "raw_stanford"

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate phase input data. "
        f"Checked: {checked}"
    )


@lru_cache(maxsize=4)
def _build_raw_stanford_tables(
    dataset_dir_str: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Convert raw Stanford MAT files into the tabular format used downstream."""
    dataset_dir = Path(dataset_dir_str)
    cell_data = load_stanford_data(dataset_dir)

    nominal_capacity_mAh = NOMINAL_CAPACITY * 1000.0
    temperatures = _default_temperature_map()

    eis_records: list[dict] = []
    capacity_records: list[dict] = []
    cycling_records: list[dict] = []
    metadata_cells: dict[str, dict] = {}

    for cell_id in CELL_IDS:
        record = cell_data[cell_id]
        freq = np.asarray(record["freq"], dtype=float)
        n_freq = int(freq.shape[0])
        X_raw = np.asarray(record["X_raw"], dtype=float)
        y = np.asarray(record["y"], dtype=float)
        block_ids = np.asarray(record["block_ids"], dtype=int)
        temp_c = temperatures[cell_id]

        metadata_cells[cell_id] = {
            "temperature_C": temp_c,
            "n_blocks": int(len(block_ids)),
            "n_spectra": int(len(block_ids)),
            "source": "stanford_diagnostic_test",
        }

        for feature_vec, soh_frac, block_id in zip(X_raw, y, block_ids):
            block_id = int(block_id)
            re_part = feature_vec[:n_freq]
            im_part = feature_vec[n_freq:]
            capacity_mAh = float(soh_frac * nominal_capacity_mAh)

            for freq_hz, re_z, im_z in zip(freq, re_part, im_part):
                eis_records.append(
                    {
                        "cell_id": cell_id,
                        "block_id": block_id,
                        "spectrum_id": block_id,
                        "temperature_C": temp_c,
                        "date_range": "",
                        "frequency_Hz": float(freq_hz),
                        "re_Z_ohm": float(re_z),
                        "im_Z_ohm": float(im_z),
                    }
                )

            capacity_records.append(
                {
                    "cell_id": cell_id,
                    "block_id": block_id,
                    "temperature_C": temp_c,
                    "capacity_mAh": capacity_mAh,
                    "source": "stanford_capacity_test",
                    "soh_pct": float(soh_frac * 100.0),
                }
            )

            cycling_records.append(
                {
                    "cell_id": cell_id,
                    "block_id": block_id,
                    "cycle_id": block_id,
                    "temperature_C": temp_c,
                    "date_range": "",
                    "discharge_capacity_mAh": capacity_mAh,
                    "charge_capacity_mAh": capacity_mAh,
                    "max_voltage_V": np.nan,
                    "min_voltage_V": np.nan,
                    "coulombic_efficiency": np.nan,
                }
            )

    metadata = {
        "battery_type": BATTERY_TYPE,
        "nominal_capacity_mAh": nominal_capacity_mAh,
        "source": "stanford_diagnostic_test",
        "cells": metadata_cells,
    }

    return (
        pd.DataFrame(eis_records),
        pd.DataFrame(capacity_records),
        pd.DataFrame(cycling_records),
        metadata,
    )


def _load_preprocessed_table(dataset_dir: Path, key: str) -> pd.DataFrame | dict:
    path = dataset_dir / PREPROCESSED_FILENAMES[key]
    if key == "metadata":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return pd.read_csv(path)


def _load_table_bundle(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    dataset_dir, source_type = _resolve_dataset_dir(data_dir)
    if source_type == "preprocessed":
        return (
            _load_preprocessed_table(dataset_dir, "eis"),
            _load_preprocessed_table(dataset_dir, "capacity"),
            _load_preprocessed_table(dataset_dir, "cycling_summary"),
            _load_preprocessed_table(dataset_dir, "metadata"),
        )
    return _build_raw_stanford_tables(str(dataset_dir.resolve()))


def load_metadata(data_dir: Optional[Path] = None) -> dict:
    """Load metadata.json or build it from raw Stanford files."""
    _, _, _, metadata = _load_table_bundle(data_dir)
    return metadata


def load_eis(
    cell_id: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load EIS data for one cell or all cells."""
    eis_df, _, _, _ = _load_table_bundle(data_dir)
    if cell_id is None:
        return eis_df.copy()
    return eis_df[eis_df["cell_id"] == cell_id].reset_index(drop=True)


def load_capacity(
    cell_id: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load block-level capacity and SOH labels."""
    _, capacity_df, _, _ = _load_table_bundle(data_dir)
    if cell_id is None:
        return capacity_df.copy()
    return capacity_df[capacity_df["cell_id"] == cell_id].reset_index(drop=True)


def load_cycling_summary(
    cell_id: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load or synthesize a simple per-block cycling summary table."""
    _, _, cycling_df, _ = _load_table_bundle(data_dir)
    if cell_id is None:
        return cycling_df.copy()
    return cycling_df[cycling_df["cell_id"] == cell_id].reset_index(drop=True)


def load_all_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load the tables used by phase 1 and phase 2."""
    eis_df, capacity_df, cycling_df, metadata = _load_table_bundle(data_dir)
    return {
        "eis": eis_df.copy(),
        "capacity": capacity_df.copy(),
        "cycling_summary": cycling_df.copy(),
        "metadata": metadata,
    }


def summarize_data(data: Dict) -> str:
    """Print a human-readable summary of loaded data."""
    eis = data["eis"]
    cap = data["capacity"]
    cyc = data["cycling_summary"]
    meta = data["metadata"]

    lines = [
        f"Battery type: {meta.get('battery_type', 'N/A')}",
        f"Nominal capacity: {meta.get('nominal_capacity_mAh', 'N/A')} mAh",
        "",
        "Per-cell summary:",
    ]

    for cid in CELL_IDS:
        cell_meta = meta["cells"].get(cid, {})
        cell_cap = cap[cap["cell_id"] == cid]
        cell_eis = eis[eis["cell_id"] == cid]
        n_spectra = cell_eis["spectrum_id"].nunique()
        n_blocks = cell_cap["block_id"].nunique()
        temp = cell_meta.get("temperature_C", "?")
        soh_range = (
            f"{cell_cap['soh_pct'].max():.1f}% -> {cell_cap['soh_pct'].min():.1f}%"
            if len(cell_cap) > 0
            else "N/A"
        )
        lines.append(
            f"  {cid} ({temp}C): {n_blocks} blocks, "
            f"{n_spectra} EIS spectra, SOH: {soh_range}"
        )

    lines.extend([
        "",
        f"Total EIS rows: {len(eis):,}",
        f"Total capacity labels: {len(cap)}",
        f"Total cycling summary rows: {len(cyc)}",
    ])

    return "\n".join(lines)
