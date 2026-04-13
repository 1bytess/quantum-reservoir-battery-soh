"""Data loader for Phase 3-Lab: reads Phase 2-Lab feature tables."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path

from .config import Phase3LabPaths, CELL_IDS


def load_features(
    paths: Phase3LabPaths = None,
    use_6d: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load EIS features and SOH labels from Phase 2-Lab outputs.

    Args:
        paths: Phase 3-Lab paths config.
        use_6d: If True, load 6D PCA features. If False, load 72D raw.

    Returns:
        (df, X, y, cell_ids):
        - df: Full DataFrame with metadata
        - X: Feature matrix (n_samples, n_features)
        - y: SOH labels as fraction [0, 1]
        - cell_ids: Array of cell IDs per sample
    """
    if paths is None:
        paths = Phase3LabPaths()

    if use_6d:
        df = pd.read_csv(paths.phase2_lab_dir / "features_eis_6d.csv")
        feature_cols = [c for c in df.columns if c.startswith("pc")]
    else:
        df = pd.read_csv(paths.phase2_lab_dir / "features_eis_72d.csv")
        feature_cols = [c for c in df.columns
                        if c.startswith("re_f") or c.startswith("im_f")]

    X = df[feature_cols].values
    y = df["soh_pct"].values / 100.0  # Convert to fraction
    cell_ids = df["cell_id"].values

    return df, X, y, cell_ids


def get_cell_data(
    paths: Phase3LabPaths = None,
) -> Dict[str, Dict]:
    """Load features organized by cell (for LOCO evaluation).

    Returns:
        {cell_id: {"X_6d": array, "X_72d": array, "y": array, "block_ids": array}}
    """
    if paths is None:
        paths = Phase3LabPaths()

    df_6d = pd.read_csv(paths.phase2_lab_dir / "features_eis_6d.csv")
    df_72d = pd.read_csv(paths.phase2_lab_dir / "features_eis_72d.csv")

    pc_cols = [c for c in df_6d.columns if c.startswith("pc")]
    eis_cols = [c for c in df_72d.columns
                if c.startswith("re_f") or c.startswith("im_f")]

    result = {}
    for cid in CELL_IDS:
        mask_6d = df_6d["cell_id"] == cid
        mask_72d = df_72d["cell_id"] == cid
        result[cid] = {
            "X_6d": df_6d.loc[mask_6d, pc_cols].values,
            "X_72d": df_72d.loc[mask_72d, eis_cols].values,
            "y": df_6d.loc[mask_6d, "soh_pct"].values / 100.0,
            "block_ids": df_6d.loc[mask_6d, "block_id"].values,
            "temperature_C": df_6d.loc[mask_6d, "temperature_C"].iloc[0],
        }

    return result
