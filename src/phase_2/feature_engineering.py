"""EIS feature engineering for lab data.

Pipeline:
1. Per-spectrum: flatten Re(Z) + Im(Z) at each frequency → 38-dim vector
2. Block aggregation: mean of all spectra within each block
3. Align with SOH labels (inner join on cell_id + block_id)
4. Dimensionality reduction: 38 → 6 via PCA (or random projection)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from typing import Tuple, Optional

from .config import (
    N_FREQUENCIES, N_EIS_RAW, N_QRC_INPUT,
    AGGREGATION_METHOD, REDUCTION_METHOD, RANDOM_STATE,
)


def flatten_eis_spectrum(eis_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot EIS data: one row per (cell, block, spectrum) with 38 features.

    Args:
        eis_df: Long-format EIS with columns: cell_id, block_id, spectrum_id,
                frequency_Hz, re_Z_ohm, im_Z_ohm

    Returns:
        DataFrame with columns: cell_id, block_id, spectrum_id,
        re_f0, re_f1, ..., re_f35, im_f0, ..., im_f35
    """
    # Sort by frequency to ensure consistent column ordering
    eis_sorted = eis_df.sort_values(
        ["cell_id", "block_id", "spectrum_id", "frequency_Hz"]
    )

    # Determine the number of frequency points per spectrum
    # (may differ per cell due to different EIS equipment settings)
    sample_group = eis_sorted.groupby(
        ["cell_id", "block_id", "spectrum_id"]
    ).size()
    n_freqs = sample_group.mode().iloc[0]  # most common count

    records = []
    for (cid, blk, sid), group in eis_sorted.groupby(
        ["cell_id", "block_id", "spectrum_id"]
    ):
        group = group.sort_values("frequency_Hz")
        if len(group) != n_freqs:
            # Skip inconsistent spectra
            continue

        row = {"cell_id": cid, "block_id": blk, "spectrum_id": sid}
        for i, (_, r) in enumerate(group.iterrows()):
            row[f"re_f{i}"] = r["re_Z_ohm"]
            row[f"im_f{i}"] = r["im_Z_ohm"]
        records.append(row)

    result = pd.DataFrame(records)
    print(f"  Flattened {len(result)} spectra × {n_freqs * 2} features")
    return result


def aggregate_eis_by_block(
    flat_eis: pd.DataFrame,
    method: str = AGGREGATION_METHOD,
) -> pd.DataFrame:
    """Aggregate spectra within each block to get one feature vector per block.

    Args:
        flat_eis: Flattened EIS data (from flatten_eis_spectrum).
        method: 'mean' or 'median'.

    Returns:
        DataFrame with one row per (cell_id, block_id), 38 feature columns.
    """
    feature_cols = [c for c in flat_eis.columns
                    if c.startswith("re_f") or c.startswith("im_f")]

    if method == "mean":
        agg = flat_eis.groupby(["cell_id", "block_id"])[feature_cols].mean()
    elif method == "median":
        agg = flat_eis.groupby(["cell_id", "block_id"])[feature_cols].median()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    agg = agg.reset_index()
    print(f"  Aggregated to {len(agg)} block-level feature vectors ({method})")
    return agg


def align_features_with_soh(
    eis_features: pd.DataFrame,
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join block-level EIS features with SOH labels.

    Args:
        eis_features: Block-aggregated EIS features (38 cols + cell_id, block_id).
        capacity_df: Capacity ground truth with cell_id, block_id, soh_pct.

    Returns:
        DataFrame with EIS features + soh_pct + temperature_C.
    """
    soh_cols = capacity_df[["cell_id", "block_id", "soh_pct", "temperature_C",
                            "capacity_mAh"]].copy()

    merged = pd.merge(
        eis_features,
        soh_cols,
        on=["cell_id", "block_id"],
        how="inner",
    )

    print(f"  Aligned: {len(merged)} samples with SOH labels "
          f"(from {len(eis_features)} EIS blocks, {len(capacity_df)} capacity rows)")

    if len(merged) == 0:
        raise ValueError("No matching blocks between EIS and capacity data!")

    return merged


def reduce_dimensionality(
    features: np.ndarray,
    n_components: int = N_QRC_INPUT,
    method: str = REDUCTION_METHOD,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, object]:
    """Reduce EIS features from 38-dim to 6-dim.

    Args:
        features: (n_samples, 38) array of EIS features.
        n_components: Target dimensionality (default: 6).
        method: 'pca' or 'random_projection'.
        random_state: Random seed.

    Returns:
        (reduced_features, fitted_reducer): Tuple of array and fitted transformer.
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(features)
        total_var = reducer.explained_variance_ratio_.sum()
        print(f"  PCA: {features.shape[1]}D -> {n_components}D "
              f"(explained variance: {total_var:.1%})")
        for i, v in enumerate(reducer.explained_variance_ratio_):
            print(f"    PC{i+1}: {v:.1%}")

    elif method == "random_projection":
        reducer = GaussianRandomProjection(
            n_components=n_components, random_state=random_state
        )
        reduced = reducer.fit_transform(features)
        print(f"  Random projection: {features.shape[1]}D -> {n_components}D")

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reduced, reducer


def build_feature_table(
    eis_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    method: str = REDUCTION_METHOD,
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Full pipeline: raw EIS → block features → aligned with SOH → reduced.

    Returns:
        (features_72d, features_6d, reducer):
        - features_72d: Full 38-dim features + metadata + soh_pct
        - features_6d: Reduced 6-dim features + metadata + soh_pct
        - reducer: Fitted PCA/projection object
    """
    print("\n  Step 1: Flattening EIS spectra...")
    flat = flatten_eis_spectrum(eis_df)

    print("  Step 2: Block-level aggregation...")
    block_features = aggregate_eis_by_block(flat)

    print("  Step 3: Aligning with SOH labels...")
    features_72d = align_features_with_soh(block_features, capacity_df)

    print("  Step 4: Dimensionality reduction...")
    feature_cols = [c for c in features_72d.columns
                    if c.startswith("re_f") or c.startswith("im_f")]
    X = features_72d[feature_cols].values
    X_reduced, reducer = reduce_dimensionality(X, method=method)

    # Build 6D feature table
    features_6d = features_72d[["cell_id", "block_id", "temperature_C",
                                 "capacity_mAh", "soh_pct"]].copy()
    for i in range(X_reduced.shape[1]):
        features_6d[f"pc{i+1}"] = X_reduced[:, i]

    return features_72d, features_6d, reducer
