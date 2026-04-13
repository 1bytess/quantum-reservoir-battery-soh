"""Phase 6c: Zero-Noise Extrapolation (ZNE) — configurable noise model.

Root-cause of v1 failure
------------------------
v1 used only readout error as the noise scale λ (constant across depths).
But ibm_fez has BETTER CZ gates (0.2688% vs 0.33%) while marrakesh has better
readout (0.95% vs 1.54%).  At d≥2 the CZ advantage of fez cancels its readout
disadvantage, so:
    d=1 : λ_fez > λ_mar  (readout dominates)     → ZNE direction correct
    d=3 : λ_fez ≈ λ_mar  (cancel)                → Richardson divides by ≈0 → explodes
    d=4 : λ_fez < λ_mar  (CZ dominates)           → ZNE extrapolates BACKWARDS

Fixes implemented
-----------------
1. COMPOSITE λ  — readout + depth × n_cz × ε_cz + t_circuit/T1
2. ORDERING CHECK — skip ZNE when |λ_high − λ_low| < MIN_DELTA (avoids near-singular)
3. PER-OBSERVABLE — separate λ_Z (readout-only) and λ_ZZ (CZ-only) for Z vs ZZ features

LAMBDA_MODE options:
    "readout"       — original (broken at d≥2)
    "composite"     — recommended: all noise channels, depth-dependent
    "per_observable"— Z features use readout λ, ZZ use CZ*depth λ (most physically motivated)

Usage:
    cd src
    python run_phase_6c.py                          # composite mode (recommended)
    python run_phase_6c.py --mode readout           # original (broken, for comparison)
    python run_phase_6c.py --mode per_observable    # per-feature noise scaling
    python run_phase_6c.py --all-modes              # run all three and compare

Outputs:
    result/phase_6/stage_3/data/qrc_zne_{mode}.csv
    result/phase_6/stage_3/data/zne_mode_comparison.csv
    result/phase_6/stage_3/plot/
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (
    CELL_IDS, FEZ_NOISE, MARRAKESH_NOISE,
    N_QUBITS, RIDGE_ALPHAS,
)
from data_loader import load_stanford_data
from phase_6.paths import Phase6StagePaths, get_stage_paths
from phase_4.config import CLAMP_RANGE, USE_ZZ_CORRELATORS

# ── Hardware geometry ────────────────────────────────────────────────────────
N_CZ_PER_LAYER  = 5          # CZ ring on 6 qubits: 5 CZ gates per layer
LAYER_TIME_US   = 0.5        # ~500 ns per layer (Heron R2 gate time estimate)
MIN_DELTA       = 5e-4       # skip ZNE when |λ_high − λ_low| < this (near-singular guard)

# Feature dimensionality
N_Z_FEATURES  = N_QUBITS                                                  # 6
N_ZZ_FEATURES = N_QUBITS * (N_QUBITS - 1) // 2 if USE_ZZ_CORRELATORS else 0  # 15
EXP_DIM       = N_Z_FEATURES + N_ZZ_FEATURES                              # 21

# ── Paths ────────────────────────────────────────────────────────────────────
MARRAKESH_STAGE = Phase6StagePaths("stage_1")
FEZ_STAGE = Phase6StagePaths("stage_2")
ZNE_STAGE = get_stage_paths("stage_3")
CKPT_MAR      = MARRAKESH_STAGE.checkpoint_dir
CKPT_FEZ      = FEZ_STAGE.checkpoint_dir
MANIFEST_MAR  = MARRAKESH_STAGE.manifest_dir / "manifest.json"
MANIFEST_FEZ  = FEZ_STAGE.manifest_dir / "manifest.json"
ZNE_DATA_DIR  = ZNE_STAGE.data_dir
ZNE_PLOT_DIR  = ZNE_STAGE.plot_dir


# ============================================================================
# Noise scale computations
# ============================================================================

def lambda_readout(noise: dict, depth: int) -> float:
    """v1 mode: only readout error (depth-independent)."""
    return noise["measurement_error"]


def lambda_composite(noise: dict, depth: int) -> float:
    """
    Depth-dependent composite noise scale:
        λ = ε_readout + depth × N_cz × ε_cz + t_circuit / T1

    Physically: readout error is measurement-layer noise, CZ errors accumulate
    per gate, T1 decay is proportional to total circuit time.
    """
    t_circuit = depth * LAYER_TIME_US
    return (
        noise["measurement_error"]
        + depth * N_CZ_PER_LAYER * noise["two_qubit_error"]
        + t_circuit / noise["t1_us"]
    )


def lambda_z_only(noise: dict, depth: int) -> float:
    """For Z-observable features: readout + T1 decay (no CZ contribution)."""
    t_circuit = depth * LAYER_TIME_US
    return noise["measurement_error"] + t_circuit / noise["t1_us"]


def lambda_zz_only(noise: dict, depth: int) -> float:
    """For ZZ-correlator features: CZ gates dominate (+ T2 dephasing ≈ T1 proxy)."""
    t_circuit = depth * LAYER_TIME_US
    return depth * N_CZ_PER_LAYER * noise["two_qubit_error"] + t_circuit / noise["t2_us"]


def get_lambdas(noise_a: dict, noise_b: dict, depth: int,
                mode: str) -> Tuple[float, float]:
    """Return (λ_a, λ_b) for the given mode. a=marrakesh, b=fez."""
    if mode == "readout":
        return lambda_readout(noise_a, depth), lambda_readout(noise_b, depth)
    elif mode == "composite":
        return lambda_composite(noise_a, depth), lambda_composite(noise_b, depth)
    else:
        raise ValueError(f"Unknown mode for get_lambdas: {mode}")


def print_lambda_table(mode: str):
    print(f"\n  λ table (mode='{mode}'):")
    print(f"  {'depth':>6} | {'λ_mar':>10} | {'λ_fez':>10} | {'Δλ':>10} | ordering")
    print(f"  {'-'*60}")
    for d in [1, 2, 3, 4]:
        if mode == "per_observable":
            lm_z  = lambda_z_only(MARRAKESH_NOISE, d)
            lf_z  = lambda_z_only(FEZ_NOISE, d)
            lm_zz = lambda_zz_only(MARRAKESH_NOISE, d)
            lf_zz = lambda_zz_only(FEZ_NOISE, d)
            print(f"  d={d}: Z  λ_mar={lm_z:.5f} λ_fez={lf_z:.5f} Δ={lf_z-lm_z:+.5f}"
                  f"  |  ZZ λ_mar={lm_zz:.5f} λ_fez={lf_zz:.5f} Δ={lf_zz-lm_zz:+.5f}")
        else:
            lm, lf = get_lambdas(MARRAKESH_NOISE, FEZ_NOISE, d, mode)
            delta = lf - lm
            status = "fez>mar ✓" if delta > MIN_DELTA else (
                     "SKIP ⚠️" if abs(delta) < MIN_DELTA else "FLIPPED ✗")
            print(f"  d={d}: λ_mar={lm:.5f} λ_fez={lf:.5f} Δ={delta:+.5f}  {status}")


# ============================================================================
# Counts → expectation values
# ============================================================================

def counts_to_expectations(counts: Dict[str, int]) -> np.ndarray:
    total = sum(counts.values())
    padded: Dict[str, int] = {}
    for bs, cnt in counts.items():
        key = bs.replace(" ", "").zfill(N_QUBITS)
        padded[key] = padded.get(key, 0) + cnt

    exps: List[float] = []
    # Z observables
    for i in range(N_QUBITS):
        ev = sum((1 - 2 * int(bs[-(i+1)])) * c / total for bs, c in padded.items())
        exps.append(ev)
    # ZZ correlators
    if USE_ZZ_CORRELATORS:
        for i, j in combinations(range(N_QUBITS), 2):
            ev = sum(
                (1 - 2*int(bs[-(i+1)])) * (1 - 2*int(bs[-(j+1)])) * c / total
                for bs, c in padded.items()
            )
            exps.append(ev)
    return np.array(exps)


# ============================================================================
# Load checkpoints
# ============================================================================

def load_checkpoint_expectations(ckpt_dir: Path, manifest_path: Path
                                  ) -> Dict[str, np.ndarray]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    n_samples = manifest["n_samples"]

    run_exps: Dict[str, np.ndarray] = {}
    for ckpt_path in sorted(ckpt_dir.glob("batch*.json")):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        if ckpt.get("status") != "completed" or "counts" not in ckpt:
            continue
        cfg     = ckpt["config"]
        run_key = f"d{cfg['depth']}_s{cfg['seed']}"
        if run_key not in run_exps:
            run_exps[run_key] = np.full((n_samples, EXP_DIM), np.nan)
        for count_key, counts in ckpt["counts"].items():
            idx = int(count_key.split("_sample")[1])
            run_exps[run_key][idx] = counts_to_expectations(counts)
    return run_exps


# ============================================================================
# Richardson extrapolation (all modes)
# ============================================================================

def zne_composite(E_low: np.ndarray, E_high: np.ndarray,
                  lam_low: float, lam_high: float) -> Optional[np.ndarray]:
    """
    Linear 2-point Richardson extrapolation.
    E(λ) = E(0) + c*λ  ⟹  E(0) = (λ_high*E_low - λ_low*E_high) / (λ_high - λ_low)

    Returns None if |Δλ| < MIN_DELTA (near-singular guard).
    The assignment high/low is automatic: function accepts either ordering and
    uses the formula correctly regardless.
    """
    delta = lam_high - lam_low
    if abs(delta) < MIN_DELTA:
        return None
    return (lam_high * E_low - lam_low * E_high) / delta


def zne_per_observable(E_low: np.ndarray, E_high: np.ndarray,
                        depth: int) -> Optional[np.ndarray]:
    """
    Per-observable ZNE: Z features use readout+T1 λ, ZZ features use CZ+T2 λ.
    Applies extrapolation independently to each feature group.
    """
    result = np.full_like(E_low, np.nan)

    # Z features (indices 0..N_Z_FEATURES-1)
    lm_z = lambda_z_only(MARRAKESH_NOISE, depth)
    lf_z = lambda_z_only(FEZ_NOISE, depth)
    dz   = lf_z - lm_z
    if abs(dz) >= MIN_DELTA:
        # Assign low/high correctly
        if dz > 0:
            zne_z = zne_composite(E_low[:, :N_Z_FEATURES], E_high[:, :N_Z_FEATURES], lm_z, lf_z)
        else:
            zne_z = zne_composite(E_high[:, :N_Z_FEATURES], E_low[:, :N_Z_FEATURES], lf_z, lm_z)
        if zne_z is not None:
            result[:, :N_Z_FEATURES] = zne_z

    # ZZ features (indices N_Z_FEATURES..EXP_DIM-1)
    if N_ZZ_FEATURES > 0:
        lm_zz = lambda_zz_only(MARRAKESH_NOISE, depth)
        lf_zz = lambda_zz_only(FEZ_NOISE, depth)
        dzz   = lf_zz - lm_zz
        if abs(dzz) >= MIN_DELTA:
            if dzz > 0:
                zne_zz = zne_composite(E_low[:, N_Z_FEATURES:], E_high[:, N_Z_FEATURES:],
                                        lm_zz, lf_zz)
            else:
                zne_zz = zne_composite(E_high[:, N_Z_FEATURES:], E_low[:, N_Z_FEATURES:],
                                        lf_zz, lm_zz)
            if zne_zz is not None:
                result[:, N_Z_FEATURES:] = zne_zz

    # Return None if both groups failed
    if np.isnan(result).all():
        return None
    return result


# ============================================================================
# LOCO evaluation
# ============================================================================

def loco_eval(X_res: np.ndarray, cell_ids: np.ndarray, soh: np.ndarray,
              depth: int, seed: int, backend: str) -> List[dict]:
    valid = ~np.isnan(X_res[:, 0])
    rows  = []
    for test_cell in CELL_IDS:
        tr_mask = np.isin(cell_ids, [c for c in CELL_IDS if c != test_cell]) & valid
        te_mask = (cell_ids == test_cell) & valid
        Xtr, ytr = X_res[tr_mask], soh[tr_mask]
        Xte, yte = X_res[te_mask], soh[te_mask]
        if len(ytr) == 0 or len(yte) == 0:
            continue
        grid = GridSearchCV(Ridge(), {"alpha": RIDGE_ALPHAS},
                            cv=min(3, len(ytr)),
                            scoring="neg_mean_absolute_error", n_jobs=-1)
        grid.fit(Xtr, ytr)
        yp    = grid.predict(Xte)
        mae   = mean_absolute_error(yte, yp)
        naive = mean_absolute_error(yte, np.full_like(yte, ytr.mean()))
        rows.append({
            "backend": backend, "depth": depth, "seed": seed,
            "test_cell": test_cell, "mae": mae, "naive_mae": naive,
            "beats_naive": bool(mae < naive),
            "best_alpha": grid.best_params_["alpha"],
        })
    return rows


# ============================================================================
# Main ZNE pipeline for a single mode
# ============================================================================

def run_mode(mode: str, exps_mar: Dict, exps_fez: Dict,
             cell_ids: np.ndarray, soh: np.ndarray,
             manifest_fez: dict) -> pd.DataFrame:

    batch_experiment = {
        f"d{b['depth']}_s{b['seed']}": b["experiment"]
        for b in manifest_fez["batches"].values()
    }

    common_keys = sorted(set(exps_mar) & set(exps_fez))
    all_rows: List[dict] = []

    print(f"\n  --- mode = {mode} ---")
    print_lambda_table(mode)

    for run_key in common_keys:
        E_mar = exps_mar[run_key]
        E_fez = exps_fez[run_key]
        depth = int(run_key.split("_")[0][1:])
        seed  = int(run_key.split("_")[1][1:])

        valid = ~np.isnan(E_mar[:, 0]) & ~np.isnan(E_fez[:, 0])
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        # ── Compute ZNE features ────────────────────────────────────────────
        E_zne = None
        skip_reason = ""

        if mode == "readout":
            lm, lf = lambda_readout(MARRAKESH_NOISE, depth), lambda_readout(FEZ_NOISE, depth)
            if abs(lf - lm) < MIN_DELTA:
                skip_reason = f"|Δλ|={abs(lf-lm):.5f} < {MIN_DELTA}"
            else:
                E_zne = np.full_like(E_mar, np.nan)
                if lf > lm:
                    E_zne[valid] = zne_composite(E_mar[valid], E_fez[valid], lm, lf)
                else:
                    E_zne[valid] = zne_composite(E_fez[valid], E_mar[valid], lf, lm)

        elif mode == "composite":
            lm = lambda_composite(MARRAKESH_NOISE, depth)
            lf = lambda_composite(FEZ_NOISE, depth)
            delta = lf - lm
            if abs(delta) < MIN_DELTA:
                skip_reason = f"|Δλ|={abs(delta):.5f} < {MIN_DELTA} (backends at equal noise)"
            else:
                # Auto-assign low/high by actual ordering
                E_low, E_high = (E_mar, E_fez) if lf > lm else (E_fez, E_mar)
                lam_low, lam_high = (lm, lf) if lf > lm else (lf, lm)
                E_zne = np.full_like(E_mar, np.nan)
                r = zne_composite(E_low[valid], E_high[valid], lam_low, lam_high)
                if r is not None:
                    E_zne[valid] = r
                else:
                    skip_reason = "singular Δλ"

        elif mode == "per_observable":
            E_zne = np.full_like(E_mar, np.nan)
            r = zne_per_observable(E_mar[valid], E_fez[valid], depth)
            if r is not None:
                E_zne[valid] = r
            else:
                skip_reason = "both Z and ZZ groups singular"

        # ── Raw hardware baselines ──────────────────────────────────────────
        for label, E in [("ibm_marrakesh", E_mar), ("ibm_fez", E_fez)]:
            rows = loco_eval(E, cell_ids, soh, depth, seed, label)
            all_rows.extend(rows)

        # ── ZNE results (if computed) ───────────────────────────────────────
        if E_zne is not None:
            zne_label = f"ZNE_{mode}"
            rows = loco_eval(E_zne, cell_ids, soh, depth, seed, zne_label)
            all_rows.extend(rows)
            if rows:
                mean_mae = np.mean([r["mae"] for r in rows])
                mar_mae  = np.mean([r["mae"] for r in all_rows
                                    if r["backend"] == "ibm_marrakesh"
                                    and r["depth"] == depth and r["seed"] == seed])
                fez_mae  = np.mean([r["mae"] for r in all_rows
                                    if r["backend"] == "ibm_fez"
                                    and r["depth"] == depth and r["seed"] == seed])
                print(f"    {run_key}: mar={mar_mae:.5f}  fez={fez_mae:.5f}"
                      f"  ZNE={mean_mae:.5f}"
                      f"  (vs fez {(fez_mae-mean_mae)/fez_mae*100:+.1f}%"
                      f"  vs mar {(mar_mae-mean_mae)/mar_mae*100:+.1f}%)")
        else:
            print(f"    {run_key}: ZNE SKIPPED — {skip_reason}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        csv_path = ZNE_DATA_DIR / f"qrc_zne_{mode}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    return df


# ============================================================================
# Comparison summary across modes
# ============================================================================

def build_comparison(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    # Add noiseless sim reference
    sim_path = PROJECT_ROOT / "result" / "phase_4" / "data" / "qrc_vs_classical.csv"
    if sim_path.exists():
        sim_df = pd.read_csv(sim_path)
        for _, row in sim_df[sim_df["method"].isin(
                ["qrc_noiseless_d1", "xgboost_38d", "qrc_noisy_d1_s4096"])].iterrows():
            rows.append({"mode": "reference", "backend": row["method"],
                         "depth": 1, "seed": 42, "mean_mae": row["mae_mean"]})

    for mode, df in dfs.items():
        if df.empty:
            continue
        for (backend, depth, seed), grp in df.groupby(["backend", "depth", "seed"]):
            rows.append({
                "mode": mode, "backend": backend,
                "depth": depth, "seed": seed,
                "mean_mae": grp["mae"].mean(),
                "std_mae":  grp["mae"].std(),
            })

    comp = pd.DataFrame(rows).sort_values(["depth", "seed", "mean_mae"])
    comp_path = ZNE_DATA_DIR / "zne_mode_comparison.csv"
    comp.to_csv(comp_path, index=False)
    return comp


# ============================================================================
# Plots
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mode_comparison(dfs: Dict[str, pd.DataFrame]):
    """Side-by-side depth sweep for all ZNE modes vs raw hardware (seed=42 only)."""
    depths = [1, 2, 3, 4]

    styles = {
        "ibm_marrakesh": dict(color="#e67e22", ls="-",  marker="o", lw=2),
        "ibm_fez":       dict(color="#e74c3c", ls="--", marker="s", lw=2),
        "ZNE_readout":   dict(color="#c0392b", ls=":",  marker="^", lw=1.5),
        "ZNE_composite": dict(color="#2ecc71", ls="-.", marker="D", lw=2),
        "ZNE_per_observable": dict(color="#9b59b6", ls=(0,(3,1,1,1)), marker="P", lw=2),
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    # Noiseless reference
    sim_path = PROJECT_ROOT / "result" / "phase_4" / "data" / "qrc_noiseless.csv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path)
        sl  = sim[sim["regime"] == "loco"].groupby("depth")["mae"].mean().reindex(depths)
        ax.plot(depths, sl.values, color="#3498db", ls=":", lw=1.5,
                marker="*", ms=8, label="Sim noiseless")

    # Plot each backend/ZNE series
    plotted = set()
    for mode, df in dfs.items():
        if df.empty:
            continue
        s42 = df[df["seed"] == 42]
        for backend, grp in s42.groupby("backend"):
            by_d = grp.groupby("depth")["mae"].mean().reindex(depths)
            if backend in plotted:
                continue
            if backend in styles:
                ax.plot(depths, by_d.values, label=backend,
                        **{k: v for k, v in styles[backend].items()})
                plotted.add(backend)

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Mean LOCO MAE")
    ax.set_title("ZNE Mode Comparison: Composite vs Per-Observable vs Readout-Only\n(seed=42, Stanford SECL)")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = ZNE_PLOT_DIR / f"zne_mode_comparison.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_zne_per_cell(dfs: Dict[str, pd.DataFrame]):
    """Per-cell MAE at d=1 seed=42, comparing all three ZNE modes."""
    cells = sorted(CELL_IDS)
    x     = np.arange(len(cells))
    w     = 0.15

    backends_to_plot = [
        ("ibm_marrakesh", "#e67e22"),
        ("ibm_fez",       "#e74c3c"),
        ("ZNE_readout",   "#c0392b"),
        ("ZNE_composite", "#2ecc71"),
        ("ZNE_per_observable", "#9b59b6"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = {}
    offset  = -(len(backends_to_plot) - 1) / 2

    for i, (backend, color) in enumerate(backends_to_plot):
        for mode, df in dfs.items():
            if df.empty:
                continue
            sub = df[(df["backend"] == backend) & (df["depth"] == 1) & (df["seed"] == 42)]
            if sub.empty or backend in plotted:
                continue
            by_cell = sub.set_index("test_cell")["mae"].reindex(cells)
            ax.bar(x + (offset + i) * w, by_cell.values, w,
                   color=color, alpha=0.85, label=backend, edgecolor="k", lw=0.4)
            plotted[backend] = True
            break

    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_xlabel("Test Cell (LOCO fold)")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Cell MAE at d=1 seed=42: All ZNE Modes")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = ZNE_PLOT_DIR / f"zne_per_cell_modes.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 6c ZNE (configurable noise model)")
    parser.add_argument("--mode", choices=["readout", "composite", "per_observable"],
                        default="composite")
    parser.add_argument("--all-modes", action="store_true",
                        help="Run all three modes and compare")
    args = parser.parse_args()

    # Load data once
    cell_data = load_stanford_data()
    X_raw_all, cell_id_all, soh_all = [], [], []
    for cid in CELL_IDS:
        n = cell_data[cid]["X_raw"].shape[0]
        X_raw_all.append(cell_data[cid]["X_raw"])
        cell_id_all.extend([cid] * n)
        soh_all.extend(cell_data[cid]["y"].tolist())
    cell_ids = np.array(cell_id_all)
    soh      = np.array(soh_all)

    print(f"Loaded {len(soh)} Stanford samples")

    exps_mar = load_checkpoint_expectations(CKPT_MAR, MANIFEST_MAR)
    exps_fez = load_checkpoint_expectations(CKPT_FEZ, MANIFEST_FEZ)
    print(f"Checkpoints: marrakesh {sorted(exps_mar)}, fez {sorted(exps_fez)}")

    with open(MANIFEST_FEZ) as f:
        manifest_fez = json.load(f)

    print(f"\nNoise parameters:")
    print(f"  ibm_marrakesh: readout={MARRAKESH_NOISE['measurement_error']:.4f}"
          f"  CZ={MARRAKESH_NOISE['two_qubit_error']:.4e}"
          f"  T1={MARRAKESH_NOISE['t1_us']} µs")
    print(f"  ibm_fez:       readout={FEZ_NOISE['measurement_error']:.4f}"
          f"  CZ={FEZ_NOISE['two_qubit_error']:.4e}"
          f"  T1={FEZ_NOISE['t1_us']} µs")
    print(f"  MIN_DELTA = {MIN_DELTA} (skip ZNE if |Δλ| < this)")
    print(f"  N_CZ_PER_LAYER = {N_CZ_PER_LAYER}, LAYER_TIME = {LAYER_TIME_US} µs")

    modes = ["readout", "composite", "per_observable"] if args.all_modes else [args.mode]
    dfs   = {}

    print("\n" + "=" * 65)
    print("  ZNE Results")
    print("=" * 65)

    for mode in modes:
        df = run_mode(mode, exps_mar, exps_fez, cell_ids, soh, manifest_fez)
        dfs[mode] = df

    # Summary across modes
    print("\n" + "=" * 65)
    print("  Summary: best LOCO MAE per backend (seed=42)")
    print("=" * 65)
    comp = build_comparison(dfs)

    for depth in [1, 2, 3, 4]:
        sub = comp[comp["depth"] == depth].sort_values("mean_mae")
        if sub.empty:
            continue
        print(f"\n  d={depth}:")
        for _, row in sub.iterrows():
            marker = " ← best" if row.name == sub.index[0] else ""
            print(f"    {row['backend']:25s}: MAE={row['mean_mae']:.5f}{marker}")

    print(f"\n  Full comparison saved: {ZNE_DATA_DIR / 'zne_mode_comparison.csv'}")

    # Plots
    if len(dfs) > 0:
        print("\nGenerating plots ...")
        plot_mode_comparison(dfs)
        plot_zne_per_cell(dfs)

    print("\nDone.")


if __name__ == "__main__":
    main()
