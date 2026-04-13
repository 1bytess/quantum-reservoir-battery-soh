"""Phase 5 — Unified Ablation Runner.

Consolidates all Phase 5 experiments into one entry point:

  Stage A  – Classical noise ablation           (stage_1_classical_noise_ablation.main)
  Stage B  – Statistical significance tests     (stage_2_statistical_significance.main)
  Stage C  – Extended QRC analyses              (stage_3_extended_qrc_analysis.main)
  Stage D  – Nested LOCO cross-validation       (stage_4_nested_loco_cv.main)
  Stage E  – Stochastic resonance sweep         (NEW)
  Stage F  – Interpolation vs Extrapolation     (NEW)

Usage:
    python run_phase_5_all.py                 # run all stages
    python run_phase_5_all.py --stage E       # run only stochastic resonance
    python run_phase_5_all.py --stage E F     # run stochastic resonance + interp/extrap
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ── project plumbing ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.phase_5.config import get_stage_paths


# =====================================================================
# Stage E  — Stochastic resonance sweep
# =====================================================================

def _run_stage_e() -> None:
    """Stochastic resonance: sweep quantum noise channels on QRC."""
    import numpy as np
    import pandas as pd
    from itertools import combinations
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    from config import CELL_IDS, N_PCA, RANDOM_STATE
    from data_loader import load_stanford_data
    from phase_4.circuit import encode_features, build_qrc_circuit
    from phase_4.config import N_QUBITS

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        amplitude_damping_error,
        phase_damping_error,
    )

    data_dir, _ = get_stage_paths("stage_stochastic_resonance")

    # ── Config ────────────────────────────────────────────────────────
    CHANNELS = ["depolarizing", "amplitude_damping", "phase_damping"]
    RATES = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    SHOTS = 8192
    REPEATS = 3
    DEPTH = 2
    USE_ZZ = True

    # ── Noise model builder ───────────────────────────────────────────
    def build_noise_model(channel: str, rate: float) -> NoiseModel:
        nm = NoiseModel()
        if rate <= 0:
            return nm
        if channel == "depolarizing":
            err_1q = depolarizing_error(rate, 1)
            err_2q = depolarizing_error(rate, 2)
        elif channel == "amplitude_damping":
            err_1q = amplitude_damping_error(rate)
            err_2q = err_1q.tensor(err_1q)
        elif channel == "phase_damping":
            err_1q = phase_damping_error(rate)
            err_2q = err_1q.tensor(err_1q)
        else:
            raise ValueError(f"Unknown channel: {channel}")
        nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry", "rz", "x", "y", "z", "h"])
        nm.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr"])
        return nm

    # ── Expectation value extraction ──────────────────────────────────
    def counts_to_expectations(counts, n_qubits, use_zz):
        total = sum(counts.values())
        exps = []
        padded = {}
        for bs, c in counts.items():
            padded[bs.replace(" ", "").zfill(n_qubits)] = c
        for i in range(n_qubits):
            v = sum((1 - 2 * int(bs[-(i+1)])) * c / total for bs, c in padded.items())
            exps.append(v)
        if use_zz:
            for i, j in combinations(range(n_qubits), 2):
                v = sum(
                    (1 - 2*int(bs[-(i+1)])) * (1 - 2*int(bs[-(j+1)])) * c / total
                    for bs, c in padded.items()
                )
                exps.append(v)
        return np.array(exps)

    # ── Reservoir feature extraction ──────────────────────────────────
    def reservoir_features(X_scaled, depth, noise_model, shots, use_zz, rotations):
        angles = encode_features(X_scaled)
        simulator = AerSimulator(noise_model=noise_model)
        result_list = []
        for i in range(X_scaled.shape[0]):
            qc = build_qrc_circuit(angles[i], depth=depth, random_rotations=rotations)
            qc_m = qc.copy()
            qc_m.measure_all()
            qc_t = transpile(qc_m, simulator, optimization_level=1)
            job = simulator.run(qc_t, shots=shots)
            cts = job.result().get_counts()
            result_list.append(counts_to_expectations(cts, N_QUBITS, use_zz))
        return np.array(result_list)

    # ── PCA helper ────────────────────────────────────────────────────
    def fit_pca_in_fold(X_train, X_test):
        n = min(N_PCA, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n, random_state=RANDOM_STATE)
        return pca.fit_transform(X_train), pca.transform(X_test)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n  Loading Stanford data...")
    cell_data = load_stanford_data()

    rng = np.random.RandomState(RANDOM_STATE)
    rotations = rng.uniform(0, 2 * np.pi, (DEPTH, N_QUBITS, 3))

    all_rows = []
    total_configs = len(CHANNELS) * len(RATES) * REPEATS
    config_idx = 0

    for channel in CHANNELS:
        for rate in RATES:
            for rep in range(REPEATS):
                config_idx += 1
                tag = f"[{config_idx}/{total_configs}] {channel} rate={rate:.4f} rep={rep+1}/{REPEATS}"
                print(f"\n    {tag}")

                nm = build_noise_model(channel, rate)

                # Full LOCO pass
                for test_cell in CELL_IDS:
                    train_cells = [c for c in CELL_IDS if c != test_cell]
                    X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
                    y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
                    X_test_raw = cell_data[test_cell]["X_raw"]
                    y_test = cell_data[test_cell]["y"]

                    X_tr_pca, X_te_pca = fit_pca_in_fold(X_train_raw, X_test_raw)

                    scaler = StandardScaler()
                    X_tr = np.clip(scaler.fit_transform(X_tr_pca), -3.0, 3.0)
                    X_te = np.clip(scaler.transform(X_te_pca), -3.0, 3.0)

                    R_train = reservoir_features(X_tr, DEPTH, nm, SHOTS, USE_ZZ, rotations)
                    R_test = reservoir_features(X_te, DEPTH, nm, SHOTS, USE_ZZ, rotations)

                    grid = GridSearchCV(
                        Ridge(), {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                        cv=min(3, len(y_train)),
                        scoring="neg_mean_absolute_error",
                        n_jobs=-1,
                    )
                    grid.fit(R_train, y_train)
                    y_pred = grid.predict(R_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")

                    all_rows.append({
                        "channel": channel,
                        "rate": rate,
                        "repeat": rep,
                        "depth": DEPTH,
                        "shots": SHOTS,
                        "test_cell": test_cell,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                    })

                avg_mae = np.mean([r["mae"] for r in all_rows if
                                   r["channel"] == channel and
                                   r["rate"] == rate and
                                   r["repeat"] == rep])
                print(f"      avg MAE = {avg_mae:.4f}")

    results_df = pd.DataFrame(all_rows)
    out_path = data_dir / "stochastic_resonance.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(results_df)} rows)")

    # Summary
    summary = results_df.groupby(["channel", "rate"])["mae"].agg(["mean", "std"]).round(4)
    print("\n  Stochastic Resonance Summary:")
    print(summary.to_string())


# =====================================================================
# Stage F  — Interpolation vs Extrapolation
# =====================================================================

def _run_stage_f() -> None:
    """Reframe LOCO = interpolation, Temporal = extrapolation."""
    import pandas as pd
    import numpy as np

    data_dir, plot_dir = get_stage_paths("stage_interp_extrap")
    phase3_dir = PROJECT_ROOT / "result" / "phase_3" / "data"
    phase4_dir = PROJECT_ROOT / "result" / "phase_4" / "data"

    rows = []

    # ── Classical from Phase 3 ────────────────────────────────────────
    loco_path = phase3_dir / "loco_results.csv"
    temp_path = phase3_dir / "temporal_results.csv"

    if loco_path.exists():
        loco = pd.read_csv(loco_path)
        for model in loco["model"].unique():
            m_df = loco[loco["model"] == model]
            rows.append({
                "model": model, "scenario": "interpolation",
                "mae": m_df["mae"].mean(), "rmse": m_df["rmse"].mean(),
                "r2": m_df["r2"].mean(), "std_mae": m_df["mae"].std(),
                "n_folds": len(m_df), "source": "phase3",
            })

    if temp_path.exists():
        temp = pd.read_csv(temp_path)
        # Exclude invalid temporal rows (n_train<3 or numerically blown-up predictions).
        if "valid" in temp.columns:
            temp = temp[temp["valid"]]
        else:
            temp = temp[temp["mae"] <= 1.0]
        for model in temp["model"].unique():
            m_df = temp[temp["model"] == model]
            rows.append({
                "model": model, "scenario": "extrapolation",
                "mae": m_df["mae"].mean(), "rmse": m_df["rmse"].mean(),
                "r2": m_df["r2"].mean(), "std_mae": m_df["mae"].std(),
                "n_folds": len(m_df), "source": "phase3",
            })

    # ── QRC from Phase 4 ─────────────────────────────────────────────
    nl_path = phase4_dir / "qrc_noiseless.csv"
    if nl_path.exists():
        nl = pd.read_csv(nl_path)
        for depth in nl["depth"].unique():
            for regime in ["loco", "temporal"]:
                r_df = nl[(nl["depth"] == depth) & (nl["regime"] == regime)]
                if r_df.empty:
                    continue
                scenario = "interpolation" if regime == "loco" else "extrapolation"
                rows.append({
                    "model": f"qrc_d{depth}", "scenario": scenario,
                    "mae": r_df["mae"].mean(), "rmse": r_df["rmse"].mean(),
                    "r2": r_df["r2"].mean(), "std_mae": r_df["mae"].std(),
                    "n_folds": len(r_df), "source": "phase4_noiseless",
                })

    ny_path = phase4_dir / "qrc_noisy.csv"
    if ny_path.exists():
        ny = pd.read_csv(ny_path)
        max_shots = ny["shots"].max()
        for depth in ny["depth"].unique():
            for regime in ["loco", "temporal"]:
                r_df = ny[(ny["depth"] == depth) & (ny["regime"] == regime) & (ny["shots"] == max_shots)]
                if r_df.empty:
                    continue
                scenario = "interpolation" if regime == "loco" else "extrapolation"
                rows.append({
                    "model": f"qrc_d{depth}_noisy", "scenario": scenario,
                    "mae": r_df["mae"].mean(), "rmse": r_df["rmse"].mean(),
                    "r2": r_df["r2"].mean(), "std_mae": r_df["mae"].std(),
                    "n_folds": len(r_df), "source": f"phase4_noisy_{int(max_shots)}",
                })

    # ── Save ──────────────────────────────────────────────────────────
    all_results = pd.DataFrame(rows)
    all_results["mae_pct"] = all_results["mae"] * 100
    out_path = data_dir / "interp_extrap_results.csv"
    all_results.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(all_results)} rows)")

    # ── Analysis ──────────────────────────────────────────────────────
    display = all_results[all_results["mae"] < 1.0].copy()
    pivot = display.pivot_table(
        values="mae_pct", index="model", columns="scenario", aggfunc="first"
    ).round(2)

    if "interpolation" in pivot.columns and "extrapolation" in pivot.columns:
        pivot["delta"] = (pivot["extrapolation"] - pivot["interpolation"]).round(2)
        pivot["|delta|"] = pivot["delta"].abs()
        pivot = pivot.sort_values("|delta|")
        print("\n  Interpolation vs Extrapolation (MAE %):")
        print(pivot.to_string())

    # Key findings
    qrc_rows = all_results[all_results["model"].str.startswith("qrc")]
    if not qrc_rows.empty:
        qrc_i = qrc_rows[qrc_rows["scenario"] == "interpolation"]
        qrc_e = qrc_rows[qrc_rows["scenario"] == "extrapolation"]
        if len(qrc_i) > 0:
            best = qrc_i.loc[qrc_i["mae"].idxmin()]
            print(f"\n  Best QRC interpolation:  {best['model']} MAE={best['mae_pct']:.2f}%")
        if len(qrc_e) > 0:
            best = qrc_e.loc[qrc_e["mae"].idxmin()]
            print(f"  Best QRC extrapolation:  {best['model']} MAE={best['mae_pct']:.2f}%")


# =====================================================================
# Main dispatch
# =====================================================================

STAGES = {
    "A": ("Classical noise ablation", "stage_1_classical_noise_ablation"),
    "B": ("Statistical significance tests", "stage_2_statistical_significance"),
    "C": ("Extended QRC analyses", "stage_3_extended_qrc_analysis"),
    "D": ("Nested LOCO cross-validation", "stage_4_nested_loco_cv"),
    "E": ("Stochastic resonance sweep", None),
    "F": ("Interpolation vs Extrapolation", None),
}


def main(stages: list[str] | None = None) -> None:
    """Run selected Phase 5 stages."""
    if stages is None:
        stages = list(STAGES.keys())

    print("=" * 70)
    print("Phase 5 — Unified Ablation Suite")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Stages: {stages}")

    for stage in stages:
        if stage not in STAGES:
            print(f"\n⚠ Unknown stage: {stage}, skipping")
            continue

        name, module_name = STAGES[stage]
        print(f"\n{'='*70}")
        print(f"Stage {stage}: {name}")
        print(f"{'='*70}")

        if stage in ("A", "B", "C", "D"):
            from importlib import import_module
            mod = import_module(f"src.phase_5.{module_name}")
            if stage == "B":
                mod.main(n_seeds=20, n_bootstrap=10000)
            elif stage == "C":
                mod.main(run_noisy=True)
            else:
                mod.main()
        elif stage == "E":
            _run_stage_e()
        elif stage == "F":
            _run_stage_f()

    print(f"\nAll stages completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 — Unified Ablation Runner")
    parser.add_argument(
        "--stage", nargs="+", choices=list(STAGES.keys()),
        help="Stage(s) to run. Default: all.",
    )
    args = parser.parse_args()
    main(stages=args.stage)
