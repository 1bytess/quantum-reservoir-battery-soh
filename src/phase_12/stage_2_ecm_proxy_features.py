"""Phase 12 Stage 2: Nonlinear circuit fitting and per-cell model selection."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_12.config import ECM_CANDIDATE_MODELS, get_stage_paths
from phase_12.ecm_features import load_warwick_impedance_records
from phase_12.ecm_nonlinear import fit_ecm_model, params_to_feature_row


class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def _save_figure(fig: plt.Figure, plot_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / stem}.png")


def main() -> None:
    data_dir, plot_dir = get_stage_paths("stage_2")
    logger = TeeLogger(data_dir / "stage_2_log.txt")
    old_stdout = sys.stdout
    sys.stdout = logger
    try:
        print(f"Started: {datetime.now().isoformat()}")
        records = load_warwick_impedance_records()
        candidate_rows = []
        selected_rows = []
        selected_fit_by_cell = {}

        for rec in records:
            model_results = []
            for model_name in ECM_CANDIDATE_MODELS:
                fit = fit_ecm_model(rec["freq"], rec["re_z"], rec["im_z"], model_name)
                model_results.append(fit)

                row = {
                    "cell_id": rec["cell_id"],
                    "soh_pct": rec["soh_pct"],
                    "model_name": model_name,
                    "success": fit["success"],
                    "nfev": fit["nfev"],
                    "sse": fit["sse"],
                    "rmse_ohm": fit["rmse_ohm"],
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                }
                row.update(params_to_feature_row(model_name, fit["params"]))
                candidate_rows.append(row)

            best = min(model_results, key=lambda item: item["aic"])
            selected_fit_by_cell[rec["cell_id"]] = best["z_fit"]

            row = {
                "cell_id": rec["cell_id"],
                "soh_frac": rec["soh_frac"],
                "soh_pct": rec["soh_pct"],
                "selected_model": best["model_name"],
                "fit_rmse_ohm": best["rmse_ohm"],
                "aic": best["aic"],
                "bic": best["bic"],
            }
            row.update(params_to_feature_row(best["model_name"], best["params"]))
            selected_rows.append(row)

            print(
                f"{rec['cell_id']}: selected {best['model_name']} "
                f"(RMSE={best['rmse_ohm']:.6f} Ohm, AIC={best['aic']:.2f})"
            )

        candidate_df = pd.DataFrame(candidate_rows).sort_values(["cell_id", "model_name"]).reset_index(drop=True)
        selected_df = pd.DataFrame(selected_rows).sort_values("cell_id").reset_index(drop=True)
        candidate_df.to_csv(data_dir / "warwick_ecm_candidate_fits.csv", index=False)
        selected_df.to_csv(data_dir / "warwick_ecm_selected_parameters.csv", index=False)

        selection_summary = (
            selected_df.groupby("selected_model", as_index=False)
            .agg(
                n_cells=("cell_id", "count"),
                median_fit_rmse_ohm=("fit_rmse_ohm", "median"),
                mean_fit_rmse_ohm=("fit_rmse_ohm", "mean"),
            )
        )
        selection_summary.to_csv(data_dir / "warwick_ecm_model_selection_summary.csv", index=False)

        md_lines = [
            "# Warwick nonlinear ECM fitting summary",
            "",
            f"- Cells fitted: {len(selected_df)}",
            f"- Candidate models: {', '.join(ECM_CANDIDATE_MODELS)}",
            f"- Median selected-model fit RMSE (Ohm): {selected_df['fit_rmse_ohm'].median():.6f}",
            "",
            "## Selected model counts",
            "",
        ]
        for _, row in selection_summary.iterrows():
            md_lines.append(
                f"- {row['selected_model']}: {int(row['n_cells'])} cells "
                f"(median RMSE {row['median_fit_rmse_ohm']:.6f} Ohm)"
            )
        (data_dir / "stage_2_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

        chosen_records = sorted(records, key=lambda item: item["soh_pct"])
        if len(chosen_records) > 4:
            idxs = [0, len(chosen_records) // 3, (2 * len(chosen_records)) // 3, len(chosen_records) - 1]
            chosen_records = [chosen_records[i] for i in idxs]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        for ax, rec in zip(axes, chosen_records):
            z_fit = selected_fit_by_cell[rec["cell_id"]]
            ax.plot(rec["re_z"], -rec["im_z"], "o", markersize=3, label="Measured")
            ax.plot(z_fit.real, -z_fit.imag, "-", linewidth=1.4, label="Selected ECM fit")
            ax.set_title(f"{rec['cell_id']} ({rec['soh_pct']:.2f}% SOH)")
            ax.set_xlabel("Re(Z) [Ohm]")
            ax.set_ylabel("-Im(Z) [Ohm]")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle("Representative nonlinear ECM fits on Warwick")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_figure(fig, plot_dir, "warwick_ecm_selected_fits")

        print(f"Completed nonlinear fitting for {len(selected_df)} cells.")
        print("Selected model counts:")
        for _, row in selection_summary.iterrows():
            print(f"  - {row['selected_model']}: {int(row['n_cells'])}")
    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    main()
