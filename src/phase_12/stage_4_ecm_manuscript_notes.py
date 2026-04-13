"""Phase 12 Stage 4: Draft manuscript-facing notes for the ECM workstream."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_12.config import get_stage_paths


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


def main() -> None:
    data_dir, _ = get_stage_paths("stage_4")
    logger = TeeLogger(data_dir / "stage_4_log.txt")
    old_stdout = sys.stdout
    sys.stdout = logger
    try:
        print(f"Started: {datetime.now().isoformat()}")

        phase_root = data_dir.parent.parent
        stage1_path = phase_root / "stage_1" / "data" / "warwick_ecm_readiness.csv"
        stage2_path = phase_root / "stage_2" / "data" / "warwick_ecm_model_selection_summary.csv"
        stage3_path = phase_root / "stage_3" / "data" / "warwick_ecm_parameter_summary.csv"

        readiness = pd.read_csv(stage1_path) if stage1_path.exists() else None
        selection = pd.read_csv(stage2_path) if stage2_path.exists() else None
        summary = pd.read_csv(stage3_path) if stage3_path.exists() else None

        lines = [
            "# ECM manuscript notes",
            "",
            "## What this phase currently provides",
            "",
            "- A Warwick-specific ECM-readiness audit on the same 25degC / 50SOC cross-cell condition used in the main paper.",
            "- Nonlinear equivalent-circuit fitting for each cell, with model selection across three candidate topologies.",
            "- A nested leave-one-cell-out ridge baseline on the fitted ECM parameters, which serves as a battery-native comparison path.",
            "",
            "## What this phase does not yet provide",
            "",
            "- It does not yet include a two-time-constant ECM, finite-length diffusion element, or a full identifiability study.",
            "- It still uses a simple regression head on fitted parameters rather than a richer hybrid electrochemical-learning pipeline.",
            "- It should therefore be described as a serious first ECM baseline, not the final physics-informed endpoint.",
            "",
        ]

        if readiness is not None and not readiness.empty:
            lines.extend(
                [
                    "## Warwick readiness snapshot",
                    "",
                    f"- Cells audited: {len(readiness)}",
                    f"- Shared frequency grid across all cells: {'yes' if readiness['freq_matches_reference'].all() else 'no'}",
                    f"- Median apparent resistance span: {readiness['delta_r_ohm'].median():.5f} Ohm",
                    "",
                ]
            )

        if selection is not None and not selection.empty:
            lines.extend(
                [
                    "## Selected model counts",
                    "",
                ]
            )
            for _, row in selection.iterrows():
                lines.append(
                    f"- {row['selected_model']}: {int(row['n_cells'])} cells "
                    f"(median fit RMSE {row['median_fit_rmse_ohm']:.6f} Ohm)"
                )
            lines.append("")

        if summary is not None and not summary.empty:
            row = summary.iloc[0]
            lines.extend(
                [
                    "## Fitted-ECM baseline result",
                    "",
                    f"- Mean LOCO MAE: {row['mean_mae_pct']:.3f}%",
                    f"- RMSE: {row['rmse_pct']:.3f}%",
                    f"- R2: {row['r2']:.3f}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Recommended paper wording",
                "",
                "A battery-native benchmark path has now been strengthened on Warwick by fitting nonlinear equivalent-circuit models to each EIS spectrum, selecting among multiple candidate topologies, and evaluating a nested LOCO ridge baseline on the fitted parameters. This substantially improves reviewer readiness, although broader circuit families and identifiability analysis would still strengthen the final benchmark.",
            ]
        )

        (data_dir / "ecm_manuscript_notes.md").write_text("\n".join(lines), encoding="utf-8")
        print("Wrote manuscript-facing ECM notes.")
    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    main()
