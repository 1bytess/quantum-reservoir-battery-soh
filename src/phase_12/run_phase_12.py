"""Phase 12: ECM baseline development for Warwick EIS SOH estimation.

Creates a reviewer-facing ECM workstream with four stages:
    1. Warwick ECM-readiness audit
    2. Nonlinear circuit fitting and per-cell model selection
    3. Nested LOCO ridge baseline on fitted ECM parameters
    4. Manuscript-facing notes and scope statement

Usage:
    python -m src.phase_12.run_phase_12
    python -m src.phase_12.run_phase_12 --stages 1 3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_12.config import PHASE_12_ROOT


STAGE_DESCRIPTIONS = {
    1: "Warwick ECM-readiness audit",
    2: "Nonlinear circuit fitting and per-cell model selection",
    3: "Nested LOCO ridge baseline on fitted ECM parameters",
    4: "Manuscript-facing ECM notes",
}


def _run_stage(stage_num: int) -> bool:
    stage_map = {
        1: "phase_12.stage_1_warwick_ecm_readiness",
        2: "phase_12.stage_2_ecm_proxy_features",
        3: "phase_12.stage_3_ecm_proxy_loco",
        4: "phase_12.stage_4_ecm_manuscript_notes",
    }

    module_name = stage_map.get(stage_num)
    if module_name is None:
        print(f"Unknown stage: {stage_num}")
        return False

    print(f"\n{'=' * 70}")
    print(f"  Stage {stage_num}: {STAGE_DESCRIPTIONS[stage_num]}")
    print(f"{'=' * 70}")

    import importlib

    try:
        module = importlib.import_module(module_name)
        module.main()
        return True
    except Exception as exc:
        print(f"\n  [FAIL] Stage {stage_num} failed: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main(stages: list[int] | None = None) -> None:
    if stages is None:
        stages = list(STAGE_DESCRIPTIONS.keys())

    print("=" * 70)
    print("Phase 12: ECM baseline development")
    print("=" * 70)
    print(f"Started:  {datetime.now().isoformat()}")
    print(f"Stages:   {stages}")
    print(f"Outputs:  {PHASE_12_ROOT}")
    print()

    results = {}
    for stage_num in stages:
        if stage_num not in STAGE_DESCRIPTIONS:
            print(f"  [WARN] Skipping unknown stage {stage_num}")
            continue
        results[stage_num] = _run_stage(stage_num)

    print(f"\n{'=' * 70}")
    print("Phase 12 Summary")
    print(f"{'=' * 70}")
    for stage_num, success in results.items():
        icon = "[OK]" if success else "[FAIL]"
        print(f"  {icon} Stage {stage_num}: {STAGE_DESCRIPTIONS[stage_num]}")

    n_success = sum(results.values())
    n_total = len(results)
    print(f"\nCompleted: {n_success}/{n_total} stages successful")

    print(f"\n{'=' * 70}")
    print("Key Outputs")
    print(f"{'=' * 70}")
    key_outputs = [
        ("Stage 1", "result/phase_12/stage_1/data/warwick_ecm_readiness.csv"),
        ("Stage 2", "result/phase_12/stage_2/data/warwick_ecm_selected_parameters.csv"),
        ("Stage 3", "result/phase_12/stage_3/data/warwick_ecm_parameter_summary.csv"),
        ("Stage 4", "result/phase_12/stage_4/data/ecm_manuscript_notes.md"),
    ]
    for stage, path in key_outputs:
        full_path = PHASE_12_ROOT.parent / path.replace("result/phase_12/", "phase_12/")
        exists = "[OK]" if full_path.exists() else "[PENDING]"
        print(f"  {exists} [{stage}] {path}")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 12: ECM baseline development")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Stage numbers to run (default: all).",
    )
    args = parser.parse_args()
    main(stages=args.stages)
