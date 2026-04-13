"""Public GitHub-facing phase map for the repository.

This file does not rename the legacy ``src/phase_*`` packages. Instead it
provides a stable public phase order that matches the current paper story:

1-5   Warwick primary workstream
6     Hardware validation (currently Stanford support)
7-10  Stanford supporting workstream
11    ESCL temporal validation
12    Cross-dataset diagnostics and transfer

Use this module for GitHub readers and release documentation, while leaving the
historical internal phase numbering intact for backward compatibility.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PublicPhase:
    phase: int
    title: str
    dataset: str
    focus: str
    default_command: tuple[str, ...]
    legacy_modules: tuple[str, ...]
    result_roots: tuple[str, ...]
    notes: str = ""
    extra_commands: tuple[tuple[str, ...], ...] = ()

    def command_strings(self) -> list[str]:
        commands = [self.default_command, *self.extra_commands]
        return ["python " + " ".join(cmd) for cmd in commands]


PUBLIC_PHASES: tuple[PublicPhase, ...] = (
    PublicPhase(
        phase=1,
        title="Warwick Unified Benchmark",
        dataset="Warwick",
        focus="Standardized train-only preprocessing and shared LOCO benchmark.",
        default_command=(
            "-m",
            "src.manuscript_support.unified_loco_benchmark",
            "--datasets",
            "warwick",
            "--models",
            "qrc",
            "xgboost",
            "ridge",
        ),
        legacy_modules=("src.manuscript_support.unified_loco_benchmark",),
        result_roots=("result/manuscript_support/unified_loco",),
    ),
    PublicPhase(
        phase=2,
        title="Warwick Nested LOCO",
        dataset="Warwick",
        focus="Nested leave-one-cell-out primary benchmark for QRC vs classical baselines.",
        default_command=("-m", "src.phase_8.stage_2_nested_warwick_cv"),
        legacy_modules=("src.phase_8.stage_2_nested_warwick_cv",),
        result_roots=("result/phase_8/stage_2",),
    ),
    PublicPhase(
        phase=3,
        title="Warwick Statistics",
        dataset="Warwick",
        focus="Primary paired stats, bootstrap confidence intervals, and multiplicity checks.",
        default_command=("-m", "src.phase_11.run_phase_11", "--stages", "2", "3", "4"),
        legacy_modules=(
            "src.phase_11.run_phase_11",
            "src.phase_11.stage_2_preregistered_stats",
            "src.phase_11.stage_3_bootstrap_primary",
            "src.phase_11.stage_4_multiple_correction",
        ),
        result_roots=(
            "result/phase_11/stage_2",
            "result/phase_11/stage_3",
            "result/phase_11/stage_4",
        ),
    ),
    PublicPhase(
        phase=4,
        title="Warwick Few-Shot",
        dataset="Warwick",
        focus="Small-data regime analysis and accuracy-efficiency framing.",
        default_command=("-m", "src.phase_11.run_phase_11", "--stages", "5"),
        legacy_modules=("src.phase_11.stage_5_fewshot_reframe",),
        result_roots=("result/phase_11/stage_5", "result/phase_5/stage_5"),
    ),
    PublicPhase(
        phase=5,
        title="Warwick ECM Support",
        dataset="Warwick",
        focus="Equivalent-circuit baselines and manuscript-facing support analysis.",
        default_command=("-m", "src.phase_12.run_phase_12"),
        legacy_modules=("src.phase_12.run_phase_12",),
        result_roots=("result/phase_12",),
    ),
    PublicPhase(
        phase=6,
        title="Hardware Validation",
        dataset="Stanford support + Warwick prep",
        focus="IBM hardware pipeline used as supporting feasibility evidence.",
        default_command=("-m", "src.phase_6.run_phase_6"),
        legacy_modules=("src.phase_6.run_phase_6", "src.phase_6.run_phase_6b", "src.phase_6.run_phase_6c"),
        result_roots=("result/phase_6",),
        notes=(
            "Hardware is still a Stanford support experiment. Do not relabel it as "
            "Warwick unless the circuits and runs are regenerated on Warwick. "
            "Offline Warwick preparation is available via src.phase_6.prepare_warwick_hardware."
        ),
        extra_commands=(
            ("-m", "src.phase_6.run_phase_6", "--prepare"),
            ("-m", "src.phase_6.run_phase_6", "--analyze"),
            ("-m", "src.phase_6.prepare_warwick_hardware"),
            ("-m", "src.phase_6.run_warwick_hardware", "--prepare"),
        ),
    ),
    PublicPhase(
        phase=7,
        title="Stanford Exploration",
        dataset="Stanford",
        focus="Initial dataset loading, Nyquist inspection, and SOH trend plotting.",
        default_command=("-m", "src.phase_1.run_phase_1_stanford"),
        legacy_modules=("src.phase_1.run_phase_1_stanford",),
        result_roots=("result/phase_1",),
    ),
    PublicPhase(
        phase=8,
        title="Stanford Classical Baselines",
        dataset="Stanford",
        focus="LOCO and temporal classical benchmark suite.",
        default_command=("-m", "src.phase_3.run_phase_3"),
        legacy_modules=("src.phase_3.run_phase_3",),
        result_roots=("result/phase_3",),
    ),
    PublicPhase(
        phase=9,
        title="Stanford QRC Simulation",
        dataset="Stanford",
        focus="Noiseless and noisy QRC simulation pipeline.",
        default_command=("-m", "src.phase_4.run_phase_4"),
        legacy_modules=("src.phase_4.run_phase_4",),
        result_roots=("result/phase_4",),
        extra_commands=(
            ("-m", "src.phase_4.run_phase_4", "--noiseless-only"),
            ("-m", "src.phase_4.run_phase_4", "--noisy-only"),
        ),
    ),
    PublicPhase(
        phase=10,
        title="Stanford Supporting Analyses",
        dataset="Stanford",
        focus="Ablations, sensitivity studies, and supporting robustness checks.",
        default_command=("-m", "src.phase_5.run_all_stages"),
        legacy_modules=(
            "src.phase_5.run_all_stages",
            "src.phase_5.run_phase_5",
        ),
        result_roots=("result/phase_5",),
    ),
    PublicPhase(
        phase=11,
        title="ESCL Temporal Validation",
        dataset="ESCL",
        focus="Single-cell temporal validation on lab data.",
        default_command=("-m", "src.phase_7.run_phase_7"),
        legacy_modules=("src.phase_7.run_phase_7",),
        result_roots=("result/phase_7",),
    ),
    PublicPhase(
        phase=12,
        title="Cross-Dataset Diagnostics",
        dataset="Stanford + Warwick + ESCL",
        focus="Kernel diagnostics, feature-space analysis, and transfer experiments.",
        default_command=("-m", "src.phase_9.run_phase_9"),
        legacy_modules=("src.phase_9.run_phase_9", "src.phase_10.transfer_learning"),
        result_roots=("result/phase_9", "result/phase_10"),
        extra_commands=(("-m", "src.phase_10.transfer_learning"),),
    ),
)


PHASE_LOOKUP = {phase.phase: phase for phase in PUBLIC_PHASES}


def _format_phase_summary(phase: PublicPhase) -> str:
    lines = [
        f"Phase {phase.phase}: {phase.title}",
        f"  Dataset: {phase.dataset}",
        f"  Focus:   {phase.focus}",
        "  Commands:",
    ]
    for command in phase.command_strings():
        lines.append(f"    {command}")
    lines.append("  Legacy modules:")
    for module in phase.legacy_modules:
        lines.append(f"    {module}")
    lines.append("  Result roots:")
    for root in phase.result_roots:
        lines.append(f"    {root}")
    if phase.notes:
        lines.append(f"  Notes:   {phase.notes}")
    return "\n".join(lines)


def _print_phase_table() -> None:
    print("Public GitHub phase order")
    print("=" * 72)
    for phase in PUBLIC_PHASES:
        print(
            f"{phase.phase:>2}  {phase.title:<28}  "
            f"{phase.dataset:<22}  {phase.default_command[2] if phase.default_command[:2] == ('-m', 'src') else phase.default_command[1]}"
        )


def _run_command(command: Sequence[str], extra_args: Sequence[str]) -> int:
    full_command = [sys.executable, *command, *extra_args]
    print("Running:", " ".join(full_command))
    completed = subprocess.run(full_command, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GitHub-facing public phase registry for the repository."
    )
    parser.add_argument("--phase", type=int, choices=range(1, 13))
    parser.add_argument("--list", action="store_true", help="List the 12 public phases.")
    parser.add_argument("--run", action="store_true", help="Run the default command for the chosen phase.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through after '--run --'.",
    )
    args = parser.parse_args(argv)

    if args.list or args.phase is None:
        _print_phase_table()
        if args.phase is None:
            return 0

    phase = PHASE_LOOKUP[args.phase]
    print()
    print(_format_phase_summary(phase))

    if args.run:
        extra_args = list(args.extra_args)
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        return _run_command(phase.default_command, extra_args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
