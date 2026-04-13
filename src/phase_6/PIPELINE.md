# Phase 6 Pipeline

Phase 6 is intentionally split by evidential role.

## Stanford

- `src.phase_6.run_phase_6`
  - `stage_1`
  - IBM Marrakesh hardware on Stanford
  - global preprocessing because hardware deployment needs fixed angles
- `src.phase_6.run_phase_6b`
  - `stage_2`
  - IBM Fez hardware on Stanford
  - backend comparison against Marrakesh
- `src.phase_6.run_phase_6c`
  - `stage_3`
  - zero-noise extrapolation and backend-noise interpretation

Stanford remains the supporting hardware-calibration path. It is where backend
comparisons, leakage audits, and ZNE live.

## Warwick

- `src.phase_6.prepare_warwick_hardware`
  - `stage_4_warwick_primary`
  - prepares the primary Warwick manifest
  - default target is `ibm_marrakesh`, `foldwise`, `primary`, `3072` shots
  - default foldwise preprocessing matches the Warwick benchmark path:
    `StandardScaler(raw) -> PCA(6) -> StandardScaler(PCA)`
- `src.phase_6.run_warwick_shadow`
  - uses the same prepared manifest
  - builds two offline references before any paid run:
    - `shadow_noiseless`
    - `shadow_digital_twin`
- `src.phase_6.run_warwick_hardware`
  - runs the paid hardware batches for the same run label
  - analyzes hardware results using the same foldwise feature records

Warwick is the primary hardware benchmark path. The intended comparison ladder
is:

1. Warwick software benchmark
2. Warwick hardware-matched noiseless shadow
3. Warwick Marrakesh digital twin
4. Warwick Marrakesh hardware

The current validated pre-hardware reference point is:

- Warwick software benchmark / noiseless shadow: `0.8335%` MAE
- Warwick Marrakesh digital twin: `1.0165%` MAE

## Provenance rule

Warwick Stage 4 writes into a run-labeled directory under
`result/phase_6/stage_4_warwick_primary/` so different backends,
preprocessing modes, and shot budgets cannot overwrite each other.

The stage-level CSV files in that folder are the current canonical outputs for
the latest validated primary Warwick run. Detailed per-run artifacts stay in
their run-label subdirectories.

Legacy `stage_4_warwick_prepare` artifacts are not canonical and should not be
reused for manuscript figures or tables.
