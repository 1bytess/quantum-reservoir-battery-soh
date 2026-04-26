# Quantum Reservoir Computing for Battery SOH

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19591235.svg)](https://doi.org/10.5281/zenodo.19591235)

This repository contains the code used for the battery state-of-health (SOH)
study built around quantum reservoir computing (QRC), classical baselines,
Warwick primary results, Stanford support experiments, ESCL temporal
validation, and IBM Quantum hardware runs.

The release is organized as a clean public codebase rather than as a mirror of
the original working directory. It includes runnable source code, the
environment specification, and instructions for obtaining the public datasets
used by the loaders and experiment pipelines.

## What Is Included

- `src/`: analysis pipelines, model code, dataset loaders, and the public
  phase registry (`src/public_release.py`).
- `environment.qiskit.yml`: conda environment specification for the study.
- `data/README.md`: dataset access notes and redistribution status.
- `.env.example`: template for IBM Quantum credentials used by Phase 6.
- `CITATION.cff` and `LICENSE`: citation metadata and Apache-2.0 license.

## What Is Not Included

- Raw datasets.
- Restricted ESCL laboratory data.
- Manuscript drafts, reference PDFs, and authoring assets.
- Internal planning notes, review notes, and local automation metadata.
- Generated `result/` folders from local runs.

## Environment

The working conda environment is `escl-quantum`.

```bash
conda env create -f environment.qiskit.yml
conda activate escl-quantum
```

Use `--no-capture-output` on Windows if your shell has encoding issues:

```bash
conda run -n escl-quantum --no-capture-output python -m src.public_release --list
```

## Public Phase Order

The internal `src/phase_*` folders are historical and preserved for backward
compatibility. For GitHub and paper readers, use the public phase order below.

| Public phase | Focus | Dataset | Default command | Legacy outputs |
| --- | --- | --- | --- | --- |
| 1  | Warwick unified benchmark      | Warwick | `python -m src.manuscript_support.unified_loco_benchmark --datasets warwick --models qrc xgboost ridge` | `result/manuscript_support/unified_loco/` |
| 2  | Warwick nested LOCO            | Warwick | `python -m src.phase_8.stage_2_nested_warwick_cv` | `result/phase_8/stage_2/` |
| 3  | Warwick statistics             | Warwick | `python -m src.phase_11.run_phase_11 --stages 2 3 4` | `result/phase_11/stage_2/`, `stage_3/`, `stage_4/` |
| 4  | Warwick few-shot               | Warwick | `python -m src.phase_11.run_phase_11 --stages 5` | `result/phase_11/stage_5/`, `result/phase_5/stage_5/` |
| 5  | Warwick ECM support            | Warwick | `python -m src.phase_12.run_phase_12` | `result/phase_12/` |
| 6  | Hardware validation            | Stanford support + Warwick prep | `python -m src.phase_6.run_phase_6` | `result/phase_6/` |
| 7  | Stanford exploration           | Stanford | `python -m src.phase_1.run_phase_1_stanford` | `result/phase_1/` |
| 8  | Stanford classical baselines   | Stanford | `python -m src.phase_3.run_phase_3` | `result/phase_3/` |
| 9  | Stanford QRC simulation        | Stanford | `python -m src.phase_4.run_phase_4` | `result/phase_4/` |
| 10 | Stanford supporting analyses   | Stanford | `python -m src.phase_5.run_all_stages` | `result/phase_5/` |
| 11 | ESCL temporal validation       | ESCL | `python -m src.phase_7.run_phase_7` | `result/phase_7/` |
| 12 | Cross-dataset diagnostics      | Stanford + Warwick + ESCL | `python -m src.phase_9.run_phase_9` | `result/phase_9/`, `result/phase_10/` |

### Public Registry CLI

Use the registry to inspect the GitHub-facing phase map:

```bash
python -m src.public_release --list
python -m src.public_release --phase 3
python -m src.public_release --phase 6 --run -- --prepare
python -m src.phase_6.prepare_warwick_hardware
python -m src.phase_6.run_warwick_hardware --prepare
```

### Hardware Notes

- Phase 6 is still a Stanford support hardware pipeline. Do not relabel it
  as Warwick unless the circuits and paid runs are regenerated on Warwick
  data.
- Warwick offline hardware preparation is available via
  `python -m src.phase_6.prepare_warwick_hardware`.
- Warwick hardware execution after preparation is available via
  `python -m src.phase_6.run_warwick_hardware`.
- Legacy module names and result folders are preserved on purpose so old
  scripts, imports, and saved results do not break.

## IBM Quantum Credentials (Phase 6 only)

Phase 6 submits circuits to IBM Cloud Quantum and reads credentials from a
local `.env` file at the repository root. Copy the template and fill in your
own values:

```bash
cp .env.example .env
# then edit .env to insert your IBM_ACC{1,2,3}_API and IBM_ACC{1,2,3}_CRN
```

The `.env` file is gitignored and must not be committed. Only Phase 6
hardware execution needs these variables; all simulation, Warwick, Stanford,
and ESCL phases run without IBM credentials.

## Data Roots

- `data/warwick/` — Warwick DIB dataset (Mendeley Data, public)
- `data/stanford/` — Stanford SECL dataset (OSF, public)
- `data/escl/` — ESCL lab dataset (restricted, not redistributed)

Generated outputs are written locally to `result/`, which is intentionally
not versioned in this public release.

## Data Access

See [data/README.md](data/README.md). In short:

- Warwick and Stanford data are obtained from their original public sources.
- ESCL laboratory data are not redistributed in this repository.
- Readers should recreate `data/warwick/`, `data/stanford/`, and `data/escl/`
  locally before running dataset-dependent pipelines.

## Compatibility

The repo has two views:

- Public release order in this README and `src.public_release`.
- Historical internal order in `src/phase_*` and `result/phase_*`.

That split is intentional. It keeps the repo readable for GitHub without
forcing a risky rename across the whole codebase.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).

## How To Cite

If you use this software, please cite both the software (via the Zenodo DOI)
and the associated manuscript. See [CITATION.cff](CITATION.cff) for machine
readable metadata, or use:

> Hernowo, B. E., Lee, M., Isaiah, E. O., Barancira, T. D., & Kim, J. (2026).
> *Quantum Reservoir Computing for Battery State-of-Health Estimation from
> Electrochemical Impedance Spectroscopy* (v1.1.0) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.19591235
