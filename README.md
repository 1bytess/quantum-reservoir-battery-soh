# Quantum Reservoir Computing for Battery SOH

This repository contains the code used for battery state-of-health estimation
experiments based on electrochemical impedance spectroscopy (EIS), quantum
reservoir computing, and matched classical baselines.

The release is organized as a clean public codebase rather than as a mirror of
the original working directory. It includes runnable source code, the
environment specification, and instructions for obtaining the public datasets
used by the loaders and experiment pipelines.

## What Is Included

- `src/`: analysis pipelines, model code, dataset loaders, and the public phase
  registry.
- `environment.qiskit.yml`: conda environment specification for the study.
- `data/README.md`: dataset access notes and redistribution status.

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

## Public Phase Map

The public phase map is exposed through:

```bash
python -m src.public_release --list
python -m src.public_release --phase 2
```

Representative commands for the main paper story:

```bash
python -m src.manuscript_support.unified_loco_benchmark --datasets warwick --models qrc xgboost ridge
python -m src.phase_8.stage_2_nested_warwick_cv
python -m src.phase_11.run_phase_11 --stages 2 3 4
python -m src.phase_6.run_phase_6 --prepare
python -m src.phase_6.run_phase_6 --analyze
python -m src.phase_1.run_phase_1_stanford
python -m src.phase_4.run_phase_4 --noiseless-only
python -m src.phase_7.run_phase_7
```

Generated outputs are written locally to `result/`, which is intentionally not
versioned in this public release.

## Data Access

See [data/README.md](data/README.md).

In short:

- Warwick and Stanford data are obtained from their original public sources.
- ESCL laboratory data are not redistributed in this repository.
- Readers should recreate `data/warwick/`, `data/stanford/`, and `data/escl/`
  locally before running dataset-dependent pipelines.
