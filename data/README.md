# Data Access

This repository does not redistribute the raw datasets used in the study.

## Expected Local Layout

Create the following folders locally before running the code:

```text
data/
  stanford/
  warwick/
  escl/
```

## Dataset Status

- `Stanford`: public dataset used for supporting experiments.
- `Warwick`: public dataset used for the primary cross-cell benchmark.
- `ESCL`: restricted laboratory dataset used for temporal validation and not
  redistributed in this repository.

## Notes For Reproduction

- The loaders resolve dataset roots from `src/config.py` and `src/config_lab.py`.
- Public-dataset users should download the Warwick and Stanford data from their
  original sources and place them in the directory structure shown above.
- The ESCL experiments cannot be rerun from raw data using this public release
  unless separate access is granted.
