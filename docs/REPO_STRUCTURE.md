# Repository Structure and Ownership

## Primary Research Code

- `Organized-FDFL/`
  - `src/utils/`: optimization utilities, prediction models, and regret helpers.
  - `src/fairness/`: fairness penalties and gradients.
  - `src/data/`: data resources used by active experiments.
  - `notebooks/`: current experiment runners and analysis notebooks.
  - `tests/`: exploratory validation notebooks/scripts.

Use this folder for all new work and reproducible runs.

## Legacy Research Code

- `FDFL/`
  - Historical FACCT-era implementation.
  - Kept for comparison and migration only.
  - Some files are script-style and execute on import; treat as experimental code.

## Third-Party Snapshots

- `fold-opt-package/`: Fold-Opt reference code (including examples).
- `LANCER-package/`: LANCER reference code.

These are vendored snapshots; keep local project edits minimal unless needed.

## Archived Materials

- `prelim exam materials (wasted)/`: old scratch files.
- `Graphics/`: plot/image outputs across phases.
- `scripts/legacy/`: root-level legacy scripts moved out of repo root.

## Maintenance Utilities

- `scripts/cleanup_workspace.ps1`: removes pycache/build/egg-info/notebook checkpoint/profile artifacts.
- Root `.gitignore`: blocks common generated files from future commits.

