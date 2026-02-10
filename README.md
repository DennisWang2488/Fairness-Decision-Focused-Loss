# Fairness Decision Focused Learning (FDFL)

Research workspace for end-to-end fairness optimization with decision-focused learning.

This repository now separates active code, legacy experiments, and third-party method snapshots.

## Project Scope

- Main topic: prediction-to-decision learning with fairness-aware objectives.
- Current focus: individual/group alpha-fair optimization in healthcare resource allocation.
- Paper artifact (workspace result): `docs/paper/end-to-end-fairness-optimization-with-fair-decision-focused-learning.pdf`

## Repository Map

| Path | Role | Status |
|---|---|---|
| `Organized-FDFL/` | Main maintained workspace (recommended entrypoint) | Active |
| `FDFL/` | Earlier FACCT-era code and notebooks | Legacy |
| `fold-opt-package/` | Fold-Opt source snapshot + examples | Third-party snapshot |
| `LANCER-package/` | LANCER source snapshot | Third-party snapshot |
| `prelim exam materials (wasted)/` | Historical scratch materials | Archive |
| `Graphics/` | Output figures and legacy plot exports | Assets |
| `scripts/` | Maintenance scripts (cleanup + legacy root scripts) | Utility |

## Quick Start (Main Workflow)

```bash
# from repo root
pip install -r Organized-FDFL/requirements.txt
pip install -e Organized-FDFL/
```

Then use notebooks in `Organized-FDFL/notebooks/`:

- `runClosedForm.ipynb`
- `runTwoStage.ipynb`
- `runFoldOPT.ipynb` (if fold-opt dependencies are available)

## External Methods (LANCER, LCGLN, etc.)

- `LANCER-package/` is included as a reference snapshot and can be run independently following its own docs.
- Fold-Opt code is included under `fold-opt-package/`.
- `LCGLN` is referenced in project planning but not implemented as a standalone package in this repository yet.

## Cleanup and Maintenance

Use the cleanup script to remove cache/build artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/cleanup_workspace.ps1
```

Ignored noise patterns are tracked in root `.gitignore` (pycache, build folders, egg-info, notebook checkpoints, profile files).

## Notes

- Legacy folders are preserved for reproducibility and historical comparison.
- For new development, prefer adding code under `Organized-FDFL/` and keep notebooks/results organized inside that folder.
