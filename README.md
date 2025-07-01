# FDFL - End-to-End Fairness Optimization

A research codebase for studying **end-to-end fairness optimization** via decision-focused learning (FDFL). We compare two-stage vs. end-to-end methods under various fairness regimes, using real-world or synthetic motivating examples.

## Table of Contents

- [Project Overview](#project-overview)  
- [Folders](#Folders)  
  - [Organized-FDFL](#organized-fdfl)  
  - [FDFL](#fdfl-facct-submission)  
  - [Fold-Opt Layer](#fold-opt-layer)  

## Folders

- **Organized-FDFL**: Current Workspace, with both ind+group alpha and more prediction fairness
- **FDFL**: FACCT submission codes, containing individual alpha-fair + acc_parity
- **Fold-Opt Layer**: Original Fold-Opt package
- [Installation](#installation)  
- [Usage](#usage) 

## Project Overview

We address the gap between **prediction fairness** and **decision fairness**, proposing an end-to-end framework that:
1. Predicts multi-treatment benefits $\hat b_{i}$ with prediction fairness.  
2. Allocates resources $d_{i}$ to maximize a individual or group $\alpha$-fair utility  
3. Compares four pipelines:  
   - M1: Two-stage, unconstrained prediction → fair decision  
   - M2: Two-stage, fair prediction → fair decision  
   - M3: End-to-end prediction → fair decision  
   - M4: End-to-end fair prediction → fair decision  
---

## Projects

### Organized-FDFL

The main workspace for ongoing development, supporting both individual and group alpha-fairness, as well as enhanced prediction fairness.

- **`src/utils/`**  
  – Core helpers for optimization, regret calculation, feature processing, and plotting  
- **`src/fairness/`**  
  – Implementations of individual/group fairness penalties  
- **`src/models/`**  
  – JSON-driven model configurations and wrappers  
- **`tests/`**  
  – Jupyter notebooks demonstrating closed-form, finite-difference, and Fold-Opt methods  

<details>
<summary>Key Notebooks</summary>

- `notebooks/FDFL-ind`: Exact copy of the FACCT version for individual alpha-fairness and accuracy parity, with 2-stage, closed-form, and finite-difference gradient methods.
- `notebooks/FDFL-group-fold-opt-working`: Fold-Opt on ind+group-alpha fairness.
- `notebooks/data_processing.ipynb`: Data preparation and processing for the new problem formulation, including updated utility (benefit) and cost.
- `notebooks/runClosedForm.ipynb`: Implements group and individual alpha-fairness using finite-difference and closed-form methods.
- `notebooks/runTwoStage.ipynb`: Runs the 2-stage version for group and individual alpha-fairness.

</details>


### FDFL (FACCT Submission)

End-to-end experimental code used in the FACCT paper:

- **FDFL/Training Notebooks** has the code for training instances.

- **Gradient Methods**: Closed-form, Fold-Opt, LANCER, CVXPYLayer, finite-difference benchmarks  
- **Key Notebooks**:  
  – `FDFL-temp-cvxpylayer-LR.ipynb`  
  – `FDFL-temp-fold-opt.ipynb`  
  – `FDFL.ipynb`



## TODO

- Change min-risk from 0.001 to 1 and verify Individual and Group Regret Performance (Done)

- Write fold-opt subsection.

- For group, report group-wise performance (MSE and Decision Solution&Objective)
  - Closed-Form, Finite-diff, and fold-opt.

- <b>For Fold-OPT Change PGD closed-form to solver</b>


---

## Installation

```bash
# Clone the repo


# Install the core package
pip install -e Organized-FDFL/

# Or try this in the Organized-FDFL dir
pip install -e .

# (import fold-opt directly, or install with downloading it from github)
pip install -e fold-opt-package/fold_opt/
# Similar for PyEPO, LANCER and lcgln package

# Notebook dependencies
pip install -r /Organized-FDFL/requirements.txt


# Package Structure
Organized-FDFL/
├── notebook/            # Jupyter notebooks for experiments and analysis
├── src/
│   ├── data/            # data loading and preprocessing scripts
│   ├── utils/           # core helpers (optimization, regret, features, plots)
│   ├── models/          # JSON configs & model glue code
│   └── fairness/        # fairness-penalty implementations
├── tests/               # Testing new function
└── setup.py             # package install script

