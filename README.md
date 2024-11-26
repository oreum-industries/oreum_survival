# README.md

## Oreum Survival `oreum_survival`

Demonstrate Survival Regression Modeling using Bayesian inference and a
Bayesian workflow, specifically using the `pymc` & `arviz` ecosystem.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![CI](https://github.com/oreum-industries/oreum_survival/workflows/ci/badge.svg)](https://github.com/oreum-industries/oreum_survival/actions/workflows/ci.yml)
[![code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![code style: interrogate](https://raw.githubusercontent.com/oreum-industries/oreum_core/master/assets/img/interrogate_badge.svg)](https://pypi.org/project/interrogate/)
[![code security: bandit](https://img.shields.io/badge/code%20security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)


### Contents

1. Project Description, Scope, Directory Structure
2. How to Install and Run on a Local Developer Machine
3. Code Standards
4. Notebook Standards
5. Data Standards
6. General Notes

---

## 1. Project Description, Technical Overview, Directory Structure

### 1.1 Project Description & Scope

This is an initial implementation of an internal project by Oreum Industries -
an implementation of copula-based Expected Loss Cost forecasting.

This project **is**:

+ A work in progress (v0.y.z) and liable to breaking changes and inconveniences
  to the user
+ Solely designed for ease of use and rapid development by employees of Oreum
  Industries, and selected clients with guidance

This project **is not**:

+ Intended for public usage and will not be supported for public usage
+ Intended for contributions by anyone not an employee of Oreum Industries,
  and unsolicited contributions will not be accepted


### 1.2 Technical Overview

+ Project began on 2024-01-25
+ The `README.md` is MacOS and POSIX oriented
+ See `LICENCE.md` for licensing and copyright details
+ See `pyproject.toml` for authors, package dependencies etc
+ For code repository access see
  [GitHub](https://github.com/oreum-industries/oreum_survival)
+ Implementation:
  + This project is enabled by a modern, open-source, advanced software stack
    for data curation, statistical analysis and predictive modelling
  + This project is end-to-end, fully reproducible data science solutions: via
    notebooks, scripts, CLI & API, automated environment & package management,
    continuous integration, version control and rich documentation
  + Specifically we use an open-source Python-based suite of software packages,
    the core of which is often known as the Scientific Python stack, supported
    by [NumFOCUS](https://numfocus.org)
  + Once installed (see section 2), see `LICENSES_THIRD_PARTY.md` for full
    details of all package licences
+ Environments: this project was originally developed on a Macbook Air M2
  (Apple Silicon ARM64) running MacOS 14.7 (Sonoma) using `osx-arm64` Accelerate



### 1.3 Project File Structure

The repo is structured for R&D usage. The major items to be
aware of are:

```txt
/
↳ dotfiles         - various dotfiles to configure the repo
↳ Makefile         - recipes to build the dev env
↳ README.md        - this readme file
↳ LICENSE.md       - licensing and copyright details
↳ assets/          - non-code based images and external pdfs
↳ data/            - placeholder for data files (to be managed via Git LFS)
↳ notebooks/       - Jupyter Notebooks
↳ plots/           - output plots saved as images
↳ sql/             - SQL files
↳ src/             - Python modules
  ↳ config/          - configs if used
  ↳ dataprep/        - data transforms / feature engineering pre-model
  ↳ engine/          - classes to operate models
  ↳ model/           - classes define statistical models
```

---


## 2. How to Install and Run on a Local Developer Machine

For local development on MacOS.


### 2.0 Pre-requisite installs via `homebrew`

1. Install Homebrew, see instructions at [https://brew.sh](https://brew.sh)
2. Install `direnv`, `git`, `git-lfs`, `graphviz`, `zsh`

```zsh
$> brew update && brew upgrade
$> brew install direnv git git-lfs graphviz zsh
```


### 2.1 Git clone the repo

Assumes `direnv`, `git`, `git-lfs`, `graphviz` and `zsh` installed as above

```zsh
$> git clone https://github.com/oreum-industries/oreum_survival
$> cd oreum_survival
```
Then allow `direnv` on MacOS to automatically run file `.envrc` upon directory open


### 2.2 Create virtual environment and install dev packages

Notes:

+ We use `conda` virtual envs controlled by `mamba` (quicker than `conda`)
+ We install packages using `miniforge` (sourced from the `conda-forge` repo)
  wherever possible and only use `pip` for packages that are handled better by
  `pip` and/or more up-to-date on [pypi](https://pypi.org)
+ Packages might not be the very latest because we want stability for `pymc`
  which is usually in a state of development flux
+ See [cheat sheet of conda commands](https://conda.io/docs/_downloads/conda-cheatsheet.pdf)
+ The `Makefile` creates a dev env and will also download and preinstall
  `miniforge` if not yet installed on your system


#### 2.2.1 Create the dev environment

From the dir above `oreum_survival/` project dir:

```zsh
$> make -C oreum_survival dev
```

This will also create some files to help confirm / diagnose successful installation:

+ `dev/install_log/blas_info.txt` for the `BLAS MKL` installation for `numpy`
+ `dev/install_log/pipdeptree[_rev].txt` lists installed package deps (and reversed)
+ `LICENSES_THIRD_PARTY.md` details the license for each package used


#### 2.2.2 (Optional best practice) Test successful installation of dev environment

```zsh
$> make test-dev-env
```

This will also add files `dev/install_log/[tests_numpy|test_scipy].txt` which
detail successful installation (or not) for `numpy`, `scipy`


#### 2.2.3 To install additional deps from the `pyproject` file:

NOTE the quotes required by zsh

```zsh
$> pip install ".[plots]"
```

#### 2.2.4 To remove the dev environment (Useful during env install experimentation):

From the dir above `oreum_survival/` project dir:

```zsh
$> make -C oreum_survival uninstall-env
```


### 2.3 Code Linting & Repo Control

#### 2.3.1 Pre-commit

We use [pre-commit](https://pre-commit.com) to run a suite of automated tests
for code linting & quality control and repo control prior to commit on local
development machines.

+ This is installed as part of `make dev` which you already ran.
+ See `.pre-commit-config.yaml` for details


#### 2.3.2 Github Actions

We use [Github Actions](https://docs.github.com/en/actions/using-workflows) aka
Workflows to run a suite of automated tests for commits received at the origin
(i.e. GitHub)

+ See `.github/workflows/*` for details


#### 2.3.3 Git LFS

We use [Git LFS](https://git-lfs.github.com) to store any large files alongside
the repo. This can be useful to replicate exact environments during development
and/or for automated tests

+ This requires a local machine install
  (see [Getting Started](https://git-lfs.github.com))
+ See `.gitattributes` for details


### 2.4 Configs for Local Development

Some notes to help configure local development environment

#### 2.4.1 Git config `~/.gitconfig`

```yaml
[user]
    name = <YOUR NAME>
    email = <YOUR EMAIL ADDRESS>
```


### 2.5 Install VSCode IDE

We strongly recommend using [VSCode](https://code.visualstudio.com) for all
development on local machines, and this is a hard pre-requisite to use
the `.devcontainer` environment (see section 3)

This repo includes relevant lightweight project control and config in:

```zsh
oreum_survival.code-workspace
.vscode/extensions.json
.vscode/settings.json
```

---

## 3. Code Standards

Even when writing R&D code, we strive to meet and exceed (even define) best
practices for code quality, documentation and reproducibility for modern
data science projects.

### 3.1 Code Linting & Repo Control

We use a suite of automated tools to check and enforce code quality. We indicate
the relevant shields at the top of this README. See section 1.4 above for how
this is enforced at precommit on developer machines and upon PR at the origin as
part of our CI process, prior to master branch merge.

These include:

+ [`ruff`](https://docs.astral.sh/ruff/) - extremely fast standardised linting
  and formatting, which replaces `black`, `flake8`, `isort`
+ [`interrogate`](https://pypi.org/project/interrogate/) - ensure complete Python
  docstrings
+ [`bandit`](https://github.com/PyCQA/bandit) - test for common Python security
  issues

We also run a suite of general tests pre-packaged in
[`precommit`](https://pre-commit.com).


### 3.2 Package-like structure

Where suitable, we break out commonly used functions and classes to module files
under the `src/` directory - this gives clear, convenient and easier code
control than when it's embedded inside notebooks. Note for clarity, that we
don't compile this code or release separately to the project.


## 4. Notebook Standards

General best practices for naming / ordering / structure.

### 4.1 General Principles

Every Notebook is:

+ Fully executable end-to-end, with linear non-cyclic flow
+ Living documentation with extensive text and plot-based explanation
+ Named starting with a 3-digit reference with group-based ordering to
indicate logical flow and dependencies, e.g:
  + `000` series: Overview, discussion, presentational documents
  + `100` series: Data Curation
  + `200` series: Exploratory Data Analysis
  + `300` series: Model Architecture and Data Transformations
  + `400` series: Model Design, Development, Evaluation and Inference
  + `500` series: Model Finalisation for Production Use
  + `600`, `700`, `800` series: used for specific extensions if needed
  + `900` series: Demos, Notes, Worked Explanations.


### 4.2. Live Notebooks

Live Notebooks are:

+ Present in the `/notebooks` directory
+ Part of the final R&D project flow, and required in order to reproduce
  the eventual findings & observations
+ Guaranteed to be up to date with the latest code in `src/`.

### 4.3 Rendered Notebooks

Rendered Notebooks are:

+ Present as rendered PDFs or `reveal.js` Slides in `notebooks/renders/`
+ Created somewhat as-needed for offline print-based discussion with
  stakeholders.

We use `nbconvert` to render to PDF or reveal.js HTML slides using configs.
From inside the `notebooks/` dir, run:

```sh
$> jupyter nbconvert --config renders/config_pdf.py
$> jupyter nbconvert --config renders/config_slides.py
```

### 4.4 Archived Notebooks

Archived Notebooks are:

+ Present in `notebooks/archive/`
+ No longer required, but kept around for historical audit, discussion,
  code examples
+ May have fallen behind the latest local code and/or methods.

---

## 5. Data Standards

See `data/README_DATA.md`

IMPORTANT NOTE on terminology / naming convention and dataset partitioning
based on the information present and dataset usage.

```text
Dataset terminology / partitioning / purpose:

|<---------- Relevant domain of all data for our analyses & models ---------->|
|<----- "Observed" historical target ------>||<- "Unobserved" future target ->|
|<----------- "Working" dataset ----------->||<----- "Forecast" dataset ----->|
|<- Training/CrossVal ->||<- Test/Holdout ->|
```

+ The **"Observed"** historical target dataset has:
  + a _known_ exogenous (target) feature value
  + known endogenous feature values to allow model regression
  + a hypothetical structure that we use to design the model

  + The **"Working"** dataset is the same as this "Observed" data, and may be
    split into:
    + A **Training/CrossVal** set used to fit the model. This may be
      partitioned into multiple Cross-Validation sets if required by the model
      architecture and fitting process
    + A **Test/Holdout** used to evaluate the model fit against a known target
    + We can use this Working set _in full_ when fitting the final model for
      Production, because this yields the most performant model

+ The **"Unobserved"** future target dataset has:
  + an _unknown_ exogenous (target) feature value
  + known endogenous feature values to allow model regression
  + a hypothetical structure that we use to design the model

  + The **"Forecast"** dataset is the same as this "Unobserved" data, and is
    generally what we will try to predict upon in Production
    + We might create predictions for individual datapoints or in bulk
    + _If_ the entities in the data evolve over time (e.g. a set of policies
      each with evolving premium payments and claim developments),
      and _if_ the endogenous features don't evolve with time (they are static
      not dynamic) then we can artificially create a Forecast dataset by
      extending the Working dataset forward in time.

Further note:

+ We may refer to "In-Sample" and "Out-of-Sample" datasets. The
  former is the data used to train the model and the latter to evaluate the
  model against a _known_ exogenous (target) value or forecast an _unknown_
  exogenous (target) value. So they can be used during Working or Forecasting.

+ Strictly speaking, our Bayesian modeling workflow does not require us to
  evaluate the model on a Test/Holdout set because we can use in-sample
  Pareto-smoothed Leave-One-Out (LOO-PIT) cross-validation testing. This is more
  powerful, and lets us fit & evaluate the model using the full Working set.

+ However, purely to aid reader comprehension and demonstrate the out-of-sample
  prediction workflow, we may use the practice of a known Test/Holdout set.


## 6. General Notes

We aim to make this project usable by all (developer, statistician, biz):

+ Logical structuring of code files with modularization and reusability
+ Small purposeful classes with abstracted object inheritance, and terse
  single-purpose functions
+ Variable and data parameterization throughout and use of config files to
  inject globals
+ Informative naming for classes / functions / variables / data, and
  human-readable, well-linted code
+ Specific and general error handling
+ Logging with rotation / archival
+ Detailed docstrings and type-hinting
+ Inline comments to explain complicated code / concepts to developers
+ Adherence to a consistent style guide and syntax, and use of linters
+ Well-organized Notebooks with logical ordering and “run-all” internal flow,
  and plenty of explanatory text and commentary to guide the reader
+ Use of virtual environments and/or containers
+ Build scripts for continuous integration and deployment
+ Unit tests and automated test scripts
+ Documentation to allow full reproducibility and maintenance
+ Commits have meaningful messages and small, iterative, manageable diffs to
  allow code review
+ Adherence to conventional branching structures, management of stale branches
+ Merges into master managed via pull requests (PRs) comprised of specific
  commits, and the PR linked to specific issue tickets
+ PRs setup to trigger manual code reviews and automated hooks to code
  formatting, unit testing, continuous integration (inc. automated integration
  and regression testing) and continuous deployment
+ New releases managed with tagging, fixed binaries, changelogs

---

Copyright 2024 Oreum OÜ t/a Oreum Industries. All rights reserved.
See LICENSE.md.

Oreum OÜ t/a Oreum Industries, Sepapaja 6, Tallinn, 15551, Estonia,
reg.16122291, [oreum.io](https://oreum.io)

---
Oreum OÜ &copy; 2024
