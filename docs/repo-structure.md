# Repository Structure

This document explains the purpose of each top-level folder in this repository, where new code should go, and the rules that keep the codebase organized.

## Guiding idea

This repository separates:

- **our importable code** from
- **runnable scripts** from
- **libraries dependencies** from
- **third-party comparison repos** from
- **configs, tests, notebooks, and docs**

The goal is to make it obvious:

- what belongs to the main project
- what is libraries
- what is experimental
- what is safe to import
- what is only meant to be run

## Top-level layout

```text
repo/
├─ src/project_name/
│  ├─ trainers/
│  ├─ models/
├─ scripts/
│  ├─ train.py
│  ├─ preprocess/
│  ├─ eval/
├─ notebooks/
│  ├─ exp1/
├─ third_party/
│  ├─ method_a/
│  └─ method_b/
├─ libraries/
│  ├─ plibs/
│  ├─ bolt/
│  ├─ blender_rendering/
├─ tests/
├─ configs/
│  ├─ project_name/
│  │  ├─ train/
│  │  ├─ preprocess/
│  │  ├─ eval/
│  │  └─ experiments/
│  │     └─ exp1/
├─ docs/
└─ pyproject.toml
```

## Folder responsibilities

### `src/project_name/`

This is the main Python package for this repository.

Put code here if it is:

- part of the core project
- meant to be imported by other project code
- reusable across scripts, tests, or notebooks

Examples:

- model definitions
- trainer implementations
- reusable preprocessing logic
- reusable evaluation utilities
- adapters for libraries libraries

Good imports look like:

```python
from project_name.models import ...
from project_name.trainers import ...
```

Do not put one-off script logic here unless it is truly reusable.

### `scripts/`

This contains runnable entrypoints and workflow wrappers.

Put code here if it is:

- mainly something a human runs
- orchestration code
- CLI-style glue code
- a thin wrapper around reusable code in `src/project_name/`

Examples:

- `scripts/train.py`
- dataset preprocessing runners
- evaluation runners
- experiment launch helpers

Scripts should stay thin. They should usually call into `project_name.*` rather than containing the full implementation themselves.

Preferred pattern:

- argument parsing in `scripts/`
- reusable logic in `src/project_name/`

### `notebooks/`

This contains exploratory and experimental notebook work.

Put things here if they are:

- temporary analysis
- visualization
- debugging
- early experiments

Rules:

- notebooks may import from `project_name`
- notebooks should not become the only place where important logic exists
- if notebook code becomes reusable or important, move it into `src/project_name/` or `scripts/`

### `third_party/`

This contains public external repositories used for comparison, benchmarking, or reference.

Examples:

- baseline methods
- public research repos
- reference implementations

Rules:

- treat code here as external
- do not refactor these repos unless there is a very specific reason
- avoid importing from `third_party/` inside `src/project_name/`
- prefer to run these repos separately, wrap them, or compare against their outputs

`third_party/` is for **comparison code**, not for code that defines the architecture of the main project.

### `external/`

This contains external dependency-like repositories that this project intentionally uses.

Examples:

- `plibs`
- `bolt`
- `blender_rendering`

These are not part of the main package, but they are closer to dependencies than to baselines.

Rules:

- code in `src/project_name/` may depend on these repositories
- prefer to use them through normal package/module imports, not filesystem-path imports
- keep adaptation logic on our side small and explicit
- if integration becomes complex, add adapters under `src/project_name/`

Examples:

- `src/project_name/integrations/`
- `src/project_name/rendering/`
- `src/project_name/backends/`

`external/` is for **dependency repos we use**.
`third_party/` is for **repos we compare against**.

### `tests/`

This contains tests for the main project.

Put here:

- unit tests
- integration tests
- regression tests
- smoke tests for scripts or configs when appropriate

Rules:

- tests should primarily exercise `project_name`
- tests may also validate integrations with `external/`
- avoid tests that depend heavily on mutable internals of `third_party/`

### `configs/`

This contains structured configuration for workflows.

Current layout:

- `configs/project_name/train/`
- `configs/project_name/preprocess/`
- `configs/project_name/eval/`
- `configs/project_name/experiments/`

Put here:

- YAML/TOML/JSON config files
- experiment settings
- evaluation settings
- preprocessing options

Rules:

- prefer configs here over hardcoded values in scripts
- experiment-specific config should live under `configs/experiments/`
- scripts should load configs from here rather than duplicating settings

### `docs/`

This contains human-readable project documentation.

Put here:

- architecture notes
- workflow docs
- experiment conventions
- repo structure docs
- onboarding docs

Good examples:

- `docs/repo-structure.md`
- `docs/training.md`
- `docs/evaluation.md`
- `docs/experiments.md`

## Dependency direction

Use this as the intended dependency flow:

```text
scripts/         -> src/project_name
notebooks/       -> src/project_name
tests/           -> src/project_name
src/project_name -> libraries
scripts/         -> third_party   (run/wrap when needed)
```

Avoid this:

```text
src/project_name -> third_party
src/project_name -> scripts
```

In other words:

- the package should not depend on runnable scripts
- the package should not be architecturally coupled to baseline repos

## Where new code should go

### Put code in `src/project_name/` if:

- it is reusable
- it is importable
- it is part of the core project
- more than one script or notebook will use it

### Put code in `scripts/` if:

- it is mainly an entrypoint
- it parses arguments
- it launches workflows
- it is glue code around reusable project functions

### Put code in `notebooks/` if:

- it is exploratory
- it is temporary
- it is for analysis or visualization

### Put code in `third_party/` if:

- it comes from a public external repo
- it is used as a baseline or comparison method
- it is not part of the main project package

### Put code in `libraries/` if:

- it is a repo we build and depend on
- it behaves more like a dependency than a benchmark
- our project uses it as part of normal workflows

## Import rules

### Good

```python
from project_name.models import ...
from project_name.trainers import ...
```

### Usually okay

```python
from plibs import ...
from bolt import ...
```

### Avoid

```python
from third_party.method_a import ...
from scripts.train import ...
```

### Strongly avoid

- scattered `sys.path` hacks
- importing code directly by filesystem location
- making `third_party/` part of the core package design

## Practical rules for contributors

1. Keep `src/project_name/` focused on reusable project code.
2. Keep `scripts/` thin.
3. Keep `third_party/` isolated.
4. Keep `libraries/` clearly dependency-like.
5. Move important notebook logic into `src/` or `scripts/`.
6. Put config in `configs/`, not inline in scripts.
7. Add docs to `docs/` when introducing new workflows or conventions.

## Quick decision checklist

When adding a new file, ask:

- Is this importable project logic?
  -> put it in `src/project_name/`

- Is this something a human runs?
  -> put it in `scripts/`

- Is this experimental analysis?
  -> put it in `notebooks/`

- Is this a public baseline/reference repo?
  -> put it in `third_party/`

- Is this a repo we build and we actively depend on?
  -> put it in `libraries/`

- Is this configuration?
  -> put it in `configs/`

- Is this documentation?
  -> put it in `docs/`

## Notes for Claude

When working in this repository:

- treat `src/project_name/` as the source of truth for reusable project code
- keep `scripts/` as workflow entrypoints and wrappers
- do not move `third_party/` code into the main package unless explicitly requested
- prefer adapters in `src/project_name/` over spreading integration details throughout the codebase
- prefer updating or creating docs in `docs/` when introducing new workflows or conventions
- promote reusable logic out of notebooks and into `src/` or `scripts/`

## Summary

This repository is organized around a simple principle:

- **`src/`** is our code
- **`scripts/`** is what we run
- **`third_party/`** is what we compare against
- **`libraries/`** is what we build and depend on
- **`configs/`** is how workflows are configured
- **`notebooks/`** is for exploration
- **`tests/`** verifies behavior
- **`docs/`** explains the system
