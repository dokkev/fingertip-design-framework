# Commands

Use the `lit` Conda environment for all repository commands:

```bash
conda activate lit
```

## Install current dependencies

```bash
python -m pip install -e ".[mesh,physics,ax,test]"
```

`mesh` supplies Gmsh, `physics` supplies Newton/Warp, and `ax` supplies Ax
1.3.1. CUDA, OptiX, and GPU drivers are externally managed.

## Focused tests

```bash
./scripts/pytest_lit tests/unit/model tests/unit/mesh -q
./scripts/pytest_lit tests/unit/contact tests/unit/physics -q
./scripts/pytest_lit tests/unit/optics tests/unit/optimization -q
./scripts/pytest_lit tests/unit/validation/test_lumo3d_trajectory_evaluator.py -q
```

The Newton smoke tests require the CUDA-capable `lit` environment:

```bash
./scripts/pytest_lit tests/smoke/physics -q -m "smoke and physics"
```

## OptiX gate before production BO

First run the environment diagnosis:

```bash
conda run -n lit python -m optics.optix.doctor --json
```

Immediately before a long production campaign, run the real runtime smoke:

```bash
conda run -n lit python -m validation.optics.production_optix_smoke
```

The distinction is important:

- `doctor` diagnoses dependencies, headers, versions, and device settings;
- `production_optix_smoke` uses the production OptiX runtime to compile, build
  a GAS/SBT, launch real rays, copy results back, and validate hit/miss output;
- the production BO preflight calls that same underlying smoke function and
  aborts before Ax creates a candidate when infrastructure is unavailable.

The smoke command is the recommended explicit gate before unattended BO runs.

## Current trajectory validation

```bash
python -m validation.optimization.lumo3d_trajectory_validation \
  --output output/validation/lumo3d_trajectory
```

This evaluates the fixed current protocol: three semantic locations, two
radii, and three absolute depths. It writes only under `output/`.

## Bounded 6D Test BO

The bounded test runner is a validation workflow, not a production campaign:

```bash
python -m validation.optimization.lumo6d_test_bo \
  --output output/validation/optimization/lumo6d_test_bo
```

It performs six Sobol and four model-based proposals only after OptiX
preflight. Do not use it as a substitute for a reviewed production campaign,
and do not run it as part of ordinary focused test execution.

## Newton viewer helpers

Interactive Newton viewer support is kept in `physics._viewer` for debugging.
It is intentionally not a general plotting framework. Production evaluation
does not open a viewer or alter solver state for display.

## Generated artifacts

Validation and benchmark outputs belong under `output/validation/`. Existing
scientific artifacts are not overwritten by cleanup or documentation commands.
