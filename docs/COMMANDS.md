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
./scripts/tools/pytest_lit tests/unit/model tests/unit/mesh -q
./scripts/tools/pytest_lit tests/unit/contact tests/unit/physics -q
./scripts/tools/pytest_lit tests/unit/optics tests/unit/optimization -q
./scripts/tools/pytest_lit tests/unit/optimization/test_evaluator.py -q
```

The Newton smoke tests require the CUDA-capable `lit` environment:

```bash
./scripts/tools/pytest_lit tests/smoke/physics -q -m "smoke and physics"
```

## OptiX gate before production BO

First run the environment diagnosis:

```bash
conda run -n lit python scripts/tools/optix_doctor.py --json
```

Immediately before a long production campaign, run the real runtime smoke:

```bash
conda run -n lit python -m scripts.tools.optix_smoke
```

The distinction is important:

- `optix_doctor.py` diagnoses dependencies, headers, versions, and device settings;
- `optix_smoke` uses the production OptiX runtime to compile, build
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

## Production BO entry point

The production runner keeps its experiment settings in a visible `USER CONFIG`
block and requires explicit trial opt-in. Run the cheap gate first:

```bash
conda run -n lit python scripts/optimization/run_bo.py --preflight
```

For a minimal production-path smoke only:

```bash
conda run -n lit python scripts/optimization/run_bo.py \
  --smoke \
  --trials 1 \
  --output output/optimization/bo_smoke
```

Without `--smoke`, the runner uses the authoritative 18-state production
protocol. `--smoke` is the only route to the reduced two-state protocol. Both
use the production evaluator, Ax adapter, and exact-contract evaluation
registry. Pass `--registry PATH` to reuse exact results across output
directories; contract IDs prevent reuse across different fixed inputs. A
shared CUDA/OptiX/Gmsh/Newton prerequisite failure aborts before candidate
registration; a morphology failure is recorded as a candidate result.

## Newton viewer helpers

Interactive Newton viewer support is kept in `physics.newton.viewer` for debugging.
It is intentionally not a general plotting framework. Production evaluation
does not open a viewer or alter solver state for display.

## Rigid OBJ asset preparation

Prepare a deterministic parametric sphere asset with:

```bash
python scripts/assets/prepare_object_mesh.py \
  --radius-mm 2.0 \
  --subdivisions 2
```

The default output is under `assets/objects/`. Runtime code can load one
asset with `mesh.load_obj()` or load all top-level OBJ files in a directory
with `mesh.load_obj_directory()`. The loader requires an explicit
`scale_mm_per_unit` and validates the neutral closed rigid-mesh contract.

## Generated artifacts

Validation, optimization, and benchmark outputs belong under `output/`. Existing
scientific artifacts are not overwritten by cleanup or documentation commands.
