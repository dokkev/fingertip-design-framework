# Commands

Use the `lit` Conda environment for all repository commands:

```bash
conda activate lit
```

## Install current dependencies

```bash
python -m pip install -e ".[mesh,physics,ax,test]"
```

`mesh` supplies Gmsh, `physics` supplies Newton/Warp and rigid asset loading,
and `ax` supplies Ax 1.3.1. CUDA, OptiX, and GPU drivers are externally managed.
The editable install exposes the sole framework namespace from `lumo/`;
repository scripts do not insert the checkout into `sys.path`.

## Focused tests

```bash
./scripts/tools/pytest_lit tests/unit/finger tests/unit/mesh -q
./scripts/tools/pytest_lit tests/unit/contact tests/unit/physics -q
./scripts/tools/pytest_lit tests/unit/ray_tracing tests/unit/optimization -q
./scripts/tools/pytest_lit tests/unit/optimization/test_evaluator.py -q
```

The Newton smoke tests require the CUDA-capable `lit` environment:

```bash
./scripts/tools/pytest_lit tests/smoke/physics -q -m "smoke and physics"
```

Visualize the analytic carrier-silicone bond in the XZ cross-section:

```bash
conda run -n lit python validation/fingertip/view_bond_geometry.py
```

Run the procedural flat-plate contact smoke explicitly:

```bash
conda run -n lit python validation/contact-physics/flat_plate_contact.py
```

The script owns its transient-force stopping threshold and maximum simulation
time locally.

To render every step and keep the final state open until the window closes:

```bash
conda run -n lit python validation/contact-physics/flat_plate_contact.py --viewer
```

Run the three-location spherical indentation validation explicitly:

```bash
conda run -n lit python validation/contact-physics/sphere_indentation.py
```

The 5, 10, and 20 mm diameter URDF spheres run in independent simulations at
`X=-7.5`, `0`, and `+7.5 mm`, respectively. Each prescribed positive-Z motion
stops at a transient reaction-force target of `20 N`.

Run the Dragon Skin 10 NV Poisson-ratio contact sweep explicitly:

```bash
conda run -n lit python validation/contact-physics/poisson_ratio_sweep.py
```

The sweep compares `0.48`, `0.49`, and `0.495` using one fixed shear modulus
and reports force-target timing and tetrahedral volume change. It performs
multiple Newton simulations and is not part of ordinary focused tests.

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
  --execution-config config/lumo_execution.yaml \
  --output output/validation/lumo3d_trajectory
```

This evaluates the fixed current protocol: three semantic locations, two
radii, and three absolute depths. It writes only under `output/`.

To verify exact evaluator reproducibility across fresh Python/Warp processes,
run the nominal 18-state gate three times:

```bash
python -m validation.optimization.lumo3d_repeatability \
  --execution-config config/lumo_execution.yaml \
  --output output/validation/optimization/lumo3d_repeatability
```

The gate compares mechanics artifact digests, contact-state fingerprints,
optical field artifact digests, scalar transport diagnostics, and hexadecimal
objective values. Any differing bit or incomplete state grid is a failure.

## Bounded 6D Test BO

The bounded test runner is a validation workflow, not a production campaign:

```bash
python -m validation.optimization.lumo6d_test_bo \
  --execution-config config/lumo_execution.yaml \
  --output output/validation/optimization/lumo6d_test_bo
```

It targets six successful Sobol and four successful model-based observations only after OptiX
preflight, using the same typed execution YAML as production. Do not use it as
a substitute for a reviewed production campaign, and do not run it as part of
ordinary focused test execution.

## Production BO entry point

Use `run_bo_ideal` as the canonical human-facing entry point. The strict YAML
owns only device, mesh, Newton, first-contact, and optical transport numerical
settings. Morphology/material/LED, the protocol, objective, and design bounds
remain visible code contracts in the campaign engine.

Run the cheap gate first:

```bash
conda run -n lit python -m scripts.optimization.run_bo_ideal \
  --preflight \
  --execution-config config/lumo_execution.yaml
```

For a minimal production-path smoke only:

```bash
conda run -n lit python -m scripts.optimization.run_bo_ideal \
  --smoke \
  --execution-config config/lumo_execution.yaml \
  --output output/optimization/bo_smoke
```

Without `--smoke`, the runner uses the authoritative 18-state production
protocol. `--smoke` is the only route to the reduced two-state protocol. Both
use the production evaluator, Ax adapter, and exact-contract evaluation
registry. External `--registry PATH` reuse is currently limited to smoke and
validation runs. Production rejects external registry reuse until
same-contract evaluator reproducibility is established; its campaign-local
registry remains available for exact resume and in-campaign duplicate handling.
A Git-tracked registry is rejected, and concurrent campaigns targeting the
same registry are serialized with an exclusive advisory lock. A
shared CUDA/OptiX/Gmsh/Newton prerequisite failure aborts before candidate
registration; a morphology failure is recorded as a candidate result. Smoke
defaults to one successful Sobol observation, no MBM observation, two total
evaluator calls including nominal, and one generated proposal.

Production requires an explicit MBM success target and independent evaluator
and proposal caps. For example:

```bash
conda run -n lit python -m scripts.optimization.run_bo_ideal \
  --execution-config config/lumo_execution.yaml \
  --search-successes 4 \
  --max-evaluations 20 \
  --max-proposals 30 \
  --output output/optimization/bo_production
```

Production defaults to six successful Sobol observations and rejects a search
target below one. Candidate failures do not reduce either success target.
`--max-evaluations` counts actual evaluator calls including nominal;
`--max-proposals` counts every Ax-generated proposal including feasibility
rejects and duplicates. Budget exhaustion, optimizer stall, nominal failure,
or an unmet success target returns exit code `3`. Infrastructure, config, or
persistence failure returns `2`.

Production requires a clean Git source snapshot by default. If a reviewed
dirty snapshot is intentional, add `--allow-dirty`; its tracked diff and
untracked source contents are hashed into provenance. Same-contract registry
records from another source snapshot are rejected unless
`--allow-cross-revision-cache` is explicit. These opt-ins are persisted in the
campaign audit and exact resume contract.

To continue an interrupted campaign, pass `--resume` explicitly with the
campaign directory or its current `checkpoint.json` pointer. Selecting an
older immutable checkpoint directory is rejected because rollback also needs
registry/trial reconciliation. The runner restores the public Ax JSON state, reconciles any
pending candidate before requesting a new proposal, and fails fast on fixed
input, budget, source, or Ax-package mismatches. Existing output alone never
triggers resume. Omitted budget and YAML arguments are restored from the
persisted campaign config; explicit values still have to match exactly.

## Representative scientific convergence harness

After the implementation gates pass and an expensive validation run is
explicitly authorized, run the representative Newton/mesh/optical workflow:

```bash
conda run -n lit python -m validation.optimization.lumo3d_scientific_convergence \
  --execution-config config/lumo_execution.yaml \
  --output output/validation/optimization/lumo3d_scientific_convergence
```

The workflow evaluates five deterministic feasible morphologies. Newton uses
the preserved displacement thresholds. The production mechanics setting is
100 VBD iterations with a 0.0125 mm load increment at 0.00025 s; its strict
reference uses 160 iterations with a 0.00625 mm increment at 0.000125 s, so
both retain the same 50 mm/s prescribed indentation rate. Mesh and optical objective sensitivity
remain `INCONCLUSIVE` until evidence-backed thresholds are reviewed. The
current mechanics artifacts do not expose an approved reaction-force metric,
so mesh force is recorded as `unsupported`, never fabricated as zero. This is
an expensive GPU/Newton/OptiX command and is not part of ordinary unit tests.

## Newton viewer helpers

Interactive Newton viewer support is kept in `lumo.physics.newton.viewer` for debugging.
It is intentionally not a general plotting framework. Production evaluation
does not open a viewer or alter solver state for display.

## Rigid OBJ asset preparation

Prepare a deterministic parametric sphere asset with:

```bash
python scripts/assets/prepare_object_mesh.py \
  --radius-mm 2.0 \
  --subdivisions 2
```

The default output is under `assets/objects/`. Runtime code loads an OBJ or STL
with `lumo.util.mesh_io.load_mesh()`. The loader requires an explicit
`scale_m_per_unit` and returns a `newton.Mesh` whose vertices are in metres.

## Generated artifacts

Validation, optimization, and benchmark outputs belong under `output/`. Existing
scientific artifacts are not overwritten by cleanup or documentation commands.
