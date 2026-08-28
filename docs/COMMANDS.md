# Commands

Run repository commands in the `lit` Conda environment.

## Install

```bash
conda run -n lit python -m pip install -e ".[mesh,physics,ax,test]"
```

OptiX and CUDA are system dependencies. The ray tracer also needs the
header-only OptiX Toolkit ShaderUtil include directory:

```bash
conda env config vars set -n lit \
  OTK_INCLUDE_DIR=/path/to/optix-toolkit/ShaderUtil/include
```

`scripts/run_mobo.py` supplies its sibling checkout as a fallback; an existing
environment value takes precedence.

## Static checks

Compile repository Python without launching a simulation:

```bash
conda run -n lit python -m compileall -q lumo scripts validation tests
```

Run Ruff:

```bash
conda run -n lit ruff check lumo scripts validation tests
```

## Focused unit tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q tests/unit
```

The fingertip objective and Ax search contract have focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q \
    tests/unit/optimization/test_fingertip_objective.py \
    tests/unit/optimization/test_design_space.py
```

## Geometry and mechanics viewers

Render the analytic bond and the complete fingertip mesh:

```bash
conda run -n lit python validation/fingertip/view_bond_geometry.py
conda run --no-capture-output -n lit \
  python -u validation/fingertip/view_fingertip.py
```

Open the full mesh in Newton ViewerGL:

```bash
conda run --no-capture-output -n lit \
  python -u validation/fingertip/view_fingertip_newton.py
```

View the mechanics-equivalent inverse-relative path for the default
`+30 deg` angled indentation scenario:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/angled_indentation_viewer.py
```

Run the short fingertip Newton compatibility smoke:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/fingertip_smoke.py
```

## Production evaluator checks

Validate the current GPU-default, constant-speed force-threshold path on four
concurrent scenarios:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/instantaneous_first_crossing.py
```

Run one nominal fingertip raw Newton-to-OptiX evaluation and reload its NPZ:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/fingertip_raw_evaluator.py
```

Run the expensive complete production-objective freeze validation:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/fingertip_production_objective_freeze.py
```

These commands perform GPU simulation and OptiX tracing; they are not part of
the focused unit suite.

## Production BO

Before a long campaign, run the single end-to-end smoke command:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/mobo_smoke.py
```

This uses the exact production settings, evaluates one successful fingertip
morphology in a fresh timestamped `output/validation/mobo_smoke/` directory,
and verifies raw NPZ/CSV output, atomic Ax state, and resume reload. It is an
expensive GPU smoke, not a lightweight unit test.

Review the user settings at the top of `scripts/run_mobo.py`, use a fresh output
directory for a new scientific contract, then run:

```bash
conda run --no-capture-output -n lit python -u scripts/run_mobo.py
```

`scripts/run_mobo.py` is the only campaign entry; `ax_bo.py` is a library module
and has no separate CLI. The campaign is sequential and resumable. It evaluates five geometry variables
on the 0.5 mm lattice, fixes `flat_pad_width_mm=30`, and maximizes `J_contact`
and `J_obs` independently. `INDENTATION_ANGLES_DEG` selects the physical
fingertip angles included in the scenario Cartesian product; `(0.0,)` is the
ordinary pad-normal case. Angled campaigns need the conservative common air
approach configured by `INITIAL_CLEARANCE_M`. Production mechanics use the fixed four-world GPU
CUDA-graph checkpoint path: a constant `5 mm/s` approach and
the first samples at or above each configured force threshold, with no servo
or dwell. The production objective requires exactly four strictly increasing
force thresholds.

`INITIAL_MORPHOLOGIES_MM` lists informed physical designs in
`(flat-pad height, semiellipse height, stem width, stem height, void width)`
order. On a fresh campaign these designs are evaluated first under the current
scientific contract; previous objective values are never imported. The five
completed initial morphologies count toward Ax's initialization budget of 13,
leaving eight fresh exact-feasible Sobol trials before `FEASIBLE_MBM` begins.

The current Dragon Skin orientation-aware campaign uses 75 scenarios per
morphology: `5 angles x 3 spheres x 5 contact-Y locations`. Before launching
it, run the one-morphology trial-117 Newton-to-OptiX smoke:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/orientation_aware_mobo_smoke.py
```

Do not mix an output directory with an older run-config schema. Ax state and one
compressed raw NPZ per completed trial are written beneath the configured
`OUTPUT_DIRECTORY`.

## Generated artifacts

Generated simulation, validation, and optimization outputs belong under
`output/`, which is ignored by Git except for its placeholder.
