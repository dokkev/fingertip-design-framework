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

The full-finger objective and Ax search contract have focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q \
    tests/unit/optimization/test_full_finger_objective.py \
    tests/unit/optimization/test_height_constraint.py
```

## Geometry and mechanics viewers

Render the analytic bond and the full five-LED mesh:

```bash
conda run -n lit python validation/fingertip/view_bond_geometry.py
conda run --no-capture-output -n lit \
  python -u validation/fingertip/view_fingertip_5led.py
```

Open the full mesh in Newton ViewerGL:

```bash
conda run --no-capture-output -n lit \
  python -u validation/fingertip/view_fingertip_5led_newton.py
```

Run the short full-finger Newton compatibility smoke:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/fingertip_5led_smoke.py
```

## Production evaluator checks

Validate the current GPU-default, constant-speed force-threshold path on four
concurrent scenarios:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/instantaneous_first_crossing.py
```

Run one nominal full-finger raw Newton-to-OptiX evaluation and reload its NPZ:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/full_finger_raw_evaluator.py
```

Run the expensive complete production-objective freeze validation:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/full_finger_production_objective_freeze.py
```

These commands perform GPU simulation and OptiX tracing; they are not part of
the focused unit suite.

## Production BO

Before a long campaign, run the single end-to-end smoke command:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/mobo_smoke.py
```

This uses the exact production settings, evaluates one successful full-finger
morphology in a fresh timestamped `output/validation/mobo_smoke/` directory,
and verifies raw NPZ/CSV output, atomic Ax state, and resume reload. It is an
expensive GPU smoke, not a lightweight unit test.

Review the user settings at the top of `scripts/run_mobo.py`, use a fresh output
directory for a new scientific contract, then run:

```bash
conda run --no-capture-output -n lit python -u scripts/run_mobo.py
```

The campaign is sequential and resumable. It evaluates five geometry variables
on the 0.5 mm lattice, fixes `flat_pad_width_mm=30` and `void_height_mm=0`, and
maximizes `J_contact` and `J_obs` independently. Production mechanics use the
GPU CUDA-graph checkpoint backend by default: a constant `5 mm/s` approach and
the first samples at or above `5/10/15/20 N`, with no servo or dwell.

Do not mix an output directory with an older run-config schema. Ax state and one
compressed raw NPZ per completed trial are written beneath the configured
`OUTPUT_DIRECTORY`.

## Generated artifacts

Generated simulation, validation, and optimization outputs belong under
`output/`, which is ignored by Git except for its placeholder.
