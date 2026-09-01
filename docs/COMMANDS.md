# Commands

Run repository commands in the `lit` Conda environment.

## Install

```bash
conda run -n lit python -m pip install -e ".[mesh,physics,ax,visualization,test]"
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
conda run -n lit python -m compileall -q experiments lumo scripts validation tests
```

Run Ruff:

```bash
conda run -n lit ruff check experiments lumo scripts validation tests
```

## Focused unit tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  python -m pytest -q tests/unit
```

## Live D435 contact localization

Install the RealSense/OpenCV GUI dependencies once, then run the online
color-image pipeline directly from the checkout:

```bash
conda run --no-capture-output -n lit \
  python -m pip install -e ".[camera]"
conda run --no-capture-output -n lit \
  python -u scripts/live_contact_localization.py
```

The default D435 color stream is 1920 x 1080 at 30 FPS. The application first
lets automatic color controls settle for 30 frames, then freezes the current
exposure, gain, and white balance before beginning the 30-frame LED calibration.
Keep the camera fixed during that geometry calibration. After that,
the five landmarks and contact dot follow gradual camera-pose changes every
frame. During confirmed no-contact operation, the absolute red detector
re-anchors the rigid array every 30 frames to limit recursive tracking drift.
If tracking is lost after a larger pose change, the viewer invalidates the old
view-dependent baseline and automatically collects 30 new frames;
press `b` again while unloaded. Pressing `b` collects 30 feature vectors and
uses their per-LED temporal median as the unloaded baseline; pressing it again
during collection restarts the acquisition. `r` explicitly starts geometry
recalibration without re-enabling automatic camera controls, and `q`, Escape,
or closing the window exits. The viewer does not save frames or estimates.
`LED_POSITIONS_IN_IMAGE_ORDER_MM` at the top of the script maps the detected
top-to-bottom image order to the physical fingertip Y axis; reverse it when the
camera is mounted from the opposite direction. A frame timeout triggers up to
ten explicit one-second reconnect attempts. A successful reconnect repeats the
photometric warmup/lock and clears the view-dependent baseline, so press `b`
again after LED recalibration.

For quantitative optical-response comparisons, exposure and gain must remain
fixed, and white balance should remain fixed when the color sensor supports it.
Acquire a new unloaded baseline after each intentional camera-viewpoint or
environmental-light change. Do not retune localization parameters between
contact locations. This protocol supports comparison with the same fixed
learning-free algorithm and a condition-specific unloaded reference; it is not
viewpoint invariance without recalibration. The response panel uses a fixed
unloaded-noise z scale with the 4-sigma contact gate marked, while retaining raw
DN values beside each LED bar.

The fingertip objective and Ax search contract have focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  python -m pytest -q \
    tests/unit/optimization/test_fingertip_objective.py \
    tests/unit/optimization/test_design_space.py
```

The publication visualization toolkit has one focused headless test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  python -m pytest -q tests/unit/visualization/test_publication_toolkit.py
```

## Publication visualization demo

Render one standalone panel and one composed 2 x 3 example as PDF/PNG, with an
additional SVG export for the composed figure:

```bash
conda run --no-capture-output -n lit \
  python -u validation/visualization/publication_toolkit_demo.py
```

The generated files are written beneath
`output/validation/publication_toolkit_demo/`.

Export the current fingertip X-Z parameterization as PDF, SVG, and PNG:

```bash
conda run --no-capture-output -n lit \
  python -u validation/visualization/fingertip_parameterization.py
```

The files are written beneath `output/publication/`.

Compose the double-column Figure 2 optomechanical pipeline from the frozen
production Newton state and a deterministic OptiX replay:

```bash
conda run --no-capture-output -n lit \
  python -u figures/fig2.py
```

The script writes `fig2.pdf`, `fig2.svg`, and `fig2.png` beside its source in
`figures/` and does not rerun Newton.

Smoke-test each structural mechanics ablation through its 2 N checkpoint:

```bash
for case in soft_only bonded_t lumo; do
  conda run --no-capture-output -n lit \
    python -u validation/contact-physics/simulation_ablation_study.py \
      --smoke "$case"
done
```

Run the matched Figure 3 Newton study. Then smoke the controlled optical replay,
run production OptiX on the exact saved mechanics states, and compose the figure
from the extended NPZ without rerunning Newton:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/simulation_ablation_study.py
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/simulation_ablation_study.py \
    --optical-smoke
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/simulation_ablation_study.py \
    --optics
conda run --no-capture-output -n lit \
  python -u validation/visualization/figure_3_hybrid_mechanics_ablation.py
```

The study writes structured mechanics and optical NPZ/CSV/JSON, Newton renders,
and a technical report beneath
`output/validation/hybrid_mechanics_ablation/`. The composition writes PDF,
SVG, and high-resolution PNG beneath `output/figures/`.

The production `--optics` replay also writes
`gap_sensitivity_samples.csv` and matching NPZ/JSON fields for the controlled
LUMO effective-gap values `0.01/0.19/0.50 mm`. It reuses the saved nominal
Newton vertices and varies only the recess floor plus LED source plane; it does
not rerun mechanics or change the BO search space.

Run the primary multi-design paper Figure 3 study. `--prepare` writes the complete
640-design campaign catalog and deterministic 40-design/120-variant manifest;
`--smoke` checks one Dragon and one Solaris morphology through 2 N. The full
command checkpoints 95 unique mechanics states, reusing compatible saved states,
and performs matched production
OptiX replay, computes paired effects, and then composes the double-column
figure:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/multi_design_void_ablation.py --prepare
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/multi_design_void_ablation.py --smoke
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/multi_design_void_ablation.py --all
conda run --no-capture-output -n lit \
  python -u figures/fig3.py
```

The primary study writes its catalog, manifest, per-variant raw NPZ states,
paired CSV, JSON summary, and report beneath
`output/validation/multi_design_void_ablation/`. The finalized composition
source and its `fig3.pdf`/`fig3.svg`/`fig3.png` exports live together beneath
`figures/`. `figures/fig3.py` also reads the four completed 160-observation BO
trial tables, validates their objective directions, recomputes empirical
Pareto membership and balanced trials, and writes `figure3_validation.md` next
to the ablation report. It uses four Pareto small multiples because the
standard and orientation-robust objective domains are not directly comparable.
All ablation source designs, including those from orientation-robust campaigns,
use the identical theta=0 fixed scenario. Campaign and material provenance
remain in metadata and are intentionally omitted from the ablation panels. The
one-location optical diagnostic `D(F)` is not `J_obs`.

Generate the learning-free brightest-10% red-channel heatmap from the current
5 mm experimental contact sweep:

```bash
conda run --no-capture-output -n lit \
  python -u figures/brightest10_red_contact_sweep.py
```

The script discovers `p0_Color.png` through `p6_Color.png` beneath
`output/experiments/`, detects the common five-LED array from their fixed-camera
median frame, and writes PDF/PNG plus a ROI debug montage beneath `figures/`.
When no matching unloaded image is present, the output filename and plot are
explicitly marked `median_centered` and `exploratory`.

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
