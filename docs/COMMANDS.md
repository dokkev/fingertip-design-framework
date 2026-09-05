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

## Physical contact dataset collection

Install the D435, OpenCV, and Bota Rokubi serial dependencies:

```bash
conda run --no-capture-output -n lit \
  python -m pip install -e ".[acquisition]"
```

On Ubuntu, give the user access to the Rokubi serial device once, then log out
and back in so the group membership takes effect:

```bash
sudo usermod -aG dialout "$USER"
```

Launch the collector with the standard 1920 x 1080, 30 FPS D435 stream and
Rokubi at `/dev/ttyUSB0`:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/collect_contact_dataset.py \
    --bota-port /dev/ttyUSB0
```

The fixed RGB defaults are 1500 µs exposure, gain 0, and 4600 K white balance.
The three corresponding CLI options can override them, but every morphology in
one comparison must use identical values. The collector disables automatic
exposure and white balance, verifies the camera read-back, and records the actual
values in `session.json`. Exposure remains expressed in microseconds at the CLI
and in the dataset; the RealSense adapter converts it to the D435 RGB sensor's
native 100-µs units (for example, 1500 µs becomes native value 15).

The saved burst rate defaults to 5 Hz and can be changed independently of the
30 FPS camera stream with `--capture-rate-hz`.

Keep the Rokubi completely unloaded during startup tare and every manual
`TARE`. After startup tare, enter the session material, morphology, and specimen
ID, then select `CREATE SESSION`. Those specimen values and the camera and
acquisition configuration are fixed for the whole session. Start a new session
when the physical specimen or camera setup changes.

For each loaded run click the 10 or 30 mm spherical-indenter button and select a
hole.
Enter `Runs in series`, then press `START SERIES`. The collector automatically
assigns the next one-based repetition index for that `indenter + hole` pair and
displays both series progress and the current read-only `Repetition Index`.
Each repetition runs the continuous
2 → 5 → 10 → 15 N progression; do not release between
successful targets. The vertical force gauge shows the current force as a bar,
the active target as a horizontal line, and the accepted margin as a shaded
band.
Hole 1 is distal and Hole 6 is proximal. The accepted band is ±1 N at 2 N,
±20% at 5 N, and ±10% at 10 and 15 N. Hold the band for the 0.25 s settling
phase and the complete 0.25 s recording interval. The default elapsed-time
schedule records at 5 Hz with the start included and the end excluded, yielding
exactly two synchronized RGB/Rokubi frames. A missed scheduled observation,
camera delivery drop, or force-band excursion discards the whole partial target
attempt. After 15 N completes, fully release the indenter. If more repetitions
remain, the collector requires force at or below 1 N continuously for 0.25 s
and then creates the next independent run automatically. `ABORT` deletes only
the current incomplete run, cancels the remainder of the series, and preserves
earlier completed repetitions.

Use `CAPTURE UNLOADED` separately within the specimen session. It saves a
synchronized burst while `F_mag ≤ 1.0 N` is maintained
through the 0.25 s settling and 0.25 s recording intervals. The same 5 Hz
elapsed-time schedule yields exactly two frames, and any force excursion
discards the entire unloaded attempt. An unloaded reference is not required
before every loaded run. By default, sessions are written to
`output/contact_dataset/`, which is
ignored by Git, using dataset format v3. `session.json` owns specimen, camera, sensor,
tare, and acquisition configuration. Session directories use
`YYYY-MM-DD_<material>_<morphology>` and add `_01`, `_02`, ... only when that
same-day name already exists. Each `run.json` owns indenter, hole, repetition,
and run status. A finalized force or unloaded directory contains only
lossless PNGs under `frames/` and raw synchronized measurements in `frames.csv`.
It has no `metadata.json` or `summary.json`. Aborting discards the entire loaded
run, including any completed force directories; incomplete `.partial` attempts
are also deleted.

Exercise the complete GUI and D435 without a physical Rokubi using the prominent
manual-force mock mode:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/collect_contact_dataset.py --mock
```

Mock sessions cannot enter the physical dataset namespace: they are written
under `output/contact_dataset/mock/MOCK_*` and carry `sensor_mode: mock`.

## Live D435 contact localization

Create the one-time manual LED ground truth for the fixed-finger reference
images, then measure the offline calibration against those labels:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_fixed_finger_calibration.py --label-ground-truth
conda run --no-capture-output -n lit \
  python -u validation/validate_fixed_finger_calibration.py
```

The first command records five clicks per image, in distal-to-proximal order,
in `validation/fixed_finger_led_ground_truth.json`. The second writes
per-condition NPZ calibrations, measured pixel errors, a summary CSV, and one
diagnostic PNG beneath `output/validation/fixed_finger_calibration/`. The PNG
includes the selected line's positive-red profile, five physical score windows,
and the maximum used from each window. The CSV independently checks that their
sum equals the stored line score and reports whole-line mean contrast as an
explicitly non-scored diagnostic. The command fails when the manual labels are
absent and performs no live tracking, joint-state processing, or per-frame
geometry reconstruction.

To regenerate only the silhouette, side lines, five score windows, LED line,
and sampling-strip diagnostic without claiming LED-position accuracy:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_fixed_finger_calibration.py --diagnostic-only
```

Run the Solaris-only fixed-camera five-lobe localizer on the saved normal and
dark-room reference images:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_solaris_led_localization.py
```

The command calibrates once from the six-frame normal temporal median and
overlays those same LED centers on all six loaded frames. It also writes
leave-one-frame-out stability, normal/dark profile diagnostics,
terminal-leakage stress artifacts, coordinates, and an
empty or populated per-LED ground-truth error CSV beneath
`output/validation/solaris_led_localization/`. It detects the first regular
five-lobe sequence in the two silhouette-side raw-red profiles; it does not use
projective geometry, physical pixel scale, or a periodic bright/dark template.
Without manual labels it reports accuracy as `UNAVAILABLE`. Record optional
validation-only labels by clicking LED 1 through LED 5 in distal-to-proximal
order:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_solaris_led_localization.py \
  --label-ground-truth
```

Characterize unloaded-relative Dragon Skin optical magnitude, longitudinal
signatures, and pairwise RMS separation with the fixed sampling strip:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_optical_morphology_analysis.py
```

Solaris is deliberately omitted because no same-condition unloaded Solaris
reference is currently checked in.

Analyze any number of format-v3 physical contact sessions and create scientific
results plus the compact, image-free `raw_data_summary`:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/analyze_morphologies.py \
  output/contact_dataset/2026-09-04_solaris_baseline \
  output/contact_dataset/2026-09-04_solaris_flat_opt \
  --output output/analysis/solaris_compare
```

Each invocation reads the raw PNGs once and writes `results/`, `figures/`, and
`raw_data_summary/`, plus `raw_data_summary.zip` for later upload. The fixed
interior strip is calibrated once per specimen. Frames within one hold are
aggregated by median, and the primary profile slope uses actual measured force.
`--expected-repetitions` changes only coverage QC. `--hole-spacing-mm` adds a
physical-spacing-normalized neighboring-location diagnostic when that spacing
is trusted. The summary contains no PNGs and no mechanical-deformation claim.
`suspect_runs.csv` is a deterministic manual-inspection ranking and never
repairs or relabels the dataset.

Replay the smooth emissive segmentation on the checked-in 13-image reference
set, report fixed-extrinsic stability/runtime, and regenerate its overlays:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_fingertip_boundary.py
```

Characterize contact representations with one fixed canonical map per recorded
sequence, write CSV/PNG/PDF evidence, and optionally compare the five fixed
dense feature definitions:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_contact_localization.py
conda run --no-capture-output -n lit \
  python -u validation/validate_contact_localization.py --compare-features
```

Ablate only the Dragon Skin longitudinal canonical span while holding the
segmentation result and optical descriptor fixed:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_contact_canonicalization.py
```

Export a Solaris dense-template model for online replay:

```bash
conda run --no-capture-output -n lit \
  python -u validation/validate_contact_localization.py \
    --export-template output/validation/contact_localization/solaris_dense_top10.npz
```

Develop and inspect the camera-extrinsic-independent fingertip boundary before
running contact localization:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/live_fingertip_boundary.py
```

The geometry viewer shows RGB, the coarse paired-LSD prior, raw selected
GrabCut component, final emissive fingertip mask, smooth contour, and the
existing red-detector LED centers/response ROIs. It reports pad width, mask
area, geometry scale, and segmentation runtime. Lab-a, grayscale, HSV, and the
emission score are geometry-only. The viewer does not run contact photometry,
tracking, or localization and writes no files.

Install the RealSense/OpenCV GUI dependencies once, then run the online
color-image pipeline directly from the checkout:

```bash
conda run --no-capture-output -n lit \
  python -m pip install -e ".[camera]"
conda run --no-capture-output -n lit \
  python -u scripts/live_contact_localization.py
```

Select one shared dense observer and optionally load an offline-generated
template model:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/live_contact_localization.py --observer dense-top10
conda run --no-capture-output -n lit \
  python -u scripts/live_contact_localization.py \
    --observer dense-top10 \
    --template-model output/validation/contact_localization/solaris_dense_top10.npz
```

The loaded model's serialized transverse interval, reduction, smoothing, and
other feature parameters are used directly online. The observer name selects
and verifies only the descriptor mode. Dense estimates report optical position
conditional on contact; this camera-only viewer has no contact-existence gate.

The default D435 color stream is 1920 x 1080 at 30 FPS. The application first
discards 30 frames while the camera's default automatic exposure and white
balance settle, then begins the 30-frame global fingertip/LED calibration
without changing any photometric controls.
Keep the camera fixed during that geometry calibration. After that,
the five landmarks and contact dot follow gradual camera-pose changes every
frame. During confirmed no-contact operation, the absolute red detector
re-anchors the rigid array every 30 frames inside a dilation of the current five
ROI polygons; it does not rerun global segmentation. Corrections larger than
half the current LED spacing are rejected. Full emissive-fingertip segmentation
runs only for initial acquisition, explicit recalibration, and recovery. Normal
30 Hz motion continues to use grayscale LK plus one rigid similarity fit. Dense
modes additionally move the reference canonical map, remap the RGB frame,
extract one profile, and run template correlation when a model is loaded. The
UI reports rolling stage medians; initialization/recovery latency is separate
from normal processing.
If tracking is lost after a larger pose change, the viewer invalidates the old
view-dependent baseline and automatically collects 30 new frames;
press `b` again while unloaded for the `led-top10` or `dense-highpass` observer.
For LED mode, `b` collects 30 feature vectors and their temporal median/MAD.
For dense high-pass mode it collects 30 canonical RGB frames and their median,
so subtraction precedes spatial high-pass filtering. Pressing `b` again during
collection restarts the acquisition. `r` explicitly starts geometry
recalibration without changing camera controls, and `q`, Escape,
or closing the window exits. The viewer does not save frames or estimates.
`LED_POSITIONS_IN_IMAGE_ORDER_MM` at the top of the script maps the detected
top-to-bottom image order to the physical fingertip Y axis; reverse it when the
camera is mounted from the opposite direction. A frame timeout triggers up to
ten explicit one-second reconnect attempts. A successful reconnect repeats the
camera warmup and clears the view-dependent baseline, so press `b`
again after LED recalibration.

This live command intentionally retains the D435's default automatic exposure
and white balance and is not the acquisition protocol for absolute
camera-intensity comparisons between morphologies. Such a quantitative
comparison must use one
explicit user-selected manual exposure, gain, and white balance for every
fingertip; it must not capture nominal manual values from a running auto mode.
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
