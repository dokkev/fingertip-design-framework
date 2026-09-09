# Commands

Run repository commands in the `lit` Conda environment.

## Install

```bash
conda run -n lit python -m pip install -e ".[mesh,physics,ax,visualization,data,test]"
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
conda run -n lit python -m compileall -q algorithm experiments lumo scripts validation tests
```

Run Ruff:

```bash
conda run -n lit ruff check algorithm experiments lumo scripts validation tests
```

## Focused unit tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  python -m pytest -q --import-mode=importlib tests/unit
```

Run only the calibration-free online-localizer tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  python -m pytest -q tests/unit/algorithm
```

## Calibration-free online contact localizer

The headless estimator under `algorithm/` does not own a camera or UI. Feed it
owned RGB8 frames from an acquisition layer:

```python
from algorithm import OnlineContactLocalizer

localizer = OnlineContactLocalizer()
localizer.initialize_geometry(first_unloaded_rgb)
localizer.acquire_unloaded_baseline(thirty_registered_unloaded_rgb_frames)

result = localizer.process(current_rgb)
if result.valid and result.contact_detected:
    print(result.position_mm)
else:
    print(result.status)
```

Before geometry initialization and throughout a run, disable automatic
exposure and automatic white balance and hold exposure, gain, and white balance
fixed. `position_mm` comes from the response centroid mapped through the five
detected physical LED anchors at 11 mm pitch; the estimator uses no hole labels
or position-labelled calibration data.

Replay the six LED-on observation conditions and audit the nominal-only global
actuator-torque calibration with:

```bash
conda run --no-capture-output -n lit \
  python validation/validate_proprioceptive_robustness.py
```

Outputs are written below
`output/validation/proprioceptive_robustness/`. The per-sample table preserves
`x_hat_mm`, `f_gt_n`, `f_hat_n`, and raw actuator torque. The JSON report marks
whether the result is eligible for Figure 6(e); a failed localization QC is a
negative validation result and must not be bypassed by plotting supplied
locations or by fitting per-run torque zeros. The same command writes a
condition-calibrated localization ablation. One condition is exactly one
0/20/40-mm triplet sharing camera extrinsics and illumination; the outputs
separate the three-run in-sample fit from leave-one-run/location-out evaluation.

Launch the NiceGUI torque/state dashboard in zero-torque monitor-only mode:

```bash
conda run --no-capture-output -n lit \
  python -u output/ak40_10_torque_gui.py \
    --channel can0 \
    --motor-id 13 \
    --max-abs-torque-nm 0.0
```

The dashboard opens at `http://127.0.0.1:8080`. Its clock hand follows measured
shaft position while the headless session sends zero-torque MIT commands and
reads feedback. To test nonzero torque, rerun with a positive
`--max-abs-torque-nm` value that has been approved for the mounted mechanism;
the program does not choose a nonzero safety limit. Closing or reconnecting the
browser, pressing `STOP / DISABLE`, losing feedback, receiving a drive error,
or shutting down the server invokes zero torque followed by disable.

Headless callers import the same session without NiceGUI:

```python
from experiments.actuation import AK40TorqueSession

session = AK40TorqueSession(
    motor,
    max_abs_torque_nm=operator_approved_limit_nm,
)
session.start()
try:
    session.set_target_torque(target_torque_nm)
    snapshot = session.snapshot()
finally:
    session.stop()
```

## Proprioceptive contact-dataset experiment

Launch the force-checkpoint NiceGUI collector. Its hardware defaults use
`can1`, decimal motor ID `13`, a 1920 x 1080 D435 stream, offline contact
processing, and output below `output/proprioceptive_contact_dataset/`:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/collect_proprioceptive_contact_dataset.py \
    --bota-port /dev/ttyUSB0 \
    --normal-axis fz \
    --normal-sign 1
```

Confirm the physical Rokubi axis/sign before using those two options. In the
page, explicitly set motor zero, apply approved Kp/Kd, and enable the motor.
Then click the 10 or 30 mm sphere and Hole 1--6 before `Start Run`. The page
guides measured force through 2, 5, 10, 15, and 20 N. After the 20 N checkpoint,
release the indenter: native motor, F/T, and optical streams continue until the
camera-synchronized force magnitude falls below 2 N, and then the run finishes
automatically. Lossless RGB is recorded at the configured 5 Hz default from
`Start Run` through this final release observation, not only inside checkpoint
bands. A large card beside the live camera shows the latest Rokubi force
magnitude as a vertical bar against the current target line and shaded
acceptance band, then switches to the below-2-N release threshold; the AK40-10 parameters are placed below that camera/tracker
row. The gauge is display-only. `Abort Run` discards the current incomplete run.
Online contact localization is disabled by default for this entry point and is not an
acquisition gate. This path does not copy rolling unloaded-reference frames by
default. Motor feedback and Rokubi samples remain
continuous from `Start Run` through automatic completion regardless of whether
an image is admitted.

Each completed run contains the ordinary `motor.csv`, `ft.csv`,
`camera_timestamps.csv`, camera frames, and metadata plus
`force_sequence.csv`. `motor.csv.timestamp_ns` and `torque_Nm` are the native
motor feedback record. `ft.csv.timestamp_ns` is the host monotonic force time.
`camera_timestamps.csv` stores host monotonic time, RealSense device time, frame
number, and the F/T timestamp paired to the image. `force_sequence.csv` carries
the same monotonic time basis together with target force, actual force, state,
and the latest motor timestamp/torque. Metadata records the fixture prior:
LED1 at 103.6 mm from the rotation axis, 11 mm LED pitch, Hole1 aligned with
LED1, 10 mm hole pitch, and distal LED1 through proximal LED5 ordering.

Online optical-conditioned force estimation is opt-in because its measured
calibration and contact thresholds are experiment-specific. Supply JSON with
fixed coefficients in this shape:

```json
{
  "region_locations_mm": [0, 10, 20, 30, 40],
  "slopes_n_per_nm": [0, 0, 0, 0, 0],
  "intercepts_n": [0, 0, 0, 0, 0]
}
```

The zero coefficients illustrate the file shape only; replace them with the
measured one-time calibration. Launch with explicitly selected motor-torque
hysteresis thresholds:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/collect_proprioceptive_contact_dataset.py \
    --force-calibration path/to/measured_force_calibration.json \
    --force-contact-enter-threshold-nm ENTER_NM \
    --force-contact-exit-threshold-nm EXIT_NM
```

Thresholds must satisfy `ENTER_NM > EXIT_NM >= 0`. The optical-to-motor offset
defaults to zero and maximum optical age defaults to 100 ms; configure them with
`--force-optical-offset-ms` and `--force-maximum-optical-age-ms` only from
measured timing evidence. After launch, keep the finger unloaded and press
`Recalibrate geometry`, then `Acquire unloaded baseline`, and finally `Set
unloaded torque bias`. None runs automatically. Geometry and baseline each use
30 frames; live optical response uses a causal three-frame median.

Package a completed run into one upload-sized HDF5 artifact:

```bash
conda run --no-capture-output -n lit \
  python -u output/compress/export_proprioceptive_h5.py \
    output/proprioceptive_contact_dataset/run_001
```

The default output is the sibling file `run_001.h5`. It preserves every saved
frame at full resolution as an independently encoded quality-95 JPEG byte
stream, the typed camera index, and byte-exact copies of the run JSON/CSV files.
It does not crop, resize, subtract a reference, or normalize intensity. JPEG is
the only lossy step. The exporter publishes no artifact if the completed file is
500 decimal MB or larger and never deletes the source PNG directory. Use an
explicit `--jpeg-quality` or lower acquisition duration/rate if the limit is
exceeded.

Verify an existing artifact with
`output/compress/export_proprioceptive_h5.py RUN.h5 --verify`.

Package every run in a proprioceptive robustness dataset into one HDF5 file:

```bash
conda run --no-capture-output -n lit \
  python -u output/compress/export_proprioceptive_h5.py \
    output/proprioceptive_robust_dataset \
    --dataset \
    --output output/upload/proprioceptive_robust_dataset.h5
```

Dataset mode retains the same full-resolution quality-95 JPEG representation
and exact per-run JSON/CSV payloads, but groups all source runs under `runs/`.

Compare segmentation-free contact-location cues on a completed proprioceptive
run without training contact-location templates or acquiring a separate
calibration recording:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/segmentation_free_contact_localization.py \
    --run output/proprioceptive_contact_dataset/run_001
```

This read-only ablation compares direct contactor edges, a visible marker,
green optical change, and temporal image change. The `auto_reference` variants
use unloaded observations already contained in the same run; the frame-only
variants require no unloaded reference. `run_001` has no contact-location ground
truth, so its reported rank agreement uses the synchronized motor-torque/F/T
moment-arm estimate only as an evaluation proxy, not as millimetre ground truth.
CSV and PNG/PDF diagnostics are written beneath
`output/validation/segmentation_free_contact_localization/run_001/`.

Runs are stored by default under:

```text
output/experiments/proprioceptive_force/
└── run_001/
    ├── metadata.json
    ├── motor.csv
    ├── ft.csv
    ├── optical.csv
    ├── force_estimate.csv
    ├── camera_timestamps.csv
    └── camera/
        ├── frame_000000.png
        └── ...
```

All streams carry host monotonic nanosecond timestamps and are saved
independently at their configured rates. Source camera images are lossless PNGs.
`metadata.json` records the fixed impedance command, Kp/Kd, optional trial
contact-location ground truth, device information, normal-axis convention, and
sample counts. It contains no morphology, material, or Git metadata.

Minimal first run:

1. Configure SocketCAN and connect the D435 and Rokubi externally.
2. Launch the console and verify the motor, camera, and F/T status.
3. Put the finger in its nominal pose and press `Set Motor Zero Position`
   explicitly.
4. Apply approved Kp/Kd, then press `Enable`.
5. In `both`/`online` mode, acquire an unloaded online optical baseline. In
   `offline` mode, leave the fingertip unloaded before starting the run so the
   rolling reference is populated.
6. Enter the trial metadata and press `Start Recording`.
7. Apply load manually with the F/T-equipped indenter.
8. Press `Stop Recording`, verify the saved path/counts, then press `Disable`.

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

## Physical contact-history dataset collection

Use the separate contact-history collector when the scientific question is
history-dependent optical response during continuous cyclic loading. Launch it
with the production D435 and Rokubi defaults:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/collect_contact_history.py \
    --bota-port /dev/ttyUSB0
```

One run establishes one physical contact near 2 N, then keeps that contact
engaged while cycling continuously between 2 and 15 N. The default loading and
unloading ramps are 11.375 N/s, with a 1 s dwell at 15 N and a 1 s dwell at
2 N.
The first two cycles are labeled `conditioning`; the following five are labeled
`measurement`, but both roles are saved. Do not release the indenter between
cycles. After the seventh cycle, the run is finalized after force remains at or
below 2 N for 0.5 s. A series creates its next independent repetition
only after that release gate.

The moving target is operator guidance, not force control or an acceptance
band. The trajectory advances only from monotonic elapsed time and never resets
when measured force leads or lags the target. Every scheduled synchronized
observation records the actual Rokubi wrench, actual force magnitude, target,
explicit loading/dwell/unloading phase, cycle role, and tracking error. Later
analysis must compare branches at matched actual measured force. Independently
of the 5 Hz saved images, every camera-rate force observation during `CYCLING`
updates a contact-continuity diagnostic. `trajectory.json` records the minimum
force seen, the number of distinct excursions at or below the default 1 N
contact-loss threshold, and a warning boolean. These diagnostics never reset or
reject acquisition. A camera-frame drop remains separately reported because a
loss occurring entirely inside that observation gap cannot be detected.

The stored `phase` is the nominal elapsed-time branch. It is deliberately not
rewritten from the measured force. Offline history analysis must therefore also
check actual-force monotonicity, handle turnaround neighborhoods, and match
loading and unloading observations by actual force.

The camera remains 1920 x 1080 at 30 FPS with fixed 1500 µs exposure, gain 0,
and 4600 K white balance. Lossless RGB observations are selected at 5 Hz by
elapsed time. At the defaults, each ramp lasts 8/7 s, each cycle lasts 30/7 s,
and the seven-cycle trajectory lasts exactly 30 s. This produces 150 scheduled
trajectory observations when none are missed. The preload and final release
each add at least 0.5 s; operator waiting time is additional.

History format v1 is isolated under `output/contact_history/`:

```text
YYYY-MM-DD_<material>_<morphology>[_NN]/
├── session.json
├── unloaded/
│   └── capture_001/
│       ├── frames.csv
│       └── frames/
└── runs/
    └── run_0001/
        ├── run.json
        └── trajectory/
            ├── trajectory.json
            ├── frames.csv
            └── frames/
```

`CAPTURE UNLOADED` stores independent diagnostic captures; it does not pair one
with a loaded run. A bounded asynchronous PNG writer reports individual missed
observations without discarding the otherwise valid continuous trajectory.
`ABORT` deletes the current partial run, cancels the remaining series, and
preserves completed repetitions.

Exercise the whole GUI without camera or Rokubi hardware using the static-image
mock camera and manual force slider:

```bash
conda run --no-capture-output -n lit \
  python -u scripts/collect_contact_history.py --mock
```

Mock sessions are written only under `output/contact_history/mock/` and are
marked as non-physical data in both the GUI and `session.json`.

## Compact physical-data export

Create the two upload artifacts from the six canonical discrete-contact
sessions and the six final contact-history sessions:

```bash
conda run --no-capture-output -n lit \
  python -u output/compress/export_compact_physical_data.py
```

The command writes `output/upload/contact_dataset.h5`,
`output/upload/contact_history.h5`, and a format README. It retains every
observation separately as an absolute 128 x 64 RGB8 canonical map, the exact
128-bin Green profile evaluated on the full 256 x 128 strip, typed force/time/
run metadata, and a compressed copy of every source JSON/CSV file. Unloaded
captures remain separate. Loaded maps use the temporally nearest unloaded
capture from the same session for geometry only; no unloaded intensity is
subtracted and no response is normalized.

The HDF5 files deliberately omit full 1920 x 1080 pixels and scene content
outside the calibrated fingertip strip. The original PNG sessions remain the
authoritative archive for full-frame perception or mechanical image tracking.
The exporter refuses to overwrite an existing artifact and writes through a
`.partial` file before atomically publishing each completed HDF5 file.

Verify existing artifacts without reopening the PNG datasets:

```bash
conda run --no-capture-output -n lit \
  python -u output/compress/export_compact_physical_data.py --verify \
    output/upload/contact_dataset.h5 \
    output/upload/contact_history.h5
```

Analyze three same-material history sessions using actual-force branch matching
(Dragon Skin example):

```bash
conda run --no-capture-output -n lit \
  python -u output/analyze_contact_history.py \
    output/contact_history/2026-09-06_dragon_skin_baseline \
    output/contact_history/2026-09-06_dragon_skin_flat_opt \
    output/contact_history/2026-09-06_dragon_skin_angled_opt \
    --repeat-metrics \
      output/analysis/dragon_skin_morphology_comparison/results/morphology_metrics.csv \
    --output output/analysis/dragon_skin_contact_history
```

The analysis uses only indenter/contact-location conditions present in all
three sessions and excludes contact-loss or incomplete runs from primary
metrics. Loading and unloading profiles are interpolated only inside their
measured actual-force overlap on a 3--14 N grid; no extrapolation is performed.
The report records the common force interval with at least 50% measurement-cycle
coverage in every morphology. Scalar `H`, `S_span`, and `H_rel` values are
reduced across eligible cycles within each run before morphology medians and
quartiles are computed, so independent contact runs remain the experimental
units. Independent-contact `W_repeat` and same-contact `W_cycle` have different
physical units, so their comparison is made only after normalizing each metric
to its own material-specific baseline.

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
  python -u output/analyze_morphologies.py \
  output/contact_dataset/Solaris-baseline \
  output/contact_dataset/Solaris-flat-opt \
  --output output/analysis/solaris_compare
```

Each invocation reads the raw PNGs once and writes `results/`, `figures/`, and
`raw_data_summary/`, plus `raw_data_summary.zip` for later upload. The fixed
interior strip is calibrated once per specimen. Frames within one hold are
aggregated by median, and the primary profile slope uses actual measured force.
`--expected-repetitions` changes only coverage QC. `--hole-spacing-mm` adds a
physical-spacing-normalized neighboring-location diagnostic when that spacing
is trusted. The summary contains no PNGs and no mechanical-deformation claim.
It preserves every unloaded capture separately in `unloaded_summary.csv`,
`unloaded_profiles.npz`, and compact `unloaded_maps.npz`; it does not infer a
loaded-run pairing or choose a preferred unloaded reference.
`suspect_runs.csv` is a deterministic manual-inspection ranking and never
repairs or relabels the dataset.

Run the staged rigid-body indentation-tracking feasibility study on the stored
Solaris Baseline 10 mm-sphere data. Inspect the first command's overlay before
running the sample. Run the full stage only if the sample's individual-frame,
fixture-drift, transverse-motion, monotonicity, and repetition evidence is
mechanically consistent:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/hardware_indentation_tracking.py --stage manual
conda run --no-capture-output -n lit \
  python -u validation/optomech/hardware_indentation_tracking.py --stage sample
conda run --no-capture-output -n lit \
  python -u validation/optomech/hardware_indentation_tracking.py --stage full
```

The two fixed manual ROIs and the corrected 0--50 mm fixture-stop mapping are
stored in `validation/optomech/hardware_indentation_tracking_config.json`.
Outputs are written beneath
`output/validation/hardware_indentation_tracking/`. This study reports pixels,
does not infer an image-to-mm scale, does not populate `S_OM`, and does not
modify Figure 5.

Run the separate vertically mean-reduced signed-profile 1-D NCC study on the
exact same fixed 12-run sample. This command has no full-session mode, never
calls the older 2-D tracking entry point, and retains the previous
vertical-median output for the controlled A/B comparison:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/hardware_indentation_tracking_1d.py
```

Its Markdown, hold/frame tables, complete NCC lag curves,
direct-versus-sequential diagnostic, synthetic translation check, and three
figures are written under
`output/validation/hardware_indentation_tracking_1d_mean/`. The result is an
image-space rigid-shaft indentation proxy in pixels only; it does not modify
the preceding 2-D output, Figure 5, or any production metric.

Run the separate direct rigid-edge geometry study on the same exact 12-run
sample. It uses row-wise signed Scharr-x peaks and robust left/right shaft
lines plus one fixed fixture edge; it does not use phase correlation or NCC:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/hardware_indentation_tracking_edges.py
```

The hold/frame geometry tables, synthetic whole-ROI translation check,
reference definitions, Markdown conclusion, and four diagnostic figures are
written under `output/validation/hardware_indentation_tracking_edges/`. The
study is pixel-domain and read-only, stops at the fixed sample, and does not
modify `S_OM` or Figure 5.

Run the unloaded-referenced optical-activation feasibility study on all five
available Figure 5 specimen sessions:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/hardware_unloaded_optical_activation.py
```

The script re-extracts every unloaded and loaded frame through one
session-global production optical strip, pairs each complete run to the nearest
unloaded capture in camera host time, and writes run, morphology, pairing,
reference-sensitivity, unloaded-stability, compact-profile, Markdown, and PNG
outputs beneath `output/validation/hardware_unloaded_optical_activation/`.
The reported RMS activation has camera-DN units and is not force-normalized.
This validation neither modifies Figure 5 nor registers a production metric.

Run the shared contact-location decoder and perception-workload analysis from
the existing compact profile artifacts:

```bash
conda run --no-capture-output -n lit \
  python -m experiments.analysis.fig5c_decoder \
  --config experiments/analysis/configs/paper_figures.yaml
```

This writes `fig5c_accuracy_summary.csv`, per-sample predictions, condition
confusion matrices, legacy exploratory Figure 6 support tables, and
`fig56_summary_bundle.csv` under `output/analysis/paper_figures/`. The primary
decoder uses signed 5 N-minus-2 N changes in six longitudinal regions and
leave-one-repetition-out nearest templates. The Figure 6(b) table uses
independent re-contact `W_contact`, never cyclic `W_cycle`.
`fig6a_magnitude_vs_accuracy.csv` contains only the eight optimized morphology
comparisons and reports baseline-relative magnitude change [%] and accuracy
change [percentage points].
`fig6c_optional_calibration_10mm_combined.csv` contains only the 10 mm sphere:
`k=0` is the geometry-prior estimator with no labelled contact calibration,
while `k=1..4` retain the calibrated six-region held-out-repetition protocol.
Those two legacy Figure 6 tables are not inputs to the corrected Figure 6(a)
or (c); use the dedicated corrected a--c command below.

Render the final IEEE double-column Figure 5 PDF/PNG from the current physical
datasets and decoder summary:

```bash
conda run --no-capture-output -n lit \
  python -m figures.fig5.fig5
```

Render the shared six-row table-layout candidate without replacing `fig.*`:

```bash
conda run --no-capture-output -n lit \
  python -m figures.fig5.fig5 --candidate
```

This writes `fig5_tablelayout_candidate.pdf/png` beside the final outputs.

Outputs are written under `figures/fig5/` as `fig.pdf` and `fig.png`; the
auditable panel-A and panel-B tables remain
`fig5a_selection_manifest.csv` and `fig5b_region_response.csv`. The raw atlas uses
the 10 mm sphere, repetition 1, and the frame closest to 15 N at three
representative physical positions: 0, 20, and 40 mm (the separate LED pitch
remains 11 mm). The 7.16 x 4.35 inch composition places panels (a), (b), and
(c) in one horizontal row. All three panels use the same six experimental rows:
Solaris Baseline/Flat-Opt/Curved-Opt above Dragon Skin
Baseline/Flat-Opt/Curved-Opt. Figure 5 retains these full material names while
the other manuscript figures abbreviate them as `Sol.` and `DS.`. Each panel's
header, data, and shared x-axis-title
area uses one thin neutral bounding box without internal table rules. Panel (a)
alone owns the figure-wide rotated material labels, morphology labels, their
paper-color identity bars, and the small separating spacer; these labels remain
outside panel (a)'s box. Panels (b) and (c)
inherit these row identities
through the shared row geometry and do not repeat them. Panel (a) shows its
four image-column headers once and uses one wider context-preserving
crop whose rotated aspect fills the shared data-row height. Panel (b) displays
all six contact positions by six longitudinal regions as rectangular median
RMS optical-change heatmaps. Its two columns are the 10 and 30 mm spheres; all
12 maps share one global Viridis scale. The same scale is shown by separate
vertical colorbars aligned with the Solaris and Dragon Skin row blocks at the
left of panel (b).
The maps place contact location on the x-axis and longitudinal region on the
y-axis, showing only 0/20/40 mm and R1/R3/R6 tick labels. They omit cell
annotations and peak overlays. All
Solaris atlas cells use one documented +0.275 EV display exposure, and all
Dragon Skin atlas cells use one documented +0.525 EV display exposure.
No per-cell normalization is applied. The six raw-image sources use the
canonical morphology directories directly under `output/contact_dataset/`.
The Dragon Skin baseline and angled-opt canonical directories combine the
selected 10 mm acquisition with the latest 30 mm repeat acquisition while
preserving every unloaded capture independently. Existing compact analysis
overrides retain the matching 30 mm repeat-session calibration. Figure 5(c)
groups 12 row-normalized 6-by-6 confusion matrices into the same six
material/morphology rows and 10/30 mm sphere columns as panel (b). The shared row-height
grammar aligns every matrix with its corresponding specimen and optical-change
profile. All matrices share one 0--100% color scale. The command reuses the
decoder's existing per-sample predictions and preserves its original raw
confusion tables.
`D_neighbor / W_contact` remains exclusive to Figure 6(b).

Recompute and render only the corrected Figure 6(a--c) analysis without
loading, rebuilding, or rendering panels (d) and (e):

```bash
conda run --no-capture-output -n lit \
  python figures/fig6/fig6abc.py \
  --config experiments/analysis/configs/paper_figures.yaml \
  --recompute
```

The versioned machine-readable artifacts are written below
`output/analysis/paper_figures/fig6c_corrected_5n_led_registered_response_v2/`; the
manuscript-scale review is `figures/fig6/fig6abc_review.pdf/png`. Panel (a) is
explicitly 10 mm-only because the maintained-contact sessions contain no 30 mm
observations. Its exact absolute plotted values are written to
`fig6a_contact_state_variability.csv`. Panel (b) retains the aggregate slope-profile
`Q_recontact = D_neighbor / W_recontact`. The directory retains the common observation NPZ, anchor table and overlays,
split manifest, per-contact predictions, `fig6c_summary.csv`,
`fig6c_provenance.json`, and `fig6c_analysis_notes.md`.

Panel (c) no longer reads `fig6c_summary.csv`, whose calibrated regime holds
every contact location fixed and therefore cannot separate the morphologies.
It plots the checked-in table
`figures/fig6/fig6c_uncalibrated_location_mae.csv`, which reports MAE at
contact locations absent from calibration. Repetitions split once into
calibration repetitions {1, 2} and evaluation repetitions {3, 4, 5}.
Calibration effort `K` is the number of distinct contact locations set up on
the rig; all `C(6, K)` location subsets are enumerated and averaged, and each
subset is evaluated only on the locations it left out. `K = 6` is therefore
undefined, and `K = 1` cannot support the affine coordinate-to-millimetre fit,
so both remain `unavailable`. `K = 0` is the LED geometry prior alone and
resolves only for Solaris. Every specimen and `K` cell carries an explicit
`status` and `failure_reason`; the empty `mae_q25_mm`/`mae_q75_mm` columns
suppress the interquartile bands until subset quartiles are supplied. Because
this table is entered by hand rather than emitted by
`experiments.analysis.fig6abc`, `tests/unit/visualization/test_fig6c_panel.py`
stands in for the provenance the generated artifacts carry.

Render the full Figure 6, including its existing panel (d), or both final
figures:

```bash
conda run --no-capture-output -n lit \
  python -m figures.fig6.fig6 \
  --config experiments/analysis/configs/paper_figures.yaml
conda run --no-capture-output -n lit \
  python -m experiments.analysis.build_fig5_fig6 \
  --config experiments/analysis/configs/paper_figures.yaml
```

The full Figure 6 command writes the canonical exact-7.16-inch double-column
`fig6.pdf/png` and the standalone single-column
`fig6c_uncalibrated_location_mae.pdf/png`. It also writes
`fig6d_force_timeseries_multilocation.pdf/png/csv` from proprioceptive dataset
runs 005--010. The composition is a 2-by-2 grid: contact-state variability and
re-contact distinguishability on the upper row, transfer to uncalibrated
locations and the six-location force time series on the lower row. Figure 6(c)
splits into `Sol.` and `DS.` subplots that share one MAE axis, marks the
`K = 0` geometry prior with an X, separates that regime from the calibrated
one with a vertical rule at `K = 1`, and draws a dashed horizontal reference at
the best prior-only result, `Sol. Curved-Opt` at 1.91 mm. Dragon Skin carries
no `K = 0` mark and is annotated `No prior available`. Figure 6(d)
uses the camera-synchronized
Rokubi force-vector magnitude, per-run unloaded torque zeroing, the prescribed
`r(x)=(103.6-x)/1000` m geometry, and one global scale fitted only on run 005.
The panel labels the represented specimen condition as `Sol. Flat-Opt`. A
muted secondary right axis shows the synchronized raw motor torque in N m for
context; it is not a separate fitted force result. Because that secondary
spine needs horizontal room, panel (d) draws into axes narrower than its grid
cell and centers its own label and title on the drawn axes.

Figure 6 writes `fig6.pdf/png`. Use the dedicated a--c command when panel (d)
must remain untouched. Add `--recompute` to the full plotting/build command
only when rebuilding the complete composition is intended.

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
  python -u scripts/live_contact_localization.py --view boundary
```

The boundary view shows RGB, the coarse paired-LSD prior, raw selected
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
  python -u scripts/live_contact_localization.py --view contact
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
  python -m figures.fig2.fig2
```

The script writes `fig2.pdf`, `fig2.svg`, and `fig2.png` beside its source in
`figures/fig2/` and does not rerun Newton.

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
  python -m figures.fig3.fig3
```

The primary study writes its catalog, manifest, per-variant raw NPZ states,
paired CSV, JSON summary, and report beneath
`output/validation/multi_design_void_ablation/`. The finalized composition
source and its `fig3.pdf`/`fig3.svg`/`fig3.png` exports live together beneath
`figures/fig3/`. `figures/fig3/fig3.py` also reads the four completed
160-observation BO
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
