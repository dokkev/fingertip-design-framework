# Commands

Use the `lit` Conda environment for all repository commands:

```bash
conda activate lit
```

## Install current dependencies

```bash
python -m pip install -e ".[mesh,physics,ax,test]"
```

`mesh` supplies Gmsh, `physics` supplies Newton 1.5/Warp and rigid asset loading,
and `ax` supplies Ax 1.3.1. CUDA, OptiX, and GPU drivers are externally managed.
The editable install exposes the sole framework namespace from `lumo/`;
repository scripts do not insert the checkout into `sys.path`.

Install the NVIDIA OptiX Toolkit source once for its header-only ShaderUtil
self-intersection implementation. No OTK build is required:

```bash
git clone --depth 1 \
  https://github.com/NVIDIA/optix-toolkit.git \
  /path/to/optix-toolkit

conda env config vars set -n lit \
  OTK_INCLUDE_DIR=/path/to/optix-toolkit/ShaderUtil/include
```

Reactivate the environment after changing its persistent variables. The
runtime also accepts an explicit `otk_include_dir` when constructing
`OptixScene`.

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

Run the focused full five-LED mesh regressions:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q tests/unit/mesh/test_fingertip_5led_mesh.py
```

Generate the longitudinal, local XZ, and 3D material views of the actual full
mesh:

```bash
conda run --no-capture-output -n lit \
  python -u validation/fingertip/view_fingertip_5led.py
```

The figure is saved to `output/validation/fingertip_5led_mesh.png`.

Open the same full five-LED mesh as a static Newton model in ViewerGL. The
green markers are the five LED reference positions below the continuous stem
rail:

```bash
conda run --no-capture-output -n lit \
  python -u validation/fingertip/view_fingertip_5led_newton.py
```

Run the short central 10 mm sphere Newton compatibility smoke without the full
force protocol:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/fingertip_5led_smoke.py
```

Run the full five-LED Newton mechanics validation at the center LED, the
`Y=+5.5 mm` midpoint, and the distal LED:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/fingertip_5led_mechanics.py
```

The procedural validation writes its report to
`output/validation/5led_newton_mechanics_validation.md` and raw deformation,
contact, profile, runtime, and visualization artifacts under
`output/validation/5led_newton/`.

Validate the full five-LED OptiX field using the saved 10 N Newton states,
without rerunning mechanics:

```bash
conda run --no-capture-output -n lit \
  python -u validation/ray-tracing/fingertip_5led_optix.py
```

The validation uses five simultaneous unit-power LED sources with 65,536
deterministic paths per emitter, saves raw escaped paths and per-emitter
responses under `output/validation/5led_optix/`, and writes the conclusions to
`output/validation/5led_optix_validation.md`. It also verifies the nominal
five 0.19 mm LED air cavities and reports which loaded states close each
cavity.

Diagnose LED-interface sensitivity using the saved full-finger Newton states:

```bash
conda run --no-capture-output -n lit \
  python -u validation/ray-tracing/led_silicone_interface.py
```

This optical-only diagnostic reuses the saved Newton states and writes the report to
`output/validation/5led_led_silicone_interface_diagnostic.md` and its signed
geometry gaps, medium traces, epsilon/gap sweeps, treatment comparisons, and
figures under `output/validation/5led_led_silicone_interface/`. Its controlled
gap sweep is measured from the fixed LED top and includes the nominal 190 µm
hardware cavity. The production geometry contract and primary cavity-closure
result are also reported by `fingertip_5led_optix.py`.

Compare the six selected Dragon Skin/Solaris physical-validation morphologies
in one common-scale 2x3 XZ cross-section figure:

```bash
conda run --no-capture-output -n lit \
  python validation/fingertip/view_physical_validation_morphologies.py
```

The figure is also saved to
`output/validation/physical_validation_morphologies.png`.

Run the focused cross-material validation for the five selected morphologies:

```bash
OPTIX_INCLUDE_DIR=/home/dk/workspace/optix-dev/include \
OTK_INCLUDE_DIR=/home/dk/workspace/optix-toolkit/ShaderUtil/include \
conda run --no-capture-output -n lit \
  python -u validation/optomech/cross_material_morphology_validation.py
```

The script reuses six existing Dragon Skin/Solaris raw evaluations, runs only
the four missing optical cross-evaluations, and does not modify either Ax
campaign. It writes the raw cross-results, comparison CSV/report, and three
plots to `output/optimization/cross_material_validation/`.

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

The 5, 10, and 20 mm diameter URDF spheres each run in independent simulations
at `X=-7.5`, `0`, and `+7.5 mm`, for nine design trials total. Each kinematic
indenter uses the proportional force servo, slows near `20 N`, and must remain
inside `20 ± 1 N` for `5 ms`. This is an explicit multi-simulation validation,
not part of ordinary focused tests.

View one centered 15 mm sphere moving continuously to a transient `20 N`
reaction force:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/sphere_15mm_viewer.py
```

The viewer renders every Newton tick and prints travel, reaction force, maximum
active silicone speed, and sphere contact count every 100 ticks. After the first
transient `20 N` crossing it holds the sphere pose fixed while continuing the
simulation for `10 s`, then freezes the final held state until the window
closes. This is an interactive contact diagnostic, not the force-servo
validation above.

Record the fixed-pose reaction-force trajectory after the same centered 15 mm
sphere first reaches transient `20 N`:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/force_traj.py
```

The script compares the baseline, a validation-only one-time particle-velocity
reset, a 30-iteration solve, and a smaller-timestep solve. Each case holds the
trigger pose for `10 s`, prints early transient checkpoints through `1 s` plus
`2`, `5`, and `10 s`, and writes combined force, speed, contact, penetration,
and tetrahedral-volume trajectories to `output/validation/force_traj.csv` and
`force_traj.png`. It is an explicit GPU validation and is not part of focused
tests.

Measure the nominal centered 15 mm sphere force-depth curve and direct sphere
penetration diagnostics without holding or force correction:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/sphere_force_depth.py
```

The script continuously pushes to `10 mm` analytic indentation depth and writes
`output/validation/sphere_force_depth.csv` and `sphere_force_depth.png`. It also
reports sphere contact count, particle/surface/centroid/tet-center penetration,
minimum tet `det(F)`, and inverted-tet count. A fresh second simulation reaches
the first transient `20 N` crossing, holds that exact pose for `1 s`, and
reports force at `0`, `5`, `100`, and `1000 ms`.

Run the focused centered-sphere rigid-soft contact diagnostic:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/sphere_contact_tuning.py
```

The script uses Newton 1.5's full-surface VBD wrench harvest, equalizes the
rigid-shape and soft-contact `ke/kd` endpoints, and compares `ke=1e4`, `3e4`,
and `1e5 N/m` with mass-scaled damping at `2 kHz`. It checks two stable cases
at `4 kHz` and writes `output/validation/sphere_contact_tuning.csv`,
`sphere_contact_tuning_force_depth.csv`, and `sphere_contact_tuning.png`.
This explicit GPU diagnostic does not run the larger mechanics or OptiX sweeps.

Run the representative numerical/contact parameter sweep explicitly:

```bash
conda run --no-capture-output -n lit \
  python -u -m lumo.benchmark.newton_parameter_sweep
```

Add the substantially finer `0.5 mm` mesh case with `--fine`. Add the current
baseline 3-sphere by 3-location robustness matrix with `--matrix`. Parameters
are varied one family at a time around the current baseline, with the approach
speed fixed at `25 mm/s` even when simulation frequency changes; neither flag
creates a Cartesian product across numerical parameters. This is an expensive
multi-simulation convergence study and is not part of ordinary focused tests.
After every requested run finishes, it writes strict JSON to
`output/benchmark/newton_parameter_sweep.json`. Use `--output PATH` to select a
different result file. An interrupted run does not write a partial result.

Run the full measured optomechanical Newton/OptiX factorial sweep explicitly:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/newton_parameter_sweep.py
```

This runs the fixed 24-configuration Newton sweep at `65,536` rays and `24`
bounces, measures mechanics and optical wall time, and writes strict JSON to
`output/validation/newton_parameter_sweep.json`. It selects the fastest
hard-valid configuration that completes the fixed sensing evaluation.

Run the complete sensing numerical-convergence study overnight:

```bash
conda run --no-capture-output -n lit \
  python -u validation/contact-physics/sensing_convergence.py
```

This one script first compares `5/20/50 ms` fixed-pose settling holds, then runs
the small one-factor-at-a-time Newton/contact study for carrier stiffness,
timestep frequency, VBD iterations, and mesh size. It evaluates the selected
hard-valid deformation with common optical samples at 16384 and 65536 rays,
using 24 bounces and three fixed seeds. It prints progress and compact final
tables, including raw per-contact quadrant responses and worst-pair diagnostics
in the JSON output, then writes strict JSON to
`output/validation/sensing_convergence.json`. Use `--output PATH` to select a
different final JSON file. This is an expensive unattended validation and is
not part of ordinary focused tests. A failed Newton setting is recorded and the
remaining settings continue; optical comparisons run from the first hard-valid
reference configuration.

Run the Dragon Skin 10 NV Poisson-ratio contact sweep explicitly:

```bash
conda run -n lit python validation/contact-physics/poisson_ratio_sweep.py
```

The sweep compares `0.48`, `0.49`, and `0.495` using one fixed shear modulus
and reports force-target timing and tetrahedral volume change. It performs
multiple Newton simulations and is not part of ordinary focused tests.

## OptiX gate before production BO

Run the static multi-instance IAS validation with the OptiX 9.1 SDK include
directory used by NVRTC:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/ias_test.py
```

This builds silicone and carrier GASes under one IAS and checks closest hits, a
miss, and visibility masking. Synchronization occurs only when the batched
results are copied to the host.

Run the CPU-only single-interface dielectric validation:

```bash
conda run --no-capture-output -n lit \
python validation/ray-tracing/interface_transport_test.py
```

This checks deterministic normal-incidence and oblique Fresnel/Snell cases,
including below-critical refraction, total internal reflection, and scalar or
per-ray power conservation.

Run the OptiX world-space geometric-normal and interface integration check:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/normal_test.py
```

This checks a planar carrier face, the analytic silicone semiellipse, and one
OptiX-hit-to-dielectric-interface operation.

Run the single refracted secondary-ray validation:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/secondary_ray_test.py
```

This uses OTK ShaderUtil safe spawn positions to trace exactly one refracted
ray from exposed silicone to carrier without a numerical self-hit.

Run the single-interface reflected/refracted power-branch validation:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/power_branch_test.py
```

This splits one incident ray's scalar power by Fresnel `R/T`, traces each
OTK-safe branch once, and stops after the reflected miss and refracted carrier
hit.

Run the one-event opaque Lambertian carrier validation:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/carrier_reflection_test.py
```

This checks deterministic cosine-weighted sampling and opaque power accounting,
then traces one `air -> silicone -> carrier -> silicone` path and stops.

Run the complete single-path silicone-exit validation:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/silicone_exit_test.py
```

This composes the validated surface operations into one deterministic
`air -> silicone -> carrier -> silicone -> air` path, verifies that the final
transmitted ray misses the scene, and checks complete scalar power accounting.

Run the idealized undeformed/deformed LED-receiver study:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/led_sensor_response_test.py
```

This uses the current Adafruit Green LED Sequin hardware metadata with an ideal
Lambertian point-source approximation, runs one central 10 mm sphere indentation
to the existing settled-force checkpoint, and evaluates a fixed 24-bounce
transport cap for the low/nominal/high Solaris and Dragon Skin 10 NV optical
sensitivity presets. All six cases use the same 4096 emitted rays, deformation,
and per-ray/per-bounce random samples before and after the silicone UPDATE.
Source placement, receiver geometry, carrier albedo, normalized optical power,
and literature-derived extinction priors remain uncalibrated validation inputs.
This is an explicit Newton/OptiX study rather than part of the focused unit-test
suite.

Run the complete 3-sphere by 3-location side-view sensing matrix:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/sensing_evaluator_test.py
```

This runs the production evaluator for 5, 10, and 20 mm spheres at
`X=-7.5, 0, +7.5 mm`. It builds one fingertip mesh and OptiX scene, traces the
no-contact state once with `65,536` paths, and runs nine independent sequential
Newton scenarios with the current `100 Hz / 10 iteration` force servo. Each
scenario advances through `5, 10, 15, 20 N` in one runtime and accepts a level
after remaining within its `+/- 10%` band for 5 s. Every checkpoint updates the
existing silicone GAS/IAS and repeats the 24-bounce trace with the same
deterministic samples, then discards full path arrays. The script prints the
nine-by-four-by-four response matrix, deltas, compact energy ledgers, actual
forces and indentations, and per-scenario and total wall runtimes.

Save the one-cell before/after sensing diagnostic without opening a window:

```bash
MPLBACKEND=Agg conda run --no-capture-output -n lit \
python validation/ray-tracing/sensing_visualization.py \
  --output /tmp/lumo_sensing.png
```

The two matched X-Z panels are projections of the existing full 3D bounded
paths at the geometry-derived extrusion center, not a separate 2D ray tracer.
The loaded panel uses the centered 15 mm sphere with the current
`500 Hz / 10 iteration` contact configuration and the accepted `20 +/- 1 N`
`DesignStudy` checkpoint after a 5 s continuous force-band hold. The plot
reuses the same 4096 deterministic diagnostic paths and 24-bounce cap in both
states.

Run the CPU-only sampled dielectric branch regression:

```bash
conda run --no-capture-output -n lit \
python validation/ray-tracing/dielectric_branch_test.py
```

This checks Fresnel branch selection, total internal reflection, medium-state
updates, and the non-double-counted lossless path weight used by bounded
transport.

Run the silicone GAS and IAS UPDATE/refit validation with the same SDK headers:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/refit_test.py
```

This translates the silicone surface by `+1 mm`, compares UPDATE against a
fresh scene build, checks that the other instances remain unchanged, and
reports representative update and full-construction timings.

Run the real Newton-state to OptiX checkpoint validation:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
conda run --no-capture-output -n lit \
python validation/ray-tracing/newton_refit_test.py
```

This drives the centered 15 mm kinematic sphere at the same `25 mm/s` used by
the interactive viewer, freezes the first transient state at or above `20 N`,
and compares an in-place silicone UPDATE against a fresh OptiX scene build.

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

## Full-finger raw evaluator

Validate the new production Newton-to-OptiX path on one nominal morphology:

```bash
OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-9.1.0/include \
OTK_INCLUDE_DIR=/path/to/optix-toolkit/ShaderUtil/include \
conda run --no-capture-output -n lit \
  python -u validation/optomech/full_finger_raw_evaluator.py
```

The procedural validation builds the full 60 mm mesh once, traces all five LEDs
once without contact, and evaluates one 15 mm sphere at `Y=0`, `+5.5`, and
`+22 mm` with sequential `5`, `10`, `15`, and `20 N` checkpoints under the
production `5 s` dwell and `+/- 10%` force band. It stores per-emitter
5x11 longitudinal-response/energy data, deformation snapshots, raw contacts, and mechanics
diagnostics in
`output/validation/full_finger_raw_evaluator/nominal_full_finger_raw.npz`.
It then reloads that NPZ without Newton or OptiX and reconstructs the combined
11-bin response, contact centroids, minimum `det(F)`, inversion counts, and
energy closure.

Reproject the existing pre-`+X` nominal artifact without rerunning Newton:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/full_finger_spatial_observation.py
```

This runs only the fixed optical trace and writes
`output/validation/full_finger_spatial_observation/spatial_response.npz`.

Compute the read-only `J_contact` and `J_obs` candidates and component plots
from that saved artifact:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/objective_prototype.py
```

The script writes contact components, diagnostic contact-onset distances, all
same-force location distances, force-trajectory diagnostics, summary CSV/NPZ data, plots,
and `j_obs_force_conditioned_report.md` under
`output/validation/full_finger_objective_prototype/`. It performs no
Newton/OptiX run and does not register either candidate objective with Ax.

Run the production objective, LED-permutation, ROI-accounting, five-dimensional
schema, and serialized-contract checks with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q \
    tests/unit/optimization/test_full_finger_objective.py \
    tests/unit/optimization/test_height_constraint.py
```

Run the expensive nominal production objective freeze validation with:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/full_finger_production_objective_freeze.py
```

It evaluates 3 sphere diameters x 7 longitudinal locations x 4 force
checkpoints, writes the complete raw NPZ and report under
`output/validation/full_finger_production_objective_freeze/`, reloads the NPZ,
and verifies mechanics integrity, energy/ROI accounting, LED-order invariance,
and exact objective reproduction. The nominal frozen result is
`J_contact=0.430140029` and `J_obs=0.002798493`; the run does not create Ax
state or start a campaign.

Run the continuous force-ramp protocol comparison against that frozen raw
reference with:

```bash
OTK_INCLUDE_DIR=/home/dk/workspace/optix-toolkit/ShaderUtil/include \
  conda run --no-capture-output -n lit \
  python -u validation/optomech/quasistatic_ramp_protocol.py
```

The procedure evaluates the exact 21-scenario production set at 1, 2, 4, and
8 N/s, capturing first actual-force crossings at 5/10/15/20 N. It writes one
raw NPZ per ramp, a 336-row checkpoint comparison, a protocol summary, and
`report.md` under `output/validation/quasistatic_ramp_protocol/`. Completed
NPZ files are reused on rerun. The measured ramps did not preserve the frozen
raw optical separations, so this command is validation-only and the Ax entry
point continues to use `reference_dwell`.

Compare point/finite-area LED emission and hard/linear longitudinal binning
without rerunning Newton:

```bash
OTK_INCLUDE_DIR=/home/dk/workspace/optix-toolkit/ShaderUtil/include \
  conda run --no-capture-output -n lit \
  python -u validation/optomech/optical_observation_model_sensitivity.py
```

The procedure replays the frozen dwell and four ramp NPZ deformation sets with
common deterministic optical samples. It traces each source model once per
state, derives hard and linear observations from the same escaped rays, and
writes CSV/NPZ results, a comparison plot, and `report.md` under
`output/validation/optical_observation_model_sensitivity/`. The finite source
uses the LuckyLight package drawing's `1.8 x 1.6 mm` water-clear resin window;
it is an optical validation choice, not the production source default.

Profile one exact frozen Newton scenario without OptiX or physics changes:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/newton_runtime_profile.py
```

The procedure runs the nominal 20 mm sphere at `Y=0` through the production
5/10/15/20 N, 5 s dwell protocol twice for cold/warm comparison. It combines
wall timers, sampled CUDA events, and one 20-step Warp activity window, then
writes `report.md` and `timing_breakdown.csv` under
`output/validation/newton_runtime_profile/`. It does not change production
settings or implement any optimization.

Run the historical strict direct-versus-partial-CUDA-graph diagnostic:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/cuda_graph_equivalence.py
```

This older diagnostic documents why bitwise equality is not a valid gate for
Newton's atomically emitted full-surface contacts. It is retained as history,
not as the production activation test.

Run the short GPU-resident force-servo semantics regression:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/gpu_servo_semantics.py
```

Run the Phase 1-B production scientific-equivalence gate:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/gpu_servo_graph_equivalence.py
```

The latter runs direct x5 and GPU-resident graph x5 on the difficult 10 mm
sphere, `Y=+11/+22 mm` pair with the unchanged four force targets and 5 s
dwell. It uses the production finite-area source and hard 11-bin observation,
checks canonical patch IoU, deformation, q-components, limiting optical
separation, inversion/contact-buffer safety, and reports control host
interventions plus measured speedup under
`output/validation/production_evaluator_acceleration/phase1_cuda_graph/`.

Run the Phase 4 fixed-work concurrency benchmark:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/gpu_scenario_parallelism.py
```

This compares 1/2/4/7 independent Newton worlds over the same seven 20 mm
sphere locations. It records throughput, GPU/CPU utilization, peak VRAM, and
scientific-equivalence diagnostics under
`output/validation/production_evaluator_acceleration/phase4_parallel/`.

Run or reload the final accelerated 21-scenario nominal evaluation:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/production_evaluator_acceleration.py
```

The result and campaign-time projection are written to
`output/validation/production_evaluator_acceleration/report.md`. The accepted
production backend uses finite-area/hard optics, GPU-resident graphs, the
validated reuse boundary, four CUDA-stream worlds, and the unchanged 5 s
force-band dwell.

## Full-finger optimization search contract

Validate the 30 mm height envelope and the five-dimensional half-millimeter Ax
encoding without running Newton or OptiX:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q tests/unit/optimization/test_height_constraint.py

conda run --no-capture-output -n lit \
  python -u validation/optomech/corrected_height_constraint.py
```

The report is written to
`output/optimization/corrected_height_constraint_validation.md`. The procedure
does not modify historical Dragon Skin or Solaris campaigns. The active search
definition fixes `flat_pad_width_mm=30` and `void_height_mm=0`; it optimizes the
remaining five geometry values on a 0.5 mm lattice.

`scripts/run_mobo.py` is the production campaign entry point. Review its user
settings, choose a fresh output directory, then run:

```bash
conda run --no-capture-output -n lit python -u scripts/run_mobo.py
```

The campaign is sequential and resumable. It uses only the five-dimensional
`discrete-05mm` search space and the exact objectives `J_contact,J_obs`.
Historical continuous `J_intensity/J_spatial` campaigns are rejected and must
not share the new output directory. The command above is documented for a
separate explicitly authorized campaign run; the objective-freeze validation
did not launch it.

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

## Sensing-objective trade-off sample

Run the deterministic baseline-plus-Sobol morphology sample with:

```bash
conda run -n lit python -u \
  validation/optomech/sensing_objective_tradeoff.py
```

The expensive validation evaluates center contact for 5, 10, and 20 mm
spheres at sequential 5, 10, 15, and 20 N force targets. It writes
`output/validation/sensing_objective_tradeoff.csv` and
`output/validation/sensing_objective_tradeoff.png`. Its trial horizon is 60 s;
when the CSV already contains the same deterministic Sobol points, completed
rows are reused and only incomplete rows are evaluated again.

## Adaptive settling validation

Compare the validation-only `0.2 s` adaptive settling rule against the
production force-band-only `5 s` dwell for centered 5, 10, and 20 mm spheres:

```bash
conda run --no-capture-output -n lit \
  python -u validation/optomech/adaptive_settling.py
```

The script keeps all mechanics and OptiX settings fixed, ray traces only at
accepted checkpoints, prints mechanical/optical differences, and writes
`output/validation/adaptive_settling.csv`.
The measured adaptive rule changes the sensing objectives materially and is
therefore not a production default.
