# LUMO Architecture

This document describes the current LUMO architecture.

It is a navigation and ownership map, not a roadmap, backlog, or description
of previous LUMO generations.

The current implementation is being rebuilt incrementally. Do not recreate
architecture that is absent from the current source tree merely because it
existed previously.

## Design principle

LUMO uses the simplest concrete architecture needed by the current research
pipeline.

Prefer:

- explicit ownership;
- direct data flow;
- upstream Newton and OptiX functionality;
- small production APIs;
- procedural validation;
- one concrete implementation before introducing abstractions.

Do not create architecture for hypothetical future requirements.

## Current dependency flow

The current high-level direction is:

```text
FingertipParameters
        ↓
    Fingertip
        ↓
 LumoSimulation
        └── internally: FingertipMesh → Newton model
```

Later validated mechanics results may be consumed by:

```text
      OptiX
        ↓
   optimization
```

Each layer must be usable and validated before downstream layers depend on it.

## Package ownership

### `lumo/fingertip/`

Owns physical fingertip inputs and analytic geometry.

Current conceptual ownership:

```text
FingertipParameters
        ↓
    Fingertip
    ├── Silicone
    ├── Carrier
    └── BondingInterface
```

`FingertipParameters` contains physical input values.

`FingertipParameters.led` is the single parameter source for the current
Adafruit Green LED Sequin. Its `LEDParameters` owns the hardware identity,
two-dimensional fingertip design envelope, normalized modeled source power,
wavelength metadata, and viewing half-angle. Ray tracing consumes this object
directly rather than maintaining a second copy of those values.

`FingertipParameters.optical` is one immutable `SiliconeOptics` value. The
default is the nominal Dragon Skin 10 NV optical sensitivity preset, matching
the current default mechanics material. Concrete low/nominal/high presets also
exist for Solaris and Dragon Skin 10 NV. These are monochromatic effective
properties, not a material hierarchy: Solaris refractive index is manufacturer
data, while Dragon Skin refractive index and every extinction coefficient are
explicit literature/modeling priors rather than measured product calibration.

The default `ViscoelasticParameters` use the current Dragon Skin 10 NV
baseline: `1070 kg/m³`, `k_mu=1.06e5 Pa`, and `k_lambda=1.0494e7 Pa`. The
Lamé values correspond to a Poisson ratio of `0.495`. The default Newton
damping value remains an uncalibrated `10 Pa·s`, not a datasheet-derived
material measurement.

`Fingertip` constructs the analytic fingertip assembly. Its `tip_z_m` property
exposes the reference silicone tip coordinate in Newton-compatible metres so
callers do not repeat the semiellipse endpoint calculation and unit conversion.

`Silicone` and `Carrier` are constructed geometry objects, not separate
parameter systems. `BondingInterface` is the derived pair of left and right
carrier-silicone polylines that receive the perfect kinematic bond. It does not
own independent physical parameters. A caller may supply a bonding interface
to select a smaller portion of that shared boundary. `Fingertip` clips any
overhang to the actual carrier-silicone boundary and prints a `[WARNING]` when
clipping occurs; an empty or disconnected result is rejected.

This package does not own:

- meshing;
- Newton;
- OptiX;
- optimization;
- validation workflows.

### `lumo/mesh/`

Owns discretization of the analytic fingertip.

The primary object is:

```text
FingertipMesh
├── fingertip
├── silicone
├── carrier
├── carrier_collision
└── bonded_vertex_indices
```

The silicone is discretized as a Newton-compatible tetrahedral mesh.

`carrier` is the complete rigid surface mesh used for visualization.
`carrier_collision` is a closed Newton collision proxy on the same carrier
body. Its silicone-reachable boundary contains only the cavity-facing lips,
stem sides, and stem bottom. Its cross-section closes through the carrier
interior, and its end caps lie outside the silicone extrusion, so Newton's
signed mesh query remains well-defined without exposing closure faces to
silicone particles. The proxy has a cached volume SDF and uses Newton's
full-surface rigid/soft contact path, which catches tetrahedral faces that
would otherwise pass between particle vertices.

`bonded_vertex_indices` identifies silicone vertices lying on
`Fingertip.bonding_interface`. The mesh layer consumes its left and right
polylines directly rather than deriving bond ownership from `Silicone`.

Mesh code may use Gmsh and geometry libraries internally.

This package does not own:

- solver execution;
- contact policy;
- Newton state;
- OptiX execution;
- optimization policy.

### `lumo/newton/`

Owns the concrete Newton implementation.

This package is responsible for the current mechanics backend, including as
needed:

- Newton model construction;
- the kinematic rigid carrier;
- the carrier's visualization-only full shape and invisible particle-contact
  proxy shape;
- the silicone-to-carrier perfect bond;
- kinematic rigid indenters created from URDF or prepared `newton.Mesh` assets;
- Newton-specific model and object identities consumed by the runtime.

Use Newton's public API directly whenever practical.

The full carrier shape has collision disabled. The collision proxy is attached
to the same kinematic body with its rigid-shape pairs explicitly filtered and
silicone particle/full-surface collision enabled. Bonded particles remain
inactive, so the bond interface is controlled only by the prescribed kinematic
bond. Silicone particle radius is explicitly zero at model construction;
contact detection distance is supplied separately by `LumoSimulation` through
the collision pipeline. The rigid carrier proxy uses a stiff shape-contact
material while VBD still permits a small, measured penalty penetration. Its
default normal contact stiffness is `1e6 N/m`; model construction exposes that
scalar only so the numerical sensitivity benchmark can vary it without
changing material parameters.

Do not introduce a generic physics-backend layer while Newton is the only
mechanics implementation.

Do not create generic attachment, constraint, simulation, runtime, or solver
frameworks without a concrete second use case.

### `lumo/simulation/`

Owns the concrete physical runtime and prescribed indentation workflows.

#### `runtime.py`

`LumoSimulation` is the one concrete runtime owner for a complete LUMO
simulation. It owns:

- the analytic `Fingertip`, its generated `FingertipMesh`, and the resulting
  `FingertipNewtonModel`;
- the fixed global `sim_frequency` in hertz, its derived Newton
  `time_step_s`, completed `step_count`, and accumulated `time_s`;
- the current and next Newton states;
- Newton control;
- SolverVBD;
- the collision pipeline and contacts;
- one global simulation step.

The runtime's default soft-contact detection margin is `1e-4 m`. This keeps
the zero-radius silicone particles discoverable by Newton's mesh query without
adding physical particle or shape thickness. Callers may override the margin
when a different scene scale requires it.

The current numerical construction defaults are a `1 mm` mesh element size,
`1000 Hz` simulation frequency, `10` SolverVBD iterations, `1e-4 m` soft
contact margin, and `1e6 N/m` carrier contact stiffness. `LumoSimulation` and
`DesignStudy` expose only these concrete scalars for the convergence
study; there is no simulation-configuration abstraction.

`LumoSimulation(fingertip, builder=...)` is the high-level construction entry
point. It meshes the fingertip, adds it to an optional caller-populated Newton
builder, and finalizes the one shared Newton model. A caller adds external
objects such as an `Indenter` to that builder before constructing the
simulation. Lower-level mesh and Newton-model construction functions remain
available to validations that inspect those stages directly.

The current step order is:

```text
caller optionally updates held fingertip and indenter poses
    ↓
reapply held fingertip pose through the carrier and kinematic bond
    ↓
collision
    ↓
SolverVBD step
    ↓
harvest current body wrenches
    ↓
state swap
    ↓
advance global step count and time
```

`LumoSimulation.step()` does not own approach trajectories, force thresholds,
validation policy, or result reporting. It is the only production API that
advances Newton state, the global step count, or simulation time.
Construction populates the initial contact buffer once, and each global step
refreshes it before SolverVBD. `soft_contact_count()` exposes total or
body-specific counts without leaking Newton contact-array indexing into
callers.
`set_fingertip_pose()` replaces a held fingertip pose that defaults to the
identity pose, and `step()` reapplies it through the rigid carrier and bonded
silicone vertices before every Newton tick.
Callers update other kinematic objects as needed before one tick and may query
the resulting indenter reaction force or maximum active silicone-particle
speed afterward. These observations reduce into preallocated scalar device
buffers rather than cloning full velocity or wrench arrays on every tick. The
runtime may later orchestrate optical work, but no ray-tracing behavior is part
of the current runtime.

#### `design_trial.py`

`DesignTrial` is one indentation definition plus lightweight scalar and pose
results. It never retains its `LumoSimulation` or `Indenter`.
`DesignStudy` owns one immutable analytic `Fingertip` design and an
ordered tuple of design trials. It constructs a fresh builder, indenter,
`LumoSimulation`, and Newton state for each trial, so every trial evaluates the
same fingertip morphology but starts from an independent reference state.
Each trial specifies a normalized `motion_direction_W` and physical
`approach_speed_m_s`; the study derives its per-tick displacement from the
simulation frequency.

The study keeps the per-trial settled-force search direct:

```text
approach until reaction reaches the target
        ↓
hold that pose for the settling duration
        ↓
evaluate the settled force
        ↓
apply one translation correction and hold again when needed
```

Only `LumoSimulation.step()` mutates Newton state or simulation time. The study
does not share mutable Newton state between trials, run trials in parallel, or
introduce a generic simulation manager. A synchronous trial-inspection callback
may validate the live final state; after it returns, the trial runtime is
released before the next trial is constructed.

### `lumo/benchmark/`

Owns explicit, long-running measurement commands that consume the current
production simulation APIs. It is not imported by the mechanics runtime.

`newton_parameter_sweep.py` holds the physical fingertip and 10 mm
center-sphere case and `25 mm/s` approach speed fixed while varying mesh element
size, simulation frequency, SolverVBD iterations, soft-contact margin, and
carrier contact stiffness one family at a time. Each frequency therefore uses
a per-tick translation derived from the fixed speed. The benchmark writes one
strict JSON artifact after every requested run has finished. The artifact
contains the raw scalar results and comparisons to the measured baseline.
`--fine` adds only the expensive `0.5 mm` mesh case; `--matrix` adds the
baseline-only 3-by-3 sphere/location robustness check.

Benchmarks may report production behavior, but they do not own simulation
state, solver policy, material calibration, or production defaults.

### `lumo/ray_tracing/`

Owns LUMO-specific optical transport behavior.

OptiX is the ray-tracing backend.

`OptixScene` is the first concrete OptiX 9.1 runtime component. It owns one
persistent CUDA stream, the OptiX context and pipeline resources, device
geometry buffers, two triangle GASes for silicone and carrier, and the IAS
containing those two instances. Its only query `trace_closest()` returns hit
state, distance, instance ID, primitive ID, triangle barycentrics, and the
world-frame geometric normal `normal_W`.
Triangle hits also return NVIDIA OptiX Toolkit ShaderUtil
`spawn_front_W` and `spawn_back_W` positions for robust secondary-ray launch;
misses return NaN normal and spawn positions.

The triangle closest-hit program fetches the current triangle vertices through
the OptiX 9.1 current-hit API. The vertices and barycentrics are passed to OTK
`getSafeTriangleSpawnOffset()`, followed by the current-hit transform and
two-sided offset operations. The world-space unit normal returned by that OTK
transform is both reported as `normal_W` and used to construct the safe spawn
points. This explicit-vertex OTK overload avoids the random vertex access build
flag, a duplicate normal transform, and any host-side triangle lookup.

`update_silicone()` accepts positions for the same vertex count and topology,
copies them into the persistent silicone vertex buffer, and performs an in-place
silicone GAS UPDATE followed by IAS UPDATE. The two acceleration structures
reuse their original output buffers and dedicated persistent update scratch
buffers. The carrier GAS remains static.

`interface_transport()` in `transport.py` is the first concrete optical
operation. It normalizes a batch of incident directions and geometric normals,
locally orients each normal against its incident ray, and evaluates one
lossless dielectric interface with caller-supplied scalar refractive indices.
It returns reflected and refracted directions, unpolarized Fresnel reflectance,
transmittance, reflected and refracted scalar ray power, and a
total-internal-reflection flag. Incident power is either one scalar for the
batch or one nonnegative value per ray. This power is a discrete optical-energy
weight: the lossless split is `P * R` and `P * T`, without radiance transport
factors. A TIR result preserves all power in reflection and uses a NaN refracted
direction. The function does not track media or launch secondary rays.

`lambertian_reflection()` models one effective opaque Lambertian carrier event.
It cosine-samples the reflected hemisphere from caller-supplied deterministic
sample coordinates and returns `albedo * incident_power` as one Monte Carlo ray
weight plus the complementary absorbed power. Albedo is supplied by the caller;
current validation values are placeholders, not calibrated white-PLA material
constants. The function does not own RNG policy, materials, media, or tracing.

`lambertian_emission()` uses the same private cosine-weighted hemisphere
sampling operation for one ideal Lambertian source. Caller-supplied sample
coordinates determine the directions, and normalized total source power is
divided equally among the rays.

`LED` combines one `LEDParameters` with a world-frame position and unit normal,
then delegates emission sampling to `lambertian_emission()`. It does not copy
or independently own hardware or optical parameter values. The current
parameters identify the Adafruit Green LED Sequin (Product ID 1756), whose
underlying LED is the LuckyLight S150PGC-G5-1B 1206 Pure Green InGaN package.
Its hardware metadata is a 525 nm dominant wavelength, 520 nm peak wavelength,
35 nm spectral half-width, and 60-degree off-axis half-intensity angle
(120-degree full viewing angle). The 60-degree half-intensity angle is
consistent with an ideal Lambertian first-order model because
`cos(60 degrees) = 0.5`, but the model does not claim to reproduce the complete
measured radiation diagram. Source power remains a normalized `1.0` until
optical calibration, and current transport remains monochromatic scalar-power
transport. Hardware values come from the
[Adafruit product page](https://www.adafruit.com/product/1756) and the
[LuckyLight datasheet](https://cdn-shop.adafruit.com/datasheets/S150PGC-G5-1B.pdf).

`trace_bounded_paths()` is the one host-side multi-bounce orchestration
operation. It keeps flat NumPy arrays for origin, direction, power, original
ray ID, and one `inside_silicone` boolean; there is no path object or generic
medium stack. Each silicone event reuses `interface_transport()` and samples
one Fresnel branch from caller-precomputed values indexed by bounce and
original ray ID. Because the selection probability equals the lossless branch
contribution, the selected path retains its power rather than multiplying by
Fresnel a second time. Before each hit reached while the path is inside
silicone, the operation applies Beer-Lambert attenuation using that OptiX hit's
metric `t` distance and accumulates the removed ballistic power as bulk loss.
Air segments are not attenuated, and no second geometry-distance calculation
exists. Carrier events reuse `lambertian_reflection()` and accumulate carrier
absorption. External misses become escaped structured rays, internal misses
remain explicit unresolved power, and active power at the caller-supplied depth
cap is reported as remaining power. The aggregate ledger closes emitted power
against escape, carrier absorption, bulk loss, internal miss, and remaining
power. The effective extinction may include unresolved scattering, especially
for translucent Dragon Skin 10 NV; volumetric scattering is not modeled. This
is a bounded concrete fingertip path operation, not a renderer.

`PathTraceResult` in `path_result.py` is the fixed public result contract for
that operation. It owns the escaped-ray array, explicit scalar power ledger,
remaining ray count, and optional diagnostic segments. The path algorithm no
longer changes tuple arity or exposes a string-keyed statistics dictionary.
OptiX hit layouts remain next to their CUDA decoding in `scene.py`, while the
short-lived vectorized Fresnel and Lambertian result layouts remain next to the
numerical operations in `transport.py`.

The optional `record_segments=True` diagnostic mode fills
`PathTraceResult.segments` with compact ray ID, bounce, start, end, power, and
hit-instance records for finite hit segments. The default is false, retains no
segment history, and returns `segments=None`. The diagnostic data comes from
the same 3D bounded transport; it does not define a second tracer.

`safe_secondary_origins()` selects the OTK front or back spawn position by the
sign of the outgoing direction dotted with `normal_W`. It does not infer media
or trace a ray. Both focused single-event validations and the bounded path loop
use this operation for every triangle departure. OptiX traversal uses `tmin=0`:
the OTK origin owns self-intersection separation, so no second scene epsilon is
combined with the official offset.

`side_view_observation()` in `observation.py` reduces escaped paths from the one
current optical-cell LED to one raw four-quadrant response. It keeps only rays
traveling toward the canonical camera-facing `+Y` side and bins their power by
the escape origin in the X-Z cross section. Quadrants are ordered upper-right,
upper-left, lower-left, lower-right around the current analytic silicone
semiellipse center. This is a directional side-view response, not a camera,
image, projection plane, pixel model, or optimization score.

The scene has no Newton dependency. A caller may explicitly pass a Newton
checkpoint through `update_silicone()`, but no production runtime currently
orchestrates Newton and OptiX. The silicone input surface is selected by the
caller; the first IAS validation removes surface triangles whose three vertices
all belong to the perfect bonded interface.

This package should define only the optical semantics required by LUMO and use
OptiX for generic ray-tracing functionality.

Do not implement a general-purpose ray tracer inside LUMO.

This layer must not own mechanics evolution.

### `lumo/optimization/`

Owns design-space and optimization policy.

Current responsibilities include design parameter bounds, feasibility
constraints, and pure sensing-objective evaluation.

`sensing_descriptors()` consumes a state array shaped `(contact states, 4)`.
For the current single optical cell it forms one scalar intensity response from
total side-visible power relative to no contact, and one normalized four-value
spatial response per state. It rejects a numerically zero no-contact reference
or state-visible total instead of adding an objective-changing epsilon.
`sensing_objectives()` returns the separate worst-case pairwise scalar-intensity
and Euclidean spatial separations. It does not know about LEDs, Newton, OptiX,
ray tracing, morphology, or objective weights.

Future optimization code should consume validated simulation outputs rather
than implementing mechanics or optical transport itself.

### `lumo/util/`

Contains only small shared helpers with clear current consumers.

`lumo/util/mesh_io.py` converts OBJ or STL triangle-surface files into prepared
`newton.Mesh` objects. Source coordinates are converted to metres with an
explicit caller-provided scale. It does not parse URDF or add Newton bodies and
shapes.

Rigid indenter asset flow is:

```text
packaged URDF resource → filesystem path → Indenter.add_urdf()

OBJ/STL path → mesh_io.load_mesh() → newton.Mesh → Indenter.add_mesh()
```

LUMO-owned URDF assets live under `lumo/assets/objects/`, with primitive sphere
indenters grouped under `lumo/assets/objects/urdf/`. They are installed as
package data and resolved with `importlib.resources`. Validation callers do not
derive repository roots from their own `__file__` paths. Sphere size names
refer to diameter in millimetres.

An external rigid body is composed with the fingertip before Newton model
finalization:

```text
caller-owned ModelBuilder
├── Indenter.add_urdf(...) or Indenter.add_mesh(...)
└── LumoSimulation(fingertip, builder=builder)
        └── finalize one shared Newton model
```

Neither `LumoSimulation` nor `build_fingertip_newton_model()` chooses or places
an indenter. Both accept a caller-populated builder so all scene bodies can be
finalized into the same Newton model.

`Indenter.add_urdf()` stores its supplied world `tf` as the initial kinematic
body pose as well as passing it to Newton's URDF importer.

`Indenter` owns only construction and the Newton body index identifying one
kinematic rigid indenter. `LumoSimulation` owns mutations of Newton state,
including fingertip and indenter pose updates.

Do not turn this package into a general service or utility layer.

## Validation boundary

`validation/` is not a production package.

Validation scripts are top-level consumers of production APIs.

`validation/contact-physics/zero_load.py` is the current procedural check that
SolverVBD preserves the unloaded fingertip reference state.

`validation/fingertip/view_bond_geometry.py` renders the analytic XZ
cross-section and highlights the exact `BondingInterface.left` and
`BondingInterface.right` polylines whose silicone vertices receive the perfect
kinematic bond.

`validation/contact-physics/flat_plate_contact.py` loads the flat-plate URDF,
moves its kinematic body toward the fingertip in positive-Z increments for a
bounded simulation duration, calls the one global simulation step per counter
increment, and stops its local loop when the transient negative-Z reaction
force reaches the script-local threshold. It separately counts flat-plate and
carrier contacts, checks the bonded-vertex drift, and measures nonbonded
vertex, surface-vertex, and tetrahedron-center penetration into the analytic
carrier. Its optional ViewerGL path only observes simulation state and
contacts; it does not advance or mutate the simulation.

`validation/contact-physics/sphere_indentation.py` creates one analytic
fingertip and uses a `DesignStudy` to run the packaged 5, 10, and 20 mm
diameter sphere URDFs at `X=-7.5`, `0`, and `+7.5 mm`. It places each sphere
from the local analytic semiellipse height at that X location. Each of the nine
independent simulations triggers at `20 N`, corrects the held pose as needed,
and holds each checkpoint for `5 ms` before evaluating the settled force. If it
is outside `20 ± 5 N`, one translation correction is applied before the next
hold. It then checks contact, finite silicone state, perfect-bond drift, and
carrier penetration before releasing that runtime.

`validation/contact-physics/sphere_15mm_viewer.py` is a focused interactive
contact diagnostic. It loads the packaged 15 mm sphere, advances that one
kinematic body at a fixed positive-Z speed, renders every Newton state and
contact set, and freezes the first state whose transient reaction reaches
`20 N`. It reports force, active-particle speed, and sphere contact count at a
throttled interval; it does not run the settled-force search.

`validation/contact-physics/poisson_ratio_sweep.py` repeats that prescribed
contact protocol for explicit near-incompressible Poisson ratios. It derives
the Lamé `k_lambda` from a fixed `k_mu`, reports force and tetrahedral volume
change, and keeps all sweep policy local to the validation script.

`validation/ray-tracing/ias_test.py` builds a static OptiX IAS from the real
undeformed silicone surface and full carrier mesh. Its deterministic rays
verify closest silicone and carrier instance hits, a full-scene miss,
visibility-mask exclusion, triangle barycentrics, and the analytic silicone
surface distance. It does not perform optical transport or couple Newton state
into OptiX.

`validation/ray-tracing/refit_test.py` translates every silicone vertex by
`+1 mm` in Z, updates the existing silicone GAS and IAS, and verifies the
expected `+1 mm` hit-distance change without changing the hit primitive or
barycentrics. It also compares displaced-scene hits against a fresh full build,
checks that carrier and miss results are unchanged, and reports
validation-local update and fresh-construction timing.

`validation/ray-tracing/newton_refit_test.py` exercises the explicit mechanics
checkpoint handoff. It uses the same centered 15 mm indenter and fixed-speed
transient `20 N` approach as the interactive contact viewer, extracts the live
Newton silicone vertices, updates the existing OptiX silicone GAS and IAS, and
compares ray results against a fresh scene built from the same deformed
vertices. Carrier and miss queries verify that the non-silicone scene state
remains unchanged; the Newton sphere indenter is not an optical scene instance.

`validation/ray-tracing/normal_test.py` traces deterministic rays against a
planar carrier face and the exposed silicone semiellipse. It compares their
world-frame geometric normals with analytic references and feeds the silicone
hit normal into one air-to-silicone `interface_transport()` call.

`validation/ray-tracing/interface_transport_test.py` independently checks
normal and oblique air-to-silicone refraction, below-critical silicone-to-air
refraction, and above-critical total internal reflection against deterministic
analytic results. It also checks scalar and per-ray power splitting,
conservation, zero power, and invalid power inputs.

`validation/ray-tracing/secondary_ray_test.py` traces one undeformed-fingertip
ray from air into exposed silicone, applies one dielectric refraction, selects
the transmitted-side OTK spawn position, and performs exactly one second OptiX
launch. The deterministic path must hit carrier rather than immediately
self-intersecting the primary silicone triangle.

`validation/ray-tracing/power_branch_test.py` applies one lossless power split
at that same real silicone interface. It traces the OTK-safe reflected and
refracted branches exactly once each, verifies power conservation and no
primary-triangle self-hit, and requires the reflected branch to leave the
fingertip while the refracted branch reaches the carrier. It does not recurse or
assign optical behavior to the carrier.

`validation/ray-tracing/carrier_reflection_test.py` checks deterministic
cosine-weighted Lambertian directions and opaque reflected/absorbed power, then
traces one real undeformed path from air through silicone to carrier and back to
the exposed silicone surface. It uses the existing OTK-safe triangle spawn and
stops at that third geometry hit without processing another interface.

`validation/ray-tracing/silicone_exit_test.py` composes the existing trace,
dielectric, Lambertian, and OTK-safe spawn operations into one deterministic
`air -> silicone -> carrier -> silicone -> air` path. The final silicone-to-air
transmission must miss the scene, and the validation accounts for all reflected,
absorbed, and escaped power without introducing a bounce loop or medium state.

`validation/ray-tracing/dielectric_branch_test.py` checks the critical sampled
dielectric rule directly: `u < R` reflects, `u >= R` transmits, TIR always
reflects, only transmission toggles the silicone-medium flag, and a selected
lossless branch retains its incident path power.

`validation/ray-tracing/led_sensor_response_test.py` is one deterministic
source-to-receiver optical-property sensitivity study. It uses the actual Green
Sequin hardware metadata through `LED`, while its point-source pose and ideal
planar receiver remain validation-local placeholders. The study reuses the
same 64-by-64 stratified LED samples and the same precomputed per-ray/per-bounce
dielectric and carrier samples before and after a force-stable central 10 mm
sphere indentation. It updates the silicone GAS from the live Newton checkpoint
and evaluates a fixed 24-bounce cap for low, nominal, and high Solaris and
Dragon Skin 10 NV optical priors.

`validation/ray-tracing/sensing_evaluator_test.py` composes the first complete
sensing evaluator for one approximately 11 mm-wide optical cell. One LED is
placed at the carrier stem-bottom center in X-Z and at the extrusion-axis center
derived from the fingertip mesh. Its Lambertian hemisphere points in the pad
direction (`-Z`). The physical point remains on the carrier boundary; an OTK
safe pad-side origin removes coincident-triangle ambiguity. The first oriented
silicone hit then determines whether a load-induced gap ahead of the fixed
carrier source is air or silicone. The LED is not a mechanics or contact
object. The script verifies every emitted primary ray against each deformed
state. It traces one no-contact state and
independent `20 N` sphere contacts at `X=-7.5`, `0`, and `+7.5 mm` with common
optical samples, produces one four-quadrant side-view response per state,
derives both descriptors, and reports each objective together with its
worst-case state pair. It does not simulate a camera or optimize morphology.

`validation/ray-tracing/sensing_visualization.py` projects a deterministic
subset of those real 3D segment records onto the LED-center X-Z plane. It plots
the actual triangle-plane section of unloaded and Newton-deformed silicone,
the LED, `+Y` escape locations, and the observation quadrants in matched axes.
This Matplotlib-only diagnostic is not a 2D optical simulation or production
rendering API.

`validation/contact-physics/sensing_convergence.py` is the single procedural
Newton/OptiX convergence study for that evaluator. It first compares 5, 20,
and 50 ms fixed-pose settling holds, then runs a small one-factor-at-a-time
study of carrier stiffness, timestep frequency, VBD iterations, and mesh size.
Only the selected hard-valid reference/contact vertex snapshots survive for the
later 16384/65536-ray, three-seed convergence comparison. The script adds no
production sweep or convergence abstraction. If a Newton setting fails the
existing mechanics acceptance checks, the script records it and continues with
the remaining settings.
Its carrier check treats tetrahedra touching the perfect-bond interface as a
separate diagnostic because that interface intentionally has no contact
constraint; only nonbonded particles and nonbonded tetrahedra determine the
penetration acceptance result.

`validation/optomech/newton_parameter_sweep.py` is a separate procedural
throughput study. It runs the requested 24 Newton combinations, evaluates only
hard-valid states with the fixed 65,536-ray optical protocol, records stage
timings and per-contact diagnostics, and writes a machine-readable result. It
does not introduce a benchmark framework or production timing hooks.

They should normally:

1. construct production objects;
2. execute the behavior being checked;
3. measure or assert the relevant property;
4. report the result.

Keep validation procedural and local by default.

Do not create reusable validation classes, runners, configuration systems, or
public APIs unless explicitly required.

Production packages must never depend on `validation/`.

## Dependency rules

Allowed high-level direction:

```text
lumo.fingertip
      ↓
lumo.mesh
      ↓
lumo.newton
      ↓
lumo.simulation
      ↓
lumo.benchmark

lumo.fingertip / lumo.mesh
      ↓
lumo.ray_tracing

validated simulation outputs
      ↓
lumo.optimization
```

Important constraints:

- `lumo.fingertip` must not import mesh, Newton, OptiX, or optimization code.
- `lumo.mesh` must not import solver, ray-tracing, optimization, or validation
  code.
- `lumo.newton` must not import ray-tracing, optimization, or validation code.
- `lumo.simulation` may compose the concrete Newton runtime but must not import
  validation or optimization policy.
- `lumo.benchmark` may consume fingertip and simulation APIs; production runtime
  packages must not import benchmark code.
- `lumo.ray_tracing` must not import Newton solver implementation.
- production packages must not import `validation/` or `tests/`.
- optional heavy dependencies should enter only at their owning execution
  boundary.

## Units and coordinate convention

Public fingertip geometry is expressed in millimetres.

Newton execution uses SI units.

The canonical LUMO frame is:

```text
X = cross-section lateral direction
Y = fingertip width / extrusion direction
Z = contact-normal direction
```

Unit conversion belongs at the appropriate backend boundary and should not be
silently duplicated across packages.

## External libraries

Newton and OptiX are primary dependencies, not reference implementations.
NVIDIA OTK ShaderUtil supplies the official header-only self-intersection
avoidance implementation consumed by the OptiX kernel.

Before adding nontrivial functionality using either library:

1. inspect the installed or targeted version;
2. inspect its current public API;
3. read the corresponding official documentation;
4. inspect upstream examples or source when necessary.

Prefer upstream functionality over repository-owned replacements.

Do not infer current behavior from old LUMO code or from a different upstream
development version.

## Current rebuild sequence

LUMO is being rebuilt one concrete capability at a time.

The current progression is:

```text
analytic fingertip
    ↓
mesh
    ↓
kinematic silicone-carrier bond
    ↓
Newton model construction
    ↓
LumoSimulation execution
    ↓
zero-load validation
    ↓
contact / indentation
    ↓
mechanics outputs and convergence
    ↓
OptiX transport
    ↓
optimization
```

This is sequencing guidance, not permission to implement downstream stages
during an earlier task.

Future work being predictable does not make it part of the current task.

## Intentionally absent architecture

Do not recreate old LUMO architecture simply because it existed previously.

In particular, do not introduce without a concrete present need:

- a generic physics package or backend interface;
- generic simulation orchestration frameworks;
- generic attachment or constraint systems;
- solver factories or registries;
- generic ray-tracing frameworks;
- compatibility layers for removed internal APIs;
- production visualization frameworks;
- reusable validation frameworks.

Legacy code may be consulted for scientific intent and failure history, but it
is not the architectural source of truth.

The current source tree is authoritative.
