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

`Indenter.add_urdf()` and `Indenter.add_mesh()` accept optional normal-contact
stiffness and damping overrides. Their default `None` values preserve Newton's
shape material because no indenter contact pair has been frozen as a production
numerical contract. URDF construction applies requested values only while the
importer creates that asset and then restores the builder defaults.
Prepared-mesh construction applies requested values to a copied shape
configuration. Neither path changes objects added later.

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

Newton is pinned to `1.5.0`. `LumoSimulation` uses its full-surface VBD proxy
wrench harvest, added upstream in Newton #3756, so indenter reaction includes
particle, edge, and face rigid-soft contact records while the indenter remains
kinematic. The runtime does not maintain a repository-owned copy of Newton's
contact-force kernel. SolverVBD's per-body rigid-soft contact list is allocated
for `2048` records; the current kinematic indenter is not truncated by that
dynamic-body list, but the larger capacity keeps dense future proxy use away
from Newton's default `256`-record limit.

The current numerical construction defaults are a `1 mm` mesh element size,
`1000 Hz` simulation frequency, `10` SolverVBD iterations, `1e-4 m` soft
contact margin, and `1e6 N/m` carrier contact stiffness. Optional
`soft_contact_stiffness_n_m` and `soft_contact_damping_n_s_m` values support the
focused rigid-soft pair study; `None` preserves Newton's model defaults. There
is no simulation-configuration abstraction.

`LumoSimulation(fingertip, builder=...)` is the high-level construction entry
point. It meshes the fingertip, adds it to an optional caller-populated Newton
builder, and finalizes the one shared Newton model. A caller adds external
objects such as an `Indenter` to that builder before constructing the
simulation. An optional caller-built `fingertip_mesh` lets the end-to-end
evaluator give Newton and OptiX the same immutable discretization without
meshing twice. Lower-level mesh and Newton-model construction functions remain
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
results. Its optional initial-clearance scalar lets consumers report physical
indentation rather than total approach travel. It never retains its
`LumoSimulation` or `Indenter`.
`DesignStudy` owns one immutable analytic `Fingertip` design and an
ordered tuple of design trials. It constructs a fresh builder, indenter,
`LumoSimulation`, and Newton state for each trial, so every trial evaluates the
same fingertip morphology but starts from an independent reference state.
An optional immutable `FingertipMesh` is reused by each fresh Newton runtime;
ordinary callers continue to let `LumoSimulation` mesh internally.
The optional `contact_stiffness_n_m` and `contact_damping_n_s_m` values apply
the same rigid-soft contact pair values to the URDF indenter endpoint and the
Newton soft endpoint. `None` preserves both Newton defaults; the study does not
freeze provisional contact values as production defaults.
Each trial specifies a normalized `motion_direction_W` and physical
`approach_speed_m_s`, which caps the magnitude of its kinematic force-servo
velocity. `DesignStudy` applies
`v = clamp(Kf * (F_target - F), -approach_speed_m_s, approach_speed_m_s)` with
the default `Kf=1.25e-3 m/(s N)`, then derives each pose increment from the
simulation timestep.

For the concrete multi-force sensing evaluation, a study may use one strictly
increasing force-target tuple. The same trial runtime then advances through
each target without resetting Newton state. Tolerance may be an absolute force
or a target-relative fraction; the sensing evaluator uses `5, 10, 15, 20 N`
with `+/- 10%` at every level. The live inspection callback runs immediately
after every accepted target while the current Newton state still exists.

The study keeps the per-trial force servo direct:

```text
last reaction force
        ↓
bounded proportional velocity
        ↓
update the kinematic indenter pose
        ↓
one LumoSimulation step and reaction measurement
        ↓
test force and commanded-displacement settling conditions
        ↓
accept after both remain true for the settling duration
```

Leaving the force tolerance or exceeding an optional per-tick commanded
displacement tolerance resets the single consecutive-tick counter. The
production sensing evaluation leaves the displacement criterion disabled and
requires `+/- 10%` target-force agreement continuously for `5 s`. The optional
displacement criterion remains available only for the focused adaptive-settling
validation. There is no force-slope or particle-speed criterion, first-crossing
stop, fixed-pose correction search, integral term, or PID controller.

Only `LumoSimulation.step()` mutates Newton state or simulation time. The study
does not share mutable Newton state between trials, run trials in parallel, or
introduce a generic simulation manager. A synchronous trial-inspection callback
may inspect each accepted live state; after the final callback returns, the
trial runtime is released before the next trial is constructed.

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

The scene has no Newton dependency. The optimization evaluator explicitly
passes each final Newton vertex checkpoint through `update_silicone()`; neither
the optical scene nor the mechanics runtime imports the other. The silicone
input surface is selected by the caller; the first IAS validation removes
surface triangles whose three vertices all belong to the perfect bonded
interface.

This package should define only the optical semantics required by LUMO and use
OptiX for generic ray-tracing functionality.

Do not implement a general-purpose ray tracer inside LUMO.

This layer must not own mechanics evolution.

### `lumo/optimization/`

Owns design-space and optimization policy.

Current responsibilities include design parameter bounds, feasibility
constraints, one concrete sensing evaluation, and pure sensing-objective
evaluation.

`evaluator.py` owns the production mechanics-to-optics orchestration. It builds
one `FingertipMesh` and one `OptixScene`, generates deterministic samples and
traces the undeformed state once, then runs supplied `DesignTrial` scenarios in
independent Newton runtimes. Each live final vertex state updates the same
silicone GAS and IAS and is traced with the same emission and bounce samples.
At each accepted `5, 10, 15, 20 N` checkpoint it immediately records actual
force, indentation, a four-quadrant response, and a compact scalar energy
ledger. It then releases the full `PathTraceResult` and escaped-ray arrays
before Newton continues toward the next target. The returned response array is
shaped `(scenario, force, quadrant)` and the energy array is shaped
`(scenario, force, energy field)`, with separate no-contact reference vectors
plus checkpoint simulation times and per-scenario wall runtime. The fixed
current mechanics contract is
`100 Hz`, `10` VBD iterations, equal `ke=3e4 N/m` endpoints,
`kd=0.282280175 N s/m`, the proportional force servo, and acceptance after
force remains within its `+/- 10%` band continuously for `5 s`. The servo uses
`Kf=2.5e-4 m/(s N)` with a `5 mm/s` trial cap, preserving the validated
`2.5 um/(N tick)` gain and `50 um/tick` maximum step. Optical transport
uses `65,536` paths and `24` bounces. This evaluator does not perform
morphology optimization or combine sensing objectives.
The LED remains on the carrier stem boundary. If a nonzero geometry void
places air between that source and silicone, transport starts in air and lets
OptiX resolve silicone entry, carrier reflection, or escape rather than
requiring every primary ray to enter silicone immediately.

`sensing_descriptors()` consumes a state array shaped `(contact states, 4)`.
For the current single optical cell it forms one scalar intensity response from
total side-visible power relative to no contact, and one normalized four-value
spatial response per state. It rejects a numerically zero no-contact reference
or state-visible total instead of adding an objective-changing epsilon.
`sensing_objectives()` returns the separate worst-case pairwise scalar-intensity
and Euclidean spatial separations. With grouped responses shaped
`(indenter, force state, quadrant)`, it compares force states only within each
indenter and returns both the per-indenter values and their indenter-wise
minima. It does not know about LEDs, Newton, OptiX, ray tracing, morphology, or
objective weights.

`ax_bo.py` owns the sequential multi-objective optimization loop and its two
separate campaign definitions. The original `continuous` campaign attaches and
verifies the 13 completed Sobol observations from the sensing trade-off
validation. The `discrete-05mm` campaign starts a separate Ax state with no
reused objective values. It fixes `flat_pad_width_mm=30`, exposes the other six
geometry dimensions as integer half-millimeter steps, and decodes those steps
to physical millimeters only at the evaluator boundary. Ax directly enforces
`flat_pad_height_step + semiellipse_height_step <= 60`; the existing
`FingertipGeometry` and `DesignSpace` checks remain the sole owners of nonlinear
geometry validity.
`scripts/run_mobo.py` is the user-edited entry point for the discrete campaign.
It exposes the physical parameter bounds, indenter URDF list, sequential force
targets, fixed force-band dwell, relative tolerance, output directory, and
successful-morphology target. The optimizer validates these inputs and records
them in `run_config.json`; mechanics and optical algorithms remain owned by
their production modules. All configured indenters share one initial center
pose derived from a 20 mm reference indenter; smaller packaged spheres simply
approach from farther away. URDF filenames identify result groups but do not
encode placement dimensions. For these common-pose trials, `DesignTrial`
records the first positive contact-force travel as the clearance so reported
indentation excludes object-dependent free approach. Callers that already know
their geometric clearance can continue supplying it directly.

The discrete campaign records both integer steps and decoded millimeters in
its CSV. Its six snapped continuous-Pareto designs are only ordered design
seeds: each receives a fresh Newton-to-OptiX evaluation before it becomes an
observation. No continuous objective is copied. Both campaigns maximize the
same two independent objectives and request exactly one candidate at a time.
Every proposed parameterization passes
through `DesignSpace.is_feasible()` before Newton or OptiX is constructed. An
invalid proposal is marked abandoned in Ax without a fabricated objective
value.
An evaluator failure is preserved as `FAILED` with its reason in the CSV while
its Ax arm is abandoned, preventing deterministic re-proposal of the same
failed morphology. It receives no objective value or penalty.

Each feasible proposal uses `evaluator.py` for the configured centered
indenter scenarios and uses `sensing_objective.py` for the indenter-wise
worst-case objectives. The optimizer does not implement mechanics, optical
transport, or objective arithmetic. Each successful new trial keeps only the
compact evaluator arrays in one compressed NPZ; Newton, OptiX, and escaped-ray
buffers are not retained between candidates.

`run_config.json` freezes the scientific contract and scientific-source digest
before the first proposal. Resume refuses a changed contract or changed
scientific production source; optimizer-only source changes are recorded
separately so a persistence bug can be corrected without invalidating completed
Newton/OptiX observations.
`ax_state.json`, `trials.csv`, and `pareto.csv` are atomically replaced after
state transitions. If interruption occurs after an NPZ is written but before
Ax completion, the `EVALUATED` CSV row lets resume report that saved result to
Ax without repeating Newton or OptiX. A trial interrupted during evaluation is
abandoned, so a crash loses at most the currently running morphology. The CLI
budget is cumulative and counts only successful new BO trials; the 13 attached
warm starts are excluded.

```text
13 completed Sobol observations
        ↓ attach to Ax
Ax multi-objective MBM, max_trials=1
        ↓
analytic DesignSpace feasibility
        ↓ valid
production evaluator: configured centered indenter scenarios
        ↓
indenter-wise J_intensity and J_spatial
        ↓
compressed raw NPZ → CSV → complete Ax → atomic Ax state + Pareto CSV
```

The discrete campaign follows the same persistence and evaluator flow, but its
initialization begins with freshly evaluated snapped design seeds instead of
completed warm-start observations. Its `run_config.json` additionally freezes
the 0.5 mm resolution, integer step bounds, decoded physical bounds, fixed pad
width, and step-space pad-depth constraint. Continuous and discrete campaigns
use different output directories and cannot share Ax state or observations.

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
independent simulations uses the kinematic proportional force servo to approach
`20 N` and accepts only after remaining inside `20 ± 1 N` for `5 ms`. It then
checks contact, finite silicone state, perfect-bond drift, and carrier
penetration before releasing that runtime.

`validation/contact-physics/sphere_15mm_viewer.py` is a focused interactive
contact diagnostic. It loads the packaged 15 mm sphere, advances that one
kinematic body at a fixed positive-Z speed, renders every Newton state and
contact set, and holds the sphere pose fixed for `10 s` after the transient
reaction first reaches `20 N`. It continues advancing and rendering Newton
during that hold, reports force, active-particle speed, and sphere contact count
at a throttled interval, and freezes the final held state until the viewer
closes. It is a fixed-speed diagnostic and does not run the production force
servo.

`validation/contact-physics/force_traj.py` repeats that centered 15 mm sphere
approach without a viewer, stops prescribed motion at the first transient
`20 N` crossing, and then holds the indenter pose exactly fixed for `10 s`.
It compares the default `1000 Hz / 10 iteration` run against a diagnostic
one-time particle-velocity reset, `1000 Hz / 30 iterations`, and
`2000 Hz / 20 iterations`. Every hold tick records reaction force, maximum
active-particle speed, sphere and carrier contact counts, nonbonded-particle
carrier penetration, and total tetrahedral volume ratio. The velocity reset is
validation-only and is not a production settling behavior. The script diagnoses
rate-dependent and numerical settling; it does not interpret the measured decay
as a calibrated material relaxation law.

`validation/contact-physics/sphere_force_depth.py` performs one nominal centered
15 mm sphere indentation with no force correction. It continuously advances
the sphere to `10 mm` analytic indentation depth while recording the
full-surface reaction force, sphere contact count, analytic sphere penetration for
nonbonded particles, free surface vertices, free surface-triangle centroids,
and fully-free tetrahedron centers, plus per-tet signed volume ratios and
inversion counts. A fresh second runtime stops at the first `20 N` crossing,
holds the sphere pose fixed for `1 s`, and reports force at `0`, `5`, `100`, and
`1000 ms`. Its CSV and plot distinguish a nonmonotonic force branch from
sphere-contact leakage or tetrahedron collapse without changing production
mechanics.

`validation/contact-physics/sphere_contact_tuning.py` is the focused contact
system diagnostic. It first measures the actual full-surface contact record
masses, then sets equal rigid-shape and soft-contact endpoint values for
`ke={1e4,3e4,1e5} N/m` and mass-scaled under/critical/over damping at `2 kHz`.
It checks two stable candidates again at `4 kHz`, records force-depth,
penetration, tetrahedral quality, fixed-pose force, contact-list overflow, and
active-particle speed, and writes CSV/PNG evidence. It does not set a production
contact default or run optics.

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

`validation/ray-tracing/sensing_evaluator_test.py` runs the production evaluator
for the Cartesian product of 5, 10, and 20 mm spheres with contact locations
`X=-7.5, 0, +7.5 mm`. Each of those nine fresh Newton runtimes advances through
`5, 10, 15, 20 N` without reset. The script reports the shared no-contact
response, accepted mechanics checkpoints, the nine-by-four-by-four side-view
response matrix, compact energy ledgers, response deltas, and wall runtimes.
All scenarios reuse one mesh, one OptiX scene, and one deterministic optical
sample set while their Newton runtimes remain independent and sequential. The
script does not simulate a camera, compute a morphology objective, or start
optimization.

`validation/ray-tracing/sensing_visualization.py` projects a deterministic
subset of those real 3D segment records onto the LED-center X-Z plane. It plots
the actual triangle-plane section of unloaded and Newton-deformed silicone,
the LED, `+Y` escape locations, and the observation quadrants in matched axes.
The loaded panel uses the current centered 15 mm sphere, `500 Hz / 10 iteration`
contact configuration and the accepted `20 +/- 1 N` checkpoint after a 5 s
continuous force-band hold. Both panels reuse the same 4096 deterministic
diagnostic paths with a 24-bounce cap.
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
