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
Green LED Sequin model. `LEDParameters` contains only dimensions and source
power that affect the simulation: the board cross-section, finite emitting
window, and normalized modeled power. Product and spectral metadata remain
documentation rather than inactive runtime parameters.

`FingertipParameters.optics` is one immutable `SiliconeOptics` value. The
default is the nominal Dragon Skin 10 NV optical sensitivity preset, matching
the current optical baseline. Concrete low/nominal/high presets also exist for
Solaris and Dragon Skin 10 NV. These are monochromatic effective properties,
not a material hierarchy: Solaris refractive index is manufacturer data, while
Dragon Skin refractive index and every extinction coefficient are explicit
literature/modeling priors rather than measured product calibration.

`FingertipParameters.mechanics` is one immutable `SiliconeMechanics` value.
The default passes `1070 kg/m³`, a `1.06e5 Pa` shear modulus, a
`1.0494e7 Pa` first Lamé parameter, and `10 Pa·s` damping directly to
Newton's damped Neo-Hookean tetrahedra. It is not a hereditary viscoelastic
model and contains no Maxwell, Prony, or relaxation state. The damping value
remains an uncalibrated numerical input rather than a datasheet measurement.
The type and its single `silicone` preset live in `mechanical_param.py`.

`Fingertip` constructs the analytic fingertip assembly. Its `tip_z_m` property
exposes the reference silicone tip coordinate in Newton-compatible metres so
callers do not repeat the semiellipse endpoint calculation and unit conversion.

`Silicone` and `Carrier` are constructed geometry objects, not separate
parameter systems. `BondingInterface` is the derived pair of left and right
carrier-silicone polylines that receive the perfect kinematic bond. It does not
own independent physical parameters and cannot be overridden by callers;
`Fingertip` always derives it from the actual shared geometry.

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

The existing `make_fingertip_mesh()` path remains the 11 mm representative
single-section mesh. `make_fingertip_5led_mesh()` is the separate simplified
full-finger path and returns a `Fingertip5LEDMesh`, which remains directly
consumable anywhere a `FingertipMesh` is accepted. It adds only the fixed
longitudinal metadata required by the physical layout: five LED reference
centers, the active-section bounds, the total bounds, and inter-LED midpoints.

The full-finger longitudinal construction is:

```text
Y = [-27.5, +27.5] mm  55 mm active section
    continuous silicone outer body with the existing XZ cutout
    continuous rigid carrier/stem rail
    LED centers at [-22, -11, 0, +11, +22] mm
    one 5.1 mm-wide, 0.19 mm-deep stem recess at each LED

Y = [+27.5, +32.5] mm  5 mm distal solid-silicone end-cap
    silicone fills the complete section below the dorsal carrier plate
    no stem/void subtraction
    rigid dorsal plate continues over the cap like a fingernail
```

Gmsh fuses the cutout active volume and solid distal volume before meshing and
requires exactly one silicone volume. The proximal cavity remains open. The
carrier stem rail ends with the 55 mm active section, while only its dorsal
plate extends across the 5 mm solid end-cap. Silicone vertices under that
distal dorsal plate belong to the carrier's perfect kinematic bond. The
single-section collision proxy still puts closure caps outside its silicone
slice; the full-finger proxy instead ends with the physical 55 mm stem rail.
The five recesses are present in both the visible carrier and its Newton
collision proxy. With nominal `void_height_mm=0`, silicone keeps the existing
stem-bottom plane while each LED emitting top lies on its recess floor,
producing a geometry-derived 0.19 mm unloaded air cavity. No optical offset or
displaced silicone surface manufactures that gap. `void_height_mm` remains
fixed at zero for the initial full-finger morphology study; the hardware
recess, not `void_height_mm`, owns this interface dimension. The local XZ
morphology and the constructed 30 mm height contract are otherwise unchanged.

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

`LumoSimulation` also owns the accelerated GPU-resident checkpoint path used
by the production full-finger evaluator. The first tick remains uncaptured so
Newton can finish lazy full-surface contact-state and rigid-history allocation.
After that warm-up, the runtime verifies the fixed contact capacities and
captures two even-length graphs for the state ping-pong parities:

```text
graph_A: prescribed motion -> A-to-B physics -> wrench -> threshold checkpoint
         prescribed motion -> B-to-A physics -> wrench -> threshold checkpoint

graph_B: prescribed motion -> B-to-A physics -> wrench -> threshold checkpoint
         prescribed motion -> A-to-B physics -> wrench -> threshold checkpoint
```

The production mode applies the constant positive approach speed, kinematic
indenter pose, collision, complete fixed-iteration VBD solve, proxy wrench
harvest, ordered threshold test, and target transition on the device. It has no
force feedback and no dwell counter. Ten graph replays advance twenty physics
ticks before one coarse host status readback. At the first tick whose measured
reaction force meets or exceeds the current threshold, device kernels copy the
particle state and full soft-contact record into that threshold's exact slot;
the synchronous evaluator callback later inspects that saved tick, not a newer
live state.

`evaluate_full_finger()` selects this GPU-resident path by default after the
direct-times-five versus graph-times-five scientific-equivalence gate passed.
The direct backend remains explicit through `use_cuda_graph=False` for
conservative reference and debugging. Bitwise equality is not an acceptance
contract because Newton full-surface contact emission and wrench accumulation
are intrinsically atomic-order nondeterministic after contact onset. The gate
instead checks force/checkpoint meaning, deformation, canonical patch support,
contact objectives, finite-area optical response, inversion, and contact-buffer
safety. A graph-captured fingertip pose is immutable after capture; callers
that need to move the whole fingertip must use the direct path.

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
`approach_speed_m_s`. The pose increment is derived directly from that speed
and the simulation timestep.

For the concrete multi-force sensing evaluation, a study may use one strictly
increasing force-target tuple. The same trial runtime then advances through
each target without resetting Newton state. The production evaluator captures
the first measured state at or above each of `5, 10, 15, 20 N`; target
tolerance is not an acceptance condition. The inspection callback receives
that exact state, either live in direct mode or from its device checkpoint slot
in accelerated mode.

The production first-crossing path is simply:

```text
move at fixed physical speed
        ↓
one LumoSimulation step and reaction measurement
        ↓
reaction force >= current ordered threshold?
        ↓ yes
copy that tick immediately, then continue toward the next threshold
```

There is no force feedback, pause, pose correction, force-band dwell, or
settled-state claim. The saved states are dynamic threshold-crossing snapshots.

With `use_cuda_graph=True`, `DesignStudy` configures production fixed motion
and ordered first-crossing thresholds in `LumoSimulation` device buffers. It
polls only every twenty physics ticks and invokes the unchanged callback
against exact device-saved checkpoint states. `use_cuda_graph=True` is the
runtime and study default; `False` remains an explicit reference/debug path.

Only `LumoSimulation` mutates Newton state or simulation time. Trials never
share mutable Newton state. For the production full-finger path, `DesignStudy`
may execute up to four independent trials concurrently in one Python process
and CUDA context. Each world owns its state pair, solver, contacts,
motion/checkpoint buffers, graphs, and CUDA stream; worlds with one sphere
diameter share only the finalized immutable model and coloring. Checkpoint
callbacks remain synchronous after the corresponding stream reaches an exact
device-saved checkpoint. Serial execution remains available by setting
`parallel_world_count=1`; no generic simulation manager was introduced.

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

`emit_from_stem_boundary()` places an `LED` emission on the resolved carrier
stem or recess boundary using that same OTK spawn contract. `source_inside_silicone()`
then queries the current silicone surface to select the initial medium. These
operations remain source-local primitives: a caller that needs several LEDs
traces their independent linear contributions and sums modeled power without a
multi-LED scene abstraction.

`side_view_observation()` in `observation.py` reduces escaped paths from the one
current optical-cell LED to one raw four-quadrant response. It keeps only rays
traveling toward `+Y` and bins their power by the escape origin in the X-Z
cross section. Quadrants are ordered upper-right, upper-left, lower-left,
lower-right around the current analytic silicone semiellipse center. In the
full 60 mm finger, `+Y` is an end-facing view because Y is longitudinal; this
four-bin reducer remains only for the representative single-section studies.

`longitudinal_side_view_observation()` is the full-finger receiver. It selects
escaped power traveling toward the canonical camera-facing `+X` side and sums
all simultaneously active LEDs into eleven fixed 5 mm bins over the 55 mm
active Y range. Thus its image coordinate is longitudinal Y and it does not use
hidden emitter identity. It is still a directional surface-power observation,
not a finite camera aperture, projection plane, lens, or pixel model. The
current production call retains hard histogram bins. The same reducer exposes
an explicit linear cloud-in-cell option for optical-model validation: power is
split between neighboring bin centers, end half-bins accumulate into their
nearest bin, and active-ROI power remains exactly conserved. This option has no
PSF width or calibrated smoothing parameter.

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

`evaluator.py` owns the production mechanics-to-optics orchestration. Its
`evaluate_full_finger()` entry builds one `Fingertip5LEDMesh` and one
`OptixScene`, generates deterministic samples and traces the undeformed state
once for each of the five LEDs, then evaluates the Cartesian product of
explicit sphere diameters and longitudinal contact-Y locations. Each location
uses an independent Newton runtime; increasing force checkpoints within that
scenario reuse the runtime. Each live checkpoint updates the same silicone GAS
and IAS and is traced with the same emission and bounce samples.

The production evaluator defaults to four independent CUDA-stream Newton
worlds. OptiX checkpoint consumption remains serialized through the one shared
scene.

`FullFingerEvaluation` is a raw-data result, not an objective result. It records
the explicit mechanics backend, checkpoint step indices, graph replay counts,
checkpoint host intervention/synchronization counts, and average ticks per
host intervention. It keeps
per-emitter longitudinal responses shaped `(scenario, force, 5, 11)`,
per-emitter energy ledgers, and the shared five-emitter no-contact state.
Simultaneous 11-bin responses are derived by summing the emitter axis and are
never the sole stored representation. It also persists active-ROI power,
outside-ROI power, total `+X` visible power, and outside-ROI fractions so every
state satisfies `sum(11 bins) + outside ROI = total +X visible power` without
changing transport or adding an objective penalty. Mechanics data includes force,
indentation, checkpoint time, maximum/mean/RMS/P95 particle speed, kinetic
energy, force overshoot, reaction-force rate, indentation rate, contact counts
and buffer overflow, minimum
tet determinant and inversion count, contact centroids, local particle
indices, barycentric coordinates, reconstructed world contact points, contact
normals, body contact positions, and deformed silicone vertices. Variable-size
contact records use one flat array plus explicit `(start, count)` offsets per
checkpoint. Newton, OptiX, and escaped-ray runtime buffers are still released
after evaluation.

The fixed current mechanics contract is
`100 Hz`, `10` VBD iterations, equal `ke=3e4 N/m` endpoints,
`kd=0.282280175 N s/m`, and a monotonic `5 mm/s` approach. Each ordered force
threshold is copied on the first Newton tick at or above it, with no force
feedback and zero dwell. At 100 Hz the prescribed displacement is
`50 um/tick`. These outputs are threshold-crossing states, not quasi-static
equilibria. Optical transport
uses a uniform finite-area `1.8 x 1.6 mm` LED emitting window, `65,536` paths
per LED, `24` bounces, and hard 11-bin observation. This evaluator does not perform
morphology optimization. `objective.py` is the pure numerical reduction layer
between this rich result and Ax.

Historical force-servo, adaptive-settling, and force-ramp paths are not part of
the production API. Focused fixed-pose studies may still use `LumoSimulation`
directly when the hold itself is the validation question.

Each full-finger LED remains on its carrier recess floor. The nominal 0.19 mm
hardware cavity places air between that source and unloaded silicone even when
`void_height_mm=0`; transport starts in air and lets OptiX resolve silicone
entry, carrier reflection, or escape. Loaded Newton geometry may close that
explicit cavity and change the initial medium without a source epsilon or
per-state gap adjustment.

`validation/optomech/optical_observation_model_sensitivity.py` replays the
frozen dwell and ramp vertices without rerunning Newton. It compares the
historical point source and hard bins against linear splatting and a uniform
finite source over the manufacturer's `1.8 x 1.6 mm` water-clear resin window.
That validation selected finite-area origins and per-ray initial media for
production while retaining hard bins. Ballistic transport, ray count, bounce count, and
`J_obs` are unchanged in this comparison.

The next full-finger discrete search contract fixes both
`flat_pad_width_mm=30` and `void_height_mm=0`. It exposes five geometry
dimensions--flat height, semiellipse height, stem width, stem height, and void
width--as integer half-millimeter steps. The 0.19 mm LED air cavity remains a
fixed carrier-recess feature and is not a design variable. The complete physical
height runs from the carrier top at `+10 mm` to the silicone ellipse tip at
`-flat_pad_height_mm-semiellipse_height_mm`. `Fingertip.full_height_mm` derives
that extent from the constructed geometry, and `DesignSpace` authoritatively
requires it to be at most `30 mm`. Ax equivalently enforces
`flat_pad_height_step + semiellipse_height_step <= 40`; the existing
`FingertipGeometry` and `DesignSpace` checks remain the sole owners of nonlinear
geometry validity.
`scripts/run_mobo.py` records the intended new full-finger campaign inputs,
including explicit sphere diameters and longitudinal contact locations.
It exposes separate mechanics and optical presets, physical parameter
bounds, indenter URDF list, sequential force thresholds, initial clearance,
output directory, and cumulative target morphology count. Mechanics and optical algorithms remain owned by their
production modules. `silicone` is the current mechanics preset.
Optical selection independently exposes the existing Solaris and Dragon Skin
10 NV low/nominal/high sensitivity presets, without claiming that Solaris uses
Dragon Skin mechanics. The evaluator pairs each sphere URDF with an explicit
diameter and places its center one radius plus the configured clearance below
the undeformed pad. The entry script supplies its sibling
`optix-toolkit/ShaderUtil/include` as the default `OTK_INCLUDE_DIR`; an
explicitly exported environment value still takes precedence.
The prepared production entry selects nominal Dragon Skin 10 NV optics, uses
the accepted four-world evaluator explicitly, and targets 120 cumulative
successful morphologies in the fresh
`mobo_full_finger_instantaneous_05mm` directory. Its
run config records the finite `1.8 x 1.6 mm` package-window source, the
four-world backend, and dependency/source hashes. Resume is refused if the
scientific source, optimizer source, dependency versions, or serialized
scientific contract differ.
`validation/optomech/mobo_smoke.py` reads those production entry settings
without changing `scripts/run_mobo.py`, targets one successful morphology in a
fresh timestamped validation directory, and invokes the same Ax loop. It then
reopens that directory through the public resume path and verifies that the
completed trial is neither lost nor repeated, without touching the
120-morphology campaign directory.

The previous continuous and six-dimensional discrete Ax campaigns remain
historical artifacts and cannot be resumed through the production entry point.
The corrected campaign uses a new output directory, run-config schema, and Ax
state, so old `J_intensity/J_spatial` observations cannot be attached to the
five-dimensional full-finger experiment.

`objective.py` owns the frozen production reductions. For every mechanical
scenario it computes finite 5 N patch formation, reference-area-weighted
5-to-20 N Lagrangian patch IoU, and progressive stiffening, then defines
`J_contact` as the minimum scenario-wise geometric mean of those three terms.
The mean-normal score remains diagnostic only. `J_obs` first sums the LED axis,
subtracts the no-contact simultaneous field, divides by total emitted power
five, and takes the minimum 11D Euclidean separation over sphere diameter,
force threshold, and distinct contact-Y pairs. Contact onset is diagnostic only; force
variation is not penalized. Both reducers accept saved raw NPZ arrays and know
nothing about Newton, OptiX, or Ax.

The production Ax contract evaluates the exact Cartesian product of sphere
diameters `5/10/20 mm` and contact Y positions
`[-22,-11,-5.5,0,5.5,11,22] mm`, with sequential instantaneous
`5/10/15/20 N` threshold snapshots.
It maximizes only `J_contact` and `J_obs`, without scalarization or objective
thresholds. Trial NPZ files retain raw mechanics, optical responses, component
scores, limiting conditions, same-force separation matrices, onset and ROI
diagnostics. The focused four-world first-crossing validation passed with 16
saved checkpoints, no inversion or contact-buffer overflow, and closed optical
energy. The previous 21-scenario dwell artifact is historical and is not
reused as an observation in this fresh Ax campaign.

`validation/optomech/objective_prototype.py` is the read-only numerical
prototype for those equations. It reconstructs a Lagrangian contact patch from
the saved vertex/edge/triangle Newton records, evaluates the proposed contact
components, and evaluates threshold-conditioned contact-location separation using
the simultaneous +X 11-bin longitudinal response. `J_obs` is the worst
same-threshold location separation. Contact-onset distance and within-location
force variation remain diagnostics because QDD proprioception owns contact
detection and force magnitude. Old +Y Q and labeled per-emitter responses are
diagnostic only. The script does not register an objective or call Newton,
OptiX, or Ax.

`validation/optomech/full_finger_spatial_observation.py` is the transitional
optical-only reprojection for the saved pre-camera-axis nominal artifact. It
reuses its Newton vertex checkpoints, applies the production 65,536-ray,
24-bounce samples, verifies the saved energy ledgers are unchanged, and stores
the +X per-emitter 5x11 responses. It does not run Newton. New calls to
`evaluate_full_finger()` already produce this spatial representation directly.

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
from the local analytic semiellipse height at that X location. Each simulation
moves at its prescribed physical speed and captures the first `20 N` threshold
crossing. It then checks contact, finite silicone state, perfect-bond drift,
and carrier penetration before releasing that runtime.

`validation/contact-physics/sphere_15mm_viewer.py` is a focused interactive
contact diagnostic. It loads the packaged 15 mm sphere, advances that one
kinematic body at a fixed positive-Z speed, renders every Newton state and
contact set, and holds the sphere pose fixed for `10 s` after the transient
reaction first reaches `20 N`. It continues advancing and rendering Newton
during that hold, reports force, active-particle speed, and sphere contact count
at a throttled interval, and freezes the final held state until the viewer
closes. It is a fixed-speed diagnostic and does not run the production force
checkpoint workflow.

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
Y = fingertip longitudinal / extrusion direction
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
