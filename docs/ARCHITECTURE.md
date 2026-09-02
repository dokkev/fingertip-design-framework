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
IndentationStudy
        ↓
four independent LumoSimulation worlds
        └── internally: FingertipMesh → Newton model
```

Later validated mechanics results may be consumed by:

```text
      OptiX
        ↓
   optimization
```

Prepared simulation, optimization, and experimental arrays may also be passed
to `lumo.visualization` for standalone publication panels or composed figures.
The visualization layer does not load scientific artifacts or own simulation
policy.

The live physical localization path is independent of simulation:

```text
RealSenseColorCamera
        ↓ owned RGB frame
experiments.localization
        ↓ LED geometry + contact estimate
scripts/live_contact_localization.py
        ↓ OpenCV display only
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
The generic `silicone` preset preserves this baseline. The `solaris` preset
uses the manufacturer's cured density of `990 kg/m³` and a `9.85e4 Pa` shear
modulus inferred from the reported `25 psi` stress at 100% elongation under an
incompressible Neo-Hookean uniaxial assumption. Its `9.75e6 Pa` first Lamé
parameter preserves the same numerical Poisson ratio of approximately `0.495`.
Solaris damping remains the same uncalibrated `10 Pa·s`; the datasheet's
uncured liquid viscosity is not a solid damping measurement. Both presets live
in `mechanical_param.py`.

`Fingertip` constructs the analytic fingertip assembly. Its `tip_z_m` property
exposes the reference silicone tip coordinate in Newton-compatible metres.
`led_source_centers_m` exposes the five physical LED source-plane centers from
the fixed longitudinal layout and carrier recess geometry. Neither property
depends on mesh resolution.

`layout.py` owns the fixed longitudinal hardware definition: the five LED
centers, 55 mm active-section bounds, 5 mm distal end-cap, and physical LED
recess dimensions. These are properties of the current fingertip hardware,
not mesh-resolution parameters. Downstream mesh, observation, and campaign
code consume this one definition.

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
interior, and its longitudinal caps close the physical 55 mm rail at its
proximal and distal ends so Newton's signed mesh query remains well-defined.
The proxy has a cached volume SDF and uses Newton's full-surface rigid/soft
contact path, which catches tetrahedral faces that would otherwise pass
between particle vertices.

`bonded_vertex_indices` identifies silicone vertices lying on
`Fingertip.bonding_interface`. The mesh layer consumes its left and right
polylines directly rather than deriving bond ownership from `Silicone`.
`FingertipMesh` requires this array to be nonempty and nonnegative, normalizes
it to unique indices, and verifies the silicone vertex range before Newton
receives it.

`make_fingertip_mesh()` is the only public construction path and always
discretizes the complete current 60 mm, five-LED fingertip. Physical LED source
centers and fixed active/total Y bounds remain owned by `lumo.fingertip`; they
are not repeated as mutable-looking mesh fields. Representative 11 mm slices
are not production objects or design parameters.

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
collision proxy ends with the physical 55 mm stem rail. The five recesses are
present in both the visible carrier and its Newton collision proxy. The
silicone cavity bottom is defined directly by `stem_height_mm`, while each LED
emitting top lies on its recess floor, producing a geometry-derived 0.19 mm
unloaded air cavity. No optical offset or displaced silicone surface
manufactures that gap. The hardware recess is the only owner of this interface
dimension. The local XZ morphology and the constructed 30 mm height contract
are otherwise unchanged.

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
- kinematic rigid indenters created from packaged URDF assets;
- Newton-specific model and object identities consumed by the runtime.

Use Newton's public API directly whenever practical.

The full carrier shape has collision disabled. The collision proxy is attached
to the same kinematic body with its rigid-shape pairs explicitly filtered and
silicone particle/full-surface collision enabled. Bonded particles remain
inactive, so the bond interface is controlled only by the fixed identity
boundary. Silicone particle radius is explicitly zero at model construction;
contact detection distance is supplied separately by `LumoSimulation` through
the collision pipeline. The rigid carrier proxy uses a stiff shape-contact
material while VBD still permits a small, measured penalty penetration. Its
normal contact stiffness is fixed at `1e6 N/m` by the Newton model. It is not a
runtime or indentation-study input.

`build_fingertip_newton_model()` defaults to this production contact model. It
also exposes two concrete structural modes used only by the hybrid-mechanics
ablation: `bonded` keeps the visible carrier, disables carrier penalty contact,
and accepts the exact fixed silicone-interface indices; `absent` omits carrier
shapes while retaining the kinematic identity anchor required by the shared
fixed-boundary update. These modes reuse Newton model construction and do not
create an alternate simulation runtime or loading protocol.

`Indenter.add_urdf()` accepts optional normal-contact stiffness and damping
overrides. Its default `None` values preserve Newton's shape material because
no indenter contact pair has been frozen as a production numerical contract.
URDF construction applies requested values only while the importer creates
that asset and then restores the builder defaults, so objects added later are
unchanged.

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

The runtime defaults to a `1 mm` mesh element size, `10` SolverVBD iterations,
and a `1e-4 m` soft-contact margin. Simulation frequency is an explicit
required input; the production fingertip evaluator uses `100 Hz`. The
Newton-owned carrier contact stiffness is fixed at `1e6 N/m`.
Optional `soft_contact_stiffness_n_m` and `soft_contact_damping_n_s_m` values
support the focused rigid-soft pair study; `None` preserves Newton's model
defaults. There is no simulation-configuration abstraction.

`LumoSimulation(fingertip, builder=...)` is the high-level construction entry
point. It meshes the fingertip, adds it to an optional caller-populated Newton
builder, and finalizes the one shared Newton model. A caller adds external
objects such as an `Indenter` to that builder before constructing the
simulation. An optional caller-built `fingertip_mesh` lets the end-to-end
evaluator give Newton and OptiX the same shared discretization without
meshing twice. Lower-level mesh and Newton-model construction functions remain
available to validations that inspect those stages directly.

The current step order is:

```text
caller optionally updates a kinematic indenter pose
    ↓
reapply the fixed identity carrier pose and kinematic bond
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

`LumoSimulation` also owns the private GPU-resident checkpoint machinery used
by `IndentationStudy`. The first tick remains uncaptured so
Newton can finish lazy full-surface contact-state and rigid-history allocation.
After that warm-up, the runtime verifies the fixed contact capacities and
captures two even-length graphs for the state ping-pong parities:

```text
graph_A: prescribed motion -> A-to-B physics -> wrench -> threshold checkpoint
         prescribed motion -> B-to-A physics -> wrench -> threshold checkpoint

graph_B: prescribed motion -> B-to-A physics -> wrench -> threshold checkpoint
         prescribed motion -> A-to-B physics -> wrench -> threshold checkpoint
```

The checkpoint machinery applies the constant positive approach speed, kinematic
indenter pose, collision, complete fixed-iteration VBD solve, proxy wrench
harvest, ordered threshold test, and target transition on the device. It has no
force feedback and no dwell counter. Ten graph replays advance twenty physics
ticks before one coarse host status readback. At the first tick whose measured
reaction force meets or exceeds the current threshold, device kernels copy the
particle state and full soft-contact record into that threshold's exact slot;
the synchronous evaluator callback later inspects that saved tick, not a newer
live state.

`evaluate_fingertip()` always uses this path. CUDA graph selection, finalized
model reuse, runtime reset reuse, and world count are not public execution
modes. Procedural validations that need an uncaptured reference use
`LumoSimulation.step()` directly instead of adding a second indentation
protocol to production.

`LumoSimulation.step()` does not own approach trajectories, force thresholds,
validation policy, or result reporting. It is the only production API that
advances Newton state, the global step count, or simulation time.
Construction populates the initial contact buffer once, and each global step
refreshes it before SolverVBD. `soft_contact_count()` exposes total or
body-specific counts without leaking Newton contact-array indexing into
callers.
Callers update other kinematic objects as needed before one tick and may query
the resulting indenter reaction force or maximum active silicone-particle
speed afterward. These observations reduce into preallocated scalar device
buffers rather than cloning full velocity or wrench arrays on every tick. The
runtime may later orchestrate optical work, but no ray-tracing behavior is part
of the current runtime.

#### `indentation.py`

`IndentationTrial` is one indentation scenario plus lightweight scalar and pose
results for the current threshold callback. Its nonnegative initial-clearance
scalar lets consumers report physical indentation rather than total approach
travel. It never retains its
`LumoSimulation` or `Indenter`.
`IndentationStudy` owns one immutable analytic `Fingertip`, an ordered tuple of
indentation trials, and the one strictly increasing force schedule. It
constructs a fresh builder, indenter,
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

The same trial runtime advances through the study's four force targets without
resetting Newton state. The production evaluator captures the first measured
state at or above each configured force threshold; target tolerance is not an
acceptance condition. Exactly four strictly increasing thresholds are required
by the production objective. The inspection callback receives that exact state
from its device checkpoint slot.

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

Only `LumoSimulation` mutates Newton state or simulation time. Trials never
share mutable Newton state. `IndentationStudy` executes batches of up to four
independent trials in one Python process and CUDA context. Each world owns its
state pair, solver, contacts,
motion/checkpoint buffers, graphs, and CUDA stream; worlds with one sphere
diameter share only the finalized immutable model and coloring. Checkpoint
callbacks remain synchronous after the corresponding stream reaches an exact
device-saved checkpoint. The four-world batch size and model sharing are fixed
implementation details, not user-facing simulation policy.

### `lumo/ray_tracing/`

Owns LUMO-specific optical transport behavior.

OptiX is the ray-tracing backend.

`OptixScene` is the first concrete OptiX 9.1 runtime component. It owns one
persistent CUDA stream, the OptiX context and pipeline resources, device
geometry buffers, triangle GASes for silicone and the production carrier, and
the IAS containing those instances. Silicone/carrier instance IDs and visibility
masks are fixed backend details owned inside `scene.py`; callers construct the
scene from one reference `FingertipMesh` and do not allocate IAS identities.
The default always includes the production carrier. The controlled Soft-only
optical ablation passes `include_carrier=False`, which omits the carrier GAS/IAS
instance and retains the complete silicone boundary instead of removing faces
normally hidden by the carrier. This is a structural validation switch, not an
alternate transport model; materials, masks, rays, and transport remain the
production path.
Its only query `trace_closest()` returns hit
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

The finalized paper Figure 3 study samples ten deterministic void-width-covering
designs from each of the current Dragon Skin normal, Dragon Skin angled,
Solaris normal, and Solaris angled campaigns. Within each campaign it selects
without replacement the lowest-trial-ID valid design nearest each empirical
`w_void` quantile at 0/12.5/25/37.5/50/62.5/75/87.5/100%, then adds the balanced
trial and deduplicates. Campaign type is provenance only: every sample is
re-evaluated under the same 20 mm sphere, Y=-5.5 mm, theta=0 deg contact and
production 1/2/5/10 N loading path. Each morphology produces the three named
conditions Soft-only, No-void carrier, and LUMO morphology. Natural zero-void
designs reuse the identical No-void carrier/LUMO morphology state as a
consistency check. The exact saved
Newton meshes are replayed through OptiX with one common finite-area five-LED
emission array and deterministic branch samples per base design. Because this
controlled study has only one contact-Y location, production `J_obs` is
undefined: it reports normalized state-change `D(F)`, visible power, channels,
and energy data without inventing an ablation objective.

The main-paper Figure 3 visualization intentionally removes material and source
campaign encoding from its three ablation panels. It presents the 40
morphologies as one controlled paired sample: a structural counterfactual
schematic, a carrier identity comparison, and a coupled scatter of the
lateral-void changes in fixed-scenario `J_contact` and `D(1 N)`. `D(1 N)`
remains a low-load optical state-change diagnostic, not `J_obs`.

A right-side fourth panel reads the four completed 160-observation BO datasets directly
from their saved trial tables. The standard and orientation-robust datasets use
different objective domains, while Dragon Skin and Solaris use different
mechanics and optical presets, so their raw objective values are not merged
into one quantitative Pareto cloud. Figure 3 instead shows four separately
scaled empirical Pareto small multiples. The composition recomputes
non-dominance, requires exact agreement with both stored `is_pareto` flags and
`pareto.csv`, and recomputes each post-hoc equal-relative-performance balanced
trial. Pareto solutions share the same blue-to-purple `w_void` encoding as the
ablation void-effect panel; all other evaluations remain neutral gray. The
final composition uses two visually bounded super-panels. Paired structural
analysis groups the structural counterfactual with stacked carrier and
lateral-void evidence; morphology optimization groups the 2 x 2 Pareto block.
Within them, all six quantitative axes occupy one shared 2 x 3 board and
therefore share exact row, column, and physical-size alignment. One `w_void`
color scale beside the coupled-void panel applies to both its scatter and the
optimized Pareto points.

Reusable axes-only panel functions live in `lumo.visualization.ablation`;
`figures/fig3.py` alone loads the completed ablation and optimization artifacts,
validates their contracts, composes the nested GridSpec, writes the validation
summary, and exports the paper artifacts.

The older single-design Soft-only/Bonded-T/LUMO study remains a supplementary
mechanistic diagnostic. Bonded-T changes the carrier-silicone boundary
condition and is deliberately excluded from the primary void-width
counterfactual, which keeps production carrier contact and changes geometry
only.

The same procedural study also performs a controlled effective-gap optical
sensitivity for the LUMO structure. It holds the nominal saved silicone states
fixed and rebuilds only the carrier recess floor and its coincident LED source
plane at `0.01`, `0.19`, and `0.50 mm`. This is deliberately not a production
geometry option, mechanics re-evaluation, or BO dimension: `h_void` remains
absent/fixed zero, void width remains the search variable, and `0.19 mm` remains
the fabrication-informed production recess depth. The comparison isolates how
that physical gap changes the source/interface boundary condition. It stores
raw 5-by-11 response, energy, visible-power, outside-ROI, and source-medium
diagnostics for every fixed Newton state. `J_obs` remains undefined for this
one-location sensitivity as well.

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

`LED` combines one `LEDParameters` with a world-frame position and unit normal.
`emit_from_stem_window()` is the sole public source operation: it samples
Lambertian directions and distributes origins uniformly across the physical
emitting window before resolving OTK-safe origins on the carrier recess floor.
It does not copy or independently own hardware or optical parameter values. The current
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

`PathTraceResult` lives beside `trace_bounded_paths()` in `path.py`. It owns the
escaped-ray array, explicit scalar power ledger, and an optional finite-segment
array requested with `record_segments=True`. Each recorded segment contains its
endpoints and start/end power after the same Beer-Lambert transport used by the
ledger. Ordinary evaluation leaves segment recording disabled and allocates no
segment history. The result does not expose a string-keyed statistics
dictionary.
OptiX hit layouts remain next to their CUDA decoding in `scene.py`, while the
short-lived vectorized Fresnel and Lambertian result layouts remain next to the
numerical operations in `transport.py`.

`safe_secondary_origins()` selects the OTK front or back spawn position by the
sign of the outgoing direction dotted with `normal_W`. It does not infer media
or trace a ray. Both focused single-event validations and the bounded path loop
use this operation for every triangle departure. OptiX traversal uses `tmin=0`:
the OTK origin owns self-intersection separation, so no second scene epsilon is
combined with the official offset.

`sources_inside_silicone()` queries the current silicone surface independently
for every finite-window origin and selects each path's initial medium. A caller
with several LEDs traces their independent linear contributions and sums
modeled power without a multi-LED scene abstraction.

`longitudinal_side_view_power()` is the sole observation reducer. It selects
escaped power traveling toward the canonical camera-facing `+X` side and
accumulates hard histogram power into eleven fixed 5 mm bins over the 55 mm
active Y range. It also returns outside-ROI and total visible-side power for
accounting. Thus its image coordinate is longitudinal Y and it does not use
hidden emitter identity. It is a directional surface-power observation, not a
finite camera aperture, projection plane, lens, or pixel model.

The package root exports only the complete scene/source/path/observation flow.
Low-level Fresnel, Lambertian, and OTK spawn operations remain available from
their owning `transport.py` and `scene.py` modules for focused validation; they
are not alternate production pipelines.

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

Current responsibilities are the concrete five-dimensional design space, one
mechanics-to-optics evaluation, pure objective reductions, the sequential Ax
loop, and durable campaign persistence.

`evaluator.py` owns the production mechanics-to-optics orchestration. Its
`evaluate_fingertip()` builds one canonical `FingertipMesh` and one
`OptixScene`, generates deterministic samples and traces the undeformed state
once for each of the five LEDs, then evaluates the Cartesian product of
explicit sphere diameters and longitudinal contact-Y locations. Each location
uses an independent Newton runtime; increasing force checkpoints within that
scenario reuse the runtime. Each live checkpoint updates the same silicone GAS
and IAS and is traced with the same emission and bounce samples.

The evaluator has one production execution path. `IndentationStudy` owns its
fixed internal GPU execution policy; the evaluator does not expose CUDA graph,
model reuse, runtime reuse, or world-count choices. OptiX checkpoint consumption
remains serialized through the one shared scene.

`FingertipEvaluation` is a raw-data result, not an objective result. It keeps
checkpoint step indices,
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

Each fingertip LED remains on its carrier recess floor. The nominal 0.19 mm
hardware cavity places air between that source and unloaded silicone;
transport starts in air and lets OptiX resolve silicone entry, carrier
reflection, or escape. Loaded Newton geometry may close that
explicit cavity and change the initial medium without a source epsilon or
per-state gap adjustment.

`validation/optomech/optical_observation_model_sensitivity.py` replays the
frozen dwell and ramp vertices without rerunning Newton. It compares the
historical point source and hard bins against linear splatting and a uniform
finite source over the manufacturer's `1.8 x 1.6 mm` water-clear resin window.
Historical point emission and linear splatting are implemented locally in that
procedural validation rather than exposed as production ray-tracing options.
The result selected finite-area origins and per-ray initial media for production
while retaining hard bins. Ballistic transport, ray count, bounce count, and
`J_obs` are unchanged in this comparison.

The discrete search contract fixes `flat_pad_width_mm=30` and
exposes five geometry dimensions--flat height, semiellipse height, stem width,
stem height, and void width--as integer half-millimeter steps. The 0.19 mm LED
air cavity remains a fixed carrier-recess feature and is not a design variable.
The complete physical height runs from the carrier top at `+10 mm` to the
silicone ellipse tip at
`-flat_pad_height_mm-semiellipse_height_mm`. `Fingertip.full_height_mm` derives
that extent from the constructed geometry, and `DesignSpace` authoritatively
requires it to be at most `30 mm`. Ax equivalently enforces
`flat_pad_height_step + semiellipse_height_step <= 40`. It also enforces the
exact lattice form of the bonded-side width requirement,
`stem_width_step + 2*void_width_step <= 39`. The existing
`DesignSpace` contains only the current five geometry bounds and fixed base
`FingertipParameters`. Constructing `FingertipGeometry`/`Fingertip` owns
nonlinear geometry validity; `DesignSpace` additionally owns the 30 mm complete
height and 5 mm minimum-silicone-thickness limits. The two Ax constraints are
proposal-time mirrors, not independent scientific owners.

Candidate generation derives a deterministic finite subset of the 0.5 mm
lattice by retaining only points accepted by `DesignSpace.is_feasible()`. A
fresh production campaign first attaches the five explicit
`INITIAL_MORPHOLOGIES_MM` designs in their listed order and evaluates them under
the current 75-scenario environment; no objective value from an older campaign
is imported. Ax 1.3.1 counts these completed manual trials toward the fixed
initialization budget of 13, so one exact-feasible Sobol point is then attached
at a time until eight fresh Sobol observations complete initialization. During
model-based generation, Ax fits its normal multi-objective surrogate and scores
a fresh pool of 256 exact-feasible Sobol points through its public acquisition
evaluation API. Only the best feasible point is attached as an Ax trial. Thus
the finite-pool acquisition search is approximate, but analytical validity is
exact and analytically invalid morphologies are never sent to Newton or stored
as abandoned Ax proposals. The pool seed is derived from the fixed campaign
seed and persisted trial count, so interruption and resume reproduce the same
proposal sequence.
`scripts/run_mobo.py` is the only executable campaign entry. It records the
current fingertip campaign inputs,
including explicit sphere diameters and longitudinal contact locations.
It exposes separate mechanics and optical presets, physical parameter
bounds, the ordered informed initial morphologies, indenter URDF list,
sequential force thresholds, initial clearance, output directory, and
cumulative target morphology count. Mechanics and optical algorithms remain owned by their
production modules. `silicone` is the current mechanics preset.
Optical selection independently exposes the existing Solaris and Dragon Skin
10 NV low/nominal/high sensitivity presets, without claiming that Solaris uses
Dragon Skin mechanics. The evaluator pairs each sphere URDF with an explicit
diameter and places its center one radius plus the configured clearance below
the undeformed pad. The entry script supplies its sibling
`optix-toolkit/ShaderUtil/include` as the default `OTK_INCLUDE_DIR`; an
explicitly exported environment value still takes precedence.
The prepared production entry selects silicone mechanics and nominal Dragon
Skin 10 NV optics and targets 160 cumulative successful morphologies by
resuming `mobo_fingertip_orientation_robust_1_2_5_10_05mm`. Its
run config records the finite `1.8 x 1.6 mm` package-window source and
dependency/source hashes. Resume is refused if the
scientific source, optimizer source, dependency versions, or serialized
scientific contract differ.
`validation/optomech/mobo_smoke.py` reads those production entry settings
without changing `scripts/run_mobo.py`, targets one successful morphology in a
fresh timestamped validation directory, and invokes the same Ax loop. It then
reopens that directory through the public resume path and verifies that the
completed trial is neither lost nor repeated, without touching the
120-morphology campaign directory.

`ax_bo.py` owns only Ax search-space conversion, feasible candidate generation,
and the sequential evaluation loop. It has no module CLI, continuous-campaign
branch, historical warm-start observations, or startup self-test.
`campaign_io.py` owns run-config provenance, atomic Ax/CSV/NPZ persistence,
resume reconciliation, Pareto tables, plots, and summary output. A fresh
campaign starts with no reused objective observations: its five informed
designs are new evaluations whose `generation_node` is
`INITIAL_MORPHOLOGY`. Generated initialization and model-based trials remain
distinguishable as `FEASIBLE_Sobol` and `FEASIBLE_MBM`. The expensive one-trial
save/resume verification remains solely in
`validation/optomech/mobo_smoke.py`.

`objective.py` owns the production reductions. For every mechanical scenario
it computes finite patch formation at the second configured force,
reference-area-weighted Lagrangian patch IoU between the second and highest
forces, and progressive stiffening from the first and last force intervals.
For the production `1/2/5/10 N` thresholds, this means patch formation at
`2 N`, patch stability from `2` to `10 N`, early stiffness from `1` to `2 N`,
and late stiffness from `5` to `10 N`. It then defines
`J_contact` as the minimum scenario-wise geometric mean of those three terms.
The mean-normal score remains diagnostic only. `J_obs` first sums the LED axis,
subtracts the no-contact simultaneous field, divides by total emitted power
five, and takes the minimum 11D Euclidean separation over sphere diameter,
contact angle, force threshold, and distinct contact-Y pairs. Location pairs
are compared only within an identical angle, diameter, and force condition.
Contact onset is diagnostic only; force variation is not penalized. Both
reducers accept saved raw NPZ arrays and know nothing about Newton, OptiX, or
Ax.

The orientation-aware production Ax contract evaluates the sphere-major,
angle-middle, contact-Y-minor Cartesian product selected in
`scripts/run_mobo.py`: `10/15/20 mm` spheres, `-30/-15/0/+15/+30 deg`, and
`-11/-5.5/0/+5.5/+11 mm`, for 75 independent scenarios per morphology. Each
scenario records sequential instantaneous first-crossing snapshots at
`1/2/5/10 N`. The fingertip and carrier stay fixed in the Newton world. For
each physical `+theta` fingertip angle, evaluator scenario construction rotates
the sphere's initial center and normalized motion direction through `-theta`
about the longitudinal world-Y line at `X=0, Z=0`.
`J_contact` takes the worst case over all resulting scenarios. `J_obs` compares
contact-Y locations only within the same diameter, angle, and force condition,
then takes the worst case over those conditions.
It maximizes only `J_contact` and `J_obs`, without scalarization or objective
thresholds. Trial NPZ files retain raw mechanics, optical responses, component
scores, limiting conditions, same-force separation matrices, onset and ROI
diagnostics, and explicit `contact_angles_deg`. `run_config.json` records the
orientation axis, pivot, inverse-relative transform convention, and complete
ordered scenario support. Strict resume refuses a changed scientific contract,
source hash, or dependency version. Neither completed 160-trial pad-normal
campaign is imported into this fresh Ax campaign.

### `experiments/`

Owns physical experiment hardware and image-processing algorithms outside the
production `lumo` simulation/optimization package.

`experiments/hardware/` owns concrete physical-device I/O and device
lifecycles. The current
`RealSenseColorCamera` is a thin owner of one `pyrealsense2` color pipeline. It
configures a 1920 x 1080 RGB stream at 30 FPS by default, returns an owned
immutable RGB frame with device timestamp and frame number, and exposes explicit
`start()`/`read()`/`stop()` plus context-manager cleanup. It does not alter
exposure, gain, white balance, or their automatic controllers. RealSense objects
do not cross this package boundary. Depth
acquisition is intentionally absent because the current localization algorithm
consumes only color images.

The package does not select localization algorithms, render a GUI, write
experimental files, or hide reconnect/retry policy. A later camera backend can
provide the same small RGB-frame lifecycle without changing image localization.

`experiments/localization/` owns pure NumPy/OpenCV image analysis for the
physical fingertip. `detect_fingertip_boundary()` is a side-view geometry
stage. It weakly smooths the Lab-a and grayscale geometry channels and runs
OpenCV's standard-refinement line-segment detector on both. Segment-side
samples from Lab-a, HSV saturation, and HSV value distinguish the physical
neutral-to-cyan dorsal edge and illuminated-to-dark palmar edge from carrier
lines, internal optical streaks, and repetitive fins. Dorsal and palmar
fragments are clustered without requiring one connected component, robustly
fitted with Huber `cv2.fitLine`, and accepted only as a positive, reasonably
stable paired width. `FingertipBoundaryRegion` owns the resulting two ordered
line curves and the image mask strictly between them. Converted color channels
are geometry-only and never enter contact photometry.

`scripts/live_fingertip_boundary.py` displays the dorsal curve in magenta, the
palmar curve in yellow, the translucent fingertip search mask, fitted pad width
and dorsal support, and the existing red-detector LED centers and response
ROIs. It writes no files. Full paired-LSD geometry runs only during initial LED
acquisition, periodic no-contact re-anchoring, and recovery after tracking loss.
Normal frame updates remain the existing grayscale pyramidal-LK and rigid
similarity-fit path.

The current learning-free contact path detects the ordered five-LED array from
a median of fixed camera frames after masking its unchanged red-high-pass
detector with `FingertipBoundaryRegion.search_mask`. It then constructs
spacing-scaled regions, measures the brightest 10% red-channel
response in small polygon bounding crops, and estimates contact position from
the positive baseline-relative response weighted by the known physical LED
positions. The crop-local implementation preserves the exact brightest-10%
definition without five full-frame masks per video frame. The five
local Lucas-Kanade correspondences update the array only through one robust
translation/rotation/uniform-scale fit, so one distorted optical landmark cannot
independently move its ROI. During confirmed no-contact operation, the absolute
red detector periodically re-anchors that rigid array to limit recursive drift.
The correction is accepted only when every constrained landmark moves by at
most half the current median LED spacing, preventing a different regular
five-peak constellation from causing a detector jump.
An explicit 30-sample unloaded feature median and per-LED MAD noise scale define
the baseline and the 4-sigma contact/no-contact gate. It does not silently infer
contact from an arbitrary initial frame.

Localization receives RGB arrays and numerical calibration values. It does not
import RealSense, own a camera lifecycle, show windows, save results, or depend
on Newton, OptiX, or Ax.

`scripts/live_contact_localization.py` is the concrete online assembly. It
discards 30 warmup frames without changing the D435's default automatic
photometric controls, then collects 30 fixed-camera frames for LED geometry.
The user captures an unloaded
30-feature baseline with `b`; normal operation applies only a three-frame median
to the final feature vector before noise-gated localization. The script draws
the detected landmarks/ROIs and live contact estimate and exits on `q` or
Escape. A lost rigid-array track invalidates the view-dependent baseline and
automatically starts a new 30-frame detection; `r` starts the same geometry
process explicitly without changing camera controls.
The displayed contact marker is the positive-response-weighted point over the
five ROI centers and is absent while the measured response remains within the
unloaded-noise gate. Its response bars use a fixed z-score axis rather than
renormalizing every frame; raw DN values remain visible. The application writes
no result artifact. A frame timeout remains a reported hardware failure;
the application, not the RealSense adapter, owns a bounded ten-attempt reconnect
loop. Successful reconnect repeats the camera warmup and discards the old LED
geometry and baseline before localization resumes. Absolute camera-intensity
comparisons under fixed camera settings are a separate acquisition contract:
they require explicit
user-selected manual exposure, gain, and white-balance values held identical
across morphologies, not values inferred from a running automatic mode.

Fingertip-boundary detection runs its image geometry at half resolution for
frames taller than 720 pixels, then maps the paired boundaries and search mask
back to the camera image. Raw-red LED features are still measured from the
original RGB frame. This keeps the periodic no-contact absolute re-anchor out
of the 1080p full-frame LSD cost without changing the photometric measurement.

### `lumo/visualization/`

Owns the small reusable Matplotlib grammar for publication figures. Scientific
panel functions receive an existing `Axes` and prepared arrays, render only
their own content, and never create, display, close, or save a figure. The
initial panels cover Pareto objectives, force-displacement, contact area,
incremental stiffness, optical response, prepared images, the current
parametric fingertip X-Z cross-section, and the controlled carrier/void
ablation panels used by Figure 3. The geometry panel receives one
constructed `Fingertip`; it derives every outline and dimension from that
object and shows the fixed LED-station recess rather than duplicating geometry
constants in a figure script.

`style.py` is the sole source for final-size single- and double-column widths,
typography, line and marker dimensions, semantic colors, and design-status
markers. Green denotes optical signal, orange denotes external mechanics,
neutral milky white denotes compliant silicone, and charcoal denotes the rigid
carrier. Dragon Skin and Solaris use a purple/blue pair so material identity
does not consume the optical green channel.

`layout.py` creates uniform axes arrays or caller-owned Matplotlib `GridSpec`
layouts, adds bold panel labels, renders one panel standalone, and saves PDF,
SVG, or high-resolution PNG output. Its `add_figure_box()` helper owns the one
rounded white-panel frame used by both Figure 2 and Figure 3, including border
color, stroke, padding, and corner radius. Figure composition owns panel labels
and file output; panel functions remain unaware of both.

The package accepts already prepared numerical arrays or image arrays. It does
not contain experiment-specific paths, campaign loading, data preprocessing,
Newton, OptiX, or Ax calls. A caller may therefore use the same panels with
simulation results, experimental measurements, standalone inspection, and
multi-panel paper figures.

`figures/fig2.py` is the finalized paper Figure 2 composition using this
contract. It loads one frozen Newton state, replays the unloaded and loaded
meshes through OptiX with common deterministic samples, and composes the
existing parameterization, mechanics, optical, and Bayesian-optimization
panels. Artifact loading, transport replay, inter-panel arrows, and export
remain in the figure composition; the reusable plotting layer stays free of
simulation ownership.

One-off experimental figure scripts own their artifact discovery and figure
composition rather than adding experiment paths to `lumo.visualization`.
`figures/brightest10_red_contact_sweep.py` is one such script: it detects the
fixed-camera five-LED array, measures the brightest-10% red response in
spacing-scaled ROIs, and explicitly falls back to a median-centered exploratory
view when the experiment directory has no unloaded frame.

### `lumo/util/`

Contains only small shared helpers with clear current consumers.

Rigid indenter asset flow is:

```text
packaged URDF resource → filesystem path → Indenter.add_urdf()
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
├── Indenter.add_urdf(...)
└── LumoSimulation(fingertip, builder=builder)
        └── finalize one shared Newton model
```

Neither `LumoSimulation` nor `build_fingertip_newton_model()` chooses or places
an indenter. Both accept a caller-populated builder so all scene bodies can be
finalized into the same Newton model.

When no builder is supplied, `build_fingertip_newton_model()` creates and
finalizes a zero-gravity builder. A caller-supplied builder owns its scene
gravity; model construction does not override it.

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
fingertip and uses an `IndentationStudy` to run the packaged 5, 10, and 20 mm
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

`validation/contact-physics/angled_indentation_viewer.py` is the focused
interactive check for the existing arbitrary-direction indentation contract.
It keeps the fingertip and carrier fixed and represents a physical fingertip
rotation about the longitudinal Y datum by applying the inverse rotation to
both the sphere center and its normalized motion direction. The default
Dragon Skin trial-117 scenario advances continuously through `5 N` and `10 N`,
then renders the final pose without advancing Newton. Angles remain
scenario-construction data: `IndentationTrial` retains only its concrete world
pose and direction, while the production campaign may select angles and the
evaluator converts them into those existing fields.

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
- `lumo.visualization` must not load simulation, optimization, or experimental
  artifacts; callers prepare the arrays supplied to its panels.
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
- a general-purpose visualization framework beyond the concrete publication
  panels and composition helpers in `lumo.visualization`;
- reusable validation frameworks.

Legacy code may be consulted for scientific intent and failure history, but it
is not the architectural source of truth.

The current source tree is authoritative.
