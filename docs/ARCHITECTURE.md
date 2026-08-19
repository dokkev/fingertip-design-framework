# Architecture

## Dependency direction

```text
model -> mesh -> fem -> neutral FEA results
   \       \                     \
    \       +-------> optics ----> validation
     +---------------> optics ----> visualization
```

Dependencies flow to the right. Production packages never import `validation`
or `tests`; `model`, `mesh`, and `fem` never import Matplotlib. Kratos and
Mitsuba remain optional environment dependencies and do not leak through
neutral result objects.

## Public workflow

The framework assumes a two-dimensional fingertip cross-section. Dimension
suffixes are therefore not part of the main API. Physical data is arranged
vertically, while subsystem boundaries use a small set of verbs:

```text
Fingertip -> mesh -> deformed mesh
                 \
                  solve(...) -> FEA result
                  trace(...) -> TransportResult
                  evaluate(...) -> proxy metrics
                  FingertipCase(Fingertip, FEA2D, RayTracing2D)
                    solve() -> FEA2D.result
                    trace() -> RayTracing2D.raw + .summary
                  render(...) -> RenderResult   [optional]
```

Typical optical use reads as:

```python
tip = Fingertip(parameters)
mesh = tip.mesh()

fea = solve(tip, mesh, indentation=1.5)
reference = trace(tip, mesh)
loaded = trace(tip, fea.deformed_mesh)
metrics = evaluate(reference, loaded)
plot_transport(loaded)
```

`trace(tip)` is also supported as a dependency-light analytic no-load preview.
It does not replace mesh-based reference/loaded comparisons.

## Ownership

- `model/` owns `Fingertip`, geometry and compliant-pad mechanical parameters,
  exact Shapely geometry, boundary semantics, LED metadata, and bulk optical
  material. LED geometry is
  metadata only: it is not unioned with or subtracted from mechanical material,
  collision, mesh, or FEM geometry.
- `mesh/` owns discrete topology, coordinates, semantic boundary groups, Gmsh
  conversion, settings, and quality. A `PadMesh` is a neutral view of the
  compliant-pad topology used across subsystem boundaries.
- `fem/` owns Kratos assembly, constitutive models, contact, constraints,
  nonlinear solves, and extraction of `FEAResult`. The pad's Young's modulus
  and Poisson ratio come from `FingertipMesh.parameters`; its public surface is
  `solve()`, `FEAResult`, and the solver-facing `IndenterSettings` fixture;
  Kratos objects do not cross into optics or visualization.
- `mechanics3d/` owns the optional NVIDIA Warp/Newton tetrahedral VBD
  prototype. Its public boundary is NumPy-only `TetMeshData`,
  `Mechanics3DSettings`, `ParticleLoad`, `Mechanics3DSession`, and
  `Mechanics3DResult`; the Newton model, CUDA arrays, and solver objects remain
  inside `mechanics3d.backends`. Its solver-neutral rigid-object boundary is
  `RigidIndenter3D`, `RigidPose3D`, `IndentationSettings`, and
  `IndentationResult`; the Newton path uses a kinematic triangle-mesh body and
  full-surface rigid-soft contact. `ParticleLoad` is the neutral external-force
  contract: local zero-based particle indices, finite Newton-valued forces, and
  an explicit deterministic ramp count. Neutral coordinates use the
  repository's millimetre convention; the backend converts to/from Newton's
  metre convention. `Mechanics3DSession` owns one persistent model/solver and
  restores a verified rest state before each independent solve. It is a fast
  mechanics surrogate and is not a replacement for `fem/` or Kratos.
- `case/` is the thin top-level research-case aggregate. `FingertipCase`
  owns one physical `Fingertip`, one `FEA2D` mechanics experiment, and one
  `RayTracing2D` optics experiment. `FEA2D` owns mesh/indenter/contact/solver
  configuration and stores its optional `FEAResult`; `RayTracing2D` owns
  PLANAR_2D settings and optional raw/summary results. `case.run_case()` is a
  convenience wrapper around `FingertipCase.run()`, while explicit users can
  call `solve()` and `trace()` separately. The case owns orchestration and
  consistency checks but neither solver algorithm. `case.save_case()` and
  `load_case()` retain the existing checked manifest and exact pose geometry.
  The mechanics result keeps its `PadMesh` contract in `FEAResult.mesh`; the
  complete `FingertipMesh` used for persistence remains
  `FEAResult.reference_mesh`. Case identity is derived from physical and
  numerical configuration, not free-form provenance.
- `optics/` owns deterministic ray transport and adapters from neutral meshes
  and displacement fields. `optics.cross_section` remains the dependency-
  light NumPy reduced 2D path; production morphology evaluation uses the
  `optics.transport3d` `Transport3DSettings`, `Transport3DResult`, and
  `trace_3d()` boundary in `PLANAR_2D` mode and therefore requires the optional
  CUDA, CuPy, PyOptiX, and OptiX-header environment.
- `mesh.indenter.IndenterPose2D` is the neutral mechanical-to-optical pose
  contract. A converged explicit-contact 2D solve carries the exact fixture,
  final prescribed travel, and mechanically identified active contact patch;
  optics does not reconstruct an independent circle.
- `optics.contact_object.IndenterOptics` owns external indenter boundary
  properties. The indenter material is not part of `Fingertip`; object
  absorption/transmission is terminal and dielectric reflection remains in the
  current medium without tracing propagation inside the object. In the current
  PLANAR_2D contact-only contract, only the exact silicone contact patch uses
  these properties; the exposed carrier is not an air-side blocker because it
  is not present as an OptiX scene surface. External air-to-indenter
  interaction remains deferred until both the scene and P2 occupancy use the
  same exposed-surface contract.
- `optics.cross_section` is the reduced deterministic 2D optical transport
  model used for design studies. `optics.transport3d` owns the independent,
  deterministic, camera-independent 3D dimensional validator. It consumes
  neutral reference/deformed pad meshes, reuses the shared extrusion topology,
  and calls the optional OptiX backend. Its public surface is
  `Transport3DSettings`, `Transport3DResult`, and `trace_3d()`; CUDA, CuPy,
  OptiX, and Kratos objects do not cross that neutral result boundary.
- `optics.mitsuba` owns the optional camera validator. Its public surface is
  `Camera`, `RenderSettings`, `RenderResult`, and `MitsubaRenderer`; extrusion,
  scene construction, and persistent renderer state are implementation details.
- `visualization/` is a thin Matplotlib layer. It exposes
  `plot_fingertip`, `plot_mesh`, `plot_fea`, `plot_transport`, `plot_camera`,
  `plot_case_comparison`, `plot_volume_mesh`, and `plot_volume_state`; these
  functions consume model/mesh/result
  objects and return an `Axes`, while `plot_case_comparison` returns one
  composed `Figure` from precomputed unloaded/loaded states. Private
  `draw_*` layers add geometry, mechanics, and optics to an existing Axes;
  they do not choose limits or figure layout. Plot wrappers own standalone
  axes policy and colorbars; the case composer owns only layout, shared
  physical bounds, shared norms, row colorbars, titles, and calls to those
  shared layers. It does not own a second scientific data model, artifact
  loader, figure DSL, panel hierarchy, or export framework. Visualization
  consumes the result-owned optical domain mask and never reconstructs it from
  deformed triangles. The volume helpers consume only the neutral
  `FingertipVolumeMesh` and `FingertipVolumeState` contracts; they do not import
  or inspect Newton, Warp, Kratos, or any mechanics solver.
- `validation/` owns scientific baselines, Phase acceptance, provenance,
  checkpointing, reports, and generated artifact schemas.
- `validation/mechanics3d/` owns read-only inspection of persisted native
  Kratos 3D reference artifacts, the timing-only prescribed-indentation Newton
  VBD benchmark, and the nominal FEA/VBD correspondence characterization. Its
  loader produces neutral validation reference states and never runs Kratos;
  its benchmark and correspondence runner call the independent `mechanics3d`
  boundary without adding contact or collision behavior. FEA-specific surface
  loading translation and geometry comparison descriptors remain in this
  validation package rather than in the generic mechanics backend.
- `validation/fem/throughput.py` owns the staged Kratos FEA throughput/fidelity
  study. It may request explicit benchmark-local mesh and solver settings, but
  it does not alter production defaults, constitutive/contact formulation, or
  downstream optical physics; generated reports remain under `output/`.
- `optimization/design_space.py` owns algorithm-independent study geometry
  variable and bound definitions. The production search has four active
  morphology variables, fixes the 30 mm/14 mm outer envelope, and freezes
  `FingertipParameters.void_height` at zero;
  the generic parameter object still supports nonzero historical/diagnostic
  geometry. It does not import NiceGUI or an optimizer.
- `OptimizationStudy` and `DesignEvaluator` use one 48-step FEM trajectory per
  diameter/location pair and four exact captured depth states per trajectory;
  low-level FEM defaults remain unchanged.
- `gui/` is a top-level interactive consumer. It owns parameter editing,
  design-space presentation, validation feedback, and embedded visualization
  composition. It does not own geometry equations, meshing, FEM, optical
  transport, or optimization algorithms.
- `optimization/` is a top-level consumer of neutral `model`, `mesh`, `fem`,
  and `optics` APIs. It owns the algorithm-independent morphology design
  space, fixed optimization-study configuration, diameter/location loading
  trajectories, captured-state aggregation, and scientific design scores. It
  does not own fingertip geometry, meshing, FEM, optical transport, camera
  rendering, GUI code, optimizer algorithms, or Ax/BoTorch models.

- `case.ContactState` is the neutral physical location/indentation/radius
  contract. `optimization.scenarios.ContactScenario` specializes that state
  for scenario-grid generation; it is not a dependency of `case` and is not a
  generalized load description.

- `optimization/ax_adapter.py` is the thin optional Ax 1.3.1 orchestration
  boundary. It maps active `DesignVariable` bounds into Ax, attaches the
  nominal trial, configures Ax's supported high-level generation strategy, performs
  sequential ask/evaluate/tell orchestration, reports failures, and returns
  compact observed trial records. It does not own GP models, kernels,
  acquisitions, BoTorch/GPyTorch objects, geometry, scientific bounds, contact
  scenarios, FEM, optical transport, the scientific objective, or GUI behavior.

  The intended future boundary is:

  ```text
  GUI / CLI
      -> DesignSpace + OptimizationStudy
  future optimizer adapter
      -> active DesignVariable bounds
      -> DesignSpace.decode(...)
      -> DesignEvaluator
  ```

### Shared 3D geometry and mesh contract

The nominal 3D mechanics paths share one authoritative geometry and TET4 mesh:

```text
FingertipModel -> FingertipSolid -> FingertipVolumeMesh
                                      ├── fem.solid3d / Kratos
                                      ├── mechanics3d.fingertip
                                      │      └── Newton runtime / SolverVBD
                                      └── visualization.volume
                                      ↓
                             FingertipVolumeState
                           ├── visualization.volume
                           └── optics.transport3d FULL_3D
```

`FingertipVolumeMesh` is the canonical 3D mechanics input. `model.solid` owns
the fixed 11 mm semantic extrusion and `mesh.volume3d` owns Gmsh
tetrahedralization. Kratos and the current Newton `SolverVBD` backend consume
that same volume mesh; neither backend defines a second fingertip geometry or
remeshes during state promotion. `visualization.volume` is a solver-neutral
consumer of the mesh and state, and `optics.transport3d` is likewise
backend-independent. Collision RT and contact RT are deferred and are not part
of this contract.

`FingertipVolumeState` is the canonical solver-neutral deformed 3D output. It
stores deformed coordinates in `tuple(sorted(volume_mesh.nodes))` order and
borrows the volume mesh's tetrahedral and semantic surface topology. Its
validation rejects non-finite coordinates, unknown source IDs, degenerate or
orientation-flipped surfaces, and inverted/degenerate tetrahedra. Mechanics
backends do not own morphology geometry, and `optics.transport3d` does not
know which backend produced the state.

Rigid-object indentation follows the same neutral-boundary rule:

```text
RigidObjectMesh
      ↓
RigidIndenter3D + RigidPose3D
      ↓
Newton kinematic triangle-mesh body
      ↓
full-surface rigid-soft contact
      ↓
SolverVBD fingertip deformation
      ↓
Mechanics3DResult
      ↓
FingertipVolumeState
```

`mesh.RigidObjectMesh` owns immutable millimetre triangle geometry and
parametric primitives; it contains no Newton, Warp, CUDA, or mechanics state.
`mechanics3d.indentation` owns pose and translation-only indentation contracts.
Only `mechanics3d.backends.newton_vbd` converts millimetres to metres and
constructs `newton.Mesh`, the kinematic body, SDF, collision pipeline, and
`SolverVBD`. The prepared fingertip's semantic support vertices are
authoritative: an optional `Mechanics3DSettings.fixed_vertex_indices` value
must be empty or exactly match them; it is never silently substituted. The
rigid SDF uses an explicit contact-scale voxel setting in millimetres rather
than a fraction of total object extent. The backend reports the actual soft
contact record count separately from Newton's per-body overflow counters.
Those counters are the only buffer-safety signal; the per-body lists skip the
kinematic indenter in Newton 1.4, so their status is explicitly marked
not-applicable for this scene. A nonzero overflow counter still fails without
returning a result. The existing prescribed-vertex function remains a
timing-only, non-contact benchmark.

The intended later arbitrary-object path is:

```text
future OBJ/STL/PLY
      ↓
explicit preparation and provenance
      ↓
RigidObjectMesh
      ↓
future Collision-RT first-contact pose
      ↓
RigidPose3D
      ↓
the same Newton full-surface contact path
```

Collision-RT and real-object conversion are future work. Primitive meshes are
generated in memory and are not an implicit asset-file dependency.

The production full-3D optical adapter builds the compliant surface directly
from `FingertipVolumeState` and uses one shared fixed rigid-carrier/virtual-
envelope builder from authoritative fingertip geometry. Persisted FEA surface
artifacts remain validation evidence and backward-compatible readers; they are
not required to construct VBD full-3D geometry.

## Fingertip and mesh state

`Fingertip` is the physical root:

```text
Fingertip
├── parameters
├── geometry
├── boundaries
├── led
├── optical
├── led_source
└── mesh(settings)
```

It is a mesh factory, not the owner of one permanent mesh: a single design can
be discretized at several resolutions. `volume_mesh(settings)` is the lazy
canonical 3D facade and defaults to the established search-tier policy; it
does not cache one permanent mesh. The returned mesh retains the rigid
carrier and contact information required by FEM while exposing the neutral pad
contract through `node_ids`, `coordinates`, `triangles`, and `boundaries`.
Its cached `.pad` view owns that neutral topology for optics.

A `PadMesh` owns reference coordinates, triangles, and a complete semantic
partition of its boundary edges. Load state is expressed by composition:

```python
loaded_mesh = reference_mesh.deformed(displacement)
```

The returned immutable view presents the same `coordinates`, `triangles`, and
`boundaries` interface. Deformation is not a mesh subtype and has no public
state-container class.

`FEAResult.displacement` follows the pad mesh's node order. On a converged
solve, `FEAResult.deformed_mesh` delegates directly to
`mesh.deformed(displacement)`; it neither rebuilds nor copies topology.
The result carries the `PadMesh` used by existing callers and may carry the
full `FingertipMesh` used by the solve in `reference_mesh`, so a
`FingertipCase` or artifact loader can verify morphology parameters and
rigid/contact topology without changing the pad-only result contract.

Strong validation remains at user, Gmsh/Kratos, and artifact boundaries.
`PadMesh.deformed()` still rejects non-finite fields and degenerate or inverted
triangles. The facade does not weaken geometry validation.

## Optical transport

The loaded silicone region is reconstructed from deformed mesh triangles. The
fixed rigid link is then excluded from the envelope used by the ray tracer.
The compliant pad's open U-shaped cutout makes semantic boundary topology
essential: the tagged external shell is closed virtually across the cutout
mouth to recover the loaded outer envelope.

That virtual closure is not silicone. Space inside the envelope but outside
the deformed silicone and fixed rigid material is air, including any gap opened
between the fixed stem/LED and the displaced cutout bottom.

`TransportResult` contains raw weighted path density, grid edges, retained ray
segments, outgoing `ExitEvent` records, silicone/air/rigid/LED regions, source
position, and energy bookkeeping. With an explicit posed indenter, the
mechanical active patch is a direct silicone-to-object interface; it is never
silicone-to-air-to-indenter. PLANAR_2D tags only deformed `pad_outer_arc`
boundary edges whose two endpoints are in the FEA-supplied active contact-node
set. It does not infer contact from a circle, distance threshold, or a rebuilt
indenter pose. Requesting object optics without that nonempty mechanical patch
is fail-closed. Object-absorbed and object-transmitted weights are terminal and
are not counted as air escape; dielectric reflection stays in silicone and the
object interior is never traced. With no `IndenterOptics`, the AIR_CONTROL path
retains the existing silicone/air transport and all object channels are zero.
The density is a deterministic light-transport proxy, not camera brightness,
irradiance, or a predicted sensor image.

`evaluate(reference, loaded)` remains camera-independent and returns a plain
dictionary. Its deliberately small initial contract is:

- `field_difference`: total-variation distance between normalized path-density
  distributions after conservative redistribution onto one physical grid;
- `centroid_shift_mm`: displacement of the path-density centroid;
- `escaped_fraction_change` and `absorbed_fraction_change`: changes relative to
  launched ray weight.

These are optimization proxies, not camera or irradiance metrics. Thresholded
effective area, entropy, and other research-dependent metrics remain undefined
until their scientific conventions are accepted.

## Optional camera validation

The three optical paths have intentionally different purposes:

```text
optics.cross_section  -> reduced deterministic 2D transport for design studies
optics.physics        -> shared NumPy interface/Fresnel math
optics.contact_object -> external posed-indenter optical boundary contract
optics.optix          -> optional CUDA/OptiX paths, doctor, and low-level runtime
optics.transport3d    -> deterministic camera-independent 3D dimensional validation
optics.mitsuba        -> optional camera/rendering validation
```

The 3D path uses one 11 mm periodic longitudinal representative cell and a
single deterministic source at `z = 0`. Its outgoing field is indexed by
reference material coordinate `u` along the exposed compliant boundary and
periodic-cell `z`; it is not a camera image or an irradiance prediction. The
numerical z planes are periodic transport boundaries, not physical escape
surfaces. `optics.mitsuba` remains independent and retains the camera role
described below.

`MitsubaRenderer` represents one design, one fixed reference topology, one
fixed extrusion depth, and one fixed camera. `render()` accepts the reference
mesh, a matching deformed mesh view, or a displacement array. Internally, only
vertex positions change between load states; face connectivity and the Python
scene remain stable. A changed design or remesh requires a new renderer.

The 2D mesh is extruded to a fixed z-depth only inside this optional boundary.
No PLY, OBJ, or STL intermediary is created.
The internal files remain separated because immutable camera/settings data,
procedural scene construction, and the mutable persistent session have distinct
owners; only `renderer.py` forms the public boundary.

## Visualization and validation figures

The public plotting helpers do not generate Gmsh meshes, start Kratos, or
import Mitsuba. Full scientific figure workflows remain explicit under
`validation/figures/` or the relevant validation package. For example,
`validation.figures.displacement_atlas` validates persisted normal-indentation
NPZ artifacts and calls `plot_fea`; `validation.figures.transfer_map`
owns the Phase 4K artifact tables and direct plots from canonical arrays.

## Artifact boundary

New full-field NPZ artifacts store semantic edge groups under
`boundary_edge_node_ids__<tag>`. The loader preserves those groups directly;
missing semantic tags are rejected by the external optical adapter rather
than reconstructed from a legacy classifier.

## Failure and artifact policy

Numerical failures remain explicit; non-finite fields, non-convergence, invalid
contact states, and invalid geometry are not clamped or hidden. Repeated solves
have step, iteration, and process timeout boundaries. `output/` is only an
untracked generated sink. Reference inputs required by a clean checkout belong
in `tests/fixtures/` or `validation/reference_data/`.
