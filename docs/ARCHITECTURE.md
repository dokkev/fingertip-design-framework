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

- `model/` owns `Fingertip`, mechanical parameters, exact Shapely geometry,
  boundary semantics, LED metadata, and bulk optical material. LED geometry is
  metadata only: it is not unioned with or subtracted from mechanical material,
  collision, mesh, or FEM geometry.
- `mesh/` owns discrete topology, coordinates, semantic boundary groups, Gmsh
  conversion, settings, and quality. A `PadMesh` is a neutral view of the
  compliant-pad topology used across subsystem boundaries.
- `fem/` owns Kratos assembly, constitutive models, contact, constraints,
  nonlinear solves, and extraction of `FEAResult`. Its public surface is
  `solve()`, `FEAResult`, and the solver-facing `IndenterSettings` fixture;
  Kratos objects do not cross into optics or visualization.
- `optics/` owns deterministic ray transport and adapters from neutral meshes
  and displacement fields. Its public transport surface is `TraceSettings`,
  `RaySegment`, `TransportResult`, `trace()`, and `evaluate()`.
- `optics.mitsuba` owns the optional camera validator. Its public surface is
  `Camera`, `RenderSettings`, `RenderResult`, and `MitsubaRenderer`; extrusion,
  scene construction, and persistent renderer state are implementation details.
- `visualization/` is a thin Matplotlib layer. It exposes only
  `plot_fingertip`, `plot_mesh`, `plot_displacement`, `plot_transport`, and
  `plot_camera`; these functions consume model/mesh/result objects and return
  an `Axes`. It does not own a second scientific data model, artifact loader,
  figure DSL, panel hierarchy, or export framework. `plot_transport(result)`
  needs no separate model or optical-domain argument because `TransportResult`
  owns the geometry used to produce the field.
- `validation/` owns scientific baselines, Phase acceptance, provenance,
  checkpointing, reports, and generated artifact schemas.

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
be discretized at several resolutions. The returned mesh retains the rigid
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
segments, silicone/air/rigid/LED regions, source position, and energy
bookkeeping. The density is a deterministic light-transport proxy, not camera
brightness, irradiance, or a predicted sensor image.

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
NPZ artifacts and calls `plot_displacement`; `validation.figures.transfer_map`
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
