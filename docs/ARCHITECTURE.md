# Architecture

## Dependency direction

```text
model -> mesh -> fem -> neutral result/artifact data
  \        \       \              \
   \        \       +-------------> validation
    \        +---------------------> validation
     +------------------------------> visualization / validation

model -> optics adapters/geometry -> neutral optical result -> visualization
                    \-> optional Mitsuba session -> camera result -> visualization

visualization -> model + mesh + neutral artifacts
validation    -> model + mesh + fem + visualization
tests         -> the package under test
```

Dependencies only flow to the right/down. Production packages never import
`validation` or `tests`. `visualization` has no Kratos import-time dependency.

## Ownership

- `model/` owns parametric Shapely geometry and physical sensor metadata.
  `FingertipSensorModel` combines a `FingertipModel`, `LEDParameters`, and
  `OpticalMaterialParameters`. LED package/source placement is sensor metadata;
  it is never unioned into, or subtracted from, FEM material, collision, or mesh
  geometry. The package owns no Gmsh, Kratos, optics algorithm, plotting, or
  file output.
- `optics/` owns immutable reference topology, replaceable deformation state,
  FEA-to-optics adapters, deterministic 2D ray transport, 2D-to-3D extrusion,
  optional Mitsuba integration, and solver-specific numerical settings. It
  consumes model-owned LED and material properties without changing mechanical
  geometry and never imports visualization.
- `mesh/` owns the deterministic conversion from `FingertipModel` to discrete
  Gmsh topology, mesh settings, quality, and the solver-independent indenter.
- `fem/` is the Kratos backend. It owns model-part assembly, materials,
  contact, constraints, nonlinear indentation, observations, and conversion to
  neutral Python results. It owns no PASS/FAIL policy, report files, or plots.
- `visualization/` owns semantic figure data, transforms, themes, panels,
  plotting, and export. Repository artifact parsers live in
  `visualization/adapters/`.
- `validation/` owns scientific benchmarks, Phase acceptance, subprocess
  isolation, checkpointing, provenance, reports, and generated artifact
  schemas.
- `tests/unit/` owns fast dependency-light contracts. `tests/smoke/` only
  verifies Gmsh, Kratos, or headless-renderer wiring.

## Runtime flow

Validation entrypoints construct immutable model parameters, build a discrete
mesh, configure a fresh Kratos model, run a bounded solve, extract neutral
results, apply scientific acceptance, and write to an explicit output
directory. Visualization reads declared input artifacts and never starts a
solver.

## Deformation-aware optical state

One physical sensor and one fixed FEA pad topology support every load condition:

```text
FingertipSensorModel
    physical geometry + LED + optical material

PadMeshTemplate2D
    reference coordinates + fixed triangles + semantic pad boundaries

PadDeformationState2D
    replaceable x-y nodal displacement field
```

No-load is not a separate geometry type. It is
`PadDeformationState2D.zero(template)`. A loaded condition uses the same
`PadMeshTemplate2D` with a nonzero state. `PadField2D` is the small aggregate
that pairs those two values.

Persisted NPZ fields and in-memory `FingertipMesh` plus displacement mappings
both converge through `optics.adapters` to the same `PadField2D`. The in-memory
path accepts the neutral mesh and displacement values returned by the FEM
indentation layer; it does not require Kratos objects or intermediary files.

The two optical consumers are:

```text
Cross-section path
    template + state
    -> deformed triangle union / optical domain
    -> deterministic 2D ray transport
    -> diagnostic Matplotlib visualization

Camera path
    template + state
    -> stable extrusion vertices
    -> persistent in-memory Mitsuba scene
    -> raw linear-RGB camera result
    -> visualization-owned normalization and PNG export
```

The cross-section mesh-state builder treats the deformed triangles as the
loaded silicone source of truth. It does not perturb analytic arc parameters or
redraw the undeformed pad. The rigid link and LED remain fixed in the stem frame
for this iteration.

Semantic boundary topology is also required to recover the completed loaded
optical envelope. The silicone mesh is an open U-shaped material region: its
cutout is connected to the top boundary, so filling polygon holes cannot close
that concavity. `PadMeshTemplate2D` therefore stores a complete partition of
its fixed boundary edges under stable pad tags, while
`PadDeformationState2D` continues to own only replaceable nodal displacement.

Loaded cross-section geometry is reconstructed as:

```text
deformed external pad shell
    + virtual closure across the cutout mouth
    - fixed rigid link/stem
    = accessible optical region
```

The virtual closure is not silicone; it only completes the optical envelope.
Space inside that envelope but outside the deformed silicone and fixed rigid
material is air. This includes a gap opened between the fixed LED/stem and a
distally displaced cutout bottom. No-load remains a zero-displacement state,
and the rigid stem and LED remain fixed in the current sensor frame.

One `MitsubaRenderSession` represents one design, one fixed 2D topology, one
fixed extrusion topology, and one fixed camera framing. Switching load state
updates only the in-memory pad vertex positions; face connectivity and the
Python scene remain unchanged. Mitsuba may rebuild its internal acceleration
structure. A changed design or remesh requires a new template and session.

Mitsuba remains optional. The analytic preview adapter is the only place that
uses Shapely triangulation for the dependency-light no-load camera example;
normal rendering creates no PLY, OBJ, or STL intermediary.

## Failure and artifact policy

Numerical failures stay explicit; non-finite fields, non-convergence, and
invalid contact states are not clamped or hidden. Every repeated solve has a
step/iteration/process timeout boundary. `output/` is only a generated sink.
Reference inputs needed by a clean checkout belong in `tests/fixtures/` or
`validation/reference_data/`.
