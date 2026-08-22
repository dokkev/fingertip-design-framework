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
   FingertipMesh
        ↓
   Newton model
        ↓
 LumoSimulation
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
    └── Carrier
```

`FingertipParameters` contains physical input values.

`Fingertip` constructs the analytic fingertip assembly.

`Silicone` and `Carrier` are constructed geometry objects, not separate
parameter systems.

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
└── bonded_vertex_indices
```

The silicone is discretized as a Newton-compatible tetrahedral mesh.

The carrier is discretized as a rigid surface mesh.

`bonded_vertex_indices` identifies silicone vertices belonging to the
silicone-to-carrier perfect-bond interface.

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
- the silicone-to-carrier perfect bond;
- kinematic rigid indenters created from URDF or prepared `newton.Mesh` assets;
- Newton-specific model and object identities consumed by the runtime.

Use Newton's public API directly whenever practical.

Do not introduce a generic physics-backend layer while Newton is the only
mechanics implementation.

Do not create generic attachment, constraint, simulation, runtime, or solver
frameworks without a concrete second use case.

### `lumo/simulation.py`

`LumoSimulation` is the one concrete runtime owner for a complete LUMO
simulation. It owns:

- the fixed global `time_step_s` and accumulated `time_s`;
- the current and next Newton states;
- Newton control;
- SolverVBD;
- the collision pipeline and contacts;
- one global simulation step.

The current step order is:

```text
caller updates carrier and indenter poses
    ↓
collision
    ↓
SolverVBD step
    ↓
state swap
    ↓
advance global time
```

`LumoSimulation.step()` does not own approach trajectories, force thresholds,
validation policy, or result reporting. It may later orchestrate optical work,
but no ray-tracing behavior is part of the current runtime.

### `lumo/ray_tracing/`

Owns LUMO-specific optical transport behavior.

OptiX is the ray-tracing backend.

This package should define only the optical semantics required by LUMO and use
OptiX for generic ray-tracing functionality.

Do not implement a general-purpose ray tracer inside LUMO.

This layer must not own mechanics evolution.

### `lumo/optimization/`

Owns design-space and optimization policy.

Current responsibilities include design parameter bounds and feasibility
constraints.

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
URDF path ───────────────────────→ Indenter.add_urdf()

OBJ/STL path → mesh_io.load_mesh() → newton.Mesh → Indenter.add_mesh()
```

An external rigid body is composed with the fingertip before Newton model
finalization:

```text
caller-owned ModelBuilder
├── Indenter.add_urdf(...) or Indenter.add_mesh(...)
└── build_fingertip_newton_model(..., builder=builder)
        └── finalize one shared Newton model
```

`build_fingertip_newton_model()` does not choose or place an indenter. It only
accepts a caller-populated builder so all scene bodies can be finalized into
the same Newton model.

`Indenter.add_urdf()` stores its supplied world `tf` as the initial kinematic
body pose as well as passing it to Newton's URDF importer.

`Indenter` owns only construction and the Newton body index identifying one
kinematic rigid indenter. `LumoSimulation` owns mutations of Newton state,
including carrier and indenter pose updates.

Do not turn this package into a general service or utility layer.

## Validation boundary

`validation/` is not a production package.

Validation scripts are top-level consumers of production APIs.

`validation/contact-physics/zero_load.py` is the current procedural check that
SolverVBD preserves the unloaded fingertip reference state.

`validation/contact-physics/flat_plate_contact.py` loads the flat-plate URDF,
moves its kinematic body toward the fingertip in bounded positive-Z increments,
and stops its local experiment loop when the transient negative-Z reaction
force reaches the caller-provided threshold.

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
Y = fingertip width / extrusion direction
Z = contact-normal direction
```

Unit conversion belongs at the appropriate backend boundary and should not be
silently duplicated across packages.

## External libraries

Newton and OptiX are primary dependencies, not reference implementations.

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
