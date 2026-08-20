# LUMO Architecture Map

This is the authoritative navigation map for the current LUMO checkout. It is
written for agents and contributors: use it to find the owner of a concept,
follow one complete runtime path, and avoid recreating removed architecture.

This document describes the code that exists now. It is not a session log,
backlog, or a compatibility guide for deleted generations.


## Read this first

Route a change to the smallest owning area:

| Change concerns | Start with | Keep out of that package |
| --- | --- | --- |
| morphology fields, geometry invariants | `finger/` | solver settings, OptiX, GUI |
| Gmsh or neutral mesh records | `mesh/` | mechanics stepping, objective policy |
| first-contact pose and approach geometry | `contact/` | Newton stepping, optical scoring |
| Newton/Warp mechanics | `physics/` | Ax policy, validation reports |
| transport geometry, OptiX, optical results | `ray_tracing/` | mechanics imports, campaign policy |
| concrete LUMO simulation flow | `lumo/` | generic backend abstractions, objective policy |
| protocol, design space, objective, Ax boundary | `optimization/` | solver implementation |
| studies, reports, regression/reference workflows | `validation/` | production dependencies |
| interactive controls or diagnostics | `gui/` | core geometry and solver ownership |

Before editing, read the relevant package and then follow the complete path in
the next sections. If ownership, a public boundary, or dependency direction
changes, update this map in the same change.


## At a glance

LUMO builds a parameterized fingertip, meshes its compliant volume, derives
first contact for each rigid indenter condition, runs a continuous Newton/Warp
indentation trajectory, transports the resulting deformed 3D state through the
FULL_3D OptiX backend, and computes a trajectory objective for morphology
search. `lumo/simulation.py` owns the reusable concrete Newton + OptiX
orchestration for one prepared morphology; `optimization/evaluator.py` owns
protocol/objective policy and persistence boundaries.

```text
finger.FingertipParameters
    -> finger / mesh geometry
    -> contact.find_first_contact
    -> physics.trajectory.indentation.solve_fingertip_indentation_trajectory
    -> lumo.LumoSimulation in-memory state handoff
    -> ray_tracing.optical_mechanics FULL_3D OptiX
    -> optimization.objectives
    -> validation reports or optimization.adapters.ax
```


## Repository map

| Path | Role | Canonical status |
| --- | --- | --- |
| `util/` | dependency-free scalar validation helpers | narrow shared utility boundary |
| `finger/` | raw morphology parameters, solids, material/LED descriptors | production domain source |
| `mesh/` | 3D neutral volume-mesh records, Gmsh meshing, rigid geometry | production discretization boundary |
| `contact/` | geometry-derived first-contact and sphere alignment | production contact initialization |
| `physics/` | Newton 1.4 / Warp mechanics and trajectory state | one production mechanics path |
| `ray_tracing/` | optical contracts and FULL_3D transport implementation | production BO path |
| `lumo/` | reusable concrete LUMO simulation state and execution orchestration | production orchestration boundary |
| `optimization/` | fixed protocol, objective, registry, Ax boundary, evaluator, persistence | production evaluation and search boundary |
| `validation/` | reports, smoke tests, regression/reference workflows, bounded campaign runners | domain/solver/transport ownership; production evaluation is in `optimization/` |
| `validation/reference/` | preserved fixed-state and Kratos reference implementations | validation-only |
| `gui/` | NiceGUI design-space shell and diagnostics | optional consumer, not core architecture |
| `tests/` | unit and dependency/runtime smoke contracts | never a production dependency |
| `docs/` | architecture and reproducible command maps | documentation only |
| `output/` | generated validation and benchmark artifacts | generated, untracked |

Only the paths above are current architecture landmarks. Empty local
directories named `case/`, `examples/`, `fem/`, `visualization/`, or
`mechanics3d/` are not production packages and must not be recreated.


## Code map

| If you need to understand... | Start here | Then inspect |
| --- | --- | --- |
| morphology, material, and optical inputs | `finger/fingertip_parameters.py::FingertipParameters` | `finger/fingertip_geometry.py`, `finger/fingertip.py`, `finger/extrusion.py` |
| neutral volume mesh | `mesh/volume/contracts.py` | `mesh/volume/mesh.py`, `mesh/volume/state.py` |
| rigid object/carrier mesh | `mesh/rigid/object.py` | `mesh/rigid/carrier.py` |
| neutral rigid pose | `mesh/rigid/object.py::RigidPose3D` | `contact/`, `physics/` |
| first contact | `contact/first_contact.py` | `contact/sphere_alignment.py` |
| mechanics public API | `physics/trajectory/indentation.py` | `physics/trajectory/fingertip.py`, `physics/contracts/` |
| Newton implementation | `physics/newton/vbd.py` | `physics/newton/session.py`, `physics/newton/viewer.py` |
| FULL_3D transport | `ray_tracing/optical_mechanics/transport.py` | `geometry.py`, `fingertip.py`, `optix_backend.py` |
| OptiX runtime/preflight | `ray_tracing/optix/runtime.py` | `scripts/tools/optix_smoke.py`, `scripts/tools/optix_doctor.py` |
| evaluation protocol | `optimization/protocol.py` | `lumo/mechanics_contract.py` |
| morphology search space | `optimization/design_space.py` | `optimization/evaluation_registry.py` |
| objective | `optimization/objectives.py` | `optimization/evaluator.py` |
| reusable LUMO simulation | `lumo/simulation.py` | `optimization/evaluator.py` |
| Ax campaign boundary | `optimization/adapters/ax.py` | `validation/optimization/lumo6d_test_bo.py` |
| production trajectory evaluator | `optimization/evaluator.py` | `lumo/simulation.py`, `validation/optimization/lumo3d_trajectory_validation.py` |
| persisted mechanics state | `optimization/deformed_state_artifact.py` | evaluator artifact writers; validation replay is in `validation/ray_tracing/deformed_state_restore.py` |
| reference comparison | `validation/physics/correspondence.py` | `validation/reference/kratos3d/` |
| interactive Newton view | `physics/newton/viewer.py` | example callers, if reintroduced explicitly |


## Package ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `util` | small typed helpers with no domain or runtime dependencies | morphology policy, solver settings, persistence, or cross-layer orchestration |
| `finger` | kinematic, viscoelastic, and bulk optical fingertip parameters, morphology constraints, 2D solid boundaries, LED source descriptor | mesh construction, mechanics, ray-tracing execution, UI |
| `mesh` | neutral mesh dataclasses, Gmsh-backed volume meshing, fingertip state conversion, sphere/carrier meshes | Newton stepping, OptiX calls, optimization policy |
| `contact` | Shapely-based collision predicate, coarse bracket/bisection, canonical sphere alignment, clear/spawn/contact poses | deformation solve and optical transport |
| `physics` | NumPy-facing Newton/Warp settings/results, prescribed indentation, continuous trajectory checkpoints, contact diagnostics | Ax generation, objective calculation, validation orchestration |
| `ray_tracing` | optical boundary/result contracts, FULL_3D surface geometry, CUDA/OptiX runtime and launches | mechanics state evolution and BO decisions |
| `lumo` | concrete `LumoSimulation` state/orchestration and named contact results | contact/Newton/OptiX implementation, persistence, and optimization policy |
| `optimization` | six-dimensional design space, fixed-depth factorial protocol, trajectory objective, exact morphology registry, Ax adapter, evaluator, artifact persistence | mesh/solver/transport implementation and reusable simulation state |
| `validation` | reports, smoke/regression/reference workflows, and bounded campaign runners | domain/solver/transport ownership; production packages must not import it |
| `gui` | optional controls and diagnostics presentation | domain rules, solver settings, transport, campaign orchestration |

The current package exports in `finger/__init__.py`, `mesh/__init__.py`,
`contact/__init__.py`, `physics/__init__.py`, `ray_tracing/__init__.py`, and
`optimization/__init__.py` are the primary lightweight API surfaces. Prefer
those exports or the canonical module named above over new wrapper layers.

`KinematicParameters` stores constructor-level geometry and representation
inputs. `ViscoelasticParameters` stores the fingertip's Newton constitutive and
inertial inputs, and `OpticalParameters` stores its bulk optical inputs.
`FingertipParameters` combines all three groups while preserving the direct
geometry field access used by mesh and contact callers. The LED remains a
separate source/package descriptor in `finger.led`; its values are fixed
evaluation inputs alongside `FingertipParameters`. Coordinates and dimensions
derived from parameter fields are computed explicitly by the owning geometry,
thickness, or reporting consumer; they are not duplicated as public parameter
properties. The physical morphology fingerprint intentionally excludes
representation, material, and optical fields, as documented by
`fingertip_parameters_fingerprint()`.


## Primary execution path

The current candidate-evaluation workflow is
`optimization.evaluator.Lumo3DTrajectoryEvaluator`. It owns candidate,
protocol, objective, and provenance handoff. `lumo.simulation.LumoSimulation`
owns one prepared morphology's reusable mesh/contact/runtime state and the
per-condition scientific call order; it does not implement contact, Newton, or
optical physics.
Validation scripts consume this evaluator for regression and campaign reports;
they are not part of the production runtime path.

```text
FingertipParameters
  -> LumoSimulation.from_fingertip
  -> reusable volume mesh / prepared mechanics / carrier / OptiX runtime
  -> run_sphere_contact(u, radius, depths)
  -> first contact -> continuous Newton trajectory
  -> checkpoint state -> FULL_3D OptiX trace
  -> TrajectoryObservation records
  -> compute_trajectory_objective
```

`LumoSimulation` runs six mechanics trajectories per morphology from the
default protocol and returns three absolute post-contact depth checkpoints per
trajectory, for 18 optical states. Radius and depth are independent protocol
axes; normalized depth/radius values are derived diagnostics only.

`LumoSimulation` keeps each Newton checkpoint in memory while constructing its
optical state and returns named `ContactSimulationResult` /
`ContactOpticalState` values. The evaluator may persist mechanics checkpoints
for campaign provenance after the in-memory handoff; persistence is not a
required production ray-tracing input. Optical artifacts include the protocol,
transport configuration, contact-state identity, and deformed 3D surface
provenance.

There is no separately generated optical fingertip mesh. The OptiX adapter
selects the lateral semantic triangles from the exact
`FingertipVolumeState` checkpoint and copies those deformed coordinates into
the runtime buffers. The rigid GAS uses the same neutral `RigidCarrierMesh`
supplied to Newton; its named lateral face group excludes the periodic z-caps.
The virtual envelope reuses the canonical reference outer/support triangles
and adds only a two-triangle air-escape closure derived from `FingertipSolid`;
it is not a second physical discretization.


## Important subsystem paths

### Contact initialization

`contact.find_first_contact()` uses a collision-free/hit bracket and refinement
to estimate the first-contact boundary. Exact touching at the floating-point
midpoint is not the contract. The returned `FirstContactResult` carries a
clear pose, contact pose, travel, approach direction, and a clear spawn pose.
The spawn clearance is a numerical initialization safeguard, not physical
indentation depth. Contact checks must remain geometry-derived rather than
hard-coded to world coordinates.

### Mechanics

`physics.trajectory.indentation.solve_fingertip_indentation_trajectory()` is the shared
incremental trajectory loop. The public units are millimetres; the Newton
backend converts positions to metres at the solver boundary and returns
millimetres in `NewtonResult` and checkpoint records. `physics.newton.vbd` is
the sole Newton implementation. `physics.newton.viewer` is debug-only and must not
change solver state or become a general visualization framework.

`lumo.mechanics_contract.DEFAULT_MECHANICS_CONTRACT` owns the frozen solver
execution settings and checkpoint-acceptance thresholds: timestep, contact
coefficients, iteration count, and admissibility limits. The fingertip's
`density`, `k_mu`, `k_lambda`, and damping are owned by
`FingertipParameters.viscoelastic` and passed to Newton at the LUMO simulation
boundary. These are backend coefficients, not a calibrated `E, nu` material
model. Do not duplicate or retune either contract in the evaluator, GUI, or
validation scripts.

### Optics

`ray_tracing.optical_mechanics` owns production FULL_3D state-to-OptiX
adaptation, field/path accumulation, carrier-interface handling, and trace
results. It consumes neutral mesh contracts and does not remesh the fingertip.
The actual runtime boundary is
`ray_tracing.optix.runtime.OptixRuntime.create()` and the production trace path
is `trace_geometry()`. Artifact persistence and contract fingerprints belong
to `optimization/optical_artifact.py`.

`ray_tracing.optix.runtime` owns only the optional CUDA/OptiX setup and execution
machinery. `scripts.tools.optix_smoke` performs the real setup, GAS build,
launch, and hit/miss verification used as the BO preflight. The
`scripts/tools/optix_doctor.py` command diagnoses an environment for human
troubleshooting; it is not part of the production ray-tracing path. A shared
CUDA/OptiX dependency failure is campaign-fatal, not a morphology-specific
ray-tracing failure.

`Transport3DResult.escaped_weight` is the complete escape-energy channel and
includes virtual-envelope escape. `outgoing_surface_weight`, the surface
field, and the recorded escape arrays cover only silicone-surface events that
have a defined `(u, z)` coordinate. Artifacts preserve both values; do not
silently treat the surface-observation subset as total escaped energy.

The old planar transport facade and cross-section implementation have been
removed. FULL_3D owns its deterministic sampling, native 3D accumulation, and
OptiX boundary directly; no reduced optical path is a hidden dependency of the
trajectory evaluator.

### Optimization

`optimization.protocol.TrajectoryEvaluationProtocol` is the authoritative
evaluation design: semantic contact locations, fixed indenter radii, and fixed
absolute depths. `optimization.design_space.DesignSpace` owns the six active
morphology variables and physical feasibility constraints. `objectives.py`
owns the objective formula.

`optimization.evaluation_registry.EvaluationRegistry` stores exact morphology
provenance and reusable results. It is a cache of scientific outcomes, not a
replacement for Ax model state. `optimization.adapters.ax` is the only Ax
boundary; it distinguishes duplicate lookup, candidate failure, and campaign
infrastructure failure.


## Public and data boundaries

| Boundary | Owner | Consumer | Contract |
| --- | --- | --- | --- |
| morphology parameters | `finger` | `mesh`, `optimization`, `validation`, `gui` | immutable six-variable design plus explicit constraints; public geometry units are mm |
| neutral volume mesh | `mesh` | `physics`, `ray_tracing`, `validation` | one canonical tetra topology plus semantic surface triangles; no solver or OptiX object |
| first-contact result | `contact` | `physics`, `validation` | geometry-derived poses and post-contact travel; `T_spawn` is clear-side initialization only |
| mechanics result/checkpoint | `physics` | `validation`, `ray_tracing` | NumPy arrays and immutable diagnostics; Newton state does not cross into ray tracing |
| in-memory deformed state | `mesh.FingertipVolumeState` via `lumo` | `ray_tracing.optical_mechanics` | exact Newton-compatible node order, deformed coordinates, and semantic triangles; authoritative production handoff |
| rigid carrier mesh | `mesh.RigidCarrierMesh` | `physics`, `ray_tracing` | one closed neutral mesh with explicit lateral/end face groups; OptiX excludes periodic z-caps |
| persisted mechanics artifact | `optimization/deformed_state_artifact` | evaluator writers | exact checkpoint mesh plus digest and source-node provenance; validation-only restoration belongs to `validation/ray_tracing/deformed_state_restore` |
| optical result | `ray_tracing.optical_mechanics` | `optimization`, `validation` | raw transport fields/weights, energy bookkeeping, and configuration fingerprints |
| objective observation | `optimization.objectives` | `validation`, `optimization.adapters.ax` | trajectory observations preserve location, radius, depth, raw field, and diagnostics |
| registry record | `optimization.evaluation_registry` | Ax adapter/campaign reports | exact contract + morphology identity; failed records carry no successful objective |

Representation conversions stay at boundaries:

```text
public geometry / poses: mm
    -> Newton backend: metres (SI)
    -> returned mechanics artifacts: mm
raw optical transport
    -> objective normalization
    -> display/report transforms only outside scientific identity
```

Do not put visualization smoothing, color normalization, or GUI state into
mechanics/ray-tracing artifacts or optimization metrics.


## Runtime dependencies

| Dependency | Owner | Boundary |
| --- | --- | --- |
| Gmsh | `mesh` | imported only when volume meshing is requested |
| Newton / Warp | `physics.newton` and execution helpers | required only for mechanics execution; keep public NumPy contracts neutral |
| CuPy / PyOptiX / CUDA Python / NVRTC | `ray_tracing.optix` and `ray_tracing.optical_mechanics` | runtime/preflight/trace boundary; environment is externally managed |
| Ax 1.3.1 | `optimization.adapters.ax` | optimizer execution boundary; importing `optimization` must stay lightweight |
| NiceGUI / Matplotlib | `gui` | optional presentation boundary |
| Kratos | `validation/reference/kratos3d` | validation-only external reference; never a production dependency |

Installation and exact commands belong in `docs/COMMANDS.md`.


## Dependency rules

The allowed high-level direction is:

```text
finger -> mesh -> contact -> physics
finger / mesh -> ray_tracing
contact / physics / ray_tracing -> lumo
lumo -> optimization / validation consumers
```

This diagram is a consumption direction, not a requirement that every package
import every predecessor. In particular, ray tracing consumes neutral finger/mesh
and in-memory mechanics states at the `lumo` orchestration boundary; ray tracing does
not import `physics`. Validation is the top-level scientific consumer and may
compose all production packages.

Important guards:

- production packages do not import `validation` or `tests`;
- `finger` remains geometry/model-only and does not import `mesh`, `physics`,
  `ray_tracing`, plotting, Gmsh, or Kratos;
- `mesh` does not import mechanics, ray tracing, validation, plotting, or Kratos;
- `physics` does not import ray tracing, validation, or tests;
- `ray_tracing` does not import physics, validation, or tests;
- low-level packages do not import GUI code;
- validation may compose production APIs and its own workflow helpers; production code never imports validation;
- shared helpers must remain small, explicitly typed, dependency-free, and
  backed by current consumers; do not turn `util` into a generic cross-layer
  service or put domain policy there;
- optional heavy dependencies enter at execution boundaries rather than
  changing neutral data contracts.

Static enforcement lives in `tests/unit/test_architecture.py`. Treat a new
violation as an architecture change, not as a test to weaken.


## Failure semantics

| Status/error | Meaning | Handling |
| --- | --- | --- |
| `invalid_design` | morphology violates explicit model/design constraints | reject candidate before or during evaluation |
| `mesh_failure` | candidate mesh cannot be generated/validated | record candidate failure |
| `domain_incompatible` | requested radius/condition does not fit the current 11 mm representative cell | record expected domain outcome |
| `mechanics_failure` | candidate-dependent Newton/trajectory failure | record candidate failure |
| `optics_failure` | expected candidate optical objective pathology, such as a singular zero field | record candidate failure |
| `VolumeMeshDependencyError`, `PhysicsDependencyError`, or `Transport3DDependencyError` | shared Gmsh/Newton-Warp/OptiX runtime infrastructure is unavailable | raise to `CampaignInfrastructureError`, abort campaign, and do not poison morphology registry |
| optical geometry/physics/trace/result contract error or unexpected exception | implementation/runtime correctness is uncertain | abandon the active Ax trial, propagate the error, and do not register a candidate outcome |

Do not turn a shared header/device/NVRTC/runtime failure into one failed
morphology. The evaluator intentionally re-raises the infrastructure class so
the Ax boundary can stop before registering a candidate failure.


## Artifact, cache, and provenance rules

- Generated validation, benchmark, campaign, and plot files belong under
  `output/` and remain untracked.
- A morphology cache key includes the contract identity and canonical exact
  six-field morphology values. The contract identity includes all non-search
  model inputs, LED geometry and active emission inputs, mesh policies,
  mechanics settings, transport settings, device, protocol, and objective.
  Do not reuse records across incompatible contracts.
- Mechanics artifacts are raw solver outputs plus provenance; ray tracing consumes
  the exact checkpoint named by its identity and digest.
- Optical fields and weights used by objective calculation remain raw. Any
  plotting normalization, smoothing, ray sampling, glow, or color transform is
  display-only and must not enter an artifact or metric.
- `validation/reference/kratos3d` is a reference comparison source, not a
  production result cache.
- Preserve valid historical artifacts; when a code defect invalidates a result,
  write a clearly separate replacement under `output/`.


## Validation boundary and commands

Validation scripts are top-level consumers. The main current checks are
documented in [`docs/COMMANDS.md`](COMMANDS.md):

```bash
conda activate lit

./scripts/tools/pytest_lit tests/unit/finger tests/unit/mesh -q
./scripts/tools/pytest_lit tests/unit/contact tests/unit/physics -q
./scripts/tools/pytest_lit tests/unit/ray_tracing tests/unit/optimization -q
./scripts/tools/pytest_lit tests/unit/optimization/test_evaluator.py -q
./scripts/tools/pytest_lit tests/smoke/physics -q -m "smoke and physics"
```

Before an unattended OptiX/BO run, use both gates from `COMMANDS.md`:

```bash
conda run -n lit python scripts/tools/optix_doctor.py --json
conda run -n lit python -m scripts.tools.optix_smoke
```

The fixed-depth trajectory validation and bounded 6D test BO are validation
workflows, not ordinary unit tests or permission to start a production BO.

`validation/reference/lumo3d_fixed_state_oracle.py` is retained as a fixed-state
regression oracle for the trajectory evaluator. `validation/physics/
multi_location_sphere_contact.py` is a validation-level orchestration fixture
that reuses the neutral contact and Newton APIs. Neither is the production Ax
evaluation entry point.


## Current deviations from the target map

These are verified current-code deviations, not recommended new architecture:

- The repository retains the fixed-state reference oracle noted above. New candidate
  evaluations must follow the `optimization.evaluator` FULL_3D trajectory
  workflow unless a validation task explicitly names a reference implementation.


## Intentionally absent architecture

Do not recreate any of the following for a new caller:

- `fem/`, `case/`, `examples/`, generic `visualization/`, or `mechanics3d/`
  production packages;
- a second mechanics backend abstraction for the current single Newton path;
- a generic plotting framework in the core packages;
- compatibility imports for deleted 2D/legacy APIs;
- a general-purpose backend/plugin registry;
- a `dict[str, Any]` core runtime schema when a focused dataclass or mapping
  contract is sufficient;
- GUI ownership of geometry, physics, transport, or objective policy.

Historical implementations remain available through repository history or
validation/reference code where explicitly retained.


## Agent change protocol

1. Read this map and the owning package before editing.
2. Identify the canonical type/function that already owns the concept.
3. Preserve units, provenance, raw-data semantics, dependency direction, and
   lazy optional-runtime boundaries.
4. Prefer a task-local function or explicit record over a new abstraction when
   only one workflow needs the behavior.
5. Update this map if ownership, execution flow, public boundaries, or
   dependency rules change.
6. Run only the focused checks authorized by the current task, in `lit`, and
   keep generated evidence under `output/`.

When this map conflicts with source, inspect the current implementation and
update the map only after resolving the ownership discrepancy. Do not use
stale tests or deleted names as a reason to restore old architecture.


## Related documents

- [`AGENTS.md`](../AGENTS.md): repository-wide agent rules and scope controls.
- [`docs/COMMANDS.md`](COMMANDS.md): environments, install options, supported
  commands, external runtime gates, and generated-output locations.
- `docs/`: domain-specific scientific contracts and design rationale.
