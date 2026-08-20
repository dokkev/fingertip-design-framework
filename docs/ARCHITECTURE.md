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
| morphology fields, geometry invariants | `model/` | solver settings, OptiX, GUI |
| Gmsh or neutral mesh records | `mesh/` | mechanics stepping, objective policy |
| first-contact pose and approach geometry | `contact/` | Newton stepping, optical scoring |
| Newton/Warp mechanics | `physics/` | Ax policy, validation reports |
| transport geometry, OptiX, optical results | `optics/` | mechanics imports, campaign policy |
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
model.FingertipParameters
    -> model / mesh geometry
    -> contact.find_first_contact
    -> physics.trajectory.indentation.solve_fingertip_indentation_trajectory
    -> mechanics checkpoint artifacts
    -> optics.transport3d FULL_3D OptiX
    -> optimization.objectives
    -> validation reports or optimization.adapters.ax
```


## Repository map

| Path | Role | Canonical status |
| --- | --- | --- |
| `model/` | raw morphology parameters, solids, material/LED descriptors | production domain source |
| `mesh/` | 2D/3D neutral mesh records, Gmsh volume meshing, rigid geometry | production discretization boundary |
| `contact/` | geometry-derived first-contact and sphere alignment | production contact initialization |
| `physics/` | Newton 1.4 / Warp mechanics and trajectory state | one production mechanics path |
| `optics/` | optical contracts and FULL_3D transport implementation | production BO path |
| `lumo/` | reusable concrete LUMO simulation state and execution orchestration | mechanics/optics implementation and optimization policy |
| `optimization/` | fixed protocol, objective, registry, Ax boundary, evaluator | solver implementation and scientific report generation |
| `validation/` | reports, smoke tests, regression/reference workflows, bounded campaign runners | domain/solver/transport ownership; production evaluation is in `optimization/` |
| `validation/reference/kratos3d/` | preserved 3D Kratos reference implementation | validation-only |
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
| morphology and constraints | `model/fingertip_model.py::FingertipParameters` | `model/fingertip.py`, `model/solid.py` |
| neutral volume mesh | `mesh/volume/contracts.py` | `mesh/volume/mesh.py`, `mesh/volume/state.py` |
| rigid object/carrier mesh | `mesh/rigid/object.py` | `mesh/rigid/carrier.py`, `mesh/rigid/indenter.py` |
| neutral rigid pose | `mesh/rigid/object.py::RigidPose3D` | `contact/`, `physics/` |
| first contact | `contact/first_contact.py` | `contact/sphere_alignment.py` |
| mechanics public API | `physics/trajectory/indentation.py` | `physics/trajectory/fingertip.py`, `physics/contracts/` |
| Newton implementation | `physics/newton/vbd.py` | `physics/newton/session.py`, `physics/newton/viewer.py` |
| FULL_3D transport | `optics/transport3d/transport.py` | `geometry.py`, `fingertip.py`, `optix_backend.py` |
| OptiX runtime/preflight | `optics/optix/runtime.py` | `validation/optics/optix_smoke.py`, `scripts/tools/optix_doctor.py` |
| evaluation protocol | `optimization/protocol.py` | `optimization/mechanics_contract.py` |
| morphology search space | `optimization/design_space.py` | `optimization/evaluation_registry.py` |
| objective | `optimization/objectives.py` | `optimization/evaluator.py` |
| reusable LUMO simulation | `lumo/simulation.py` | `optimization/evaluator.py` |
| Ax campaign boundary | `optimization/adapters/ax.py` | `validation/optimization/lumo6d_test_bo.py` |
| production trajectory evaluator | `optimization/evaluator.py` | `lumo/simulation.py`, `validation/optimization/lumo3d_trajectory_validation.py` |
| persisted mechanics state | `optimization/deformed_state_artifact.py` | evaluator artifact writers |
| reference comparison | `validation/physics/correspondence.py` | `validation/reference/kratos3d/` |
| interactive Newton view | `physics/newton/viewer.py` | example callers, if reintroduced explicitly |


## Package ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `model` | raw `FingertipParameters`, morphology constraints, 2D solid boundaries, optical material/source descriptors | mesh construction, mechanics, optics execution, UI |
| `mesh` | neutral mesh dataclasses, Gmsh-backed volume meshing, fingertip state conversion, sphere/carrier meshes | Newton stepping, OptiX calls, optimization policy |
| `contact` | Shapely-based collision predicate, coarse bracket/bisection, canonical sphere alignment, clear/spawn/contact poses | deformation solve and optical transport |
| `physics` | NumPy-facing Newton/Warp settings/results, prescribed indentation, continuous trajectory checkpoints, contact diagnostics | Ax generation, objective calculation, validation orchestration |
| `optics` | optical boundary/result contracts, FULL_3D surface geometry, CUDA/OptiX runtime and launches | mechanics state evolution and BO decisions |
| `lumo` | concrete `LumoSimulation` state/orchestration and named contact results | contact/Newton/OptiX implementation and optimization policy |
| `optimization` | six-dimensional design space, fixed-depth factorial protocol, mechanics contract, trajectory objective, exact morphology registry, Ax adapter | mesh/solver/transport implementation and simulation state |
| `validation` | reports, smoke/regression/reference workflows, and bounded campaign runners | domain/solver/transport ownership; production packages must not import it |
| `gui` | optional controls and diagnostics presentation | domain rules, solver settings, transport, campaign orchestration |

The current package exports in `model/__init__.py`, `mesh/__init__.py`,
`contact/__init__.py`, `physics/__init__.py`, `optics/__init__.py`, and
`optimization/__init__.py` are the primary lightweight API surfaces. Prefer
those exports or the canonical module named above over new wrapper layers.

`FingertipParameters` stores constructor-level physical fields only. Coordinates
and dimensions derived from those fields are computed explicitly by the owning
geometry, thickness, or reporting consumer; they are not duplicated as public
parameter properties.


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

`LumoSimulation` persists each mechanics checkpoint before using it for optics
and returns named `ContactSimulationResult` / `ContactOpticalState` values.
The evaluator converts those values at the artifact boundary. The optical
artifact includes the mechanics artifact digest, protocol, transport
configuration, contact-state identity, and deformed 3D surface provenance.


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

`optimization.mechanics_contract.DEFAULT_MECHANICS_CONTRACT` owns the frozen
search numerical settings. Do not duplicate or retune those settings in the
evaluator, GUI, or validation scripts.

### Optics

`optics.transport3d` owns production FULL_3D geometry construction, field/path
accumulation, carrier-interface handling, and trace results. The actual runtime
boundary is `optics.optix.runtime.OptixRuntime.create()` and the production
trace path is `trace_geometry()`. Artifact persistence and contract
fingerprints belong to `optimization/optical_artifact.py`.

`optics.optix.runtime` owns only the optional CUDA/OptiX setup and execution
machinery. `validation.optics.optix_smoke` performs the real setup, GAS build,
launch, and hit/miss verification used as the BO preflight. The
`scripts/tools/optix_doctor.py` command diagnoses an environment for human
troubleshooting; it is not part of the production optics path. A shared
CUDA/OptiX dependency failure is campaign-fatal, not a morphology-specific
optics failure.

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
| morphology parameters | `model` | `mesh`, `optimization`, `validation`, `gui` | immutable six-variable design plus explicit constraints; public geometry units are mm |
| neutral volume mesh | `mesh` | `physics`, `optics`, `validation` | NumPy-backed nodes/elements and surface metadata; no solver object |
| first-contact result | `contact` | `physics`, `validation` | geometry-derived poses and post-contact travel; `T_spawn` is clear-side initialization only |
| mechanics result/checkpoint | `physics` | `validation`, `optics` | NumPy arrays and immutable diagnostics; Newton state does not cross into optics |
| deformed state artifact | `optimization/deformed_state_artifact` | `optics.transport3d` | exact checkpoint mesh plus digest and source-node provenance |
| optical result | `optics.transport3d` | `optimization`, `validation` | raw transport fields/weights, energy bookkeeping, and configuration fingerprints |
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
mechanics/optics artifacts or optimization metrics.


## Runtime dependencies

| Dependency | Owner | Boundary |
| --- | --- | --- |
| Gmsh | `mesh` | imported only when volume meshing is requested |
| Newton / Warp | `physics.newton` and execution helpers | required only for mechanics execution; keep public NumPy contracts neutral |
| CuPy / PyOptiX / CUDA Python / NVRTC | `optics.optix` and `optics.transport3d` | runtime/preflight/trace boundary; environment is externally managed |
| Ax 1.3.1 | `optimization.adapters.ax` | optimizer execution boundary; importing `optimization` must stay lightweight |
| NiceGUI / Matplotlib | `gui` | optional presentation boundary |
| Kratos | `validation/reference/kratos3d` | validation-only external reference; never a production dependency |

Installation and exact commands belong in `docs/COMMANDS.md`.


## Dependency rules

The allowed high-level direction is:

```text
model
  -> mesh
  -> contact
  -> physics
  -> optics
  -> optimization / validation consumers
```

This diagram is a consumption direction, not a requirement that every package
import every predecessor. In particular, optics consumes neutral model/mesh
and restored mechanics artifacts at orchestration boundaries; optics does not
import `physics`. Validation is the top-level scientific consumer and may
compose all production packages.

Important guards:

- production packages do not import `validation` or `tests`;
- `model` remains geometry/model-only and does not import `mesh`, `physics`,
  `optics`, plotting, Gmsh, or Kratos;
- `mesh` does not import mechanics, optics, validation, plotting, or Kratos;
- `physics` does not import optics, validation, or tests;
- `optics` does not import physics, validation, or tests;
- low-level packages do not import GUI code;
- validation may compose production APIs and its own workflow helpers; production code never imports validation;
- no new generic `utils`, backend registry, compatibility package, or
  cross-layer wrapper is justified without a concrete current consumer;
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
| `optics_failure` | candidate-dependent geometry/physics/trace result failure | record candidate failure |
| `VolumeMeshDependencyError`, `PhysicsDependencyError`, or `Transport3DDependencyError` | shared Gmsh/Newton-Warp/OptiX runtime infrastructure is unavailable | raise to `CampaignInfrastructureError`, abort campaign, and do not poison morphology registry |

Do not turn a shared header/device/NVRTC/runtime failure into one failed
morphology. The evaluator intentionally re-raises the infrastructure class so
the Ax boundary can stop before registering a candidate failure.


## Artifact, cache, and provenance rules

- Generated validation, benchmark, campaign, and plot files belong under
  `output/` and remain untracked.
- A morphology cache key includes the contract identity and canonical exact
  six-field morphology values. Do not reuse records across incompatible
  protocol, mechanics, transport, or objective fingerprints.
- Mechanics artifacts are raw solver outputs plus provenance; optics consumes
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

./scripts/tools/pytest_lit tests/unit/model tests/unit/mesh -q
./scripts/tools/pytest_lit tests/unit/contact tests/unit/physics -q
./scripts/tools/pytest_lit tests/unit/optics tests/unit/optimization -q
./scripts/tools/pytest_lit tests/unit/optimization/test_evaluator.py -q
./scripts/tools/pytest_lit tests/smoke/physics -q -m "smoke and physics"
```

Before an unattended OptiX/BO run, use both gates from `COMMANDS.md`:

```bash
conda run -n lit python scripts/tools/optix_doctor.py --json
conda run -n lit python -m validation.optics.optix_smoke
```

The fixed-depth trajectory validation and bounded 6D test BO are validation
workflows, not ordinary unit tests or permission to start a production BO.

`validation/optimization/lumo3d_evaluator.py` is retained as a fixed-state
regression oracle for the trajectory evaluator. `validation/physics/
multi_location_sphere_contact.py` is a validation-level orchestration fixture
that reuses the neutral contact and Newton APIs. Neither is the production Ax
evaluation entry point.


## Current deviations from the target map

These are verified current-code deviations, not recommended new architecture:

- The repository retains the fixed-state evaluator noted above. New candidate
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
