"""Algorithm-independent morphology evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Literal, Mapping

import numpy as np

from fem import solve
from fem.contact import InternalContactTopologyError
from fem.indentation import InvalidIndentationSettings
from fem.kratos_settings import validate_basal_interface_configuration
from fem.kratos_adapter import KratosAdapterError, KratosDependencyError
from fem.results import IndentationPostprocessError
from mesh import InvalidPadMesh, InvalidMeshSettings, MeshSettings
from mesh.fingertip import FingertipMeshingError, GmshDependencyError
from mesh.indenter import (
    IndenterMeshingError,
    IndenterSettings,
    InvalidIndenterSettings,
)
from model import (
    Fingertip,
    FingertipParameters,
    InvalidFingertip,
    InvalidFingertipParameters,
    LED,
    OpticalMaterial,
    validate_minimum_silicone_thickness,
)
from model.fingertip_model import InvalidFingertipGeometry
from optics import IndenterOptics
from optics.transport3d import (
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DResult,
    Transport3DResultError,
    Transport3DSettings,
    Transport3DTraceError,
    trace_3d,
)

from optimization.scenarios import ContactScenario, ScenarioGrid, TrajectoryScenario


EvaluationStatus = Literal[
    "success",
    "invalid_design",
    "mesh_failure",
    "fea_failure",
    "optics_failure",
]


def _finite_metric(name: str, value: Real, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return resolved


def _freeze_diagnostics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_diagnostics(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_diagnostics(item) for item in value.tolist())
    if isinstance(value, Real) and not isinstance(value, bool):
        resolved = float(value)
        return resolved if math.isfinite(resolved) else None
    return value


@dataclass(frozen=True)
class StateEvaluation:
    """One of the 48 exact loaded optical states."""

    state: ContactScenario
    contact_metric: float
    reaction_force_n: float | None
    contact_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    optical_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, ContactScenario):
            raise TypeError("state must be a ContactScenario")
        object.__setattr__(
            self,
            "contact_metric",
            _finite_metric("contact_metric", self.contact_metric, nonnegative=True),
        )
        if self.reaction_force_n is not None:
            object.__setattr__(
                self,
                "reaction_force_n",
                _finite_metric("reaction_force_n", self.reaction_force_n),
            )
        object.__setattr__(
            self,
            "contact_diagnostics",
            _freeze_diagnostics(self.contact_diagnostics),
        )
        object.__setattr__(
            self,
            "optical_diagnostics",
            _freeze_diagnostics(self.optical_diagnostics),
        )


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """The four captured metrics and depth AUC for one FEM trajectory."""

    trajectory: TrajectoryScenario
    states: tuple[StateEvaluation, ...]
    auc: float

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TrajectoryScenario):
            raise TypeError("trajectory must be a TrajectoryScenario")
        states = tuple(self.states)
        if any(not isinstance(state, StateEvaluation) for state in states):
            raise TypeError("states must contain StateEvaluation values")
        if any(
            state.state.location_x_mm != self.trajectory.location_x_mm
            or state.state.indenter_radius_mm != self.trajectory.indenter_radius_mm
            for state in states
        ):
            raise ValueError("trajectory states do not match their trajectory")
        object.__setattr__(self, "states", states)
        object.__setattr__(
            self,
            "auc",
            _finite_metric("auc", self.auc, nonnegative=True),
        )


@dataclass(frozen=True)
class DesignEvaluation:
    """Immutable result for one complete morphology evaluation.

    ``score`` is ``J_morph = min A(D,x)`` and is maximized by the search
    backend. A minimization backend should use the explicit cost ``-score``.
    """

    status: EvaluationStatus
    score: float | None
    minimum_auc: float | None
    mean_auc: float | None
    median_auc: float | None
    minimum_raw_contact_metric: float | None
    mean_raw_contact_metric: float | None
    limiting_trajectory: TrajectoryScenario | None
    limiting_diameter_mm: float | None
    limiting_location_x_mm: float | None
    minimum_raw_contact_state: ContactScenario | None
    minimum_raw_contact_depth_mm: float | None
    trajectories: tuple[TrajectoryEvaluation, ...]
    states: tuple[StateEvaluation, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (
            "success",
            "invalid_design",
            "mesh_failure",
            "fea_failure",
            "optics_failure",
        ):
            raise ValueError(f"unsupported evaluation status: {self.status!r}")
        if self.status == "success" and self.score is None:
            raise ValueError("successful evaluation requires a score")
        if self.status != "success" and self.score is not None:
            raise ValueError("failed evaluation score must be None")
        if self.score is not None:
            object.__setattr__(
                self,
                "score",
                _finite_metric("score", self.score, nonnegative=True),
            )
        for name in (
            "minimum_auc",
            "mean_auc",
            "median_auc",
            "minimum_raw_contact_metric",
            "mean_raw_contact_metric",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_metric(name, value, nonnegative=True),
                )
        if self.limiting_trajectory is not None and not isinstance(
            self.limiting_trajectory, TrajectoryScenario
        ):
            raise TypeError("limiting_trajectory must be a TrajectoryScenario or None")
        if self.minimum_raw_contact_state is not None and not isinstance(
            self.minimum_raw_contact_state, ContactScenario
        ):
            raise TypeError(
                "minimum_raw_contact_state must be a ContactScenario or None"
            )
        for name in ("limiting_diameter_mm", "limiting_location_x_mm"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_metric(name, value),
                )
        if self.minimum_raw_contact_depth_mm is not None:
            object.__setattr__(
                self,
                "minimum_raw_contact_depth_mm",
                _finite_metric(
                    "minimum_raw_contact_depth_mm",
                    self.minimum_raw_contact_depth_mm,
                    nonnegative=True,
                ),
            )
        object.__setattr__(self, "trajectories", tuple(self.trajectories))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "diagnostics", _freeze_diagnostics(self.diagnostics))


_DESIGN_ERRORS = (
    InvalidFingertip,
    InvalidFingertipParameters,
    InvalidFingertipGeometry,
)
_MESH_ERRORS = (
    GmshDependencyError,
    FingertipMeshingError,
    InvalidMeshSettings,
    InvalidPadMesh,
    IndenterMeshingError,
)
_FEA_ERRORS = (
    KratosDependencyError,
    KratosAdapterError,
    InternalContactTopologyError,
    IndentationPostprocessError,
    InvalidIndentationSettings,
    InvalidIndenterSettings,
)
_OPTICS_ERRORS = (
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DResultError,
    Transport3DTraceError,
)

# A missing or unusable OptiX runtime is an environment failure, not a
# morphology-specific optical result.  Transport3DDependencyError is
# intentionally excluded from _OPTICS_ERRORS so the campaign can abort before
# registering the candidate in the persistent evaluation registry.


def _failure(
    status: EvaluationStatus,
    message: str,
    *,
    trajectories: tuple[TrajectoryEvaluation, ...] = (),
    states: tuple[StateEvaluation, ...] = (),
) -> DesignEvaluation:
    return DesignEvaluation(
        status=status,
        score=None,
        minimum_auc=None,
        mean_auc=None,
        median_auc=None,
        minimum_raw_contact_metric=None,
        mean_raw_contact_metric=None,
        limiting_trajectory=None,
        limiting_diameter_mm=None,
        limiting_location_x_mm=None,
        minimum_raw_contact_state=None,
        minimum_raw_contact_depth_mm=None,
        trajectories=trajectories,
        states=states,
        failure_message=message,
    )


def _profile_pair(result: Transport3DResult) -> tuple[np.ndarray, np.ndarray]:
    _, left, right = result.lateral_outgoing_profiles()
    return left, right


def _contact_metric(
    reference: Transport3DResult,
    loaded: Transport3DResult,
) -> float:
    """Compute lateral outgoing L1 redistribution using one launch weight."""
    if reference.launched_weight <= 0.0:
        raise Transport3DResultError("reference launched weight must be positive")
    reference_edges, reference_left, reference_right = (
        reference.lateral_outgoing_profiles()
    )
    loaded_edges, loaded_left, loaded_right = loaded.lateral_outgoing_profiles()
    if not np.array_equal(reference_edges, loaded_edges):
        raise Transport3DResultError(
            "reference and loaded lateral profiles must share exactly one binning"
        )
    return float(
        np.sum(np.abs(loaded_left - reference_left))
        + np.sum(np.abs(loaded_right - reference_right))
    ) / float(reference.launched_weight)


def _auc(depths: tuple[float, ...], values: tuple[float, ...]) -> float:
    if len(depths) != len(values) or not depths or depths[0] <= 0.0:
        raise ValueError("AUC requires positive, paired captured depths")
    x = np.asarray((0.0, *depths), dtype=float)
    y = np.asarray((0.0, *values), dtype=float)
    return float(np.sum(0.5 * np.diff(x) * (y[1:] + y[:-1])) / 2.0)


def _quadrant_energy(result: Transport3DResult) -> dict[str, float] | None:
    density = result.internal_z_integrated_path_density
    x_edges = result.internal_path_x_edges_mm
    y_edges = result.internal_path_y_edges_mm
    if density is None or x_edges is None or y_edges is None:
        return None
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    source_x, source_y = result.source_position_mm[:2]
    x_relative = x_centers - source_x
    y_relative = y_centers - source_y
    return {
        "upper_left": float(
            np.sum(
                density[
                    (y_relative[:, None] > 0.0)
                    & (x_relative[None, :] < 0.0)
                ]
            )
        ),
        "upper_right": float(
            np.sum(
                density[
                    (y_relative[:, None] > 0.0)
                    & (x_relative[None, :] > 0.0)
                ]
            )
        ),
        "lower_left": float(
            np.sum(
                density[
                    (y_relative[:, None] < 0.0)
                    & (x_relative[None, :] < 0.0)
                ]
            )
        ),
        "lower_right": float(
            np.sum(
                density[
                    (y_relative[:, None] < 0.0)
                    & (x_relative[None, :] > 0.0)
                ]
            )
        ),
    }


def _transport_diagnostics(
    result: Transport3DResult,
    reference: Transport3DResult | None = None,
) -> dict[str, Any]:
    metadata = result.geometry_metadata
    _, lateral_left, lateral_right = result.lateral_outgoing_profiles()
    lateral_weight = float(np.sum(lateral_left) + np.sum(lateral_right))
    quadrant = _quadrant_energy(result)
    reference_quadrant = None if reference is None else _quadrant_energy(reference)
    quadrant_delta = None
    if quadrant is not None and reference_quadrant is not None:
        quadrant_delta = {
            name: quadrant[name] - reference_quadrant[name]
            for name in quadrant
        }
    return {
        "launched_weight": result.launched_weight,
        "escaped_weight": result.escaped_weight,
        "absorbed_weight": result.absorbed_weight,
        "terminated_weight": result.terminated_weight,
        "outgoing_surface_weight": result.outgoing_surface_weight,
        "lateral_outgoing_weight": lateral_weight,
        "left_lateral_weight": float(np.sum(lateral_left)),
        "right_lateral_weight": float(np.sum(lateral_right)),
        "lateral_throughput": lateral_weight / max(result.launched_weight, 1.0e-30),
        "left_lateral_throughput": float(np.sum(lateral_left)) / max(result.launched_weight, 1.0e-30),
        "right_lateral_throughput": float(np.sum(lateral_right)) / max(result.launched_weight, 1.0e-30),
        "object_absorbed_weight": result.object_absorbed_weight,
        "object_transmitted_weight": result.object_transmitted_weight,
        "object_interface_incident_weight": result.object_interface_incident_weight,
        "object_reflected_weight": result.object_reflected_weight,
        "quadrant": quadrant if quadrant is not None else metadata.get("quadrant"),
        "quadrant_delta": quadrant_delta,
        "energy_balance_error": result.energy_balance_error,
    }


class DesignEvaluator:
    """Evaluate one morphology against twelve trajectories and 48 states."""

    def __init__(
        self,
        scenario_grid: ScenarioGrid,
        *,
        mesh_settings: MeshSettings,
        trace_settings: Transport3DSettings,
        led: LED | None = None,
        optical: OpticalMaterial | None = None,
        indenter_optics: IndenterOptics,
        fem_steps: int = 48,
        internal_contact: str = "sides_separate",
        basal_interface: str = "bonded",
    ) -> None:
        if not isinstance(scenario_grid, ScenarioGrid):
            raise TypeError("scenario_grid must be a ScenarioGrid")
        if not scenario_grid.is_production_protocol:
            raise ValueError(
                "production DesignEvaluator requires the fixed 12-trajectory "
                "and 48-state scenario protocol"
            )
        if not isinstance(mesh_settings, MeshSettings):
            raise TypeError("mesh_settings must be MeshSettings")
        if not isinstance(trace_settings, Transport3DSettings):
            raise TypeError("trace_settings must be Transport3DSettings")
        if trace_settings.mode != "planar":
            raise ValueError("production morphology evaluation requires mode='planar'")
        if led is not None and not isinstance(led, LED):
            raise TypeError("led must be an LED or None")
        if optical is not None and not isinstance(optical, OpticalMaterial):
            raise TypeError("optical must be an OpticalMaterial or None")
        if indenter_optics is None:
            raise ValueError(
                "production morphology evaluation requires explicit indenter_optics"
            )
        if not isinstance(indenter_optics, IndenterOptics):
            raise TypeError("indenter_optics must be an IndenterOptics")
        if fem_steps != 48:
            raise ValueError("production morphology evaluation requires fem_steps=48")
        try:
            basal, internal = validate_basal_interface_configuration(
                basal_interface,
                internal_contact,
            )
        except ValueError as exception:
            raise ValueError(str(exception)) from exception
        if basal != "bonded" or internal != "sides_separate":
            raise ValueError(
                "production morphology evaluation requires bonded+sides_separate"
            )
        self.scenario_grid = scenario_grid
        self.mesh_settings = mesh_settings
        self.trace_settings = trace_settings
        self.led = LED() if led is None else led
        self.optical = OpticalMaterial() if optical is None else optical
        self.indenter_optics = indenter_optics
        self.fem_steps = 48
        self.internal_contact = internal
        self.basal_interface = basal

    def evaluate(self, parameters: FingertipParameters) -> DesignEvaluation:
        """Trace one unloaded state, then all exact captured loaded states."""
        try:
            validate_minimum_silicone_thickness(parameters)
            tip = Fingertip(parameters, led=self.led, optical=self.optical)
        except _DESIGN_ERRORS as exc:
            return _failure("invalid_design", f"{type(exc).__name__}: {exc}")

        try:
            mesh = tip.mesh(self.mesh_settings)
        except _MESH_ERRORS as exc:
            return _failure("mesh_failure", f"{type(exc).__name__}: {exc}")

        try:
            reference = trace_3d(tip, mesh, settings=self.trace_settings)
        except _OPTICS_ERRORS as exc:
            return _failure("optics_failure", f"{type(exc).__name__}: {exc}")

        trajectory_results: list[TrajectoryEvaluation] = []
        state_results: list[StateEvaluation] = []
        for trajectory in self.scenario_grid.trajectories:
            try:
                fea = solve(
                    tip,
                    mesh,
                    indentation=trajectory.maximum_indentation_mm,
                    surface_x_mm=trajectory.location_x_mm,
                    steps=self.fem_steps,
                    indenter=IndenterSettings(radius_mm=trajectory.indenter_radius_mm),
                    internal_contact=self.internal_contact,
                    basal_interface=self.basal_interface,
                )
            except _MESH_ERRORS as exc:
                return _failure(
                    "mesh_failure",
                    f"trajectory {trajectory}: {type(exc).__name__}: {exc}",
                    trajectories=tuple(trajectory_results),
                    states=tuple(state_results),
                )
            except _FEA_ERRORS as exc:
                return _failure(
                    "fea_failure",
                    f"trajectory {trajectory}: {type(exc).__name__}: {exc}",
                    trajectories=tuple(trajectory_results),
                    states=tuple(state_results),
                )
            if not fea.converged:
                return _failure(
                    "fea_failure",
                    f"trajectory {trajectory}: FEM solve did not converge",
                    trajectories=tuple(trajectory_results),
                    states=tuple(state_results),
                )

            trajectory_states: list[StateEvaluation] = []
            for depth in self.scenario_grid.captured_depths_mm:
                state = ContactScenario(
                    trajectory.location_x_mm,
                    depth,
                    trajectory.indenter_radius_mm,
                )
                try:
                    captured = fea.captured_state(depth)
                    loaded = trace_3d(
                        tip,
                        captured.deformed_mesh,
                        reference_mesh=mesh,
                        settings=self.trace_settings,
                        indenter_pose=captured.indenter_pose,
                        indenter_optics=self.indenter_optics,
                    )
                    metric = _contact_metric(reference, loaded)
                except KeyError as exc:
                    return _failure(
                        "fea_failure",
                        f"state {state}: missing captured FEM state: {exc}",
                        trajectories=tuple(trajectory_results),
                        states=tuple(state_results),
                    )
                except _MESH_ERRORS as exc:
                    return _failure(
                        "mesh_failure",
                        f"state {state}: {type(exc).__name__}: {exc}",
                        trajectories=tuple(trajectory_results),
                        states=tuple(state_results),
                    )
                except _OPTICS_ERRORS as exc:
                    return _failure(
                        "optics_failure",
                        f"state {state}: {type(exc).__name__}: {exc}",
                        trajectories=tuple(trajectory_results),
                        states=tuple(state_results),
                    )
                evaluated = StateEvaluation(
                    state=state,
                    contact_metric=metric,
                    reaction_force_n=captured.reaction_force_n,
                    contact_diagnostics={
                        "active_external_node_ids": captured.active_external_node_ids,
                        "active_internal_node_ids": captured.active_internal_node_ids,
                        "contact_groups": captured.contact,
                        "external_contact_width": captured.details[
                            "external_contact_width"
                        ],
                        "depth_mm": captured.depth_mm,
                        "exact_indenter_pose": captured.indenter_pose,
                    },
                    optical_diagnostics=_transport_diagnostics(loaded, reference),
                )
                trajectory_states.append(evaluated)
                state_results.append(evaluated)
            values = tuple(item.contact_metric for item in trajectory_states)
            trajectory_results.append(
                TrajectoryEvaluation(
                    trajectory=trajectory,
                    states=tuple(trajectory_states),
                    auc=_auc(self.scenario_grid.captured_depths_mm, values),
                )
            )

        aucs = tuple(item.auc for item in trajectory_results)
        raw_metrics = tuple(item.contact_metric for item in state_results)
        minimum_auc = min(aucs)
        limiting_trajectory = next(
            item.trajectory for item in trajectory_results if item.auc == minimum_auc
        )
        limiting_state = min(state_results, key=lambda item: item.contact_metric)
        ordered_aucs = sorted(aucs)
        middle = len(ordered_aucs) // 2
        median_auc = (
            ordered_aucs[middle]
            if len(ordered_aucs) % 2
            else 0.5 * (ordered_aucs[middle - 1] + ordered_aucs[middle])
        )
        return DesignEvaluation(
            status="success",
            score=minimum_auc,
            minimum_auc=minimum_auc,
            mean_auc=sum(aucs) / len(aucs),
            median_auc=median_auc,
            minimum_raw_contact_metric=min(raw_metrics),
            mean_raw_contact_metric=sum(raw_metrics) / len(raw_metrics),
            limiting_trajectory=limiting_trajectory,
            limiting_diameter_mm=limiting_trajectory.diameter_mm,
            limiting_location_x_mm=limiting_trajectory.location_x_mm,
            minimum_raw_contact_state=limiting_state.state,
            minimum_raw_contact_depth_mm=limiting_state.state.indentation_mm,
            trajectories=tuple(trajectory_results),
            states=tuple(state_results),
            diagnostics={
                "trajectory_count": len(trajectory_results),
                "captured_state_count": len(state_results),
                "reference_launched_weight": reference.launched_weight,
                "auc_depths_mm": self.scenario_grid.captured_depths_mm,
                "objective": "J_morph=min_AUC_contact",
                "cost_for_minimizer": -minimum_auc,
            },
        )


__all__ = [
    "DesignEvaluation",
    "DesignEvaluator",
    "EvaluationStatus",
    "StateEvaluation",
    "TrajectoryEvaluation",
]
