"""Algorithm-independent optomechanical design evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

from fem import solve
from fem.contact import InternalContactTopologyError
from fem.indentation import InvalidIndentationSettings
from fem.kratos_adapter import KratosAdapterError, KratosDependencyError
from fem.results import IndentationPostprocessError
from mesh import InvalidPadMesh, InvalidMeshSettings, MeshSettings
from mesh.fingertip import (
    FingertipMeshingError,
    GmshDependencyError,
)
from mesh.indenter import IndenterMeshingError, IndenterSettings, InvalidIndenterSettings
from model import (
    Fingertip,
    FingertipParameters,
    InvalidFingertip,
    InvalidFingertipParameters,
)
from model.fingertip_model import InvalidFingertipGeometry
from optics import TraceSettings, evaluate as evaluate_transport, field_difference, trace
from optics.cross_section.domain import CrossSectionOpticsError
from optics.metrics import OpticalMetricError

from optimization.scenarios import ContactScenario, ScenarioGrid, ScenarioPair


EvaluationStatus = Literal[
    "success",
    "invalid_design",
    "mesh_failure",
    "fea_failure",
    "optics_failure",
]


def _finite_metric(name: str, value: Real, *, bounded: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if bounded and not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return resolved


@dataclass(frozen=True)
class ScenarioEvaluation:
    """Compact diagnostics for one successfully simulated contact scenario."""

    scenario: ContactScenario
    detectability: float
    reaction_force_n: float | None
    centroid_shift_mm: float
    escaped_fraction_change: float
    absorbed_fraction_change: float

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, ContactScenario):
            raise TypeError("scenario must be a ContactScenario")
        object.__setattr__(
            self,
            "detectability",
            _finite_metric("detectability", self.detectability, bounded=True),
        )
        if self.reaction_force_n is not None:
            object.__setattr__(
                self,
                "reaction_force_n",
                _finite_metric("reaction_force_n", self.reaction_force_n),
            )
        for name in (
            "centroid_shift_mm",
            "escaped_fraction_change",
            "absorbed_fraction_change",
        ):
            object.__setattr__(
                self,
                name,
                _finite_metric(name, getattr(self, name)),
            )


@dataclass(frozen=True)
class PairEvaluation:
    """Optical separability for one required adjacent scenario pair."""

    pair: ScenarioPair
    separability: float

    def __post_init__(self) -> None:
        if not isinstance(self.pair, ScenarioPair):
            raise TypeError("pair must be a ScenarioPair")
        object.__setattr__(
            self,
            "separability",
            _finite_metric("separability", self.separability, bounded=True),
        )


@dataclass(frozen=True)
class DesignEvaluation:
    """Immutable compact result for one complete design evaluation."""

    status: EvaluationStatus
    score: float | None
    minimum_separability: float | None
    mean_separability: float | None
    median_separability: float | None
    minimum_detectability: float | None
    limiting_pair: ScenarioPair | None
    scenarios: tuple[ScenarioEvaluation, ...]
    pairs: tuple[PairEvaluation, ...]
    failure_message: str | None

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
                _finite_metric("score", self.score, bounded=True),
            )
        for name in (
            "minimum_separability",
            "mean_separability",
            "median_separability",
            "minimum_detectability",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_metric(name, value, bounded=True),
                )
        if self.limiting_pair is not None and not isinstance(
            self.limiting_pair, ScenarioPair
        ):
            raise TypeError("limiting_pair must be a ScenarioPair or None")


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
_OPTICS_ERRORS = (CrossSectionOpticsError, OpticalMetricError)


def _failure(
    status: EvaluationStatus,
    message: str,
    *,
    scenarios: tuple[ScenarioEvaluation, ...] = (),
    pairs: tuple[PairEvaluation, ...] = (),
) -> DesignEvaluation:
    return DesignEvaluation(
        status=status,
        score=None,
        minimum_separability=None,
        mean_separability=None,
        median_separability=None,
        minimum_detectability=None,
        limiting_pair=None,
        scenarios=scenarios,
        pairs=pairs,
        failure_message=message,
    )


class DesignEvaluator:
    """Evaluate one physical fingertip against a fixed scenario protocol."""

    def __init__(
        self,
        scenario_grid: ScenarioGrid,
        *,
        mesh_settings: MeshSettings,
        trace_settings: TraceSettings,
        fem_steps: int = 48,
        internal_contact: str = "three_pairs",
    ) -> None:
        if not isinstance(scenario_grid, ScenarioGrid):
            raise TypeError("scenario_grid must be a ScenarioGrid")
        if not scenario_grid.adjacent_pairs:
            raise ValueError(
                "scenario_grid must contain at least one required adjacent pair"
            )
        if not isinstance(mesh_settings, MeshSettings):
            raise TypeError("mesh_settings must be MeshSettings")
        if not isinstance(trace_settings, TraceSettings):
            raise TypeError("trace_settings must be TraceSettings")
        if (
            not isinstance(fem_steps, int)
            or isinstance(fem_steps, bool)
            or fem_steps <= 0
        ):
            raise ValueError("fem_steps must be a positive integer")
        self.scenario_grid = scenario_grid
        self.mesh_settings = mesh_settings
        self.trace_settings = trace_settings
        self.fem_steps = fem_steps
        self.internal_contact = internal_contact

    def evaluate(self, parameters: FingertipParameters) -> DesignEvaluation:
        """Run one deterministic mesh/FEM/optical design evaluation."""
        try:
            tip = Fingertip(parameters)
        except _DESIGN_ERRORS as exc:
            return _failure("invalid_design", f"{type(exc).__name__}: {exc}")

        try:
            mesh = tip.mesh(self.mesh_settings)
        except _MESH_ERRORS as exc:
            return _failure("mesh_failure", f"{type(exc).__name__}: {exc}")

        try:
            reference_transport = trace(
                tip,
                mesh,
                settings=self.trace_settings,
            )
        except _OPTICS_ERRORS as exc:
            return _failure("optics_failure", f"{type(exc).__name__}: {exc}")

        scenario_results: list[ScenarioEvaluation] = []
        loaded_transport: dict[ContactScenario, object] = {}
        for scenario in self.scenario_grid.scenarios:
            indenter = IndenterSettings(radius_mm=scenario.indenter_radius_mm)
            try:
                fea = solve(
                    tip,
                    mesh,
                    indentation=scenario.indentation_mm,
                    surface_x_mm=scenario.location_x_mm,
                    steps=self.fem_steps,
                    indenter=indenter,
                    internal_contact=self.internal_contact,
                )
            except _MESH_ERRORS as exc:
                return _failure(
                    "mesh_failure",
                    f"scenario {scenario}: {type(exc).__name__}: {exc}",
                    scenarios=tuple(scenario_results),
                )
            except _FEA_ERRORS as exc:
                return _failure(
                    "fea_failure",
                    f"scenario {scenario}: {type(exc).__name__}: {exc}",
                    scenarios=tuple(scenario_results),
                )

            if not fea.converged:
                return _failure(
                    "fea_failure",
                    f"scenario {scenario}: FEM solve did not converge",
                    scenarios=tuple(scenario_results),
                )

            try:
                loaded = trace(
                    tip,
                    fea.deformed_mesh,
                    settings=self.trace_settings,
                )
                metrics = evaluate_transport(reference_transport, loaded)
                scenario_result = ScenarioEvaluation(
                    scenario=scenario,
                    detectability=metrics["field_difference"],
                    reaction_force_n=fea.reaction_force,
                    centroid_shift_mm=metrics["centroid_shift_mm"],
                    escaped_fraction_change=metrics["escaped_fraction_change"],
                    absorbed_fraction_change=metrics["absorbed_fraction_change"],
                )
            except _OPTICS_ERRORS as exc:
                return _failure(
                    "optics_failure",
                    f"scenario {scenario}: {type(exc).__name__}: {exc}",
                    scenarios=tuple(scenario_results),
                )
            scenario_results.append(scenario_result)
            loaded_transport[scenario] = loaded

        pair_results: list[PairEvaluation] = []
        for pair in self.scenario_grid.adjacent_pairs:
            try:
                pair_result = PairEvaluation(
                    pair=pair,
                    separability=field_difference(
                        loaded_transport[pair.first],
                        loaded_transport[pair.second],
                    ),
                )
            except _OPTICS_ERRORS as exc:
                return _failure(
                    "optics_failure",
                    f"pair {pair}: {type(exc).__name__}: {exc}",
                    scenarios=tuple(scenario_results),
                    pairs=tuple(pair_results),
                )
            pair_results.append(pair_result)

        separabilities = tuple(result.separability for result in pair_results)
        minimum_separability = min(separabilities)
        limiting_pair = next(
            result.pair
            for result in pair_results
            if result.separability == minimum_separability
        )
        detectabilities = tuple(result.detectability for result in scenario_results)
        ordered_separabilities = sorted(separabilities)
        midpoint = len(ordered_separabilities) // 2
        if len(ordered_separabilities) % 2:
            median_separability = ordered_separabilities[midpoint]
        else:
            median_separability = 0.5 * (
                ordered_separabilities[midpoint - 1]
                + ordered_separabilities[midpoint]
            )
        return DesignEvaluation(
            status="success",
            score=minimum_separability,
            minimum_separability=minimum_separability,
            mean_separability=sum(separabilities) / len(separabilities),
            median_separability=median_separability,
            minimum_detectability=min(detectabilities),
            limiting_pair=limiting_pair,
            scenarios=tuple(scenario_results),
            pairs=tuple(pair_results),
            failure_message=None,
        )


__all__ = [
    "DesignEvaluation",
    "DesignEvaluator",
    "EvaluationStatus",
    "PairEvaluation",
    "ScenarioEvaluation",
]
