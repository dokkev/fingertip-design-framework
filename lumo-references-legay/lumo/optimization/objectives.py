"""Typed trajectory observations and objective calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ObjectiveIdentifier:
    """Stable identity for one objective definition and its interpretation."""

    name: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("objective name must be a non-empty string")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("objective version must be an integer")
        if self.version < 1:
            raise ValueError("objective version must be positive")

    @property
    def serialized_name(self) -> str:
        return f"{self.name}_v{self.version}"


TRAJECTORY_SEPARATION_OBJECTIVE = ObjectiveIdentifier(
    name="trajectory_separation_margin_fixed_depth",
    version=1,
)


@dataclass(frozen=True)
class TrajectoryObjectiveConfig:
    """Configuration for the canonical trajectory-separation objective."""

    radius_penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        value = float(self.radius_penalty_weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("radius_penalty_weight must be finite and non-negative")
        object.__setattr__(self, "radius_penalty_weight", value)


@dataclass(frozen=True)
class TrajectoryObservationKey:
    """Physical labels identifying one trajectory observation."""

    location_u: float
    radius_mm: float
    checkpoint_depth_mm: float

    def __post_init__(self) -> None:
        for name in ("location_u", "radius_mm", "checkpoint_depth_mm"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.location_u <= 1.0:
            raise ValueError("location_u must lie in [0, 1]")
        if self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive")
        if self.checkpoint_depth_mm <= 0.0:
            raise ValueError("checkpoint_depth_mm must be positive")


@dataclass(frozen=True)
class TrajectoryObservation:
    """One optical field and required transport values for the objective."""

    location_u: float
    radius_mm: float
    checkpoint_depth_mm: float
    field: np.ndarray
    total_transport: float
    escaped_weight: float
    debug_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        field = np.array(self.field, dtype=float, copy=True)
        if field.ndim < 1 or not np.all(np.isfinite(field)) or np.any(field < 0.0):
            raise ValueError("observation field must be finite and non-negative")
        field.setflags(write=False)
        object.__setattr__(self, "field", field)
        key = TrajectoryObservationKey(
            self.location_u,
            self.radius_mm,
            self.checkpoint_depth_mm,
        )
        object.__setattr__(self, "location_u", key.location_u)
        object.__setattr__(self, "radius_mm", key.radius_mm)
        object.__setattr__(self, "checkpoint_depth_mm", key.checkpoint_depth_mm)
        for name in ("total_transport", "escaped_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.debug_diagnostics is not None:
            if not isinstance(self.debug_diagnostics, Mapping):
                raise TypeError("debug_diagnostics must be a mapping or None")
            object.__setattr__(self, "debug_diagnostics", dict(self.debug_diagnostics))

    @property
    def key(self) -> TrajectoryObservationKey:
        return TrajectoryObservationKey(
            self.location_u,
            self.radius_mm,
            self.checkpoint_depth_mm,
        )


@dataclass(frozen=True)
class TrajectoryPairDistance:
    """Distance and labels for one compared observation pair."""

    first: TrajectoryObservationKey
    second: TrajectoryObservationKey
    distance: float

    def __post_init__(self) -> None:
        value = float(self.distance)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("pair distance must be finite and non-negative")
        object.__setattr__(self, "distance", value)


def normalized_field_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the native normalized redistribution L1 distance."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape:
        raise ValueError("observation fields must share a shape")
    left_mass = float(np.sum(left))
    right_mass = float(np.sum(right))
    if not left_mass > 0.0 or not right_mass > 0.0:
        raise ValueError("normalized field distance is undefined for zero mass")
    return 0.5 * float(np.sum(np.abs(left / left_mass - right / right_mass)))


@dataclass(frozen=True)
class TrajectoryObjectiveResult:
    """Immutable objective value and typed diagnostics for one trajectory set."""

    objective: ObjectiveIdentifier
    objective_value: float | None
    d_inter: float | None
    d_radius: float | None
    worst_inter_location_pair: TrajectoryPairDistance | None
    worst_radius_pair: TrajectoryPairDistance | None
    all_pairwise_distances: tuple[float, ...]
    objective_pathology: bool
    objective_pathology_reason: str | None
    objective_pathology_state_indices: tuple[int, ...]
    minimum_field_mass: float
    maximum_field_mass: float
    observation_count: int
    observation_diagnostics: tuple[Mapping[str, Any], ...]
    radius_penalty_weight: float

    @property
    def objective_name(self) -> str:
        """Return the boundary name derived from the typed identifier."""

        return self.objective.serialized_name

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON/report representation at the persistence boundary."""

        def pair_record(pair: TrajectoryPairDistance | None) -> dict[str, Any] | None:
            if pair is None:
                return None
            return {
                "first": {
                    "location_u": pair.first.location_u,
                    "radius_mm": pair.first.radius_mm,
                    "checkpoint_depth_mm": pair.first.checkpoint_depth_mm,
                },
                "second": {
                    "location_u": pair.second.location_u,
                    "radius_mm": pair.second.radius_mm,
                    "checkpoint_depth_mm": pair.second.checkpoint_depth_mm,
                },
                "distance": pair.distance,
            }

        return {
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
            "D_inter": self.d_inter,
            "D_radius": self.d_radius,
            "worst_inter_location_pair": pair_record(self.worst_inter_location_pair),
            "worst_radius_pair": pair_record(self.worst_radius_pair),
            "all_pairwise_distances": list(self.all_pairwise_distances),
            "objective_pathology": self.objective_pathology,
            "objective_pathology_reason": self.objective_pathology_reason,
            "objective_pathology_state_indices": list(
                self.objective_pathology_state_indices
            ),
            "minimum_field_mass": self.minimum_field_mass,
            "maximum_field_mass": self.maximum_field_mass,
            "observation_count": self.observation_count,
            "observation_diagnostics": list(self.observation_diagnostics),
            "radius_penalty_weight": self.radius_penalty_weight,
        }


def _pair_record(
    first: TrajectoryObservation,
    second: TrajectoryObservation,
    distance: float,
) -> TrajectoryPairDistance:
    return TrajectoryPairDistance(first.key, second.key, distance)


def _compute_trajectory_objective(
    observations: Iterable[TrajectoryObservation],
    config: TrajectoryObjectiveConfig,
    objective: ObjectiveIdentifier,
) -> TrajectoryObjectiveResult:
    items = tuple(observations)
    if not items:
        raise ValueError("at least one trajectory observation is required")
    if any(not isinstance(item, TrajectoryObservation) for item in items):
        raise TypeError("observations must contain TrajectoryObservation values")
    field_masses = tuple(float(np.sum(item.field)) for item in items)
    diagnostics = tuple(item.debug_diagnostics or {} for item in items)
    zero_mass_indices = [
        index for index, mass in enumerate(field_masses) if mass <= 1.0e-12
    ]
    if zero_mass_indices:
        return TrajectoryObjectiveResult(
            objective=objective,
            objective_value=None,
            d_inter=None,
            d_radius=None,
            worst_inter_location_pair=None,
            worst_radius_pair=None,
            all_pairwise_distances=(),
            objective_pathology=True,
            objective_pathology_reason="near-total signal extinction",
            objective_pathology_state_indices=tuple(zero_mass_indices),
            minimum_field_mass=float(min(field_masses)),
            maximum_field_mass=float(max(field_masses)),
            observation_count=len(items),
            observation_diagnostics=diagnostics,
            radius_penalty_weight=config.radius_penalty_weight,
        )
    inter: list[tuple[float, TrajectoryObservation, TrajectoryObservation]] = []
    radius: list[tuple[float, TrajectoryObservation, TrajectoryObservation]] = []
    all_distances: list[float] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            distance = normalized_field_distance(first.field, second.field)
            all_distances.append(distance)
            if first.location_u != second.location_u:
                inter.append((distance, first, second))
            if (
                first.location_u == second.location_u
                and first.checkpoint_depth_mm == second.checkpoint_depth_mm
                and first.radius_mm != second.radius_mm
            ):
                radius.append((distance, first, second))
    if not inter:
        raise ValueError("at least two contact locations are required for D_inter")
    d_inter, inter_first, inter_second = min(inter, key=lambda item: item[0])
    d_radius = max((item[0] for item in radius), default=0.0)
    radius_pair = max(radius, key=lambda item: item[0]) if radius else None
    extinct_states = [
        index
        for index, item in enumerate(items)
        if item.total_transport <= 1.0e-6
        or item.escaped_weight <= 1.0e-6
        or field_masses[index] <= 1.0e-3
    ]
    pathology = bool(extinct_states) or all(mass <= 1.0e-12 for mass in field_masses)
    objective_value = d_inter - config.radius_penalty_weight * d_radius
    return TrajectoryObjectiveResult(
        objective=objective,
        objective_value=float(objective_value),
        d_inter=float(d_inter),
        d_radius=float(d_radius),
        worst_inter_location_pair=_pair_record(inter_first, inter_second, d_inter),
        worst_radius_pair=(
            None
            if radius_pair is None
            else _pair_record(radius_pair[1], radius_pair[2], radius_pair[0])
        ),
        all_pairwise_distances=tuple(float(value) for value in all_distances),
        objective_pathology=bool(pathology),
        objective_pathology_reason=(
            "near-total signal extinction" if pathology else None
        ),
        objective_pathology_state_indices=tuple(extinct_states),
        minimum_field_mass=float(min(field_masses)),
        maximum_field_mass=float(max(field_masses)),
        observation_count=len(items),
        observation_diagnostics=diagnostics,
        radius_penalty_weight=config.radius_penalty_weight,
    )


def compute_trajectory_objective(
    observations: Iterable[TrajectoryObservation],
    config: TrajectoryObjectiveConfig | None = None,
) -> TrajectoryObjectiveResult:
    """Evaluate the canonical trajectory-separation objective."""

    return _compute_trajectory_objective(
        observations,
        config or TrajectoryObjectiveConfig(),
        TRAJECTORY_SEPARATION_OBJECTIVE,
    )


__all__ = [
    "ObjectiveIdentifier",
    "TRAJECTORY_SEPARATION_OBJECTIVE",
    "TrajectoryObjectiveConfig",
    "TrajectoryObservationKey",
    "TrajectoryObservation",
    "TrajectoryPairDistance",
    "TrajectoryObjectiveResult",
    "compute_trajectory_objective",
    "normalized_field_distance",
]
