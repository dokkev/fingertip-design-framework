"""Pure trajectory observations and objective calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np


OBJECTIVE_NAME = "trajectory_separation_margin_fixed_depth_v1"


@dataclass(frozen=True)
class TrajectoryObjectiveConfig:
    radius_penalty_weight: float = 1.0
    version: str = "trajectory-separation-margin-v1"

    def __post_init__(self) -> None:
        value = float(self.radius_penalty_weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("radius_penalty_weight must be finite and non-negative")
        object.__setattr__(self, "radius_penalty_weight", value)


@dataclass(frozen=True)
class TrajectoryObservation:
    """One normalized optical field with physical trajectory labels."""

    location_u: float
    radius_mm: float
    checkpoint_depth_mm: float
    field: np.ndarray
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        field = np.array(self.field, dtype=float, copy=True)
        if field.ndim < 1 or not np.all(np.isfinite(field)) or np.any(field < 0.0):
            raise ValueError("observation field must be finite and non-negative")
        field.setflags(write=False)
        object.__setattr__(self, "field", field)
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
    """Immutable objective value and diagnostics for one trajectory set."""

    objective_name: str
    objective_value: float | None
    d_inter: float | None
    d_radius: float | None
    worst_inter_location_pair: Mapping[str, Any] | None
    worst_radius_pair: Mapping[str, Any] | None
    all_pairwise_distances: tuple[float, ...]
    objective_pathology: bool
    objective_pathology_reason: str | None
    objective_pathology_state_indices: tuple[int, ...]
    minimum_field_mass: float
    maximum_field_mass: float
    observation_count: int
    observation_diagnostics: tuple[Mapping[str, Any], ...]
    radius_penalty_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON/report representation at the persistence boundary."""

        return {
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
            "D_inter": self.d_inter,
            "D_radius": self.d_radius,
            "worst_inter_location_pair": self.worst_inter_location_pair,
            "worst_radius_pair": self.worst_radius_pair,
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


def _pair_record(first: TrajectoryObservation, second: TrajectoryObservation, distance: float) -> dict[str, Any]:
    return {
        "first": {
            "location_u": first.location_u,
            "radius_mm": first.radius_mm,
            "checkpoint_depth_mm": first.checkpoint_depth_mm,
        },
        "second": {
            "location_u": second.location_u,
            "radius_mm": second.radius_mm,
            "checkpoint_depth_mm": second.checkpoint_depth_mm,
        },
        "distance": float(distance),
    }


def compute_trajectory_objective(
    observations: Iterable[TrajectoryObservation],
    config: TrajectoryObjectiveConfig | None = None,
) -> TrajectoryObjectiveResult:
    """Compute inter-location separation minus radius nuisance variation."""

    selected_config = config or TrajectoryObjectiveConfig()
    items = tuple(observations)
    if not items:
        raise ValueError("at least one trajectory observation is required")
    if any(not isinstance(item, TrajectoryObservation) for item in items):
        raise TypeError("observations must contain TrajectoryObservation values")
    field_masses = tuple(float(np.sum(item.field)) for item in items)
    diagnostics = tuple(item.diagnostics or {} for item in items)
    zero_mass_indices = [index for index, mass in enumerate(field_masses) if mass <= 1.0e-12]
    if zero_mass_indices:
        return TrajectoryObjectiveResult(
            objective_name=OBJECTIVE_NAME,
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
            radius_penalty_weight=selected_config.radius_penalty_weight,
        )
    inter: list[tuple[float, TrajectoryObservation, TrajectoryObservation]] = []
    radius: list[tuple[float, TrajectoryObservation, TrajectoryObservation]] = []
    all_distances: list[float] = []
    for index, first in enumerate(items):
        for second in items[index + 1:]:
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
        for index, diagnostic in enumerate(diagnostics)
        if float(diagnostic.get("total_transport", float("inf"))) <= 1.0e-6
        or float(diagnostic.get("escaped_weight", float("inf"))) <= 1.0e-6
        or field_masses[index] <= 1.0e-3
    ]
    pathology = bool(extinct_states) or all(mass <= 1.0e-12 for mass in field_masses)
    objective = d_inter - selected_config.radius_penalty_weight * d_radius
    return TrajectoryObjectiveResult(
        objective_name=OBJECTIVE_NAME,
        objective_value=float(objective),
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
        radius_penalty_weight=selected_config.radius_penalty_weight,
    )


__all__ = [
    "OBJECTIVE_NAME",
    "TrajectoryObjectiveConfig",
    "TrajectoryObservation",
    "TrajectoryObjectiveResult",
    "compute_trajectory_objective",
    "normalized_field_distance",
]
