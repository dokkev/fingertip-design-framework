"""Fixed morphology-search loading trajectories and captured contact states."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from case.state import ContactState


def _finite_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


class ContactScenario(ContactState):
    """One captured loaded state, retained for neutral optics provenance."""


@dataclass(frozen=True, order=True)
class TrajectoryScenario:
    """One monotonic displacement-controlled trajectory."""

    location_x_mm: float
    indenter_radius_mm: float
    maximum_indentation_mm: float = 2.0

    def __post_init__(self) -> None:
        location = _finite_value("location_x_mm", self.location_x_mm)
        radius = _finite_value("indenter_radius_mm", self.indenter_radius_mm)
        maximum = _finite_value(
            "maximum_indentation_mm", self.maximum_indentation_mm
        )
        if radius <= 0.0 or maximum <= 0.0:
            raise ValueError("trajectory radius and maximum indentation must be positive")
        object.__setattr__(self, "location_x_mm", location)
        object.__setattr__(self, "indenter_radius_mm", radius)
        object.__setattr__(self, "maximum_indentation_mm", maximum)

    @property
    def diameter_mm(self) -> float:
        return 2.0 * self.indenter_radius_mm


def _validated_levels(
    name: str,
    values: tuple[float, ...],
    *,
    positive: bool,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    resolved = tuple(
        _finite_value(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )
    if positive and any(value <= 0.0 for value in resolved):
        raise ValueError(f"{name} values must be positive")
    if any(first >= second for first, second in zip(resolved, resolved[1:])):
        raise ValueError(f"{name} values must be strictly increasing")
    return resolved


@dataclass(frozen=True)
class ScenarioGrid:
    """The production 12-trajectory, 48-state loading protocol.

    Defaults are diameters ``6, 10, 14, 20`` mm, one-sided locations
    ``0, 1.5, 3`` mm, and captures at ``0.5, 1, 1.5, 2`` mm. Custom levels
    remain useful for focused synthetic tests; production evaluation rejects
    a grid that is not the complete protocol.
    """

    locations_x_mm: tuple[float, ...] = (0.0, 1.5, 3.0)
    indenter_radii_mm: tuple[float, ...] = (3.0, 5.0, 7.0, 10.0)
    captured_depths_mm: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    maximum_indentation_mm: float = 2.0

    def __post_init__(self) -> None:
        locations = _validated_levels(
            "locations_x_mm", self.locations_x_mm, positive=False
        )
        radii = _validated_levels(
            "indenter_radii_mm", self.indenter_radii_mm, positive=True
        )
        depths = _validated_levels(
            "captured_depths_mm", self.captured_depths_mm, positive=True
        )
        maximum = _finite_value(
            "maximum_indentation_mm", self.maximum_indentation_mm
        )
        if maximum <= 0.0:
            raise ValueError("maximum_indentation_mm must be positive")
        if depths[-1] > maximum + 1.0e-12:
            raise ValueError("captured depths cannot exceed maximum indentation")
        if abs(depths[-1] - maximum) > 1.0e-12:
            raise ValueError(
                "the final captured depth must equal maximum_indentation_mm"
            )
        object.__setattr__(self, "locations_x_mm", locations)
        object.__setattr__(self, "indenter_radii_mm", radii)
        object.__setattr__(self, "captured_depths_mm", depths)
        object.__setattr__(self, "maximum_indentation_mm", maximum)

    @property
    def trajectories(self) -> tuple[TrajectoryScenario, ...]:
        """Return one trajectory for every radius/location combination."""
        return tuple(
            TrajectoryScenario(
                location_x_mm=location,
                indenter_radius_mm=radius,
                maximum_indentation_mm=self.maximum_indentation_mm,
            )
            for radius in self.indenter_radii_mm
            for location in self.locations_x_mm
        )

    @property
    def captured_states(self) -> tuple[ContactScenario, ...]:
        """Return the captured state labels in trajectory/depth order."""
        return tuple(
            ContactScenario(
                trajectory.location_x_mm,
                depth,
                trajectory.indenter_radius_mm,
            )
            for trajectory in self.trajectories
            for depth in self.captured_depths_mm
        )

    @property
    def trajectory_count(self) -> int:
        return len(self.trajectories)

    @property
    def captured_state_count(self) -> int:
        return self.trajectory_count * len(self.captured_depths_mm)

    @property
    def is_production_protocol(self) -> bool:
        return (
            self.locations_x_mm == (0.0, 1.5, 3.0)
            and self.indenter_radii_mm == (3.0, 5.0, 7.0, 10.0)
            and self.captured_depths_mm == (0.5, 1.0, 1.5, 2.0)
            and self.maximum_indentation_mm == 2.0
        )


__all__ = ["ContactScenario", "ScenarioGrid", "TrajectoryScenario"]
