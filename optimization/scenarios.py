"""Immutable contact scenarios and their required Cartesian transitions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal


ScenarioAxis = Literal["location", "indentation", "radius"]


def _finite_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


@dataclass(frozen=True, order=True)
class ContactScenario:
    """One prescribed contact location, indentation, and indenter radius."""

    location_x_mm: float
    indentation_mm: float
    indenter_radius_mm: float

    def __post_init__(self) -> None:
        location = _finite_value("location_x_mm", self.location_x_mm)
        indentation = _finite_value("indentation_mm", self.indentation_mm)
        radius = _finite_value("indenter_radius_mm", self.indenter_radius_mm)
        if indentation <= 0.0:
            raise ValueError("indentation_mm must be positive")
        if radius <= 0.0:
            raise ValueError("indenter_radius_mm must be positive")
        object.__setattr__(self, "location_x_mm", location)
        object.__setattr__(self, "indentation_mm", indentation)
        object.__setattr__(self, "indenter_radius_mm", radius)


def _validated_levels(
    name: str,
    values: tuple[float, ...],
    *,
    positive: bool,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    resolved = tuple(_finite_value(f"{name}[{index}]", value) for index, value in enumerate(values))
    if positive and any(value <= 0.0 for value in resolved):
        raise ValueError(f"{name} values must be positive")
    if any(first >= second for first, second in zip(resolved, resolved[1:])):
        raise ValueError(f"{name} values must be strictly increasing")
    return resolved


@dataclass(frozen=True)
class ScenarioGrid:
    """Cartesian scenario levels with deterministic scenario/pair ordering.

    Scenarios are ordered by radius, then indentation, then location. Required
    pairs are ordered by axis (location, indentation, radius), then by the
    unchanged outer levels and the adjacent level index within that axis.
    """

    locations_x_mm: tuple[float, ...]
    indentations_mm: tuple[float, ...]
    indenter_radii_mm: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locations_x_mm",
            _validated_levels("locations_x_mm", self.locations_x_mm, positive=False),
        )
        object.__setattr__(
            self,
            "indentations_mm",
            _validated_levels("indentations_mm", self.indentations_mm, positive=True),
        )
        object.__setattr__(
            self,
            "indenter_radii_mm",
            _validated_levels("indenter_radii_mm", self.indenter_radii_mm, positive=True),
        )

    @property
    def scenarios(self) -> tuple[ContactScenario, ...]:
        """Return scenarios in radius, indentation, location order."""
        return tuple(
            ContactScenario(location, indentation, radius)
            for radius in self.indenter_radii_mm
            for indentation in self.indentations_mm
            for location in self.locations_x_mm
        )

    @property
    def adjacent_pairs(self) -> tuple[ScenarioPair, ...]:
        """Return only adjacent transitions along one scenario axis."""
        pairs: list[ScenarioPair] = []
        for radius in self.indenter_radii_mm:
            for indentation in self.indentations_mm:
                for index in range(len(self.locations_x_mm) - 1):
                    pairs.append(
                        ScenarioPair(
                            ContactScenario(
                                self.locations_x_mm[index], indentation, radius
                            ),
                            ContactScenario(
                                self.locations_x_mm[index + 1], indentation, radius
                            ),
                            "location",
                        )
                    )
        for radius in self.indenter_radii_mm:
            for location in self.locations_x_mm:
                for index in range(len(self.indentations_mm) - 1):
                    pairs.append(
                        ScenarioPair(
                            ContactScenario(
                                location, self.indentations_mm[index], radius
                            ),
                            ContactScenario(
                                location, self.indentations_mm[index + 1], radius
                            ),
                            "indentation",
                        )
                    )
        for indentation in self.indentations_mm:
            for location in self.locations_x_mm:
                for index in range(len(self.indenter_radii_mm) - 1):
                    pairs.append(
                        ScenarioPair(
                            ContactScenario(
                                location, indentation, self.indenter_radii_mm[index]
                            ),
                            ContactScenario(
                                location,
                                indentation,
                                self.indenter_radii_mm[index + 1],
                            ),
                            "radius",
                        )
                    )
        return tuple(pairs)


@dataclass(frozen=True)
class ScenarioPair:
    """One required adjacent transition along exactly one scenario axis."""

    first: ContactScenario
    second: ContactScenario
    axis: ScenarioAxis

    def __post_init__(self) -> None:
        if not isinstance(self.first, ContactScenario) or not isinstance(
            self.second, ContactScenario
        ):
            raise TypeError("first and second must be ContactScenario values")
        if self.axis not in ("location", "indentation", "radius"):
            raise ValueError(f"unsupported scenario axis: {self.axis!r}")
        differences = {
            "location": self.first.location_x_mm != self.second.location_x_mm,
            "indentation": self.first.indentation_mm != self.second.indentation_mm,
            "radius": self.first.indenter_radius_mm != self.second.indenter_radius_mm,
        }
        if sum(differences.values()) != 1 or not differences[self.axis]:
            raise ValueError(
                "scenario pair must differ along exactly its declared axis"
            )


__all__ = ["ContactScenario", "ScenarioAxis", "ScenarioGrid", "ScenarioPair"]
