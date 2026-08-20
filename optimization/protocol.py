"""Authoritative scientific contact and trajectory evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real


PROTOCOL_SCHEMA = "lumo3d-fixed-depth-trajectory-protocol-v1"


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _levels(name: str, values: tuple[float, ...], *, positive: bool) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    resolved = tuple(_finite(f"{name}[{index}]", value) for index, value in enumerate(values))
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} values must be unique")
    if positive and any(value <= 0.0 for value in resolved):
        raise ValueError(f"{name} values must be positive")
    return resolved


@dataclass(frozen=True)
class TrajectoryEvaluationProtocol:
    """Physical conditions evaluated for one morphology.

    ``checkpoint_depths_mm`` are absolute post-contact indentation depths. They
    are deliberately independent of indenter radius, so radius and depth form
    a factorial nuisance design around the contact-location signal.

    This object contains no solver, mesh, OptiX, or Ax settings. The normalized
    indentation ratio is a derived diagnostic only and is never an input to the
    evaluation protocol.
    """

    contact_locations_u: tuple[float, ...]
    indenter_radii_mm: tuple[float, ...]
    checkpoint_depths_mm: tuple[float, ...]
    initial_gap_mm: float = 0.25

    def __post_init__(self) -> None:
        locations = _levels("contact_locations_u", self.contact_locations_u, positive=False)
        if any(value < 0.0 or value > 1.0 for value in locations):
            raise ValueError("contact_locations_u values must lie in [0, 1]")
        radii = _levels("indenter_radii_mm", self.indenter_radii_mm, positive=True)
        depths = _levels("checkpoint_depths_mm", self.checkpoint_depths_mm, positive=True)
        if any(left >= right for left, right in zip(depths, depths[1:])):
            raise ValueError("checkpoint_depths_mm must be strictly increasing")
        gap = _finite("initial_gap_mm", self.initial_gap_mm)
        if gap <= 0.0:
            raise ValueError("initial_gap_mm must be positive")
        object.__setattr__(self, "contact_locations_u", locations)
        object.__setattr__(self, "indenter_radii_mm", radii)
        object.__setattr__(self, "checkpoint_depths_mm", depths)
        object.__setattr__(self, "initial_gap_mm", gap)

    @property
    def trajectory_count(self) -> int:
        return len(self.contact_locations_u) * len(self.indenter_radii_mm)

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoint_depths_mm)

    @property
    def optical_state_count(self) -> int:
        return self.trajectory_count * self.checkpoint_count

    def normalized_indentation_ratios(self, radius_mm: float) -> tuple[float, ...]:
        """Return radius-normalized depths as derived diagnostics."""

        radius = _finite("radius_mm", radius_mm)
        if radius <= 0.0:
            raise ValueError("radius_mm must be positive")
        return tuple(depth / radius for depth in self.checkpoint_depths_mm)

    def checkpoint_travels_mm(self, radius_mm: float | None = None) -> tuple[float, ...]:
        """Return absolute post-contact travels independent of radius.

        The optional radius argument is accepted only as a migration aid for
        callers that used the old derived-travel method. It is validated when
        supplied but cannot affect the returned fixed-depth values.
        """

        if radius_mm is not None:
            radius = _finite("radius_mm", radius_mm)
            if radius <= 0.0:
                raise ValueError("radius_mm must be positive")
        return self.checkpoint_depths_mm

    @property
    def maximum_depth_mm(self) -> float:
        return self.checkpoint_depths_mm[-1]

    @property
    def checkpoint_fractions(self) -> tuple[float, ...]:
        """Derived plotting/scheduling fractions, not scientific inputs."""

        maximum = self.maximum_depth_mm
        return tuple(depth / maximum for depth in self.checkpoint_depths_mm)

    @property
    def trajectories(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (location, radius)
            for radius in self.indenter_radii_mm
            for location in self.contact_locations_u
        )

    def checkpoint_states(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (location, radius, depth)
            for location, radius in self.trajectories
            for depth in self.checkpoint_depths_mm
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "design": "fixed_absolute_depth_factorial",
            "contact_locations_u": list(self.contact_locations_u),
            "indenter_radii_mm": list(self.indenter_radii_mm),
            "checkpoint_depths_mm": list(self.checkpoint_depths_mm),
            "initial_gap_mm": self.initial_gap_mm,
            "derived_checkpoint_fractions": list(self.checkpoint_fractions),
            "derived_normalized_indentation_ratios_by_radius": {
                f"{radius:g}": list(self.normalized_indentation_ratios(radius))
                for radius in self.indenter_radii_mm
            },
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_TRAJECTORY_PROTOCOL = TrajectoryEvaluationProtocol(
    contact_locations_u=(0.25, 0.50, 0.75),
    indenter_radii_mm=(4.0, 5.0),
    checkpoint_depths_mm=(0.5, 1.0, 1.5),
    initial_gap_mm=0.25,
)


__all__ = [
    "DEFAULT_TRAJECTORY_PROTOCOL",
    "PROTOCOL_SCHEMA",
    "TrajectoryEvaluationProtocol",
]
