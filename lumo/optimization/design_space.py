"""Five-dimensional feasible design space for fingertip optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType

from lumo.fingertip.fingertip import Fingertip
from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.util.scalar_validation import require_finite


MAX_FINGERTIP_HEIGHT_MM = 30.0
MINIMUM_SILICONE_THICKNESS_MM = 5.0
_GEOMETRY_PARAMETER_NAMES = (
    "flat_pad_height_mm",
    "semiellipse_height_mm",
    "stem_width_mm",
    "stem_height_mm",
    "void_width_mm",
)


@dataclass(frozen=True)
class ParameterBound:
    """Inclusive lower and upper bounds for one geometry parameter."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        require_finite("lower", self.lower)
        require_finite("upper", self.upper)
        lower = float(self.lower)
        upper = float(self.upper)
        if lower >= upper:
            raise ValueError("parameter bound must satisfy lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class DesignSpace:
    """Current five-parameter fingertip morphology domain.

    ``base_parameters`` owns every fixed physical input. ``geometry_bounds``
    names only the geometry fields varied by Ax. Constructing ``Fingertip`` is
    the authoritative nonlinear geometry check; the complete height and minimum
    silicone thickness are the two optimization-level feasibility limits.
    """

    base_parameters: FingertipParameters = field(default_factory=FingertipParameters)
    geometry_bounds: Mapping[str, ParameterBound] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_parameters, FingertipParameters):
            raise TypeError("base_parameters must be FingertipParameters")
        if not isinstance(self.geometry_bounds, Mapping):
            raise TypeError("geometry_bounds must be a mapping")

        bounds = dict(self.geometry_bounds)
        if set(bounds) != set(_GEOMETRY_PARAMETER_NAMES):
            raise ValueError(
                "geometry_bounds must define exactly the current five design "
                f"parameters: {_GEOMETRY_PARAMETER_NAMES!r}"
            )
        for name, bound in bounds.items():
            if not isinstance(bound, ParameterBound):
                raise TypeError(f"geometry bound for {name!r} must be ParameterBound")
        object.__setattr__(
            self,
            "geometry_bounds",
            MappingProxyType(
                {name: bounds[name] for name in _GEOMETRY_PARAMETER_NAMES}
            ),
        )

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return qualified optimization parameter names in declared order."""
        return tuple(f"geometry.{name}" for name in self.geometry_bounds)

    def to_parameters(self, candidate: Mapping[str, float]) -> FingertipParameters:
        """Construct complete fingertip parameters from one candidate."""
        expected = set(self.variable_names)
        received = set(candidate)
        if received != expected:
            details = []
            if missing := expected - received:
                details.append(f"missing={sorted(missing)!r}")
            if extra := received - expected:
                details.append(f"unexpected={sorted(extra)!r}")
            raise ValueError(
                "candidate must define exactly the free parameters: "
                + ", ".join(details)
            )

        geometry_values = {}
        for name, bound in self.geometry_bounds.items():
            key = f"geometry.{name}"
            value = float(candidate[key])
            if not isfinite(value):
                raise ValueError(f"{key!r} must be finite")
            if not bound.lower <= value <= bound.upper:
                raise ValueError(f"{key!r} lies outside its parameter bounds")
            geometry_values[name] = value

        geometry = replace(self.base_parameters.geometry, **geometry_values)
        return replace(self.base_parameters, geometry=geometry)

    def is_feasible(self, candidate: Mapping[str, float]) -> bool:
        """Return whether a candidate belongs to the physical design domain."""
        try:
            fingertip = Fingertip(self.to_parameters(candidate))
        except (TypeError, ValueError):
            return False
        return (
            fingertip.full_height_mm <= MAX_FINGERTIP_HEIGHT_MM
            and fingertip.silicone.minimum_silicone_thickness_mm
            >= MINIMUM_SILICONE_THICKNESS_MM
        )


__all__ = [
    "DesignSpace",
    "MAX_FINGERTIP_HEIGHT_MM",
    "MINIMUM_SILICONE_THICKNESS_MM",
    "ParameterBound",
]
