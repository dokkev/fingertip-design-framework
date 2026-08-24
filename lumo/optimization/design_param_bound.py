"""Design-variable bounds for fingertip optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType

from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.fingertip.geometric_param import FingertipGeometry
from lumo.fingertip.optical_param import LEDParameters
from lumo.util.scalar_validation import require_finite


@dataclass(frozen=True)
class ParameterBound:
    """Lower and upper bounds for one optimization parameter."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        require_finite("lower", self.lower)
        require_finite("upper", self.upper)

        lower = float(self.lower)
        upper = float(self.upper)

        if lower >= upper:
            raise ValueError(
                "parameter bound must satisfy lower < upper"
            )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class DesignParameterBounds:
    """Optimization bounds applied to a complete fingertip definition.

    ``parameters`` stores the complete nominal fingertip. Parameters not listed
    in a bound mapping remain fixed at their values in ``parameters``.

    Parameters listed in ``geometry`` or ``led`` are optimization variables.
    Physical compatibility is not reimplemented here. Constructed candidate
    ``FingertipParameters`` remain responsible for validating physical
    invariants such as LED fit within the stem.
    """

    parameters: FingertipParameters = field(
        default_factory=FingertipParameters
    )

    geometry: Mapping[str, ParameterBound] = field(
        default_factory=dict
    )
    led: Mapping[str, ParameterBound] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, FingertipParameters):
            raise TypeError(
                "parameters must be FingertipParameters"
            )

        geometry = self._validate_bounds(
            self.geometry,
            parameter_type=FingertipGeometry,
            group_name="geometry",
        )
        led = self._validate_bounds(
            self.led,
            parameter_type=LEDParameters,
            group_name="led",
        )

        object.__setattr__(
            self,
            "geometry",
            MappingProxyType(geometry),
        )
        object.__setattr__(
            self,
            "led",
            MappingProxyType(led),
        )

    @staticmethod
    def _validate_bounds(
        bounds: Mapping[str, ParameterBound],
        *,
        parameter_type: type,
        group_name: str,
    ) -> dict[str, ParameterBound]:
        if not isinstance(bounds, Mapping):
            raise TypeError(
                f"{group_name} bounds must be a mapping"
            )

        valid_names = {
            item.name
            for item in fields(parameter_type)
            if item.init
        }

        result = dict(bounds)

        unknown = set(result) - valid_names
        if unknown:
            raise ValueError(
                f"unknown {group_name} parameters: "
                f"{sorted(unknown)!r}"
            )

        for name, bound in result.items():
            if not isinstance(bound, ParameterBound):
                raise TypeError(
                    f"{group_name} bound for {name!r} "
                    "must be ParameterBound"
                )

        return result

    @property
    def free_geometry_parameters(self) -> tuple[str, ...]:
        """Return geometry parameters varied during optimization."""
        return tuple(self.geometry)

    @property
    def free_led_parameters(self) -> tuple[str, ...]:
        """Return LED parameters varied during optimization."""
        return tuple(self.led)


__all__ = [
    "DesignParameterBounds",
    "ParameterBound",
]
