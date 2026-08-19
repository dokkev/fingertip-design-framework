"""Algorithm-independent production morphology design-space definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import isfinite
from numbers import Real
from typing import Literal, Mapping

from model import (
    FingertipParameters,
    validate_minimum_silicone_thickness,
    validate_silicone_ligament,
)


OptimizableParameterName = Literal[
    "flat_pad_height",
    "semielliptical_pad_height",
    "stem_width",
    "stem_height",
    "void_width",
    "void_height",
]

OPTIMIZABLE_PARAMETER_NAMES: tuple[OptimizableParameterName, ...] = (
    "flat_pad_height",
    "semielliptical_pad_height",
    "stem_width",
    "stem_height",
    "void_width",
    "void_height",
)
_OPTIMIZABLE_PARAMETER_SET = frozenset(OPTIMIZABLE_PARAMETER_NAMES)
_FIXED_FLAT_PAD_WIDTH_MM = 30.0
PRODUCTION_NOMINAL_VOID_HEIGHT_MM = 0.25
PRODUCTION_SEARCH_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("flat_pad_height", 3.0, 8.0),
    ("semielliptical_pad_height", 5.0, 12.0),
    ("stem_width", 5.0, 10.0),
    ("stem_height", 4.0, 9.0),
    ("void_width", 0.5, 4.0),
    ("void_height", 0.25, 3.0),
)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


@dataclass(frozen=True)
class DesignVariable:
    """One supported physical morphology field and its explicit box bounds."""

    name: OptimizableParameterName
    optimize: bool
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in _OPTIMIZABLE_PARAMETER_SET:
            raise ValueError(f"unsupported design variable: {self.name!r}")
        if not isinstance(self.optimize, bool):
            raise TypeError("optimize must be a bool")
        lower = _finite_real("lower", self.lower)
        upper = _finite_real("upper", self.upper)
        if lower > upper:
            raise ValueError(
                f"lower bound must not exceed upper bound for {self.name}: "
                f"lower={lower:g}, upper={upper:g}"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class DesignSpace:
    """Immutable nominal parameters plus the six-variable production contract.

    Flat and semi-elliptical pad heights are independent.  Candidate feasibility
    is enforced by the authoritative fingertip validation plus the explicit
    minimum-silicone-thickness rule; total pad depth is therefore free to vary.
    """

    nominal_parameters: FingertipParameters
    variables: tuple[DesignVariable, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nominal_parameters, FingertipParameters):
            raise TypeError("nominal_parameters must be FingertipParameters")
        if self.nominal_parameters.flat_pad_width != _FIXED_FLAT_PAD_WIDTH_MM:
            raise ValueError("production DesignSpace requires flat_pad_width=30.0")

        variables = tuple(self.variables)
        if len(variables) != len(OPTIMIZABLE_PARAMETER_NAMES):
            raise ValueError(
                "DesignSpace must contain exactly one entry for each of the six "
                "optimizable parameters"
            )
        if any(not isinstance(variable, DesignVariable) for variable in variables):
            raise TypeError("variables must contain DesignVariable values")

        by_name: dict[str, DesignVariable] = {}
        for variable in variables:
            if variable.name in by_name:
                raise ValueError(f"duplicate design variable name: {variable.name}")
            by_name[variable.name] = variable
        if set(by_name) != _OPTIMIZABLE_PARAMETER_SET:
            missing = _OPTIMIZABLE_PARAMETER_SET - set(by_name)
            unknown = set(by_name) - _OPTIMIZABLE_PARAMETER_SET
            raise ValueError(
                "DesignSpace variables must contain exactly the six supported "
                f"parameters; missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
            )
        if any(not variable.optimize for variable in by_name.values()):
            raise ValueError(
                "production DesignSpace requires all six morphology variables "
                "to be active"
            )

        object.__setattr__(
            self,
            "variables",
            tuple(by_name[name] for name in OPTIMIZABLE_PARAMETER_NAMES),
        )

    @property
    def active_variables(self) -> tuple[DesignVariable, ...]:
        """Return active variables in the canonical scientific order."""
        return tuple(variable for variable in self.variables if variable.optimize)

    def lower_corner_values(self) -> dict[str, float]:
        return {variable.name: variable.lower for variable in self.active_variables}

    def upper_corner_values(self) -> dict[str, float]:
        return {variable.name: variable.upper for variable in self.active_variables}

    def decode(self, values: Mapping[str, float]) -> FingertipParameters:
        """Decode exactly one named active candidate without repairing it."""
        active = self.active_variables
        active_names = {variable.name for variable in active}
        supplied_names = set(values)
        unknown = supplied_names - active_names
        missing = active_names - supplied_names
        if unknown:
            raise ValueError(f"unknown or inactive design variables: {sorted(unknown)!r}")
        if missing:
            raise ValueError(f"missing active design variables: {sorted(missing)!r}")

        updates: dict[str, float] = {}
        for variable in active:
            value = _finite_real(variable.name, values[variable.name])
            if value < variable.lower or value > variable.upper:
                raise ValueError(
                    f"{variable.name}={value:g} is outside inclusive bounds "
                    f"[{variable.lower:g}, {variable.upper:g}]"
                )
            updates[variable.name] = value

        candidate = replace(
            self.nominal_parameters,
            **updates,
            flat_pad_width=_FIXED_FLAT_PAD_WIDTH_MM,
        )
        # Retain the established conservative diagnostics and add the true
        # Euclidean corner-to-ellipse wall-thickness gate.  Invalid candidates
        # fail here, before meshing/mechanics/optics.
        validate_silicone_ligament(candidate)
        validate_minimum_silicone_thickness(candidate)
        return candidate

    def corner_values(self) -> tuple[dict[str, float], ...]:
        active = self.active_variables
        if not active:
            return ({},)
        return tuple(
            {
                variable.name: value
                for variable, value in zip(active, choice, strict=True)
            }
            for choice in product(
                *((variable.lower, variable.upper) for variable in active)
            )
        )


__all__ = [
    "DesignSpace",
    "DesignVariable",
    "OPTIMIZABLE_PARAMETER_NAMES",
    "PRODUCTION_SEARCH_BOUNDS",
    "PRODUCTION_NOMINAL_VOID_HEIGHT_MM",
    "OptimizableParameterName",
]
