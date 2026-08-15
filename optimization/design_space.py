"""Algorithm-independent morphology design-space definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import isfinite
from numbers import Real
from typing import Literal, Mapping

from model import FingertipParameters, validate_silicone_ligament


OptimizableParameterName = Literal[
    "flat_pad_width",
    "flat_pad_height",
    "semielliptical_pad_height",
    "stem_width",
    "stem_height",
    "void_width",
    "void_height",
]

OPTIMIZABLE_PARAMETER_NAMES: tuple[OptimizableParameterName, ...] = (
    "flat_pad_width",
    "flat_pad_height",
    "semielliptical_pad_height",
    "stem_width",
    "stem_height",
    "void_width",
    "void_height",
)
_OPTIMIZABLE_PARAMETER_SET = frozenset(OPTIMIZABLE_PARAMETER_NAMES)


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
    """Immutable nominal parameters plus the complete seven-variable morphology
    design-space contract.
    """

    nominal_parameters: FingertipParameters
    variables: tuple[DesignVariable, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nominal_parameters, FingertipParameters):
            raise TypeError("nominal_parameters must be FingertipParameters")

        variables = tuple(self.variables)
        if len(variables) != len(OPTIMIZABLE_PARAMETER_NAMES):
            raise ValueError(
                "DesignSpace must contain exactly one entry for each of the seven "
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
                "DesignSpace variables must contain exactly the seven supported "
                f"parameters; missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
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
        """Return active-variable values for the all-lower box corner."""
        return {
            variable.name: variable.lower for variable in self.active_variables
        }

    def upper_corner_values(self) -> dict[str, float]:
        """Return active-variable values for the all-upper box corner."""
        return {
            variable.name: variable.upper for variable in self.active_variables
        }

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

        # FingertipParameters remains the authority for physical constraints;
        # the optimization-only coupled ligament rule is enforced here.
        candidate = replace(self.nominal_parameters, **updates)
        validate_silicone_ligament(candidate)
        return candidate

    def corner_values(self) -> tuple[dict[str, float], ...]:
        """Enumerate every active lower/upper corner deterministically."""
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
    "OptimizableParameterName",
]
