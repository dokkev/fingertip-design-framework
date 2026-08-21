"""Feasible design space for fingertip optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite

from lumo.fingertip.fingertip import Fingertip
from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.fingertip.geometric_param import InvalidFingertipParameters

from .design_param_bound import DesignParameterBounds


@dataclass(frozen=True)
class LinearConstraint:
    """Linear constraint over fingertip design parameters.

    The constraint represents

        lower <= sum(a_i * x_i) <= upper

    Parameter names use qualified paths such as

        geometry.flat_pad_height_mm
        led.width_mm
    """

    coefficients: Mapping[str, float]
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        coefficients = {
            name: float(value)
            for name, value in self.coefficients.items()
            if float(value) != 0.0
        }

        if not coefficients:
            raise ValueError(
                "linear constraint requires at least one nonzero coefficient"
            )

        for name, coefficient in coefficients.items():
            if not isinstance(name, str) or "." not in name:
                raise ValueError(
                    "constraint parameter names must use '<group>.<parameter>'"
                )

            if not isfinite(coefficient):
                raise ValueError(
                    f"coefficient for {name!r} must be finite"
                )

        lower = None if self.lower is None else float(self.lower)
        upper = None if self.upper is None else float(self.upper)

        if lower is None and upper is None:
            raise ValueError(
                "linear constraint requires lower or upper"
            )

        if lower is not None and not isfinite(lower):
            raise ValueError("constraint lower bound must be finite")

        if upper is not None and not isfinite(upper):
            raise ValueError("constraint upper bound must be finite")

        if (
            lower is not None
            and upper is not None
            and lower > upper
        ):
            raise ValueError(
                "constraint must satisfy lower <= upper"
            )

        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def is_satisfied(
        self,
        values: Mapping[str, float],
    ) -> bool:
        """Return whether the supplied parameter values satisfy the constraint."""

        try:
            value = sum(
                coefficient * values[name]
                for name, coefficient in self.coefficients.items()
            )
        except KeyError as exc:
            raise ValueError(
                f"constraint references unknown parameter {exc.args[0]!r}"
            ) from exc

        if self.lower is not None and value < self.lower:
            return False

        if self.upper is not None and value > self.upper:
            return False

        return True


@dataclass(frozen=True)
class DesignSpace:
    """Feasible fingertip design space.

    ``parameter_bounds`` defines the raw optimization domain. Linear and
    geometric constraints restrict that domain to the feasible subset.
    """

    parameter_bounds: DesignParameterBounds = field(
        default_factory=DesignParameterBounds
    )

    linear_constraints: tuple[LinearConstraint, ...] = ()

    minimum_silicone_thickness_mm: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.parameter_bounds,
            DesignParameterBounds,
        ):
            raise TypeError(
                "parameter_bounds must be DesignParameterBounds"
            )

        constraints = tuple(self.linear_constraints)

        if not all(
            isinstance(constraint, LinearConstraint)
            for constraint in constraints
        ):
            raise TypeError(
                "linear_constraints must contain LinearConstraint objects"
            )

        minimum_thickness = self.minimum_silicone_thickness_mm

        if minimum_thickness is not None:
            minimum_thickness = float(minimum_thickness)

            if (
                not isfinite(minimum_thickness)
                or minimum_thickness < 0.0
            ):
                raise ValueError(
                    "minimum_silicone_thickness_mm must be "
                    "finite and non-negative"
                )

        object.__setattr__(
            self,
            "linear_constraints",
            constraints,
        )
        object.__setattr__(
            self,
            "minimum_silicone_thickness_mm",
            minimum_thickness,
        )

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return all free optimization parameter names."""

        geometry = tuple(
            f"geometry.{name}"
            for name in self.parameter_bounds.geometry
        )

        led = tuple(
            f"led.{name}"
            for name in self.parameter_bounds.led
        )

        return geometry + led

    def to_parameters(
        self,
        candidate: Mapping[str, float],
    ) -> FingertipParameters:
        """Construct complete fingertip parameters from one candidate."""

        expected = set(self.variable_names)
        received = set(candidate)

        if received != expected:
            missing = expected - received
            extra = received - expected

            details = []

            if missing:
                details.append(f"missing={sorted(missing)!r}")

            if extra:
                details.append(f"unexpected={sorted(extra)!r}")

            raise ValueError(
                "candidate must define exactly the free parameters: "
                + ", ".join(details)
            )

        geometry_values = {}
        led_values = {}

        for name, bound in self.parameter_bounds.geometry.items():
            key = f"geometry.{name}"
            value = float(candidate[key])

            if not isfinite(value):
                raise ValueError(
                    f"{key!r} must be finite"
                )

            if not bound.lower <= value <= bound.upper:
                raise ValueError(
                    f"{key!r} lies outside its parameter bounds"
                )

            geometry_values[name] = value

        for name, bound in self.parameter_bounds.led.items():
            key = f"led.{name}"
            value = float(candidate[key])

            if not isfinite(value):
                raise ValueError(
                    f"{key!r} must be finite"
                )

            if not bound.lower <= value <= bound.upper:
                raise ValueError(
                    f"{key!r} lies outside its parameter bounds"
                )

            led_values[name] = value

        base = self.parameter_bounds.parameters

        geometry = replace(
            base.geometry,
            **geometry_values,
        )

        led = replace(
            base.led,
            **led_values,
        )

        return replace(
            base,
            geometry=geometry,
            led=led,
        )

    def parameter_values(
        self,
        parameters: FingertipParameters,
    ) -> dict[str, float]:
        """Return qualified scalar parameter values used by constraints."""

        geometry = {
            f"geometry.{name}": float(value)
            for name, value in vars(parameters.geometry).items()
        }

        led = {
            f"led.{name}": float(value)
            for name, value in vars(parameters.led).items()
        }

        return geometry | led

    def is_feasible(
        self,
        candidate: Mapping[str, float],
    ) -> bool:
        """Return whether a candidate belongs to the feasible design space."""

        try:
            parameters = self.to_parameters(candidate)
        except (ValueError, TypeError, InvalidFingertipParameters):
            return False

        values = self.parameter_values(parameters)

        if any(
            not constraint.is_satisfied(values)
            for constraint in self.linear_constraints
        ):
            return False

        if self.minimum_silicone_thickness_mm is not None:
            fingertip = Fingertip(parameters)

            if (
                fingertip.minimum_silicone_thickness_mm
                < self.minimum_silicone_thickness_mm
            ):
                return False

        return True


__all__ = [
    "DesignSpace",
    "LinearConstraint",
]
