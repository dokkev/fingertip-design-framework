"""Algorithm-independent production morphology design-space definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import product
from math import isfinite, sqrt
from numbers import Real
from typing import Mapping

from lumo.finger import (
    FingertipParameters,
    InvalidFingertipParameters,
    MAX_TOTAL_PAD_DEPTH_MM,
    PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
    validate_minimum_silicone_thickness,
)


def _lerp(lower: float, upper: float, fraction: float) -> float:
    return float(lower + (upper - lower) * fraction)


def _fraction(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1.0e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _unit_interval(value: float, name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < -1.0e-9 or resolved > 1.0 + 1.0e-9:
        raise DesignSpaceFeasibilityError(
            f"{name} cannot encode the supplied physical morphology: {resolved!r}",
            constraint="latent_parameterization",
        )
    return min(1.0, max(0.0, resolved))


class OptimizableParameterName(StrEnum):
    """Canonical names shared by morphology, Ax, and registry boundaries."""

    FLAT_PAD_HEIGHT = "flat_pad_height"
    SEMIELLIPTICAL_PAD_HEIGHT = "semielliptical_pad_height"
    STEM_WIDTH = "stem_width"
    STEM_HEIGHT = "stem_height"
    VOID_WIDTH = "void_width"
    VOID_HEIGHT = "void_height"

OPTIMIZABLE_PARAMETER_NAMES: tuple[OptimizableParameterName, ...] = tuple(
    OptimizableParameterName
)
_OPTIMIZABLE_PARAMETER_SET = frozenset(OptimizableParameterName)
_FIXED_FLAT_PAD_WIDTH_MM = 30.0
PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM = MAX_TOTAL_PAD_DEPTH_MM
PRODUCTION_NOMINAL_VOID_HEIGHT_MM = 0.25
FEASIBLE_PARAMETERIZATION_VERSION = "feasible-morphology-v1"
LATENT_PARAMETER_NAMES: tuple[str, ...] = (
    "latent_cutout_width",
    "latent_pad_depth",
    "latent_pad_split",
    "latent_stem_width_split",
    "latent_cutout_depth",
    "latent_stem_height_split",
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
        try:
            name = OptimizableParameterName(self.name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported design variable: {self.name!r}") from exc
        object.__setattr__(self, "name", name)
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

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "optimize": self.optimize,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class ParameterSpec:
    """Named numerical envelope for one morphology parameter."""

    name: OptimizableParameterName
    lower: float
    upper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", OptimizableParameterName(self.name))
        lower = _finite_real("lower", self.lower)
        upper = _finite_real("upper", self.upper)
        if lower > upper:
            raise ValueError(f"lower bound must not exceed upper bound for {self.name}")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name.value, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class LinearConstraint:
    """One named feasibility constraint shared by search backends."""

    expression: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("linear constraint expression must be non-empty")
        object.__setattr__(self, "expression", self.expression.strip())


class DesignSpaceFeasibilityError(ValueError):
    """Reject one latent proposal before mesh, mechanics, or optics."""

    def __init__(self, message: str, *, constraint: str) -> None:
        super().__init__(message)
        if not isinstance(constraint, str) or not constraint.strip():
            raise ValueError("constraint must be a non-empty string")
        self.constraint = constraint


@dataclass(frozen=True)
class LatentVariable:
    """One normalized variable exposed to a search backend."""

    name: str
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in LATENT_PARAMETER_NAMES:
            raise ValueError(f"unsupported latent variable: {self.name!r}")
        lower = _finite_real("lower", self.lower)
        upper = _finite_real("upper", self.upper)
        if lower != 0.0 or upper != 1.0:
            raise ValueError("latent variables must span [0, 1]")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "lower": self.lower, "upper": self.upper}

PRODUCTION_SEARCH_BOUNDS: tuple[ParameterSpec, ...] = (
    ParameterSpec(OptimizableParameterName.FLAT_PAD_HEIGHT, 0.5, 29.5),
    ParameterSpec(OptimizableParameterName.SEMIELLIPTICAL_PAD_HEIGHT, 0.5, 29.5),
    ParameterSpec(OptimizableParameterName.STEM_WIDTH, 1.0, 20.0),
    ParameterSpec(OptimizableParameterName.STEM_HEIGHT, 1.0, 25.0),
    ParameterSpec(OptimizableParameterName.VOID_WIDTH, 0.0, 10.0),
    ParameterSpec(OptimizableParameterName.VOID_HEIGHT, 0.0, 25.0),
)

PRODUCTION_LINEAR_CONSTRAINTS: tuple[LinearConstraint, ...] = (
    LinearConstraint("flat_pad_height + semielliptical_pad_height <= 30.0"),
    LinearConstraint("stem_width + 2*void_width <= 20.0"),
)


@dataclass(frozen=True)
class DesignSpace:
    """Immutable nominal parameters and a feasible latent search contract.

    Ax samples six normalized latent variables. ``decode`` maps them to the
    physical six-dimensional morphology while enforcing the coupled pad,
    cutout, envelope, and silicone-thickness constraints. The physical bounds
    remain available as ``variables`` and are never sent directly to Ax.
    """

    nominal_parameters: FingertipParameters
    variables: tuple[DesignVariable, ...]
    linear_constraints: tuple[LinearConstraint, ...] = PRODUCTION_LINEAR_CONSTRAINTS

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

        constraints = tuple(self.linear_constraints)
        if any(not isinstance(item, LinearConstraint) for item in constraints):
            raise TypeError("linear_constraints must contain LinearConstraint values")
        object.__setattr__(self, "linear_constraints", constraints)

        object.__setattr__(
            self,
            "variables",
            tuple(by_name[name] for name in OPTIMIZABLE_PARAMETER_NAMES),
        )

        self.validate_physical_parameters(self.nominal_parameters)

    @property
    def active_variables(self) -> tuple[DesignVariable, ...]:
        """Return physical variables in canonical scientific order."""
        return tuple(variable for variable in self.variables if variable.optimize)

    @property
    def parameterization_version(self) -> str:
        return FEASIBLE_PARAMETERIZATION_VERSION

    @property
    def search_variables(self) -> tuple[LatentVariable, ...]:
        """Return the normalized variables exposed to Ax or another backend."""
        return tuple(LatentVariable(name) for name in LATENT_PARAMETER_NAMES)

    @property
    def latent_variables(self) -> tuple[LatentVariable, ...]:
        """Alias emphasizing that the backend sees latent, not physical, values."""
        return self.search_variables

    def to_dict(self) -> dict[str, object]:
        """Return the search contract for campaign and Ax snapshot metadata."""
        return {
            "parameterization_version": self.parameterization_version,
            "physical_variables": [variable.to_dict() for variable in self.variables],
            "latent_variables": [variable.to_dict() for variable in self.search_variables],
            "linear_constraints": [
                constraint.expression for constraint in self.linear_constraints
            ],
        }

    def lower_corner_values(self) -> dict[str, float]:
        return {variable.name: variable.lower for variable in self.active_variables}

    def upper_corner_values(self) -> dict[str, float]:
        return {variable.name: variable.upper for variable in self.active_variables}

    def decode(self, values: Mapping[str, float]) -> FingertipParameters:
        """Map one normalized latent proposal to an authoritative morphology."""
        active = self.search_variables
        active_names = {variable.name for variable in active}
        supplied_names = set(values)
        unknown = supplied_names - active_names
        missing = active_names - supplied_names
        if unknown:
            raise DesignSpaceFeasibilityError(
                f"unknown or inactive latent variables: {sorted(unknown)!r}",
                constraint="latent_parameterization",
            )
        if missing:
            raise DesignSpaceFeasibilityError(
                f"missing latent variables: {sorted(missing)!r}",
                constraint="latent_parameterization",
            )

        try:
            latent = {
                variable.name: _finite_real(variable.name, values[variable.name])
                for variable in active
            }
        except (TypeError, ValueError) as exc:
            raise DesignSpaceFeasibilityError(
                f"latent values must be finite real numbers: {exc}",
                constraint="latent_bounds",
            ) from exc
        for name, value in latent.items():
            if not 0.0 <= value <= 1.0:
                raise DesignSpaceFeasibilityError(
                    f"{name}={value:g} is outside latent bounds [0, 1]",
                    constraint="latent_bounds",
                )

        cutout_width_upper = self._maximum_feasible_cutout_width()
        cutout_width_lower = self._minimum_cutout_width()
        cutout_width = _lerp(
            cutout_width_lower,
            cutout_width_upper,
            latent["latent_cutout_width"],
        )
        flat_height, semi_height = self._decode_pad_heights(
            latent["latent_pad_depth"],
            latent["latent_pad_split"],
            cutout_width,
        )
        stem_width, void_width = self._decode_width_split(
            cutout_width,
            latent["latent_stem_width_split"],
        )
        cutout_depth_upper = self._maximum_feasible_cutout_depth(
            flat_height, semi_height, cutout_width
        )
        cutout_depth_lower = self._minimum_cutout_depth()
        cutout_depth = _lerp(
            cutout_depth_lower,
            cutout_depth_upper,
            latent["latent_cutout_depth"],
        )
        stem_height, void_height = self._decode_height_split(
            cutout_depth,
            latent["latent_stem_height_split"],
        )
        try:
            candidate = replace(
                self.nominal_parameters,
                flat_pad_height=flat_height,
                semielliptical_pad_height=semi_height,
                stem_width=stem_width,
                stem_height=stem_height,
                void_width=void_width,
                void_height=void_height,
                flat_pad_width=_FIXED_FLAT_PAD_WIDTH_MM,
            )
        except InvalidFingertipParameters as exc:
            raise DesignSpaceFeasibilityError(
                str(exc), constraint="fingertip_geometry"
            ) from exc
        self.validate_physical_parameters(candidate)
        return candidate

    def encode(self, parameters: FingertipParameters) -> dict[str, float]:
        """Return latent values that reproduce one valid physical morphology."""
        self.validate_physical_parameters(parameters)
        flat = float(parameters.flat_pad_height)
        semi = float(parameters.semielliptical_pad_height)
        cutout_width = parameters.stem_width + 2.0 * parameters.void_width
        total_lower = self._minimum_pad_depth(cutout_width)
        total_upper = self._maximum_pad_depth()
        total = flat + semi
        pad_depth = _unit_interval(
            _fraction(total - total_lower, total_upper - total_lower),
            "latent_pad_depth",
        )
        flat_lower, flat_upper = self._pad_height_interval_with_margin(
            total, cutout_width
        )
        pad_split = _unit_interval(
            _fraction(flat - flat_lower, flat_upper - flat_lower),
            "latent_pad_split",
        )
        cutout_upper = self._maximum_feasible_cutout_width()
        cutout = _unit_interval(
            _fraction(
                cutout_width - self._minimum_cutout_width(),
                cutout_upper - self._minimum_cutout_width(),
            ),
            "latent_cutout_width",
        )
        stem_lower, stem_upper = self._width_split_bounds(cutout_width)
        width_split = _unit_interval(
            _fraction(parameters.stem_width - stem_lower, stem_upper - stem_lower),
            "latent_stem_width_split",
        )
        depth = parameters.stem_height + parameters.void_height
        depth_upper = self._maximum_feasible_cutout_depth(flat, semi, cutout_width)
        depth = _unit_interval(
            _fraction(
                depth - self._minimum_cutout_depth(),
                depth_upper - self._minimum_cutout_depth(),
            ),
            "latent_cutout_depth",
        )
        height_lower, height_upper = self._height_split_bounds(
            parameters.stem_height + parameters.void_height
        )
        height_split = _unit_interval(
            _fraction(parameters.stem_height - height_lower, height_upper - height_lower),
            "latent_stem_height_split",
        )
        return {
            "latent_pad_depth": pad_depth,
            "latent_pad_split": pad_split,
            "latent_cutout_width": cutout,
            "latent_stem_width_split": width_split,
            "latent_cutout_depth": depth,
            "latent_stem_height_split": height_split,
        }

    def physical_values(
        self,
        parameters: FingertipParameters,
    ) -> dict[str, float]:
        """Return only the six physical search fields for the registry."""
        self.validate_physical_parameters(parameters)
        return {
            variable.name.value: float(getattr(parameters, variable.name.value))
            for variable in self.active_variables
        }

    def from_physical_values(
        self,
        values: Mapping[str, float],
    ) -> FingertipParameters:
        """Construct and validate a morphology from registry fields."""
        expected = {variable.name.value for variable in self.active_variables}
        supplied = set(values)
        if supplied != expected:
            raise DesignSpaceFeasibilityError(
                "physical morphology must contain exactly the six search fields",
                constraint="physical_parameterization",
            )
        candidate = replace(
            self.nominal_parameters,
            **{name: float(values[name]) for name in expected},
            flat_pad_width=_FIXED_FLAT_PAD_WIDTH_MM,
        )
        self.validate_physical_parameters(candidate)
        return candidate

    def validate_physical_parameters(
        self,
        parameters: FingertipParameters,
    ) -> None:
        """Apply physical bounds and authoritative geometry feasibility rules."""
        if not isinstance(parameters, FingertipParameters):
            raise TypeError("parameters must be FingertipParameters")
        for variable in self.active_variables:
            value = float(getattr(parameters, variable.name.value))
            if value < variable.lower or value > variable.upper:
                raise DesignSpaceFeasibilityError(
                    f"{variable.name.value}={value:g} is outside physical bounds "
                    f"[{variable.lower:g}, {variable.upper:g}]",
                    constraint=f"bounds:{variable.name.value}",
                )
        self._validate_linear_constraints(parameters)
        try:
            parameters.validate()
        except InvalidFingertipParameters as exc:
            raise DesignSpaceFeasibilityError(
                str(exc), constraint="fingertip_geometry"
            ) from exc
        try:
            validate_minimum_silicone_thickness(
                parameters,
                minimum_mm=PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
            )
        except InvalidFingertipParameters as exc:
            raise DesignSpaceFeasibilityError(
                str(exc), constraint="minimum_silicone_thickness"
            ) from exc

    def _validate_linear_constraints(self, parameters: FingertipParameters) -> None:
        normalized = {
            "flat_pad_height+semielliptical_pad_height<=30.0": (
                parameters.flat_pad_height + parameters.semielliptical_pad_height,
                MAX_TOTAL_PAD_DEPTH_MM,
                "total_pad_depth",
            ),
            "stem_width+2*void_width<=20.0": (
                parameters.stem_width + 2.0 * parameters.void_width,
                20.0,
                "cutout_width",
            ),
        }
        for constraint in self.linear_constraints:
            key = constraint.expression.replace(" ", "")
            if key not in normalized:
                raise DesignSpaceFeasibilityError(
                    f"unsupported feasibility constraint: {constraint.expression}",
                    constraint="linear_constraint",
                )
            value, upper, name = normalized[key]
            if value > upper + parameters.geometry_length_tolerance_mm:
                raise DesignSpaceFeasibilityError(
                    f"{name}={value:g} exceeds {upper:g}",
                    constraint=name,
                )

    def _minimum_pad_depth(self, cutout_width: float | None = None) -> float:
        lower = self._physical_variable("flat_pad_height").lower + self._physical_variable(
            "semielliptical_pad_height"
        ).lower
        if cutout_width is None:
            return lower
        feasible_lower = lower
        feasible_upper = self._maximum_pad_depth()
        if self._pad_split_has_margin(feasible_lower, cutout_width):
            return feasible_lower
        if not self._pad_split_has_margin(feasible_upper, cutout_width):
            raise DesignSpaceFeasibilityError(
                "pad bounds cannot provide the minimum silicone thickness",
                constraint="minimum_silicone_thickness",
            )
        for _ in range(48):
            middle = 0.5 * (feasible_lower + feasible_upper)
            if self._pad_split_has_margin(middle, cutout_width):
                feasible_upper = middle
            else:
                feasible_lower = middle
        return feasible_upper

    def _maximum_pad_depth(self) -> float:
        return min(
            self._physical_variable("flat_pad_height").upper
            + self._physical_variable("semielliptical_pad_height").upper,
            MAX_TOTAL_PAD_DEPTH_MM,
        )

    def _decode_pad_heights(
        self,
        depth: float,
        split: float,
        cutout_width: float,
    ) -> tuple[float, float]:
        total = _lerp(
            self._minimum_pad_depth(cutout_width),
            self._maximum_pad_depth(),
            depth,
        )
        lower, upper = self._pad_height_interval_with_margin(total, cutout_width)
        flat = _lerp(lower, upper, split)
        return flat, total - flat

    def _pad_height_interval(self, total: float) -> tuple[float, float]:
        flat = self._physical_variable("flat_pad_height")
        semi = self._physical_variable("semielliptical_pad_height")
        lower = max(flat.lower, total - semi.upper)
        upper = min(flat.upper, total - semi.lower)
        if upper < lower:
            raise DesignSpaceFeasibilityError(
                f"no pad-height split exists for total depth {total:g} mm",
                constraint="total_pad_depth",
            )
        return lower, upper

    def _pad_height_interval_with_margin(
        self,
        total: float,
        cutout_width: float,
    ) -> tuple[float, float]:
        lower, upper = self._pad_height_interval(total)
        factor = sqrt(
            1.0 - (cutout_width / _FIXED_FLAT_PAD_WIDTH_MM) ** 2
        )
        required_flat = (
            PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM
            + self._minimum_cutout_depth()
            - total * factor
        ) / (1.0 - factor)
        lower = max(lower, required_flat)
        if upper < lower:
            raise DesignSpaceFeasibilityError(
                f"no pad-height split provides a 5 mm wall for cutout {cutout_width:g} mm",
                constraint="minimum_silicone_thickness",
            )
        return lower, upper

    def _pad_split_has_margin(self, total: float, cutout_width: float) -> bool:
        try:
            self._pad_height_interval_with_margin(total, cutout_width)
        except DesignSpaceFeasibilityError:
            return False
        return True

    def _minimum_cutout_width(self) -> float:
        return self._physical_variable("stem_width").lower + 2.0 * self._physical_variable(
            "void_width"
        ).lower

    def _maximum_cutout_width(self) -> float:
        stem = self._physical_variable("stem_width")
        void = self._physical_variable("void_width")
        bond_limit = _FIXED_FLAT_PAD_WIDTH_MM - 2.0 * self.nominal_parameters.bond_extension_width
        return min(stem.upper + 2.0 * void.upper, 20.0, bond_limit)

    def _maximum_feasible_cutout_width(self) -> float:
        lower = self._minimum_cutout_width()
        upper = self._maximum_cutout_width()
        if upper < lower:
            raise DesignSpaceFeasibilityError(
                "physical bounds leave no cutout-width interval",
                constraint="cutout_width",
            )
        if not self._pad_split_has_margin(self._maximum_pad_depth(), lower):
            raise DesignSpaceFeasibilityError(
                "minimum cutout width cannot satisfy silicone thickness",
                constraint="minimum_silicone_thickness",
            )
        if self._pad_split_has_margin(self._maximum_pad_depth(), upper):
            return upper
        high = upper
        low = lower
        for _ in range(48):
            middle = 0.5 * (low + high)
            if self._pad_split_has_margin(self._maximum_pad_depth(), middle):
                low = middle
            else:
                high = middle
        return low

    def _maximum_feasible_cutout_depth(
        self,
        flat: float,
        semi: float,
        cutout: float,
    ) -> float:
        lower = self._minimum_cutout_depth()
        upper = min(
            self._physical_variable("stem_height").upper
            + self._physical_variable("void_height").upper,
            flat
            + self._ellipse_depth(semi, cutout)
            - PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM
            - self.nominal_parameters.geometry_length_tolerance_mm,
        )
        if upper < lower:
            raise DesignSpaceFeasibilityError(
                "minimum cutout depth cannot satisfy silicone thickness",
                constraint="minimum_silicone_thickness",
            )
        return upper

    def _decode_width_split(
        self,
        cutout: float,
        split: float,
    ) -> tuple[float, float]:
        lower, upper = self._width_split_bounds(cutout)
        stem = _lerp(lower, upper, split)
        return stem, (cutout - stem) / 2.0

    def _width_split_bounds(self, cutout: float) -> tuple[float, float]:
        stem = self._physical_variable("stem_width")
        void = self._physical_variable("void_width")
        lower = max(stem.lower, cutout - 2.0 * void.upper)
        upper = min(stem.upper, cutout - 2.0 * void.lower)
        if upper < lower:
            raise DesignSpaceFeasibilityError(
                f"no stem/void width split exists for cutout {cutout:g} mm",
                constraint="cutout_width",
            )
        return lower, upper

    def _minimum_cutout_depth(self) -> float:
        return self._physical_variable("stem_height").lower + self._physical_variable(
            "void_height"
        ).lower

    def _decode_height_split(self, depth: float, split: float) -> tuple[float, float]:
        lower, upper = self._height_split_bounds(depth)
        stem = _lerp(lower, upper, split)
        return stem, depth - stem

    def _decode_height_split_at_lower(self, depth: float) -> tuple[float, float]:
        lower, _ = self._height_split_bounds(depth)
        return lower, depth - lower

    def _height_split_bounds(self, depth: float) -> tuple[float, float]:
        stem = self._physical_variable("stem_height")
        void = self._physical_variable("void_height")
        lower = max(stem.lower, depth - void.upper)
        upper = min(stem.upper, depth - void.lower)
        if upper < lower:
            raise DesignSpaceFeasibilityError(
                f"no stem/void height split exists for cutout depth {depth:g} mm",
                constraint="cutout_depth",
            )
        return lower, upper

    def _ellipse_depth(self, semi: float, cutout: float) -> float:
        half_width = _FIXED_FLAT_PAD_WIDTH_MM / 2.0
        normalized = (cutout / 2.0) / half_width
        if not 0.0 <= normalized < 1.0:
            raise DesignSpaceFeasibilityError(
                "cutout lies outside the pad envelope",
                constraint="cutout_width",
            )
        return semi * sqrt(1.0 - normalized**2)

    def _physical_variable(self, name: str) -> DesignVariable:
        return next(variable for variable in self.variables if variable.name.value == name)

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
    "DesignSpaceFeasibilityError",
    "FEASIBLE_PARAMETERIZATION_VERSION",
    "LATENT_PARAMETER_NAMES",
    "LatentVariable",
    "LinearConstraint",
    "ParameterSpec",
    "OPTIMIZABLE_PARAMETER_NAMES",
    "PRODUCTION_SEARCH_BOUNDS",
    "PRODUCTION_LINEAR_CONSTRAINTS",
    "PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM",
    "PRODUCTION_NOMINAL_VOID_HEIGHT_MM",
    "OptimizableParameterName",
]
