"""Focused tests for the algorithm-independent six-variable design space."""

from __future__ import annotations

import math

import pytest

from model import FingertipParameters
from optimization import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
)


def _space(
    nominal_parameters: FingertipParameters | None = None,
    *,
    active: tuple[str, ...] = (),
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
    reversed_input: bool = False,
) -> DesignSpace:
    nominal_parameters = nominal_parameters or FingertipParameters()
    lower = lower or {}
    upper = upper or {}
    variables = []
    for name in OPTIMIZABLE_PARAMETER_NAMES:
        nominal_value = getattr(nominal_parameters, name)
        variables.append(
            DesignVariable(
                name=name,
                optimize=name in active,
                lower=lower.get(name, nominal_value),
                upper=upper.get(name, nominal_value),
            )
        )
    if reversed_input:
        variables.reverse()
    return DesignSpace(nominal_parameters, tuple(variables))


def test_exact_supported_set_includes_void_width_but_not_void_height() -> None:
    assert OPTIMIZABLE_PARAMETER_NAMES == (
        "flat_pad_width",
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
    )
    assert "void_width" in OPTIMIZABLE_PARAMETER_NAMES
    assert "void_height" not in OPTIMIZABLE_PARAMETER_NAMES


@pytest.mark.parametrize(
    "variable",
    (
        DesignVariable("void_width", False, 0.0, 0.0),
        DesignVariable("stem_height", True, 6.0, 6.0),
    ),
)
def test_design_variable_accepts_finite_zero_width_bounds(variable) -> None:
    assert variable.lower <= variable.upper


def test_design_variable_rejects_reversed_nonfinite_and_unsupported_values() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        DesignVariable("stem_height", True, 7.0, 6.0)
    with pytest.raises(ValueError, match="finite"):
        DesignVariable("stem_height", True, math.nan, 6.0)
    with pytest.raises(ValueError, match="finite"):
        DesignVariable("stem_height", True, 6.0, math.inf)
    with pytest.raises(ValueError, match="unsupported"):
        DesignVariable("void_height", False, 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        DesignVariable("stem_height", 1, 6.0, 7.0)  # type: ignore[arg-type]


def test_design_space_requires_exactly_one_entry_for_each_supported_name() -> None:
    variables = [
        DesignVariable(name, False, 1.0, 1.0)
        for name in OPTIMIZABLE_PARAMETER_NAMES
    ]
    with pytest.raises(ValueError, match="duplicate"):
        DesignSpace(FingertipParameters(), tuple(variables[:-1] + [variables[0]]))
    with pytest.raises(ValueError, match="exactly one entry"):
        DesignSpace(FingertipParameters(), tuple(variables[:-1]))
    variables[-1] = DesignVariable("stem_height", False, 1.0, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        DesignSpace(FingertipParameters(), tuple(variables))


def test_active_variables_are_canonical_and_include_void_width() -> None:
    space = _space(
        active=("void_width", "flat_pad_width", "stem_width"),
        reversed_input=True,
    )
    assert tuple(variable.name for variable in space.active_variables) == (
        "flat_pad_width",
        "stem_width",
        "void_width",
    )


def test_zero_active_space_has_one_empty_corner_and_decodes_nominal() -> None:
    space = _space()
    assert space.active_variables == ()
    assert space.lower_corner_values() == {}
    assert space.upper_corner_values() == {}
    assert space.corner_values() == ({},)
    assert space.decode({}) == space.nominal_parameters


def test_nominal_parameters_is_the_only_reference_field() -> None:
    nominal_parameters = FingertipParameters()
    space = _space(nominal_parameters)

    assert space.nominal_parameters is nominal_parameters
    assert not hasattr(space, "baseline")


def test_decode_requires_exact_active_names_and_inclusive_bounds() -> None:
    space = _space(
        active=("stem_height", "void_width"),
        lower={"stem_height": 5.0, "void_width": 0.0},
        upper={"stem_height": 7.0, "void_width": 1.0},
    )
    assert space.decode({"stem_height": 5.0, "void_width": 0.0}).stem_height == 5.0
    assert space.decode({"stem_height": 7.0, "void_width": 1.0}).void_width == 1.0
    with pytest.raises(ValueError, match="missing"):
        space.decode({"stem_height": 6.0})
    with pytest.raises(ValueError, match="unknown or inactive"):
        space.decode({"stem_height": 6.0, "void_width": 0.5, "void_height": 0.0})
    with pytest.raises(TypeError, match="finite real"):
        space.decode({"stem_height": True, "void_width": 0.5})
    with pytest.raises(ValueError, match="finite"):
        space.decode({"stem_height": math.inf, "void_width": 0.5})
    with pytest.raises(ValueError, match="outside"):
        space.decode({"stem_height": 7.1, "void_width": 0.5})


def test_decode_preserves_fixed_geometry_and_mechanical_values() -> None:
    nominal_parameters = FingertipParameters(
        link_thickness=4.0,
        bond_extension_width=3.0,
        bond_extension_height=1.5,
        void_height=0.5,
        young_modulus_mpa=0.8,
        poisson_ratio=0.2,
        arc_resolution=64,
    )
    space = _space(
        nominal_parameters,
        active=("stem_width",),
        lower={"stem_width": 7.6},
        upper={"stem_width": 8.0},
    )
    decoded = space.decode({"stem_width": 8.0})
    for name in (
        "link_thickness",
        "bond_extension_width",
        "bond_extension_height",
        "void_height",
        "young_modulus_mpa",
        "poisson_ratio",
        "arc_resolution",
    ):
        assert getattr(decoded, name) == getattr(nominal_parameters, name)


def test_void_width_decode_changes_only_void_width() -> None:
    space = _space(
        active=("void_width",),
        lower={"void_width": 0.0},
        upper={"void_width": 1.0},
    )
    zero = space.decode({"void_width": 0.0})
    one = space.decode({"void_width": 1.0})
    assert zero.void_width == 0.0
    assert one.void_width == 1.0
    assert one.stem_width == zero.stem_width
    assert one.void_height == zero.void_height


def test_physical_decode_failure_propagates_without_repair() -> None:
    space = _space(
        active=("flat_pad_width", "stem_width"),
        lower={"flat_pad_width": 15.0, "stem_width": 7.6},
        upper={"flat_pad_width": 20.0, "stem_width": 9.0},
    )
    with pytest.raises(ValueError):
        space.decode({"flat_pad_width": 15.0, "stem_width": 9.0})


def test_corner_values_are_deterministic_and_contain_only_active_names() -> None:
    space = _space(
        active=("flat_pad_width", "stem_height", "void_width"),
        lower={"flat_pad_width": 19.0, "stem_height": 5.0, "void_width": 0.0},
        upper={"flat_pad_width": 21.0, "stem_height": 7.0, "void_width": 1.0},
    )
    corners = space.corner_values()
    assert len(corners) == 8
    assert corners == space.corner_values()
    assert corners[0] == {
        "flat_pad_width": 19.0,
        "stem_height": 5.0,
        "void_width": 0.0,
    }
    assert all(set(corner) == {"flat_pad_width", "stem_height", "void_width"} for corner in corners)
    assert all(
        corner[name] in {
            next(variable for variable in space.variables if variable.name == name).lower,
            next(variable for variable in space.variables if variable.name == name).upper,
        }
        for corner in corners
        for name in corner
    )
