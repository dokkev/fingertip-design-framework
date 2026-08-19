"""Current five-variable production design-space contracts."""

from __future__ import annotations

from dataclasses import asdict
import math

import pytest

from model import FingertipParameters
from model import Fingertip
from optimization import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    PRODUCTION_SEARCH_BOUNDS,
    create_production_study,
)


def _space(
    nominal: FingertipParameters | None = None,
    *,
    bounds: tuple[tuple[str, float, float], ...] = PRODUCTION_SEARCH_BOUNDS,
) -> DesignSpace:
    return DesignSpace(
        FingertipParameters(void_height=0.25) if nominal is None else nominal,
        tuple(DesignVariable(name, True, lower, upper) for name, lower, upper in bounds),
    )


def test_supported_variables_are_exactly_the_current_five() -> None:
    assert OPTIMIZABLE_PARAMETER_NAMES == (
        "flat_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
        "void_height",
    )
    assert tuple(
        (variable.name, variable.lower, variable.upper)
        for variable in create_production_study().design_space.active_variables
    ) == PRODUCTION_SEARCH_BOUNDS


def test_design_variable_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        DesignVariable("stem_height", True, 7.0, 6.0)
    with pytest.raises(ValueError, match="finite"):
        DesignVariable("stem_height", True, math.nan, 6.0)
    with pytest.raises(ValueError, match="unsupported"):
        DesignVariable("flat_pad_width", True, 20.0, 30.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        DesignVariable("stem_height", 1, 5.0, 7.0)  # type: ignore[arg-type]


def test_space_requires_all_five_active_variables_once() -> None:
    variables = tuple(
        DesignVariable(name, True, lower, upper)
        for name, lower, upper in PRODUCTION_SEARCH_BOUNDS
    )
    with pytest.raises(ValueError, match="exactly one entry"):
        DesignSpace(FingertipParameters(), variables[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        DesignSpace(FingertipParameters(), (*variables[:-1], variables[0]))
    inactive = tuple(
        DesignVariable(
            variable.name,
            variable.name != "void_width",
            variable.lower,
            variable.upper,
        )
        for variable in variables
    )
    with pytest.raises(ValueError, match="all five"):
        DesignSpace(FingertipParameters(), inactive)


def test_decode_uses_exact_names_bounds_and_derived_height() -> None:
    space = _space()
    values = {
        "flat_pad_height": 6.25,
        "stem_width": 8.25,
        "stem_height": 7.0,
        "void_width": 1.5,
        "void_height": 1.25,
    }
    decoded = space.decode(values)

    assert decoded.flat_pad_width == 30.0
    assert decoded.flat_pad_height == 6.25
    assert decoded.semielliptical_pad_height == 7.75
    assert decoded.void_height == 1.25
    with pytest.raises(ValueError, match="missing"):
        space.decode(
            {name: value for name, value in values.items() if name != "void_width"}
        )
    with pytest.raises(ValueError, match="unknown"):
        space.decode({**values, "semielliptical_pad_height": 8.0})
    with pytest.raises(ValueError, match="outside"):
        space.decode({**values, "void_width": 2.1})


def test_decode_preserves_fixed_material_and_link_fields() -> None:
    nominal = FingertipParameters(
        link_thickness=4.0,
        bond_extension_width=3.0,
        bond_extension_height=1.5,
        young_modulus_mpa=0.8,
        poisson_ratio=0.2,
        arc_resolution=64,
    )
    decoded = _space(nominal).decode(
        {
            "flat_pad_height": 5.5,
            "stem_width": 8.0,
        "stem_height": 6.5,
        "void_width": 1.25,
        "void_height": 0.75,
        }
    )
    before = asdict(nominal)
    for name in (
        "link_thickness",
        "bond_extension_width",
        "bond_extension_height",
        "young_modulus_mpa",
        "poisson_ratio",
        "arc_resolution",
    ):
        assert getattr(decoded, name) == before[name]


def test_production_study_rejects_zero_nominal_void_height() -> None:
    with pytest.raises(ValueError, match="nominal void_height"):
        create_production_study(nominal_parameters=FingertipParameters())


def test_corner_values_are_deterministic_for_five_variables() -> None:
    space = _space()
    corners = space.corner_values()
    assert len(corners) == 32
    assert corners[0] == space.lower_corner_values()
    assert corners[-1] == space.upper_corner_values()
    assert corners == space.corner_values()


@pytest.mark.parametrize("void_height", (0.25, 1.5, 3.0))
def test_void_height_decodes_into_authoritative_geometry(void_height: float) -> None:
    space = create_production_study().design_space
    values = {
        variable.name: (
            void_height
            if variable.name == "void_height"
            else (variable.lower + variable.upper) / 2.0
        )
        for variable in space.active_variables
    }
    parameters = space.decode(values)
    solid = Fingertip(parameters).solid()

    assert parameters.void_height == void_height
    assert parameters.void_bottom_y == -(parameters.stem_height + void_height)
    assert solid.parameters.void_height == void_height
