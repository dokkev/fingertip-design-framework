"""Current six-variable production design-space contracts."""

from __future__ import annotations

from dataclasses import asdict
import math

import pytest

from model import Fingertip, FingertipParameters
from optimization import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    PRODUCTION_SEARCH_BOUNDS,
)


def _space(nominal: FingertipParameters | None = None) -> DesignSpace:
    return DesignSpace(
        FingertipParameters(void_height=0.25) if nominal is None else nominal,
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in PRODUCTION_SEARCH_BOUNDS
        ),
    )


def test_supported_variables_are_exactly_the_current_six() -> None:
    assert OPTIMIZABLE_PARAMETER_NAMES == (
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
        "void_height",
    )
    assert tuple(
        (variable.name, variable.lower, variable.upper)
        for variable in _space().active_variables
    ) == tuple(spec.to_tuple() for spec in PRODUCTION_SEARCH_BOUNDS)


def test_design_variable_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        DesignVariable("stem_height", True, 7.0, 6.0)
    with pytest.raises(ValueError, match="finite"):
        DesignVariable("stem_height", True, math.nan, 6.0)
    with pytest.raises(ValueError, match="unsupported"):
        DesignVariable("flat_pad_width", True, 20.0, 30.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        DesignVariable("stem_height", 1, 5.0, 7.0)  # type: ignore[arg-type]


def test_space_requires_all_six_active_variables_once() -> None:
    variables = tuple(
        DesignVariable(spec.name, True, spec.lower, spec.upper)
        for spec in PRODUCTION_SEARCH_BOUNDS
    )
    with pytest.raises(ValueError, match="exactly one entry"):
        DesignSpace(FingertipParameters(), variables[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        DesignSpace(FingertipParameters(), (*variables[:-1], variables[0]))
    inactive = tuple(
        DesignVariable(variable.name, variable.name != "void_width", variable.lower, variable.upper)
        for variable in variables
    )
    with pytest.raises(ValueError, match="all six"):
        DesignSpace(FingertipParameters(), inactive)


def test_decode_uses_independent_flat_and_semielliptical_heights() -> None:
    space = _space()
    values = {
        "flat_pad_height": 6.25,
        "semielliptical_pad_height": 10.5,
        "stem_width": 7.0,
        "stem_height": 7.0,
        "void_width": 1.5,
        "void_height": 1.25,
    }
    decoded = space.decode(values)
    assert decoded.flat_pad_width == 30.0
    assert decoded.flat_pad_height == 6.25
    assert decoded.semielliptical_pad_height == 10.5
    assert decoded.flat_pad_height + decoded.semielliptical_pad_height == 16.75
    assert decoded.void_height == 1.25
    with pytest.raises(ValueError, match="missing"):
        space.decode({name: value for name, value in values.items() if name != "void_width"})
    with pytest.raises(ValueError, match="unknown"):
        space.decode({**values, "flat_pad_width": 31.0})
    with pytest.raises(ValueError, match="outside"):
        space.decode({**values, "void_width": 10.1})


def test_decode_preserves_fixed_material_and_link_fields() -> None:
    nominal = FingertipParameters(
        link_thickness=4.0,
        bond_extension_width=3.0,
        bond_extension_height=1.5,
        young_modulus_mpa=0.8,
        poisson_ratio=0.2,
        arc_resolution=64,
        void_height=0.25,
    )
    decoded = _space(nominal).decode(
        {
            "flat_pad_height": 5.5,
            "semielliptical_pad_height": 9.5,
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


def test_corner_values_are_deterministic_for_six_variables() -> None:
    space = _space()
    corners = space.corner_values()
    assert len(corners) == 64
    assert corners[0] == space.lower_corner_values()
    assert corners[-1] == space.upper_corner_values()
    assert corners == space.corner_values()


@pytest.mark.parametrize("void_height", (0.25, 1.5, 2.0))
def test_void_height_decodes_into_authoritative_geometry(void_height: float) -> None:
    parameters = _space().decode(
        {
            variable.name: (
                void_height
                if variable.name == "void_height"
                else getattr(_space().nominal_parameters, variable.name)
            )
            for variable in _space().active_variables
        }
    )
    solid = Fingertip(parameters).solid()
    assert parameters.void_height == void_height
    assert solid.parameters.void_height == void_height
