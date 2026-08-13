"""Tests for fingertip parameter validation and derived dimensions."""

from __future__ import annotations

from dataclasses import fields
import math

import pytest

from model.fingertip_parameters import FingertipParameters, InvalidFingertipParameters


def _removed_width_name(prefix: str) -> str:
    """Build a removed width name without keeping it as a public symbol."""
    return "_".join((prefix, "width"))


def test_default_parameters_use_the_canonical_geometry_api() -> None:
    parameters = FingertipParameters()
    assert parameters.flat_pad_width == 20.0
    assert parameters.flat_pad_height == 3.0
    assert parameters.semielliptical_pad_height == 7.0
    assert parameters.link_thickness == 3.5
    assert parameters.bond_extension_width == 4.0
    assert parameters.bond_extension_height == 2.0
    assert parameters.cutout_width == parameters.stem_width
    assert parameters.cutout_height == parameters.stem_height
    assert parameters.void_area == 0.0
    assert parameters.young_modulus_mpa == 0.55
    assert parameters.poisson_ratio == 0.49


def test_link_and_ellipse_widths_are_not_constructor_parameters() -> None:
    parameter_names = {field.name for field in fields(FingertipParameters)}
    assert "flat_pad_width" in parameter_names
    assert _removed_width_name("link") not in parameter_names
    assert _removed_width_name("semielliptical_pad") not in parameter_names
    with pytest.raises(TypeError):
        FingertipParameters(**{_removed_width_name("link"): 20.0})
    with pytest.raises(TypeError):
        FingertipParameters(**{_removed_width_name("semielliptical_pad"): 20.0})


def test_derived_coordinates_and_dimensions_have_one_definition() -> None:
    parameters = FingertipParameters(
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        bond_extension_width=3.0,
        bond_extension_height=2.5,
        stem_width=6.0,
        stem_height=7.0,
        void_width=1.5,
        void_height=2.0,
    )
    assert parameters.ellipse_start_y == pytest.approx(-5.0)
    assert parameters.stem_tip_y == pytest.approx(-7.0)
    assert parameters.void_bottom_y == pytest.approx(-9.0)
    assert parameters.pad_tip_y == pytest.approx(-14.0)
    assert parameters.cutout_width == pytest.approx(9.0)
    assert parameters.cutout_half_width == pytest.approx(4.5)
    assert parameters.cutout_height == pytest.approx(9.0)
    assert parameters.total_pad_depth == pytest.approx(14.0)


@pytest.mark.parametrize(
    "name",
    [
        "flat_pad_width",
        "flat_pad_height",
        "semielliptical_pad_height",
        "link_thickness",
        "bond_extension_width",
        "bond_extension_height",
        "stem_width",
        "stem_height",
    ],
)
def test_nonpositive_primary_dimensions_are_rejected(name: str) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: -1.0})


@pytest.mark.parametrize("name", ["void_width", "void_height"])
def test_negative_clearance_is_rejected(name: str) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: -0.1})


def test_bond_extension_height_must_be_strictly_below_link_thickness() -> None:
    with pytest.raises(InvalidFingertipParameters, match="smaller than"):
        FingertipParameters(link_thickness=2.0, bond_extension_height=2.0)


def test_extensions_and_cutout_must_fit_within_the_full_width() -> None:
    with pytest.raises(InvalidFingertipParameters, match="2\*bond_extension_width"):
        FingertipParameters(
            bond_extension_width=6.2,
            stem_width=7.6,
        )


def test_cutout_inside_flat_region_is_valid() -> None:
    parameters = FingertipParameters(stem_height=2.0)
    assert parameters.cutout_height <= parameters.flat_pad_height


def test_cutout_penetrating_semiellipse_can_remain_inside() -> None:
    parameters = FingertipParameters()
    assert parameters.cutout_height > parameters.flat_pad_height


def test_cutout_too_deep_for_semiellipse_is_rejected() -> None:
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(void_height=4.0)


def test_cutout_width_and_depth_coupling_is_rejected() -> None:
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(
            bond_extension_width=1.0,
            stem_width=8.0,
            void_width=4.0,
            void_height=2.6,
        )


def test_cutout_on_or_within_tolerance_of_ellipse_is_rejected() -> None:
    half_width = 20.0 / 2.0
    cutout_half_width = 7.6 / 2.0
    available_depth = 7.0 * math.sqrt(
        1.0 - (cutout_half_width / half_width) ** 2
    )

    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(
            void_height=available_depth + 3.0 - 6.0,
        )

    tolerance = 1.0e-3
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(
            geometry_tolerance=tolerance,
            void_height=available_depth - tolerance / 2.0 + 3.0 - 6.0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flat_pad_width", math.inf),
        ("flat_pad_height", -math.inf),
        ("semielliptical_pad_height", math.nan),
        ("bond_extension_width", math.inf),
        ("bond_extension_height", math.nan),
        ("stem_width", math.nan),
        ("void_height", math.nan),
        ("young_modulus_mpa", math.inf),
        ("poisson_ratio", math.nan),
        ("geometry_tolerance", math.inf),
    ],
)
def test_nonfinite_values_are_rejected(name: str, value: float) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: value})


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_nonpositive_young_modulus_is_rejected(value: float) -> None:
    with pytest.raises(InvalidFingertipParameters, match="young_modulus_mpa"):
        FingertipParameters(young_modulus_mpa=value)


@pytest.mark.parametrize("value", [-1.0, 0.5])
def test_poisson_ratio_must_be_in_open_physical_range(value: float) -> None:
    with pytest.raises(InvalidFingertipParameters, match="poisson_ratio"):
        FingertipParameters(poisson_ratio=value)
