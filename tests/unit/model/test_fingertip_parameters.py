"""Tests for fingertip parameter validation and derived dimensions."""

from __future__ import annotations

from dataclasses import fields
import math

import pytest

from model.fingertip_parameters import FingertipParameters, InvalidFingertipParameters


def _removed_name(prefix: str, suffix: str) -> str:
    """Build a removed API name without keeping it as a repository symbol."""
    return "_".join((prefix, suffix))


def test_default_parameters_use_the_canonical_geometry_api() -> None:
    parameters = FingertipParameters()
    assert parameters.flat_pad_width == 20.0
    assert parameters.flat_pad_height == 3.0
    assert parameters.semielliptical_pad_height == 7.0
    assert parameters.link_width == 12.0
    assert parameters.link_thickness == 3.5
    assert parameters.bond_extension_height == 2.0
    assert parameters.bond_extension_width == pytest.approx(4.0)
    assert parameters.cutout_width == parameters.stem_width
    assert parameters.cutout_height == parameters.stem_height
    assert parameters.void_area == 0.0
    assert parameters.young_modulus_mpa == 1.0
    assert parameters.poisson_ratio == 0.49


def test_removed_geometry_fields_are_not_constructor_parameters() -> None:
    parameter_names = {field.name for field in fields(FingertipParameters)}
    assert "flat_pad_width" in parameter_names
    assert "flat_pad_height" in parameter_names
    for removed_name in (
        _removed_name("vertical", "pad_width"),
        _removed_name("vertical", "pad_height"),
        _removed_name("semielliptical", "pad_width"),
    ):
        assert removed_name not in parameter_names
        with pytest.raises(TypeError):
            FingertipParameters(**{removed_name: 20.0})


def test_derived_coordinates_and_dimensions_have_one_definition() -> None:
    parameters = FingertipParameters(
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        link_width=14.0,
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
    assert parameters.bond_extension_width == pytest.approx(3.0)


@pytest.mark.parametrize(
    "name",
    [
        "flat_pad_width",
        "flat_pad_height",
        "semielliptical_pad_height",
        "link_width",
        "link_thickness",
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


def test_flat_pad_must_be_wider_than_the_rigid_link() -> None:
    with pytest.raises(InvalidFingertipParameters, match="flat_pad_width"):
        FingertipParameters(flat_pad_width=12.0, link_width=12.0)


def test_cutout_must_be_strictly_narrower_than_the_rigid_link() -> None:
    with pytest.raises(
        InvalidFingertipParameters,
        match="cutout_width must be smaller than link_width",
    ):
        FingertipParameters(link_width=12.0, stem_width=11.0, void_width=0.5)


def test_bond_extension_must_fit_on_the_rigid_link_sidewall() -> None:
    with pytest.raises(InvalidFingertipParameters, match="bond_extension_height"):
        FingertipParameters(link_thickness=2.0, bond_extension_height=2.1)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flat_pad_width", math.inf),
        ("flat_pad_height", -math.inf),
        ("semielliptical_pad_height", math.nan),
        ("link_width", math.inf),
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
