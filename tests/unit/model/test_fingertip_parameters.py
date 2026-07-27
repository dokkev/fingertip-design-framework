"""Tests for LIT pad parameter validation and derived dimensions."""

from __future__ import annotations

import math

import pytest

from model.fingertip_parameters import FingertipParameters, InvalidFingertipParameters


def test_default_parameters_are_component_specific() -> None:
    parameters = FingertipParameters()
    assert parameters.vertical_pad_width == 20.0
    assert parameters.vertical_pad_height == 3.0
    assert parameters.semielliptical_pad_width == 20.0
    assert parameters.semielliptical_pad_height == 7.0
    assert parameters.stem_height == 6.0
    assert parameters.stem_height > parameters.vertical_pad_height
    assert parameters.link_width == parameters.vertical_pad_width
    assert parameters.cutout_width == parameters.stem_width
    assert parameters.cutout_height == parameters.stem_height
    assert parameters.void_area == 0.0


def test_derived_coordinates_and_dimensions_have_one_definition() -> None:
    parameters = FingertipParameters(
        vertical_pad_height=5.0,
        semielliptical_pad_height=9.0,
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
        "vertical_pad_width",
        "vertical_pad_height",
        "semielliptical_pad_width",
        "semielliptical_pad_height",
        "link_thickness",
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


def test_vertical_and_semielliptical_widths_must_match() -> None:
    with pytest.raises(InvalidFingertipParameters, match="must be equal"):
        FingertipParameters(semielliptical_pad_width=19.0)
    FingertipParameters(semielliptical_pad_width=20.0 + 0.5e-9)


def test_cutout_must_be_strictly_narrower_than_vertical_pad() -> None:
    with pytest.raises(
        InvalidFingertipParameters,
        match="cutout_width must be smaller",
    ):
        FingertipParameters(stem_width=10.0, void_width=5.0)


def test_stem_and_vertical_pad_heights_are_not_coupled() -> None:
    short_stem = FingertipParameters(stem_height=2.0, vertical_pad_height=4.0)
    long_stem = FingertipParameters(stem_height=6.0, vertical_pad_height=4.0)
    assert short_stem.stem_height < short_stem.vertical_pad_height
    assert long_stem.stem_height > long_stem.vertical_pad_height


def test_legacy_mapping_requires_explicit_new_vertical_height() -> None:
    migrated = FingertipParameters.from_legacy_mapping(
        {
            "pad_width": 24.0,
            "pad_height": 10.0,
            "stem_width": 8.0,
            "stem_height": 7.0,
        },
        vertical_pad_height=5.0,
    )
    assert migrated.vertical_pad_width == 24.0
    assert migrated.semielliptical_pad_width == 24.0
    assert migrated.vertical_pad_height == 5.0
    assert migrated.semielliptical_pad_height == 10.0
    assert migrated.stem_height == 7.0


def test_legacy_mapping_rejects_mixed_schema() -> None:
    with pytest.raises(InvalidFingertipParameters, match="cannot mix"):
        FingertipParameters.from_legacy_mapping(
            {
                "pad_width": 24.0,
                "pad_height": 10.0,
                "vertical_pad_width": 24.0,
            },
            vertical_pad_height=5.0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("vertical_pad_width", math.inf),
        ("vertical_pad_height", -math.inf),
        ("semielliptical_pad_width", math.nan),
        ("semielliptical_pad_height", math.inf),
        ("stem_width", math.nan),
        ("void_height", math.nan),
        ("geometry_tolerance", math.inf),
    ],
)
def test_nonfinite_values_are_rejected(name: str, value: float) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: value})
