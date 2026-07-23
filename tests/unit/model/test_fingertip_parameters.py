"""Tests for LIT pad parameter validation and derived dimensions."""

from __future__ import annotations

import math

import pytest

from model.fingertip_parameters import FingertipParameters, InvalidFingertipParameters


def test_default_parameters_match_reference_geometry() -> None:
    parameters = FingertipParameters()
    assert parameters.pad_width == 30.0
    assert parameters.pad_height == 18.0
    assert parameters.link_width == parameters.pad_width
    assert parameters.cutout_width == parameters.stem_width
    assert parameters.cutout_depth == parameters.stem_height
    assert parameters.void_area == 0.0


@pytest.mark.parametrize(
    "name",
    ["pad_width", "pad_height", "link_thickness", "stem_width", "stem_height"],
)
def test_nonpositive_primary_dimensions_are_rejected(name: str) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: -1.0})


@pytest.mark.parametrize("name", ["void_width", "void_height"])
def test_negative_clearance_is_rejected(name: str) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: -0.1})


def test_side_and_bottom_clearance_are_independent() -> None:
    parameters = FingertipParameters(void_width=2.5, void_height=3.0)
    assert parameters.cutout_width == pytest.approx(12.0)
    assert parameters.cutout_half_width == pytest.approx(6.0)
    assert parameters.cutout_depth == pytest.approx(10.0)
    assert parameters.void_area == pytest.approx(71.0)


def test_cutout_must_leave_interface_on_both_sides() -> None:
    with pytest.raises(InvalidFingertipParameters, match="positive bonded segment"):
        FingertipParameters(stem_width=10.0, void_width=10.0)


def test_cutout_lower_corners_must_remain_inside_pad() -> None:
    with pytest.raises(
        InvalidFingertipParameters,
        match="outside the half-ellipse.*normalized_ellipse_value",
    ):
        FingertipParameters(void_height=20.0)


def test_cutout_lower_corners_may_lie_on_ellipse_boundary() -> None:
    base = FingertipParameters()
    boundary_depth = base.pad_height * math.sqrt(
        1.0 - (base.cutout_half_width / (base.pad_width / 2.0)) ** 2
    )
    parameters = FingertipParameters(stem_height=boundary_depth)
    assert parameters.cutout_depth == pytest.approx(boundary_depth)


def test_stem_wider_than_pad_has_specific_error() -> None:
    with pytest.raises(
        InvalidFingertipParameters,
        match=r"stem_width=31, pad_width=30",
    ):
        FingertipParameters(stem_width=31.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pad_width", math.inf),
        ("pad_height", -math.inf),
        ("stem_width", math.nan),
        ("void_height", math.nan),
        ("geometry_tolerance", math.inf),
    ],
)
def test_nonfinite_values_are_rejected(name: str, value: float) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: value})
