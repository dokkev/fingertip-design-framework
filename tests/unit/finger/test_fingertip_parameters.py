"""Tests for fingertip parameter validation and derived dimensions."""

from __future__ import annotations

from dataclasses import asdict, fields
import math

import pytest

from lumo.finger.fingertip_parameters import (
    FingertipParameters,
    InvalidFingertipParameters,
    OpticalParameters,
    ViscoelasticParameters,
)


def _removed_width_name(prefix: str) -> str:
    """Build a removed width name without keeping it as a public symbol."""
    return "_".join((prefix, "width"))


def test_default_parameters_use_the_canonical_geometry_api() -> None:
    parameters = FingertipParameters()
    assert parameters.flat_pad_width == 30.0
    assert parameters.flat_pad_height == 5.0
    assert parameters.semielliptical_pad_height == 9.0
    assert parameters.link_thickness == 3.5
    assert parameters.bond_extension_width == 4.0
    assert parameters.bond_extension_height == 2.0
    assert parameters.stem_width == 7.6
    assert parameters.stem_height == 6.0
    assert parameters.void_width == 1.0
    assert parameters.void_height == 0.0
    assert not hasattr(parameters, "cutout_width")
    assert not hasattr(parameters, "cutout_height")
    assert not hasattr(parameters, "void_area")


def test_fingertip_parameters_expose_a_typed_viscoelastic_group() -> None:
    parameters = FingertipParameters()

    assert parameters.viscoelastic == ViscoelasticParameters()
    assert parameters.viscoelastic.k_mu_pa == 1.0e5
    assert ViscoelasticParameters(k_damp=0.0).k_damp == 0.0


def test_serialized_nested_viscoelastic_parameters_are_restored_as_a_type() -> None:
    parameters = FingertipParameters(
        viscoelastic=ViscoelasticParameters(k_mu_pa=2.0e5),
    )

    restored = FingertipParameters(**asdict(parameters))

    assert restored == parameters
    assert isinstance(restored.viscoelastic, ViscoelasticParameters)


def test_optical_parameters_are_owned_by_fingertip_parameters() -> None:
    parameters = FingertipParameters(
        optical=OpticalParameters(absorption_per_mm=0.03),
    )

    assert parameters.optical.absorption_per_mm == 0.03
    assert FingertipParameters(**asdict(parameters)) == parameters


def test_optical_parameters_reject_invalid_absorption() -> None:
    with pytest.raises(InvalidFingertipParameters, match="absorption"):
        OpticalParameters(absorption_per_mm=-0.01)


@pytest.mark.parametrize("name", ["density_kg_m3", "k_mu_pa", "k_lambda_pa"])
def test_viscoelastic_positive_inputs_are_validated(name: str) -> None:
    with pytest.raises(InvalidFingertipParameters):
        ViscoelasticParameters(**{name: 0.0})


def test_viscoelastic_damping_may_be_zero_but_not_negative() -> None:
    with pytest.raises(InvalidFingertipParameters, match="greater than or equal"):
        ViscoelasticParameters(k_damp=-1.0)


def test_link_and_ellipse_widths_are_not_constructor_parameters() -> None:
    parameter_names = {field.name for field in fields(FingertipParameters)}
    assert "flat_pad_width" in parameter_names
    assert _removed_width_name("link") not in parameter_names
    assert _removed_width_name("semielliptical_pad") not in parameter_names
    with pytest.raises(TypeError):
        FingertipParameters(**{_removed_width_name("link"): 20.0})
    with pytest.raises(TypeError):
        FingertipParameters(**{_removed_width_name("semielliptical_pad"): 20.0})


def test_derived_coordinates_and_dimensions_are_not_parameter_attributes() -> None:
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
    for name in (
        "ellipse_start_y",
        "stem_tip_y",
        "void_bottom_y",
        "pad_tip_y",
        "total_pad_depth",
        "cutout_width",
        "cutout_half_width",
        "cutout_height",
        "cutout_depth",
        "bonded_segment_length",
        "void_area",
    ):
        assert not hasattr(parameters, name)


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
    with pytest.raises(InvalidFingertipParameters, match=r"2\*bond_extension_width"):
        FingertipParameters(
            bond_extension_width=10.2,
            stem_width=7.6,
        )


def test_cutout_inside_flat_region_is_valid() -> None:
    parameters = FingertipParameters(stem_height=2.0)
    assert parameters.stem_height + parameters.void_height <= parameters.flat_pad_height


def test_cutout_penetrating_semiellipse_can_remain_inside() -> None:
    parameters = FingertipParameters()
    assert parameters.stem_height + parameters.void_height > parameters.flat_pad_height


def test_cutout_too_deep_for_semiellipse_is_rejected() -> None:
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(void_height=8.0)


def test_cutout_width_and_depth_coupling_is_rejected() -> None:
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(
            bond_extension_width=1.0,
            stem_width=8.0,
            void_width=4.0,
            void_height=8.0,
        )


def test_cutout_on_or_within_tolerance_of_ellipse_is_rejected() -> None:
    half_width = 30.0 / 2.0
    cutout_half_width = 7.6 / 2.0
    available_depth = 9.0 * math.sqrt(
        1.0 - (cutout_half_width / half_width) ** 2
    )

    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(
            void_height=available_depth + 5.0 - 6.0,
        )

    tolerance = 1.0e-3
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        FingertipParameters(
            geometry_length_tolerance_mm=tolerance,
            void_height=available_depth - tolerance / 2.0 + 5.0 - 6.0,
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
        ("geometry_length_tolerance_mm", math.inf),
        ("geometry_area_tolerance_mm2", math.inf),
    ],
)
def test_nonfinite_values_are_rejected(name: str, value: float) -> None:
    with pytest.raises(InvalidFingertipParameters):
        FingertipParameters(**{name: value})
