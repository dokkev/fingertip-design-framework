from __future__ import annotations

import pytest

from model import Fingertip, FingertipParameters, LED, OpticalMaterial


def test_fingertip_uses_authoritative_nominal_parameters_by_default() -> None:
    default = Fingertip()
    explicit = Fingertip(FingertipParameters())

    assert default.parameters == FingertipParameters()
    assert default.geometry.material_geometry.equals(explicit.geometry.material_geometry)
    assert default.geometry.raw_material_geometry.equals(
        explicit.geometry.raw_material_geometry
    )
    assert default.geometry.void_geometry.equals(explicit.geometry.void_geometry)


def test_fingertip_explicit_parameters_override_nominal_defaults() -> None:
    custom = FingertipParameters(void_height=1.0)
    tip = Fingertip(custom)

    assert tip.parameters == custom
    assert tip.parameters != FingertipParameters()


def test_fingertip_default_led_and_optical_material_are_unchanged() -> None:
    tip = Fingertip()

    assert tip.led == LED()
    assert tip.optical == OpticalMaterial()


def test_fingertip_owns_led_metadata_without_changing_mechanical_geometry() -> None:
    parameters = FingertipParameters()
    reference = Fingertip(parameters)
    alternate = Fingertip(
        parameters,
        led=LED(width_mm=2.0, height_mm=1.0, relative_radiant_power=2.0),
    )

    assert reference.led_source == (0.0, parameters.stem_tip_y)
    assert alternate.led_source == (0.0, parameters.stem_tip_y)
    assert alternate.led_package_geometry.bounds != pytest.approx(
        reference.led_package_geometry.bounds
    )
    assert alternate.geometry.material_geometry.equals(
        reference.geometry.material_geometry
    )
    assert alternate.geometry.material_geometry.area == pytest.approx(
        reference.geometry.material_geometry.area
    )
    assert alternate.boundaries.segments.keys() == reference.boundaries.segments.keys()
