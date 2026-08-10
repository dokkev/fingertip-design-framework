from __future__ import annotations

import pytest

from model import Fingertip, FingertipParameters, LED


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
