"""Core invariants for fingertip physical parameters and analytic assembly."""

from __future__ import annotations

from dataclasses import fields

import pytest

from lumo.fingertip import (
    ACTIVE_Y_BOUNDS_MM,
    LED_CENTERS_Y_MM,
    LED_RECESS_DEPTH_MM,
    TOTAL_Y_BOUNDS_MM,
    Fingertip,
    FingertipGeometry,
    FingertipParameters,
    LEDParameters,
    SiliconeMechanics,
)


def test_default_fingertip_derives_its_physical_assembly() -> None:
    fingertip = Fingertip()

    assert isinstance(fingertip.parameters.mechanics, SiliconeMechanics)
    assert fingertip.parameters.optics.name == "Dragon Skin 10 NV"
    assert fingertip.bonding_interface.left[0] == (
        fingertip.silicone.cavity_left_x_mm,
        0.0,
    )
    assert fingertip.bonding_interface.right[-1] == (
        fingertip.silicone.cavity_right_x_mm,
        0.0,
    )
    bonding_field = next(
        field for field in fields(Fingertip) if field.name == "bonding_interface"
    )
    assert not bonding_field.init


def test_stem_height_is_the_only_cavity_depth() -> None:
    geometry = FingertipGeometry(stem_height_mm=7.0)
    fingertip = Fingertip(FingertipParameters(geometry=geometry))

    assert "void_height_mm" not in {
        field.name for field in fields(FingertipGeometry)
    }
    assert fingertip.silicone.cavity_bottom_z_mm == -7.0


def test_hardware_layout_defines_led_source_centers() -> None:
    fingertip = Fingertip()
    centers_m = fingertip.led_source_centers_m
    source_z_mm = (
        min(z_mm for _, z_mm in fingertip.carrier.cross_section)
        + LED_RECESS_DEPTH_MM
    )

    assert ACTIVE_Y_BOUNDS_MM == (-27.5, 27.5)
    assert TOTAL_Y_BOUNDS_MM == (-27.5, 32.5)
    assert tuple(
        1.0e3 * center[1] for center in centers_m
    ) == pytest.approx(LED_CENTERS_Y_MM)
    assert all(center[0] == 0.0 for center in centers_m)
    assert tuple(
        1.0e3 * center[2] for center in centers_m
    ) == pytest.approx((source_z_mm,) * len(LED_CENTERS_Y_MM))


def test_invalid_geometry_raises_value_error() -> None:
    with pytest.raises(ValueError, match="flat_pad_height_mm"):
        FingertipGeometry(flat_pad_height_mm=0.0)


@pytest.mark.parametrize(
    ("geometry", "led", "message"),
    (
        (
            FingertipGeometry(stem_width_mm=7.0),
            LEDParameters(width_mm=7.1),
            "LED width",
        ),
        (
            FingertipGeometry(stem_height_mm=4.0),
            LEDParameters(height_mm=4.1),
            "LED height",
        ),
    ),
)
def test_led_must_fit_inside_stem(
    geometry: FingertipGeometry,
    led: LEDParameters,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FingertipParameters(geometry=geometry, led=led)


def test_mechanics_requires_physical_scalar_ranges() -> None:
    with pytest.raises(ValueError, match="damping_pa_s"):
        SiliconeMechanics(damping_pa_s=-1.0)
