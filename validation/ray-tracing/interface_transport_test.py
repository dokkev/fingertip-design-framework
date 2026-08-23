"""Validate one lossless dielectric-interface interaction analytically."""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np

from lumo.ray_tracing import interface_transport


_NORMAL = np.array((0.0, 0.0, 1.0))


def _direction(angle_degrees: float) -> np.ndarray:
    angle = radians(angle_degrees)
    return np.array((sin(angle), 0.0, -cos(angle)))


def _require_unit(label: str, direction: np.ndarray) -> None:
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-12):
        raise AssertionError(f"{label} is not normalized")


def main() -> None:
    normal_incidence = interface_transport(
        np.array(((0.0, 0.0, -1.0),)),
        _NORMAL[None, :],
        n_incident=1.0,
        n_transmitted=1.4,
    )[0]
    expected_reflectance = ((1.0 - 1.4) / (1.0 + 1.4)) ** 2
    if not np.allclose(
        normal_incidence["reflected_direction"],
        (0.0, 0.0, 1.0),
        atol=1.0e-12,
    ):
        raise AssertionError("normal-incidence reflection direction is wrong")
    if not np.allclose(
        normal_incidence["refracted_direction"],
        (0.0, 0.0, -1.0),
        atol=1.0e-12,
    ):
        raise AssertionError("normal-incidence refraction direction is wrong")
    if not np.isclose(
        normal_incidence["reflectance"],
        expected_reflectance,
        atol=1.0e-12,
    ):
        raise AssertionError("normal-incidence Fresnel reflectance is wrong")
    if not np.isclose(
        normal_incidence["reflectance"]
        + normal_incidence["transmittance"],
        1.0,
        atol=1.0e-12,
    ):
        raise AssertionError("normal-incidence R + T is not one")
    reversed_normal = interface_transport(
        np.array(((0.0, 0.0, -1.0),)),
        -_NORMAL[None, :],
        n_incident=1.0,
        n_transmitted=1.4,
    )[0]
    for field in (
        "reflected_direction",
        "refracted_direction",
        "reflectance",
        "transmittance",
    ):
        if not np.allclose(
            reversed_normal[field],
            normal_incidence[field],
            atol=1.0e-12,
        ):
            raise AssertionError("local normal orientation is inconsistent")

    oblique_direction = _direction(30.0)
    oblique = interface_transport(
        oblique_direction[None, :],
        _NORMAL[None, :],
        n_incident=1.0,
        n_transmitted=1.4,
    )[0]
    reflected = oblique["reflected_direction"]
    refracted = oblique["refracted_direction"]
    expected_reflected = np.array(
        (sin(radians(30.0)), 0.0, cos(radians(30.0)))
    )
    if not np.allclose(reflected, expected_reflected, atol=1.0e-12):
        raise AssertionError("oblique reflection angle is wrong")
    if not np.isclose(
        1.0 * sin(radians(30.0)),
        1.4 * abs(refracted[0]),
        atol=1.0e-12,
    ):
        raise AssertionError("oblique refraction violates Snell's law")
    _require_unit("oblique reflected direction", reflected)
    _require_unit("oblique refracted direction", refracted)
    if not (
        0.0 <= oblique["reflectance"] <= 1.0
        and 0.0 <= oblique["transmittance"] <= 1.0
        and np.isclose(
            oblique["reflectance"] + oblique["transmittance"],
            1.0,
            atol=1.0e-12,
        )
    ):
        raise AssertionError("oblique Fresnel powers are invalid")

    below_critical_direction = _direction(30.0)
    below_critical = interface_transport(
        below_critical_direction[None, :],
        _NORMAL[None, :],
        n_incident=1.4,
        n_transmitted=1.0,
    )[0]
    if bool(below_critical["total_internal_reflection"]):
        raise AssertionError("below-critical ray incorrectly reported TIR")
    if not np.all(np.isfinite(below_critical["refracted_direction"])):
        raise AssertionError("below-critical refraction is not finite")
    if not np.isclose(
        1.4 * sin(radians(30.0)),
        abs(below_critical["refracted_direction"][0]),
        atol=1.0e-12,
    ):
        raise AssertionError("silicone-to-air refraction violates Snell's law")

    above_critical = interface_transport(
        _direction(60.0)[None, :],
        _NORMAL[None, :],
        n_incident=1.4,
        n_transmitted=1.0,
    )[0]
    if not bool(above_critical["total_internal_reflection"]):
        raise AssertionError("above-critical ray did not report TIR")
    if above_critical["reflectance"] != 1.0:
        raise AssertionError("TIR reflectance is not one")
    if above_critical["transmittance"] != 0.0:
        raise AssertionError("TIR transmittance is not zero")
    if np.any(np.isfinite(above_critical["refracted_direction"])):
        raise AssertionError("TIR returned a finite refracted direction")
    _require_unit(
        "TIR reflected direction",
        above_critical["reflected_direction"],
    )

    print("normal incidence air -> silicone: PASS")
    print("oblique incidence air -> silicone: PASS")
    print("below-critical silicone -> air: PASS")
    print("above-critical silicone -> air: PASS")
    print()
    print("Dielectric interface transport: PASS")


if __name__ == "__main__":
    main()
