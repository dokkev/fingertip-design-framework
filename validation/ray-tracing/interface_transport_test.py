"""Validate one lossless dielectric-interface interaction analytically."""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np

from lumo.ray_tracing.transport import interface_transport


_NORMAL = np.array((0.0, 0.0, 1.0))


def _direction(angle_degrees: float) -> np.ndarray:
    angle = radians(angle_degrees)
    return np.array((sin(angle), 0.0, -cos(angle)))


def _require_unit(label: str, direction: np.ndarray) -> None:
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-12):
        raise AssertionError(f"{label} is not normalized")


def _require_power_split(
    label: str,
    optical: np.ndarray,
    incident_power: float | np.ndarray,
) -> None:
    incident_power = np.broadcast_to(
        np.asarray(incident_power, dtype=np.float64),
        optical.shape,
    )
    if not np.allclose(
        optical["reflected_power"],
        incident_power * optical["reflectance"],
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError(f"{label} reflected power is wrong")
    if not np.allclose(
        optical["refracted_power"],
        incident_power * optical["transmittance"],
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError(f"{label} refracted power is wrong")
    if not np.allclose(
        optical["reflected_power"] + optical["refracted_power"],
        incident_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError(f"{label} does not conserve ray power")


def main() -> None:
    normal_incidence_result = interface_transport(
        np.array(((0.0, 0.0, -1.0),)),
        _NORMAL[None, :],
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=0.37,
    )
    _require_power_split("normal incidence", normal_incidence_result, 0.37)
    normal_incidence = normal_incidence_result[0]
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
    reversed_normal_result = interface_transport(
        np.array(((0.0, 0.0, -1.0),)),
        -_NORMAL[None, :],
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=0.37,
    )
    _require_power_split("reversed normal", reversed_normal_result, 0.37)
    reversed_normal = reversed_normal_result[0]
    for field in (
        "reflected_direction",
        "refracted_direction",
        "reflectance",
        "transmittance",
        "reflected_power",
        "refracted_power",
    ):
        if not np.allclose(
            reversed_normal[field],
            normal_incidence[field],
            atol=1.0e-12,
        ):
            raise AssertionError("local normal orientation is inconsistent")

    oblique_direction = _direction(30.0)
    oblique_powers = np.array((0.37, 0.0))
    oblique_result = interface_transport(
        np.repeat(oblique_direction[None, :], 2, axis=0),
        np.repeat(_NORMAL[None, :], 2, axis=0),
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=oblique_powers,
    )
    _require_power_split("oblique incidence", oblique_result, oblique_powers)
    oblique = oblique_result[0]
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
    below_critical_result = interface_transport(
        below_critical_direction[None, :],
        _NORMAL[None, :],
        n_incident=1.4,
        n_transmitted=1.0,
        incident_power=0.37,
    )
    _require_power_split(
        "below-critical incidence",
        below_critical_result,
        0.37,
    )
    below_critical = below_critical_result[0]
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

    above_critical_result = interface_transport(
        _direction(60.0)[None, :],
        _NORMAL[None, :],
        n_incident=1.4,
        n_transmitted=1.0,
        incident_power=0.37,
    )
    _require_power_split(
        "total internal reflection",
        above_critical_result,
        0.37,
    )
    above_critical = above_critical_result[0]
    if not bool(above_critical["total_internal_reflection"]):
        raise AssertionError("above-critical ray did not report TIR")
    if above_critical["reflectance"] != 1.0:
        raise AssertionError("TIR reflectance is not one")
    if above_critical["transmittance"] != 0.0:
        raise AssertionError("TIR transmittance is not zero")
    if above_critical["reflected_power"] != 0.37:
        raise AssertionError("TIR did not preserve incident power")
    if above_critical["refracted_power"] != 0.0:
        raise AssertionError("TIR produced refracted power")
    if np.any(np.isfinite(above_critical["refracted_direction"])):
        raise AssertionError("TIR returned a finite refracted direction")
    _require_unit(
        "TIR reflected direction",
        above_critical["reflected_direction"],
    )

    for label, invalid_power in (
        ("negative", -1.0),
        ("non-finite", np.nan),
        ("wrong-count", np.array((0.2, 0.3))),
    ):
        try:
            interface_transport(
                np.array(((0.0, 0.0, -1.0),)),
                _NORMAL[None, :],
                n_incident=1.0,
                n_transmitted=1.4,
                incident_power=invalid_power,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label} incident power was accepted")

    print("normal incidence air -> silicone: PASS")
    print("oblique incidence air -> silicone: PASS")
    print("below-critical silicone -> air: PASS")
    print("above-critical silicone -> air: PASS")
    print()
    print("Dielectric interface transport: PASS")


if __name__ == "__main__":
    main()
