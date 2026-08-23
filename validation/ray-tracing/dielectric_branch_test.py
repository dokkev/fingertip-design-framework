"""Validate sampled smooth-dielectric branch selection and path weight."""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np

from lumo.ray_tracing import interface_transport
from lumo.ray_tracing.path import _sample_dielectric_branches


def main() -> None:
    normal = np.array(((0.0, 0.0, 1.0),))
    normal_incidence = np.array(((0.0, 0.0, -1.0),))
    air_to_silicone = interface_transport(
        np.repeat(normal_incidence, 2, axis=0),
        np.repeat(normal, 2, axis=0),
        n_incident=1.0,
        n_transmitted=1.41,
        incident_power=np.array((0.2, 0.3)),
    )

    angle = radians(60.0)
    tir = interface_transport(
        np.array(((sin(angle), 0.0, -cos(angle)),)),
        normal,
        n_incident=1.41,
        n_transmitted=1.0,
        incident_power=0.5,
    )
    optical = np.concatenate((air_to_silicone, tir))
    reflectance = float(air_to_silicone["reflectance"][0])
    incident_power = np.array((0.2, 0.3, 0.5))
    directions, inside, selected_power = _sample_dielectric_branches(
        optical,
        np.array((0.5 * reflectance, 0.5 * (1.0 + reflectance), 0.999)),
        np.array((False, False, True)),
        incident_power,
    )

    if not np.allclose(
        directions[0],
        optical["reflected_direction"][0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("u < R did not select reflection")
    if not np.allclose(
        directions[1],
        optical["refracted_direction"][1],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("u >= R did not select transmission")
    if not np.allclose(
        directions[2],
        optical["reflected_direction"][2],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("total internal reflection was not reflected")
    if not np.array_equal(inside, np.array((False, True, True))):
        raise AssertionError("dielectric branch changed the wrong medium state")
    if not np.array_equal(selected_power, incident_power):
        raise AssertionError("Fresnel branch probability was applied to power")

    print(f"normal-incidence reflectance: {reflectance:.9f}")
    print("reflection/transmission/TIR selection: PASS")
    print("medium toggle and lossless path power: PASS")
    print()
    print("Sampled dielectric branch regression: PASS")


if __name__ == "__main__":
    main()
