"""Validate one opaque Lambertian carrier reflection in the static fingertip."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.ray_tracing import OptixScene
from lumo.ray_tracing.scene import safe_secondary_origins
from lumo.ray_tracing.transport import (
    interface_transport,
    lambertian_reflection,
)


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
ALL_MASK = SILICONE_MASK | CARRIER_MASK
_CARRIER_ALBEDO = 0.7


def main() -> None:
    incoming = np.repeat(np.array(((0.0, 0.0, -1.0),)), 3, axis=0)
    normals = np.repeat(np.array(((0.0, 0.0, 1.0),)), 3, axis=0)
    incident_powers = np.array((1.0, 1.0, 0.37))
    albedos = np.array((0.7, 0.0, 1.0))
    local = lambertian_reflection(
        incoming,
        normals,
        incident_power=incident_powers,
        albedo=albedos,
        u1=0.36,
        u2=0.125,
    )
    if not np.allclose(
        local["reflected_power"],
        (0.7, 0.0, 0.37),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("Lambertian reflected power is wrong")
    if not np.allclose(
        local["absorbed_power"],
        (0.3, 1.0, 0.0),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("Lambertian absorbed power is wrong")
    if not np.allclose(
        local["reflected_power"] + local["absorbed_power"],
        incident_powers,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("Lambertian interaction does not conserve power")
    if not np.allclose(
        np.linalg.norm(local["reflected_direction"], axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("Lambertian direction is not normalized")
    if not np.all(
        np.sum(local["reflected_direction"] * normals, axis=1) > 0.0
    ):
        raise AssertionError("Lambertian direction left the reflected hemisphere")
    if np.allclose(
        local["reflected_direction"][0],
        normals[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("Lambertian direction test is not oblique")

    reversed_normal = lambertian_reflection(
        incoming[:1],
        -normals[:1],
        incident_power=1.0,
        albedo=0.7,
        u1=0.36,
        u2=0.125,
    )
    if not np.allclose(
        reversed_normal["reflected_direction"],
        local["reflected_direction"][:1],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("Lambertian reflection depends on mesh winding")

    scene = OptixScene(
        make_fingertip_mesh(Fingertip(FingertipParameters()))
    )

    primary_origins = np.array(
        ((-0.0070, 0.00090, -0.030),),
        dtype=np.float32,
    )
    primary_directions = np.array(((0.0, 0.0, 1.0),), dtype=np.float32)
    primary_hits = scene.trace_closest(
        primary_origins,
        primary_directions,
        mask=ALL_MASK,
    )
    primary_hit = primary_hits[0]
    if (
        not bool(primary_hit["hit"])
        or int(primary_hit["instance_id"]) != SILICONE_INSTANCE_ID
    ):
        raise AssertionError("primary ray did not hit exposed silicone")

    interface = interface_transport(
        primary_directions,
        primary_hits["normal_W"],
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=1.0,
    )
    if bool(interface["total_internal_reflection"][0]):
        raise AssertionError("air-to-silicone ray unexpectedly reported TIR")

    carrier_origins = safe_secondary_origins(
        primary_hits,
        interface["refracted_direction"],
    )
    carrier_hits = scene.trace_closest(
        carrier_origins,
        interface["refracted_direction"],
        mask=ALL_MASK,
    )
    carrier_hit = carrier_hits[0]
    if (
        not bool(carrier_hit["hit"])
        or int(carrier_hit["instance_id"]) != CARRIER_INSTANCE_ID
    ):
        raise AssertionError("refracted ray did not hit the carrier")

    carrier_scatter = lambertian_reflection(
        interface["refracted_direction"],
        carrier_hits["normal_W"],
        incident_power=interface["refracted_power"],
        albedo=_CARRIER_ALBEDO,
        u1=0.0,
        u2=0.0,
    )
    expected_reflected_power = (
        interface["refracted_power"][0] * _CARRIER_ALBEDO
    )
    expected_absorbed_power = (
        interface["refracted_power"][0] * (1.0 - _CARRIER_ALBEDO)
    )
    if not np.isclose(
        carrier_scatter["reflected_power"][0],
        expected_reflected_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("carrier reflected power is wrong")
    if not np.isclose(
        carrier_scatter["absorbed_power"][0],
        expected_absorbed_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("carrier absorbed power is wrong")
    if not np.isclose(
        carrier_scatter["reflected_power"][0]
        + carrier_scatter["absorbed_power"][0],
        interface["refracted_power"][0],
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("carrier interaction does not conserve power")

    carrier_direction = carrier_scatter["reflected_direction"][0]
    if not np.all(np.isfinite(carrier_direction)) or not np.isclose(
        np.linalg.norm(carrier_direction),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("carrier-reflected direction is invalid")
    oriented_carrier_normal = np.asarray(
        carrier_hit["normal_W"],
        dtype=np.float64,
    )
    if (
        np.dot(interface["refracted_direction"][0], oriented_carrier_normal)
        > 0.0
    ):
        oriented_carrier_normal = -oriented_carrier_normal
    if np.dot(carrier_direction, oriented_carrier_normal) <= 0.0:
        raise AssertionError("carrier reflection entered the wrong hemisphere")

    return_origins = safe_secondary_origins(
        carrier_hits,
        carrier_scatter["reflected_direction"],
    )
    return_hit = scene.trace_closest(
        return_origins,
        carrier_scatter["reflected_direction"],
        mask=ALL_MASK,
    )[0]
    if (
        bool(return_hit["hit"])
        and int(return_hit["instance_id"]) == CARRIER_INSTANCE_ID
        and int(return_hit["primitive_id"])
        == int(carrier_hit["primitive_id"])
    ):
        raise AssertionError("Lambertian ray self-hit the carrier triangle")
    if (
        not bool(return_hit["hit"])
        or int(return_hit["instance_id"]) != SILICONE_INSTANCE_ID
    ):
        raise AssertionError("carrier-reflected ray did not hit exposed silicone")
    if not np.isfinite(return_hit["t"]) or float(return_hit["t"]) <= 0.0:
        raise AssertionError("silicone return-hit distance is invalid")

    print("local Lambertian power and direction: PASS")
    print(
        "primary: silicone | "
        f"primitive={int(primary_hit['primitive_id'])} | "
        f"t={1.0e3 * float(primary_hit['t']):.6f} mm"
    )
    print(
        "carrier: hit      | "
        f"primitive={int(carrier_hit['primitive_id'])} | "
        f"incident_power={float(interface['refracted_power'][0]):.9f} | "
        f"reflected_power="
        f"{float(carrier_scatter['reflected_power'][0]):.9f} | "
        f"absorbed_power={float(carrier_scatter['absorbed_power'][0]):.9f}"
    )
    print(
        "return:  silicone | "
        f"primitive={int(return_hit['primitive_id'])} | "
        f"t={1.0e3 * float(return_hit['t']):.6f} mm"
    )
    print()
    print("One opaque Lambertian carrier reflection: PASS")


if __name__ == "__main__":
    main()
