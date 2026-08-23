"""Trace one safely spawned refracted ray through the static fingertip."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.ray_tracing import (
    OptixScene,
    interface_transport,
    safe_secondary_origins,
)


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2
SPHERE_INSTANCE_ID = 3

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
SPHERE_MASK = 0x04
ALL_MASK = SILICONE_MASK | CARRIER_MASK | SPHERE_MASK

_SPHERE_CENTER_M = np.array((0.030, 0.0, 0.0), dtype=np.float32)
_SPHERE_RADIUS_M = 0.005


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    scene = OptixScene(
        make_fingertip_mesh(fingertip),
        sphere_center=_SPHERE_CENTER_M,
        sphere_radius=_SPHERE_RADIUS_M,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        sphere_instance_id=SPHERE_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
        sphere_visibility_mask=SPHERE_MASK,
    )

    primary_origins = np.array(
        ((-0.0070, 0.00090, -0.030),),
        dtype=np.float32,
    )
    primary_directions = np.array(
        ((0.0, 0.0, 1.0),),
        dtype=np.float32,
    )
    first_hits = scene.trace_closest(
        primary_origins,
        primary_directions,
        mask=ALL_MASK,
    )
    first_hit = first_hits[0]
    if (
        not bool(first_hit["hit"])
        or int(first_hit["instance_id"]) != SILICONE_INSTANCE_ID
    ):
        raise AssertionError("primary ray did not hit exposed silicone")

    barycentrics = np.asarray(first_hit["barycentrics"])
    barycentric_weights = np.array(
        (
            1.0 - barycentrics.sum(),
            barycentrics[0],
            barycentrics[1],
        )
    )
    if barycentric_weights.min() < 0.2:
        raise AssertionError("primary hit is too close to a triangle edge")
    if not (
        np.all(np.isfinite(first_hit["spawn_front_W"]))
        and np.all(np.isfinite(first_hit["spawn_back_W"]))
    ):
        raise AssertionError("primary triangle did not return OTK spawn points")

    optical = interface_transport(
        primary_directions,
        first_hits["normal_W"],
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=1.0,
    )
    refracted_directions = optical["refracted_direction"]
    if bool(optical["total_internal_reflection"][0]):
        raise AssertionError("air-to-silicone ray unexpectedly reported TIR")
    if not np.all(np.isfinite(refracted_directions)):
        raise AssertionError("refracted direction is not finite")
    if not np.isclose(
        np.linalg.norm(refracted_directions[0]),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("refracted direction is not normalized")

    secondary_origins = safe_secondary_origins(
        first_hits,
        refracted_directions,
    )
    second_hits = scene.trace_closest(
        secondary_origins,
        refracted_directions,
        mask=ALL_MASK,
    )
    second_hit = second_hits[0]
    if (
        int(second_hit["instance_id"]) == SILICONE_INSTANCE_ID
        and int(second_hit["primitive_id"])
        == int(first_hit["primitive_id"])
    ):
        raise AssertionError("secondary ray self-hit the primary triangle")
    if (
        not bool(second_hit["hit"])
        or int(second_hit["instance_id"]) != CARRIER_INSTANCE_ID
    ):
        raise AssertionError("refracted secondary ray did not hit the carrier")
    if not np.isfinite(second_hit["t"]) or float(second_hit["t"]) <= 0.0:
        raise AssertionError("secondary carrier hit distance is invalid")

    print(
        "first hit:  silicone | "
        f"primitive={int(first_hit['primitive_id'])} | "
        f"t={1.0e3 * float(first_hit['t']):.6f} mm"
    )
    print(
        "second hit: carrier  | "
        f"primitive={int(second_hit['primitive_id'])} | "
        f"t={1.0e3 * float(second_hit['t']):.6f} mm"
    )
    print("OTK safe spawn: PASS")
    print()
    print("One refracted secondary ray: PASS")


if __name__ == "__main__":
    main()
