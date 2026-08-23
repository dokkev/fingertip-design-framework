"""Validate one reflected/refracted power split in the static fingertip."""

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

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
ALL_MASK = SILICONE_MASK | CARRIER_MASK


def main() -> None:
    scene = OptixScene(
        make_fingertip_mesh(Fingertip(FingertipParameters())),
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
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

    barycentrics = np.asarray(primary_hit["barycentrics"])
    barycentric_weights = np.array(
        (1.0 - barycentrics.sum(), barycentrics[0], barycentrics[1])
    )
    if barycentric_weights.min() < 0.2:
        raise AssertionError("primary hit is too close to a triangle edge")
    if not (
        np.all(np.isfinite(primary_hit["spawn_front_W"]))
        and np.all(np.isfinite(primary_hit["spawn_back_W"]))
    ):
        raise AssertionError("primary triangle did not return OTK spawn points")

    incident_power = 1.0
    optical = interface_transport(
        primary_directions,
        primary_hits["normal_W"],
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=incident_power,
    )
    if bool(optical["total_internal_reflection"][0]):
        raise AssertionError("air-to-silicone ray unexpectedly reported TIR")
    if not np.isclose(
        optical["reflected_power"][0] + optical["refracted_power"][0],
        incident_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("interface did not conserve incident ray power")
    if not (
        optical["reflected_power"][0] > 0.0
        and optical["refracted_power"][0] > 0.0
    ):
        raise AssertionError("interface did not produce two positive branches")

    reflected_origins = safe_secondary_origins(
        primary_hits,
        optical["reflected_direction"],
    )
    refracted_origins = safe_secondary_origins(
        primary_hits,
        optical["refracted_direction"],
    )
    reflected_hit = scene.trace_closest(
        reflected_origins,
        optical["reflected_direction"],
        mask=ALL_MASK,
    )[0]
    refracted_hit = scene.trace_closest(
        refracted_origins,
        optical["refracted_direction"],
        mask=ALL_MASK,
    )[0]

    for label, branch_hit in (
        ("reflected", reflected_hit),
        ("refracted", refracted_hit),
    ):
        if (
            bool(branch_hit["hit"])
            and int(branch_hit["instance_id"]) == SILICONE_INSTANCE_ID
            and int(branch_hit["primitive_id"])
            == int(primary_hit["primitive_id"])
        ):
            raise AssertionError(f"{label} branch self-hit the primary triangle")

    if bool(reflected_hit["hit"]):
        raise AssertionError("reflected branch did not leave the fingertip")
    if (
        not bool(refracted_hit["hit"])
        or int(refracted_hit["instance_id"]) != CARRIER_INSTANCE_ID
    ):
        raise AssertionError("refracted branch did not hit the carrier")
    if not np.isfinite(refracted_hit["t"]) or float(refracted_hit["t"]) <= 0.0:
        raise AssertionError("refracted carrier hit distance is invalid")

    print(
        "primary:   silicone | "
        f"primitive={int(primary_hit['primitive_id'])} | "
        f"t={1.0e3 * float(primary_hit['t']):.6f} mm"
    )
    print(
        "reflected: miss     | "
        f"power={float(optical['reflected_power'][0]):.9f}"
    )
    print(
        "refracted: carrier  | "
        f"primitive={int(refracted_hit['primitive_id'])} | "
        f"t={1.0e3 * float(refracted_hit['t']):.6f} mm | "
        f"power={float(optical['refracted_power'][0]):.9f}"
    )
    print()
    print("One dielectric power branch: PASS")


if __name__ == "__main__":
    main()
