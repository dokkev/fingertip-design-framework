"""Validate one complete air-to-silicone-to-carrier-to-air optical path."""

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

_N_AIR = 1.0
_N_SILICONE = 1.4
_CARRIER_ALBEDO = 0.7
_MIN_BARYCENTRIC_WEIGHT = 0.1


def main() -> None:
    scene = OptixScene(
        make_fingertip_mesh(Fingertip(FingertipParameters()))
    )

    incident_power = 1.0
    primary_origins = np.array(
        ((-0.0070, 0.00090, -0.030),),
        dtype=np.float32,
    )
    primary_directions = np.array(((0.0, 0.0, 1.0),), dtype=np.float32)
    silicone_entry_hits = scene.trace_closest(
        primary_origins,
        primary_directions,
        mask=ALL_MASK,
    )
    silicone_entry_hit = silicone_entry_hits[0]
    if (
        not bool(silicone_entry_hit["hit"])
        or int(silicone_entry_hit["instance_id"]) != SILICONE_INSTANCE_ID
    ):
        raise AssertionError("primary ray did not hit exposed silicone")
    entry_barycentrics = np.asarray(silicone_entry_hit["barycentrics"])
    entry_barycentric_weights = np.array(
        (
            1.0 - entry_barycentrics.sum(),
            entry_barycentrics[0],
            entry_barycentrics[1],
        )
    )
    if entry_barycentric_weights.min() < _MIN_BARYCENTRIC_WEIGHT:
        raise AssertionError("silicone entry is too close to a triangle edge")

    entry_transport = interface_transport(
        primary_directions,
        silicone_entry_hits["normal_W"],
        n_incident=_N_AIR,
        n_transmitted=_N_SILICONE,
        incident_power=incident_power,
    )
    if bool(entry_transport["total_internal_reflection"][0]):
        raise AssertionError("air-to-silicone ray unexpectedly reported TIR")
    if not np.isclose(
        entry_transport["reflected_power"][0]
        + entry_transport["refracted_power"][0],
        incident_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("silicone entry did not conserve power")

    carrier_origins = safe_secondary_origins(
        silicone_entry_hits,
        entry_transport["refracted_direction"],
    )
    carrier_hits = scene.trace_closest(
        carrier_origins,
        entry_transport["refracted_direction"],
        mask=ALL_MASK,
    )
    carrier_hit = carrier_hits[0]
    if (
        bool(carrier_hit["hit"])
        and int(carrier_hit["instance_id"]) == SILICONE_INSTANCE_ID
        and int(carrier_hit["primitive_id"])
        == int(silicone_entry_hit["primitive_id"])
    ):
        raise AssertionError("entry secondary ray self-hit the silicone triangle")
    if (
        not bool(carrier_hit["hit"])
        or int(carrier_hit["instance_id"]) != CARRIER_INSTANCE_ID
    ):
        raise AssertionError("silicone-transmitted ray did not hit the carrier")
    carrier_barycentrics = np.asarray(carrier_hit["barycentrics"])
    carrier_barycentric_weights = np.array(
        (
            1.0 - carrier_barycentrics.sum(),
            carrier_barycentrics[0],
            carrier_barycentrics[1],
        )
    )
    if carrier_barycentric_weights.min() < _MIN_BARYCENTRIC_WEIGHT:
        raise AssertionError("carrier hit is too close to a triangle edge")

    carrier_reflection = lambertian_reflection(
        entry_transport["refracted_direction"],
        carrier_hits["normal_W"],
        incident_power=entry_transport["refracted_power"],
        albedo=_CARRIER_ALBEDO,
        u1=0.25,
        u2=0.75,
    )
    if not np.isclose(
        carrier_reflection["reflected_power"][0]
        + carrier_reflection["absorbed_power"][0],
        entry_transport["refracted_power"][0],
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("carrier reflection did not conserve power")

    silicone_return_origins = safe_secondary_origins(
        carrier_hits,
        carrier_reflection["reflected_direction"],
    )
    silicone_return_hits = scene.trace_closest(
        silicone_return_origins,
        carrier_reflection["reflected_direction"],
        mask=ALL_MASK,
    )
    silicone_return_hit = silicone_return_hits[0]
    if (
        bool(silicone_return_hit["hit"])
        and int(silicone_return_hit["instance_id"]) == CARRIER_INSTANCE_ID
        and int(silicone_return_hit["primitive_id"])
        == int(carrier_hit["primitive_id"])
    ):
        raise AssertionError("Lambertian secondary ray self-hit the carrier")
    if (
        not bool(silicone_return_hit["hit"])
        or int(silicone_return_hit["instance_id"]) != SILICONE_INSTANCE_ID
    ):
        raise AssertionError("carrier-reflected ray did not hit exposed silicone")
    if not (
        np.all(np.isfinite(silicone_return_hit["normal_W"]))
        and np.all(np.isfinite(silicone_return_hit["spawn_front_W"]))
        and np.all(np.isfinite(silicone_return_hit["spawn_back_W"]))
    ):
        raise AssertionError("silicone exit hit has invalid normal or spawn data")
    exit_barycentrics = np.asarray(silicone_return_hit["barycentrics"])
    exit_barycentric_weights = np.array(
        (
            1.0 - exit_barycentrics.sum(),
            exit_barycentrics[0],
            exit_barycentrics[1],
        )
    )
    if exit_barycentric_weights.min() < _MIN_BARYCENTRIC_WEIGHT:
        raise AssertionError("silicone exit is too close to a triangle edge")

    exit_transport = interface_transport(
        carrier_reflection["reflected_direction"],
        silicone_return_hits["normal_W"],
        n_incident=_N_SILICONE,
        n_transmitted=_N_AIR,
        incident_power=carrier_reflection["reflected_power"],
    )
    if bool(exit_transport["total_internal_reflection"][0]):
        raise AssertionError("selected silicone exit ray reported TIR")
    exit_direction = exit_transport["refracted_direction"][0]
    if not np.all(np.isfinite(exit_direction)) or not np.isclose(
        np.linalg.norm(exit_direction),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("silicone-to-air refracted direction is invalid")
    if not np.isclose(
        exit_transport["reflected_power"][0]
        + exit_transport["refracted_power"][0],
        carrier_reflection["reflected_power"][0],
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("silicone exit did not conserve power")
    escaped_power = float(exit_transport["refracted_power"][0])
    if escaped_power <= 0.0:
        raise AssertionError("silicone exit produced no transmitted power")

    exit_origins = safe_secondary_origins(
        silicone_return_hits,
        exit_transport["refracted_direction"],
    )
    final_hit = scene.trace_closest(
        exit_origins,
        exit_transport["refracted_direction"],
        mask=ALL_MASK,
    )[0]
    if (
        bool(final_hit["hit"])
        and int(final_hit["instance_id"]) == SILICONE_INSTANCE_ID
        and int(final_hit["primitive_id"])
        == int(silicone_return_hit["primitive_id"])
    ):
        raise AssertionError("exit secondary ray self-hit the silicone triangle")
    if bool(final_hit["hit"]):
        raise AssertionError("silicone-transmitted ray did not escape the scene")

    accounted_power = float(
        entry_transport["reflected_power"][0]
        + carrier_reflection["absorbed_power"][0]
        + exit_transport["reflected_power"][0]
        + escaped_power
    )
    if not np.isclose(
        accounted_power,
        incident_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("complete optical path did not conserve power")

    print(
        "entry:   silicone | "
        f"P_in={incident_power:.6f} | "
        f"P_t={float(entry_transport['refracted_power'][0]):.6f}"
    )
    print(
        "carrier: hit      | "
        f"P_reflected={float(carrier_reflection['reflected_power'][0]):.6f} | "
        f"P_absorbed={float(carrier_reflection['absorbed_power'][0]):.6f}"
    )
    print(
        "exit:    silicone | "
        f"P_out={float(carrier_reflection['reflected_power'][0]):.6f} | "
        f"P_escaped={escaped_power:.6f}"
    )
    print("escape:  miss")
    print(f"power:   PASS | accounted={accounted_power:.6f}")
    print()
    print("One air -> silicone -> carrier -> silicone -> air path: PASS")


if __name__ == "__main__":
    main()
