"""Validate OptiX world-space geometric surface normals."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.mesh import make_fingertip_mesh
from lumo.ray_tracing import OptixScene
from lumo.ray_tracing.transport import interface_transport


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
_NORMAL_TOLERANCE = 2.0e-5


def _require_unit_normal(label: str, normal: np.ndarray) -> None:
    if not np.all(np.isfinite(normal)):
        raise AssertionError(f"{label} normal is not finite")
    if not np.isclose(
        np.linalg.norm(normal),
        1.0,
        rtol=0.0,
        atol=_NORMAL_TOLERANCE,
    ):
        raise AssertionError(f"{label} normal is not normalized")


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    scene = OptixScene(make_fingertip_mesh(fingertip))

    positive_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    negative_z = np.array((0.0, 0.0, -1.0), dtype=np.float32)

    carrier = scene.trace_closest(
        np.array(((0.0011, 0.00073, 0.010),), dtype=np.float32),
        negative_z[None, :],
        mask=CARRIER_MASK,
    )[0]
    if not bool(carrier["hit"]) or int(carrier["instance_id"]) != CARRIER_INSTANCE_ID:
        raise AssertionError("carrier ray did not hit the carrier")
    carrier_normal = np.asarray(carrier["normal_W"], dtype=np.float64)
    _require_unit_normal("carrier", carrier_normal)
    if abs(float(np.dot(carrier_normal, positive_z))) < 1.0 - _NORMAL_TOLERANCE:
        raise AssertionError("carrier normal is not parallel to the planar face")

    silicone_x_m = -0.0032
    silicone_origin = np.array(
        (silicone_x_m, 0.00090, -0.030),
        dtype=np.float32,
    )
    silicone = scene.trace_closest(
        silicone_origin[None, :],
        positive_z[None, :],
        mask=SILICONE_MASK,
    )[0]
    if (
        not bool(silicone["hit"])
        or int(silicone["instance_id"]) != SILICONE_INSTANCE_ID
    ):
        raise AssertionError("silicone ray did not hit the silicone")
    silicone_normal = np.asarray(silicone["normal_W"], dtype=np.float64)
    _require_unit_normal("silicone", silicone_normal)

    silicone_x_mm = 1.0e3 * silicone_x_m
    silicone_z_mm = (
        fingertip.silicone.ellipse_center_z_mm
        - semiellipse_depth_at_x_mm(
            half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
            height_mm=fingertip.silicone.ellipse_radius_z_mm,
            x_mm=silicone_x_mm,
        )
    )
    expected_silicone_normal = np.array(
        (
            silicone_x_mm / fingertip.silicone.ellipse_radius_x_mm**2,
            0.0,
            (
                silicone_z_mm - fingertip.silicone.ellipse_center_z_mm
            )
            / fingertip.silicone.ellipse_radius_z_mm**2,
        )
    )
    expected_silicone_normal /= np.linalg.norm(expected_silicone_normal)
    if abs(float(np.dot(silicone_normal, expected_silicone_normal))) < 0.995:
        raise AssertionError(
            "silicone mesh normal is inconsistent with the analytic semiellipse"
        )

    optical = interface_transport(
        positive_z[None, :],
        silicone["normal_W"][None, :],
        n_incident=1.0,
        n_transmitted=1.4,
        incident_power=1.0,
    )[0]
    if bool(optical["total_internal_reflection"]):
        raise AssertionError("air-to-silicone integration ray reported TIR")
    for field in ("reflected_direction", "refracted_direction"):
        direction = optical[field]
        if not np.all(np.isfinite(direction)) or not np.isclose(
            np.linalg.norm(direction),
            1.0,
            atol=1.0e-12,
        ):
            raise AssertionError(f"integration {field} is invalid")
    if not np.isclose(
        optical["reflectance"] + optical["transmittance"],
        1.0,
        atol=1.0e-12,
    ):
        raise AssertionError("integration R + T is not one")

    print(f"carrier normal:  PASS | n_W={carrier_normal}")
    print(f"silicone normal: PASS | n_W={silicone_normal}")
    print("OptiX -> dielectric interface: PASS")
    print()
    print("OptiX geometric surface normals: PASS")


if __name__ == "__main__":
    main()
