"""Verify one closest-hit query across a static multi-object OptiX IAS."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.mesh import make_fingertip_mesh
from lumo.ray_tracing import OptixScene


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
ALL_MASK = SILICONE_MASK | CARRIER_MASK

_BARYCENTRIC_TOLERANCE = 1.0e-5


def _require_hit(
    label: str,
    result: np.void,
    *,
    instance_id: int,
    primitive_count: int,
) -> None:
    if not bool(result["hit"]):
        raise AssertionError(f"{label} unexpectedly missed")
    if int(result["instance_id"]) != instance_id:
        raise AssertionError(
            f"{label} hit instance {int(result['instance_id'])}, "
            f"expected {instance_id}"
        )
    if not np.isfinite(result["t"]) or float(result["t"]) <= 0.0:
        raise AssertionError(f"{label} returned an invalid hit distance")
    primitive_id = int(result["primitive_id"])
    if not 0 <= primitive_id < primitive_count:
        raise AssertionError(f"{label} returned an invalid primitive ID")

    barycentrics = np.asarray(result["barycentrics"])
    u, v = (float(value) for value in barycentrics)
    if not np.all(np.isfinite(barycentrics)):
        raise AssertionError(f"{label} barycentrics are not finite")
    if (
        u < -_BARYCENTRIC_TOLERANCE
        or v < -_BARYCENTRIC_TOLERANCE
        or u + v > 1.0 + _BARYCENTRIC_TOLERANCE
    ):
        raise AssertionError(f"{label} barycentrics lie outside the triangle")


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)

    silicone_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int64,
    ).reshape(-1, 3)
    bonded_vertices = np.zeros(
        len(fingertip_mesh.silicone.vertices),
        dtype=bool,
    )
    bonded_vertices[fingertip_mesh.bonded_vertex_indices] = True
    exposed_silicone_triangle_count = int(
        np.count_nonzero(
            ~np.all(bonded_vertices[silicone_triangles], axis=1),
        )
    )
    carrier_triangle_count = np.asarray(
        fingertip_mesh.carrier.indices,
    ).size // 3

    scene = OptixScene(
        fingertip_mesh,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
    )

    silicone_x_m = -0.0065
    silicone_origin = np.array(
        (silicone_x_m, 0.00073, -0.030),
        dtype=np.float32,
    )
    carrier_origin = np.array((0.0011, 0.00073, 0.010), dtype=np.float32)
    miss_origin = np.array((0.050, 0.0, -0.020), dtype=np.float32)
    positive_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    negative_z = np.array((0.0, 0.0, -1.0), dtype=np.float32)

    origins = np.stack((silicone_origin, carrier_origin, miss_origin))
    directions = np.stack((positive_z, negative_z, positive_z))
    results = scene.trace_closest(origins, directions, mask=ALL_MASK)

    silicone_result, carrier_result, miss_result = results
    _require_hit(
        "silicone",
        silicone_result,
        instance_id=SILICONE_INSTANCE_ID,
        primitive_count=exposed_silicone_triangle_count,
    )
    _require_hit(
        "carrier",
        carrier_result,
        instance_id=CARRIER_INSTANCE_ID,
        primitive_count=carrier_triangle_count,
    )

    if bool(miss_result["hit"]):
        raise AssertionError("miss ray unexpectedly hit the IAS")
    if (
        float(miss_result["t"]) != -1.0
        or int(miss_result["instance_id"]) != -1
        or int(miss_result["primitive_id"]) != -1
        or not np.array_equal(miss_result["barycentrics"], (-1.0, -1.0))
        or not np.all(np.isnan(miss_result["normal_W"]))
        or not np.all(np.isnan(miss_result["spawn_front_W"]))
        or not np.all(np.isnan(miss_result["spawn_back_W"]))
    ):
        raise AssertionError("miss ray did not return the designated miss values")

    masked_result = scene.trace_closest(
        carrier_origin[None, :],
        negative_z[None, :],
        mask=ALL_MASK & ~CARRIER_MASK,
    )[0]
    _require_hit(
        "mask",
        masked_result,
        instance_id=SILICONE_INSTANCE_ID,
        primitive_count=exposed_silicone_triangle_count,
    )

    expected_silicone_z_m = 1.0e-3 * (
        fingertip.silicone.ellipse_center_z_mm
        - semiellipse_depth_at_x_mm(
            half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
            height_mm=fingertip.silicone.ellipse_radius_z_mm,
            x_mm=1.0e3 * silicone_x_m,
        )
    )
    expected_silicone_t_m = expected_silicone_z_m - silicone_origin[2]
    if not np.isclose(
        silicone_result["t"],
        expected_silicone_t_m,
        rtol=0.0,
        atol=2.0e-4,
    ):
        raise AssertionError("silicone hit is inconsistent with analytic geometry")

    print(
        "silicone: PASS | "
        f"instance={int(silicone_result['instance_id'])} | "
        f"t={float(silicone_result['t']):.9e} m"
    )
    print(
        "carrier:  PASS | "
        f"instance={int(carrier_result['instance_id'])} | "
        f"t={float(carrier_result['t']):.9e} m"
    )
    print("miss:     PASS")
    print(
        "mask:     PASS | "
        f"next instance={int(masked_result['instance_id'])} | "
        f"t={float(masked_result['t']):.9e} m"
    )
    print()
    print("OptiX IAS closest-hit query: PASS")


if __name__ == "__main__":
    main()
