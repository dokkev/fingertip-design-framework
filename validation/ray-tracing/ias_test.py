"""Verify one closest-hit query across a static multi-object OptiX IAS."""

from __future__ import annotations

from math import sqrt

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.mesh import make_fingertip_mesh
from lumo.ray_tracing import OptixScene


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2
OBJECT_INSTANCE_ID = 3

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
OBJECT_MASK = 0x04
ALL_MASK = SILICONE_MASK | CARRIER_MASK | OBJECT_MASK

_SPHERE_CENTER_M = np.array((0.030, 0.0, 0.0), dtype=np.float32)
_SPHERE_RADIUS_M = 0.005
_BARYCENTRIC_TOLERANCE = 1.0e-5


def _require_hit(
    label: str,
    result: np.void,
    *,
    instance_id: int,
    primitive_count: int,
    triangle: bool,
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
    if triangle:
        u, v = (float(value) for value in barycentrics)
        if not np.all(np.isfinite(barycentrics)):
            raise AssertionError(f"{label} barycentrics are not finite")
        if (
            u < -_BARYCENTRIC_TOLERANCE
            or v < -_BARYCENTRIC_TOLERANCE
            or u + v > 1.0 + _BARYCENTRIC_TOLERANCE
        ):
            raise AssertionError(f"{label} barycentrics lie outside the triangle")
    elif not np.array_equal(barycentrics, (-1.0, -1.0)):
        raise AssertionError(f"{label} should not report triangle barycentrics")


def _sphere_distance_m(origin: np.ndarray, direction: np.ndarray) -> float:
    offset = origin - _SPHERE_CENTER_M
    half_b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - _SPHERE_RADIUS_M**2
    discriminant = half_b**2 - c
    if discriminant < 0.0:
        raise AssertionError("CPU sphere reference unexpectedly missed")
    return -half_b - sqrt(discriminant)


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)

    silicone_triangle_count = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
    ).size // 3
    carrier_triangle_count = np.asarray(
        fingertip_mesh.carrier.indices,
    ).size // 3

    scene = OptixScene(
        fingertip_mesh,
        sphere_center=_SPHERE_CENTER_M,
        sphere_radius=_SPHERE_RADIUS_M,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        sphere_instance_id=OBJECT_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
        sphere_visibility_mask=OBJECT_MASK,
    )

    silicone_x_m = -0.0065
    silicone_origin = np.array(
        (silicone_x_m, 0.00073, -0.030),
        dtype=np.float32,
    )
    carrier_origin = np.array((0.0011, 0.00073, 0.010), dtype=np.float32)
    sphere_origin = np.array((0.030, 0.0, -0.020), dtype=np.float32)
    miss_origin = np.array((0.050, 0.0, -0.020), dtype=np.float32)
    positive_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    negative_z = np.array((0.0, 0.0, -1.0), dtype=np.float32)

    origins = np.stack(
        (silicone_origin, carrier_origin, sphere_origin, miss_origin)
    )
    directions = np.stack((positive_z, negative_z, positive_z, positive_z))
    results = scene.trace_closest(origins, directions, mask=ALL_MASK)

    silicone_result, carrier_result, sphere_result, miss_result = results
    _require_hit(
        "silicone",
        silicone_result,
        instance_id=SILICONE_INSTANCE_ID,
        primitive_count=silicone_triangle_count,
        triangle=True,
    )
    _require_hit(
        "carrier",
        carrier_result,
        instance_id=CARRIER_INSTANCE_ID,
        primitive_count=carrier_triangle_count,
        triangle=True,
    )
    _require_hit(
        "sphere",
        sphere_result,
        instance_id=OBJECT_INSTANCE_ID,
        primitive_count=1,
        triangle=False,
    )

    expected_sphere_t_m = _sphere_distance_m(sphere_origin, positive_z)
    if not np.isclose(
        sphere_result["t"],
        expected_sphere_t_m,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise AssertionError(
            f"sphere t={float(sphere_result['t']):.9e} m, "
            f"CPU reference={expected_sphere_t_m:.9e} m"
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
        primitive_count=silicone_triangle_count,
        triangle=True,
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
    print(
        "sphere:   PASS | "
        f"instance={int(sphere_result['instance_id'])} | "
        f"t={float(sphere_result['t']):.9e} m"
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
