"""Verify silicone GAS and IAS UPDATE against a fresh OptiX build."""

from __future__ import annotations

from time import perf_counter

import cupy as cp
import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
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
_DELTA_Z_M = 1.0e-3
_T_TOLERANCE_M = 2.0e-6
_BARYCENTRIC_TOLERANCE = 2.0e-5


def _make_scene(
    silicone_vertices: np.ndarray,
    silicone_triangles: np.ndarray,
    carrier_vertices: np.ndarray,
    carrier_triangles: np.ndarray,
) -> OptixScene:
    return OptixScene(
        silicone_vertices=silicone_vertices,
        silicone_triangles=silicone_triangles,
        carrier_vertices=carrier_vertices,
        carrier_triangles=carrier_triangles,
        sphere_center=_SPHERE_CENTER_M,
        sphere_radius=_SPHERE_RADIUS_M,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        sphere_instance_id=OBJECT_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
        sphere_visibility_mask=OBJECT_MASK,
    )


def _assert_same_result(
    label: str,
    left: np.void,
    right: np.void,
) -> None:
    for field in ("hit", "instance_id", "primitive_id"):
        if left[field] != right[field]:
            raise AssertionError(f"{label} differs in {field}")
    if not np.isclose(
        left["t"],
        right["t"],
        rtol=0.0,
        atol=_T_TOLERANCE_M,
    ):
        raise AssertionError(f"{label} differs in hit distance")
    if not np.allclose(
        left["barycentrics"],
        right["barycentrics"],
        rtol=0.0,
        atol=_BARYCENTRIC_TOLERANCE,
    ):
        raise AssertionError(f"{label} differs in barycentrics")


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)

    silicone_vertices = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float32,
    )
    silicone_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    bonded_vertices = np.zeros(len(silicone_vertices), dtype=bool)
    bonded_vertices[fingertip_mesh.bonded_vertex_indices] = True
    silicone_triangles = silicone_triangles[
        ~np.all(bonded_vertices[silicone_triangles], axis=1)
    ]

    carrier_vertices = np.asarray(
        fingertip_mesh.carrier.vertices,
        dtype=np.float32,
    )
    carrier_triangles = np.asarray(
        fingertip_mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)

    triangle_vertices = silicone_vertices[silicone_triangles]
    triangle_edges_1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    triangle_edges_2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    triangle_normals = np.cross(triangle_edges_1, triangle_edges_2)
    normal_lengths = np.linalg.norm(triangle_normals, axis=1)
    bottom_candidates = np.flatnonzero(
        np.abs(triangle_normals[:, 2]) > 0.8 * normal_lengths
    )
    triangle_centroids = triangle_vertices.mean(axis=1)
    silicone_primitive_id = int(
        bottom_candidates[
            np.argmin(triangle_centroids[bottom_candidates, 2])
        ]
    )
    silicone_hit_point = triangle_centroids[silicone_primitive_id]
    silicone_origin = silicone_hit_point.copy()
    silicone_origin[2] -= 0.010

    carrier_origin = np.array((0.0011, 0.00073, 0.010), dtype=np.float32)
    sphere_origin = np.array((0.030, 0.0, -0.020), dtype=np.float32)
    miss_origin = np.array((0.050, 0.0, -0.020), dtype=np.float32)
    positive_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    negative_z = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    origins = np.stack(
        (silicone_origin, carrier_origin, sphere_origin, miss_origin)
    )
    directions = np.stack((positive_z, negative_z, positive_z, positive_z))

    updated_scene = _make_scene(
        silicone_vertices,
        silicone_triangles,
        carrier_vertices,
        carrier_triangles,
    )
    initial_results = updated_scene.trace_closest(
        origins,
        directions,
        mask=ALL_MASK,
    )
    initial_silicone = initial_results[0]
    if not bool(initial_silicone["hit"]):
        raise AssertionError("initial silicone ray missed")
    if int(initial_silicone["instance_id"]) != SILICONE_INSTANCE_ID:
        raise AssertionError("initial ray did not hit silicone")
    if int(initial_silicone["primitive_id"]) != silicone_primitive_id:
        raise AssertionError("initial ray did not hit the selected triangle")
    if not np.isfinite(initial_silicone["t"]) or initial_silicone["t"] <= 0.0:
        raise AssertionError("initial silicone ray has invalid distance")
    for label, result, expected_instance_id in (
        ("carrier", initial_results[1], CARRIER_INSTANCE_ID),
        ("sphere", initial_results[2], OBJECT_INSTANCE_ID),
    ):
        if not bool(result["hit"]):
            raise AssertionError(f"initial {label} ray missed")
        if int(result["instance_id"]) != expected_instance_id:
            raise AssertionError(f"initial {label} ray hit the wrong instance")
        if not np.isfinite(result["t"]) or float(result["t"]) <= 0.0:
            raise AssertionError(f"initial {label} ray has invalid distance")
    if bool(initial_results[3]["hit"]):
        raise AssertionError("initial miss ray unexpectedly hit the scene")
    initial_barycentrics = np.asarray(initial_silicone["barycentrics"])
    barycentric_weights = np.array(
        (
            1.0 - initial_barycentrics.sum(),
            initial_barycentrics[0],
            initial_barycentrics[1],
        )
    )
    if np.min(barycentric_weights) < 0.2:
        raise AssertionError("silicone validation ray is too close to an edge")

    displaced_vertices = silicone_vertices.copy()
    displaced_vertices[:, 2] += _DELTA_Z_M

    update_start = cp.cuda.Event()
    update_end = cp.cuda.Event()
    update_start.record(updated_scene._stream)
    updated_scene.update_silicone(displaced_vertices)
    update_end.record(updated_scene._stream)
    update_end.synchronize()
    update_time_ms = cp.cuda.get_elapsed_time(update_start, update_end)

    updated_results = updated_scene.trace_closest(
        origins,
        directions,
        mask=ALL_MASK,
    )
    updated_silicone = updated_results[0]
    if not bool(updated_silicone["hit"]):
        raise AssertionError("translated silicone ray missed")
    if int(updated_silicone["instance_id"]) != SILICONE_INSTANCE_ID:
        raise AssertionError("translated ray did not hit silicone")
    if updated_silicone["primitive_id"] != initial_silicone["primitive_id"]:
        raise AssertionError("silicone primitive changed after rigid translation")
    measured_delta_t_m = float(updated_silicone["t"] - initial_silicone["t"])
    if not np.isclose(
        measured_delta_t_m,
        _DELTA_Z_M,
        rtol=0.0,
        atol=_T_TOLERANCE_M,
    ):
        raise AssertionError("silicone hit did not move by the prescribed 1 mm")
    if not np.allclose(
        updated_silicone["barycentrics"],
        initial_silicone["barycentrics"],
        rtol=0.0,
        atol=_BARYCENTRIC_TOLERANCE,
    ):
        raise AssertionError("silicone barycentrics changed after translation")

    for label, result_index in (("carrier", 1), ("sphere", 2), ("miss", 3)):
        _assert_same_result(
            label,
            initial_results[result_index],
            updated_results[result_index],
        )

    fresh_build_start = perf_counter()
    fresh_scene = _make_scene(
        displaced_vertices,
        silicone_triangles,
        carrier_vertices,
        carrier_triangles,
    )
    fresh_scene._stream.synchronize()
    fresh_build_time_ms = 1.0e3 * (perf_counter() - fresh_build_start)
    fresh_results = fresh_scene.trace_closest(
        origins,
        directions,
        mask=ALL_MASK,
    )

    for label, updated, fresh in zip(
        ("silicone", "carrier", "sphere", "miss"),
        updated_results,
        fresh_results,
        strict=True,
    ):
        _assert_same_result(label, updated, fresh)

    print(
        "initial silicone:    PASS | "
        f"t={1.0e3 * float(initial_silicone['t']):.6f} mm"
    )
    print(
        "translated silicone: PASS | "
        f"dt={1.0e3 * measured_delta_t_m:.6f} mm"
    )
    print("carrier unchanged:   PASS")
    print("sphere unchanged:    PASS")
    print("miss unchanged:      PASS")
    print("update vs rebuild:   PASS")
    print()
    print(f"update time:      {update_time_ms:.3f} ms")
    print(f"fresh build time: {fresh_build_time_ms:.3f} ms")
    print()
    print("OptiX silicone GAS/IAS refit: PASS")


if __name__ == "__main__":
    main()
