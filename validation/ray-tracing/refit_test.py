"""Verify silicone GAS and IAS UPDATE against a fresh OptiX build."""

from __future__ import annotations

from time import perf_counter

import cupy as cp
import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.ray_tracing import OptixScene


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
ALL_MASK = SILICONE_MASK | CARRIER_MASK

_DELTA_Z_M = 1.0e-3
_T_TOLERANCE_M = 2.0e-6
_BARYCENTRIC_TOLERANCE = 2.0e-5


def _make_scene(
    fingertip_mesh: FingertipMesh,
) -> OptixScene:
    return OptixScene(fingertip_mesh)


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
    if bool(left["hit"]) and not np.allclose(
        left["normal_W"],
        right["normal_W"],
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise AssertionError(f"{label} differs in world-space normal")
    if not bool(left["hit"]) and not (
        np.all(np.isnan(left["normal_W"]))
        and np.all(np.isnan(right["normal_W"]))
    ):
        raise AssertionError(f"{label} differs in miss normal sentinel")


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)

    silicone_vertices = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float32,
    )

    silicone_origin = np.array((-0.0070, 0.00090, -0.030), dtype=np.float32)
    carrier_origin = np.array((0.0011, 0.00073, 0.010), dtype=np.float32)
    miss_origin = np.array((0.050, 0.0, -0.020), dtype=np.float32)
    positive_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    negative_z = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    origins = np.stack((silicone_origin, carrier_origin, miss_origin))
    directions = np.stack((positive_z, negative_z, positive_z))

    updated_scene = _make_scene(fingertip_mesh)
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
    if not np.isfinite(initial_silicone["t"]) or initial_silicone["t"] <= 0.0:
        raise AssertionError("initial silicone ray has invalid distance")
    initial_carrier = initial_results[1]
    if not bool(initial_carrier["hit"]):
        raise AssertionError("initial carrier ray missed")
    if int(initial_carrier["instance_id"]) != CARRIER_INSTANCE_ID:
        raise AssertionError("initial carrier ray hit the wrong instance")
    if not np.isfinite(initial_carrier["t"]) or initial_carrier["t"] <= 0.0:
        raise AssertionError("initial carrier ray has invalid distance")
    if bool(initial_results[2]["hit"]):
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

    for label, result_index in (("carrier", 1), ("miss", 2)):
        _assert_same_result(
            label,
            initial_results[result_index],
            updated_results[result_index],
        )

    fresh_build_start = perf_counter()
    fresh_scene = _make_scene(fingertip_mesh)
    fresh_scene.update_silicone(displaced_vertices)
    fresh_scene._stream.synchronize()
    fresh_build_time_ms = 1.0e3 * (perf_counter() - fresh_build_start)
    fresh_results = fresh_scene.trace_closest(
        origins,
        directions,
        mask=ALL_MASK,
    )

    for label, updated, fresh in zip(
        ("silicone", "carrier", "miss"),
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
    print("miss unchanged:      PASS")
    print("update vs rebuild:   PASS")
    print()
    print(f"update time:      {update_time_ms:.3f} ms")
    print(f"fresh build time: {fresh_build_time_ms:.3f} ms")
    print()
    print("OptiX silicone GAS/IAS refit: PASS")


if __name__ == "__main__":
    main()
