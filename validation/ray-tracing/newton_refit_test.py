"""Validate the explicit Newton-state to OptiX silicone handoff."""

from __future__ import annotations

from importlib.resources import as_file, files

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import FingertipMesh
from lumo.newton import Indenter
from lumo.ray_tracing import OptixScene
from lumo.simulation import LumoSimulation


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2
OBJECT_INSTANCE_ID = 3

SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
OBJECT_MASK = 0x04
ALL_MASK = SILICONE_MASK | CARRIER_MASK | OBJECT_MASK

_SIM_FREQUENCY_HZ = 1.0e3
_INDENTER_RADIUS_M = 7.5e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_MAX_SIM_TIME_S = 10.0
_TARGET_FORCE_N = 20.0
_MAX_BONDED_DRIFT_M = 1.0e-8
_MOTION_DIRECTION_W = wp.vec3(0.0, 0.0, 1.0)

_SPHERE_CENTER_M = np.array((0.030, 0.0, 0.0), dtype=np.float32)
_SPHERE_RADIUS_M = 0.005
_T_TOLERANCE_M = 2.0e-6
_BARYCENTRIC_TOLERANCE = 2.0e-5


def _make_scene(
    fingertip_mesh: FingertipMesh,
    *,
    silicone_vertices: np.ndarray | None = None,
) -> OptixScene:
    return OptixScene(
        fingertip_mesh,
        sphere_center=_SPHERE_CENTER_M,
        sphere_radius=_SPHERE_RADIUS_M,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        sphere_instance_id=OBJECT_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
        sphere_visibility_mask=OBJECT_MASK,
        silicone_vertices=silicone_vertices,
    )


def _assert_expected_hits(results: np.ndarray) -> None:
    for label, result, expected_instance_id in (
        ("silicone", results[0], SILICONE_INSTANCE_ID),
        ("carrier", results[1], CARRIER_INSTANCE_ID),
        ("sphere", results[2], OBJECT_INSTANCE_ID),
    ):
        if not bool(result["hit"]):
            raise AssertionError(f"{label} ray missed")
        if int(result["instance_id"]) != expected_instance_id:
            raise AssertionError(f"{label} ray hit the wrong instance")
        if not np.isfinite(result["t"]) or float(result["t"]) <= 0.0:
            raise AssertionError(f"{label} ray returned an invalid distance")
    if bool(results[3]["hit"]):
        raise AssertionError("miss ray unexpectedly hit the scene")


def _assert_same_results(updated: np.ndarray, fresh: np.ndarray) -> None:
    for label, updated_result, fresh_result in zip(
        ("silicone", "carrier", "sphere", "miss"),
        updated,
        fresh,
        strict=True,
    ):
        for field in ("hit", "instance_id", "primitive_id"):
            if updated_result[field] != fresh_result[field]:
                raise AssertionError(f"{label} differs in {field}")
        if not np.isclose(
            updated_result["t"],
            fresh_result["t"],
            rtol=0.0,
            atol=_T_TOLERANCE_M,
        ):
            raise AssertionError(f"{label} differs in hit distance")
        if not np.allclose(
            updated_result["barycentrics"],
            fresh_result["barycentrics"],
            rtol=0.0,
            atol=_BARYCENTRIC_TOLERANCE,
        ):
            raise AssertionError(f"{label} differs in barycentrics")


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    reference_simulation = LumoSimulation(
        fingertip,
        sim_frequency=_SIM_FREQUENCY_HZ,
    )
    reference_mesh = reference_simulation.fingertip_mesh
    reference_vertices = reference_simulation.silicone_vertices()
    mesh_vertices = np.asarray(
        reference_mesh.silicone.vertices,
        dtype=np.float32,
    )
    if not np.allclose(reference_vertices, mesh_vertices, rtol=0.0, atol=1.0e-7):
        raise AssertionError(
            "initial Newton silicone ordering differs from FingertipMesh"
        )
    initial_mapping_error_m = float(
        np.linalg.norm(reference_vertices - mesh_vertices, axis=1).max()
    )

    scene = _make_scene(reference_mesh)
    candidate_origins = np.array(
        [
            (x_m, y_m, -0.030)
            for x_m in (-0.001, 0.0, 0.001)
            for y_m in (-0.001, 0.0, 0.001)
        ],
        dtype=np.float32,
    )
    positive_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    candidate_results = scene.trace_closest(
        candidate_origins,
        np.tile(positive_z, (len(candidate_origins), 1)),
        mask=ALL_MASK,
    )
    silicone_origin: np.ndarray | None = None
    for origin, result in zip(
        candidate_origins,
        candidate_results,
        strict=True,
    ):
        barycentrics = np.asarray(result["barycentrics"])
        barycentric_weights = np.array(
            (
                1.0 - barycentrics.sum(),
                barycentrics[0],
                barycentrics[1],
            )
        )
        if (
            int(result["instance_id"]) == SILICONE_INSTANCE_ID
            and np.min(barycentric_weights) > 0.1
        ):
            silicone_origin = origin
            break
    if silicone_origin is None:
        raise AssertionError("no unambiguous central silicone ray was found")

    carrier_origin = np.array((0.0011, 0.00073, 0.010), dtype=np.float32)
    sphere_origin = np.array((0.030, 0.0, -0.020), dtype=np.float32)
    miss_origin = np.array((0.050, 0.0, -0.020), dtype=np.float32)
    negative_z = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    origins = np.stack(
        (silicone_origin, carrier_origin, sphere_origin, miss_origin)
    )
    directions = np.stack((positive_z, negative_z, positive_z, positive_z))
    reference_results = scene.trace_closest(
        origins,
        directions,
        mask=ALL_MASK,
    )
    _assert_expected_hits(reference_results)
    del reference_simulation

    initial_center_z_m = (
        fingertip.tip_z_m
        - _INITIAL_CLEARANCE_M
        - _INDENTER_RADIUS_M
    )
    initial_tf = wp.transform(
        wp.vec3(0.0, 0.0, initial_center_z_m),
        wp.quat_identity(),
    )

    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )

    with as_file(sphere_resource) as sphere_urdf_path:
        builder = newton.ModelBuilder(gravity=0.0)
        indenter = Indenter.add_urdf(
            builder,
            sphere_urdf_path,
            tf=initial_tf,
        )
        simulation = LumoSimulation(
            fingertip,
            builder=builder,
            sim_frequency=_SIM_FREQUENCY_HZ,
        )

        if simulation.soft_contact_count(indenter.body_index):
            raise AssertionError(
                "15 mm sphere has contacts before prescribed motion"
            )

        translation_step_m = (
            _APPROACH_SPEED_M_S / simulation.sim_frequency
        )
        max_step_count = int(
            _MAX_SIM_TIME_S * simulation.sim_frequency
        )
        reaction_force_n = 0.0
        for approach_step in range(1, max_step_count + 1):
            travel_m = approach_step * translation_step_m
            simulation.apply_indenter_pose(
                indenter,
                wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        initial_center_z_m + travel_m,
                    ),
                    wp.quat_identity(),
                ),
            )
            simulation.step()
            reaction_force_n = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=_MOTION_DIRECTION_W,
            )
            if reaction_force_n >= _TARGET_FORCE_N:
                break
        else:
            raise RuntimeError(
                "15 mm sphere did not reach the transient 20 N target within "
                f"{_MAX_SIM_TIME_S:g} s; last force was "
                f"{reaction_force_n:.9e} N"
            )

        sphere_contact_count = simulation.soft_contact_count(
            indenter.body_index
        )
        if sphere_contact_count == 0:
            raise AssertionError("20 N target was reached without sphere contact")

        final_vertices = simulation.silicone_vertices()
        initial_vertices = np.asarray(
            simulation.fingertip_mesh.silicone.vertices,
            dtype=np.float32,
        )
        if final_vertices.shape != initial_vertices.shape:
            raise AssertionError("silicone vertex count changed")
        if not np.all(np.isfinite(final_vertices)):
            raise AssertionError("final silicone vertices are not finite")
        if not np.allclose(
            initial_vertices,
            reference_vertices,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise AssertionError("indentation mesh vertex ordering changed")
        if not np.array_equal(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            reference_mesh.silicone.surface_tri_indices,
        ):
            raise AssertionError("indentation mesh topology changed")

        bonded_indices = simulation.fingertip_mesh.bonded_vertex_indices
        nonbonded = np.ones(len(final_vertices), dtype=bool)
        nonbonded[bonded_indices] = False
        displacement_m = np.linalg.norm(
            final_vertices - initial_vertices,
            axis=1,
        )
        maximum_nonbonded_displacement_m = float(
            displacement_m[nonbonded].max()
        )
        if maximum_nonbonded_displacement_m <= 0.0:
            raise AssertionError("nonbonded silicone did not move")
        maximum_bonded_drift_m = float(
            displacement_m[bonded_indices].max()
        )
        if maximum_bonded_drift_m > _MAX_BONDED_DRIFT_M:
            raise AssertionError("bonded silicone vertices drifted")

        scene.update_silicone(final_vertices)
        updated_results = scene.trace_closest(
            origins,
            directions,
            mask=ALL_MASK,
        )
        _assert_expected_hits(updated_results)
        fresh_scene = _make_scene(
            simulation.fingertip_mesh,
            silicone_vertices=final_vertices,
        )
        fresh_results = fresh_scene.trace_closest(
            origins,
            directions,
            mask=ALL_MASK,
        )
        _assert_expected_hits(fresh_results)
        _assert_same_results(updated_results, fresh_results)

        print(
            "initial mapping:     PASS | "
            f"max error={initial_mapping_error_m:.3e} m"
        )
        print(
            "transient state:     PASS | "
            f"F={reaction_force_n:.4f} N | "
            f"travel={1.0e3 * travel_m:.4f} mm | "
            f"contacts={sphere_contact_count}"
        )
        print(
            "silicone extraction: PASS | "
            f"vertices={len(final_vertices)} | "
            f"nonbonded move={maximum_nonbonded_displacement_m:.6e} m | "
            f"bond drift={maximum_bonded_drift_m:.3e} m"
        )
        print("OptiX refit:          PASS")
        print("updated vs rebuild:   PASS")

    print()
    print("Newton -> OptiX silicone checkpoint: PASS")


if __name__ == "__main__":
    main()
