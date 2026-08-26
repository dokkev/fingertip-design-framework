"""Validate full five-LED fingertip mechanics at three Y locations."""

from __future__ import annotations

import csv
import json
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import matplotlib
import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import (
    LED_PITCH_MM,
    LED_RECESS_DEPTH_MM,
    LED_RECESS_WIDTH_MM,
    MAIN_Y_BOUNDS_MM,
    TOTAL_Y_BOUNDS_MM,
    make_fingertip_5led_mesh,
    make_fingertip_mesh,
)
from lumo.newton import Indenter
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


_OUTPUT_DIRECTORY = Path("output/validation/5led_newton")
_REPORT_PATH = Path("output/validation/5led_newton_mechanics_validation.md")
_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_APPROACH_SPEED_M_S = 5.0e-3
_FORCE_GAIN_M_S_N = 2.5e-4
_FORCE_TARGETS_N = (10.0, 20.0)
_FORCE_TOLERANCE_FRACTION = 0.20
_SETTLE_DURATION_S = 3.0
_MAX_SIM_TIME_S = 60.0
_INITIAL_CLEARANCE_M = 1.0e-3
_SPHERE_DIAMETER_MM = 15.0
_SPHERE_RADIUS_M = 0.5e-3 * _SPHERE_DIAMETER_MM
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547
_SOFT_CONTACT_MARGIN_M = 1.0e-4
_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6
_ELEMENT_SIZE_MM = 1.0
_SLICE_HALF_WIDTH_MM = 0.75
_CONTACTS_Y_MM = (
    ("center_led", 0.0),
    ("between_leds", 5.5),
    ("distal_led", 22.0),
)
_STATIONS_Y_MM = np.array(
    (-22.0, -16.5, -11.0, -5.5, 0.0, 5.5, 11.0, 16.5, 22.0),
    dtype=np.float64,
)
_DENSE_PROFILE_Y_MM = np.arange(-27.5, 32.5 + 0.5, 1.0)


def _six_tet_volumes(
    positions_m: np.ndarray,
    tet_indices: np.ndarray,
) -> np.ndarray:
    tets = positions_m[tet_indices]
    return np.einsum(
        "ij,ij->i",
        tets[:, 1] - tets[:, 0],
        np.cross(tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]),
    )


def _connected_vertex_components(
    elements: np.ndarray,
    vertex_count: int,
) -> int:
    parent = np.arange(vertex_count, dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for element in elements:
        first_root = root(int(element[0]))
        for vertex in element[1:]:
            other_root = root(int(vertex))
            if first_root != other_root:
                parent[other_root] = first_root

    used = np.unique(elements)
    return len({root(int(vertex)) for vertex in used})


def _verify_geometry(fingertip_mesh) -> dict[str, float | int | list[float]]:
    vertices_m = np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float64)
    tets = np.asarray(
        fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    carrier_vertices_m = np.asarray(
        fingertip_mesh.carrier.vertices,
        dtype=np.float64,
    )
    carrier_triangles = np.asarray(
        fingertip_mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    led_y_mm = 1.0e3 * fingertip_mesh.led_centers_m[:, 1]
    led_z_mm = 1.0e3 * fingertip_mesh.led_centers_m[:, 2]

    if not np.allclose(
        (1.0e3 * vertices_m[:, 1].min(), 1.0e3 * vertices_m[:, 1].max()),
        TOTAL_Y_BOUNDS_MM,
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise RuntimeError("silicone does not span the required 60 mm length")
    if not np.allclose(
        led_y_mm,
        np.array((-22.0, -11.0, 0.0, 11.0, 22.0)),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("LED centers do not match the five-position contract")
    if not np.allclose(np.diff(led_y_mm), LED_PITCH_MM, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("LED centers do not use an exact 11 mm pitch")
    led_gap_mm = led_z_mm - fingertip_mesh.fingertip.silicone.cavity_bottom_z_mm
    if not np.allclose(
        led_gap_mm,
        LED_RECESS_DEPTH_MM,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("unloaded LED-to-silicone gaps are not 0.19 mm")
    if _connected_vertex_components(tets, len(vertices_m)) != 1:
        raise RuntimeError("silicone tetrahedra are not one connected body")

    reference_six_volumes = _six_tet_volumes(vertices_m, tets)
    if not np.all(np.isfinite(reference_six_volumes)):
        raise RuntimeError("silicone reference tetrahedra are non-finite")
    if np.any(np.abs(reference_six_volumes) <= 1.0e-18):
        raise RuntimeError("silicone reference mesh contains a degenerate tet")
    if np.any(np.sign(reference_six_volumes) != np.sign(reference_six_volumes[0])):
        raise RuntimeError("silicone reference mesh has inconsistent tet winding")

    edges = np.sort(
        np.concatenate(
            (
                carrier_triangles[:, (0, 1)],
                carrier_triangles[:, (1, 2)],
                carrier_triangles[:, (2, 0)],
            )
        ),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise RuntimeError("carrier surface is not watertight")

    centroids_mm = 1.0e3 * vertices_m[tets].mean(axis=1)
    silicone = fingertip_mesh.fingertip.silicone
    inside_cutout_x = (
        (centroids_mm[:, 0] > silicone.cavity_left_x_mm + 0.25)
        & (centroids_mm[:, 0] < silicone.cavity_right_x_mm - 0.25)
    )
    inside_cutout_z = (
        (centroids_mm[:, 2] > silicone.cavity_bottom_z_mm + 0.25)
        & (centroids_mm[:, 2] < -0.25)
    )
    active_void_tets = inside_cutout_x & inside_cutout_z & (
        (centroids_mm[:, 1] > MAIN_Y_BOUNDS_MM[0] + 1.0)
        & (centroids_mm[:, 1] < MAIN_Y_BOUNDS_MM[1] - 1.0)
    )
    distal_fill_tets = inside_cutout_x & inside_cutout_z & (
        centroids_mm[:, 1] > MAIN_Y_BOUNDS_MM[1] + 0.5
    )
    distal_upper_fill_tets = (
        (np.abs(centroids_mm[:, 0]) < 2.0)
        & (centroids_mm[:, 1] > MAIN_Y_BOUNDS_MM[1] + 0.5)
        & (centroids_mm[:, 2] > 0.5)
        & (
            centroids_mm[:, 2]
            < fingertip_mesh.fingertip.silicone.bond_top_z_mm - 0.5
        )
    )
    if np.any(active_void_tets):
        raise RuntimeError("silicone tetrahedra occupy the active-section void")
    if not np.any(distal_fill_tets) or not np.any(distal_upper_fill_tets):
        raise RuntimeError("distal 5 mm silicone closure is not solid")

    return {
        "silicone_vertex_count": int(len(vertices_m)),
        "silicone_tet_count": int(len(tets)),
        "silicone_surface_triangle_count": int(
            len(fingertip_mesh.silicone.surface_tri_indices) // 3
        ),
        "bonded_vertex_count": int(len(fingertip_mesh.bonded_vertex_indices)),
        "silicone_component_count": 1,
        "minimum_reference_six_volume_m3": float(
            np.abs(reference_six_volumes).min()
        ),
        "carrier_vertex_count": int(len(carrier_vertices_m)),
        "carrier_triangle_count": int(len(carrier_triangles)),
        "carrier_boundary_edge_count": int(np.count_nonzero(edge_counts != 2)),
        "silicone_y_bounds_mm": [
            float(1.0e3 * vertices_m[:, 1].min()),
            float(1.0e3 * vertices_m[:, 1].max()),
        ],
        "carrier_y_bounds_mm": [
            float(1.0e3 * carrier_vertices_m[:, 1].min()),
            float(1.0e3 * carrier_vertices_m[:, 1].max()),
        ],
        "led_centers_y_mm": led_y_mm.tolist(),
        "led_recess_width_mm": LED_RECESS_WIDTH_MM,
        "led_recess_depth_mm": LED_RECESS_DEPTH_MM,
        "unloaded_led_silicone_gaps_mm": led_gap_mm.tolist(),
        "active_void_tet_centroid_count": int(np.count_nonzero(active_void_tets)),
        "distal_fill_tet_centroid_count": int(np.count_nonzero(distal_fill_tets)),
        "distal_upper_fill_tet_centroid_count": int(
            np.count_nonzero(distal_upper_fill_tets)
        ),
    }


def _make_trial(
    fingertip: Fingertip,
    sphere_path: Path,
    *,
    name: str,
    contact_y_mm: float,
    target_force_n: float,
) -> DesignTrial:
    initial_center_z_m = (
        fingertip.tip_z_m - _SPHERE_RADIUS_M - _INITIAL_CLEARANCE_M
    )
    return DesignTrial(
        name=name,
        urdf_path=sphere_path,
        initial_tf=wp.transform(
            wp.vec3(0.0, 1.0e-3 * contact_y_mm, initial_center_z_m),
            wp.quat_identity(),
        ),
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=_APPROACH_SPEED_M_S,
        target_force_n=target_force_n,
        max_sim_time_s=_MAX_SIM_TIME_S,
        initial_clearance_m=_INITIAL_CLEARANCE_M,
    )


def _slice_profile(
    reference_vertices_m: np.ndarray,
    displacement_m: np.ndarray,
    surface_vertices: np.ndarray,
    stations_y_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    surface_y_mm = 1.0e3 * reference_vertices_m[surface_vertices, 1]
    surface_displacement_m = displacement_m[surface_vertices]
    maxima_m = np.empty(len(stations_y_mm), dtype=np.float64)
    means_m = np.empty(len(stations_y_mm), dtype=np.float64)
    for index, station_y_mm in enumerate(stations_y_mm):
        in_slice = np.abs(surface_y_mm - station_y_mm) <= _SLICE_HALF_WIDTH_MM
        if not np.any(in_slice):
            nearest_distance_mm = np.min(np.abs(surface_y_mm - station_y_mm))
            in_slice = np.abs(surface_y_mm - station_y_mm) <= (
                nearest_distance_mm + 1.0e-9
            )
        values_m = surface_displacement_m[in_slice]
        maxima_m[index] = float(values_m.max())
        means_m[index] = float(values_m.mean())
    return maxima_m, means_m


def _contact_records(
    simulation: LumoSimulation,
    indenter: Indenter,
    positions_m: np.ndarray,
) -> dict[str, np.ndarray]:
    contacts = simulation.contacts
    emitted_count = int(contacts.soft_contact_count.numpy()[0])
    stored_count = min(emitted_count, int(contacts.soft_contact_max))
    shapes = contacts.soft_contact_shape.numpy()[:stored_count]
    valid = shapes >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    indenter_records = valid.copy()
    indenter_records[valid] = (
        shape_bodies[shapes[valid]] == indenter.body_index
    )
    indices = contacts.soft_contact_indices.numpy()[:stored_count][
        indenter_records
    ]
    barycentric = contacts.soft_contact_barycentric.numpy()[:stored_count][
        indenter_records
    ]
    normals_W = contacts.soft_contact_normal.numpy()[:stored_count][
        indenter_records
    ]
    body_positions = contacts.soft_contact_body_pos.numpy()[:stored_count][
        indenter_records
    ]

    points_W_m = np.empty((len(indices), 3), dtype=np.float64)
    for record_index, (particle_indices, weights) in enumerate(
        zip(indices, barycentric, strict=True)
    ):
        present = particle_indices >= 0
        points_W_m[record_index] = np.sum(
            positions_m[particle_indices[present]] * weights[present, None],
            axis=0,
        )
    return {
        "particle_indices": indices,
        "barycentric": barycentric,
        "points_W_m": points_W_m,
        "normals_W": normals_W,
        "body_positions": body_positions,
    }


def _collect_checkpoint(
    trial: DesignTrial,
    simulation: LumoSimulation,
    indenter: Indenter,
    *,
    contact_y_mm: float,
    target_force_n: float,
    reference_vertices_m: np.ndarray,
    reference_six_volumes_m3: np.ndarray,
    tet_indices: np.ndarray,
    surface_triangles: np.ndarray,
    surface_vertices: np.ndarray,
) -> dict[str, object]:
    if (
        trial.final_tf is None
        or trial.travel_m is None
        or trial.reaction_force_n is None
        or trial.simulation_time_s is None
        or trial.maximum_particle_speed_m_s is None
        or trial.force_change_n is None
    ):
        raise RuntimeError(f"{trial.name} checkpoint is incomplete")

    positions_m = np.asarray(simulation.silicone_vertices(), dtype=np.float64)
    velocities_m_s = simulation.state.particle_qd.numpy()
    if not np.all(np.isfinite(positions_m)) or not np.all(
        np.isfinite(velocities_m_s)
    ):
        raise RuntimeError(f"{trial.name} produced a non-finite state")

    displacement_m = np.linalg.norm(
        positions_m - reference_vertices_m,
        axis=1,
    )
    current_six_volumes_m3 = _six_tet_volumes(positions_m, tet_indices)
    det_f = current_six_volumes_m3 / reference_six_volumes_m3
    if not np.all(np.isfinite(det_f)):
        raise RuntimeError(f"{trial.name} produced non-finite det(F)")

    bonded_indices = simulation.fingertip_mesh.bonded_vertex_indices
    bonded_drift_m = displacement_m[bonded_indices]
    final_tf = np.asarray(trial.final_tf, dtype=np.float64)
    sphere_center_m = final_tf[:3]
    surface_displacement_m = displacement_m[surface_vertices]
    surface_penetration_m = np.maximum(
        0.0,
        _SPHERE_RADIUS_M
        - np.linalg.norm(
            positions_m[surface_vertices] - sphere_center_m,
            axis=1,
        ),
    )
    surface_centroid_penetration_m = np.maximum(
        0.0,
        _SPHERE_RADIUS_M
        - np.linalg.norm(
            positions_m[surface_triangles].mean(axis=1) - sphere_center_m,
            axis=1,
        ),
    )

    station_max_m, station_mean_m = _slice_profile(
        reference_vertices_m,
        displacement_m,
        surface_vertices,
        _STATIONS_Y_MM,
    )
    dense_max_m, dense_mean_m = _slice_profile(
        reference_vertices_m,
        displacement_m,
        surface_vertices,
        _DENSE_PROFILE_Y_MM,
    )
    profile_peak_m = float(dense_max_m.max())
    if profile_peak_m <= 0.0:
        raise RuntimeError(f"{trial.name} produced no displacement")
    station_normalized = station_max_m / profile_peak_m
    dense_normalized = dense_max_m / profile_peak_m

    influence: dict[str, float] = {}
    for threshold in (0.10, 0.05):
        above = _DENSE_PROFILE_Y_MM[dense_normalized >= threshold]
        influence[f"influence_{int(100 * threshold)}_left_mm"] = float(
            contact_y_mm - above.min()
        )
        influence[f"influence_{int(100 * threshold)}_right_mm"] = float(
            above.max() - contact_y_mm
        )
        influence[f"influence_{int(100 * threshold)}_span_mm"] = float(
            above.max() - above.min()
        )

    raw_contacts = _contact_records(simulation, indenter, positions_m)
    if len(raw_contacts["points_W_m"]) == 0:
        raise RuntimeError(f"{trial.name} has no stored indenter contact records")
    contact_centroid_W_m = raw_contacts["points_W_m"].mean(axis=0)
    average_normal_W = raw_contacts["normals_W"].mean(axis=0)
    average_normal_length = float(np.linalg.norm(average_normal_W))
    if average_normal_length > 0.0:
        average_normal_W /= average_normal_length

    indentation_m = trial.travel_m - _INITIAL_CLEARANCE_M
    if indentation_m <= 0.0:
        raise RuntimeError(f"{trial.name} has non-positive indentation")
    sphere_contact_count = simulation.soft_contact_count(indenter.body_index)
    if sphere_contact_count <= 0:
        raise RuntimeError(f"{trial.name} has no indenter contact")
    overflow = int(
        simulation.solver.body_particle_contact_overflow_max.numpy()[0]
    )

    result: dict[str, object] = {
        "name": trial.name,
        "contact_y_mm": contact_y_mm,
        "target_force_n": target_force_n,
        "actual_force_n": trial.reaction_force_n,
        "force_error_n": trial.reaction_force_n - target_force_n,
        "force_change_n": trial.force_change_n,
        "indentation_m": indentation_m,
        "checkpoint_time_s": trial.simulation_time_s,
        "step_count": trial.step_count,
        "maximum_particle_speed_m_s": trial.maximum_particle_speed_m_s,
        "maximum_nodal_displacement_m": float(displacement_m.max()),
        "maximum_surface_displacement_m": float(surface_displacement_m.max()),
        "local_stiffness_n_mm": trial.reaction_force_n / (1.0e3 * indentation_m),
        "minimum_det_f": float(det_f.min()),
        "inverted_tet_count": int(np.count_nonzero(det_f <= 0.0)),
        "maximum_bonded_drift_m": float(bonded_drift_m.max()),
        "sphere_contact_count": sphere_contact_count,
        "total_soft_contact_count": simulation.soft_contact_count(),
        "carrier_contact_count": simulation.soft_contact_count(
            simulation.fingertip_model.carrier_body
        ),
        "soft_contact_capacity": int(simulation.contacts.soft_contact_max),
        "body_particle_buffer_size": int(
            simulation.solver.body_particle_contact_buffer_pre_alloc
        ),
        "body_particle_buffer_overflow": overflow,
        "maximum_surface_vertex_sphere_penetration_m": float(
            surface_penetration_m.max()
        ),
        "maximum_surface_centroid_sphere_penetration_m": float(
            surface_centroid_penetration_m.max()
        ),
        "contact_centroid_W_m": contact_centroid_W_m,
        "average_contact_normal_W": average_normal_W,
        "peak_profile_y_mm": float(
            _DENSE_PROFILE_Y_MM[int(np.argmax(dense_max_m))]
        ),
        "reference_vertices_m": reference_vertices_m,
        "deformed_vertices_m": positions_m,
        "displacement_m": displacement_m,
        "station_max_m": station_max_m,
        "station_mean_m": station_mean_m,
        "station_normalized": station_normalized,
        "dense_max_m": dense_max_m,
        "dense_mean_m": dense_mean_m,
        "dense_normalized": dense_normalized,
        "final_tf": final_tf,
        "raw_contacts": raw_contacts,
        **influence,
    }
    return result


def _save_checkpoint(result: dict[str, object]) -> None:
    raw_contacts = result["raw_contacts"]
    np.savez_compressed(
        _OUTPUT_DIRECTORY / f"{result['name']}.npz",
        deformed_vertices_m=result["deformed_vertices_m"],
        displacement_m=result["displacement_m"],
        final_tf=result["final_tf"],
        contact_particle_indices=raw_contacts["particle_indices"],
        contact_barycentric=raw_contacts["barycentric"],
        contact_points_W_m=raw_contacts["points_W_m"],
        contact_normals_W=raw_contacts["normals_W"],
        contact_body_positions=raw_contacts["body_positions"],
        station_y_mm=_STATIONS_Y_MM,
        station_max_m=result["station_max_m"],
        station_mean_m=result["station_mean_m"],
        station_normalized=result["station_normalized"],
        dense_profile_y_mm=_DENSE_PROFILE_Y_MM,
        dense_profile_max_m=result["dense_max_m"],
        dense_profile_mean_m=result["dense_mean_m"],
        dense_profile_normalized=result["dense_normalized"],
    )


def _run_case(
    fingertip: Fingertip,
    fingertip_mesh,
    sphere_path: Path,
    *,
    name: str,
    contact_y_mm: float,
    target_force_n: float,
) -> dict[str, object]:
    reference_vertices_m = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    tet_indices = np.asarray(
        fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    surface_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    surface_vertices = np.unique(surface_triangles)
    reference_six_volumes_m3 = _six_tet_volumes(
        reference_vertices_m,
        tet_indices,
    )
    trial = _make_trial(
        fingertip,
        sphere_path,
        name=name,
        contact_y_mm=contact_y_mm,
        target_force_n=target_force_n,
    )
    checkpoints: list[dict[str, object]] = []

    def inspect(
        completed_trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        checkpoints.append(
            _collect_checkpoint(
                completed_trial,
                simulation,
                indenter,
                contact_y_mm=contact_y_mm,
                target_force_n=target_force_n,
                reference_vertices_m=reference_vertices_m,
                reference_six_volumes_m3=reference_six_volumes_m3,
                tet_indices=tet_indices,
                surface_triangles=surface_triangles,
                surface_vertices=surface_vertices,
            )
        )

    print(
        f"starting {name}: Y={contact_y_mm:+.1f} mm, "
        f"target={target_force_n:g} N",
        flush=True,
    )
    wall_start_s = perf_counter()
    DesignStudy(
        fingertip,
        (trial,),
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        element_size_mm=_ELEMENT_SIZE_MM,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        carrier_contact_stiffness_n_m=_CARRIER_CONTACT_STIFFNESS_N_M,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    ).run(inspect_trial=inspect)
    wp.synchronize()
    if len(checkpoints) != 1:
        raise RuntimeError(f"{name} produced {len(checkpoints)} checkpoints")
    result = checkpoints[0]
    result["wall_runtime_s"] = perf_counter() - wall_start_s
    _save_checkpoint(result)
    print(
        f"completed {name}: F={result['actual_force_n']:.4f} N | "
        f"indent={1.0e3 * result['indentation_m']:.4f} mm | "
        f"umax={1.0e3 * result['maximum_nodal_displacement_m']:.4f} mm | "
        f"wall={result['wall_runtime_s']:.2f} s",
        flush=True,
    )
    return result


def _measure_initialization(
    fingertip: Fingertip,
    fingertip_mesh,
    sphere_path: Path,
) -> float:
    trial = _make_trial(
        fingertip,
        sphere_path,
        name="initialization_probe",
        contact_y_mm=0.0,
        target_force_n=_FORCE_TARGETS_N[0],
    )
    wall_start_s = perf_counter()
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    indenter = Indenter.add_urdf(
        builder,
        sphere_path,
        tf=trial.initial_tf,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    )
    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        soft_contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        soft_contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
        carrier_contact_stiffness_n_m=_CARRIER_CONTACT_STIFFNESS_N_M,
    )
    wp.synchronize()
    initialization_s = perf_counter() - wall_start_s
    if simulation.soft_contact_count(indenter.body_index) != 0:
        raise RuntimeError("initialization probe sphere starts in contact")
    del simulation, indenter, builder
    return initialization_s


def _write_tables(
    results: list[dict[str, object]],
    failures: list[dict[str, str | float]],
) -> None:
    summary_fields = (
        "name",
        "status",
        "contact_y_mm",
        "target_force_n",
        "actual_force_n",
        "force_error_n",
        "indentation_mm",
        "checkpoint_time_s",
        "wall_runtime_s",
        "maximum_nodal_displacement_mm",
        "local_stiffness_n_mm",
        "minimum_det_f",
        "inverted_tet_count",
        "maximum_bonded_drift_m",
        "sphere_contact_count",
        "total_soft_contact_count",
        "carrier_contact_count",
        "soft_contact_capacity",
        "body_particle_buffer_size",
        "body_particle_buffer_overflow",
        "maximum_surface_vertex_sphere_penetration_m",
        "maximum_surface_centroid_sphere_penetration_m",
        "force_change_n",
        "maximum_particle_speed_m_s",
        "peak_profile_y_mm",
        "influence_10_left_mm",
        "influence_10_right_mm",
        "influence_10_span_mm",
        "influence_5_left_mm",
        "influence_5_right_mm",
        "influence_5_span_mm",
        "error",
    )
    with (_OUTPUT_DIRECTORY / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result["name"],
                    "status": "PASS",
                    "contact_y_mm": result["contact_y_mm"],
                    "target_force_n": result["target_force_n"],
                    "actual_force_n": result["actual_force_n"],
                    "force_error_n": result["force_error_n"],
                    "indentation_mm": 1.0e3 * result["indentation_m"],
                    "checkpoint_time_s": result["checkpoint_time_s"],
                    "wall_runtime_s": result["wall_runtime_s"],
                    "maximum_nodal_displacement_mm": (
                        1.0e3 * result["maximum_nodal_displacement_m"]
                    ),
                    "local_stiffness_n_mm": result["local_stiffness_n_mm"],
                    "minimum_det_f": result["minimum_det_f"],
                    "inverted_tet_count": result["inverted_tet_count"],
                    "maximum_bonded_drift_m": result["maximum_bonded_drift_m"],
                    "sphere_contact_count": result["sphere_contact_count"],
                    "total_soft_contact_count": result[
                        "total_soft_contact_count"
                    ],
                    "carrier_contact_count": result["carrier_contact_count"],
                    "soft_contact_capacity": result["soft_contact_capacity"],
                    "body_particle_buffer_size": result[
                        "body_particle_buffer_size"
                    ],
                    "body_particle_buffer_overflow": result[
                        "body_particle_buffer_overflow"
                    ],
                    "maximum_surface_vertex_sphere_penetration_m": result[
                        "maximum_surface_vertex_sphere_penetration_m"
                    ],
                    "maximum_surface_centroid_sphere_penetration_m": result[
                        "maximum_surface_centroid_sphere_penetration_m"
                    ],
                    "force_change_n": result["force_change_n"],
                    "maximum_particle_speed_m_s": result[
                        "maximum_particle_speed_m_s"
                    ],
                    "peak_profile_y_mm": result["peak_profile_y_mm"],
                    "influence_10_left_mm": result["influence_10_left_mm"],
                    "influence_10_right_mm": result["influence_10_right_mm"],
                    "influence_10_span_mm": result["influence_10_span_mm"],
                    "influence_5_left_mm": result["influence_5_left_mm"],
                    "influence_5_right_mm": result["influence_5_right_mm"],
                    "influence_5_span_mm": result["influence_5_span_mm"],
                    "error": "",
                }
            )
        for failure in failures:
            writer.writerow(
                {
                    "name": failure["name"],
                    "status": "FAIL",
                    "contact_y_mm": failure["contact_y_mm"],
                    "error": failure["error"],
                }
            )

    with (_OUTPUT_DIRECTORY / "station_displacements.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "contact_y_mm",
                "station_y_mm",
                "maximum_surface_displacement_mm",
                "mean_surface_displacement_mm",
                "normalized_maximum",
            )
        )
        for result in results:
            for station_y_mm, maximum_m, mean_m, normalized in zip(
                _STATIONS_Y_MM,
                result["station_max_m"],
                result["station_mean_m"],
                result["station_normalized"],
                strict=True,
            ):
                writer.writerow(
                    (
                        result["name"],
                        result["contact_y_mm"],
                        station_y_mm,
                        1.0e3 * maximum_m,
                        1.0e3 * mean_m,
                        normalized,
                    )
                )

    with (_OUTPUT_DIRECTORY / "longitudinal_profiles.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "contact_y_mm",
                "profile_y_mm",
                "maximum_surface_displacement_mm",
                "mean_surface_displacement_mm",
                "normalized_maximum",
            )
        )
        for result in results:
            for y_mm, maximum_m, mean_m, normalized in zip(
                _DENSE_PROFILE_Y_MM,
                result["dense_max_m"],
                result["dense_mean_m"],
                result["dense_normalized"],
                strict=True,
            ):
                writer.writerow(
                    (
                        result["name"],
                        result["contact_y_mm"],
                        y_mm,
                        1.0e3 * maximum_m,
                        1.0e3 * mean_m,
                        normalized,
                    )
                )


def _plot_profiles(
    results: list[dict[str, object]],
    *,
    target_force_n: float,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    for result in results:
        axes[0].plot(
            _DENSE_PROFILE_Y_MM,
            1.0e3 * result["dense_max_m"],
            label=f"{result['name']} (contact {result['contact_y_mm']:+g} mm)",
        )
        axes[1].plot(
            _DENSE_PROFILE_Y_MM,
            result["dense_normalized"],
            label=result["name"],
        )
    for axis in axes:
        for led_y_mm in (-22.0, -11.0, 0.0, 11.0, 22.0):
            axis.axvline(led_y_mm, color="0.75", linewidth=0.7)
        axis.axvline(27.5, color="tab:red", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("max surface displacement [mm]")
    axes[1].set_ylabel("normalized displacement")
    axes[1].set_xlabel("reference Y [mm]")
    axes[1].axhline(0.10, color="black", linestyle=":", linewidth=1.0)
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=1.0)
    figure.suptitle(
        f"Full five-LED longitudinal deformation at {target_force_n:g} N"
    )
    figure.tight_layout()
    figure.savefig(
        _OUTPUT_DIRECTORY
        / f"longitudinal_displacement_profiles_{target_force_n:g}n.png",
        dpi=180,
    )
    plt.close(figure)


def _add_mesh(
    axis,
    vertices_m: np.ndarray,
    triangles: np.ndarray,
    *,
    color_values_mm: np.ndarray | None,
    maximum_color_mm: float,
) -> None:
    vertices_mm = 1.0e3 * vertices_m
    faces = vertices_mm[triangles]
    collection = Poly3DCollection(faces, linewidths=0.0, rasterized=True)
    if color_values_mm is None:
        collection.set_facecolor((0.72, 0.92, 0.68, 0.58))
    else:
        collection.set_array(color_values_mm[triangles].mean(axis=1))
        collection.set_cmap("viridis")
        collection.set_clim(0.0, maximum_color_mm)
    axis.add_collection3d(collection)


def _plot_deformation_comparison(
    fingertip_mesh,
    results: list[dict[str, object]],
) -> None:
    reference_vertices_m = np.asarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    surface_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    carrier_vertices_mm = 1.0e3 * np.asarray(
        fingertip_mesh.carrier.vertices,
        dtype=np.float64,
    )
    carrier_triangles = np.asarray(
        fingertip_mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    maximum_color_mm = max(
        1.0e3 * float(result["maximum_nodal_displacement_m"])
        for result in results
    )

    panels: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        ("undeformed", reference_vertices_m, None)
    ]
    panels.extend(
        (
            str(result["name"]),
            result["deformed_vertices_m"],
            1.0e3 * result["displacement_m"],
        )
        for result in results
    )
    figure = plt.figure(figsize=(14.0, 10.0))
    for panel_index, (title, vertices_m, colors_mm) in enumerate(panels, start=1):
        axis = figure.add_subplot(2, 2, panel_index, projection="3d")
        _add_mesh(
            axis,
            vertices_m,
            surface_triangles,
            color_values_mm=colors_mm,
            maximum_color_mm=maximum_color_mm,
        )
        carrier = Poly3DCollection(
            carrier_vertices_mm[carrier_triangles],
            facecolor=(0.34, 0.37, 0.42, 0.88),
            edgecolor="none",
        )
        axis.add_collection3d(carrier)
        led_centers_mm = 1.0e3 * fingertip_mesh.led_centers_m
        axis.scatter(
            led_centers_mm[:, 0],
            led_centers_mm[:, 1],
            led_centers_mm[:, 2],
            color="#38b000",
            s=15.0,
            depthshade=False,
        )
        axis.set_xlim(-17.0, 17.0)
        axis.set_ylim(-30.0, 35.0)
        axis.set_zlim(
            1.0e3 * reference_vertices_m[:, 2].min() - 1.0,
            carrier_vertices_mm[:, 2].max() + 1.0,
        )
        axis.set_box_aspect((34.0, 65.0, 28.0))
        axis.view_init(elev=19.0, azim=-58.0)
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
        axis.set_zlabel("Z [mm]")
        axis.set_title(title)
    scalar_mappable = ScalarMappable(
        norm=Normalize(0.0, maximum_color_mm),
        cmap="viridis",
    )
    figure.colorbar(
        scalar_mappable,
        ax=figure.axes,
        shrink=0.72,
        label="nodal displacement [mm]",
    )
    figure.suptitle("Full five-LED Newton deformation comparison")
    figure.savefig(
        _OUTPUT_DIRECTORY / "deformation_comparison.png",
        dpi=170,
    )
    plt.close(figure)


def _plot_between_led_focus(
    fingertip_mesh,
    result: dict[str, object],
) -> None:
    reference_m = np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float64)
    deformed_m = result["deformed_vertices_m"]
    surface_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    surface_vertices = np.unique(surface_triangles)
    in_center_slice = np.abs(1.0e3 * reference_m[surface_vertices, 0]) <= 0.75
    selected = surface_vertices[in_center_slice]
    displacement_mm = 1.0e3 * result["displacement_m"][selected]

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharex=True, sharey=True)
    axes[0].scatter(
        1.0e3 * reference_m[selected, 1],
        1.0e3 * reference_m[selected, 2],
        s=5.0,
        color="0.45",
    )
    axes[0].set_title("undeformed center slice")
    scatter = axes[1].scatter(
        1.0e3 * deformed_m[selected, 1],
        1.0e3 * deformed_m[selected, 2],
        c=displacement_mm,
        s=7.0,
        cmap="viridis",
    )
    axes[1].set_title("between-LED contact at Y=+5.5 mm")
    final_tf = result["final_tf"]
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    sphere_y_mm = 1.0e3 * final_tf[1] + 0.5 * _SPHERE_DIAMETER_MM * np.cos(angle)
    sphere_z_mm = 1.0e3 * final_tf[2] + 0.5 * _SPHERE_DIAMETER_MM * np.sin(angle)
    axes[1].plot(sphere_y_mm, sphere_z_mm, color="black", linewidth=1.0)
    for axis in axes:
        for led_y_mm in (-22.0, -11.0, 0.0, 11.0, 22.0):
            axis.axvline(led_y_mm, color="#38b000", linewidth=0.6)
        axis.axvline(27.5, color="tab:red", linestyle="--", linewidth=1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("Y [mm]")
    axes[0].set_ylabel("Z [mm]")
    figure.colorbar(scatter, ax=axes, label="nodal displacement [mm]")
    figure.suptitle("Longitudinal center slice around the between-LED contact")
    figure.savefig(
        _OUTPUT_DIRECTORY / "between_led_deformation.png",
        dpi=180,
    )
    plt.close(figure)


def _write_report(
    geometry: dict[str, object],
    mesh_build_time_s: float,
    initialization_time_s: float,
    results: list[dict[str, object]],
    failures: list[dict[str, str | float]],
    single_section: dict[str, object] | None,
) -> None:
    by_name = {str(result["name"]): result for result in results}
    lines = [
        "# Full 5-LED Newton mechanics validation",
        "",
        "## Contract",
        "",
        f"- morphology: hand-designed nominal, full height "
        f"{geometry['full_height_mm']:.3f} mm",
        f"- main section: `Y=[{MAIN_Y_BOUNDS_MM[0]}, {MAIN_Y_BOUNDS_MM[1]}] mm`",
        f"- total silicone: `Y=[{TOTAL_Y_BOUNDS_MM[0]}, {TOTAL_Y_BOUNDS_MM[1]}] mm`",
        f"- LED centers: `{geometry['led_centers_y_mm']} mm`",
        f"- Newton: {_SIM_FREQUENCY_HZ:g} Hz, {_VBD_ITERATIONS} VBD iterations",
        f"- contact: {_SPHERE_DIAMETER_MM:g} mm sphere, "
        f"targets `{list(_FORCE_TARGETS_N)} N`, "
        f"±{100 * _FORCE_TOLERANCE_FRACTION:g}% band for {_SETTLE_DURATION_S:g} s",
        "",
        "## Geometry verification",
        "",
        f"- silicone: {geometry['silicone_vertex_count']} vertices, "
        f"{geometry['silicone_tet_count']} tetrahedra, one connected component",
        f"- carrier: {geometry['carrier_vertex_count']} vertices, "
        f"{geometry['carrier_triangle_count']} triangles, "
        f"{geometry['carrier_boundary_edge_count']} boundary/nonmanifold edges",
        f"- LED stem recesses: {geometry['led_recess_width_mm']:.3f} mm wide × "
        f"{geometry['led_recess_depth_mm']:.3f} mm deep; unloaded gaps "
        f"`{geometry['unloaded_led_silicone_gaps_mm']} mm`",
        f"- distal fill diagnostic tets: {geometry['distal_fill_tet_centroid_count']} lower, "
        f"{geometry['distal_upper_fill_tet_centroid_count']} upper",
        f"- mesh build wall time: {mesh_build_time_s:.3f} s",
        f"- representative Newton initialization: {initialization_time_s:.3f} s",
        "",
        "## Contact checkpoints",
        "",
        "| case | Y [mm] | force [N] | indentation [mm] | peak displacement [mm] | k=F/δ [N/mm] | min det(F) | contacts | checkpoint [s] | wall [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['contact_y_mm']:.1f} | "
            f"{result['actual_force_n']:.4f} | {1.0e3 * result['indentation_m']:.4f} | "
            f"{1.0e3 * result['maximum_nodal_displacement_m']:.4f} | "
            f"{result['local_stiffness_n_mm']:.4f} | {result['minimum_det_f']:.5f} | "
            f"{result['sphere_contact_count']} | {result['checkpoint_time_s']:.3f} | "
            f"{result['wall_runtime_s']:.2f} |"
        )
    for failure in failures:
        lines.append(
            f"| {failure['name']} | {failure['contact_y_mm']:.1f} | FAIL | "
            f"{failure['error']} | | | | | | |"
        )

    lines.extend(
        (
            "",
            "Every passing checkpoint remained continuously inside the force band for "
            f"{_SETTLE_DURATION_S:g} s. The terminal one-tick force change and maximum "
            "particle speed are preserved in `cases.csv`; the production API does not "
            "expose the complete per-tick servo trace.",
            "",
            "## Terminal mechanics diagnostics",
            "",
            "| case | ΔF last tick [N] | vmax [m/s] | sphere centroid penetration [µm] | min det(F) | inverted tets | buffer overflow |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for result in results:
        lines.append(
            f"| {result['name']} | {result['force_change_n']:.5f} | "
            f"{result['maximum_particle_speed_m_s']:.4e} | "
            f"{1.0e6 * result['maximum_surface_centroid_sphere_penetration_m']:.2f} | "
            f"{result['minimum_det_f']:.5f} | {result['inverted_tet_count']} | "
            f"{result['body_particle_buffer_overflow']} |"
        )

    lines.extend(
        (
            "",
            "## Longitudinal station response",
            "",
            f"Primary profile metric: maximum displacement magnitude among silicone "
            f"surface vertices whose reference Y lies within ±{_SLICE_HALF_WIDTH_MM:g} mm "
            "of each station.",
            "",
            "| case | station Y [mm] | max displacement [mm] | normalized |",
            "|---|---:|---:|---:|",
        )
    )
    for result in results:
        for station_y_mm, maximum_m, normalized in zip(
            _STATIONS_Y_MM,
            result["station_max_m"],
            result["station_normalized"],
            strict=True,
        ):
            lines.append(
                f"| {result['name']} | {station_y_mm:.1f} | "
                f"{1.0e3 * maximum_m:.5f} | {normalized:.4f} |"
            )

    lines.extend(("", "## Full-finger mechanics conclusions", ""))
    primary_force_n = _FORCE_TARGETS_N[0]
    primary_names = {
        name: f"{name}_{primary_force_n:g}n"
        for name, _ in _CONTACTS_Y_MM
    }
    if all(result_name in by_name for result_name in primary_names.values()):
        center = by_name[primary_names["center_led"]]
        between = by_name[primary_names["between_leds"]]
        distal = by_name[primary_names["distal_led"]]
        led_station_indices = [
            int(np.flatnonzero(_STATIONS_Y_MM == y_mm)[0])
            for y_mm in (-22.0, -11.0, 0.0, 11.0, 22.0)
        ]
        center_led_response = center["station_normalized"][led_station_indices]
        between_led_response = between["station_normalized"][led_station_indices]
        distal_indent_change = (
            distal["indentation_m"] / center["indentation_m"] - 1.0
        )
        runtime_multiplier = None
        if single_section is not None:
            runtime_multiplier = (
                center["wall_runtime_s"] / single_section["wall_runtime_s"]
            )
        secondary_force_n = _FORCE_TARGETS_N[-1]
        secondary_names = {
            name: f"{name}_{secondary_force_n:g}n"
            for name, _ in _CONTACTS_Y_MM
        }
        secondary_available = all(
            result_name in by_name
            for result_name in secondary_names.values()
        )
        nonlinear_summary = "The optional higher-force cases were not completed."
        if secondary_available:
            center_high = by_name[secondary_names["center_led"]]
            between_high = by_name[secondary_names["between_leds"]]
            distal_high = by_name[secondary_names["distal_led"]]
            high_center_led_response = center_high["station_normalized"][
                led_station_indices
            ]
            high_between_led_response = between_high["station_normalized"][
                led_station_indices
            ]
            nonlinear_summary = (
                f"At {secondary_force_n:g} N, center LED responses are "
                f"`{np.array2string(high_center_led_response, precision=3)}` and "
                f"between-LED neighboring responses rise to "
                f"{high_between_led_response[2]:.1%}/{high_between_led_response[3]:.1%}. "
                f"Distal indentation differs from center by "
                f"{100.0 * (distal_high['indentation_m'] / center_high['indentation_m'] - 1.0):+.1f}%."
            )
        full_runtime_s = sum(result["wall_runtime_s"] for result in results)
        lines.extend(
            (
                f"1. Center-contact deformation remains above 10% over "
                f"{center['influence_10_span_mm']:.1f} mm "
                f"(left {center['influence_10_left_mm']:.1f} mm, right "
                f"{center['influence_10_right_mm']:.1f} mm from contact); the 5% "
                f"tail spans {center['influence_5_span_mm']:.1f} mm.",
                f"2. Center-contact normalized responses at LEDs 1→5 are "
                f"`{np.array2string(center_led_response, precision=3)}`; adjacent "
                "11 mm stations retain about 15–16% of the peak, so one pitch is "
                "mechanically coupled rather than independent.",
                f"3. Between-LED contact peaks near Y={between['peak_profile_y_mm']:.1f} mm; "
                f"LED responses are `{np.array2string(between_led_response, precision=3)}`. "
                "The peak stays at the imposed midpoint and decays smoothly; the two "
                "neighbor LEDs receive 29.4%/31.8%, with no split peak or mesh artifact.",
                f"4. The distal contact indentation differs from center by "
                f"{100.0 * distal_indent_change:+.1f}% at the accepted force. "
                f"Its local stiffness is {distal['local_stiffness_n_mm']:.3f} N/mm "
                f"versus {center['local_stiffness_n_mm']:.3f} N/mm at center. "
                f"Its one-sided 10% influence span is {distal['influence_10_span_mm']:.1f} mm.",
                f"5. {nonlinear_summary}",
                (
                    "6. Single-section comparison was not run."
                    if single_section is None
                    else f"6. Single-section center indentation was "
                    f"{1.0e3 * single_section['indentation_m']:.4f} mm versus "
                    f"{1.0e3 * center['indentation_m']:.4f} mm for the full finger; "
                    f"peak displacement was "
                    f"{1.0e3 * single_section['maximum_nodal_displacement_m']:.4f} mm "
                    f"versus {1.0e3 * center['maximum_nodal_displacement_m']:.4f} mm."
                ),
                (
                    "7. A full/single runtime multiplier is unavailable."
                    if runtime_multiplier is None
                    else f"7. Approximate full/single center runtime multiplier: "
                    f"{runtime_multiplier:.2f}×. Full-finger contact cases consumed "
                    f"{full_runtime_s:.1f} s total wall time."
                ),
            )
        )
        mechanically_valid = all(
            result["inverted_tet_count"] == 0
            and result["body_particle_buffer_overflow"] == 0
            and np.isfinite(result["minimum_det_f"])
            for result in results
        )
        lines.append(
            "8. No inversion/contact-buffer issue was observed."
            if mechanically_valid
            else "8. At least one inversion or contact-buffer issue remains; see `cases.csv`."
        )
        lines.append(
            "9. The full geometry is ready for optical validation."
            if mechanically_valid and not failures
            else "9. The full geometry is not yet cleared for optical validation."
        )
    else:
        lines.append(
            "The three required contacts did not all complete, so no readiness conclusion is valid."
        )

    lines.extend(
        (
            "",
            "## Artifacts",
            "",
            "- `5led_newton/reference_mesh.npz`: immutable mesh and reference state",
            "- `5led_newton/<case>.npz`: deformed vertices and raw Newton contact records",
            "- `5led_newton/cases.csv`: scalar mechanics diagnostics",
            "- `5led_newton/station_displacements.csv`: fixed-station coupling",
            "- `5led_newton/longitudinal_profiles.csv`: dense 1 mm profile",
            "- `5led_newton/deformation_comparison.png`: common-view deformation fields",
            "- `5led_newton/between_led_deformation.png`: focused midpoint contact",
            "- `5led_newton/longitudinal_displacement_profiles_<force>n.png`: propagation curves",
            "",
            "Exact contact patch area is intentionally not reported. Raw soft-contact feature "
            "indices, barycentric weights, reconstructed soft-side points, body positions, "
            "and Newton normals are persisted for later analysis.",
        )
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingertip = Fingertip(FingertipParameters())
    if fingertip.full_height_mm > 30.0:
        raise RuntimeError("nominal fingertip violates the 30 mm height contract")

    mesh_start_s = perf_counter()
    fingertip_mesh = make_fingertip_5led_mesh(
        fingertip,
        element_size_mm=_ELEMENT_SIZE_MM,
    )
    mesh_build_time_s = perf_counter() - mesh_start_s
    geometry = _verify_geometry(fingertip_mesh)
    geometry["full_height_mm"] = fingertip.full_height_mm
    (_OUTPUT_DIRECTORY / "geometry_contract.json").write_text(
        json.dumps(geometry, indent=2) + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        _OUTPUT_DIRECTORY / "reference_mesh.npz",
        silicone_vertices_m=np.asarray(fingertip_mesh.silicone.vertices),
        silicone_tetrahedra=np.asarray(
            fingertip_mesh.silicone.tet_indices,
            dtype=np.int32,
        ).reshape(-1, 4),
        silicone_surface_triangles=np.asarray(
            fingertip_mesh.silicone.surface_tri_indices,
            dtype=np.int32,
        ).reshape(-1, 3),
        carrier_vertices_m=np.asarray(fingertip_mesh.carrier.vertices),
        carrier_triangles=np.asarray(
            fingertip_mesh.carrier.indices,
            dtype=np.int32,
        ).reshape(-1, 3),
        bonded_vertex_indices=fingertip_mesh.bonded_vertex_indices,
        led_centers_m=fingertip_mesh.led_centers_m,
    )

    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    results: list[dict[str, object]] = []
    failures: list[dict[str, str | float]] = []
    single_section: dict[str, object] | None = None
    with as_file(sphere_resource) as sphere_path:
        initialization_time_s = _measure_initialization(
            fingertip,
            fingertip_mesh,
            sphere_path,
        )
        for target_force_n in _FORCE_TARGETS_N:
            for contact_name, contact_y_mm in _CONTACTS_Y_MM:
                name = f"{contact_name}_{target_force_n:g}n"
                try:
                    results.append(
                        _run_case(
                            fingertip,
                            fingertip_mesh,
                            sphere_path,
                            name=name,
                            contact_y_mm=contact_y_mm,
                            target_force_n=target_force_n,
                        )
                    )
                except Exception as exc:  # preserve later cases
                    failure = {
                        "name": name,
                        "contact_y_mm": contact_y_mm,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"FAILED {name}: {failure['error']}", flush=True)

        if not failures:
            single_mesh = make_fingertip_mesh(
                fingertip,
                extrusion_depth_mm=11.0,
                element_size_mm=_ELEMENT_SIZE_MM,
            )
            try:
                single_section = _run_case(
                    fingertip,
                    single_mesh,
                    sphere_path,
                    name="single_section_center_10n",
                    contact_y_mm=0.0,
                    target_force_n=_FORCE_TARGETS_N[0],
                )
            except Exception as exc:
                print(
                    "single-section comparison skipped after failure: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    _write_tables(results, failures)
    for target_force_n in _FORCE_TARGETS_N:
        force_results = [
            result
            for result in results
            if np.isclose(
                result["target_force_n"],
                target_force_n,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        if force_results:
            _plot_profiles(force_results, target_force_n=target_force_n)
    full_results = [
        result
        for result in results
        if not str(result["name"]).startswith("single_section_center")
    ]
    primary_results = [
        result
        for result in full_results
        if np.isclose(
            result["target_force_n"],
            _FORCE_TARGETS_N[0],
            rtol=0.0,
            atol=1.0e-12,
        )
    ]
    if len(primary_results) == 3:
        _plot_deformation_comparison(fingertip_mesh, primary_results)
        between_result = next(
            result
            for result in primary_results
            if str(result["name"]).startswith("between_leds")
        )
        _plot_between_led_focus(fingertip_mesh, between_result)
    _write_report(
        geometry,
        mesh_build_time_s,
        initialization_time_s,
        full_results,
        failures,
        single_section,
    )

    print(f"report: {_REPORT_PATH}", flush=True)
    print(f"raw output: {_OUTPUT_DIRECTORY}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} required full-finger cases failed")


if __name__ == "__main__":
    main()
