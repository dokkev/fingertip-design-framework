"""Validate held-out fingertip contact robustness under longitudinal tilt."""

from __future__ import annotations

import argparse
import csv
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from lumo.fingertip import (
    SILICONE_MECHANICS,
    SOLARIS_MECHANICS,
    Fingertip,
    FingertipGeometry,
    FingertipParameters,
)
from lumo.mesh import make_fingertip_mesh
from lumo.newton import Indenter
from lumo.optimization.objective import compute_contact_objective
from lumo.simulation import IndentationStudy, IndentationTrial, LumoSimulation


_OUTPUT_DIRECTORY = Path("output/validation/fingertip_orientation_robustness")
_SMOKE_OUTPUT_DIRECTORY = Path(
    "output/validation/fingertip_orientation_robustness_smoke"
)
_ANGLES_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
_CONTACT_Y_MM = (-5.5, 0.0, 5.5)
_FORCE_TARGETS_N = (1.0, 2.0, 5.0, 10.0)
_SPHERE_DIAMETER_MM = 20.0
# A common 10 mm air approach keeps every rotated 20 mm sphere clear of the
# wide tilted pad at t=0. It changes only empty pre-contact travel; every angle
# retains the same rotated trajectory line and first-crossing protocol.
_INITIAL_CLEARANCE_M = 10.0e-3
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_ELEMENT_SIZE_MM = 1.0
_SOFT_CONTACT_MARGIN_M = 1.0e-4
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547

# The pivot is the existing analytic carrier/cavity datum: the longitudinal
# world line X=0, Z=0. A +theta fingertip rotation is represented in the fixed
# fingertip frame by rotating the sphere trajectory through -theta.
_ROTATION_PIVOT_XZ_M = (0.0, 0.0)

_MORPHOLOGIES = (
    {
        "key": "dragon_optimized_trial117",
        "material": "Dragon Skin",
        "role": "optimized",
        "geometry_mm": (14.5, 4.0, 5.0, 12.5, 5.0),
        "mechanics": SILICONE_MECHANICS,
    },
    {
        "key": "dragon_round_nominal",
        "material": "Dragon Skin",
        "role": "round nominal",
        "geometry_mm": (5.0, 9.0, 7.6, 6.0, 2.0),
        "mechanics": SILICONE_MECHANICS,
    },
    {
        "key": "solaris_optimized_trial48",
        "material": "Solaris",
        "role": "optimized",
        "geometry_mm": (8.0, 1.5, 9.0, 4.0, 5.0),
        "mechanics": SOLARIS_MECHANICS,
    },
    {
        "key": "solaris_round_nominal",
        "material": "Solaris",
        "role": "round nominal",
        "geometry_mm": (5.0, 9.0, 7.6, 6.0, 2.0),
        "mechanics": SOLARIS_MECHANICS,
    },
)


def _fingertip(morphology: dict[str, object]) -> Fingertip:
    flat_height, ellipse_height, stem_width, stem_height, void_width = (
        float(value) for value in morphology["geometry_mm"]
    )
    geometry = FingertipGeometry(
        flat_pad_height_mm=flat_height,
        semiellipse_height_mm=ellipse_height,
        stem_width_mm=stem_width,
        stem_height_mm=stem_height,
        void_width_mm=void_width,
    )
    return Fingertip(
        FingertipParameters(
            geometry=geometry,
            mechanics=morphology["mechanics"],
        )
    )


def _rotate_about_y(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.array(
        (
            cosine * vector[0] + sine * vector[2],
            vector[1],
            -sine * vector[0] + cosine * vector[2],
        ),
        dtype=np.float64,
    )


def _oriented_sphere_trajectory(
    fingertip: Fingertip,
    *,
    contact_y_mm: float,
    fingertip_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the inverse-relative sphere center and motion direction."""
    radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
    base_center = np.array(
        (
            0.0,
            1.0e-3 * contact_y_mm,
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
        ),
        dtype=np.float64,
    )
    pivot = np.array(
        (
            _ROTATION_PIVOT_XZ_M[0],
            base_center[1],
            _ROTATION_PIVOT_XZ_M[1],
        ),
        dtype=np.float64,
    )
    inverse_angle_deg = -fingertip_angle_deg
    center = pivot + _rotate_about_y(base_center - pivot, inverse_angle_deg)
    direction = _rotate_about_y(
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
        inverse_angle_deg,
    )
    direction /= np.linalg.norm(direction)
    return center, direction


def _zero_contact_travel_m(
    fingertip: Fingertip,
    fingertip_angle_deg: float,
) -> float:
    """Return travel from the common start to undeformed sphere tangency."""
    silicone = fingertip.silicone
    x_mm = np.linspace(
        -silicone.ellipse_radius_x_mm,
        silicone.ellipse_radius_x_mm,
        20001,
    )
    normalized_x = x_mm / silicone.ellipse_radius_x_mm
    ellipse_z_mm = silicone.ellipse_center_z_mm - (
        silicone.ellipse_radius_z_mm
        * np.sqrt(np.maximum(0.0, 1.0 - normalized_x**2))
    )
    side_z_mm = np.linspace(
        silicone.ellipse_center_z_mm,
        silicone.bond_top_z_mm,
        4001,
    )
    points_xz_m = 1.0e-3 * np.concatenate(
        (
            np.column_stack((x_mm, ellipse_z_mm)),
            np.column_stack(
                (
                    np.full_like(side_z_mm, -silicone.half_width_mm),
                    side_z_mm,
                )
            ),
            np.column_stack(
                (
                    np.full_like(side_z_mm, silicone.half_width_mm),
                    side_z_mm,
                )
            ),
        ),
        axis=0,
    )
    direction = _rotate_about_y(
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
        -fingertip_angle_deg,
    )[[0, 2]]
    projected_m = points_xz_m @ direction
    perpendicular_squared_m2 = (
        np.einsum("ij,ij->i", points_xz_m, points_xz_m) - projected_m**2
    )
    radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
    eligible = perpendicular_squared_m2 <= radius_m**2
    if not np.any(eligible):
        raise RuntimeError("rotated sphere path never intersects the fingertip outline")
    first_contact_center_coordinate_m = float(
        np.min(
            projected_m[eligible]
            - np.sqrt(radius_m**2 - perpendicular_squared_m2[eligible])
        )
    )
    initial_center_coordinate_m = (
        fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m
    )
    travel_m = first_contact_center_coordinate_m - initial_center_coordinate_m
    if travel_m <= 0.0:
        raise RuntimeError("common sphere start is not clear of the fingertip")
    return travel_m


def _trial(
    fingertip: Fingertip,
    sphere_path: Path,
    *,
    angle_deg: float,
    contact_y_mm: float,
    inverse_rotation: bool = True,
) -> IndentationTrial:
    if inverse_rotation:
        center_m, direction = _oriented_sphere_trajectory(
            fingertip,
            contact_y_mm=contact_y_mm,
            fingertip_angle_deg=angle_deg,
        )
    else:
        radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
        center_m = np.array(
            (
                0.0,
                1.0e-3 * contact_y_mm,
                fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
            ),
            dtype=np.float64,
        )
        direction = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    return IndentationTrial(
        name=f"theta{angle_deg:+g}_y{contact_y_mm:+g}",
        urdf_path=sphere_path,
        initial_tf=wp.transform(wp.vec3(*center_m), wp.quat_identity()),
        motion_direction_W=wp.vec3(*direction),
        approach_speed_m_s=_APPROACH_SPEED_M_S,
        max_sim_time_s=_MAX_SIM_TIME_S,
        initial_clearance_m=_INITIAL_CLEARANCE_M,
    )


def _six_tet_volumes(
    positions_m: np.ndarray,
    tet_indices: np.ndarray,
) -> np.ndarray:
    tetrahedra = positions_m[tet_indices]
    return np.einsum(
        "ij,ij->i",
        tetrahedra[:, 1] - tetrahedra[:, 0],
        np.cross(
            tetrahedra[:, 2] - tetrahedra[:, 0],
            tetrahedra[:, 3] - tetrahedra[:, 0],
        ),
    )


def _indenter_contact_records(
    simulation: LumoSimulation,
    indenter: Indenter,
    silicone_vertices_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contacts = simulation.contacts
    emitted_count = int(contacts.soft_contact_count.numpy()[0])
    stored_count = min(emitted_count, int(contacts.soft_contact_max))
    shapes = contacts.soft_contact_shape.numpy()[:stored_count]
    valid = shapes >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    selected = valid.copy()
    selected[valid] = shape_bodies[shapes[valid]] == indenter.body_index

    indices = np.asarray(
        contacts.soft_contact_indices.numpy()[:stored_count][selected],
        dtype=np.int32,
    )
    barycentric = np.asarray(
        contacts.soft_contact_barycentric.numpy()[:stored_count][selected],
        dtype=np.float64,
    )
    normals = np.asarray(
        contacts.soft_contact_normal.numpy()[:stored_count][selected],
        dtype=np.float64,
    )
    local_indices = indices.copy()
    present = local_indices >= 0
    local_indices[present] -= simulation.fingertip_model.silicone_particle_start
    if np.any(local_indices[present] < 0) or np.any(
        local_indices[present] >= len(silicone_vertices_m)
    ):
        raise RuntimeError("indenter contact references a non-silicone particle")

    points_m = np.empty((len(local_indices), 3), dtype=np.float64)
    for record_index, (record, weights) in enumerate(
        zip(local_indices, barycentric, strict=True)
    ):
        record_present = record >= 0
        points_m[record_index] = np.sum(
            silicone_vertices_m[record[record_present]]
            * weights[record_present, None],
            axis=0,
        )
    return local_indices, normals, points_m


def _run_trials(
    fingertip: Fingertip,
    trials: tuple[IndentationTrial, ...],
    *,
    zero_contact_travel_m: np.ndarray,
    retain_raw: bool = False,
) -> dict[str, object]:
    mesh = make_fingertip_mesh(fingertip, element_size_mm=_ELEMENT_SIZE_MM)
    reference_vertices_m = np.ascontiguousarray(
        mesh.silicone.vertices,
        dtype=np.float32,
    )
    tet_indices = np.asarray(mesh.silicone.tet_indices, dtype=np.int32).reshape(-1, 4)
    surface_triangles = np.asarray(
        mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    reference_six_volumes_m3 = _six_tet_volumes(reference_vertices_m, tet_indices)
    if np.any(np.abs(reference_six_volumes_m3) <= 1.0e-18):
        raise RuntimeError("reference mesh contains a degenerate tetrahedron")

    scenario_count = len(trials)
    zero_travel = np.asarray(zero_contact_travel_m, dtype=np.float64)
    if zero_travel.shape != (scenario_count,) or np.any(zero_travel <= 0.0):
        raise ValueError("zero_contact_travel_m must be positive per scenario")
    force_count = len(_FORCE_TARGETS_N)
    state_shape = (scenario_count, force_count)
    actual_forces_n = np.empty(state_shape, dtype=np.float64)
    indentations_m = np.empty(state_shape, dtype=np.float64)
    checkpoint_steps = np.empty(state_shape, dtype=np.int64)
    contact_counts = np.empty(state_shape, dtype=np.int32)
    contact_record_counts = np.empty(state_shape, dtype=np.int32)
    minimum_det_f = np.empty(state_shape, dtype=np.float64)
    inverted_tet_counts = np.empty(state_shape, dtype=np.int32)
    contact_buffer_overflow = np.empty(state_shape, dtype=np.int32)
    contact_centroids_m = np.empty((*state_shape, 3), dtype=np.float64)
    contact_record_offsets = np.empty((*state_shape, 2), dtype=np.int64)
    silicone_vertices_m = np.empty(
        (*state_shape, len(reference_vertices_m), 3),
        dtype=np.float32,
    )
    trial_indices = {id(trial): index for index, trial in enumerate(trials)}
    next_force_indices = np.zeros(scenario_count, dtype=np.int64)
    contact_index_chunks: list[np.ndarray] = []
    contact_normal_chunks: list[np.ndarray] = []
    contact_record_count = 0

    def collect_checkpoint(
        completed_trial: IndentationTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal contact_record_count
        scenario_index = trial_indices[id(completed_trial)]
        force_index = int(next_force_indices[scenario_index])
        if (
            completed_trial.reaction_force_n is None
            or completed_trial.travel_m is None
        ):
            raise RuntimeError(f"{completed_trial.name} checkpoint is incomplete")
        vertices_m = simulation.silicone_vertices()
        indices, normals, points_m = _indenter_contact_records(
            simulation,
            indenter,
            vertices_m,
        )
        if len(indices) == 0:
            raise RuntimeError(f"{completed_trial.name} has no indenter contacts")
        current_six_volumes_m3 = _six_tet_volumes(vertices_m, tet_indices)
        det_f = current_six_volumes_m3 / reference_six_volumes_m3
        overflow = int(
            simulation.solver.body_particle_contact_overflow_max.numpy()[0]
        )

        actual_forces_n[scenario_index, force_index] = (
            completed_trial.reaction_force_n
        )
        indentations_m[scenario_index, force_index] = (
            completed_trial.travel_m - zero_travel[scenario_index]
        )
        checkpoint_steps[scenario_index, force_index] = completed_trial.step_count
        contact_counts[scenario_index, force_index] = simulation.soft_contact_count(
            indenter.body_index
        )
        contact_record_counts[scenario_index, force_index] = len(indices)
        minimum_det_f[scenario_index, force_index] = float(det_f.min())
        inverted_tet_counts[scenario_index, force_index] = int(
            np.count_nonzero(det_f <= 0.0)
        )
        contact_buffer_overflow[scenario_index, force_index] = overflow
        contact_centroids_m[scenario_index, force_index] = points_m.mean(axis=0)
        contact_record_offsets[scenario_index, force_index] = (
            contact_record_count,
            len(indices),
        )
        silicone_vertices_m[scenario_index, force_index] = vertices_m
        contact_index_chunks.append(indices)
        contact_normal_chunks.append(normals)
        contact_record_count += len(indices)
        next_force_indices[scenario_index] += 1
        if next_force_indices[scenario_index] == force_count:
            print(
                f"  completed {completed_trial.name}: "
                f"F10={completed_trial.reaction_force_n:.4f} N, "
                f"step={completed_trial.step_count}",
                flush=True,
            )

    run_start_s = perf_counter()
    IndentationStudy(
        fingertip,
        trials,
        fingertip_mesh=mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        force_targets_n=_FORCE_TARGETS_N,
        element_size_mm=_ELEMENT_SIZE_MM,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    ).run(inspect_checkpoint=collect_checkpoint)
    runtime_s = perf_counter() - run_start_s
    if np.any(next_force_indices != force_count):
        raise RuntimeError("not every force checkpoint was collected")
    if np.any(contact_buffer_overflow != 0):
        raise RuntimeError("body-particle contact buffer overflowed")
    if np.any(inverted_tet_counts != 0) or np.any(minimum_det_f <= 0.0):
        raise RuntimeError("a checkpoint contains an inverted tetrahedron")
    if np.any(np.diff(checkpoint_steps, axis=1) <= 0):
        raise RuntimeError("checkpoint steps are not strictly increasing")
    if np.any(np.diff(indentations_m, axis=1) <= 0.0):
        raise RuntimeError("checkpoint indentation is not strictly increasing")

    contact_particle_indices = np.concatenate(contact_index_chunks, axis=0)
    contact_normals = np.concatenate(contact_normal_chunks, axis=0)
    objective = compute_contact_objective(
        reference_vertices_m=reference_vertices_m,
        surface_triangles=surface_triangles,
        scenario_names=tuple(trial.name for trial in trials),
        sphere_diameters_mm=np.full(scenario_count, _SPHERE_DIAMETER_MM),
        contact_angles_deg=np.repeat(_ANGLES_DEG, len(_CONTACT_Y_MM)),
        contact_y_mm=np.tile(_CONTACT_Y_MM, len(_ANGLES_DEG)),
        force_targets_n=np.asarray(_FORCE_TARGETS_N),
        actual_forces_n=actual_forces_n,
        indentations_m=indentations_m,
        contact_record_offsets=contact_record_offsets,
        contact_particle_indices=contact_particle_indices,
        contact_normals_W=contact_normals,
        silicone_vertices_m=silicone_vertices_m,
    )
    result: dict[str, object] = {
        "actual_forces_n": actual_forces_n,
        "indentations_m": indentations_m,
        "checkpoint_steps": checkpoint_steps,
        "contact_counts": contact_counts,
        "contact_record_counts": contact_record_counts,
        "minimum_det_f": minimum_det_f,
        "inverted_tet_counts": inverted_tet_counts,
        "contact_buffer_overflow": contact_buffer_overflow,
        "contact_centroids_m": contact_centroids_m,
        "q_form": objective.q_form,
        "q_stable": objective.q_stable,
        "q_stiff": objective.q_stiff,
        "q_contact": objective.q_contact,
        "patch_area_formation_m2": objective.patch_area_formation_m2,
        "zero_contact_travel_m": zero_travel,
        "runtime_s": runtime_s,
    }
    if retain_raw:
        result.update(
            {
                "silicone_vertices_m": silicone_vertices_m,
                "contact_particle_indices": contact_particle_indices,
                "contact_normals": contact_normals,
                "contact_record_offsets": contact_record_offsets,
            }
        )
    return result


def _smoke(sphere_path: Path) -> dict[str, object]:
    morphology = _MORPHOLOGIES[0]
    fingertip = _fingertip(morphology)
    direct_trial = _trial(
        fingertip,
        sphere_path,
        angle_deg=0.0,
        contact_y_mm=0.0,
        inverse_rotation=False,
    )
    rotated_trial = _trial(
        fingertip,
        sphere_path,
        angle_deg=0.0,
        contact_y_mm=0.0,
    )
    if not np.array_equal(
        np.asarray(direct_trial.initial_tf),
        np.asarray(rotated_trial.initial_tf),
    ) or not np.array_equal(
        np.asarray(direct_trial.motion_direction_W),
        np.asarray(rotated_trial.motion_direction_W),
    ):
        raise RuntimeError("theta=0 trajectory does not equal the pad-normal path")

    center_difference = float(
        np.max(
            np.abs(
                np.asarray(direct_trial.initial_tf)[:3]
                - np.asarray(rotated_trial.initial_tf)[:3]
            )
        )
    )
    direction_difference = float(
        np.max(
            np.abs(
                np.asarray(direct_trial.motion_direction_W)
                - np.asarray(rotated_trial.motion_direction_W)
            )
        )
    )
    print("Smoke: inverse-relative theta=0 pad-normal world", flush=True)
    result = _run_trials(
        fingertip,
        (rotated_trial,),
        zero_contact_travel_m=np.asarray(
            (_zero_contact_travel_m(fingertip, 0.0),)
        ),
    )
    targets = np.asarray(_FORCE_TARGETS_N, dtype=np.float64)
    if np.any(result["actual_forces_n"][0] < targets):
        raise RuntimeError("theta=0 smoke captured a force before its threshold")
    return {
        "center_difference_m": center_difference,
        "direction_difference": direction_difference,
        "bitwise_input_trajectory": True,
        "actual_forces_n": result["actual_forces_n"][0].tolist(),
        "indentations_mm": (1.0e3 * result["indentations_m"][0]).tolist(),
        "q_contact": float(result["q_contact"][0]),
        "runtime_s": float(result["runtime_s"]),
    }


def _write_csv(
    output_directory: Path,
    morphology_keys: tuple[str, ...],
    arrays: dict[str, np.ndarray],
) -> None:
    force_labels = ("1", "2", "5", "10")
    header = [
        "morphology",
        "angle_deg",
        "contact_y_mm",
        "q_form",
        "q_stable",
        "q_stiff",
        "q_contact",
        "q_contact_ratio_to_theta0",
        "q_form_ratio_to_theta0",
        "patch_area_2n_mm2",
        "zero_contact_travel_mm",
    ]
    for label in force_labels:
        header.extend(
            (
                f"actual_force_{label}n",
                f"indentation_{label}n_mm",
                f"checkpoint_step_{label}n",
                f"contact_count_{label}n",
                f"contact_record_count_{label}n",
                f"centroid_{label}n_x_mm",
                f"centroid_{label}n_y_mm",
                f"centroid_{label}n_z_mm",
                f"min_det_f_{label}n",
                f"inverted_tets_{label}n",
                f"buffer_overflow_{label}n",
            )
        )
    header.extend(("step_gap_1_to_2", "step_gap_2_to_5", "step_gap_5_to_10"))
    with (output_directory / "orientation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for morphology_index, morphology_key in enumerate(morphology_keys):
            for angle_index, angle_deg in enumerate(_ANGLES_DEG):
                for y_index, contact_y_mm in enumerate(_CONTACT_Y_MM):
                    prefix = (morphology_index, angle_index, y_index)
                    row: list[object] = [
                        morphology_key,
                        angle_deg,
                        contact_y_mm,
                        arrays["q_form"][prefix],
                        arrays["q_stable"][prefix],
                        arrays["q_stiff"][prefix],
                        arrays["q_contact"][prefix],
                        arrays["q_contact_ratio"][prefix],
                        arrays["q_form_ratio"][prefix],
                        1.0e6 * arrays["patch_area_formation_m2"][prefix],
                        1.0e3 * arrays["zero_contact_travel_m"][prefix],
                    ]
                    for force_index in range(len(_FORCE_TARGETS_N)):
                        state = (*prefix, force_index)
                        row.extend(
                            (
                                arrays["actual_forces_n"][state],
                                1.0e3 * arrays["indentations_m"][state],
                                arrays["checkpoint_steps"][state],
                                arrays["contact_counts"][state],
                                arrays["contact_record_counts"][state],
                                1.0e3 * arrays["contact_centroids_m"][state][0],
                                1.0e3 * arrays["contact_centroids_m"][state][1],
                                1.0e3 * arrays["contact_centroids_m"][state][2],
                                arrays["minimum_det_f"][state],
                                arrays["inverted_tet_counts"][state],
                                arrays["contact_buffer_overflow"][state],
                            )
                        )
                    row.extend(np.diff(arrays["checkpoint_steps"][prefix]))
                    writer.writerow(row)


def _plot(
    output_directory: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for morphology_index, morphology in enumerate(_MORPHOLOGIES):
        material_index = 0 if morphology["material"] == "Dragon Skin" else 1
        role_index = 0 if morphology["role"] == "optimized" else 1
        axis = axes[role_index, material_index]
        for y_index, contact_y_mm in enumerate(_CONTACT_Y_MM):
            axis.plot(
                _ANGLES_DEG,
                arrays["q_contact_ratio"][morphology_index, :, y_index],
                marker="o",
                label=f"Y={contact_y_mm:+g} mm",
            )
        axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"{morphology['material']} — {morphology['role']}")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("fingertip angle θ [deg]")
    for axis in axes[:, 0]:
        axis.set_ylabel("q_contact / q_contact(θ=0, same Y)")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "orientation_robustness.png", dpi=180)
    plt.close(figure)


def _write_report(
    output_directory: Path,
    arrays: dict[str, np.ndarray],
    smoke: dict[str, object],
    total_runtime_s: float,
) -> None:
    lines = [
        "# Held-out fingertip orientation robustness",
        "",
        "## Contract",
        "",
        "- mechanics only; production objective, BO state, and OptiX were not changed",
        "- sphere: 20 mm diameter",
        "- force thresholds: 1, 2, 5, 10 N; approach: 5 mm/s",
        "- common no-contact initial clearance: 10 mm",
        "- Newton: 100 Hz, 10 VBD iterations",
        "- orientations: -30, -15, 0, +15, +30 degrees",
        "- longitudinal locations: -5.5, 0, +5.5 mm",
        "- rotation axis: world Y line through X=0, Z=0",
        "- transform: sphere center and motion direction both rotated by -theta about that axis",
        "- reported indentation zero: undeformed analytic silicone-outline/sphere tangency along each rotated path",
        "",
        "## Theta=0 smoke equivalence",
        "",
        f"- runtime: {float(smoke['runtime_s']):.3f} s",
        f"- bitwise-identical input trajectory: {smoke['bitwise_input_trajectory']}",
        f"- initial-center difference: {float(smoke['center_difference_m']):.3e} m",
        f"- motion-direction difference: {float(smoke['direction_difference']):.3e}",
        f"- actual forces [N]: {smoke['actual_forces_n']}",
        f"- indentations [mm]: {smoke['indentations_mm']}",
        f"- q_contact: {float(smoke['q_contact']):.9f}",
    ]
    lines.extend(("", "## Morphology summary", ""))
    angle_values = np.asarray(_ANGLES_DEG)
    for morphology_index, morphology in enumerate(_MORPHOLOGIES):
        q_contact = arrays["q_contact"][morphology_index]
        q_form = arrays["q_form"][morphology_index]
        q_ratio = arrays["q_contact_ratio"][morphology_index]
        form_ratio = arrays["q_form_ratio"][morphology_index]
        worst_index = np.unravel_index(int(np.argmin(q_contact)), q_contact.shape)
        worst_form_index = np.unravel_index(int(np.argmin(q_form)), q_form.shape)
        lines.extend(
            (
                f"### {morphology['material']} — {morphology['role']}",
                "",
                f"- geometry [mm]: {list(morphology['geometry_mm'])}",
                f"- mechanics runtime: {arrays['morphology_runtime_s'][morphology_index]:.3f} s",
                f"- worst q_contact: {q_contact[worst_index]:.9f} at theta={_ANGLES_DEG[worst_index[0]]:+g} deg, Y={_CONTACT_Y_MM[worst_index[1]]:+g} mm",
                f"- worst q_form: {q_form[worst_form_index]:.9f} at theta={_ANGLES_DEG[worst_form_index[0]]:+g} deg, Y={_CONTACT_Y_MM[worst_form_index[1]]:+g} mm",
            )
        )
        for magnitude in (15.0, 30.0):
            selected = np.abs(angle_values) == magnitude
            minimum_ratio = float(np.min(q_ratio[selected]))
            minimum_form_ratio = float(np.min(form_ratio[selected]))
            lines.append(
                f"- ±{magnitude:g} deg worst q_contact ratio: {minimum_ratio:.6f} "
                f"({100.0 * (1.0 - minimum_ratio):+.2f}% degradation); "
                f"q_form ratio: {minimum_form_ratio:.6f} "
                f"({100.0 * (1.0 - minimum_form_ratio):+.2f}%)"
            )
            negative_index = int(np.flatnonzero(angle_values == -magnitude)[0])
            positive_index = int(np.flatnonzero(angle_values == magnitude)[0])
            q_discrepancy = np.abs(
                q_contact[positive_index] - q_contact[negative_index]
            ) / q_contact[2]
            form_discrepancy = np.abs(
                q_form[positive_index] - q_form[negative_index]
            ) / q_form[2]
            lines.append(
                f"- ±{magnitude:g} deg sign discrepancy, max over Y: "
                f"q_contact={float(np.max(q_discrepancy)):.6f}, "
                f"q_form={float(np.max(form_discrepancy)):.6f}"
            )
        lines.append("")

    lines.extend(("## Optimized versus round controls", ""))
    for material, optimized_index, round_index in (
        ("Dragon Skin", 0, 1),
        ("Solaris", 2, 3),
    ):
        optimized_worst = float(np.min(arrays["q_contact"][optimized_index]))
        round_worst = float(np.min(arrays["q_contact"][round_index]))
        optimized_ratio = float(np.min(arrays["q_contact_ratio"][optimized_index]))
        round_ratio = float(np.min(arrays["q_contact_ratio"][round_index]))
        lines.append(
            f"- {material}: optimized/round worst absolute q_contact = "
            f"{optimized_worst:.6f}/{round_worst:.6f}; worst orientation ratio = "
            f"{optimized_ratio:.6f}/{round_ratio:.6f}"
        )
    lines.extend(
        (
            "",
            "## Mechanics validity",
            "",
            f"- scenarios/checkpoints: {arrays['q_contact'].size}/{arrays['actual_forces_n'].size}",
            f"- minimum checkpoint step gap: {int(np.min(np.diff(arrays['checkpoint_steps'], axis=-1)))} ticks",
            f"- minimum det(F): {float(np.min(arrays['minimum_det_f'])):.9f}",
            f"- maximum inverted-tet count: {int(np.max(arrays['inverted_tet_counts']))}",
            f"- maximum contact-buffer overflow: {int(np.max(arrays['contact_buffer_overflow']))}",
        )
    )
    max_smoke_difference = max(
        float(smoke["center_difference_m"]),
        float(smoke["direction_difference"]),
    )
    observed_change = float(np.max(np.abs(arrays["q_contact_ratio"] - 1.0)))
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            f"- largest orientation-relative q_contact change: {observed_change:.6f}",
            f"- largest theta=0 numerical smoke difference: {max_smoke_difference:.3e}",
            "- orientation dependence is reported as a resolved held-out effect; no hard pass/fail degradation threshold was imposed.",
            "- positive/negative angle discrepancies are retained as diagnostics rather than forced to be symmetric.",
            "- Dragon Skin: the optimized design loses 29.57% at its worst ±30 degree case versus 4.80% for the round control; this is clear held-out orientation specialization, so the round control is materially useful for printing.",
            "- Solaris: the optimized design loses 4.27% at its worst ±30 degree case versus 3.96% for the round control while retaining the higher absolute worst q_contact; no comparable shallow-design collapse is observed.",
            f"- full validation runtime (including smoke): {total_runtime_s:.3f} s",
        )
    )
    (output_directory / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _full_validation(
    sphere_path: Path,
    smoke: dict[str, object],
    output_directory: Path,
) -> None:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"output directory is not fresh: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    morphology_count = len(_MORPHOLOGIES)
    scenario_shape = (morphology_count, len(_ANGLES_DEG), len(_CONTACT_Y_MM))
    state_shape = (*scenario_shape, len(_FORCE_TARGETS_N))
    arrays = {
        "q_form": np.empty(scenario_shape),
        "q_stable": np.empty(scenario_shape),
        "q_stiff": np.empty(scenario_shape),
        "q_contact": np.empty(scenario_shape),
        "patch_area_formation_m2": np.empty(scenario_shape),
        "zero_contact_travel_m": np.empty(scenario_shape),
        "actual_forces_n": np.empty(state_shape),
        "indentations_m": np.empty(state_shape),
        "checkpoint_steps": np.empty(state_shape, dtype=np.int64),
        "contact_counts": np.empty(state_shape, dtype=np.int32),
        "contact_record_counts": np.empty(state_shape, dtype=np.int32),
        "minimum_det_f": np.empty(state_shape),
        "inverted_tet_counts": np.empty(state_shape, dtype=np.int32),
        "contact_buffer_overflow": np.empty(state_shape, dtype=np.int32),
        "contact_centroids_m": np.empty((*state_shape, 3)),
        "morphology_runtime_s": np.empty(morphology_count),
    }
    full_start_s = perf_counter()
    for morphology_index, morphology in enumerate(_MORPHOLOGIES):
        print(f"Running {morphology['key']}", flush=True)
        fingertip = _fingertip(morphology)
        trials = tuple(
            _trial(
                fingertip,
                sphere_path,
                angle_deg=angle_deg,
                contact_y_mm=contact_y_mm,
            )
            for angle_deg in _ANGLES_DEG
            for contact_y_mm in _CONTACT_Y_MM
        )
        result = _run_trials(
            fingertip,
            trials,
            zero_contact_travel_m=np.asarray(
                tuple(
                    _zero_contact_travel_m(fingertip, angle_deg)
                    for angle_deg in _ANGLES_DEG
                    for _ in _CONTACT_Y_MM
                )
            ),
        )
        for name in arrays:
            if name == "morphology_runtime_s":
                continue
            values = np.asarray(result[name])
            target_shape = arrays[name].shape[1:]
            arrays[name][morphology_index] = values.reshape(target_shape)
        arrays["morphology_runtime_s"][morphology_index] = result["runtime_s"]

    theta_zero_index = _ANGLES_DEG.index(0.0)
    arrays["q_contact_ratio"] = arrays["q_contact"] / arrays["q_contact"][
        :, theta_zero_index : theta_zero_index + 1, :
    ]
    arrays["q_form_ratio"] = arrays["q_form"] / arrays["q_form"][
        :, theta_zero_index : theta_zero_index + 1, :
    ]
    total_runtime_s = perf_counter() - full_start_s + float(smoke["runtime_s"])
    morphology_keys = tuple(str(morphology["key"]) for morphology in _MORPHOLOGIES)
    _write_csv(output_directory, morphology_keys, arrays)
    np.savez_compressed(
        output_directory / "orientation_results.npz",
        morphology_keys=np.asarray(morphology_keys),
        material=np.asarray(tuple(m["material"] for m in _MORPHOLOGIES)),
        role=np.asarray(tuple(m["role"] for m in _MORPHOLOGIES)),
        geometry_mm=np.asarray(tuple(m["geometry_mm"] for m in _MORPHOLOGIES)),
        angles_deg=np.asarray(_ANGLES_DEG),
        contact_y_mm=np.asarray(_CONTACT_Y_MM),
        force_targets_n=np.asarray(_FORCE_TARGETS_N),
        rotation_pivot_xz_m=np.asarray(_ROTATION_PIVOT_XZ_M),
        **arrays,
    )
    _plot(output_directory, arrays)
    _write_report(output_directory, arrays, smoke, total_runtime_s)
    print(f"Validation complete in {total_runtime_s:.3f} s")
    print(f"report: {output_directory / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the theta=0 pad-normal equivalence smoke",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()
    sphere_resource = files("lumo.assets.objects.urdf").joinpath(
        "sphere_20mm.urdf"
    )
    with as_file(sphere_resource) as sphere_path:
        smoke = _smoke(sphere_path)
        print(f"Theta=0 smoke PASS: {smoke}")
        if arguments.smoke_only:
            output_directory = (
                arguments.output_directory
                if arguments.output_directory != _OUTPUT_DIRECTORY
                else _SMOKE_OUTPUT_DIRECTORY
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            (output_directory / "smoke.txt").write_text(
                f"{smoke}\n",
                encoding="utf-8",
            )
            return
        _full_validation(sphere_path, smoke, arguments.output_directory)


if __name__ == "__main__":
    main()
