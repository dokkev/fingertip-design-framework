"""Extract sentinel force states with fixed-pose relaxed indentation loading."""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.newton import Indenter
from lumo.optimization.evaluator import (
    _CARRIER_INSTANCE_ID,
    _CARRIER_MASK,
    _ENERGY_FIELDS,
    _SILICONE_INSTANCE_ID,
    _SILICONE_MASK,
    _full_finger_emissions,
    _full_finger_optical_samples,
    _indenter_contact_records,
    _make_full_finger_leds,
    _six_tet_volumes,
    _trace_full_finger_state,
)
from lumo.optimization.objective import compute_contact_objective
from lumo.ray_tracing import OptixScene
from lumo.simulation import LumoSimulation


_OUTPUT_DIRECTORY = (
    Path("output/validation/production_evaluator_acceleration")
    / "post_phase4_indentation_control"
)
_REFERENCE_DIRECTORY = (
    Path("output/validation/production_evaluator_acceleration")
    / "phase3_dwell"
    / "five_second_convergence"
)
_SCENARIOS = (
    ("sphere_20mm_y+22mm", "sphere_20mm.urdf", 20.0, 22.0),
    ("sphere_10mm_y+11mm", "sphere_10mm.urdf", 10.0, 11.0),
    ("sphere_10mm_y+22mm", "sphere_10mm.urdf", 10.0, 22.0),
    ("sphere_15mm_y+0mm", "sphere_15mm.urdf", 15.0, 0.0),
)
_TARGET_FORCES_N = np.asarray((5.0, 10.0, 15.0, 20.0))
_SIM_FREQUENCY_HZ = 100.0
_TIME_STEP_S = 1.0 / _SIM_FREQUENCY_HZ
_INITIAL_CLEARANCE_M = 1.0e-3
_WINDOW_TICKS = 50
_MIN_RELAXATION_WINDOWS = 4
_MAX_RELAXATION_WINDOWS = 25
# Validation-local plateau levels measured by the preceding 10 s Gate A run.
_MAX_FORCE_WINDOW_DRIFT_N = 0.06
_MAX_VERTEX_WINDOW_RMS_M = 5.0e-6
_MAX_PARTICLE_P95_SPEED_M_S = 7.0e-5


def _active_speeds(simulation: LumoSimulation) -> np.ndarray:
    velocities = simulation.state.particle_qd
    flags = simulation.fingertip_model.model.particle_flags
    if velocities is None or flags is None:
        raise RuntimeError("Newton state lacks particle-speed diagnostics")
    start = simulation.fingertip_model.silicone_particle_start
    stop = start + simulation.fingertip_model.silicone_particle_count
    velocity_values = velocities.numpy()[start:stop]
    flag_values = flags.numpy()[start:stop]
    active = (flag_values & int(newton.ParticleFlags.ACTIVE)) != 0
    return np.linalg.norm(velocity_values[active], axis=1)


def _fixed_pose_relax(
    simulation: LumoSimulation,
    indenter: Indenter,
    expected_pose: np.ndarray,
) -> dict[str, float]:
    last: dict[str, float] | None = None
    for window_index in range(1, _MAX_RELAXATION_WINDOWS + 1):
        start_vertices = simulation.silicone_vertices().astype(np.float64)
        forces_n = np.empty(_WINDOW_TICKS, dtype=np.float64)
        for tick in range(_WINDOW_TICKS):
            simulation.step()
            forces_n[tick] = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
            )
        vertices_m = simulation.silicone_vertices().astype(np.float64)
        speeds_m_s = _active_speeds(simulation)
        pose = np.asarray(
            simulation.state.body_q.numpy()[indenter.body_index],
            dtype=np.float64,
        )
        if not np.array_equal(pose, expected_pose):
            raise RuntimeError("indenter pose changed during fixed-pose relaxation")
        vertex_delta = vertices_m - start_vertices
        last = {
            "force_n": float(forces_n[-1]),
            "force_mean_n": float(np.mean(forces_n)),
            "force_drift_n": float(forces_n[-1] - forces_n[0]),
            "force_std_n": float(np.std(forces_n)),
            "vertex_rms_drift_m": float(np.sqrt(np.mean(vertex_delta**2))),
            "vertex_max_drift_m": float(np.max(np.abs(vertex_delta))),
            "particle_rms_speed_m_s": float(
                np.sqrt(np.mean(speeds_m_s**2))
            ),
            "particle_p95_speed_m_s": float(
                np.percentile(speeds_m_s, 95.0)
            ),
            "relaxation_time_s": window_index
            * _WINDOW_TICKS
            * _TIME_STEP_S,
            "relaxation_windows": float(window_index),
        }
        if (
            window_index >= _MIN_RELAXATION_WINDOWS
            and abs(last["force_drift_n"]) <= _MAX_FORCE_WINDOW_DRIFT_N
            and last["vertex_rms_drift_m"] <= _MAX_VERTEX_WINDOW_RMS_M
            and last["particle_p95_speed_m_s"]
            <= _MAX_PARTICLE_P95_SPEED_M_S
        ):
            return last
    if last is None:
        raise RuntimeError("fixed-pose relaxation did not execute")
    raise RuntimeError(
        "fixed-pose relaxation did not reach the Gate-A numerical floor "
        f"within {last['relaxation_time_s']:g} s: force drift="
        f"{last['force_drift_n']:.6g} N, vertex RMS="
        f"{1.0e6 * last['vertex_rms_drift_m']:.3f} um, P95 speed="
        f"{last['particle_p95_speed_m_s']:.6g} m/s"
    )


def _run_scenario(
    fingertip: Fingertip,
    fingertip_mesh: object,
    scenario: tuple[str, str, float, float],
    reference_six_volumes_m3: np.ndarray,
    tet_indices: np.ndarray,
) -> dict[str, object]:
    name, urdf_name, diameter_mm, contact_y_mm = scenario
    radius_m = 0.5e-3 * diameter_mm
    initial_translation_m = np.asarray(
        (
            0.0,
            1.0e-3 * contact_y_mm,
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
        ),
        dtype=np.float64,
    )
    initial_tf = wp.transform(
        wp.vec3(*initial_translation_m),
        wp.quat_identity(),
    )
    resource = files("lumo.assets.objects.urdf").joinpath(urdf_name)
    with as_file(resource) as sphere_path:
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
        indenter = Indenter.add_urdf(
            builder,
            sphere_path,
            tf=initial_tf,
            contact_stiffness_n_m=3.0e4,
            contact_damping_n_s_m=0.28228017516945547,
        )
        simulation = LumoSimulation(
            fingertip,
            builder=builder,
            fingertip_mesh=fingertip_mesh,
            sim_frequency=_SIM_FREQUENCY_HZ,
            iterations=10,
            soft_contact_margin_m=1.0e-4,
            soft_contact_stiffness_n_m=3.0e4,
            soft_contact_damping_n_s_m=0.28228017516945547,
            element_size_mm=1.0,
            carrier_contact_stiffness_n_m=1.0e6,
            use_cuda_graph=True,
        )

        curve_indentations_m: list[float] = []
        curve_forces_n: list[float] = []
        curve_relaxation_times_s: list[float] = []
        checkpoints: list[dict[str, object]] = []
        indentation_m = 0.0
        refinements = 0
        relaxation_blocks = 0
        start_s = perf_counter()
        for target_force_n in _TARGET_FORCES_N:
            for attempt in range(20):
                force_n = curve_forces_n[-1] if curve_forces_n else 0.0
                if len(curve_forces_n) >= 2:
                    delta_force_n = curve_forces_n[-1] - curve_forces_n[-2]
                    delta_indentation_m = (
                        curve_indentations_m[-1] - curve_indentations_m[-2]
                    )
                    slope_n_m = delta_force_n / delta_indentation_m
                else:
                    slope_n_m = 0.0
                if slope_n_m > 0.0:
                    step_m = 0.7 * (target_force_n - force_n) / slope_n_m
                else:
                    step_m = 0.5e-3
                step_m = float(np.clip(step_m, 20.0e-6, 0.75e-3))
                indentation_m += step_m
                translation_m = initial_translation_m + np.asarray(
                    (0.0, 0.0, _INITIAL_CLEARANCE_M + indentation_m)
                )
                pose = wp.transform(
                    wp.vec3(*translation_m),
                    wp.quat_identity(),
                )
                simulation.apply_indenter_pose(indenter, pose)
                expected_pose = np.asarray(pose, dtype=np.float64)
                relaxation = _fixed_pose_relax(
                    simulation,
                    indenter,
                    expected_pose,
                )
                relaxation_blocks += int(relaxation["relaxation_windows"])
                force_n = relaxation["force_mean_n"]
                curve_indentations_m.append(indentation_m)
                curve_forces_n.append(force_n)
                curve_relaxation_times_s.append(relaxation["relaxation_time_s"])
                upper_n = 1.1 * target_force_n
                preferred_lower_n = target_force_n - min(
                    0.02 * target_force_n,
                    0.2,
                )
                if force_n > upper_n:
                    raise RuntimeError(
                        f"{name} overshot {target_force_n:g} N outside its "
                        f"+/-10% band at {force_n:.6f} N"
                    )
                if force_n < preferred_lower_n:
                    refinements += 1
                    continue

                vertices_m = simulation.silicone_vertices().astype(np.float64)
                records = _indenter_contact_records(
                    simulation,
                    indenter,
                    vertices_m,
                )
                if len(records[0]) == 0:
                    raise RuntimeError(f"{name} target has no indenter contacts")
                det_f = (
                    _six_tet_volumes(vertices_m, tet_indices)
                    / reference_six_volumes_m3
                )
                overflow = int(
                    simulation.solver.body_particle_contact_overflow_max.numpy()[0]
                )
                checkpoints.append(
                    {
                        "target_force_n": target_force_n,
                        "actual_force_n": force_n,
                        "indentation_m": indentation_m,
                        "vertices_m": vertices_m,
                        "contact_indices": records[0],
                        "contact_barycentric": records[1],
                        "contact_positions_m": records[2],
                        "contact_normals": records[3],
                        "contact_body_positions": records[4],
                        "contact_count": len(records[0]),
                        "minimum_det_f": float(np.min(det_f)),
                        "inverted_tet_count": int(np.count_nonzero(det_f <= 0.0)),
                        "contact_buffer_overflow": overflow,
                        **relaxation,
                    }
                )
                break
            else:
                raise RuntimeError(f"{name} exhausted target refinements")

        exceed_attempts = 0
        while curve_forces_n[-1] <= 20.0:
            indentation_m += 0.1e-3
            translation_m = initial_translation_m + np.asarray(
                (0.0, 0.0, _INITIAL_CLEARANCE_M + indentation_m)
            )
            pose = wp.transform(wp.vec3(*translation_m), wp.quat_identity())
            simulation.apply_indenter_pose(indenter, pose)
            relaxation = _fixed_pose_relax(
                simulation,
                indenter,
                np.asarray(pose, dtype=np.float64),
            )
            relaxation_blocks += int(relaxation["relaxation_windows"])
            curve_indentations_m.append(indentation_m)
            curve_forces_n.append(relaxation["force_mean_n"])
            curve_relaxation_times_s.append(relaxation["relaxation_time_s"])
            exceed_attempts += 1
            if exceed_attempts >= 10:
                raise RuntimeError(f"{name} could not exceed 20 N")
        wall_s = perf_counter() - start_s

    curve_forces = np.asarray(curve_forces_n)
    force_drops = np.minimum(np.diff(curve_forces), 0.0)
    return {
        "name": name,
        "diameter_mm": diameter_mm,
        "contact_y_mm": contact_y_mm,
        "checkpoints": checkpoints,
        "curve_indentations_m": np.asarray(curve_indentations_m),
        "curve_forces_n": curve_forces,
        "curve_relaxation_times_s": np.asarray(curve_relaxation_times_s),
        "maximum_force_drop_n": float(-np.min(force_drops, initial=0.0)),
        "strictly_monotonic_force": bool(np.all(np.diff(curve_forces) > 0.0)),
        "refinements": refinements,
        "relaxation_blocks": relaxation_blocks,
        "wall_s": wall_s,
    }


def _flatten_contacts(
    scenarios: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.empty((len(scenarios), 4, 2), dtype=np.int64)
    index_chunks = []
    normal_chunks = []
    count = 0
    for scenario_index, scenario in enumerate(scenarios):
        for force_index, checkpoint in enumerate(scenario["checkpoints"]):
            indices = checkpoint["contact_indices"]
            normals = checkpoint["contact_normals"]
            offsets[scenario_index, force_index] = (count, len(indices))
            index_chunks.append(indices)
            normal_chunks.append(normals)
            count += len(indices)
    return offsets, np.concatenate(index_chunks), np.concatenate(normal_chunks)


def _load_reference() -> dict[str, object]:
    selections = (
        ("dwell_5p0_contact_limiter.npz", (0,)),
        ("dwell_5p0_observation_pair.npz", (0, 1)),
        ("dwell_5p0_interior.npz", (0,)),
    )
    state_fields = (
        "scenario_names",
        "sphere_diameters_mm",
        "contact_y_mm",
        "actual_forces_n",
        "indentations_m",
        "silicone_vertices_m",
        "response_matrix",
        "energy_matrix",
        "inside_roi_power",
        "outside_roi_power",
        "visible_side_power",
        "minimum_det_f",
        "inverted_tet_counts",
        "contact_buffer_overflow",
        "scenario_runtime_s",
    )
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in state_fields}
    contact_indices = []
    contact_normals = []
    offsets = np.empty((len(_SCENARIOS), 4, 2), dtype=np.int64)
    contact_count = 0
    destination = 0
    shared: dict[str, np.ndarray] = {}
    for filename, selected_indices in selections:
        with np.load(_REFERENCE_DIRECTORY / filename) as saved:
            if not shared:
                for name in (
                    "reference_vertices_m",
                    "surface_triangles",
                    "force_targets_n",
                    "no_contact_response",
                    "no_contact_energy",
                    "energy_fields",
                ):
                    shared[name] = np.asarray(saved[name])
            for source_index in selected_indices:
                for field in state_fields:
                    chunks[field].append(np.asarray(saved[field][source_index]))
                for force_index in range(4):
                    start, count = saved["contact_record_offsets"][
                        source_index, force_index
                    ]
                    indices = np.asarray(
                        saved["contact_particle_indices"][start : start + count]
                    )
                    normals = np.asarray(
                        saved["contact_normals_W"][start : start + count]
                    )
                    offsets[destination, force_index] = (contact_count, count)
                    contact_indices.append(indices)
                    contact_normals.append(normals)
                    contact_count += count
                destination += 1
    return {
        **shared,
        **{field: np.asarray(values) for field, values in chunks.items()},
        "contact_record_offsets": offsets,
        "contact_particle_indices": np.concatenate(contact_indices),
        "contact_normals_W": np.concatenate(contact_normals),
    }


def _trace_optics(
    fingertip: Fingertip,
    fingertip_mesh: object,
    scenarios: list[dict[str, object]],
) -> dict[str, np.ndarray | float]:
    scene = OptixScene(
        fingertip_mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    leds = _make_full_finger_leds(fingertip, fingertip_mesh)
    emissions = _full_finger_emissions(scene, leds)
    dielectric_u, carrier_u1, carrier_u2 = _full_finger_optical_samples(
        len(emissions[0])
    )
    no_contact = _trace_full_finger_state(
        scene,
        fingertip,
        leds,
        emissions,
        dielectric_branch_u=dielectric_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        require_air_sources=True,
    )
    response = np.empty((len(scenarios), 4, 5, 11), dtype=np.float64)
    energy = np.empty((len(scenarios), 4, 5, len(_ENERGY_FIELDS)))
    outside = np.empty((len(scenarios), 4, 5), dtype=np.float64)
    visible = np.empty_like(outside)
    start_s = perf_counter()
    for scenario_index, scenario in enumerate(scenarios):
        for force_index, checkpoint in enumerate(scenario["checkpoints"]):
            scene.update_silicone(checkpoint["vertices_m"])
            (
                response[scenario_index, force_index],
                energy[scenario_index, force_index],
                outside[scenario_index, force_index],
                visible[scenario_index, force_index],
            ) = _trace_full_finger_state(
                scene,
                fingertip,
                leds,
                emissions,
                dielectric_branch_u=dielectric_u,
                carrier_u1=carrier_u1,
                carrier_u2=carrier_u2,
            )
    return {
        "no_contact_response": no_contact[0],
        "no_contact_energy": no_contact[1],
        "response_matrix": response,
        "energy_matrix": energy,
        "outside_roi_power": outside,
        "visible_side_power": visible,
        "wall_s": perf_counter() - start_s,
    }


def _contact_objective_inputs(
    fingertip_mesh: object,
    scenarios: list[dict[str, object]],
) -> dict[str, object]:
    offsets, indices, normals = _flatten_contacts(scenarios)
    return {
        "reference_vertices_m": np.asarray(fingertip_mesh.silicone.vertices),
        "surface_triangles": np.asarray(
            fingertip_mesh.silicone.surface_tri_indices
        ).reshape(-1, 3),
        "scenario_names": tuple(scenario["name"] for scenario in scenarios),
        "sphere_diameters_mm": np.asarray(
            [scenario["diameter_mm"] for scenario in scenarios]
        ),
        "force_targets_n": _TARGET_FORCES_N,
        "actual_forces_n": np.asarray(
            [
                [point["actual_force_n"] for point in scenario["checkpoints"]]
                for scenario in scenarios
            ]
        ),
        "indentations_m": np.asarray(
            [
                [point["indentation_m"] for point in scenario["checkpoints"]]
                for scenario in scenarios
            ]
        ),
        "contact_record_offsets": offsets,
        "contact_particle_indices": indices,
        "contact_normals_W": normals,
        "silicone_vertices_m": np.asarray(
            [
                [point["vertices_m"] for point in scenario["checkpoints"]]
                for scenario in scenarios
            ]
        ),
    }


def _difficult_separation(
    response: np.ndarray,
    no_contact_response: np.ndarray,
) -> float:
    normalized = (
        response.sum(axis=2) - no_contact_response.sum(axis=0)[None, None, :]
    ) / 5.0
    return float(np.linalg.norm(normalized[1, 0] - normalized[2, 0]))


def _relative(candidate: np.ndarray | float, reference: np.ndarray | float) -> float:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    scale = np.maximum(np.abs(reference_array), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(candidate_array - reference_array) / scale))


def _save_raw(
    scenarios: list[dict[str, object]],
    contact_inputs: dict[str, object],
    optics: dict[str, np.ndarray | float],
) -> None:
    offsets, indices, normals = _flatten_contacts(scenarios)
    checkpoints = [
        checkpoint
        for scenario in scenarios
        for checkpoint in scenario["checkpoints"]
    ]
    curve_point_counts = np.asarray(
        [len(scenario["curve_forces_n"]) for scenario in scenarios],
        dtype=np.int64,
    )
    maximum_curve_points = int(curve_point_counts.max())
    curve_indentations_m = np.full(
        (len(scenarios), maximum_curve_points),
        np.nan,
        dtype=np.float64,
    )
    curve_forces_n = np.full_like(curve_indentations_m, np.nan)
    curve_relaxation_times_s = np.full_like(curve_indentations_m, np.nan)
    for scenario_index, scenario in enumerate(scenarios):
        count = curve_point_counts[scenario_index]
        curve_indentations_m[scenario_index, :count] = scenario[
            "curve_indentations_m"
        ]
        curve_forces_n[scenario_index, :count] = scenario["curve_forces_n"]
        curve_relaxation_times_s[scenario_index, :count] = scenario[
            "curve_relaxation_times_s"
        ]
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "indentation_controlled_sentinels.npz",
        scenario_names=np.asarray([scenario["name"] for scenario in scenarios]),
        sphere_diameters_mm=np.asarray(
            [scenario["diameter_mm"] for scenario in scenarios]
        ),
        contact_y_mm=np.asarray([scenario["contact_y_mm"] for scenario in scenarios]),
        force_targets_n=_TARGET_FORCES_N,
        actual_forces_n=contact_inputs["actual_forces_n"],
        indentations_m=contact_inputs["indentations_m"],
        silicone_vertices_m=np.asarray(
            contact_inputs["silicone_vertices_m"],
            dtype=np.float32,
        ),
        reference_vertices_m=contact_inputs["reference_vertices_m"],
        surface_triangles=contact_inputs["surface_triangles"],
        contact_record_offsets=offsets,
        contact_particle_indices=indices,
        contact_barycentric=np.concatenate(
            [checkpoint["contact_barycentric"] for checkpoint in checkpoints]
        ),
        contact_positions_W_m=np.concatenate(
            [checkpoint["contact_positions_m"] for checkpoint in checkpoints]
        ),
        contact_normals_W=normals,
        contact_body_positions=np.concatenate(
            [checkpoint["contact_body_positions"] for checkpoint in checkpoints]
        ),
        response_matrix=optics["response_matrix"],
        no_contact_response=optics["no_contact_response"],
        energy_matrix=optics["energy_matrix"],
        no_contact_energy=optics["no_contact_energy"],
        energy_fields=np.asarray(_ENERGY_FIELDS),
        outside_roi_power=optics["outside_roi_power"],
        visible_side_power=optics["visible_side_power"],
        curve_point_counts=curve_point_counts,
        curve_indentations_m=curve_indentations_m,
        curve_forces_n=curve_forces_n,
        curve_relaxation_times_s=curve_relaxation_times_s,
        scenario_wall_s=np.asarray([scenario["wall_s"] for scenario in scenarios]),
    )


def _write_report(
    scenarios: list[dict[str, object]],
    candidate_contact: object,
    reference: dict[str, object],
    reference_contact: object,
    optics: dict[str, np.ndarray | float],
) -> None:
    candidate_response = np.asarray(optics["response_matrix"])
    reference_response = np.asarray(reference["response_matrix"])
    candidate_separation = _difficult_separation(
        candidate_response,
        np.asarray(optics["no_contact_response"]),
    )
    reference_separation = _difficult_separation(
        reference_response,
        np.asarray(reference["no_contact_response"]),
    )
    indentation_difference_m = np.asarray(
        [
            [point["indentation_m"] for point in scenario["checkpoints"]]
            for scenario in scenarios
        ]
    ) - np.asarray(reference["indentations_m"])
    response_delta = candidate_response - reference_response
    energy_fields = tuple(str(value) for value in reference["energy_fields"])
    closure_index = energy_fields.index("closure_error")
    max_closure = float(
        np.max(np.abs(np.asarray(optics["energy_matrix"])[..., closure_index]))
    )
    q_errors = {
        "q_form": _relative(candidate_contact.q_form, reference_contact.q_form),
        "q_stable": _relative(
            candidate_contact.q_stable, reference_contact.q_stable
        ),
        "q_stiff": _relative(candidate_contact.q_stiff, reference_contact.q_stiff),
        "q_contact": _relative(
            candidate_contact.q_contact, reference_contact.q_contact
        ),
    }
    safety_pass = all(
        checkpoint["inverted_tet_count"] == 0
        and checkpoint["contact_buffer_overflow"] == 0
        for scenario in scenarios
        for checkpoint in scenario["checkpoints"]
    )
    force_pass = all(
        abs(checkpoint["actual_force_n"] - checkpoint["target_force_n"])
        <= 0.1 * checkpoint["target_force_n"]
        for scenario in scenarios
        for checkpoint in scenario["checkpoints"]
    )
    monotonic_pass = all(
        scenario["strictly_monotonic_force"] for scenario in scenarios
    )
    optical_pass = (
        abs(candidate_separation - reference_separation) / reference_separation
        <= 0.02
    )
    component_pass = (
        q_errors["q_form"] <= 0.05
        and q_errors["q_stable"] <= 0.05
        and q_errors["q_stiff"] <= 0.05
        and q_errors["q_contact"] <= 0.02
    )
    recommend_full_grid = (
        safety_pass
        and force_pass
        and monotonic_pass
        and optical_pass
        and component_pass
    )
    mechanics_wall_s = sum(float(scenario["wall_s"]) for scenario in scenarios)
    reference_wall_s = float(np.sum(reference["scenario_runtime_s"]))

    with (_OUTPUT_DIRECTORY / "force_state_summary.csv").open(
        "w", newline=""
    ) as stream:
        fieldnames = (
            "scenario",
            "target_force_n",
            "actual_force_n",
            "indentation_mm",
            "reference_indentation_mm",
            "relaxation_time_s",
            "force_window_drift_n",
            "vertex_window_rms_um",
            "particle_p95_speed_m_s",
            "contact_count",
            "minimum_det_f",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_index, scenario in enumerate(scenarios):
            for force_index, checkpoint in enumerate(scenario["checkpoints"]):
                writer.writerow(
                    {
                        "scenario": scenario["name"],
                        "target_force_n": checkpoint["target_force_n"],
                        "actual_force_n": checkpoint["actual_force_n"],
                        "indentation_mm": 1.0e3 * checkpoint["indentation_m"],
                        "reference_indentation_mm": 1.0e3
                        * reference["indentations_m"][scenario_index, force_index],
                        "relaxation_time_s": checkpoint["relaxation_time_s"],
                        "force_window_drift_n": checkpoint["force_drift_n"],
                        "vertex_window_rms_um": 1.0e6
                        * checkpoint["vertex_rms_drift_m"],
                        "particle_p95_speed_m_s": checkpoint[
                            "particle_p95_speed_m_s"
                        ],
                        "contact_count": checkpoint["contact_count"],
                        "minimum_det_f": checkpoint["minimum_det_f"],
                    }
                )

    plt.figure(figsize=(7.0, 4.5))
    for scenario in scenarios:
        plt.plot(
            1.0e3 * scenario["curve_indentations_m"],
            scenario["curve_forces_n"],
            marker="o",
            label=scenario["name"],
        )
    plt.xlabel("fixed indentation [mm]")
    plt.ylabel("relaxed reaction force [N]")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIRECTORY / "force_vs_indentation_relaxed.png", dpi=180)
    plt.close()

    x = np.arange(len(scenarios))
    width = 0.19
    plt.figure(figsize=(8.0, 4.5))
    for offset, (name, candidate, reference_values) in enumerate(
        (
            ("q_form", candidate_contact.q_form, reference_contact.q_form),
            ("q_stable", candidate_contact.q_stable, reference_contact.q_stable),
            ("q_stiff", candidate_contact.q_stiff, reference_contact.q_stiff),
            ("q_contact", candidate_contact.q_contact, reference_contact.q_contact),
        )
    ):
        plt.bar(x + (offset - 1.5) * width, candidate, width, label=name)
        plt.scatter(
            x + (offset - 1.5) * width,
            reference_values,
            color="black",
            marker="_",
            zorder=3,
        )
    plt.xticks(x, [scenario["name"] for scenario in scenarios], rotation=15)
    plt.ylabel("contact component")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIRECTORY / "q_components_by_protocol.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 4.0))
    plt.bar(
        ("force servo", "fixed indentation"),
        (reference_separation, candidate_separation),
    )
    plt.ylabel("10 mm, 5 N, Y=+11 vs +22 separation")
    plt.tight_layout()
    plt.savefig(
        _OUTPUT_DIRECTORY / "difficult_pair_optical_separation.png",
        dpi=180,
    )
    plt.close()

    lines = [
        "# Post-Phase 4 indentation-controlled quasi-static loading",
        "",
        "## Gate A summary",
        "",
        "The first-crossing 10 N trigger followed by an exactly fixed 10 s "
        "pose hold passed. Force, particle motion, and geometry drift decayed; "
        "patch IoU remained 1.0; no inversion or overflow occurred. Full Gate A "
        "details remain in `gate_a_report.md`.",
        "",
        "## Relaxed force-state extraction",
        "",
        "Each listed checkpoint is an actually simulated fixed-pose state. No "
        "geometry or optical response was interpolated, and force feedback was "
        "disabled during every relaxation window.",
        "",
        "| scenario | F actual [N] | indentation [mm] | reference indentation [mm] | relaxation [s] | min det(F) |",
        "|---|---|---|---|---|---|",
    ]
    for scenario_index, scenario in enumerate(scenarios):
        lines.append(
            f"| {scenario['name']} | "
            f"{[round(point['actual_force_n'], 4) for point in scenario['checkpoints']]} | "
            f"{[round(1.0e3 * point['indentation_m'], 4) for point in scenario['checkpoints']]} | "
            f"{np.round(1.0e3 * reference['indentations_m'][scenario_index], 4).tolist()} | "
            f"{[round(point['relaxation_time_s'], 2) for point in scenario['checkpoints']]} | "
            f"{min(point['minimum_det_f'] for point in scenario['checkpoints']):.6f} |"
        )
    lines.extend(
        (
            "",
            "## Force-indentation behavior",
            "",
        )
    )
    for scenario in scenarios:
        lines.append(
            f"- {scenario['name']}: strict monotonic={scenario['strictly_monotonic_force']}, "
            f"maximum relaxed one-step force drop={scenario['maximum_force_drop_n']:.6f} N, "
            f"refinements={scenario['refinements']}, relaxation windows="
            f"{scenario['relaxation_blocks']}"
        )
    lines.extend(
        (
            "",
            "## Contact comparison",
            "",
            "| component | maximum relative difference |",
            "|---|---:|",
            f"| q_form | {100.0 * q_errors['q_form']:.3f}% |",
            f"| q_stable | {100.0 * q_errors['q_stable']:.3f}% |",
            f"| q_stiff | {100.0 * q_errors['q_stiff']:.3f}% |",
            f"| q_contact | {100.0 * q_errors['q_contact']:.3f}% |",
            f"| sentinel J_contact | {candidate_contact.J_contact:.9f} vs reference {reference_contact.J_contact:.9f} |",
            "",
            "## Optical comparison",
            "",
            f"- response RMS / maximum bin difference: {float(np.sqrt(np.mean(response_delta**2))):.9f} / {float(np.max(np.abs(response_delta))):.9f}",
            f"- difficult-pair separation: {candidate_separation:.9f} vs reference {reference_separation:.9f} ({100.0 * abs(candidate_separation-reference_separation)/reference_separation:.3f}% difference)",
            f"- maximum energy closure error: {max_closure:.3e}",
            "",
            "## Runtime",
            "",
            f"- force-servo sentinel scenario runtime sum: {reference_wall_s:.3f} s",
            f"- indentation-controlled mechanics wall: {mechanics_wall_s:.3f} s",
            f"- indentation-controlled OptiX wall: {float(optics['wall_s']):.3f} s",
            f"- indentation refinements: {sum(int(s['refinements']) for s in scenarios)}",
            f"- relaxation blocks: {sum(int(s['relaxation_blocks']) for s in scenarios)}",
            "",
            "## Explicit answers",
            "",
            "1. Fixed-pose relaxation converges more cleanly: YES for Gate A.",
            "2. Force drift decays at fixed pose: YES.",
            "3. Geometry and contact support drift decay: YES; Gate A patch IoU was 1.0.",
            f"4. F(delta) monotonic and smooth enough: {'YES' if monotonic_pass else 'NO'}.",
            f"5. Valid states near 5/10/15/20 N without servo: {'YES' if force_pass else 'NO'}.",
            f"6. Maximum indentation difference from servo: {1.0e6 * float(np.max(np.abs(indentation_difference_m))):.3f} um.",
            f"7. q components preserve the reference gate: {'YES' if component_pass else 'NO'}.",
            "8. q_stiff becomes more stable: NOT ESTABLISHED by one extraction repeat; its reference difference is reported above.",
            f"9. Finite-area optical response consistent: {'YES' if optical_pass else 'NO'}.",
            f"10. Difficult-pair separation consistent: {'YES' if optical_pass else 'NO'}.",
            "11. A convergence condition is more defensible than fixed 5 s: PROMISING, but the validation-local floor still requires repeatability validation.",
            f"12. Runtime impact: {mechanics_wall_s + float(optics['wall_s']):.3f} s for four sentinels versus {reference_wall_s:.3f} s reference scenario time.",
            f"13. Recommend full 21-scenario validation: {'YES' if recommend_full_grid else 'NO'}.",
            "14. Recommend changing production loading now: NO; the full-grid and repeatability gate has not been approved or run.",
            "15. No production BO campaign was started and production loading was not replaced: CONFIRMED.",
            "",
            f"Overall sentinel gate: {'PASS' if recommend_full_grid else 'FAIL'}.",
        )
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    gate_a_report = _OUTPUT_DIRECTORY / "gate_a_report.md"
    if not gate_a_report.is_file():
        raise FileNotFoundError(
            "run post_phase4_indentation_control.py and pass Gate A before "
            "extracting fixed-indentation force states"
        )
    reference = _load_reference()
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    reference_vertices_m = np.asarray(fingertip_mesh.silicone.vertices)
    tet_indices = np.asarray(fingertip_mesh.silicone.tet_indices).reshape(-1, 4)
    reference_six_volumes_m3 = _six_tet_volumes(
        reference_vertices_m,
        tet_indices,
    )
    scenarios = []
    for scenario in _SCENARIOS:
        print(f"running {scenario[0]}", flush=True)
        scenarios.append(
            _run_scenario(
                fingertip,
                fingertip_mesh,
                scenario,
                reference_six_volumes_m3,
                tet_indices,
            )
        )
    contact_inputs = _contact_objective_inputs(fingertip_mesh, scenarios)
    candidate_contact = compute_contact_objective(**contact_inputs)
    reference_contact = compute_contact_objective(
        reference_vertices_m=reference["reference_vertices_m"],
        surface_triangles=reference["surface_triangles"],
        scenario_names=tuple(str(value) for value in reference["scenario_names"]),
        sphere_diameters_mm=reference["sphere_diameters_mm"],
        force_targets_n=reference["force_targets_n"],
        actual_forces_n=reference["actual_forces_n"],
        indentations_m=reference["indentations_m"],
        contact_record_offsets=reference["contact_record_offsets"],
        contact_particle_indices=reference["contact_particle_indices"],
        contact_normals_W=reference["contact_normals_W"],
        silicone_vertices_m=reference["silicone_vertices_m"],
    )
    print("tracing fixed-indentation checkpoints", flush=True)
    optics = _trace_optics(fingertip, fingertip_mesh, scenarios)
    _save_raw(scenarios, contact_inputs, optics)
    _write_report(
        scenarios,
        candidate_contact,
        reference,
        reference_contact,
        optics,
    )
    print((_OUTPUT_DIRECTORY / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
