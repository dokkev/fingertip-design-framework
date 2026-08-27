"""One concrete Newton-to-OptiX sensing evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import newton
import numpy as np
import warp as wp

from lumo.fingertip import ACTIVE_Y_BOUNDS_MM, Fingertip
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton import Indenter
from lumo.ray_tracing import (
    LED,
    LONGITUDINAL_SIDE_BIN_COUNT,
    OptixScene,
    PathTraceResult,
    emit_from_stem_window,
    longitudinal_side_view_power,
    sources_inside_silicone,
    trace_bounded_paths,
)
from lumo.simulation import IndentationStudy, IndentationTrial, LumoSimulation


_SAMPLE_SIDE_COUNT = 256
_MAX_BOUNCES = 24
_RNG_SEED = 20260823
_SOURCE_RNG_SEED = 20260826
_CARRIER_ALBEDO = 0.7

_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547
_ELEMENT_SIZE_MM = 1.0
_SOFT_CONTACT_MARGIN_M = 1.0e-4
_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_ENERGY_FIELDS = (
    "emitted_power",
    "escaped_power",
    "carrier_absorbed_power",
    "bulk_loss_power",
    "unresolved_internal_miss_power",
    "remaining_power",
    "accounted_power",
    "closure_error",
)


@dataclass(frozen=True)
class FingertipEvaluation:
    """Raw mechanics and per-emitter optics for one fingertip."""

    reference_vertices_m: np.ndarray
    tet_indices: np.ndarray
    surface_triangles: np.ndarray
    bonded_vertex_indices: np.ndarray
    led_source_centers_m: np.ndarray
    no_contact_response: np.ndarray
    no_contact_energy: np.ndarray
    no_contact_inside_roi_power: np.ndarray
    no_contact_outside_roi_power: np.ndarray
    no_contact_visible_side_power: np.ndarray
    no_contact_outside_roi_power_fraction: float
    scenario_names: tuple[str, ...]
    sphere_diameters_mm: np.ndarray
    contact_y_mm: np.ndarray
    force_targets_n: np.ndarray
    actual_forces_n: np.ndarray
    indentations_m: np.ndarray
    checkpoint_steps: np.ndarray
    checkpoint_times_s: np.ndarray
    maximum_particle_speeds_m_s: np.ndarray
    mean_particle_speeds_m_s: np.ndarray
    rms_particle_speeds_m_s: np.ndarray
    particle_speed_p95_m_s: np.ndarray
    kinetic_energy_j: np.ndarray
    force_overshoots_n: np.ndarray
    reaction_force_rates_n_s: np.ndarray
    indentation_rates_m_s: np.ndarray
    indenter_contact_counts: np.ndarray
    total_contact_counts: np.ndarray
    contact_buffer_overflow: np.ndarray
    minimum_det_f: np.ndarray
    inverted_tet_counts: np.ndarray
    contact_centroids_W_m: np.ndarray
    contact_record_offsets: np.ndarray
    contact_particle_indices: np.ndarray
    contact_barycentric: np.ndarray
    contact_positions_W_m: np.ndarray
    contact_normals_W: np.ndarray
    contact_body_positions: np.ndarray
    silicone_vertices_m: np.ndarray
    response_matrix: np.ndarray
    energy_fields: tuple[str, ...]
    energy_matrix: np.ndarray
    inside_roi_power: np.ndarray
    outside_roi_power: np.ndarray
    visible_side_power: np.ndarray
    outside_roi_power_fraction: np.ndarray
    scenario_runtime_s: np.ndarray
    checkpoint_optics_runtime_s: np.ndarray
    no_contact_optics_runtime_s: float

    @property
    def combined_no_contact_response(self) -> np.ndarray:
        """Return the simultaneous five-emitter no-contact spatial response."""
        return self.no_contact_response.sum(axis=0)

    @property
    def combined_response_matrix(self) -> np.ndarray:
        """Return simultaneous checkpoint responses without storing a copy."""
        return self.response_matrix.sum(axis=2)


def _trace_paths(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    *,
    inside_silicone: bool | np.ndarray,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
) -> PathTraceResult:
    optics = fingertip.parameters.optics
    return trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=inside_silicone,
        n_air=1.0,
        n_silicone=optics.refractive_index,
        extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_MAX_BOUNCES,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
    )


def _path_energy(paths: PathTraceResult) -> np.ndarray:
    return np.array(
        (
            paths.emitted_power,
            paths.escaped_power,
            paths.absorbed_power,
            paths.bulk_loss_power,
            paths.unresolved_internal_miss_power,
            paths.remaining_power,
            paths.accounted_power,
            paths.closure_error,
        ),
        dtype=np.float64,
    )


def _make_leds(fingertip: Fingertip) -> tuple[LED, ...]:
    normal_W = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    return tuple(
        LED(
            position_W_m=center_m,
            normal_W=normal_W,
            parameters=fingertip.parameters.led,
        )
        for center_m in fingertip.led_source_centers_m
    )


def _optical_samples(
    ray_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_RNG_SEED)
    sample_shape = (_MAX_BOUNCES, ray_count)
    return (
        rng.random(sample_shape),
        rng.random(sample_shape),
        rng.random(sample_shape),
    )


def _emissions(
    scene: OptixScene,
    leds: tuple[LED, ...],
) -> tuple[np.ndarray, ...]:
    """Emit the production finite-area deterministic path set."""
    coordinate = (
        np.arange(_SAMPLE_SIDE_COUNT, dtype=np.float64) + 0.5
    ) / _SAMPLE_SIDE_COUNT
    angular_u1, angular_u2 = np.meshgrid(
        coordinate,
        coordinate,
        indexing="ij",
    )
    angular_u1 = angular_u1.ravel()
    angular_u2 = angular_u2.ravel()
    source_coordinate = (np.arange(len(angular_u1), dtype=np.float64) + 0.5) / len(
        angular_u1
    )
    source_rng = np.random.default_rng(_SOURCE_RNG_SEED)
    source_u_x = source_coordinate[source_rng.permutation(len(source_coordinate))]
    source_u_y = source_coordinate[source_rng.permutation(len(source_coordinate))]
    return tuple(
        emit_from_stem_window(
            scene,
            led,
            angular_u1,
            angular_u2,
            source_u_x,
            source_u_y,
        )
        for led in leds
    )


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    leds: tuple[LED, ...],
    emissions: tuple[np.ndarray, ...],
    *,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
    require_air_sources: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    responses = np.empty(
        (len(leds), LONGITUDINAL_SIDE_BIN_COUNT),
        dtype=np.float64,
    )
    energies = np.empty((len(leds), len(_ENERGY_FIELDS)), dtype=np.float64)
    outside_roi_power = np.empty(len(leds), dtype=np.float64)
    visible_side_power = np.empty(len(leds), dtype=np.float64)
    for led_index, (led, emission) in enumerate(zip(leds, emissions, strict=True)):
        inside_silicone = sources_inside_silicone(
            scene,
            led,
            emission,
        )
        if require_air_sources and np.any(inside_silicone):
            raise RuntimeError(
                f"unloaded LED {led_index + 1} is not inside its air recess"
            )
        paths = _trace_paths(
            scene,
            fingertip,
            emission,
            inside_silicone=inside_silicone,
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
        )
        (
            responses[led_index],
            outside_roi_power[led_index],
            visible_side_power[led_index],
        ) = longitudinal_side_view_power(
            paths.escaped_rays,
        )
        energies[led_index] = _path_energy(paths)
        del paths
    if not np.allclose(
        responses.sum(axis=1) + outside_roi_power,
        visible_side_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("longitudinal ROI accounting does not close")
    return responses, energies, outside_roi_power, visible_side_power


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contacts = simulation.contacts
    emitted_count = int(contacts.soft_contact_count.numpy()[0])
    stored_count = min(emitted_count, int(contacts.soft_contact_max))
    shapes = contacts.soft_contact_shape.numpy()[:stored_count]
    valid = shapes >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    indenter_records = valid.copy()
    indenter_records[valid] = shape_bodies[shapes[valid]] == indenter.body_index

    particle_indices = np.asarray(
        contacts.soft_contact_indices.numpy()[:stored_count][indenter_records],
        dtype=np.int32,
    )
    barycentric = np.asarray(
        contacts.soft_contact_barycentric.numpy()[:stored_count][indenter_records],
        dtype=np.float64,
    )
    normals_W = np.asarray(
        contacts.soft_contact_normal.numpy()[:stored_count][indenter_records],
        dtype=np.float64,
    )
    body_positions = np.asarray(
        contacts.soft_contact_body_pos.numpy()[:stored_count][indenter_records],
        dtype=np.float64,
    )

    local_indices = particle_indices.copy()
    present = local_indices >= 0
    local_indices[present] -= simulation.fingertip_model.silicone_particle_start
    if np.any(local_indices[present] < 0) or np.any(
        local_indices[present] >= len(silicone_vertices_m)
    ):
        raise RuntimeError("indenter contact references a non-silicone particle")

    points_W_m = np.empty((len(local_indices), 3), dtype=np.float64)
    for record_index, (indices, weights) in enumerate(
        zip(local_indices, barycentric, strict=True)
    ):
        record_present = indices >= 0
        points_W_m[record_index] = np.sum(
            silicone_vertices_m[indices[record_present]]
            * weights[record_present, None],
            axis=0,
        )
    return local_indices, barycentric, points_W_m, normals_W, body_positions


def evaluate_fingertip(
    fingertip: Fingertip,
    sphere_urdf_paths: Iterable[str | Path],
    sphere_diameters_mm: Iterable[float],
    contact_y_mm: Iterable[float],
    *,
    force_targets_n: Iterable[float] = _FORCE_TARGETS_N,
    initial_clearance_m: float = 1.0e-3,
    approach_speed_m_s: float = 5.0e-3,
    max_sim_time_s: float = 60.0,
) -> FingertipEvaluation:
    """Evaluate raw mechanics and five-emitter optics for one morphology.

    Sphere diameters and URDF paths are paired. Their Cartesian product with
    ``contact_y_mm`` defines independent Newton scenarios. Increasing force
    targets reuse one runtime within each scenario; different longitudinal
    contact locations never share a runtime.
    """
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")
    urdf_paths = tuple(Path(path) for path in sphere_urdf_paths)
    diameters_mm = tuple(float(value) for value in sphere_diameters_mm)
    locations_y_mm = tuple(float(value) for value in contact_y_mm)
    if not urdf_paths or len(urdf_paths) != len(diameters_mm):
        raise ValueError(
            "sphere_urdf_paths and sphere_diameters_mm must be nonempty and paired"
        )
    if any(path.suffix.lower() != ".urdf" or not path.is_file() for path in urdf_paths):
        raise ValueError("every sphere path must identify an existing URDF file")
    if any(not np.isfinite(value) or value <= 0.0 for value in diameters_mm):
        raise ValueError("sphere diameters must be finite and positive")
    if len(set(diameters_mm)) != len(diameters_mm):
        raise ValueError("sphere diameters must be unique")
    if not locations_y_mm or any(not np.isfinite(value) for value in locations_y_mm):
        raise ValueError("contact_y_mm must contain finite locations")
    if len(set(locations_y_mm)) != len(locations_y_mm):
        raise ValueError("contact_y_mm must be unique")

    force_targets = tuple(float(target) for target in force_targets_n)
    if len(force_targets) < 3 or any(
        not np.isfinite(target) or target <= 0.0 for target in force_targets
    ):
        raise ValueError("force_targets_n must contain at least three positive values")
    if any(
        current <= previous
        for previous, current in zip(force_targets, force_targets[1:])
    ):
        raise ValueError("force_targets_n must be strictly increasing")
    for name, value in (
        ("initial_clearance_m", initial_clearance_m),
        ("approach_speed_m_s", approach_speed_m_s),
        ("max_sim_time_s", max_sim_time_s),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    fingertip_mesh = make_fingertip_mesh(
        fingertip,
        element_size_mm=_ELEMENT_SIZE_MM,
    )
    active_y_min_mm, active_y_max_mm = ACTIVE_Y_BOUNDS_MM
    if any(
        value < active_y_min_mm or value > active_y_max_mm for value in locations_y_mm
    ):
        raise ValueError("contact_y_mm must lie inside the 55 mm active section")

    reference_vertices_m = np.ascontiguousarray(
        fingertip_mesh.silicone.vertices,
        dtype=np.float32,
    )
    tet_indices = np.asarray(
        fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    surface_triangles = np.asarray(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    reference_six_volumes_m3 = _six_tet_volumes(
        reference_vertices_m,
        tet_indices,
    )
    if np.any(np.abs(reference_six_volumes_m3) <= 1.0e-18):
        raise RuntimeError("fingertip reference mesh contains a degenerate tet")

    scene = OptixScene(fingertip_mesh)
    leds = _make_leds(fingertip)
    emissions = _emissions(scene, leds)
    dielectric_branch_u, carrier_u1, carrier_u2 = _optical_samples(len(emissions[0]))
    no_contact_optics_start_s = perf_counter()
    (
        no_contact_response,
        no_contact_energy,
        no_contact_outside_roi_power,
        no_contact_visible_side_power,
    ) = _trace_state(
        scene,
        fingertip,
        leds,
        emissions,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        require_air_sources=True,
    )
    no_contact_optics_runtime_s = perf_counter() - no_contact_optics_start_s
    no_contact_inside_roi_power = no_contact_response.sum(axis=1)
    no_contact_visible_power = float(no_contact_visible_side_power.sum())
    no_contact_outside_roi_power_fraction = (
        float(no_contact_outside_roi_power.sum()) / no_contact_visible_power
        if no_contact_visible_power > 0.0
        else 0.0
    )

    trials = []
    scenario_diameters_mm = []
    scenario_locations_y_mm = []
    for urdf_path, diameter_mm in zip(urdf_paths, diameters_mm, strict=True):
        radius_m = 0.5e-3 * diameter_mm
        for location_y_mm in locations_y_mm:
            trials.append(
                IndentationTrial(
                    name=f"sphere_{diameter_mm:g}mm_y{location_y_mm:+g}mm",
                    urdf_path=urdf_path,
                    initial_tf=wp.transform(
                        wp.vec3(
                            0.0,
                            1.0e-3 * location_y_mm,
                            fingertip.tip_z_m - initial_clearance_m - radius_m,
                        ),
                        wp.quat_identity(),
                    ),
                    motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                    approach_speed_m_s=approach_speed_m_s,
                    max_sim_time_s=max_sim_time_s,
                    initial_clearance_m=initial_clearance_m,
                )
            )
            scenario_diameters_mm.append(diameter_mm)
            scenario_locations_y_mm.append(location_y_mm)
    trial_tuple = tuple(trials)

    scenario_count = len(trial_tuple)
    force_count = len(force_targets)
    vertex_count = len(reference_vertices_m)
    led_count = len(leds)
    shape = (scenario_count, force_count)
    actual_forces_n = np.empty(shape, dtype=np.float64)
    indentations_m = np.empty(shape, dtype=np.float64)
    checkpoint_steps = np.empty(shape, dtype=np.int64)
    checkpoint_times_s = np.empty(shape, dtype=np.float64)
    maximum_particle_speeds_m_s = np.empty(shape, dtype=np.float64)
    mean_particle_speeds_m_s = np.empty(shape, dtype=np.float64)
    rms_particle_speeds_m_s = np.empty(shape, dtype=np.float64)
    particle_speed_p95_m_s = np.empty(shape, dtype=np.float64)
    kinetic_energy_j = np.empty(shape, dtype=np.float64)
    force_overshoots_n = np.empty(shape, dtype=np.float64)
    reaction_force_rates_n_s = np.empty(shape, dtype=np.float64)
    indentation_rates_m_s = np.empty(shape, dtype=np.float64)
    indenter_contact_counts = np.empty(shape, dtype=np.int32)
    total_contact_counts = np.empty(shape, dtype=np.int32)
    contact_buffer_overflow = np.empty(shape, dtype=np.int32)
    minimum_det_f = np.empty(shape, dtype=np.float64)
    inverted_tet_counts = np.empty(shape, dtype=np.int32)
    contact_centroids_W_m = np.empty((*shape, 3), dtype=np.float64)
    contact_record_offsets = np.empty((*shape, 2), dtype=np.int64)
    silicone_vertices_m = np.empty(
        (*shape, vertex_count, 3),
        dtype=np.float32,
    )
    response_matrix = np.empty(
        (*shape, led_count, LONGITUDINAL_SIDE_BIN_COUNT),
        dtype=np.float64,
    )
    energy_matrix = np.empty(
        (*shape, led_count, len(_ENERGY_FIELDS)),
        dtype=np.float64,
    )
    inside_roi_power = np.empty((*shape, led_count), dtype=np.float64)
    outside_roi_power = np.empty((*shape, led_count), dtype=np.float64)
    visible_side_power = np.empty((*shape, led_count), dtype=np.float64)
    outside_roi_power_fraction = np.empty(shape, dtype=np.float64)
    scenario_runtime_s = np.empty(scenario_count, dtype=np.float64)
    checkpoint_optics_runtime_s = np.empty(shape, dtype=np.float64)
    scenario_indices = {id(trial): index for index, trial in enumerate(trial_tuple)}
    next_force_indices = np.zeros(scenario_count, dtype=np.int64)
    contact_index_chunks: list[np.ndarray] = []
    contact_barycentric_chunks: list[np.ndarray] = []
    contact_position_chunks: list[np.ndarray] = []
    contact_normal_chunks: list[np.ndarray] = []
    contact_body_position_chunks: list[np.ndarray] = []
    contact_record_count = 0

    def collect_checkpoint(
        completed_trial: IndentationTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal contact_record_count
        scenario_index = scenario_indices[id(completed_trial)]
        force_index = int(next_force_indices[scenario_index])
        if force_index >= force_count:
            raise RuntimeError(f"{completed_trial.name} produced an extra checkpoint")
        if (
            completed_trial.reaction_force_n is None
            or completed_trial.travel_m is None
            or completed_trial.simulation_time_s is None
            or completed_trial.maximum_particle_speed_m_s is None
            or completed_trial.force_overshoot_n is None
            or completed_trial.reaction_force_rate_n_s is None
            or completed_trial.indentation_rate_m_s is None
        ):
            raise RuntimeError(f"{completed_trial.name} checkpoint is incomplete")

        vertices_m = simulation.silicone_vertices()
        if not np.all(np.isfinite(vertices_m)):
            raise RuntimeError(f"{completed_trial.name} has non-finite vertices")
        overflow = int(simulation.solver.body_particle_contact_overflow_max.numpy()[0])
        if overflow != 0:
            raise RuntimeError(
                f"{completed_trial.name} overflowed the body-particle contact buffer"
            )
        records = _indenter_contact_records(simulation, indenter, vertices_m)
        record_count = len(records[0])
        if record_count == 0:
            raise RuntimeError(f"{completed_trial.name} has no indenter contacts")

        current_six_volumes_m3 = _six_tet_volumes(vertices_m, tet_indices)
        det_f = current_six_volumes_m3 / reference_six_volumes_m3
        if not np.all(np.isfinite(det_f)):
            raise RuntimeError(f"{completed_trial.name} has non-finite det(F)")

        particle_qd = simulation.state.particle_qd
        particle_flags = simulation.fingertip_model.model.particle_flags
        particle_mass = simulation.fingertip_model.model.particle_mass
        if particle_qd is None or particle_flags is None or particle_mass is None:
            raise RuntimeError(
                f"{completed_trial.name} has no particle motion diagnostics"
            )
        particle_start = simulation.fingertip_model.silicone_particle_start
        particle_stop = (
            particle_start + simulation.fingertip_model.silicone_particle_count
        )
        velocities_m_s = particle_qd.numpy()[particle_start:particle_stop]
        flags = particle_flags.numpy()[particle_start:particle_stop]
        masses_kg = particle_mass.numpy()[particle_start:particle_stop]
        active = (flags & int(newton.ParticleFlags.ACTIVE)) != 0
        active_speeds_m_s = np.linalg.norm(velocities_m_s[active], axis=1)
        if len(active_speeds_m_s) == 0 or not np.all(np.isfinite(active_speeds_m_s)):
            raise RuntimeError(
                f"{completed_trial.name} has invalid active particle speeds"
            )

        optics_start_s = perf_counter()
        scene.update_silicone(vertices_m)
        responses, energies, outside_power, visible_power = _trace_state(
            scene,
            fingertip,
            leds,
            emissions,
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
        )
        checkpoint_optics_runtime_s[scenario_index, force_index] = (
            perf_counter() - optics_start_s
        )

        actual_forces_n[scenario_index, force_index] = completed_trial.reaction_force_n
        indentations_m[scenario_index, force_index] = (
            completed_trial.travel_m - initial_clearance_m
        )
        checkpoint_steps[scenario_index, force_index] = completed_trial.step_count
        checkpoint_times_s[scenario_index, force_index] = (
            completed_trial.simulation_time_s
        )
        maximum_particle_speeds_m_s[scenario_index, force_index] = (
            completed_trial.maximum_particle_speed_m_s
        )
        mean_particle_speeds_m_s[scenario_index, force_index] = float(
            np.mean(active_speeds_m_s)
        )
        rms_particle_speeds_m_s[scenario_index, force_index] = float(
            np.sqrt(np.mean(active_speeds_m_s**2))
        )
        particle_speed_p95_m_s[scenario_index, force_index] = float(
            np.percentile(active_speeds_m_s, 95.0)
        )
        kinetic_energy_j[scenario_index, force_index] = float(
            0.5
            * np.sum(
                masses_kg[active] * active_speeds_m_s**2,
            )
        )
        force_overshoots_n[scenario_index, force_index] = (
            completed_trial.force_overshoot_n
        )
        reaction_force_rates_n_s[scenario_index, force_index] = (
            completed_trial.reaction_force_rate_n_s
        )
        indentation_rates_m_s[scenario_index, force_index] = (
            completed_trial.indentation_rate_m_s
        )
        indenter_contact_counts[scenario_index, force_index] = (
            simulation.soft_contact_count(indenter.body_index)
        )
        total_contact_counts[scenario_index, force_index] = (
            simulation.soft_contact_count()
        )
        contact_buffer_overflow[scenario_index, force_index] = overflow
        minimum_det_f[scenario_index, force_index] = float(det_f.min())
        inverted_tet_counts[scenario_index, force_index] = int(
            np.count_nonzero(det_f <= 0.0)
        )
        contact_centroids_W_m[scenario_index, force_index] = records[2].mean(axis=0)
        contact_record_offsets[scenario_index, force_index] = (
            contact_record_count,
            record_count,
        )
        silicone_vertices_m[scenario_index, force_index] = vertices_m
        response_matrix[scenario_index, force_index] = responses
        energy_matrix[scenario_index, force_index] = energies
        inside_roi_power[scenario_index, force_index] = responses.sum(axis=1)
        outside_roi_power[scenario_index, force_index] = outside_power
        visible_side_power[scenario_index, force_index] = visible_power
        combined_visible_power = float(visible_power.sum())
        outside_roi_power_fraction[scenario_index, force_index] = (
            float(outside_power.sum()) / combined_visible_power
            if combined_visible_power > 0.0
            else 0.0
        )

        contact_index_chunks.append(records[0])
        contact_barycentric_chunks.append(records[1])
        contact_position_chunks.append(records[2])
        contact_normal_chunks.append(records[3])
        contact_body_position_chunks.append(records[4])
        contact_record_count += record_count
        next_force_indices[scenario_index] += 1
        if next_force_indices[scenario_index] == force_count:
            if completed_trial.wall_runtime_s is None:
                raise RuntimeError(f"{completed_trial.name} has no scenario runtime")
            scenario_runtime_s[scenario_index] = completed_trial.wall_runtime_s

    IndentationStudy(
        fingertip,
        trial_tuple,
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        force_targets_n=force_targets,
        element_size_mm=_ELEMENT_SIZE_MM,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    ).run(inspect_checkpoint=collect_checkpoint)

    if np.any(next_force_indices != force_count):
        raise RuntimeError("not every fingertip force checkpoint was collected")
    return FingertipEvaluation(
        reference_vertices_m=reference_vertices_m,
        tet_indices=tet_indices,
        surface_triangles=surface_triangles,
        bonded_vertex_indices=fingertip_mesh.bonded_vertex_indices,
        led_source_centers_m=np.asarray(
            fingertip.led_source_centers_m,
            dtype=np.float64,
        ),
        no_contact_response=no_contact_response,
        no_contact_energy=no_contact_energy,
        no_contact_inside_roi_power=no_contact_inside_roi_power,
        no_contact_outside_roi_power=no_contact_outside_roi_power,
        no_contact_visible_side_power=no_contact_visible_side_power,
        no_contact_outside_roi_power_fraction=(no_contact_outside_roi_power_fraction),
        scenario_names=tuple(trial.name for trial in trial_tuple),
        sphere_diameters_mm=np.asarray(scenario_diameters_mm, dtype=np.float64),
        contact_y_mm=np.asarray(scenario_locations_y_mm, dtype=np.float64),
        force_targets_n=np.asarray(force_targets, dtype=np.float64),
        actual_forces_n=actual_forces_n,
        indentations_m=indentations_m,
        checkpoint_steps=checkpoint_steps,
        checkpoint_times_s=checkpoint_times_s,
        maximum_particle_speeds_m_s=maximum_particle_speeds_m_s,
        mean_particle_speeds_m_s=mean_particle_speeds_m_s,
        rms_particle_speeds_m_s=rms_particle_speeds_m_s,
        particle_speed_p95_m_s=particle_speed_p95_m_s,
        kinetic_energy_j=kinetic_energy_j,
        force_overshoots_n=force_overshoots_n,
        reaction_force_rates_n_s=reaction_force_rates_n_s,
        indentation_rates_m_s=indentation_rates_m_s,
        indenter_contact_counts=indenter_contact_counts,
        total_contact_counts=total_contact_counts,
        contact_buffer_overflow=contact_buffer_overflow,
        minimum_det_f=minimum_det_f,
        inverted_tet_counts=inverted_tet_counts,
        contact_centroids_W_m=contact_centroids_W_m,
        contact_record_offsets=contact_record_offsets,
        contact_particle_indices=np.concatenate(contact_index_chunks),
        contact_barycentric=np.concatenate(contact_barycentric_chunks),
        contact_positions_W_m=np.concatenate(contact_position_chunks),
        contact_normals_W=np.concatenate(contact_normal_chunks),
        contact_body_positions=np.concatenate(contact_body_position_chunks),
        silicone_vertices_m=silicone_vertices_m,
        response_matrix=response_matrix,
        energy_fields=_ENERGY_FIELDS,
        energy_matrix=energy_matrix,
        inside_roi_power=inside_roi_power,
        outside_roi_power=outside_roi_power,
        visible_side_power=visible_side_power,
        outside_roi_power_fraction=outside_roi_power_fraction,
        scenario_runtime_s=scenario_runtime_s,
        checkpoint_optics_runtime_s=checkpoint_optics_runtime_s,
        no_contact_optics_runtime_s=no_contact_optics_runtime_s,
    )


__all__ = [
    "FingertipEvaluation",
    "evaluate_fingertip",
]
