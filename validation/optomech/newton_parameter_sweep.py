"""Measure the full optomechanical cost of a Newton parameter sweep."""

from __future__ import annotations

import gc
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np
import newton
import warp as wp
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import FingertipMesh
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.newton import Indenter
from lumo.optimization import sensing_descriptors, sensing_objectives
from lumo.ray_tracing import (
    LED,
    OptixScene,
    safe_secondary_origins,
    side_view_observation,
    trace_bounded_paths,
)
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_CONTACT_X_MM = (-7.5, 0.0, 7.5)
_SPHERE_DIAMETER_MM = 10.0
_SPHERE_RADIUS_M = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 1.0
_APPROACH_SPEED_M_S = 2.5e-2
_SETTLE_DURATION_S = 1.0
# The 1 s hold plus approach work fits in this validation-local cap. Keeping
# failed configurations bounded prevents a servo failure from dominating
# the full factorial throughput measurement.
_MAX_SIM_TIME_S = 2.0

_MAX_BONDED_DRIFT_M = 1.0e-8
_MAX_CARRIER_PENETRATION_M = 1.0e-5

_STIFFNESSES_N_M = (4.0e6, 8.0e6)
_FREQUENCIES_HZ = (500.0, 1000.0, 2000.0)
_VBD_ITERATIONS = (10, 20)
_ELEMENT_SIZES_MM = (1.0, 0.75)
_SOFT_CONTACT_MARGIN_M = 1.0e-4

_RAY_COUNT = 65_536
_RAY_SIDE_COUNT = 256
_MAX_BOUNCES = 24
_OPTICAL_SEED = 20260823
_CARRIER_ALBEDO = 0.7

_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK

_DEFAULT_OUTPUT = Path("output/validation/newton_parameter_sweep.json")


def _configurations() -> list[dict[str, object]]:
    """Return the requested 2 x 3 x 2 x 2 full-factorial sweep."""
    configurations = []
    for stiffness in _STIFFNESSES_N_M:
        for frequency in _FREQUENCIES_HZ:
            for iterations in _VBD_ITERATIONS:
                for element_size in _ELEMENT_SIZES_MM:
                    configurations.append(
                        {
                            "carrier_contact_stiffness_n_m": stiffness,
                            "sim_frequency": frequency,
                            "iterations": iterations,
                            "element_size_mm": element_size,
                        }
                    )
    if len(configurations) != 24:
        raise AssertionError("the Newton sweep must contain exactly 24 cases")
    return configurations


def _carrier_interior_depths_m(
    positions_m: np.ndarray,
    *,
    carrier_cross_section: Polygon,
    carrier_y_limits_m: tuple[float, float],
) -> np.ndarray:
    y_min_m, y_max_m = carrier_y_limits_m
    inside = (
        (positions_m[:, 1] > y_min_m)
        & (positions_m[:, 1] < y_max_m)
        & contains_xy(
            carrier_cross_section,
            1.0e3 * positions_m[:, 0],
            1.0e3 * positions_m[:, 2],
        )
    )
    depths_m = np.zeros(len(positions_m), dtype=np.float64)
    if np.any(inside):
        depths_m[inside] = 1.0e-3 * distance(
            carrier_cross_section.boundary,
            points(
                1.0e3 * positions_m[inside, 0],
                1.0e3 * positions_m[inside, 2],
            ),
        )
    return depths_m


def _carrier_surface_penetration_depths_m(
    positions_m: np.ndarray,
    surface_triangles: np.ndarray,
    *,
    free_surface_mask: np.ndarray,
    carrier_cross_section: Polygon,
    carrier_y_limits_m: tuple[float, float],
) -> np.ndarray:
    """Measure exposed surface overlap without using it as a hard gate."""
    y_min_m, y_max_m = carrier_y_limits_m
    depths_m = np.zeros(len(surface_triangles), dtype=np.float64)
    for triangle_index, triangle_indices in enumerate(surface_triangles):
        if not free_surface_mask[triangle_index]:
            continue
        triangle_positions = positions_m[triangle_indices]
        if (
            float(triangle_positions[:, 1].max()) <= y_min_m
            or float(triangle_positions[:, 1].min()) >= y_max_m
        ):
            continue
        triangle = Polygon(1.0e3 * triangle_positions[:, (0, 2)])
        if triangle.is_empty or not triangle.is_valid or triangle.area == 0.0:
            continue
        overlap = triangle.intersection(carrier_cross_section)
        if overlap.is_empty or overlap.area == 0.0:
            continue
        candidate_points: list[tuple[float, float]] = []
        geometries = (
            overlap.geoms if hasattr(overlap, "geoms") else (overlap,)
        )
        for geometry in geometries:
            if geometry.geom_type == "Polygon":
                candidate_points.extend(
                    (float(x), float(y))
                    for x, y in geometry.exterior.coords
                )
            if not geometry.is_empty:
                candidate_points.append(
                    tuple(
                        float(value)
                        for value in geometry.representative_point()
                        .coords[0]
                    )
                )
        if candidate_points:
            candidate_array = np.asarray(candidate_points, dtype=np.float64)
            depths_m[triangle_index] = float(
                1.0e-3
                * distance(
                    carrier_cross_section.boundary,
                    points(candidate_array[:, 0], candidate_array[:, 1]),
                ).max()
            )
    return depths_m


def _make_trial(
    fingertip: Fingertip,
    sphere_urdf_path: Path,
    *,
    contact_x_mm: float,
) -> DesignTrial:
    surface_z_mm = (
        fingertip.silicone.ellipse_center_z_mm
        - semiellipse_depth_at_x_mm(
            half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
            height_mm=fingertip.silicone.ellipse_radius_z_mm,
            x_mm=contact_x_mm,
        )
    )
    initial_center_z_m = (
        1.0e-3 * surface_z_mm
        - _INITIAL_CLEARANCE_M
        - _SPHERE_RADIUS_M
    )
    return DesignTrial(
        name=f"contact_x={contact_x_mm:+.1f}mm",
        urdf_path=sphere_urdf_path,
        initial_tf=wp.transform(
            wp.vec3(1.0e-3 * contact_x_mm, 0.0, initial_center_z_m),
            wp.quat_identity(),
        ),
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=_APPROACH_SPEED_M_S,
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )


def _measure_trial(
    trial: DesignTrial,
    simulation: LumoSimulation,
    indenter: Indenter,
    *,
    contact_x_mm: float,
) -> tuple[dict[str, object], np.ndarray]:
    """Collect one live checkpoint and apply only the hard validity gates."""
    if (
        trial.reaction_force_n is None
        or trial.travel_m is None
        or trial.maximum_particle_speed_m_s is None
        or trial.force_change_n is None
        or trial.final_tf is None
    ):
        raise RuntimeError("completed trial has incomplete scalar results")

    positions_m = simulation.silicone_vertices()
    velocities_m_s = simulation.state.particle_qd.numpy()
    reference_positions_m = np.asarray(
        simulation.fingertip_mesh.silicone.vertices,
        dtype=np.float64,
    )
    finite_state = bool(
        np.all(np.isfinite(positions_m))
        and np.all(np.isfinite(velocities_m_s))
    )

    bonded_indices = np.asarray(
        simulation.fingertip_model.bonded_particle_indices.numpy(),
        dtype=np.int64,
    )
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    surface_triangles = np.asarray(
        simulation.fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)

    maximum_displacement_m = None
    maximum_bonded_drift_m = None
    maximum_particle_penetration_m = None
    maximum_free_tet_penetration_m = None
    maximum_exposed_surface_penetration_m = None
    if finite_state:
        maximum_displacement_m = float(
            np.linalg.norm(positions_m - reference_positions_m, axis=1).max()
        )
        maximum_bonded_drift_m = float(
            np.linalg.norm(
                positions_m[bonded_indices]
                - simulation.fingertip_model.bonded_local_positions.numpy(),
                axis=1,
            ).max()
        )
        carrier_cross_section = Polygon(
            simulation.fingertip.carrier.cross_section
        )
        carrier_vertices = np.asarray(
            simulation.fingertip_mesh.carrier.vertices,
            dtype=np.float64,
        )
        carrier_y_limits_m = (
            float(carrier_vertices[:, 1].min()),
            float(carrier_vertices[:, 1].max()),
        )
        nonbonded = np.ones(len(positions_m), dtype=bool)
        nonbonded[bonded_indices] = False
        particle_depths_m = _carrier_interior_depths_m(
            positions_m,
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        particle_depths_m[~nonbonded] = 0.0
        tet_depths_m = _carrier_interior_depths_m(
            positions_m[tet_indices].mean(axis=1),
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        free_tet_mask = np.all(nonbonded[tet_indices], axis=1)
        free_tet_depths_m = tet_depths_m[free_tet_mask]
        free_surface_mask = np.all(nonbonded[surface_triangles], axis=1)
        surface_depths_m = _carrier_surface_penetration_depths_m(
            positions_m,
            surface_triangles,
            free_surface_mask=free_surface_mask,
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        maximum_particle_penetration_m = float(particle_depths_m.max())
        maximum_free_tet_penetration_m = float(
            free_tet_depths_m.max() if len(free_tet_depths_m) else 0.0
        )
        maximum_exposed_surface_penetration_m = float(surface_depths_m.max())

    indenter_contact_count = simulation.soft_contact_count(
        indenter.body_index
    )
    force_error_n = abs(trial.reaction_force_n - _TARGET_FORCE_N)
    failures: list[str] = []
    if force_error_n > _FORCE_TOLERANCE_N:
        failures.append("force tolerance")
    if not finite_state:
        failures.append("non-finite state")
    if indenter_contact_count == 0:
        failures.append("no indenter contact")
    if (
        maximum_bonded_drift_m is None
        or maximum_bonded_drift_m > _MAX_BONDED_DRIFT_M
    ):
        failures.append("bonded drift")
    if (
        maximum_particle_penetration_m is None
        or maximum_particle_penetration_m > _MAX_CARRIER_PENETRATION_M
        or maximum_free_tet_penetration_m is None
        or maximum_free_tet_penetration_m > _MAX_CARRIER_PENETRATION_M
    ):
        failures.append("carrier penetration")

    record = {
        "contact_x_mm": contact_x_mm,
        "settled_force_n": trial.reaction_force_n,
        "force_error_n": force_error_n,
        "travel_m": trial.travel_m,
        "maximum_displacement_m": maximum_displacement_m,
        "maximum_active_particle_speed_m_s": (
            trial.maximum_particle_speed_m_s
        ),
        "force_change_n": trial.force_change_n,
        "simulation_step_count": trial.step_count,
        "maximum_particle_penetration_m": maximum_particle_penetration_m,
        "maximum_free_tet_penetration_m": maximum_free_tet_penetration_m,
        "maximum_exposed_surface_penetration_m": (
            maximum_exposed_surface_penetration_m
        ),
        "maximum_bonded_drift_m": maximum_bonded_drift_m,
        "indenter_contact_count": indenter_contact_count,
        "hard_valid": not failures,
        "failure_reason": ", ".join(failures),
    }
    return record, positions_m.copy()


def _run_mechanics(
    fingertip: Fingertip,
    sphere_urdf_path: Path,
    parameters: dict[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Run three fresh Newton contact states for one parameter tuple."""
    sim_frequency = float(parameters["sim_frequency"])
    records: dict[float, dict[str, object]] = {}
    snapshots: dict[float, np.ndarray] = {}
    mesh: FingertipMesh | None = None
    start = perf_counter()

    for contact_x_mm in _CONTACT_X_MM:
        trial = _make_trial(
            fingertip,
            sphere_urdf_path,
            contact_x_mm=contact_x_mm,
        )

        def inspect_trial(
            completed_trial: DesignTrial,
            simulation: LumoSimulation,
            indenter: Indenter,
        ) -> None:
            nonlocal mesh
            record, snapshot = _measure_trial(
                completed_trial,
                simulation,
                indenter,
                contact_x_mm=contact_x_mm,
            )
            records[contact_x_mm] = record
            snapshots[contact_x_mm] = snapshot
            if mesh is None:
                mesh = simulation.fingertip_mesh

        try:
            DesignStudy(
                fingertip,
                (trial,),
                sim_frequency=sim_frequency,
                force_tolerance_n=_FORCE_TOLERANCE_N,
                settle_duration_s=_SETTLE_DURATION_S,
                element_size_mm=float(parameters["element_size_mm"]),
                iterations=int(parameters["iterations"]),
                soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
                carrier_contact_stiffness_n_m=float(
                    parameters["carrier_contact_stiffness_n_m"]
                ),
            ).run(inspect_trial=inspect_trial)
        except Exception as error:
            records[contact_x_mm] = {
                "contact_x_mm": contact_x_mm,
                "hard_valid": False,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        del trial

    hard_valid = bool(all(record["hard_valid"] for record in records.values()))
    failure_reasons = [
        f"x={contact_x_mm:+g}: {record['failure_reason']}"
        for contact_x_mm, record in records.items()
        if record["failure_reason"]
    ]
    mechanics = {
        "hard_valid": hard_valid,
        "failure_reason": "; ".join(failure_reasons),
        "contacts": [records[x] for x in _CONTACT_X_MM],
        "wall_time_s": perf_counter() - start,
    }
    if not hard_valid or mesh is None:
        return mechanics, None
    bundle = {
        "mesh": mesh,
        "reference_vertices": np.asarray(
            mesh.silicone.vertices,
            dtype=np.float64,
        ).copy(),
        "contact_snapshots": snapshots,
    }
    return mechanics, bundle


def _make_led(fingertip: Fingertip, mesh: FingertipMesh) -> LED:
    vertices = np.asarray(mesh.silicone.vertices, dtype=np.float64)
    center_y_m = 0.5 * float(vertices[:, 1].min() + vertices[:, 1].max())
    stem_bottom_z_m = -1.0e-3 * fingertip.parameters.geometry.stem_height_mm
    return LED(
        position_W_m=np.array((0.0, center_y_m, stem_bottom_z_m)),
        normal_W=np.array((0.0, 0.0, -1.0)),
        parameters=fingertip.parameters.led,
    )


def _emit_from_stem_boundary(
    scene: OptixScene,
    led: LED,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    probe_distance_m = 0.5e-3 * led.parameters.height_mm
    probe_origin = (led.position_W_m - probe_distance_m * led.normal_W)[None]
    carrier_hit = scene.trace_closest(
        probe_origin,
        led.normal_W[None],
        mask=_CARRIER_MASK,
    )
    if not carrier_hit["hit"][0]:
        raise AssertionError("carrier probe did not find the LED boundary")
    hit_position = probe_origin[0] + carrier_hit["t"][0] * led.normal_W
    if not np.allclose(hit_position, led.position_W_m, rtol=0.0, atol=1.0e-7):
        raise AssertionError("carrier probe found the wrong LED boundary")
    emission = led.emit(u1, u2)
    emission["origin_W_m"] = safe_secondary_origins(carrier_hit, led.normal_W[None])[0]
    return emission


def _source_inside_silicone(
    scene: OptixScene,
    led: LED,
    emission: np.ndarray,
    *,
    state_label: str,
) -> np.ndarray:
    initial_hits = scene.trace_closest(
        emission["origin_W_m"],
        emission["direction_W"],
        mask=_ALL_MASK,
    )
    unknown_hit = initial_hits["hit"] & ~np.isin(
        initial_hits["instance_id"],
        (_SILICONE_INSTANCE_ID, _CARRIER_INSTANCE_ID),
    )
    if np.any(unknown_hit):
        raise AssertionError(f"{state_label} has an unknown source hit")
    silicone_hit = scene.trace_closest(
        emission["origin_W_m"][:1],
        led.normal_W[None],
        mask=_SILICONE_MASK,
    )[0]
    if not silicone_hit["hit"]:
        raise AssertionError(f"{state_label} has no silicone source path")
    source_inside = float(np.dot(silicone_hit["normal_W"], led.normal_W)) > 0.0
    if state_label == "no_contact" and not source_inside:
        raise AssertionError("no-contact source normal faces away from silicone")
    return np.full(len(initial_hits), source_inside, dtype=bool)


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    *,
    inside_silicone: np.ndarray,
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    transport_start = perf_counter()
    result = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=inside_silicone,
        n_air=1.0,
        n_silicone=fingertip.parameters.optical.refractive_index,
        extinction_coefficient_m_inv=(
            fingertip.parameters.optical.extinction_coefficient_m_inv
        ),
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_MAX_BOUNCES,
        dielectric_branch_u=samples[0],
        carrier_u1=samples[1],
        carrier_u2=samples[2],
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        mask=_ALL_MASK,
    )
    transport_s = perf_counter() - transport_start
    if not np.isclose(
        result.accounted_power,
        result.emitted_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("bounded transport failed energy closure")
    observation_start = perf_counter()
    observation = side_view_observation(
        result.escaped_rays,
        fingertip=fingertip,
    )
    observation_s = perf_counter() - observation_start
    return observation, {
        "bounded_ray_transport_s": transport_s,
        "observation_s": observation_s,
    }


def _fixed_optical_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_OPTICAL_SEED)
    return tuple(
        rng.random((_MAX_BOUNCES, _RAY_COUNT))
        for _ in range(3)
    )


def _minimum_pair(values: np.ndarray) -> tuple[float, tuple[int, int]]:
    best_distance = float("inf")
    best_pair = (-1, -1)
    for first in range(len(values) - 1):
        for second in range(first + 1, len(values)):
            current = float(
                np.linalg.norm(np.atleast_1d(values[first] - values[second]))
            )
            if current < best_distance:
                best_distance = current
                best_pair = (first, second)
    return best_distance, best_pair


def _run_optics(
    fingertip: Fingertip,
    bundle: dict[str, object],
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    """Evaluate one valid Newton bundle with one common optical sample set."""
    mesh = bundle["mesh"]
    if not isinstance(mesh, FingertipMesh):
        raise TypeError("mechanics bundle has no FingertipMesh")
    reference_vertices = np.asarray(bundle["reference_vertices"])
    snapshots = bundle["contact_snapshots"]
    if not isinstance(snapshots, dict):
        raise TypeError("mechanics bundle has no contact snapshots")

    optics_start = perf_counter()
    scene_start = perf_counter()
    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    scene_construction_s = perf_counter() - scene_start

    side_indices = np.arange(_RAY_SIDE_COUNT)
    sample_i, sample_j = np.meshgrid(
        side_indices,
        side_indices,
        indexing="ij",
    )
    emission_u1 = (sample_i.ravel() + 0.5) / _RAY_SIDE_COUNT
    emission_u2 = (sample_j.ravel() + 0.5) / _RAY_SIDE_COUNT
    led = _make_led(fingertip, mesh)
    emission_start = perf_counter()
    emission = _emit_from_stem_boundary(
        scene,
        led,
        emission_u1,
        emission_u2,
    )
    emission_setup_s = perf_counter() - emission_start

    labels = ["no_contact"] + [
        f"contact_x={contact_x_mm:+.1f}mm"
        for contact_x_mm in _CONTACT_X_MM
    ]
    states = [reference_vertices] + [
        np.asarray(snapshots[contact_x_mm], dtype=np.float64)
        for contact_x_mm in _CONTACT_X_MM
    ]
    observations = []
    state_timings = []
    gas_update_s = 0.0
    transport_s = 0.0
    observation_s = 0.0
    for label, vertices in zip(labels, states, strict=True):
        update_start = perf_counter()
        scene.update_silicone(vertices)
        update_elapsed = perf_counter() - update_start
        gas_update_s += update_elapsed

        source_start = perf_counter()
        inside_silicone = _source_inside_silicone(
            scene,
            led,
            emission,
            state_label=label,
        )
        source_elapsed = perf_counter() - source_start
        observation, timing = _trace_state(
            scene,
            fingertip,
            emission,
            inside_silicone=inside_silicone,
            samples=samples,
        )
        transport_s += timing["bounded_ray_transport_s"]
        observation_s += timing["observation_s"]
        observations.append(observation)
        state_timings.append(
            {
                "state": label,
                "silicone_gas_update_s": update_elapsed,
                "source_check_s": source_elapsed,
                **timing,
            }
        )

    response = np.stack(observations)
    if response.shape != (4, 4):
        raise AssertionError("side-view response must have shape (4, 4)")
    intensity, spatial = sensing_descriptors(response)
    j_intensity, j_spatial = sensing_objectives(response)
    intensity_pair = _minimum_pair(intensity)[1]
    spatial_pair = _minimum_pair(spatial)[1]
    optics_s = perf_counter() - optics_start
    del scene
    gc.collect()
    return {
        "ray_count": _RAY_COUNT,
        "max_bounces": _MAX_BOUNCES,
        "quadrant_responses": {
            label: response[index].tolist()
            for index, label in enumerate(labels)
        },
        "intensity_responses": {
            label: float(intensity[index])
            for index, label in enumerate(labels)
        },
        "spatial_responses": {
            label: spatial[index].tolist()
            for index, label in enumerate(labels)
        },
        "J_intensity": float(j_intensity),
        "worst_intensity_pair": [
            labels[intensity_pair[0]],
            labels[intensity_pair[1]],
        ],
        "J_spatial": float(j_spatial),
        "worst_spatial_pair": [
            labels[spatial_pair[0]],
            labels[spatial_pair[1]],
        ],
        "timing_s": {
            "optix_scene_construction_s": scene_construction_s,
            "emission_setup_s": emission_setup_s,
            "silicone_gas_update_s": gas_update_s,
            "bounded_ray_transport_s": transport_s,
            "observation_objective_s": observation_s,
            "total_optics_s": optics_s,
            "states": state_timings,
        },
    }


def _warm_up(fingertip: Fingertip, sphere_urdf_path: Path) -> float:
    """Warm Warp/Newton and one OptiX path before measured configurations."""
    start = perf_counter()
    newton_builder = None
    simulation = None
    indenter = None
    scene = None
    try:
        newton_builder = newton.ModelBuilder(
            gravity=wp.vec3(0.0, 0.0, 0.0)
        )
        warmup_center_z_m = (
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - _SPHERE_RADIUS_M
        )
        indenter = Indenter.add_urdf(
            newton_builder,
            sphere_urdf_path,
            tf=wp.transform(
                wp.vec3(0.0, 0.0, warmup_center_z_m),
                wp.quat_identity(),
            ),
        )
        simulation = LumoSimulation(
            fingertip,
            builder=newton_builder,
            sim_frequency=1000.0,
            iterations=10,
            soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
            element_size_mm=1.0,
            carrier_contact_stiffness_n_m=4.0e6,
        )
        simulation.step()
        mesh = simulation.fingertip_mesh
        scene = OptixScene(
            mesh,
            silicone_instance_id=_SILICONE_INSTANCE_ID,
            carrier_instance_id=_CARRIER_INSTANCE_ID,
            silicone_visibility_mask=_SILICONE_MASK,
            carrier_visibility_mask=_CARRIER_MASK,
        )
        scene.trace_closest(
            np.zeros((1, 3), dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            mask=_ALL_MASK,
        )
        wp.synchronize()
    finally:
        del scene, simulation, indenter, newton_builder
        gc.collect()
    return perf_counter() - start


def _comparison_to_slowest(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    valid = [result for result in results if result["valid"]]
    if len(valid) < 2:
        return []
    slowest = max(valid, key=lambda result: float(result["total_s"]))
    slowest_optics = slowest["optics"]
    if not isinstance(slowest_optics, dict):
        return []
    slowest_contacts = slowest["mechanics"]["contacts"]
    comparisons = []
    for result in sorted(valid, key=lambda item: float(item["total_s"])):
        optics = result["optics"]
        if not isinstance(optics, dict):
            continue
        contacts = result["mechanics"]["contacts"]
        force_delta = max(
            abs(
                float(contact["settled_force_n"])
                - float(reference["settled_force_n"])
            )
            for contact, reference in zip(contacts, slowest_contacts, strict=True)
        )
        displacement_delta = max(
            abs(
                float(contact["maximum_displacement_m"])
                - float(reference["maximum_displacement_m"])
            )
            for contact, reference in zip(contacts, slowest_contacts, strict=True)
        )
        travel_delta = max(
            abs(
                float(contact["travel_m"])
                - float(reference["travel_m"])
            )
            for contact, reference in zip(contacts, slowest_contacts, strict=True)
        )
        comparisons.append(
            {
                "configuration": result["parameters"],
                "reference_slowest_configuration": slowest["parameters"],
                "total_s_delta": float(result["total_s"])
                - float(slowest["total_s"]),
                "maximum_settled_force_delta_n": force_delta,
                "maximum_displacement_delta_m": displacement_delta,
                "maximum_travel_delta_m": travel_delta,
                "J_intensity_delta": float(optics["J_intensity"])
                - float(slowest_optics["J_intensity"]),
                "J_spatial_delta": float(optics["J_spatial"])
                - float(slowest_optics["J_spatial"]),
            }
        )
    return comparisons


def _print_results(results: list[dict[str, object]]) -> None:
    print()
    print("Hard-valid configurations sorted by measured end-to-end cost")
    print(
        f"{'stiffness':>11} {'Hz':>6} {'iter':>5} {'mesh':>6} "
        f"{'valid':>7} {'J_i':>11} {'J_s':>11} "
        f"{'mechanics[s]':>13} {'optics[s]':>11} {'total[s]':>10}"
    )
    valid = [result for result in results if result["valid"]]
    for result in sorted(valid, key=lambda item: float(item["total_s"])):
        parameters = result["parameters"]
        optics = result["optics"]
        print(
            f"{float(parameters['carrier_contact_stiffness_n_m']):11.3e} "
            f"{float(parameters['sim_frequency']):6.0f} "
            f"{int(parameters['iterations']):5d} "
            f"{float(parameters['element_size_mm']):6.2f} "
            f"{'PASS':>7} "
            f"{float(optics['J_intensity']):11.5e} "
            f"{float(optics['J_spatial']):11.5e} "
            f"{float(result['mechanics_s']):13.3f} "
            f"{float(result['optics']['timing_s']['total_optics_s']):11.3f} "
            f"{float(result['total_s']):10.3f}"
        )

    print()
    print("Invalid configurations")
    for result in results:
        if result["valid"]:
            continue
        parameters = result["parameters"]
        print(
            f"  k={float(parameters['carrier_contact_stiffness_n_m']):.3e}, "
            f"f={float(parameters['sim_frequency']):g}, "
            f"iter={int(parameters['iterations'])}, "
            f"mesh={float(parameters['element_size_mm']):g}: "
            f"{result['failure_reason']}"
        )


def main() -> None:
    output_path = _DEFAULT_OUTPUT
    fingertip_start = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    fingertip_setup_s = perf_counter() - fingertip_start
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_10mm.urdf",
    )

    with ExitStack() as resources:
        sphere_urdf_path = resources.enter_context(as_file(sphere_resource))
        warmup_s = _warm_up(fingertip, sphere_urdf_path)
        print(f"warm-up complete in {warmup_s:.3f} s")
        samples = _fixed_optical_samples()
        results = []
        configurations = _configurations()
        sweep_start = perf_counter()
        for index, parameters in enumerate(configurations, start=1):
            print(
                f"[{index}/{len(configurations)}] "
                f"k={float(parameters['carrier_contact_stiffness_n_m']):.3e}, "
                f"f={float(parameters['sim_frequency']):g}, "
                f"iter={int(parameters['iterations'])}, "
                f"mesh={float(parameters['element_size_mm']):g}",
                flush=True,
            )
            config_start = perf_counter()
            mechanics, bundle = _run_mechanics(
                fingertip,
                sphere_urdf_path,
                parameters,
            )
            optics = None
            optics_failure = ""
            if bundle is not None:
                try:
                    optics = _run_optics(fingertip, bundle, samples)
                except Exception as error:
                    optics_failure = f"{type(error).__name__}: {error}"
                del bundle
                gc.collect()
            failure_reason = str(mechanics["failure_reason"])
            if optics_failure:
                failure_reason = "; ".join(
                    value for value in (failure_reason, optics_failure) if value
                )
            result = {
                "parameters": parameters,
                "hard_valid": mechanics["hard_valid"],
                "valid": bool(mechanics["hard_valid"] and optics is not None),
                "failure_reason": failure_reason,
                "mechanics": mechanics,
                "mechanics_s": float(mechanics["wall_time_s"]),
                "optics": optics,
                "total_s": perf_counter() - config_start,
            }
            results.append(result)
            print(
                f"  {'PASS' if result['valid'] else 'INVALID'} | "
                f"mechanics={float(result['mechanics_s']):.3f}s | "
                f"total={float(result['total_s']):.3f}s",
                flush=True,
            )

    recommendation = None
    valid_results = [result for result in results if result["valid"]]
    if valid_results:
        recommended = min(valid_results, key=lambda result: float(result["total_s"]))
        recommendation = {
            "parameters": recommended["parameters"],
            "approach_speed_m_s": _APPROACH_SPEED_M_S,
            "settle_duration_s": _SETTLE_DURATION_S,
            "max_sim_time_s": _MAX_SIM_TIME_S,
            "ray_count": _RAY_COUNT,
            "max_bounces": _MAX_BOUNCES,
            "measured_seconds_per_fingertip": float(recommended["total_s"]),
            "estimated_minutes_per_100_designs": (
                100.0 * float(recommended["total_s"]) / 60.0
            ),
            "estimated_hours_per_1000_designs": (
                1000.0 * float(recommended["total_s"]) / 3600.0
            ),
            "selection": (
                "fastest measured configuration that is hard-valid for all "
                "three loaded contacts and completes the fixed optical evaluation"
            ),
            "comparison_to_slower_valid": _comparison_to_slowest(results),
        }

    payload = {
        "schema_version": 2,
        "study": "optomechanical_newton_parameter_sweep",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_configuration": {
            "carrier_contact_stiffness_n_m": list(_STIFFNESSES_N_M),
            "sim_frequency_hz": list(_FREQUENCIES_HZ),
            "vbd_iterations": list(_VBD_ITERATIONS),
            "element_size_mm": list(_ELEMENT_SIZES_MM),
            "configuration_count": len(configurations),
        },
        "fixed_evaluation_protocol": {
            "contact_states": ["no_contact", *_CONTACT_X_MM],
            "target_force_n": _TARGET_FORCE_N,
            "force_tolerance_n": _FORCE_TOLERANCE_N,
            "approach_speed_m_s": _APPROACH_SPEED_M_S,
            "settle_duration_s": _SETTLE_DURATION_S,
            "ray_count": _RAY_COUNT,
            "max_bounces": _MAX_BOUNCES,
            "optical_seed": _OPTICAL_SEED,
            "carrier_albedo": _CARRIER_ALBEDO,
        },
        "timing_context": {
            "fingertip_setup_s": fingertip_setup_s,
            "warmup_s": warmup_s,
            "measured_sweep_wall_time": perf_counter() - sweep_start,
        },
        "configurations": results,
        "recommendation": recommendation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_results(results)
    if recommendation is None:
        print("No configuration completed as hard-valid with optical results.")
    else:
        print()
        print("Recommended candidate")
        print(json.dumps(recommendation, indent=2, sort_keys=True))
    print(f"\nJSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()
