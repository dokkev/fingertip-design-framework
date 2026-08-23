"""Run Newton and optical convergence studies for the sensing evaluator."""

from __future__ import annotations

import argparse
import json
from importlib.resources import as_file, files
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.mesh import FingertipMesh
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


_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK

_CONTACT_X_MM = (-7.5, 0.0, 7.5)
_SPHERE_RADIUS_M = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 5.0
# The settling study keeps 5 ms as the smallest duration that remained
# hard-valid at all three contact locations in the measured run.
_SETTLE_DURATION_S = 5.0e-3
_SETTLE_DURATIONS_S = (5.0e-3, 20.0e-3, 50.0e-3)
_MAX_SIM_TIME_S = 30.0
_MAX_SEARCH_ITERATIONS = 256
_MAX_BONDED_DRIFT_M = 1.0e-8
_MAX_CARRIER_PENETRATION_M = 1.0e-5

_BASELINE = {
    "sim_frequency": 1.0e3,
    "iterations": 10,
    "soft_contact_margin_m": 1.0e-4,
    # 4 MPa is the first candidate intended to pass all contact locations.
    # Keep 8 MPa in the stiffness sweep as the higher-stiffness comparison.
    "carrier_contact_stiffness_n_m": 4.0e6,
    "element_size_mm": 1.0,
}
_SWEEPS = (
    ("carrier_contact_stiffness_n_m", (4.0e6, 8.0e6)),
    ("sim_frequency", (500.0, 1000.0, 2000.0)),
    ("iterations", (10, 20)),
    ("element_size_mm", (1.0, 0.75)),
)

_FIXED_OPTICAL_SIDE_COUNT = 128
_RAY_SIDE_COUNTS = (128, 256)
_RAY_SEEDS = (20260823, 20260824, 20260825)
_BOUNCE_CAP = 24
_CARRIER_ALBEDO = 0.7
_DEFAULT_OUTPUT = Path("output/validation/sensing_convergence.json")

_EXPECTED_NUMERICAL_FAILURES = (
    "did not hold",
    "soft contacts before prescribed motion",
    "non-finite",
)


def _configurations() -> list[dict[str, object]]:
    configurations = [
        {
            "name": "baseline",
            "family": "baseline",
            "value": None,
            "parameters": dict(_BASELINE),
        }
    ]
    for family, values in _SWEEPS:
        baseline_value = _BASELINE[family]
        for value in values:
            if value == baseline_value:
                continue
            parameters = dict(_BASELINE)
            parameters[family] = value
            configurations.append(
                {
                    "name": f"{family}={value:g}",
                    "family": family,
                    "value": value,
                    "parameters": parameters,
                }
            )
    return configurations


def _make_trial(
    fingertip: Fingertip,
    sphere_urdf_path: Path,
    *,
    contact_x_mm: float,
    approach_speed_m_s: float,
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
        approach_speed_m_s=approach_speed_m_s,
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )


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
    depths_m = np.zeros(positions_m.shape[0], dtype=np.float64)
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
    """Measure exposed silicone surface area inside the carrier polygon."""
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

        triangle = Polygon(
            1.0e3 * triangle_positions[:, (0, 2)]
        )
        if triangle.is_empty or not triangle.is_valid or triangle.area == 0.0:
            continue
        overlap = triangle.intersection(carrier_cross_section)
        if overlap.is_empty or overlap.area == 0.0:
            continue

        candidate_points: list[tuple[float, float]] = []
        geometries = (
            overlap.geoms
            if hasattr(overlap, "geoms")
            else (overlap,)
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
        if not candidate_points:
            continue
        candidate_array = np.asarray(candidate_points, dtype=np.float64)
        depths_m[triangle_index] = float(
            1.0e-3
            * distance(
                carrier_cross_section.boundary,
                points(candidate_array[:, 0], candidate_array[:, 1]),
            ).max()
        )

    return depths_m


def _measure_trial(
    trial: DesignTrial,
    simulation: LumoSimulation,
    indenter: Indenter,
    *,
    contact_x_mm: float,
) -> tuple[dict[str, object], np.ndarray]:
    if (
        trial.final_tf is None
        or trial.travel_m is None
        or trial.reaction_force_n is None
        or trial.maximum_particle_speed_m_s is None
        or trial.force_change_n is None
    ):
        raise RuntimeError(f"{trial.name} completed without scalar results")

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

    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    surface_triangles = np.asarray(
        simulation.fingertip_mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    nonbonded = np.ones(len(positions_m), dtype=bool)
    nonbonded[bonded_indices] = False

    maximum_displacement_m = float("inf")
    maximum_bonded_drift_m = float("inf")
    maximum_carrier_penetration_m = float("inf")
    maximum_particle_carrier_penetration_m = float("inf")
    maximum_tet_center_carrier_penetration_m = float("inf")
    maximum_free_tet_center_carrier_penetration_m = float("inf")
    maximum_bond_adjacent_tet_center_carrier_penetration_m = float("inf")
    maximum_exposed_surface_carrier_penetration_m = float("inf")
    deep_particle_count = 0
    deep_free_tet_count = 0
    deep_bond_adjacent_tet_count = 0
    deep_exposed_surface_count = 0
    if finite_state:
        maximum_displacement_m = float(
            np.linalg.norm(
                positions_m - reference_positions_m,
                axis=1,
            ).max()
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
        tet_touches_bond = np.any(~nonbonded[tet_indices], axis=1)
        free_tet_depths_m = tet_depths_m[free_tet_mask]
        bond_adjacent_tet_depths_m = tet_depths_m[tet_touches_bond]
        # Surface triangles touching the perfect-bond interface remain
        # diagnostic geometry. Only fully free triangles enter acceptance.
        free_surface_mask = np.all(nonbonded[surface_triangles], axis=1)
        exposed_surface_depths_m = _carrier_surface_penetration_depths_m(
            positions_m,
            surface_triangles,
            free_surface_mask=free_surface_mask,
            carrier_cross_section=carrier_cross_section,
            carrier_y_limits_m=carrier_y_limits_m,
        )
        maximum_particle_carrier_penetration_m = float(
            particle_depths_m.max()
        )
        maximum_tet_center_carrier_penetration_m = float(tet_depths_m.max())
        maximum_free_tet_center_carrier_penetration_m = (
            float(free_tet_depths_m.max())
            if len(free_tet_depths_m)
            else 0.0
        )
        maximum_bond_adjacent_tet_center_carrier_penetration_m = (
            float(bond_adjacent_tet_depths_m.max())
            if len(bond_adjacent_tet_depths_m)
            else 0.0
        )
        maximum_exposed_surface_carrier_penetration_m = float(
            exposed_surface_depths_m.max()
        )
        # Surface overlap is reported as a discretization diagnostic. The hard
        # mechanics gate uses particle and fully-free tet penetration only.
        maximum_carrier_penetration_m = max(
            maximum_particle_carrier_penetration_m,
            maximum_free_tet_center_carrier_penetration_m,
        )
        deep_particle_count = int(
            np.count_nonzero(
                particle_depths_m > _MAX_CARRIER_PENETRATION_M
            )
        )
        deep_free_tet_count = int(
            np.count_nonzero(
                free_tet_depths_m > _MAX_CARRIER_PENETRATION_M
            )
        )
        deep_bond_adjacent_tet_count = int(
            np.count_nonzero(
                bond_adjacent_tet_depths_m > _MAX_CARRIER_PENETRATION_M
            )
        )
        deep_exposed_surface_count = int(
            np.count_nonzero(
                exposed_surface_depths_m > _MAX_CARRIER_PENETRATION_M
            )
        )

    indenter_contact_count = simulation.soft_contact_count(
        indenter.body_index
    )
    carrier_contact_count = simulation.soft_contact_count(
        simulation.fingertip_model.carrier_body
    )
    force_error_n = abs(trial.reaction_force_n - trial.target_force_n)
    failures = []
    if force_error_n > _FORCE_TOLERANCE_N:
        failures.append("force tolerance")
    if not finite_state:
        failures.append("non-finite state")
    if indenter_contact_count == 0:
        failures.append("no indenter contact")
    if maximum_bonded_drift_m > _MAX_BONDED_DRIFT_M:
        failures.append("bonded drift")
    if maximum_carrier_penetration_m > _MAX_CARRIER_PENETRATION_M:
        failures.append("carrier penetration")
    sphere_center_m = np.asarray(trial.final_tf, dtype=np.float64)[:3]
    led_center_y_m = 0.5 * float(
        reference_positions_m[:, 1].min() + reference_positions_m[:, 1].max()
    )
    led_position_m = np.array(
        (
            0.0,
            led_center_y_m,
            -1.0e-3
            * simulation.fingertip.parameters.geometry.stem_height_mm,
        ),
        dtype=np.float64,
    )
    led_distance_m = float(np.linalg.norm(sphere_center_m - led_position_m))
    if led_distance_m <= _SPHERE_RADIUS_M:
        failures.append("indenter touched LED source")

    result = {
        "contact_x_mm": contact_x_mm,
        "valid": not failures,
        "failure": ", ".join(failures),
        "settled_force_n": trial.reaction_force_n,
        "force_error_n": force_error_n,
        "travel_m": trial.travel_m,
        "maximum_displacement_m": maximum_displacement_m,
        "maximum_active_particle_speed_m_s": (
            trial.maximum_particle_speed_m_s
        ),
        "final_force_change_n": trial.force_change_n,
        "simulation_step_count": trial.step_count,
        "search_correction_count": trial.search_iteration_count,
        "indenter_contact_count": indenter_contact_count,
        "carrier_contact_count": carrier_contact_count,
        "maximum_carrier_penetration_m": maximum_carrier_penetration_m,
        "maximum_particle_carrier_penetration_m": (
            maximum_particle_carrier_penetration_m
        ),
        "maximum_tet_center_carrier_penetration_m": (
            maximum_tet_center_carrier_penetration_m
        ),
        "maximum_free_tet_center_carrier_penetration_m": (
            maximum_free_tet_center_carrier_penetration_m
        ),
        "maximum_bond_adjacent_tet_center_carrier_penetration_m": (
            maximum_bond_adjacent_tet_center_carrier_penetration_m
        ),
        "deep_particle_count": deep_particle_count,
        "deep_free_tet_count": deep_free_tet_count,
        "deep_bond_adjacent_tet_count": deep_bond_adjacent_tet_count,
        "maximum_exposed_surface_carrier_penetration_m": (
            maximum_exposed_surface_carrier_penetration_m
        ),
        "deep_exposed_surface_count": deep_exposed_surface_count,
        "maximum_bonded_drift_m": maximum_bonded_drift_m,
        "led_distance_m": led_distance_m,
    }
    return result, positions_m.copy()


def _is_expected_numerical_failure(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in _EXPECTED_NUMERICAL_FAILURES)


def _run_newton_configuration(
    fingertip: Fingertip,
    sphere_urdf_path: Path,
    configuration: dict[str, object],
    *,
    settle_duration_s: float = _SETTLE_DURATION_S,
) -> tuple[dict[str, object], dict[str, object] | None]:
    parameters = configuration["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError("configuration parameters must be a dictionary")

    sim_frequency = float(parameters["sim_frequency"])
    configuration = {
        **configuration,
        "settle_duration_s": settle_duration_s,
        "approach_speed_m_s": _APPROACH_SPEED_M_S,
        "approach_step_m": _APPROACH_SPEED_M_S / sim_frequency,
    }
    trials = tuple(
        _make_trial(
            fingertip,
            sphere_urdf_path,
            contact_x_mm=contact_x_mm,
            approach_speed_m_s=_APPROACH_SPEED_M_S,
        )
        for contact_x_mm in _CONTACT_X_MM
    )
    contact_x_by_trial = dict(zip(map(id, trials), _CONTACT_X_MM, strict=True))
    study = DesignStudy(
        fingertip,
        trials,
        sim_frequency=sim_frequency,
        force_tolerance_n=_FORCE_TOLERANCE_N,
        settle_duration_s=settle_duration_s,
        max_search_iterations=_MAX_SEARCH_ITERATIONS,
        element_size_mm=float(parameters["element_size_mm"]),
        iterations=int(parameters["iterations"]),
        soft_contact_margin_m=float(parameters["soft_contact_margin_m"]),
        carrier_contact_stiffness_n_m=float(
            parameters["carrier_contact_stiffness_n_m"]
        ),
    )

    mechanics: list[dict[str, object]] = []
    snapshots: dict[float, np.ndarray] = {}
    mesh: FingertipMesh | None = None
    reference_vertices: np.ndarray | None = None
    reference_triangles: np.ndarray | None = None

    def inspect_trial(
        trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal mesh, reference_vertices, reference_triangles
        current_reference = np.asarray(
            simulation.fingertip_mesh.silicone.vertices,
            dtype=np.float64,
        )
        current_triangles = np.asarray(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            dtype=np.int32,
        )
        if mesh is None:
            mesh = simulation.fingertip_mesh
            reference_vertices = current_reference.copy()
            reference_triangles = current_triangles.copy()
        elif (
            reference_vertices is None
            or reference_triangles is None
            or not np.allclose(
                current_reference,
                reference_vertices,
                rtol=0.0,
                atol=1.0e-7,
            )
            or not np.array_equal(current_triangles, reference_triangles)
        ):
            raise RuntimeError(
                "independent trials changed silicone topology or ordering"
            )

        contact_x_mm = contact_x_by_trial[id(trial)]
        result, snapshot = _measure_trial(
            trial,
            simulation,
            indenter,
            contact_x_mm=contact_x_mm,
        )
        mechanics.append(result)
        snapshots[contact_x_mm] = snapshot
        print(
            "    "
            f"x={float(result['contact_x_mm']):+4.1f} mm | "
            f"F={float(result['settled_force_n']):7.3f} N | "
            f"travel={1.0e3 * float(result['travel_m']):7.3f} mm | "
            "pen="
            f"{1.0e6 * float(result['maximum_carrier_penetration_m']):7.3f} um "
            "(particle/free-tet/bond-tet/free-surface="
            f"{int(result['deep_particle_count'])}/"
            f"{int(result['deep_free_tet_count'])}/"
            f"{int(result['deep_bond_adjacent_tet_count'])}/"
            f"{int(result['deep_exposed_surface_count'])}) | "
            f"{'PASS' if result['valid'] else 'INVALID'}",
            flush=True,
        )

    start = perf_counter()
    try:
        study.run(inspect_trial=inspect_trial)
        wp.synchronize()
    except RuntimeError as error:
        if not _is_expected_numerical_failure(error):
            raise
        record = {
            **configuration,
            "valid": False,
            "failure": f"{type(error).__name__}: {error}",
            "mechanics": mechanics,
            "wall_runtime_s": perf_counter() - start,
            "sensing": None,
        }
        return record, None

    valid = (
        len(mechanics) == len(_CONTACT_X_MM)
        and all(bool(result["valid"]) for result in mechanics)
    )
    failures = [
        f"x={float(result['contact_x_mm']):+g}: {result['failure']}"
        for result in mechanics
        if not result["valid"]
    ]
    record = {
        **configuration,
        "valid": valid,
        "failure": "; ".join(failures),
        "mechanics": mechanics,
        "wall_runtime_s": perf_counter() - start,
        "sensing": None,
    }
    if not valid:
        return record, None
    if mesh is None or reference_vertices is None or reference_triangles is None:
        raise RuntimeError("valid Newton configuration did not retain its mesh")

    bundle = {
        "mesh": mesh,
        "reference_vertices": reference_vertices,
        "contact_snapshots": snapshots,
    }
    return record, bundle


def _run_settling_study(
    fingertip: Fingertip,
    sphere_urdf_path: Path,
) -> list[dict[str, object]]:
    records = []
    for settle_duration_s in _SETTLE_DURATIONS_S:
        duration_ms = 1.0e3 * settle_duration_s
        print(
            f"[settling {duration_ms:g} ms] center and off-center contacts",
            flush=True,
        )
        configuration = {
            "name": f"settle_duration_ms={duration_ms:g}",
            "family": "settle_duration_s",
            "value": settle_duration_s,
            "parameters": dict(_BASELINE),
        }
        record, bundle = _run_newton_configuration(
            fingertip,
            sphere_urdf_path,
            configuration,
            settle_duration_s=settle_duration_s,
        )
        records.append(record)
        del bundle
    return records


def _print_settling_table(records: list[dict[str, object]]) -> None:
    print()
    print("Settling-duration study")
    print(
        f"{'duration[ms]':>12s} {'x[mm]':>7s} {'F[N]':>8s} "
        f"{'travel[mm]':>11s} {'disp[mm]':>10s} {'vmax[m/s]':>11s} "
        f"{'dF[N]':>10s} {'status':>8s}"
    )
    for configuration in records:
        duration_ms = 1.0e3 * float(configuration["settle_duration_s"])
        for result in configuration["mechanics"]:
            print(
                f"{duration_ms:12.1f} "
                f"{float(result['contact_x_mm']):+7.1f} "
                f"{float(result['settled_force_n']):8.3f} "
                f"{1.0e3 * float(result['travel_m']):11.3f} "
                f"{1.0e3 * float(result['maximum_displacement_m']):10.3f} "
                f"{float(result['maximum_active_particle_speed_m_s']):11.3e} "
                f"{float(result['final_force_change_n']):10.3e} "
                f"{'PASS' if result['valid'] else 'INVALID':>8s}"
            )


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
    probe_origin = (
        led.position_W_m - probe_distance_m * led.normal_W
    )[None, :]
    direction = led.normal_W[None, :]
    carrier_hit = scene.trace_closest(
        probe_origin,
        direction,
        mask=_CARRIER_MASK,
    )
    if not carrier_hit["hit"][0]:
        raise AssertionError("carrier probe did not find the LED boundary")
    hit_position = probe_origin[0] + carrier_hit["t"][0] * led.normal_W
    if not np.allclose(
        hit_position,
        led.position_W_m,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise AssertionError("carrier probe found the wrong LED boundary")
    safe_origin = safe_secondary_origins(carrier_hit, direction)[0]
    emission = led.emit(u1, u2)
    emission["origin_W_m"] = safe_origin
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
        led.normal_W[None, :],
        mask=_SILICONE_MASK,
    )[0]
    if not silicone_hit["hit"]:
        raise AssertionError(f"{state_label} has no silicone source path")
    normal_projection = float(np.dot(silicone_hit["normal_W"], led.normal_W))
    if abs(normal_projection) <= 1.0e-6:
        raise AssertionError(f"{state_label} has an ambiguous source interface")
    source_inside_silicone = normal_projection > 0.0
    if state_label == "no_contact" and not source_inside_silicone:
        raise AssertionError(f"{state_label} source normal faces away from silicone")
    return np.full(
        len(initial_hits),
        source_inside_silicone,
        dtype=bool,
    )


def _trace_observation(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    *,
    inside_silicone: bool | np.ndarray,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
) -> np.ndarray:
    optics = fingertip.parameters.optical
    result = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=inside_silicone,
        n_air=1.0,
        n_silicone=optics.refractive_index,
        extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_BOUNCE_CAP,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        mask=_ALL_MASK,
    )
    if not np.isclose(
        result.accounted_power,
        result.emitted_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("bounded transport failed energy closure")
    return side_view_observation(result.escaped_rays, fingertip=fingertip)


def _minimum_pair(descriptors: np.ndarray) -> tuple[float, tuple[int, int]]:
    best_distance = float("inf")
    best_pair = (-1, -1)
    for first in range(len(descriptors) - 1):
        for second in range(first + 1, len(descriptors)):
            distance_value = float(
                np.linalg.norm(
                    np.atleast_1d(descriptors[first] - descriptors[second])
                )
            )
            if distance_value < best_distance:
                best_distance = distance_value
                best_pair = (first, second)
    return best_distance, best_pair


def _pairwise_distances(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float64)
    distances = np.zeros((len(values), len(values)), dtype=np.float64)
    for first in range(len(values) - 1):
        for second in range(first + 1, len(values)):
            distances[first, second] = distances[second, first] = float(
                np.linalg.norm(
                    np.atleast_1d(values[first] - values[second])
                )
            )
    return distances


def _optical_samples(
    *,
    seed: int,
    ray_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return tuple(
        rng.random((_BOUNCE_CAP, ray_count))
        for _ in range(3)
    )


def _evaluate_sensing(
    scene: OptixScene,
    fingertip: Fingertip,
    bundle: dict[str, object],
    *,
    side_count: int,
    seed: int,
) -> dict[str, object]:
    mesh = bundle["mesh"]
    if not isinstance(mesh, FingertipMesh):
        raise TypeError("snapshot bundle has no FingertipMesh")
    reference_vertices = np.asarray(
        bundle["reference_vertices"],
        dtype=np.float64,
    )
    contact_snapshots = bundle["contact_snapshots"]
    if not isinstance(contact_snapshots, dict):
        raise TypeError("snapshot bundle contact states must be a dictionary")

    sample_i, sample_j = np.meshgrid(
        np.arange(side_count),
        np.arange(side_count),
        indexing="ij",
    )
    u1 = (sample_i.ravel() + 0.5) / side_count
    u2 = (sample_j.ravel() + 0.5) / side_count
    led = _make_led(fingertip, mesh)
    emission = _emit_from_stem_boundary(scene, led, u1, u2)
    ray_count = len(emission)
    (
        dielectric_branch_u,
        carrier_u1,
        carrier_u2,
    ) = _optical_samples(
        seed=seed,
        ray_count=ray_count,
    )

    labels = ["no_contact"] + [
        f"contact_x={contact_x_mm:+.1f}mm"
        for contact_x_mm in _CONTACT_X_MM
    ]
    states = [reference_vertices] + [
        np.asarray(contact_snapshots[contact_x_mm], dtype=np.float64)
        for contact_x_mm in _CONTACT_X_MM
    ]
    observations = []
    for label, vertices in zip(labels, states, strict=True):
        scene.update_silicone(vertices)
        inside_silicone = _source_inside_silicone(
            scene,
            led,
            emission,
            state_label=label,
        )
        if label == "no_contact" and not np.all(inside_silicone):
            raise AssertionError(
                "reference stem source is not touching silicone"
            )
        observations.append(
            _trace_observation(
                scene,
                fingertip,
                emission,
                inside_silicone=inside_silicone,
                dielectric_branch_u=dielectric_branch_u,
                carrier_u1=carrier_u1,
                carrier_u2=carrier_u2,
            )
        )

    response = np.stack(observations)
    if response.shape != (4, 4):
        raise AssertionError("side-view response does not have shape (4, 4)")
    intensity, spatial = sensing_descriptors(response)
    j_intensity, j_spatial = sensing_objectives(response)
    measured_intensity, intensity_pair = _minimum_pair(intensity)
    measured_spatial, spatial_pair = _minimum_pair(spatial)
    intensity_distances = _pairwise_distances(intensity)
    spatial_distances = _pairwise_distances(spatial)
    if not np.isclose(j_intensity, measured_intensity) or not np.isclose(
        j_spatial,
        measured_spatial,
    ):
        raise AssertionError("objective and worst-pair report disagree")
    if not np.allclose(intensity[0], 0.0, rtol=0.0, atol=1.0e-15):
        raise AssertionError("no-contact intensity descriptor is not zero")
    if not np.allclose(spatial.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("spatial descriptors are not normalized")
    return {
        "ray_count": ray_count,
        "seed": seed,
        "quadrant_responses": response.tolist(),
        "intensity_responses": intensity.tolist(),
        "spatial_responses": spatial.tolist(),
        "pairwise_intensity_distances": intensity_distances.tolist(),
        "pairwise_spatial_distances": spatial_distances.tolist(),
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
        "center_intensity_response": float(intensity[2]),
    }


def _relative_change(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return (value - baseline) / abs(baseline)


def _evaluate_configuration_sensing(
    fingertip: Fingertip,
    bundle: dict[str, object],
) -> dict[str, object]:
    mesh = bundle["mesh"]
    if not isinstance(mesh, FingertipMesh):
        raise TypeError("valid configuration has no FingertipMesh")
    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    start = perf_counter()
    sensing = _evaluate_sensing(
        scene,
        fingertip,
        bundle,
        side_count=_FIXED_OPTICAL_SIDE_COUNT,
        seed=_RAY_SEEDS[0],
    )
    sensing["wall_runtime_s"] = perf_counter() - start
    del scene
    return sensing


def _run_ray_convergence(
    fingertip: Fingertip,
    reference_bundle: dict[str, object],
) -> list[dict[str, object]]:
    mesh = reference_bundle["mesh"]
    if not isinstance(mesh, FingertipMesh):
        raise TypeError("reference snapshot has no FingertipMesh")
    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    convergence = []
    for side_count in _RAY_SIDE_COUNTS:
        ray_count = side_count**2
        print(f"[ray convergence] rays={ray_count}", flush=True)
        start = perf_counter()
        seed_results = []
        for seed in _RAY_SEEDS:
            print(f"    seed={seed}", flush=True)
            seed_results.append(
                _evaluate_sensing(
                    scene,
                    fingertip,
                    reference_bundle,
                    side_count=side_count,
                    seed=seed,
                )
            )
        j_intensity = np.array(
            [float(result["J_intensity"]) for result in seed_results]
        )
        j_spatial = np.array(
            [float(result["J_spatial"]) for result in seed_results]
        )
        center_response = np.array(
            [
                float(result["center_intensity_response"])
                for result in seed_results
            ]
        )
        convergence.append(
            {
                "ray_count": ray_count,
                "J_intensity_mean": float(j_intensity.mean()),
                "J_intensity_std": float(j_intensity.std(ddof=1)),
                "J_spatial_mean": float(j_spatial.mean()),
                "J_spatial_std": float(j_spatial.std(ddof=1)),
                "center_intensity_response_mean": float(
                    center_response.mean()
                ),
                "center_intensity_response_std": float(
                    center_response.std(ddof=1)
                ),
                "worst_intensity_pairs": [
                    result["worst_intensity_pair"]
                    for result in seed_results
                ],
                "worst_spatial_pairs": [
                    result["worst_spatial_pair"]
                    for result in seed_results
                ],
                "seed_results": seed_results,
                "wall_runtime_s": perf_counter() - start,
            }
        )
    del scene
    return convergence


def _print_newton_tables(configurations: list[dict[str, object]]) -> None:
    print()
    print("Newton mechanics sensitivity")
    print(
        f"{'configuration':44s} {'x[mm]':>7s} {'F[N]':>8s} "
        f"{'travel':>8s} {'disp':>8s} {'vmax':>9s} {'dF':>9s} "
        f"{'ticks':>7s} {'corr':>5s} {'Ci':>5s} {'Cc':>5s} "
        f"{'hard[um]':>8s} {'surf[um]':>8s} {'status':>8s}"
    )
    for configuration in configurations:
        mechanics = configuration["mechanics"]
        if not isinstance(mechanics, list) or not mechanics:
            print(
                f"{str(configuration['name']):44s} "
                f"{'-':>72s} {'FAIL':>8s}  {configuration['failure']}"
            )
            continue
        for result in mechanics:
            print(
                f"{str(configuration['name']):44s} "
                f"{float(result['contact_x_mm']):+7.1f} "
                f"{float(result['settled_force_n']):8.3f} "
                f"{1.0e3 * float(result['travel_m']):8.3f} "
                f"{1.0e3 * float(result['maximum_displacement_m']):8.3f} "
                f"{float(result['maximum_active_particle_speed_m_s']):9.2e} "
                f"{float(result['final_force_change_n']):9.2e} "
                f"{int(result['simulation_step_count']):7d} "
                f"{int(result['search_correction_count']):5d} "
                f"{int(result['indenter_contact_count']):5d} "
                f"{int(result['carrier_contact_count']):5d} "
                f"{1.0e6 * float(result['maximum_carrier_penetration_m']):8.3f} "
                f"{1.0e6 * float(result['maximum_exposed_surface_carrier_penetration_m']):8.3f} "
                f"{'PASS' if result['valid'] else 'INVALID':>8s}"
            )
        if configuration["failure"]:
            print(f"  failure: {configuration['failure']}")

    print()
    print("Exposed-surface overlap versus mesh size (diagnostic only)")
    print(
        f"{'element[mm]':>12s} {'x[mm]':>7s} {'surface[um]':>13s} "
        f"{'particle[um]':>13s} {'free-tet[um]':>13s}"
    )
    for configuration in configurations:
        if configuration["family"] not in ("baseline", "element_size_mm"):
            continue
        element_size_mm = float(
            configuration["parameters"]["element_size_mm"]
        )
        for result in configuration["mechanics"]:
            print(
                f"{element_size_mm:12.3f} "
                f"{float(result['contact_x_mm']):+7.1f} "
                f"{1.0e6 * float(result['maximum_exposed_surface_carrier_penetration_m']):13.3f} "
                f"{1.0e6 * float(result['maximum_particle_carrier_penetration_m']):13.3f} "
                f"{1.0e6 * float(result['maximum_free_tet_center_carrier_penetration_m']):13.3f}"
            )

    print()
    print("Newton sensing sensitivity at 16384 rays")
    print(
        f"{'configuration':44s} {'J_intensity':>13s} {'dJ_i[%]':>10s} "
        f"{'J_spatial':>13s} {'dJ_s[%]':>10s} {'status':>8s}"
    )
    for configuration in configurations:
        sensing = configuration["sensing"]
        if not isinstance(sensing, dict):
            status = "SKIPPED" if configuration["valid"] else "INVALID"
            print(
                f"{str(configuration['name']):44s} "
                f"{'-':>13s} {'-':>10s} {'-':>13s} {'-':>10s} "
                f"{status:>8s}"
            )
            if configuration.get("sensing_skip_reason"):
                print(f"  reason: {configuration['sensing_skip_reason']}")
            continue
        relative_i = sensing.get("relative_J_intensity_change")
        relative_s = sensing.get("relative_J_spatial_change")
        relative_i_text = (
            "-"
            if relative_i is None
            else f"{100.0 * float(relative_i):+.3f}"
        )
        relative_s_text = (
            "-"
            if relative_s is None
            else f"{100.0 * float(relative_s):+.3f}"
        )
        print(
            f"{str(configuration['name']):44s} "
            f"{float(sensing['J_intensity']):13.6e} "
            f"{relative_i_text:>10s} "
            f"{float(sensing['J_spatial']):13.6e} "
            f"{relative_s_text:>10s} "
            f"{'PASS':>8s}"
        )
        print(
            "  worst pairs: intensity="
            f"{tuple(sensing['worst_intensity_pair'])}, spatial="
            f"{tuple(sensing['worst_spatial_pair'])}"
        )


def _print_ray_table(convergence: list[dict[str, object]]) -> None:
    print()
    print("Optical ray-count convergence (three fixed seeds)")
    print(
        f"{'rays':>7s} {'J_intensity mean+-std':>27s} "
        f"{'J_spatial mean+-std':>27s} "
        f"{'center response mean+-std':>30s} {'wall[s]':>10s}"
    )
    for result in convergence:
        print(
            f"{int(result['ray_count']):7d} "
            f"{float(result['J_intensity_mean']):11.4e} +- "
            f"{float(result['J_intensity_std']):9.2e} "
            f"{float(result['J_spatial_mean']):11.4e} +- "
            f"{float(result['J_spatial_std']):9.2e} "
            f"{float(result['center_intensity_response_mean']):12.4e} +- "
            f"{float(result['center_intensity_response_std']):9.2e} "
            f"{float(result['wall_runtime_s']):10.2f}"
        )
        print(
            f"  intensity pairs={result['worst_intensity_pairs']} | "
            f"spatial pairs={result['worst_spatial_pairs']}"
        )
    low_result, high_result = convergence
    print(
        f"{int(low_result['ray_count'])} -> {int(high_result['ray_count'])}: "
        "dJ_intensity="
        f"{float(high_result['J_intensity_mean']) - float(low_result['J_intensity_mean']):+.6e}, "
        "dJ_spatial="
        f"{float(high_result['J_spatial_mean']) - float(low_result['J_spatial_mean']):+.6e}, "
        "dcenter="
        f"{float(high_result['center_intensity_response_mean']) - float(low_result['center_intensity_response_mean']):+.6e}"
    )


def _within_seed_uncertainty(
    first: dict[str, object],
    second: dict[str, object],
) -> bool:
    intensity_uncertainty = float(
        np.hypot(
            float(first["J_intensity_std"]),
            float(second["J_intensity_std"]),
        )
    )
    spatial_uncertainty = float(
        np.hypot(
            float(first["J_spatial_std"]),
            float(second["J_spatial_std"]),
        )
    )
    return (
        abs(
            float(first["J_intensity_mean"])
            - float(second["J_intensity_mean"])
        )
        <= intensity_uncertainty
        and abs(
            float(first["J_spatial_mean"])
            - float(second["J_spatial_mean"])
        )
        <= spatial_uncertainty
    )


def _mechanics_deltas(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, float]:
    baseline_results = {
        float(result["contact_x_mm"]): result
        for result in baseline["mechanics"]
    }
    candidate_results = {
        float(result["contact_x_mm"]): result
        for result in candidate["mechanics"]
    }
    if baseline_results.keys() != candidate_results.keys():
        raise RuntimeError("mechanics configurations have different contacts")

    fields = (
        ("force_n", "settled_force_n"),
        ("travel_m", "travel_m"),
        ("displacement_m", "maximum_displacement_m"),
        ("speed_m_s", "maximum_active_particle_speed_m_s"),
        ("force_change_n", "final_force_change_n"),
        ("penetration_m", "maximum_carrier_penetration_m"),
    )
    return {
        f"maximum_absolute_{label}_change": max(
            abs(
                float(candidate_results[contact_x_mm][field])
                - float(baseline_results[contact_x_mm][field])
            )
            for contact_x_mm in baseline_results
        )
        for label, field in fields
    }


def _recommendations(
    configurations: list[dict[str, object]],
    convergence: list[dict[str, object]],
    reference_name: str,
) -> dict[str, object]:
    by_name = {
        str(configuration["name"]): configuration
        for configuration in configurations
    }
    baseline = by_name[reference_name]
    if not baseline["valid"]:
        raise RuntimeError("reference configuration is not mechanics-valid")
    baseline_sensing = baseline["sensing"]
    if not isinstance(baseline_sensing, dict):
        raise RuntimeError("reference sensing is unavailable")

    by_count = {
        int(result["ray_count"]): result for result in convergence
    }
    if _within_seed_uncertainty(by_count[16384], by_count[65536]):
        recommended_rays = 16384
    else:
        recommended_rays = 65536

    reference_noise = by_count[16384]
    objective_uncertainty = (
        float(reference_noise["J_intensity_std"]),
        float(reference_noise["J_spatial_std"]),
    )

    def objectives_within_seed_uncertainty(name: str) -> bool:
        candidate = by_name[name]
        sensing = candidate["sensing"]
        if not candidate["valid"] or not isinstance(sensing, dict):
            return False
        return (
            abs(
                float(sensing["J_intensity"])
                - float(baseline_sensing["J_intensity"])
            )
            <= objective_uncertainty[0]
            and abs(
                float(sensing["J_spatial"])
                - float(baseline_sensing["J_spatial"])
            )
            <= objective_uncertainty[1]
        )

    comparison_names = {
        "carrier_contact_stiffness_n_m": (
            "baseline",
            "carrier_contact_stiffness_n_m=8e+06",
        ),
        "iterations": ("iterations=20",),
        "sim_frequency": ("sim_frequency=500", "sim_frequency=2000"),
        "element_size_mm": ("element_size_mm=0.75",),
    }
    baseline_component_evidence = {}
    for family, names in comparison_names.items():
        mechanics_valid = all(bool(by_name[name]["valid"]) for name in names)
        objective_stable = mechanics_valid and all(
            objectives_within_seed_uncertainty(name) for name in names
        )
        baseline_component_evidence[family] = {
            "comparison_configurations": list(names),
            "all_mechanics_valid": mechanics_valid,
            "objectives_within_seed_uncertainty": objective_stable,
            "mechanics_deltas": {
                name: _mechanics_deltas(baseline, by_name[name])
                for name in names
                if by_name[name]["valid"]
            },
        }
    baseline_stable = all(
        bool(evidence["all_mechanics_valid"])
        and bool(evidence["objectives_within_seed_uncertainty"])
        for evidence in baseline_component_evidence.values()
    )

    family_changes: dict[str, float] = {}
    for configuration in configurations[1:]:
        sensing = configuration["sensing"]
        if not isinstance(sensing, dict):
            continue
        family = str(configuration["family"])
        relative_changes = [
            abs(float(value))
            for value in (
                sensing.get("relative_J_intensity_change"),
                sensing.get("relative_J_spatial_change"),
            )
            if value is not None
        ]
        if not relative_changes:
            continue
        family_changes[family] = max(
            family_changes.get(family, 0.0),
            *relative_changes,
        )
    most_sensitive_sensing_parameter = (
        max(family_changes, key=family_changes.get)
        if family_changes
        else "unresolved"
    )

    mechanics_invalid_families: dict[str, list[str]] = {}
    for configuration in configurations[1:]:
        if configuration["valid"]:
            continue
        family = str(configuration["family"])
        mechanics_invalid_families.setdefault(family, []).append(
            str(configuration["name"])
        )

    stiffness_names = (
        "baseline",
        "carrier_contact_stiffness_n_m=8e+06",
    )
    stiffness_mechanics_valid = all(
        bool(by_name[name]["valid"]) for name in stiffness_names
    )
    stiffness_objectives_stable = stiffness_mechanics_valid and all(
        objectives_within_seed_uncertainty(name) for name in stiffness_names
    )
    if recommended_rays == 16384:
        ray_recommendation = (
            "16384 rays is sufficient: the 16384-to-65536 objective change "
            "is within the combined three-seed standard deviations."
        )
    else:
        ray_recommendation = (
            "65536 rays is recommended because the 16384-to-65536 objective "
            "change exceeds the combined three-seed standard deviations."
        )
    return {
        "recommended_newton_settings": dict(baseline["parameters"]),
        "newton_contract_status": (
            "stable" if baseline_stable else "candidate_not_frozen"
        ),
        "reference_configuration": reference_name,
        "newton_recommendation": (
            "Use the selected hard-valid reference for sensing evaluation: every compared "
            "setting passes the existing mechanics acceptance checks and its "
            "sensing objectives remain within the measured 16384-ray seed "
            "standard deviations. Mechanics scalar deltas are reported "
            "separately; no additional convergence threshold was invented."
            if baseline_stable
            else "Do not freeze the selected Newton reference automatically: "
            "at least one comparison either fails the existing mechanics "
                "checks or changes the sensing objectives beyond measured "
                "16384-ray seed variation. Inspect the reported deltas."
        ),
        "recommended_ray_count": recommended_rays,
        "ray_recommendation": ray_recommendation,
        "largest_valid_sensing_sensitivity_parameter": (
            most_sensitive_sensing_parameter
        ),
        "mechanics_invalid_families": mechanics_invalid_families,
        "baseline_component_evidence": baseline_component_evidence,
        "carrier_contact_stiffness_evidence": {
            "comparison_configurations": list(stiffness_names),
            "all_mechanics_valid": stiffness_mechanics_valid,
            "objectives_within_seed_uncertainty": (
                stiffness_objectives_stable
            ),
        },
        "comparison_noise_reference": {
            "ray_count": 16384,
            "J_intensity_std": objective_uncertainty[0],
            "J_spatial_std": objective_uncertainty[1],
        },
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete LUMO sensing convergence study."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"final JSON output path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    wall_start = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    configuration_specs = _configurations()
    configurations: list[dict[str, object]] = []
    reference_name: str | None = None
    reference_bundle: dict[str, object] | None = None
    reference_sensing: dict[str, object] | None = None

    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_10mm.urdf",
    )
    with as_file(sphere_resource) as sphere_urdf_path:
        settling_records = _run_settling_study(
            fingertip,
            sphere_urdf_path,
        )
        for index, configuration in enumerate(configuration_specs, start=1):
            print(
                f"[Newton {index}/{len(configuration_specs)}] "
                f"{configuration['name']}",
                flush=True,
            )
            record, bundle = _run_newton_configuration(
                fingertip,
                sphere_urdf_path,
                configuration,
            )
            if bundle is None:
                configurations.append(record)
                if record["failure"]:
                    print(
                        f"    continued after: {record['failure']}",
                        flush=True,
                    )
                continue

            print(
                f"[optical {index}/{len(configuration_specs)}] "
                f"{record['name']} | "
                f"rays={_FIXED_OPTICAL_SIDE_COUNT**2}",
                flush=True,
            )
            sensing = _evaluate_configuration_sensing(
                fingertip,
                bundle,
            )
            record["sensing"] = sensing
            if reference_sensing is None:
                reference_name = str(record["name"])
                reference_bundle = bundle
                reference_sensing = sensing
            else:
                sensing["relative_J_intensity_change"] = _relative_change(
                    float(sensing["J_intensity"]),
                    float(reference_sensing["J_intensity"]),
                )
                sensing["relative_J_spatial_change"] = _relative_change(
                    float(sensing["J_spatial"]),
                    float(reference_sensing["J_spatial"]),
                )
                del bundle
            configurations.append(record)

    study_complete = (
        reference_name is not None
        and reference_sensing is not None
        and reference_bundle is not None
    )
    ray_convergence: list[dict[str, object]] = []
    recommendations: dict[str, object] | None = None
    blocked_reason: str | None = None
    if study_complete:
        ray_convergence = _run_ray_convergence(fingertip, reference_bundle)
        del reference_bundle
        recommendations = _recommendations(
            configurations,
            ray_convergence,
            reference_name,
        )
    else:
        blocked_reason = (
            "no hard-valid Newton configuration was available; optical "
            "sensitivity and ray convergence were skipped"
        )

    _print_settling_table(settling_records)
    _print_newton_tables(configurations)
    if recommendations is None:
        print()
        print(f"Optical convergence: SKIPPED | {blocked_reason}")
    else:
        _print_ray_table(ray_convergence)
        print()
        print("Evidence-based recommendation")
        print(
            "  largest valid sensing sensitivity: "
            f"{recommendations['largest_valid_sensing_sensitivity_parameter']}"
        )
        print(
            "  mechanics-invalid families: "
            f"{recommendations['mechanics_invalid_families'] or 'none'}"
        )
        print(f"  {recommendations['newton_recommendation']}")
        for family, evidence in recommendations[
            "baseline_component_evidence"
        ].items():
            print(
                f"    {family}: mechanics_valid="
                f"{evidence['all_mechanics_valid']}, objective_stable="
                f"{evidence['objectives_within_seed_uncertainty']}"
            )
            for name, deltas in evidence["mechanics_deltas"].items():
                print(
                    f"      {name}: dF="
                    f"{deltas['maximum_absolute_force_n_change']:.3e} N, "
                    "dtravel="
                    f"{1.0e3 * deltas['maximum_absolute_travel_m_change']:.3e} mm, "
                    "ddisplacement="
                    f"{1.0e3 * deltas['maximum_absolute_displacement_m_change']:.3e} mm, "
                    "dpenetration="
                    f"{1.0e6 * deltas['maximum_absolute_penetration_m_change']:.3e} um"
                )
        stiffness_evidence = recommendations[
            "carrier_contact_stiffness_evidence"
        ]
        print(
            "  carrier-contact-stiffness sensing-evaluation evidence: "
            f"mechanics_valid={stiffness_evidence['all_mechanics_valid']}, "
            "objective_stable="
            f"{stiffness_evidence['objectives_within_seed_uncertainty']}"
        )
        print(f"  {recommendations['ray_recommendation']}")
        print(
            "  final evaluation contract candidate: Newton="
            f"{recommendations['recommended_newton_settings']}, "
            f"rays={recommendations['recommended_ray_count']}, "
            f"bounces={_BOUNCE_CAP}"
        )

    failed = [
        {
            "name": configuration["name"],
            "reason": configuration["failure"],
        }
        for configuration in configurations
        if not configuration["valid"]
    ]
    total_wall_runtime_s = perf_counter() - wall_start
    payload = {
        "schema_version": 1,
        "study": "sensing_convergence",
        "study_complete": study_complete,
        "blocked_reason": blocked_reason,
        "mechanics_reference_configuration": reference_name,
        "settling_study": settling_records,
        "fixed_contract": {
            "contact_x_mm": list(_CONTACT_X_MM),
            "target_force_n": _TARGET_FORCE_N,
            "force_tolerance_n": _FORCE_TOLERANCE_N,
            "settle_duration_s": _SETTLE_DURATION_S,
            "settle_durations_tested_s": list(_SETTLE_DURATIONS_S),
            "approach_speed_m_s": _APPROACH_SPEED_M_S,
            "approach_step_m_by_frequency": {
                str(int(frequency)): _APPROACH_SPEED_M_S / frequency
                for frequency in (500.0, 1000.0, 2000.0)
            },
            "carrier_albedo": _CARRIER_ALBEDO,
            "max_optical_bounces": _BOUNCE_CAP,
            "fixed_newton_sensing_ray_count": (
                _FIXED_OPTICAL_SIDE_COUNT**2
            ),
            "ray_side_counts": list(_RAY_SIDE_COUNTS),
            "ray_seeds": list(_RAY_SEEDS),
        },
        "newton_configurations": configurations,
        "ray_convergence": ray_convergence,
        "recommendations": recommendations,
        "failed_configurations": failed,
        "total_wall_runtime_s": total_wall_runtime_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  failed configurations: {len(failed)}")
    for failure in failed:
        print(f"    {failure['name']}: {failure['reason']}")
    print(f"  total wall runtime: {total_wall_runtime_s:.1f} s")
    print(f"  JSON: {args.output.resolve()}")
    if not study_complete:
        raise SystemExit(f"study incomplete: {blocked_reason}")


if __name__ == "__main__":
    main()
