"""Indent three spherical URDFs at three fingertip locations to 20 N."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import warp as wp
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.newton import Indenter
from lumo.simulation import LumoSimulation
from lumo.simulation.indentation import IndentationCase, IndentationStudy


_SIM_FREQUENCY_HZ = 1.0e3
_TRANSLATION_STEP_M = 2.5e-5
_INITIAL_CLEARANCE_M = 1.0e-3
_MAX_SIM_TIME_S = 30.0
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 0.1
_VELOCITY_TOLERANCE_M_S = 1.0e-4
_FORCE_CHANGE_TOLERANCE_N = 1.0e-2
_SETTLED_TICK_COUNT = 5
_MAX_SEARCH_ITERATIONS = 256
_MAX_BONDED_DRIFT_M = 1.0e-8
# Match the established flat-plate carrier-penetration acceptance threshold.
_MAX_CARRIER_PENETRATION_M = 1.0e-5
_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
)
_CONTACT_X_MM = (-7.5, 0.0, 7.5)


def _soft_contact_count_for_body(
    simulation: LumoSimulation,
    body_index: int,
) -> int:
    contact_count = int(
        simulation.contacts.soft_contact_count.numpy()[0]
    )
    shape_indices = simulation.contacts.soft_contact_shape.numpy()[
        :contact_count
    ]
    valid = shape_indices >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    return int(
        np.count_nonzero(
            shape_bodies[shape_indices[valid]] == body_index
        )
    )


def _make_case(
    fingertip: Fingertip,
    urdf_path: Path,
    urdf_filename: str,
    *,
    sphere_diameter_mm: float,
    contact_x_mm: float,
) -> IndentationCase:
    local_surface_z_mm = (
        fingertip.silicone.ellipse_center_z_mm
        - semiellipse_depth_at_x_mm(
            half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
            height_mm=fingertip.silicone.ellipse_radius_z_mm,
            x_mm=contact_x_mm,
        )
    )
    sphere_radius_m = 0.5e-3 * sphere_diameter_mm
    initial_center_z_m = (
        1.0e-3 * local_surface_z_mm
        - _INITIAL_CLEARANCE_M
        - sphere_radius_m
    )
    contact_x_m = 1.0e-3 * contact_x_mm
    initial_pose = wp.transform(
        wp.vec3(contact_x_m, 0.0, initial_center_z_m),
        wp.quat_identity(),
    )

    return IndentationCase(
        name=f"{Path(urdf_filename).stem}_x{contact_x_mm:+g}mm",
        urdf_path=urdf_path,
        initial_tf=initial_pose,
        translation_step_W_m=wp.vec3(
            0.0,
            0.0,
            _TRANSLATION_STEP_M,
        ),
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )


def _carrier_interior_depths_m(
    positions_m: np.ndarray,
    *,
    carrier_cross_section: Polygon,
    carrier_y_limits_m: tuple[float, float],
) -> np.ndarray:
    """Return cross-section penetration depth for points inside the carrier."""
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


def _validate_and_report(
    case: IndentationCase,
    simulation: LumoSimulation,
    indenter: Indenter,
) -> None:
    reaction_force_n = case.reaction_force_n
    if (
        reaction_force_n is None
        or case.travel_m is None
        or case.simulation_time_s is None
        or case.maximum_particle_speed_m_s is None
        or case.force_change_n is None
    ):
        raise RuntimeError(f"{case.name} has not completed")

    contact_x_mm = 1.0e3 * float(np.asarray(case.initial_tf)[0])
    force_error_n = abs(reaction_force_n - case.target_force_n)
    if force_error_n > _FORCE_TOLERANCE_N:
        raise RuntimeError(
            f"{case.name} missed the quasi-static force tolerance"
        )

    bonded_indices = (
        simulation.fingertip_model.bonded_particle_indices.numpy()
    )

    sphere_contact_count = _soft_contact_count_for_body(
        simulation,
        indenter.body_index,
    )
    if sphere_contact_count == 0:
        raise RuntimeError(
            f"{case.name} reached the force target without contact"
        )

    particle_q = simulation.state.particle_q.numpy()
    particle_qd = simulation.state.particle_qd.numpy()
    if not np.all(np.isfinite(particle_q)) or not np.all(
        np.isfinite(particle_qd)
    ):
        raise RuntimeError(
            f"{case.name} produced a non-finite silicone state"
        )

    nonbonded = np.ones(particle_q.shape[0], dtype=bool)
    nonbonded[bonded_indices] = False
    surface_indices = np.unique(
        np.asarray(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            dtype=np.int32,
        ).reshape(-1)
    )
    surface_indices = surface_indices[nonbonded[surface_indices]]
    tet_indices = np.asarray(
        simulation.fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
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
    carrier_depths_m = _carrier_interior_depths_m(
        particle_q,
        carrier_cross_section=carrier_cross_section,
        carrier_y_limits_m=carrier_y_limits_m,
    )
    carrier_depths_m[~nonbonded] = 0.0
    surface_carrier_depths_m = carrier_depths_m[surface_indices]
    tet_carrier_depths_m = _carrier_interior_depths_m(
        particle_q[tet_indices].mean(axis=1),
        carrier_cross_section=carrier_cross_section,
        carrier_y_limits_m=carrier_y_limits_m,
    )
    maximum_carrier_penetration_m = max(
        float(carrier_depths_m.max()),
        float(tet_carrier_depths_m.max()),
    )
    deep_nonbonded_count = int(
        np.count_nonzero(
            carrier_depths_m > _MAX_CARRIER_PENETRATION_M
        )
    )
    deep_surface_count = int(
        np.count_nonzero(
            surface_carrier_depths_m > _MAX_CARRIER_PENETRATION_M
        )
    )
    deep_tet_count = int(
        np.count_nonzero(
            tet_carrier_depths_m > _MAX_CARRIER_PENETRATION_M
        )
    )
    if deep_nonbonded_count or deep_surface_count or deep_tet_count:
        raise RuntimeError(
            f"{case.name} penetrated the carrier beyond the allowed "
            f"{_MAX_CARRIER_PENETRATION_M:.9e} m tolerance"
        )

    bonded_drift_m = np.linalg.norm(
        particle_q[bonded_indices]
        - simulation.fingertip_model.bonded_local_positions.numpy(),
        axis=1,
    )
    max_bonded_drift_m = float(bonded_drift_m.max())
    if max_bonded_drift_m > _MAX_BONDED_DRIFT_M:
        raise RuntimeError(
            f"{case.name} caused bonded silicone vertices to drift"
        )

    print(
        f"{case.name}: PASS | x={contact_x_mm:+.1f} mm | "
        f"F={reaction_force_n:.4f} N | error={force_error_n:.4f} N | "
        f"travel={1.0e3 * case.travel_m:.4f} mm | "
        f"ticks={case.step_count} | t={case.simulation_time_s:.3f} s | "
        f"search={case.search_iteration_count} | "
        f"vmax={case.maximum_particle_speed_m_s:.3e} m/s | "
        f"dF={case.force_change_n:.3e} N | contacts={sphere_contact_count} | "
        f"bond={max_bonded_drift_m:.3e} m | "
        f"carrier={maximum_carrier_penetration_m:.3e} m"
    )


def main() -> None:
    fingertip = Fingertip(FingertipParameters())
    resource_root = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
    )

    with ExitStack() as resources:
        cases = []
        for urdf_filename, sphere_diameter_mm in _SPHERES:
            urdf_path = resources.enter_context(
                as_file(resource_root.joinpath(urdf_filename))
            )
            for contact_x_mm in _CONTACT_X_MM:
                cases.append(
                    _make_case(
                        fingertip,
                        urdf_path,
                        urdf_filename,
                        sphere_diameter_mm=sphere_diameter_mm,
                        contact_x_mm=contact_x_mm,
                    )
                )

        study = IndentationStudy(
            fingertip,
            cases,
            sim_frequency=_SIM_FREQUENCY_HZ,
            force_tolerance_n=_FORCE_TOLERANCE_N,
            velocity_tolerance_m_s=_VELOCITY_TOLERANCE_M_S,
            force_change_tolerance_n=_FORCE_CHANGE_TOLERANCE_N,
            settled_tick_count=_SETTLED_TICK_COUNT,
            max_search_iterations=_MAX_SEARCH_ITERATIONS,
        )
        study.run(inspect_case=_validate_and_report)

    print("nine-case spherical indentation matrix: PASS")


if __name__ == "__main__":
    main()
