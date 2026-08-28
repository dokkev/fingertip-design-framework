"""View the inverse-relative trajectory for one angled indentation."""

from __future__ import annotations

from importlib.resources import as_file, files

import newton
import newton.viewer
import numpy as np
import warp as wp

from lumo.fingertip import (
    SILICONE_MECHANICS,
    Fingertip,
    FingertipGeometry,
    FingertipParameters,
)
from lumo.newton import Indenter
from lumo.simulation import IndentationTrial, LumoSimulation
from lumo.util.viewer_util import configure_fingertip_camera


# Dragon Skin optimized trial 117.
_GEOMETRY_MM = (14.5, 4.0, 5.0, 12.5, 5.0)
_SPHERE_DIAMETER_MM = 20.0
_CONTACT_Y_MM = -5.5
_FINGERTIP_ANGLE_DEG = 30.0
_INITIAL_CLEARANCE_M = 10.0e-3
_APPROACH_SPEED_M_S = 5.0e-3
_TARGET_FORCES_N = (5.0, 10.0)
_MAX_SIM_TIME_S = 60.0
_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_SOFT_CONTACT_MARGIN_M = 1.0e-4
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547
_REPORT_INTERVAL_TICKS = 20

# A physical +theta fingertip rotation is represented with the fingertip fixed
# by rotating the sphere center and motion direction through -theta about this
# longitudinal world line.
_ROTATION_PIVOT_XZ_M = (0.0, 0.0)


def _fingertip() -> Fingertip:
    flat_height, ellipse_height, stem_width, stem_height, void_width = (
        _GEOMETRY_MM
    )
    return Fingertip(
        FingertipParameters(
            geometry=FingertipGeometry(
                flat_pad_height_mm=flat_height,
                semiellipse_height_mm=ellipse_height,
                stem_width_mm=stem_width,
                stem_height_mm=stem_height,
                void_width_mm=void_width,
            ),
            mechanics=SILICONE_MECHANICS,
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


def _trajectory(
    fingertip: Fingertip,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
    ordinary_center_m = np.array(
        (
            0.0,
            1.0e-3 * _CONTACT_Y_MM,
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
        ),
        dtype=np.float64,
    )
    pivot_m = np.array(
        (
            _ROTATION_PIVOT_XZ_M[0],
            ordinary_center_m[1],
            _ROTATION_PIVOT_XZ_M[1],
        ),
        dtype=np.float64,
    )
    center_m = pivot_m + _rotate_about_y(
        ordinary_center_m - pivot_m,
        -angle_deg,
    )
    direction = _rotate_about_y(
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
        -angle_deg,
    )
    direction /= np.linalg.norm(direction)
    return center_m, direction


def _zero_contact_travel_m(
    fingertip: Fingertip,
    angle_deg: float,
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
    boundary_xz_m = 1.0e-3 * np.concatenate(
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
    direction_xz = _rotate_about_y(
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
        -angle_deg,
    )[[0, 2]]
    projected_m = boundary_xz_m @ direction_xz
    perpendicular_squared_m2 = (
        np.einsum("ij,ij->i", boundary_xz_m, boundary_xz_m)
        - projected_m**2
    )
    radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
    eligible = perpendicular_squared_m2 <= radius_m**2
    if not np.any(eligible):
        raise RuntimeError("angled sphere path misses the fingertip outline")
    first_center_coordinate_m = float(
        np.min(
            projected_m[eligible]
            - np.sqrt(radius_m**2 - perpendicular_squared_m2[eligible])
        )
    )
    initial_center_coordinate_m = (
        fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m
    )
    travel_m = first_center_coordinate_m - initial_center_coordinate_m
    if travel_m <= 0.0:
        raise RuntimeError("common angled sphere start is not clear")
    return travel_m


def _assert_trajectory(fingertip: Fingertip) -> None:
    ordinary_center_m, ordinary_direction = _trajectory(fingertip, 0.0)
    radius_m = 0.5e-3 * _SPHERE_DIAMETER_MM
    expected_ordinary_center_m = np.array(
        (
            0.0,
            1.0e-3 * _CONTACT_Y_MM,
            fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        ordinary_center_m,
        expected_ordinary_center_m,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        ordinary_direction,
        (0.0, 0.0, 1.0),
        rtol=0.0,
        atol=1.0e-15,
    )

    angled_center_m, angled_direction = _trajectory(
        fingertip,
        _FINGERTIP_ANGLE_DEG,
    )
    np.testing.assert_allclose(
        np.linalg.norm(angled_direction),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    pivot_m = np.array(
        (0.0, expected_ordinary_center_m[1], 0.0),
        dtype=np.float64,
    )
    for travel_m in (0.0, 8.0e-3, 12.0e-3):
        ordinary_sample_m = expected_ordinary_center_m + travel_m * np.array(
            (0.0, 0.0, 1.0),
            dtype=np.float64,
        )
        expected_angled_sample_m = pivot_m + _rotate_about_y(
            ordinary_sample_m - pivot_m,
            -_FINGERTIP_ANGLE_DEG,
        )
        actual_angled_sample_m = angled_center_m + travel_m * angled_direction
        np.testing.assert_allclose(
            actual_angled_sample_m,
            expected_angled_sample_m,
            rtol=0.0,
            atol=1.0e-14,
        )


def _render(
    viewer: newton.viewer.ViewerGL,
    simulation: LumoSimulation,
) -> None:
    viewer.begin_frame(simulation.time_s)
    viewer.log_state(simulation.state)
    viewer.log_contacts(simulation.contacts, simulation.state)
    viewer.end_frame()


def _log_trajectory(
    viewer: newton.viewer.ViewerGL,
    fingertip: Fingertip,
    initial_center_m: np.ndarray,
    direction: np.ndarray,
) -> None:
    ordinary_center_m, ordinary_direction = _trajectory(fingertip, 0.0)
    line_length_m = 18.0e-3
    guide_offset_m = 13.0e-3 * np.array(
        (direction[2], 0.0, -direction[0]),
        dtype=np.float64,
    )
    starts = wp.array(
        (
            wp.vec3(*initial_center_m),
            wp.vec3(*ordinary_center_m),
            wp.vec3(0.0, -32.5e-3, 0.0),
            wp.vec3(*(initial_center_m + guide_offset_m)),
        ),
        dtype=wp.vec3,
    )
    ends = wp.array(
        (
            wp.vec3(*(initial_center_m + line_length_m * direction)),
            wp.vec3(*(ordinary_center_m + line_length_m * ordinary_direction)),
            wp.vec3(0.0, 32.5e-3, 0.0),
            wp.vec3(
                *(initial_center_m + guide_offset_m + line_length_m * direction)
            ),
        ),
        dtype=wp.vec3,
    )
    colors = wp.array(
        (
            wp.vec3(1.0, 0.7, 0.0),
            wp.vec3(0.4, 0.4, 0.4),
            wp.vec3(0.1, 0.4, 1.0),
            wp.vec3(1.0, 0.2, 0.8),
        ),
        dtype=wp.vec3,
    )
    viewer.log_lines("/reference/trajectories", starts, ends, colors)


def main() -> None:
    fingertip = _fingertip()
    _assert_trajectory(fingertip)
    initial_center_m, direction = _trajectory(
        fingertip,
        _FINGERTIP_ANGLE_DEG,
    )
    zero_contact_travel_m = _zero_contact_travel_m(
        fingertip,
        _FINGERTIP_ANGLE_DEG,
    )

    sphere_resource = files("lumo.assets.objects.urdf").joinpath(
        "sphere_20mm.urdf"
    )
    with as_file(sphere_resource) as sphere_path:
        trial = IndentationTrial(
            name="dragon_trial117_theta+30_y-5.5",
            urdf_path=sphere_path,
            initial_tf=wp.transform(
                wp.vec3(*initial_center_m),
                wp.quat_identity(),
            ),
            motion_direction_W=wp.vec3(*direction),
            approach_speed_m_s=_APPROACH_SPEED_M_S,
            max_sim_time_s=_MAX_SIM_TIME_S,
            initial_clearance_m=zero_contact_travel_m,
        )
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
        indenter = Indenter.add_urdf(
            builder,
            trial.urdf_path,
            tf=trial.initial_tf,
            contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
            contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
        )

    simulation = LumoSimulation(
        fingertip,
        builder=builder,
        sim_frequency=_SIM_FREQUENCY_HZ,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        soft_contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        soft_contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    )
    initial_contact_count = simulation.soft_contact_count(indenter.body_index)

    print("Angled indentation viewer", flush=True)
    print(f"  fingertip theta: {_FINGERTIP_ANGLE_DEG:+.1f} deg", flush=True)
    print("  inverse relative rotation: -30.0 deg about world Y", flush=True)
    print("  pivot axis: X=0, Z=0 (longitudinal Y line)", flush=True)
    print(f"  contact Y: {_CONTACT_Y_MM:+.1f} mm", flush=True)
    print(
        "  initial sphere center [m]: "
        f"{np.array2string(initial_center_m, precision=9)}",
        flush=True,
    )
    print(
        "  motion direction W: "
        f"{np.array2string(direction, precision=9)}",
        flush=True,
    )
    print(
        f"  analytic zero-contact travel: "
        f"{1.0e3 * zero_contact_travel_m:.6f} mm",
        flush=True,
    )
    print(f"  initial sphere contacts: {initial_contact_count}", flush=True)
    if initial_contact_count != 0:
        raise RuntimeError("sphere has contacts before prescribed motion")

    viewer = newton.viewer.ViewerGL(vsync=False)
    try:
        viewer.set_model(simulation.fingertip_model.model)
        configure_fingertip_camera(viewer)
        # Look along Y so the X-Z trajectory angle is directly visible. The
        # shared helper still supplies the centimeter-scale interaction
        # sensitivities used by the other fingertip viewers.
        viewer.set_camera(wp.vec3(0.0, -0.12, -0.01), 0.0, 0.0)
        viewer.camera.look_at(
            np.array((0.0, 1.0e-3 * _CONTACT_Y_MM, -0.015))
        )
        _log_trajectory(viewer, fingertip, initial_center_m, direction)
        _render(viewer, simulation)

        step_distance_m = trial.approach_speed_m_s / simulation.sim_frequency
        max_steps = int(trial.max_sim_time_s * simulation.sim_frequency)
        threshold_index = 0
        reaction_force_n = 0.0

        for step_index in range(1, max_steps + 1):
            if not viewer.is_running():
                print("viewer closed before 10 N was reached", flush=True)
                return

            travel_m = step_index * step_distance_m
            center_m = initial_center_m + travel_m * direction
            simulation.apply_indenter_pose(
                indenter,
                wp.transform(wp.vec3(*center_m), wp.quat_identity()),
            )
            simulation.step()
            reaction_force_n = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=trial.motion_direction_W,
            )
            _render(viewer, simulation)

            if (
                step_index % _REPORT_INTERVAL_TICKS == 0
                or reaction_force_n >= _TARGET_FORCES_N[threshold_index]
            ):
                indentation_m = travel_m - zero_contact_travel_m
                print(
                    f"t={simulation.time_s:6.3f} s | "
                    f"travel={1.0e3 * travel_m:8.4f} mm | "
                    f"indent={1.0e3 * indentation_m:8.4f} mm | "
                    f"F={reaction_force_n:9.4f} N | "
                    f"contacts={simulation.soft_contact_count(indenter.body_index)}",
                    flush=True,
                )

            if reaction_force_n < _TARGET_FORCES_N[threshold_index]:
                continue

            print(
                f"  crossed {_TARGET_FORCES_N[threshold_index]:g} N at "
                f"{1.0e3 * (travel_m - zero_contact_travel_m):.4f} mm "
                "physical indentation",
                flush=True,
            )
            threshold_index += 1
            if threshold_index < len(_TARGET_FORCES_N):
                continue

            body_pose = simulation.state.body_q.numpy()[indenter.body_index]
            actual_center_m = np.asarray(body_pose[:3], dtype=np.float64)
            line_error_m = float(np.linalg.norm(actual_center_m - center_m))
            if line_error_m > 1.0e-7:
                raise RuntimeError(
                    "Newton indenter pose left the prescribed trajectory: "
                    f"error={line_error_m:.9e} m"
                )
            print(
                "10 N reached; final indenter pose is now completely frozen.\n"
                f"  final center [m]: {np.array2string(actual_center_m, precision=9)}\n"
                f"  trajectory error: {line_error_m:.3e} m\n"
                "  close the viewer to finish.",
                flush=True,
            )
            try:
                while viewer.is_running():
                    _render(viewer, simulation)
            except KeyboardInterrupt:
                print("viewer interrupted after frozen-pose inspection", flush=True)
            return

        raise RuntimeError(
            "sphere did not reach 10 N within "
            f"{trial.max_sim_time_s:g} s; last force was "
            f"{reaction_force_n:.9e} N"
        )
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
