"""Load a LUMO fingertip mesh into Newton and visualize it."""

from __future__ import annotations

import newton
import newton.viewer
import warp as wp

from lumo.fingertip.fingertip import Fingertip
from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.mesh.fingertip_mesh import make_fingertip_mesh


def _make_reference_lines() -> tuple[wp.array, wp.array, wp.array]:
    """Return an X-Y floor grid and colored LUMO-coordinate axes."""
    grid_extent_m = 0.04
    grid_step_m = 0.005
    floor_z_m = -0.02

    starts: list[wp.vec3] = []
    ends: list[wp.vec3] = []
    colors: list[wp.vec3] = []
    grid_color = wp.vec3(0.25, 0.25, 0.28)

    grid_count = int(grid_extent_m / grid_step_m)
    for index in range(-grid_count, grid_count + 1):
        coordinate_m = index * grid_step_m

        starts.append(
            wp.vec3(-grid_extent_m, coordinate_m, floor_z_m)
        )
        ends.append(
            wp.vec3(grid_extent_m, coordinate_m, floor_z_m)
        )
        colors.append(grid_color)

        starts.append(
            wp.vec3(coordinate_m, -grid_extent_m, floor_z_m)
        )
        ends.append(
            wp.vec3(coordinate_m, grid_extent_m, floor_z_m)
        )
        colors.append(grid_color)

    axis_length_m = 0.025
    starts.extend(
        (
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        )
    )
    ends.extend(
        (
            wp.vec3(axis_length_m, 0.0, 0.0),
            wp.vec3(0.0, axis_length_m, 0.0),
            wp.vec3(0.0, 0.0, axis_length_m),
        )
    )
    colors.extend(
        (
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        )
    )

    return (
        wp.array(starts, dtype=wp.vec3),
        wp.array(ends, dtype=wp.vec3),
        wp.array(colors, dtype=wp.vec3),
    )


def main() -> None:
    parameters = FingertipParameters()
    fingertip = Fingertip(parameters)

    mesh = make_fingertip_mesh(
        fingertip,
        extrusion_depth_mm=11.0,
        element_size_mm=1.0,
    )

    material = mesh.fingertip.parameters.viscoelastic

    builder = newton.ModelBuilder(
        gravity=0.0
    )

    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        mesh=mesh.silicone,
        density=material.density_kg_m3,
        k_mu=material.k_mu_pa,
        k_lambda=material.k_lambda_pa,
        k_damp=material.damping,
    )

    # The carrier is a rigid world-attached shape.  It comes from the same
    # FingertipMesh as the silicone volume, so both geometries share one
    # canonical LUMO coordinate frame.
    builder.add_shape_mesh(
        body=-1,
        mesh=mesh.carrier,
        color=wp.vec3(0.85, 0.45, 0.10),
        label="fingertip_carrier",
    )

    model = builder.finalize(
        requires_grad=False,
    )
    state = model.state()

    print("Newton fingertip")
    print("-----------------")
    print(f"mesh vertices:     {mesh.silicone.vertex_count}")
    print(f"mesh tetrahedra:   {mesh.silicone.tet_count}")
    print(f"model particles:   {model.particle_count}")
    print(f"model tetrahedra:  {model.tet_count}")

    if model.particle_count != mesh.silicone.vertex_count:
        raise RuntimeError(
            "Newton particle count does not match mesh vertex count"
        )

    if model.tet_count != mesh.silicone.tet_count:
        raise RuntimeError(
            "Newton tetrahedron count does not match mesh tetrahedron count"
        )

    viewer = newton.viewer.ViewerGL()

    try:
        viewer.set_model(model)
        reference_starts, reference_ends, reference_colors = (
            _make_reference_lines()
        )
        viewer.log_lines(
            "/reference/grid_and_axes",
            reference_starts,
            reference_ends,
            reference_colors,
        )

        # The fingertip is only a few centimeters across.
        viewer.set_camera(
            wp.vec3(0.08, -0.08, 0.05),
            -25.0,
            135.0,
        )
        viewer.camera.fov = 25.0
        viewer.camera_speed = 0.03

        # ViewerGL does not expose mouse sensitivities as public settings.
        # This script is a validation-only viewer, so keep the local tuning
        # explicit while leaving production code independent of private API.
        viewer.gui._camera_orbit_sensitivity = 0.03
        viewer.gui._camera_dolly_scroll_sensitivity = 0.03
        viewer.gui._camera_dolly_drag_sensitivity = 0.003

        while viewer.is_running():
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()

    finally:
        viewer.close()


if __name__ == "__main__":
    main()
