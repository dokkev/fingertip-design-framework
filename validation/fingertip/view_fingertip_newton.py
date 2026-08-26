"""Visualize the complete fingertip in Newton ViewerGL."""

from __future__ import annotations

import newton
import newton.viewer
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import make_fingertip_mesh
from lumo.newton import FingertipNewtonModel, build_fingertip_newton_model
from lumo.util.viewer_util import make_reference_lines


_SILICONE_COLOR = (0.72, 0.92, 0.68)
_LED_COLOR = wp.vec3(0.15, 0.95, 0.25)


def main() -> None:
    fingertip_mesh = make_fingertip_mesh(
        Fingertip(),
        element_size_mm=1.0,
    )
    fingertip_newton: FingertipNewtonModel = build_fingertip_newton_model(
        fingertip_mesh,
    )
    model = fingertip_newton.model
    state = model.state()

    print("Newton full five-LED fingertip")
    print("-------------------------------")
    print(f"silicone vertices:   {fingertip_mesh.silicone.vertex_count}")
    print(f"silicone tetrahedra: {fingertip_mesh.silicone.tet_count}")
    print(f"bonded vertices:     {len(fingertip_mesh.bonded_vertex_indices)}")
    print("LED centers [m]:")
    for index, center_m in enumerate(fingertip_mesh.led_centers_m, start=1):
        print(f"  LED {index}: {center_m.tolist()}")
    print("Green markers show the five LED reference positions.")

    if model.particle_count != fingertip_mesh.silicone.vertex_count:
        raise RuntimeError("Newton particle count does not match the full mesh")
    if model.tet_count != fingertip_mesh.silicone.tet_count:
        raise RuntimeError(
            "Newton tetrahedron count does not match the full mesh"
        )

    silicone_surface_indices = wp.array(
        fingertip_mesh.silicone.surface_tri_indices,
        dtype=wp.int32,
    )
    led_centers = wp.array(
        fingertip_mesh.led_centers_m,
        dtype=wp.vec3,
    )
    led_colors = wp.full(
        len(fingertip_mesh.led_centers_m),
        _LED_COLOR,
        dtype=wp.vec3,
    )
    reference_starts, reference_ends, reference_colors = make_reference_lines()
    silicone_vertices = state.particle_q.numpy()
    scene_center = 0.5 * (
        silicone_vertices.min(axis=0) + silicone_vertices.max(axis=0)
    )

    viewer = newton.viewer.ViewerGL(vsync=False)
    try:
        viewer.set_model(model)
        viewer.show_triangles = False
        viewer.log_lines(
            "/reference/grid_and_axes",
            reference_starts,
            reference_ends,
            reference_colors,
        )
        viewer.log_points(
            "/reference/led_centers",
            led_centers,
            radii=1.2e-3,
            colors=led_colors,
        )
        camera_position = scene_center + np.array(
            (0.09, 0.11, 0.065),
            dtype=np.float32,
        )
        viewer.set_camera(wp.vec3(*camera_position), 0.0, 0.0)
        viewer.camera.look_at(scene_center)
        viewer.camera.fov = 32.0
        viewer.camera_speed = 0.01
        viewer.gui._camera_orbit_sensitivity = 0.015
        viewer.gui._camera_dolly_scroll_sensitivity = 0.02
        viewer.gui._camera_dolly_drag_sensitivity = 0.0015

        while viewer.is_running():
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.log_mesh(
                "/fingertip/silicone_surface",
                state.particle_q,
                silicone_surface_indices,
                color=_SILICONE_COLOR,
                roughness=0.85,
                metallic=0.0,
                backface_culling=False,
            )
            viewer.end_frame()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
