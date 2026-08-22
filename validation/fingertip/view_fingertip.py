"""Load a LUMO fingertip mesh into Newton and visualize it."""

from __future__ import annotations

import newton
import newton.viewer
import warp as wp

from lumo.fingertip.fingertip import Fingertip
from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.mechanics import build_fingertip_newton_model
from lumo.mesh.fingertip_mesh import make_fingertip_mesh
from lumo.util.viewer_util import (
    configure_fingertip_camera,
    make_reference_lines,
)


_SILICONE_COLOR = (0.72, 0.92, 0.68)
_ALUMINUM_COLOR = wp.vec3(0.36, 0.39, 0.43)


def main() -> None:
    parameters = FingertipParameters()
    fingertip = Fingertip(parameters)

    mesh = make_fingertip_mesh(
        fingertip,
        extrusion_depth_mm=11.0,
        element_size_mm=1.0,
    )

    mechanics = build_fingertip_newton_model(
        mesh,
        gravity=0.0,
        carrier_color=_ALUMINUM_COLOR,
    )
    model = mechanics.model
    state = model.state()

    print("Newton fingertip")
    print("-----------------")
    print(f"mesh vertices:     {mesh.silicone.vertex_count}")
    print(f"mesh tetrahedra:   {mesh.silicone.tet_count}")
    print(
        f"bonded vertices:   {mechanics.bonded_particle_indices.shape[0]}"
    )
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
        # ViewerGL's public mesh API has no alpha channel. Render the
        # deforming silicone surface as a pale white overlay instead of the
        # default soft-body triangle pass so the carrier remains easy to see.
        viewer.show_triangles = False
        silicone_surface_indices = wp.array(
            mesh.silicone.surface_tri_indices,
            dtype=wp.int32,
        )

        reference_starts, reference_ends, reference_colors = (
            make_reference_lines()
        )
        viewer.log_lines(
            "/reference/grid_and_axes",
            reference_starts,
            reference_ends,
            reference_colors,
        )

        configure_fingertip_camera(viewer)

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
