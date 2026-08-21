"""Load a LUMO fingertip mesh into Newton and visualize it."""

from __future__ import annotations

import newton
import newton.viewer
import warp as wp

from lumo.fingertip.fingertip import Fingertip
from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.mesh.fingertip_mesh import make_fingertip_mesh


def main() -> None:
    parameters = FingertipParameters()
    fingertip = Fingertip(parameters)

    mesh = make_fingertip_mesh(
        fingertip,
        extrusion_depth_mm=11.0,
        element_size_mm=1.0,
    )

    material = parameters.viscoelastic

    builder = newton.ModelBuilder(
        gravity=0.0
    )

    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        mesh=mesh,
        density=material.density_kg_m3,
        k_mu=material.k_mu_pa,
        k_lambda=material.k_lambda_pa,
        k_damp=material.damping,
    )

    model = builder.finalize(
        requires_grad=False,
    )
    state = model.state()

    print("Newton fingertip")
    print("-----------------")
    print(f"mesh vertices:     {mesh.vertex_count}")
    print(f"mesh tetrahedra:   {mesh.tet_count}")
    print(f"model particles:   {model.particle_count}")
    print(f"model tetrahedra:  {model.tet_count}")

    if model.particle_count != mesh.vertex_count:
        raise RuntimeError(
            "Newton particle count does not match mesh vertex count"
        )

    if model.tet_count != mesh.tet_count:
        raise RuntimeError(
            "Newton tetrahedron count does not match mesh tetrahedron count"
        )

    viewer = newton.viewer.ViewerGL()

    try:
        viewer.set_model(model)

        while viewer.is_running():
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()

    finally:
        viewer.close()


if __name__ == "__main__":
    main()