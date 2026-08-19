"""Newton 1.4 VBD backend for the isolated mechanics3d prototype."""

from __future__ import annotations

import numpy as np
import warp as wp
import newton

from mechanics3d.solve import Mechanics3DSettings
from mechanics3d.types import Mechanics3DResult, TetMeshData


def solve_newton_vbd(
    mesh: TetMeshData,
    settings: Mechanics3DSettings,
) -> Mechanics3DResult:
    """Solve a neutral tetrahedral mesh using Newton's supported VBD API."""

    wp.init()
    if not settings.device.startswith("cuda:"):
        raise ValueError("mechanics3d requires a CUDA device, for example cuda:0")
    if settings.device not in wp.get_cuda_devices():
        raise RuntimeError(f"CUDA device {settings.device!r} is not available")
    if any(index >= mesh.vertices.shape[0] for index in settings.fixed_vertex_indices):
        raise ValueError("fixed_vertex_indices contain an out-of-range vertex")

    builder = newton.ModelBuilder(gravity=settings.gravity)
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=mesh.vertices.tolist(),
        indices=mesh.tetrahedra.reshape(-1).tolist(),
        density=settings.density,
        k_mu=settings.k_mu,
        k_lambda=settings.k_lambda,
        k_damp=settings.k_damp,
    )
    builder.color()
    model = builder.finalize(device=settings.device)

    if settings.fixed_vertex_indices:
        flags = np.asarray(model.particle_flags.numpy(), dtype=np.int32)
        flags[list(settings.fixed_vertex_indices)] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags = wp.array(flags, dtype=wp.int32, device=settings.device)

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=settings.iterations,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    rest_vertices = np.asarray(state_in.particle_q.numpy(), dtype=np.float32).copy()

    for _ in range(settings.steps):
        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, settings.dt)
        state_in, state_out = state_out, state_in

    wp.synchronize_device(settings.device)
    deformed_vertices = np.asarray(state_in.particle_q.numpy(), dtype=np.float32).copy()
    return Mechanics3DResult(
        rest_vertices=rest_vertices,
        deformed_vertices=deformed_vertices,
        tetrahedra=mesh.tetrahedra,
        steps=settings.steps,
    )
