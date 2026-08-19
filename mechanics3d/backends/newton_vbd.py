"""Newton 1.4 VBD backend for the isolated mechanics3d prototype."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import warp as wp
import newton

from mechanics3d.solve import Mechanics3DSettings
from mechanics3d.load import ParticleLoad
from mechanics3d.types import Mechanics3DResult, TetMeshData


@wp.kernel
def _apply_prescribed_displacement(
    particle_q: wp.array(dtype=wp.vec3f),
    rest_vertices: wp.array(dtype=wp.vec3f),
    vertex_indices: wp.array(dtype=wp.int32),
    displacement_m: wp.vec3f,
) -> None:
    """Apply a benchmark-local kinematic displacement to selected vertices."""

    index = vertex_indices[wp.tid()]
    particle_q[index] = rest_vertices[index] + displacement_m


@wp.kernel
def _add_particle_forces(
    particle_f: wp.array(dtype=wp.vec3f),
    vertex_indices: wp.array(dtype=wp.int32),
    forces_n: wp.array(dtype=wp.vec3f),
    scale: float,
) -> None:
    """Add one neutral per-particle force contribution in Newtons."""

    index = vertex_indices[wp.tid()]
    particle_f[index] += forces_n[wp.tid()] * scale


@dataclass
class _VBDContext:
    mesh: TetMeshData
    settings: Mechanics3DSettings
    device: object
    model: object
    solver: object
    state_in: object
    state_out: object
    rest_state: object
    control: object
    contacts: object
    rest_vertices_m: np.ndarray


def _build_vbd_context(mesh: TetMeshData, settings: Mechanics3DSettings):
    wp.init()
    if not wp.is_device_available(settings.device):
        raise RuntimeError(f"CUDA device {settings.device!r} is not available")
    device = wp.get_device(settings.device)
    if not device.is_cuda:
        raise ValueError("mechanics3d requires a CUDA device, for example cuda:0")
    if any(index >= mesh.vertices.shape[0] for index in settings.fixed_vertex_indices):
        raise ValueError("fixed_vertex_indices contain an out-of-range vertex")

    # The repository-facing neutral geometry uses millimetres. Newton uses SI
    # metres for particle positions, so convert only at this backend boundary.
    vertices_m = np.asarray(mesh.vertices, dtype=np.float32) * 1.0e-3
    builder = newton.ModelBuilder(gravity=settings.gravity)
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=vertices_m.tolist(),
        indices=mesh.tetrahedra.reshape(-1).tolist(),
        density=settings.density,
        k_mu=settings.k_mu,
        k_lambda=settings.k_lambda,
        k_damp=settings.k_damp,
    )
    builder.color()
    model = builder.finalize(device=device)

    if settings.fixed_vertex_indices:
        flags = np.asarray(model.particle_flags.numpy(), dtype=np.int32)
        flags[list(settings.fixed_vertex_indices)] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags = wp.array(flags, dtype=wp.int32, device=device)

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
    rest_vertices_m = np.asarray(state_in.particle_q.numpy(), dtype=np.float32).copy()
    rest_state = model.state()
    rest_state.assign(state_in)
    return _VBDContext(
        mesh=mesh,
        settings=settings,
        device=device,
        model=model,
        solver=solver,
        state_in=state_in,
        state_out=state_out,
        rest_state=rest_state,
        control=control,
        contacts=contacts,
        rest_vertices_m=rest_vertices_m,
    )


def _reset_vbd_context(context: _VBDContext) -> None:
    context.state_in.assign(context.rest_state)
    context.state_out.assign(context.rest_state)
    wp.synchronize_device(context.device)


def _solve_vbd_context(
    context: _VBDContext,
    mesh: TetMeshData,
    settings: Mechanics3DSettings,
    load: ParticleLoad,
) -> tuple[Mechanics3DResult, dict[str, float | int | str]]:
    """Reset and solve one force-ramped load without rebuilding the model."""

    if mesh is not context.mesh or settings != context.settings:
        raise ValueError("persistent session topology/settings do not match its build")

    reset_started = time.perf_counter()
    _reset_vbd_context(context)
    reset_wall_s = time.perf_counter() - reset_started

    index_device = None
    force_device = None
    if len(load.vertex_indices):
        index_device = wp.array(
            np.asarray(load.vertex_indices, dtype=np.int32),
            dtype=wp.int32,
            device=context.device,
        )
        force_device = wp.array(
            np.asarray(load.forces_n, dtype=np.float32),
            dtype=wp.vec3f,
            device=context.device,
        )

    solve_started = time.perf_counter()
    for step in range(load.load_steps):
        context.state_in.clear_forces()
        if index_device is not None and force_device is not None:
            scale = np.float32((step + 1) / load.load_steps)
            wp.launch(
                _add_particle_forces,
                dim=len(load.vertex_indices),
                inputs=[context.state_in.particle_f, index_device, force_device, scale],
                device=context.device,
            )
        context.solver.step(
            context.state_in,
            context.state_out,
            context.control,
            context.contacts,
            settings.dt,
        )
        context.state_in, context.state_out = context.state_out, context.state_in
    wp.synchronize_device(context.device)
    solver_loop_wall_s = time.perf_counter() - solve_started

    deformed_vertices = (
        np.asarray(context.state_in.particle_q.numpy(), dtype=np.float32).copy() * 1.0e3
    )
    rest_vertices = context.rest_vertices_m * 1.0e3
    total_wall_s = time.perf_counter() - reset_started
    result = Mechanics3DResult(
        rest_vertices=rest_vertices,
        deformed_vertices=deformed_vertices,
        tetrahedra=mesh.tetrahedra,
        steps=load.load_steps,
    )
    return result, {
        "device": settings.device,
        "reset_wall_s": float(reset_wall_s),
        "solver_loop_wall_s": float(solver_loop_wall_s),
        "per_solve_wall_s": float(total_wall_s),
        "load_steps": load.load_steps,
    }


def solve_newton_vbd(
    mesh: TetMeshData,
    settings: Mechanics3DSettings,
) -> Mechanics3DResult:
    """Solve a neutral tetrahedral mesh using Newton's supported VBD API."""

    context = _build_vbd_context(mesh, settings)

    for _ in range(settings.steps):
        context.state_in.clear_forces()
        context.model.collide(context.state_in, context.contacts)
        context.solver.step(
            context.state_in,
            context.state_out,
            context.control,
            context.contacts,
            settings.dt,
        )
        context.state_in, context.state_out = context.state_out, context.state_in

    wp.synchronize_device(context.device)
    deformed_vertices = (
        np.asarray(context.state_in.particle_q.numpy(), dtype=np.float32).copy() * 1.0e3
    )
    rest_vertices = context.rest_vertices_m * 1.0e3
    return Mechanics3DResult(
        rest_vertices=rest_vertices,
        deformed_vertices=deformed_vertices,
        tetrahedra=mesh.tetrahedra,
        steps=settings.steps,
    )


def solve_newton_vbd_prescribed(
    mesh: TetMeshData,
    settings: Mechanics3DSettings,
    *,
    vertex_indices: Sequence[int],
    displacement_mm: Sequence[float],
    load_steps: int,
) -> tuple[Mechanics3DResult, dict[str, float | int | str]]:
    """Run a minimal prescribed-vertex timing solve without contact search.

    This is intentionally a benchmark-only kinematic patch.  It does not
    model an indenter, collision, contact, or reaction force.
    """

    indices = tuple(sorted(int(index) for index in vertex_indices))
    if not indices:
        raise ValueError("vertex_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("vertex_indices must be unique")
    if any(index < 0 or index >= mesh.vertices.shape[0] for index in indices):
        raise ValueError("vertex_indices contain an out-of-range vertex")
    displacement = np.asarray(displacement_mm, dtype=np.float32)
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise ValueError("displacement_mm must contain three finite values")
    if int(load_steps) < 1:
        raise ValueError("load_steps must be positive")
    load_steps = int(load_steps)

    # Device initialization is deliberately outside the reported mechanics
    # interval.  All actual model/solver GPU work is synchronized below.
    wp.init()
    if not wp.is_device_available(settings.device):
        raise RuntimeError(f"CUDA device {settings.device!r} is not available")
    device = wp.get_device(settings.device)
    if not device.is_cuda:
        raise ValueError("mechanics3d requires a CUDA device, for example cuda:0")
    wp.synchronize_device(device)
    mechanics_start = time.perf_counter()
    build_start = mechanics_start
    context = _build_vbd_context(mesh, settings)
    wp.synchronize_device(context.device)
    model_build_wall_s = time.perf_counter() - build_start

    rest_vertices_device = wp.array(context.rest_vertices_m, dtype=wp.vec3f, device=device)
    index_device = wp.array(np.asarray(indices, dtype=np.int32), dtype=wp.int32, device=device)
    displacement_m = displacement * 1.0e-3

    solver_start = time.perf_counter()
    for step in range(load_steps):
        context.state_in.clear_forces()
        # `contacts` is intentionally never populated: this benchmark has no
        # collision/contact path.  The selected vertices are imposed after the
        # unconstrained VBD step as a deterministic kinematic patch.
        context.solver.step(
            context.state_in,
            context.state_out,
            context.control,
            context.contacts,
            settings.dt,
        )
        fraction = np.float32((step + 1) / load_steps)
        target_displacement = wp.vec3f(
            displacement_m[0] * fraction,
            displacement_m[1] * fraction,
            displacement_m[2] * fraction,
        )
        wp.launch(
            _apply_prescribed_displacement,
            dim=len(indices),
            inputs=[context.state_out.particle_q, rest_vertices_device, index_device, target_displacement],
            device=device,
        )
        context.state_in, context.state_out = context.state_out, context.state_in
    wp.synchronize_device(device)
    solver_loop_wall_s = time.perf_counter() - solver_start

    deformed_vertices = np.asarray(context.state_in.particle_q.numpy(), dtype=np.float32).copy() * 1.0e3
    rest_vertices = context.rest_vertices_m * 1.0e3
    total_mechanics_wall_s = time.perf_counter() - mechanics_start
    result = Mechanics3DResult(
        rest_vertices=rest_vertices,
        deformed_vertices=deformed_vertices,
        tetrahedra=mesh.tetrahedra,
        steps=load_steps,
    )
    timing: dict[str, float | int | str] = {
        "device": settings.device,
        "model_build_wall_s": float(model_build_wall_s),
        "solver_loop_wall_s": float(solver_loop_wall_s),
        "total_mechanics_wall_s": float(total_mechanics_wall_s),
        "load_steps": load_steps,
    }
    return result, timing
