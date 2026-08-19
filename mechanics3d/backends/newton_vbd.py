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
from mechanics3d.fingertip import FingertipMechanicsMesh
from mechanics3d.indentation import IndentationResult, IndentationSettings, RigidIndenter3D


# These are Newton implementation capacities for the current single
# kinematic-indenter scene, not public indentation physics settings.  The
# corresponding per-body lists skip kinematic bodies in Newton 1.4.
_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 1024
_RIGID_CONTACT_BUFFER_SIZE = 64


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


@wp.kernel
def _set_kinematic_body_state(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_index: int,
    pose: wp.transform,
    velocity: wp.spatial_vector,
) -> None:
    """Set one prescribed translation and spatial velocity on a kinematic body."""

    if wp.tid() == 0:
        body_q[body_index] = pose
        body_qd[body_index] = velocity


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


@dataclass
class _IndentationContext:
    prepared: FingertipMechanicsMesh
    indenter: RigidIndenter3D
    mechanics_settings: Mechanics3DSettings
    indentation_settings: IndentationSettings
    device: object
    model: object
    solver: object
    pipeline: object
    state_in: object
    state_out: object
    control: object
    contacts: object
    indenter_body: int
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


def _warp_pose(pose, *, device: object):
    translation_m = np.asarray(pose.translation_mm, dtype=np.float32) * 1.0e-3
    return wp.transform(
        wp.vec3(float(translation_m[0]), float(translation_m[1]), float(translation_m[2])),
        wp.quat(*pose.quaternion_xyzw),
    )


def _build_indentation_context(
    prepared_fingertip: FingertipMechanicsMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: Mechanics3DSettings,
    indentation_settings: IndentationSettings,
) -> _IndentationContext:
    """Build one soft-tet plus kinematic triangle-mesh contact scene."""

    wp.init()
    if not wp.is_device_available(mechanics_settings.device):
        raise RuntimeError(
            f"CUDA device {mechanics_settings.device!r} is not available"
        )
    device = wp.get_device(mechanics_settings.device)
    if not device.is_cuda:
        raise ValueError("mechanics3d requires a CUDA device, for example cuda:0")

    configured_support = tuple(sorted(mechanics_settings.fixed_vertex_indices))
    authoritative_support = tuple(sorted(prepared_fingertip.support_vertex_indices))
    if configured_support and configured_support != authoritative_support:
        raise ValueError(
            "mechanics3d indentation requires fixed_vertex_indices to be empty or "
            "equal to prepared_fingertip.support_vertex_indices; the authoritative "
            f"support is {authoritative_support!r}, received {configured_support!r}"
        )

    vertices_m = np.asarray(prepared_fingertip.tet_mesh.vertices, dtype=np.float32) * 1.0e-3
    rigid_vertices_m = np.asarray(indenter.mesh.vertices_mm, dtype=np.float32) * 1.0e-3
    rigid_mesh = newton.Mesh(
        rigid_vertices_m,
        np.asarray(indenter.mesh.faces, dtype=np.int32).reshape(-1),
        compute_inertia=False,
        is_solid=True,
    )
    rigid_cfg = newton.ModelBuilder.ShapeConfig(
        # The indenter is prescribed kinematic geometry, not a simulated mass.
        # Keep collision material properties here while preventing shape mass
        # accumulation into the kinematic body.
        density=0.0,
        ke=indentation_settings.soft_contact_ke,
        kd=indentation_settings.soft_contact_kd,
        mu=indentation_settings.soft_contact_mu,
        gap=0.0,
    )
    # The repository mesh is converted to metres above.  Use an explicit
    # contact-scale voxel size rather than deriving resolution from total
    # object size, so future large imported objects do not lose local contact
    # features.  Mesh-backed shapes own their cooked SDF on ``newton.Mesh``.
    object_extent_m = float(np.max(np.ptp(rigid_vertices_m, axis=0)))
    sdf_target_voxel_m = indentation_settings.rigid_sdf_target_voxel_mm * 1.0e-3
    rigid_mesh.build_sdf(
        device=device,
        narrow_band_range=(-object_extent_m, object_extent_m),
        target_voxel_size=sdf_target_voxel_m,
        margin=0.0,
    )
    # Newton 1.4 rejects cfg.sdf_* resolution fields for mesh-backed shapes.
    # Manual Mesh.build_sdf() is therefore authoritative; this flag requires
    # the already-cooked mesh SDF for full-surface contact.
    rigid_cfg.configure_sdf(
        force_sdf=True,
    )

    builder = newton.ModelBuilder(gravity=mechanics_settings.gravity)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices_m.tolist(),
        indices=prepared_fingertip.tet_mesh.tetrahedra.reshape(-1).tolist(),
        density=mechanics_settings.density,
        k_mu=mechanics_settings.k_mu,
        k_lambda=mechanics_settings.k_lambda,
        k_damp=mechanics_settings.k_damp,
        # The coarse tet mesh is handled by the full-surface edge/face path;
        # do not let Newton's metre-scale default particle radius create a
        # spurious contact shell around a millimetre-scale fingertip.
        particle_radius=0.0,
    )
    indenter_body = builder.add_body(
        xform=_warp_pose(indenter.initial_pose, device=device),
        # Kinematic bodies still need valid inertial data for Newton's model
        # validation.  This placeholder is locked and never participates in
        # the prescribed trajectory dynamics.
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
        lock_inertia=True,
        is_kinematic=True,
        label=f"{indenter.mesh.name}_kinematic",
    )
    builder.add_shape_mesh(
        body=indenter_body,
        mesh=rigid_mesh,
        cfg=rigid_cfg,
        label=f"{indenter.mesh.name}_collision_mesh",
    )
    builder.color()
    model = builder.finalize(device=device)

    if prepared_fingertip.support_vertex_indices:
        flags = np.asarray(model.particle_flags.numpy(), dtype=np.int32)
        flags[list(prepared_fingertip.support_vertex_indices)] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags = wp.array(flags, dtype=wp.int32, device=device)

    model.soft_contact_ke = indentation_settings.soft_contact_ke
    model.soft_contact_kd = indentation_settings.soft_contact_kd
    model.soft_contact_mu = indentation_settings.soft_contact_mu
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=indentation_settings.soft_contact_margin_mm * 1.0e-3,
        rigid_contact_max=_RIGID_CONTACT_BUFFER_SIZE,
        enable_rigid_soft_full_surface_contact=True,
        deterministic=True,
    )
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=mechanics_settings.iterations,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
        rigid_contact_hard=True,
        rigid_body_particle_contact_buffer_size=_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = pipeline.contacts()
    if (
        state_in.body_q is None
        or state_in.body_qd is None
        or state_out.body_q is None
        or state_out.body_qd is None
    ):
        raise RuntimeError("Newton state did not allocate kinematic body pose and velocity arrays")
    rest_vertices_m = np.asarray(state_in.particle_q.numpy(), dtype=np.float32).copy()
    return _IndentationContext(
        prepared=prepared_fingertip,
        indenter=indenter,
        mechanics_settings=mechanics_settings,
        indentation_settings=indentation_settings,
        device=device,
        model=model,
        solver=solver,
        pipeline=pipeline,
        state_in=state_in,
        state_out=state_out,
        control=control,
        contacts=contacts,
        indenter_body=indenter_body,
        rest_vertices_m=rest_vertices_m,
    )


def solve_newton_vbd_indentation(
    prepared_fingertip: FingertipMechanicsMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: Mechanics3DSettings,
    indentation_settings: IndentationSettings,
    *,
    viewer: object | None = None,
) -> IndentationResult:
    """Run translation-only kinematic rigid-mesh contact with standalone VBD.

    ``viewer`` is an optional Newton viewer owned by a Newton-specific example
    or application.  It is deliberately kept out of the neutral public
    indentation contract; when supplied, each accepted solver state and its
    contacts are sent to the viewer after the VBD step.
    """

    context = _build_indentation_context(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
    )
    if viewer is not None:
        viewer.set_model(context.model)
    previous_pose = indenter.initial_pose
    max_soft_contact_count = 0
    max_rigid_contact_count = 0
    max_soft_contact_overflow = 0
    max_rigid_contact_overflow = 0
    timestep_s = float(mechanics_settings.dt)
    for step in range(indentation_settings.load_steps):
        fraction = float(step + 1) / indentation_settings.load_steps
        target_pose = indenter.pose_at_travel(fraction * indentation_settings.travel_mm)
        delta_mm = np.asarray(target_pose.translation_mm) - np.asarray(previous_pose.translation_mm)
        velocity_m = delta_mm * 1.0e-3 / timestep_s
        pose = _warp_pose(target_pose, device=context.device)
        velocity = wp.spatial_vector(
            float(velocity_m[0]),
            float(velocity_m[1]),
            float(velocity_m[2]),
            0.0,
            0.0,
            0.0,
        )
        wp.launch(
            _set_kinematic_body_state,
            dim=1,
            inputs=[
                context.state_in.body_q,
                context.state_in.body_qd,
                context.indenter_body,
                pose,
                velocity,
            ],
            device=context.device,
        )
        # SolverVBD writes the accepted kinematic pose to its output state.
        # Seed both ping-pong states explicitly so every load step starts from
        # the prescribed pose even when the solver has no rigid-body solve to
        # perform for a kinematic body.
        wp.launch(
            _set_kinematic_body_state,
            dim=1,
            inputs=[
                context.state_out.body_q,
                context.state_out.body_qd,
                context.indenter_body,
                pose,
                velocity,
            ],
            device=context.device,
        )
        context.state_in.clear_forces()
        context.model.collide(
            context.state_in,
            context.contacts,
            collision_pipeline=context.pipeline,
            enable_rigid_soft_full_surface_contact=True,
        )
        context.solver.step(
            context.state_in,
            context.state_out,
            context.control,
            context.contacts,
            timestep_s,
        )
        soft_contact_overflow = int(
            context.solver.body_particle_contact_overflow_max.numpy()[0]
        )
        rigid_contact_overflow = int(
            context.solver.body_body_contact_overflow_max.numpy()[0]
        )
        max_soft_contact_overflow = max(max_soft_contact_overflow, soft_contact_overflow)
        max_rigid_contact_overflow = max(max_rigid_contact_overflow, rigid_contact_overflow)
        accepted_state = context.state_out
        if viewer is not None:
            viewer.begin_frame((step + 1) * timestep_s)
            viewer.log_state(accepted_state)
            viewer.log_contacts(context.contacts, accepted_state)
            viewer.end_frame()
        max_soft_contact_count = max(
            max_soft_contact_count,
            int(context.contacts.soft_contact_count.numpy()[0]),
        )
        if hasattr(context.contacts, "rigid_contact_count"):
            max_rigid_contact_count = max(
                max_rigid_contact_count,
                int(context.contacts.rigid_contact_count.numpy()[0]),
            )
        # Newton's counters are the only valid overflow signal here.  The
        # per-body lists skip static/kinematic bodies, so total soft-contact
        # records must not be compared with their capacities.
        if max_soft_contact_overflow > _RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE:
            raise RuntimeError(
                "Newton reported rigid body particle contact buffer overflow: "
                f"{max_soft_contact_overflow} > "
                f"{_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE}"
            )
        if max_rigid_contact_overflow > _RIGID_CONTACT_BUFFER_SIZE:
            raise RuntimeError(
                "Newton reported rigid contact buffer overflow: "
                f"{max_rigid_contact_overflow} > "
                f"{_RIGID_CONTACT_BUFFER_SIZE}"
            )
        context.state_in, context.state_out = context.state_out, context.state_in
        previous_pose = target_pose

    wp.synchronize_device(context.device)
    deformed_vertices = (
        np.asarray(context.state_in.particle_q.numpy(), dtype=np.float32).copy() * 1.0e3
    )
    rest_vertices = context.rest_vertices_m * 1.0e3
    final_body_translation_mm = (
        np.asarray(
            context.state_in.body_q.numpy()[context.indenter_body][:3],
            dtype=np.float32,
        )
        * 1.0e3
    )
    mechanics_result = Mechanics3DResult(
        rest_vertices=rest_vertices,
        deformed_vertices=deformed_vertices,
        tetrahedra=prepared_fingertip.tet_mesh.tetrahedra,
        steps=indentation_settings.load_steps,
    )
    return IndentationResult(
        mechanics_result=mechanics_result,
        final_indenter_pose=indenter.pose_at_travel(indentation_settings.travel_mm),
        diagnostics={
            "device": mechanics_settings.device,
            "full_surface_contact": True,
            "contact_buffer_status": "not_applicable_for_kinematic_indenter",
            "load_steps": indentation_settings.load_steps,
            "rigid_sdf_target_voxel_mm": indentation_settings.rigid_sdf_target_voxel_mm,
            "max_soft_contact_count": max_soft_contact_count,
            "max_rigid_contact_count": max_rigid_contact_count,
            "max_soft_contact_overflow": max_soft_contact_overflow,
            "max_rigid_contact_overflow": max_rigid_contact_overflow,
            "rigid_body_particle_contact_buffer_size": _RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
            "rigid_contact_buffer_size": _RIGID_CONTACT_BUFFER_SIZE,
            "final_body_x_mm": float(final_body_translation_mm[0]),
            "final_body_y_mm": float(final_body_translation_mm[1]),
            "final_body_z_mm": float(final_body_translation_mm[2]),
            "max_displacement_mm": float(np.max(np.linalg.norm(mechanics_result.displacement, axis=1))),
        },
    )


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
