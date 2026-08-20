"""Newton 1.4 VBD backend for production mechanics execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import warp as wp
import newton
from shapely import wkt as shapely_wkt
from shapely.geometry import Point

from physics.solve import NewtonSettings, PhysicsDependencyError
from physics.load import ParticleLoad
from physics.types import NewtonResult, TetMeshData
from physics.fingertip import PreparedFingertipMesh
from physics.indentation import (
    IndentationCheckpoint,
    IndentationResult,
    IndentationSettings,
    IndentationTrajectoryResult,
    RigidIndenter3D,
    checkpoint_step_schedule,
)
from mesh.rigid_object import RigidObjectMesh, RigidPose3D

if TYPE_CHECKING:
    from contact.first_contact import FirstContactResult


# These are Newton implementation capacities for the current single
# kinematic-indenter scene, not public indentation physics settings.  The
# corresponding per-body lists skip kinematic bodies in Newton 1.4.
_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 1024
_RIGID_CONTACT_BUFFER_SIZE = 64


def _require_cuda_device(device_name: str):
    """Initialize Warp and require the configured CUDA device."""

    try:
        wp.init()
        if not wp.is_device_available(device_name):
            raise PhysicsDependencyError(
                f"CUDA device {device_name!r} is not available"
            )
        device = wp.get_device(device_name)
    except PhysicsDependencyError:
        raise
    except (ImportError, OSError, RuntimeError) as exc:
        raise PhysicsDependencyError(
            "Warp/CUDA runtime could not initialize: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not device.is_cuda:
        raise PhysicsDependencyError(
            f"physics requires a CUDA device, received {device_name!r}"
        )
    return device


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
    settings: NewtonSettings
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
    prepared: PreparedFingertipMesh
    indenter: RigidIndenter3D
    mechanics_settings: NewtonSettings
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
    indenter_shape: int
    carrier_shape: int | None
    carrier_mesh: RigidObjectMesh | None
    void_bottom_vertex_indices: frozenset[int]
    rest_vertices_m: np.ndarray


def _signed_carrier_clearance_mm(
    vertices_mm: np.ndarray,
    surface_triangles: np.ndarray,
    carrier_mesh: RigidObjectMesh,
) -> float:
    """Return the minimum signed 2D clearance of a void surface to the carrier.

    The carrier is an 11 mm extrusion of the authoritative rigid XY polygon.
    Positive values mean separation, zero is contact, and negative values mean
    that a void-surface vertex has entered the carrier cross-section.  This is
    an independent diagnostic; Newton contact provenance remains authoritative
    for whether a contact record was generated.
    """

    cross_section_wkt = carrier_mesh.metadata.get("cross_section_wkt")
    if not isinstance(cross_section_wkt, str) or not cross_section_wkt:
        return float("nan")
    polygon = shapely_wkt.loads(cross_section_wkt)
    z_min = float(carrier_mesh.metadata.get("z_min_mm", np.min(carrier_mesh.vertices_mm[:, 2])))
    z_max = float(carrier_mesh.metadata.get("z_max_mm", np.max(carrier_mesh.vertices_mm[:, 2])))
    local_indices = np.unique(np.asarray(surface_triangles, dtype=np.int64).reshape(-1))
    points = np.asarray(vertices_mm, dtype=float)[local_indices]
    clearances: list[float] = []
    for x_mm, y_mm, z_mm in points:
        if z_mm < z_min or z_mm > z_max:
            continue
        point = Point(float(x_mm), float(y_mm))
        distance = float(point.distance(polygon.boundary))
        clearances.append(-distance if polygon.covers(point) else distance)
    return float(min(clearances)) if clearances else float("nan")


def _contact_shape_details(
    context: _IndentationContext,
) -> tuple[int, int, int, int, frozenset[int]]:
    """Return contact counts and exact soft-particle carrier provenance."""

    count = int(context.contacts.soft_contact_count.numpy()[0])
    shapes = np.asarray(context.contacts.soft_contact_shape.numpy()[:count], dtype=np.int64)
    sphere_count = int(np.count_nonzero(shapes == context.indenter_shape))
    carrier_count = (
        int(np.count_nonzero(shapes == context.carrier_shape))
        if context.carrier_shape is not None
        else 0
    )
    soft_indices = np.asarray(
        context.contacts.soft_contact_indices.numpy()[:count], dtype=np.int64
    )
    void_bottom_mask = np.any(
        np.isin(soft_indices, tuple(context.void_bottom_vertex_indices))
        & (soft_indices >= 0),
        axis=1,
    ) if count else np.zeros(0, dtype=bool)
    carrier_void_bottom_count = (
        int(np.count_nonzero((shapes == context.carrier_shape) & void_bottom_mask))
        if context.carrier_shape is not None
        else 0
    )
    carrier_vertex_indices: frozenset[int] = frozenset()
    if count and context.carrier_shape is not None:
        rows = soft_indices[shapes == context.carrier_shape]
        all_carrier_vertices = frozenset(
            int(index)
            for index in np.asarray(rows, dtype=np.int64).reshape(-1)
            if int(index) >= 0
        )
        # The current carrier-contact contract is the compliant pad's
        # semantic void-bottom interface.  Other carrier soft-contact records
        # can involve the bonded/support region and must not be promoted to an
        # optical void triangle merely because the static carrier is nearby.
        carrier_vertex_indices = frozenset(
            all_carrier_vertices.intersection(context.void_bottom_vertex_indices)
        )
    rigid_count = int(context.contacts.rigid_contact_count.numpy()[0])
    rigid_pairs = 0
    if context.carrier_shape is not None and rigid_count:
        shape0 = np.asarray(context.contacts.rigid_contact_shape0.numpy()[:rigid_count], dtype=np.int64)
        shape1 = np.asarray(context.contacts.rigid_contact_shape1.numpy()[:rigid_count], dtype=np.int64)
        rigid_pairs = int(
            np.count_nonzero(
                ((shape0 == context.indenter_shape) & (shape1 == context.carrier_shape))
                | ((shape0 == context.carrier_shape) & (shape1 == context.indenter_shape))
            )
        )
    return (
        sphere_count,
        carrier_count,
        carrier_void_bottom_count,
        rigid_pairs,
        carrier_vertex_indices,
    )


def _contact_shape_counts(context: _IndentationContext) -> tuple[int, int, int, int]:
    """Return sphere/carrier records, void-bottom records, and rigid pairs."""

    return _contact_shape_details(context)[:4]


def _build_vbd_context(mesh: TetMeshData, settings: NewtonSettings):
    device = _require_cuda_device(settings.device)
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
    settings: NewtonSettings,
    load: ParticleLoad,
) -> tuple[NewtonResult, dict[str, float | int | str]]:
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
    result = NewtonResult(
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
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings,
    indentation_settings: IndentationSettings,
    *,
    initial_pose: RigidPose3D | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidObjectMesh | None = None,
) -> _IndentationContext:
    """Build one soft-tet plus kinematic sphere and optional static carrier."""

    for name, mesh in (
        ("visual_carrier_mesh", visual_carrier_mesh),
        ("rigid_carrier_mesh", rigid_carrier_mesh),
    ):
        if mesh is not None and not isinstance(mesh, RigidObjectMesh):
            raise TypeError(f"{name} must be a RigidObjectMesh or None")
    if initial_pose is None:
        initial_pose = indenter.initial_pose
    if not isinstance(initial_pose, RigidPose3D):
        raise TypeError("initial_pose must be RigidPose3D or None")

    device = _require_cuda_device(mechanics_settings.device)

    configured_support = tuple(sorted(mechanics_settings.fixed_vertex_indices))
    authoritative_support = tuple(sorted(prepared_fingertip.support_vertex_indices))
    if configured_support and configured_support != authoritative_support:
        raise ValueError(
            "physics indentation requires fixed_vertex_indices to be empty or "
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
    carrier_shape = None
    if rigid_carrier_mesh is not None:
        carrier_vertices_m = (
            np.asarray(rigid_carrier_mesh.vertices_mm, dtype=np.float32) * 1.0e-3
        )
        carrier_mesh = newton.Mesh(
            carrier_vertices_m,
            np.asarray(rigid_carrier_mesh.faces, dtype=np.int32).reshape(-1),
            compute_inertia=False,
            is_solid=True,
        )
        carrier_extent_m = float(np.max(np.ptp(carrier_vertices_m, axis=0)))
        carrier_mesh.build_sdf(
            device=device,
            narrow_band_range=(-carrier_extent_m, carrier_extent_m),
            target_voxel_size=sdf_target_voxel_m,
            margin=0.0,
        )
        carrier_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=indentation_settings.soft_contact_ke,
            kd=indentation_settings.soft_contact_kd,
            mu=indentation_settings.soft_contact_mu,
            gap=0.0,
            has_shape_collision=False,
            has_particle_collision=True,
            collision_group=1,
            is_visible=True,
        )
        carrier_cfg.configure_sdf(force_sdf=True)
        carrier_shape = builder.add_shape_mesh(
            body=-1,
            mesh=carrier_mesh,
            cfg=carrier_cfg,
            color=wp.vec3(0.68, 0.70, 0.74),
            label="distal_phalanx_carrier_collision_enabled",
        )

    if visual_carrier_mesh is not None:
        carrier_vertices_m = (
            np.asarray(visual_carrier_mesh.vertices_mm, dtype=np.float32) * 1.0e-3
        )
        carrier_mesh = newton.Mesh(
            carrier_vertices_m,
            np.asarray(visual_carrier_mesh.faces, dtype=np.int32).reshape(-1),
            compute_inertia=False,
            is_solid=True,
        )
        carrier_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
            collision_group=0,
            is_visible=True,
        )
        # Newton body -1 is static world geometry.  The collision flags are
        # explicitly disabled so this separate shape can only be rendered.
        builder.add_shape_mesh(
            body=-1,
            mesh=carrier_mesh,
            cfg=carrier_cfg,
            color=wp.vec3(0.68, 0.70, 0.74),
            label="distal_phalanx_carrier_render_only",
        )
    indenter_body = builder.add_body(
        xform=_warp_pose(initial_pose, device=device),
        # Kinematic bodies still need valid inertial data for Newton's model
        # validation.  This placeholder is locked and never participates in
        # the prescribed trajectory dynamics.
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
        lock_inertia=True,
        is_kinematic=True,
        label=f"{indenter.mesh.name}_kinematic",
    )
    indenter_shape = builder.add_shape_mesh(
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
        indenter_shape=indenter_shape,
        carrier_shape=carrier_shape,
        carrier_mesh=rigid_carrier_mesh,
        void_bottom_vertex_indices=(
            frozenset(
                np.unique(
                    np.asarray(prepared_fingertip.surface_triangles["void_bottom"], dtype=np.int64)
                    .reshape(-1)
                ).tolist()
            )
            if "void_bottom" in prepared_fingertip.surface_triangles
            else frozenset()
        ),
        rest_vertices_m=rest_vertices_m,
    )


def _solve_newton_vbd_indentation_path(
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings,
    indentation_settings: IndentationSettings,
    *,
    viewer: object | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidObjectMesh | None = None,
    first_contact: FirstContactResult | None = None,
) -> IndentationTrajectoryResult:
    """Run translation-only kinematic rigid-mesh contact with standalone VBD.

    ``viewer`` is an optional Newton viewer owned by a Newton-specific example
    or application.  It is deliberately kept out of the neutral public
    indentation contract; when supplied, each accepted solver state and its
    contacts are sent to the viewer after the VBD step.

    ``visual_carrier_mesh`` is a viewer-only mesh and never participates in
    collision. ``rigid_carrier_mesh`` is a separate explicit collision-enabled
    static world mesh. The two arguments are intentionally not interchangeable.
    """

    checkpoint_travels = (float(indentation_settings.travel_mm),)
    checkpoint_fractions = (1.0,)
    if indentation_settings.travel_mm == 0.0:
        schedule = tuple(
            (0.0, step, step)
            for step in range(1, indentation_settings.load_steps + 1)
        )
    else:
        schedule = checkpoint_step_schedule(
            checkpoint_travels,
            max_load_increment_mm=float(indentation_settings.travel_mm)
            / float(indentation_settings.load_steps),
        )
    return _solve_newton_vbd_indentation_path_with_schedule(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        schedule,
        checkpoint_travels=checkpoint_travels,
        checkpoint_fractions=checkpoint_fractions,
        viewer=viewer,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
        first_contact=first_contact,
    )


def _solve_newton_vbd_indentation_path_with_schedule(
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings,
    indentation_settings: IndentationSettings,
    schedule: tuple[tuple[float, int, int], ...],
    *,
    checkpoint_travels: tuple[float, ...],
    checkpoint_fractions: tuple[float, ...],
    normalized_indentation_ratios: tuple[float, ...] | None = None,
    viewer: object | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidObjectMesh | None = None,
    first_contact: FirstContactResult | None = None,
) -> IndentationTrajectoryResult:
    """Execute the one shared incremental VBD loop for all path APIs."""

    if first_contact is not None:
        from contact.first_contact import FirstContactResult

        if not isinstance(first_contact, FirstContactResult):
            raise TypeError("first_contact must be FirstContactResult or None")
        if not np.allclose(
            np.asarray(first_contact.approach_direction, dtype=float),
            np.asarray(indenter.approach_direction, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "first_contact.approach_direction must match the indenter "
                "approach_direction"
            )
    if len(checkpoint_travels) != len(checkpoint_fractions):
        raise ValueError("checkpoint travel/fraction lengths must match")
    ratios = (
        checkpoint_fractions
        if normalized_indentation_ratios is None
        else normalized_indentation_ratios
    )
    if len(ratios) != len(checkpoint_travels):
        raise ValueError("checkpoint travel/ratio lengths must match")
    # The schedule may contain intermediate steps.  Build the exact mapping
    # from each requested cumulative endpoint instead of relying on a global
    # load-step fraction.
    checkpoint_indices = {}
    for target in checkpoint_travels:
        matching = [cumulative for travel, _, cumulative in schedule if np.isclose(travel, target, rtol=0.0, atol=1.0e-12)]
        if not matching:
            raise ValueError(f"checkpoint travel {target:g} is absent from the schedule")
        checkpoint_indices[matching[-1]] = checkpoint_travels.index(target)
    checkpoint_by_index = {index: cumulative for cumulative, index in checkpoint_indices.items()}
    checkpoint_by_step = {cumulative: index for index, cumulative in checkpoint_by_index.items()}

    initial_pose = (
        first_contact.spawn_pose
        if first_contact is not None
        else indenter.initial_pose
    )
    context = _build_indentation_context(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        initial_pose=initial_pose,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
    )
    if viewer is not None:
        viewer.set_model(context.model)
    previous_pose_for_velocity = (
        first_contact.contact_pose
        if first_contact is not None
        else initial_pose
    )
    max_soft_contact_count = 0
    max_sphere_soft_contact_count = 0
    max_carrier_soft_contact_count = 0
    max_void_bottom_carrier_contact_count = 0
    first_carrier_contact_step: int | None = None
    max_sphere_carrier_rigid_contact_count = 0
    carrier_contact_vertex_indices: set[int] = set()
    max_rigid_contact_count = 0
    max_soft_contact_overflow = 0
    max_rigid_contact_overflow = 0
    min_carrier_clearance_mm = float("nan")
    initial_carrier_clearance_mm = float("nan")
    final_carrier_clearance_mm = float("nan")
    carrier_surface = (
        prepared_fingertip.surface_triangles.get("void_bottom")
        if rigid_carrier_mesh is not None
        else None
    )
    if rigid_carrier_mesh is not None and carrier_surface is not None:
        initial_carrier_clearance_mm = _signed_carrier_clearance_mm(
            context.rest_vertices_m * 1.0e3,
            carrier_surface,
            rigid_carrier_mesh,
        )
    timestep_s = float(mechanics_settings.dt)
    checkpoints: list[IndentationCheckpoint] = []
    for target_travel, interval_step, cumulative_step in schedule:
        if first_contact is None:
            target_pose = indenter.pose_at_travel(target_travel)
        else:
            # Free-space motion to contact has already been normalized by the
            # geometry-only search. The VBD schedule is post-contact only.
            target_pose = first_contact.pose_at_post_contact_travel(
                target_travel
            )
        delta_mm = np.asarray(target_pose.translation_mm) - np.asarray(
            previous_pose_for_velocity.translation_mm
        )
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
        (
            sphere_contact_count,
            carrier_contact_count,
            void_bottom_carrier_contact_count,
            sphere_carrier_rigid_count,
            contact_vertices,
        ) = _contact_shape_details(context)
        carrier_contact_vertex_indices.update(contact_vertices)
        max_sphere_soft_contact_count = max(max_sphere_soft_contact_count, sphere_contact_count)
        max_carrier_soft_contact_count = max(max_carrier_soft_contact_count, carrier_contact_count)
        max_void_bottom_carrier_contact_count = max(
            max_void_bottom_carrier_contact_count,
            void_bottom_carrier_contact_count,
        )
        if void_bottom_carrier_contact_count and first_carrier_contact_step is None:
            first_carrier_contact_step = cumulative_step
        max_sphere_carrier_rigid_contact_count = max(
            max_sphere_carrier_rigid_contact_count,
            sphere_carrier_rigid_count,
        )
        if sphere_carrier_rigid_count:
            raise RuntimeError(
                "sphere-carrier rigid collision is active; carrier collision must "
                "be particle-only for this indentation experiment"
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
        if rigid_carrier_mesh is not None and carrier_surface is not None:
            current_clearance_mm = _signed_carrier_clearance_mm(
                np.asarray(accepted_state.particle_q.numpy(), dtype=np.float32) * 1.0e3,
                carrier_surface,
                rigid_carrier_mesh,
            )
            if np.isfinite(current_clearance_mm):
                min_carrier_clearance_mm = (
                    current_clearance_mm
                    if not np.isfinite(min_carrier_clearance_mm)
                    else min(min_carrier_clearance_mm, current_clearance_mm)
                )
                final_carrier_clearance_mm = current_clearance_mm
        if viewer is not None:
            viewer.begin_frame(cumulative_step * timestep_s)
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
        previous_pose_for_velocity = target_pose

        checkpoint_index = checkpoint_by_step.get(cumulative_step)
        if checkpoint_index is not None:
            wp.synchronize_device(context.device)
            snapshot_vertices = (
                np.asarray(context.state_in.particle_q.numpy(), dtype=np.float32).copy()
                * 1.0e3
            )
            rest_vertices = context.rest_vertices_m * 1.0e3
            snapshot_result = NewtonResult(
                rest_vertices=rest_vertices,
                deformed_vertices=snapshot_vertices,
                tetrahedra=prepared_fingertip.tet_mesh.tetrahedra,
                steps=cumulative_step,
            )
            support_indices = tuple(prepared_fingertip.support_vertex_indices)
            support_displacement = snapshot_result.displacement[list(support_indices)] if support_indices else np.empty((0, 3))
            points = snapshot_vertices[prepared_fingertip.tet_mesh.tetrahedra]
            six_volumes = np.einsum(
                "ij,ij->i",
                np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
                points[:, 3] - points[:, 0],
            )
            snapshot_diagnostics: dict[str, object] = {
                "device": mechanics_settings.device,
                "full_surface_contact": True,
                "contact_buffer_status": "not_applicable_for_kinematic_indenter",
                "first_contact_normalized": first_contact is not None,
                "post_contact_travel_mm": target_travel,
                "load_steps": cumulative_step,
                "max_load_increment_mm": max(
                    travel - previous
                    for previous, travel in zip(
                        (0.0,) + tuple(item[0] for item in schedule[: cumulative_step - 1]),
                        tuple(item[0] for item in schedule[:cumulative_step]),
                    )
                ),
                "rigid_sdf_target_voxel_mm": indentation_settings.rigid_sdf_target_voxel_mm,
                "max_soft_contact_count": max_soft_contact_count,
                "max_sphere_soft_contact_count": max_sphere_soft_contact_count,
                "max_carrier_soft_contact_count": max_carrier_soft_contact_count,
                "max_void_bottom_carrier_contact_count": max_void_bottom_carrier_contact_count,
                "carrier_interface_contact_count": max_void_bottom_carrier_contact_count,
                "first_carrier_contact_step": first_carrier_contact_step,
                # This checkpoint describes the contacts present at the
                # current accepted solver state. Keep trajectory-level onset
                # provenance separate from the current-state flag.
                "carrier_contact_active": bool(contact_vertices),
                "carrier_contact_occurred": first_carrier_contact_step is not None,
                "max_sphere_carrier_rigid_contact_count": max_sphere_carrier_rigid_contact_count,
                "carrier_contact_vertex_indices": tuple(sorted(carrier_contact_vertex_indices)),
                "carrier_contact_vertex_count": len(carrier_contact_vertex_indices),
                "active_carrier_contact_vertex_indices": tuple(sorted(contact_vertices)),
                "carrier_collision_enabled": rigid_carrier_mesh is not None,
                "initial_carrier_clearance_mm": initial_carrier_clearance_mm,
                "min_carrier_clearance_mm": min_carrier_clearance_mm,
                "final_carrier_clearance_mm": final_carrier_clearance_mm,
                "max_carrier_penetration_mm": (
                    max(0.0, -min_carrier_clearance_mm)
                    if np.isfinite(min_carrier_clearance_mm)
                    else float("nan")
                ),
                "max_rigid_contact_count": max_rigid_contact_count,
                "max_soft_contact_overflow": max_soft_contact_overflow,
                "max_rigid_contact_overflow": max_rigid_contact_overflow,
                "rigid_body_particle_contact_buffer_size": _RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "rigid_contact_buffer_size": _RIGID_CONTACT_BUFFER_SIZE,
                "max_displacement_mm": float(np.max(np.linalg.norm(snapshot_result.displacement, axis=1))),
                "rms_displacement_mm": float(np.sqrt(np.mean(np.square(snapshot_result.displacement)))),
                "max_support_displacement_mm": float(np.max(np.linalg.norm(support_displacement, axis=1))) if len(support_displacement) else 0.0,
                "inverted_tetrahedra": int(np.count_nonzero(six_volumes <= 0.0)),
                "min_six_volume": float(np.min(six_volumes)),
                "final_body_x_mm": float(np.asarray(context.state_in.body_q.numpy()[context.indenter_body][:3], dtype=np.float32)[0] * 1.0e3),
                "final_body_y_mm": float(np.asarray(context.state_in.body_q.numpy()[context.indenter_body][:3], dtype=np.float32)[1] * 1.0e3),
                "final_body_z_mm": float(np.asarray(context.state_in.body_q.numpy()[context.indenter_body][:3], dtype=np.float32)[2] * 1.0e3),
            }
            if first_contact is not None:
                snapshot_diagnostics.update(
                    {
                        "first_contact_travel_mm": first_contact.travel_to_contact_mm,
                        "first_contact_bracket_width_mm": first_contact.bracket_width_mm,
                        "spawn_clearance_mm": first_contact.spawn_clearance_mm,
                        "reference_pose_collision_free": True,
                        "spawn_pose_collision_free": True,
                    }
                )
            checkpoints.append(
                IndentationCheckpoint(
                    checkpoint_index=checkpoint_index,
                    checkpoint_fraction=checkpoint_fractions[checkpoint_index],
                    normalized_indentation_ratio=ratios[checkpoint_index],
                    post_contact_travel_mm=target_travel,
                    cumulative_step_index=cumulative_step,
                    indenter_pose=target_pose,
                    mechanics_result=snapshot_result,
                    diagnostics=snapshot_diagnostics,
                )
            )

    wp.synchronize_device(context.device)
    if not checkpoints:
        raise RuntimeError("indentation path produced no requested checkpoints")
    return IndentationTrajectoryResult(
        checkpoints=tuple(checkpoints),
        total_steps=len(schedule),
        diagnostics=checkpoints[-1].diagnostics,
    )


def solve_newton_vbd_indentation(
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings,
    indentation_settings: IndentationSettings,
    *,
    viewer: object | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidObjectMesh | None = None,
    first_contact: FirstContactResult | None = None,
) -> IndentationResult:
    """Run one indentation path and return its final checkpoint."""

    trajectory = _solve_newton_vbd_indentation_path(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        viewer=viewer,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
        first_contact=first_contact,
    )
    final = trajectory.final
    return IndentationResult(
        mechanics_result=final.mechanics_result,
        final_indenter_pose=final.indenter_pose,
        diagnostics=final.diagnostics,
    )


def solve_newton_vbd_indentation_trajectory(
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings,
    indentation_settings: IndentationSettings,
    checkpoint_travels_mm: Sequence[float],
    *,
    checkpoint_fractions: Sequence[float],
    normalized_indentation_ratios: Sequence[float] | None = None,
    max_load_increment_mm: float = 0.05,
    viewer: object | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidObjectMesh | None = None,
    first_contact: FirstContactResult | None = None,
) -> IndentationTrajectoryResult:
    """Run one continuous VBD path and capture exact checkpoint states."""

    travels = tuple(float(value) for value in checkpoint_travels_mm)
    fractions = tuple(float(value) for value in checkpoint_fractions)
    ratios = None if normalized_indentation_ratios is None else tuple(
        float(value) for value in normalized_indentation_ratios
    )
    schedule = checkpoint_step_schedule(
        travels,
        max_load_increment_mm=max_load_increment_mm,
    )
    return _solve_newton_vbd_indentation_path_with_schedule(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        schedule,
        checkpoint_travels=travels,
        checkpoint_fractions=fractions,
        normalized_indentation_ratios=ratios,
        viewer=viewer,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
        first_contact=first_contact,
    )


def solve_newton_vbd(
    mesh: TetMeshData,
    settings: NewtonSettings,
) -> NewtonResult:
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
    return NewtonResult(
        rest_vertices=rest_vertices,
        deformed_vertices=deformed_vertices,
        tetrahedra=mesh.tetrahedra,
        steps=settings.steps,
    )


def solve_newton_vbd_prescribed(
    mesh: TetMeshData,
    settings: NewtonSettings,
    *,
    vertex_indices: Sequence[int],
    displacement_mm: Sequence[float],
    load_steps: int,
) -> tuple[NewtonResult, dict[str, float | int | str]]:
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
    device = _require_cuda_device(settings.device)
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
    result = NewtonResult(
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
