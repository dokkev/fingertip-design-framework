"""Concrete runtime for one LUMO simulation."""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton.indenter import Indenter
from lumo.newton.model import FingertipNewtonModel, build_fingertip_newton_model
from lumo.util.scalar_validation import require_nonnegative, require_positive


_DEFAULT_SOFT_CONTACT_MARGIN_M = 1.0e-4
_DEFAULT_ELEMENT_SIZE_MM = 1.0
_VBD_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 2048
_FORCE_CHECKPOINT_CAPACITY = 4

_MOTION_REACTION_FORCE = 0
_MOTION_PREVIOUS_FORCE = 1
_MOTION_TRAVEL = 2
_MOTION_FLOAT_COUNT = 3

_MOTION_TARGET_INDEX = 0
_MOTION_CHECKPOINT_COUNT = 1
_MOTION_FINISHED = 2
_MOTION_ERROR = 3
_MOTION_STEP_COUNT = 4
_MOTION_EVENT_SLOT = 5
_MOTION_INT_COUNT = 6

_CHECKPOINT_FORCE = 0
_CHECKPOINT_TRAVEL = 1
_CHECKPOINT_FORCE_CHANGE = 2
_CHECKPOINT_THRESHOLD = 3
_CHECKPOINT_FORCE_RATE = 4
_CHECKPOINT_INDENTATION_RATE = 5
_CHECKPOINT_FORCE_OVERSHOOT = 6
_CHECKPOINT_FLOAT_COUNT = 7


@wp.kernel
def _set_body_pose(
    pose: wp.transform,
    body_index: int,
    body_q: wp.array(dtype=wp.transform),
):
    body_q[body_index] = pose


@wp.kernel
def _project_body_reaction_force(
    body_wrenches: wp.array(dtype=wp.spatial_vector),
    body_index: int,
    motion_direction_W: wp.vec3,
    reaction_force_n: wp.array(dtype=float),
):
    force_on_body_W = wp.spatial_top(body_wrenches[body_index])
    reaction_force_n[0] = wp.max(
        0.0,
        -wp.dot(force_on_body_W, motion_direction_W),
    )


@wp.kernel
def _force_checkpoint_before_step(
    initial_translation_W_m: wp.array(dtype=wp.vec3),
    initial_rotation: wp.array(dtype=wp.quat),
    motion_direction_W: wp.vec3,
    indenter_body_index: int,
    approach_speed_m_s: float,
    time_step_s: float,
    motion_float: wp.array(dtype=float),
    motion_int: wp.array(dtype=wp.int32),
    body_q_a: wp.array(dtype=wp.transform),
    body_q_b: wp.array(dtype=wp.transform),
):
    """Advance the kinematic indenter at the prescribed physical speed."""
    if motion_int[_MOTION_FINISHED] == 0 and motion_int[_MOTION_ERROR] == 0:
        motion_float[_MOTION_TRAVEL] = (
            motion_float[_MOTION_TRAVEL] + approach_speed_m_s * time_step_s
        )

    translation_W_m = (
        initial_translation_W_m[0]
        + motion_float[_MOTION_TRAVEL] * motion_direction_W
    )
    pose = wp.transform(translation_W_m, initial_rotation[0])
    body_q_a[indenter_body_index] = pose
    body_q_b[indenter_body_index] = pose


@wp.kernel
def _force_checkpoint_after_step(
    time_step_s: float,
    target_count: int,
    target_forces_n: wp.array(dtype=float),
    approach_speed_m_s: float,
    motion_float: wp.array(dtype=float),
    motion_int: wp.array(dtype=wp.int32),
    checkpoint_float: wp.array2d(dtype=float),
    checkpoint_step: wp.array(dtype=wp.int32),
):
    """Save the first tick at or above the current ordered force threshold."""
    motion_int[_MOTION_EVENT_SLOT] = -1
    motion_int[_MOTION_STEP_COUNT] = motion_int[_MOTION_STEP_COUNT] + 1

    reaction_force_n = motion_float[_MOTION_REACTION_FORCE]
    previous_force_n = motion_float[_MOTION_PREVIOUS_FORCE]
    if not wp.isfinite(reaction_force_n):
        motion_int[_MOTION_ERROR] = 1
        return
    motion_float[_MOTION_PREVIOUS_FORCE] = reaction_force_n

    if motion_int[_MOTION_FINISHED] != 0 or motion_int[_MOTION_ERROR] != 0:
        return

    target_index = motion_int[_MOTION_TARGET_INDEX]
    target_force_n = target_forces_n[target_index]
    checkpoint_slot = motion_int[_MOTION_CHECKPOINT_COUNT]
    if checkpoint_slot >= target_count:
        motion_int[_MOTION_ERROR] = 2
        return
    if reaction_force_n < target_force_n:
        return

    signed_force_change_n = reaction_force_n - previous_force_n
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE] = reaction_force_n
    checkpoint_float[checkpoint_slot, _CHECKPOINT_TRAVEL] = motion_float[
        _MOTION_TRAVEL
    ]
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE_CHANGE] = wp.abs(
        signed_force_change_n
    )
    checkpoint_float[checkpoint_slot, _CHECKPOINT_THRESHOLD] = target_force_n
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE_RATE] = (
        signed_force_change_n / time_step_s
    )
    checkpoint_float[checkpoint_slot, _CHECKPOINT_INDENTATION_RATE] = (
        approach_speed_m_s
    )
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE_OVERSHOOT] = (
        reaction_force_n - target_force_n
    )
    checkpoint_step[checkpoint_slot] = motion_int[_MOTION_STEP_COUNT]
    motion_int[_MOTION_EVENT_SLOT] = checkpoint_slot
    motion_int[_MOTION_CHECKPOINT_COUNT] = checkpoint_slot + 1
    if checkpoint_slot + 1 == target_count:
        motion_int[_MOTION_FINISHED] = 1
    else:
        motion_int[_MOTION_TARGET_INDEX] = target_index + 1


@wp.kernel
def _snapshot_force_checkpoint_particles(
    event_slot: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    checkpoint_q_0: wp.array(dtype=wp.vec3),
    checkpoint_qd_0: wp.array(dtype=wp.vec3),
    checkpoint_q_1: wp.array(dtype=wp.vec3),
    checkpoint_qd_1: wp.array(dtype=wp.vec3),
    checkpoint_q_2: wp.array(dtype=wp.vec3),
    checkpoint_qd_2: wp.array(dtype=wp.vec3),
    checkpoint_q_3: wp.array(dtype=wp.vec3),
    checkpoint_qd_3: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()
    slot = event_slot[_MOTION_EVENT_SLOT]
    if slot == 0:
        checkpoint_q_0[particle_index] = particle_q[particle_index]
        checkpoint_qd_0[particle_index] = particle_qd[particle_index]
    elif slot == 1:
        checkpoint_q_1[particle_index] = particle_q[particle_index]
        checkpoint_qd_1[particle_index] = particle_qd[particle_index]
    elif slot == 2:
        checkpoint_q_2[particle_index] = particle_q[particle_index]
        checkpoint_qd_2[particle_index] = particle_qd[particle_index]
    elif slot == 3:
        checkpoint_q_3[particle_index] = particle_q[particle_index]
        checkpoint_qd_3[particle_index] = particle_qd[particle_index]


@wp.kernel
def _snapshot_force_checkpoint_contacts(
    event_slot: wp.array(dtype=wp.int32),
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_shape: wp.array(dtype=wp.int32),
    soft_contact_indices: wp.array(dtype=wp.vec3i),
    soft_contact_barycentric: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    count_0: wp.array(dtype=wp.int32),
    shape_0: wp.array(dtype=wp.int32),
    indices_0: wp.array(dtype=wp.vec3i),
    barycentric_0: wp.array(dtype=wp.vec3),
    normal_0: wp.array(dtype=wp.vec3),
    body_pos_0: wp.array(dtype=wp.vec3),
    count_1: wp.array(dtype=wp.int32),
    shape_1: wp.array(dtype=wp.int32),
    indices_1: wp.array(dtype=wp.vec3i),
    barycentric_1: wp.array(dtype=wp.vec3),
    normal_1: wp.array(dtype=wp.vec3),
    body_pos_1: wp.array(dtype=wp.vec3),
    count_2: wp.array(dtype=wp.int32),
    shape_2: wp.array(dtype=wp.int32),
    indices_2: wp.array(dtype=wp.vec3i),
    barycentric_2: wp.array(dtype=wp.vec3),
    normal_2: wp.array(dtype=wp.vec3),
    body_pos_2: wp.array(dtype=wp.vec3),
    count_3: wp.array(dtype=wp.int32),
    shape_3: wp.array(dtype=wp.int32),
    indices_3: wp.array(dtype=wp.vec3i),
    barycentric_3: wp.array(dtype=wp.vec3),
    normal_3: wp.array(dtype=wp.vec3),
    body_pos_3: wp.array(dtype=wp.vec3),
):
    contact_index = wp.tid()
    slot = event_slot[_MOTION_EVENT_SLOT]
    emitted_count = soft_contact_count[0]
    if slot == 0:
        if contact_index == 0:
            count_0[0] = emitted_count
        shape_0[contact_index] = soft_contact_shape[contact_index]
        indices_0[contact_index] = soft_contact_indices[contact_index]
        barycentric_0[contact_index] = soft_contact_barycentric[contact_index]
        normal_0[contact_index] = soft_contact_normal[contact_index]
        body_pos_0[contact_index] = soft_contact_body_pos[contact_index]
    elif slot == 1:
        if contact_index == 0:
            count_1[0] = emitted_count
        shape_1[contact_index] = soft_contact_shape[contact_index]
        indices_1[contact_index] = soft_contact_indices[contact_index]
        barycentric_1[contact_index] = soft_contact_barycentric[contact_index]
        normal_1[contact_index] = soft_contact_normal[contact_index]
        body_pos_1[contact_index] = soft_contact_body_pos[contact_index]
    elif slot == 2:
        if contact_index == 0:
            count_2[0] = emitted_count
        shape_2[contact_index] = soft_contact_shape[contact_index]
        indices_2[contact_index] = soft_contact_indices[contact_index]
        barycentric_2[contact_index] = soft_contact_barycentric[contact_index]
        normal_2[contact_index] = soft_contact_normal[contact_index]
        body_pos_2[contact_index] = soft_contact_body_pos[contact_index]
    elif slot == 3:
        if contact_index == 0:
            count_3[0] = emitted_count
        shape_3[contact_index] = soft_contact_shape[contact_index]
        indices_3[contact_index] = soft_contact_indices[contact_index]
        barycentric_3[contact_index] = soft_contact_barycentric[contact_index]
        normal_3[contact_index] = soft_contact_normal[contact_index]
        body_pos_3[contact_index] = soft_contact_body_pos[contact_index]


@wp.kernel
def _measure_active_particle_speed(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    active_flag: int,
    maximum_speed_m_s: wp.array(dtype=float),
    nonfinite_velocity_count: wp.array(dtype=wp.int32),
):
    particle_index = wp.tid()
    if (particle_flags[particle_index] & active_flag) == 0:
        return

    velocity_W_m_s = particle_qd[particle_index]
    if not wp.isfinite(velocity_W_m_s):
        wp.atomic_add(nonfinite_velocity_count, 0, 1)
        return

    wp.atomic_max(
        maximum_speed_m_s,
        0,
        wp.length(velocity_W_m_s),
    )


class LumoSimulation:
    """Construct and own the runtime for one analytic LUMO fingertip."""

    def __init__(
        self,
        fingertip: Fingertip,
        *,
        builder: newton.ModelBuilder | None = None,
        fingertip_mesh: FingertipMesh | None = None,
        fingertip_model: FingertipNewtonModel | None = None,
        sim_frequency: float,
        iterations: int = 10,
        soft_contact_margin_m: float = _DEFAULT_SOFT_CONTACT_MARGIN_M,
        soft_contact_stiffness_n_m: float | None = None,
        soft_contact_damping_n_s_m: float | None = None,
        element_size_mm: float = _DEFAULT_ELEMENT_SIZE_MM,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        if builder is not None and not isinstance(
            builder,
            newton.ModelBuilder,
        ):
            raise TypeError("builder must be a newton.ModelBuilder")
        if fingertip_model is not None:
            if not isinstance(fingertip_model, FingertipNewtonModel):
                raise TypeError("fingertip_model must be a FingertipNewtonModel")
            if builder is not None:
                raise ValueError("builder and fingertip_model are mutually exclusive")
            if fingertip_model.fingertip_mesh.fingertip is not fingertip:
                raise ValueError("fingertip_model must belong to the supplied fingertip")
            if (
                fingertip_mesh is not None
                and fingertip_mesh is not fingertip_model.fingertip_mesh
            ):
                raise ValueError("fingertip_mesh must match fingertip_model")
        if fingertip_mesh is not None:
            if not isinstance(fingertip_mesh, FingertipMesh):
                raise TypeError("fingertip_mesh must be a FingertipMesh")
            if fingertip_mesh.fingertip is not fingertip:
                raise ValueError(
                    "fingertip_mesh must belong to the supplied fingertip"
                )
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")
        require_positive("sim_frequency", sim_frequency)
        require_nonnegative(
            "soft_contact_margin_m",
            soft_contact_margin_m,
        )
        if soft_contact_stiffness_n_m is not None:
            require_positive(
                "soft_contact_stiffness_n_m",
                soft_contact_stiffness_n_m,
            )
        if soft_contact_damping_n_s_m is not None:
            require_nonnegative(
                "soft_contact_damping_n_s_m",
                soft_contact_damping_n_s_m,
            )
        require_positive("element_size_mm", element_size_mm)
        self.fingertip = fingertip
        if fingertip_model is None:
            self.fingertip_mesh = (
                make_fingertip_mesh(
                    fingertip,
                    element_size_mm=element_size_mm,
                )
                if fingertip_mesh is None
                else fingertip_mesh
            )
            self.fingertip_model = build_fingertip_newton_model(
                self.fingertip_mesh,
                builder=builder,
            )
        else:
            self.fingertip_mesh = fingertip_model.fingertip_mesh
            self.fingertip_model = fingertip_model
        model = self.fingertip_model.model
        if soft_contact_stiffness_n_m is not None:
            model.soft_contact_ke = float(soft_contact_stiffness_n_m)
        if soft_contact_damping_n_s_m is not None:
            model.soft_contact_kd = float(soft_contact_damping_n_s_m)

        self.sim_frequency = float(sim_frequency)
        self.time_step_s = 1.0 / self.sim_frequency
        self.step_count = 0
        self.time_s = 0.0
        self.solver = newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            particle_enable_self_contact=False,
            rigid_body_particle_contact_buffer_size=(
                _VBD_BODY_PARTICLE_CONTACT_BUFFER_SIZE
            ),
        )
        self.collision_pipeline = newton.CollisionPipeline(
            model,
            soft_contact_margin=soft_contact_margin_m,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self._state_a = model.state()
        self._state_b = model.state()
        self.state = self._state_a
        self._next_state = self._state_b
        self.control = model.control()
        if self.state.body_qd is None:
            raise RuntimeError("simulation state has no rigid-body velocities")
        self._body_qd_before = wp.empty_like(self.state.body_qd)
        self._body_to_wrench = wp.array(
            np.arange(model.body_count, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self._body_wrenches = wp.zeros(
            model.body_count,
            dtype=wp.spatial_vector,
            device=model.device,
        )
        self._reaction_force_n = wp.zeros(
            1,
            dtype=float,
            device=model.device,
        )
        self._maximum_particle_speed_m_s = wp.zeros(
            1,
            dtype=float,
            device=model.device,
        )
        self._nonfinite_particle_velocity_count = wp.zeros(
            1,
            dtype=wp.int32,
            device=model.device,
        )
        self._cuda_graph_replay_count = 0
        self._checkpoint_graph_a: wp.Graph | None = None
        self._checkpoint_graph_b: wp.Graph | None = None
        self._motion_float: wp.array | None = None
        self._motion_int: wp.array | None = None
        self._target_forces_n: wp.array | None = None
        self._checkpoint_float: wp.array | None = None
        self._checkpoint_step: wp.array | None = None
        self._checkpoint_states: tuple[newton.State, ...] = ()
        self._checkpoint_contacts: tuple[newton.Contacts, ...] = ()
        self._initial_translation_W_m: wp.array | None = None
        self._initial_rotation: wp.array | None = None
        self._motion_direction_W: wp.vec3 | None = None
        self._indenter_body_index: int | None = None
        self._approach_speed_m_s: float | None = None
        self._target_count = 0
        self._checkpoint_host_intervention_count = 0
        self._checkpoint_host_sync_count = 0
        self._live_checkpoint_selection: tuple[
            newton.State,
            newton.Contacts,
            int,
            float,
        ] | None = None
        self._has_step_result = False
        self.collision_pipeline.collide(self.state, self.contacts)

    def silicone_vertices(self) -> np.ndarray:
        """Return current silicone positions in fingertip-mesh vertex order."""
        return self.fingertip_model.silicone_vertices(self.state)

    def apply_indenter_pose(
        self,
        indenter: Indenter,
        pose: wp.transform,
    ) -> None:
        """Apply one prescribed indenter pose to both state buffers."""
        if not isinstance(indenter, Indenter):
            raise TypeError("indenter must be an Indenter")
        body_count = self.fingertip_model.model.body_count
        if indenter.body_index < 0 or indenter.body_index >= body_count:
            raise ValueError("indenter body index is outside this simulation")

        for state in (self.state, self._next_state):
            if state.body_q is None:
                raise ValueError("simulation state has no rigid-body poses")
            wp.launch(
                _set_body_pose,
                dim=1,
                inputs=[pose, indenter.body_index],
                outputs=[state.body_q],
                device=state.body_q.device,
            )

    def soft_contact_count(self, body_index: int | None = None) -> int:
        """Return the current total or body-specific soft-contact count."""
        contact_count = int(self.contacts.soft_contact_count.numpy()[0])
        if body_index is None:
            return contact_count
        if (
            isinstance(body_index, bool)
            or not isinstance(body_index, (int, np.integer))
            or body_index < 0
            or body_index >= self.fingertip_model.model.body_count
        ):
            raise ValueError("body_index is outside this simulation")
        if contact_count == 0:
            return 0

        shape_indices = self.contacts.soft_contact_shape.numpy()[
            :contact_count
        ]
        valid = shape_indices >= 0
        shape_bodies = self.fingertip_model.model.shape_body.numpy()
        return int(
            np.count_nonzero(
                shape_bodies[shape_indices[valid]] == body_index
            )
        )

    def _launch_step_device_operations(
        self,
        state_in: newton.State,
        state_out: newton.State,
    ) -> None:
        """Launch the fixed device work for one Newton tick."""
        self.fingertip_model.prepare_step(state_in, state_out)
        if state_in.body_qd is None:
            raise RuntimeError("simulation state has no rigid-body velocities")
        wp.copy(self._body_qd_before, state_in.body_qd)
        self.solver.coupling_notify_input_state_update(
            state_in,
            newton.StateFlags.BODY_Q,
            dt=self.time_step_s,
        )

        state_in.clear_forces()
        self.collision_pipeline.collide(state_in, self.contacts)
        self.solver.step(
            state_in,
            state_out,
            self.control,
            self.contacts,
            self.time_step_s,
        )
        self._body_wrenches.zero_()
        self.solver.coupling_harvest_proxy_wrenches(
            self._body_to_wrench,
            self._body_wrenches,
            body_qd_before=self._body_qd_before,
            state=state_in,
            state_out=state_out,
            contacts=self.contacts,
            dt=self.time_step_s,
        )

    def _configure_force_checkpoints(
        self,
        indenter: Indenter,
        *,
        initial_tf: wp.transform,
        motion_direction_W: wp.vec3,
        approach_speed_m_s: float,
        target_forces_n: tuple[float, ...],
    ) -> None:
        """Prepare GPU-resident motion and ordered force checkpoints."""
        if not self.fingertip_model.model.device.is_cuda:
            raise RuntimeError("force checkpoints require a CUDA Newton model")
        if self._motion_float is not None:
            raise RuntimeError("force checkpoints are already configured")
        if not isinstance(indenter, Indenter):
            raise TypeError("indenter must be an Indenter")
        if not 0 <= indenter.body_index < self.fingertip_model.model.body_count:
            raise ValueError("indenter body index is outside this simulation")

        initial_tf_values = np.asarray(initial_tf, dtype=np.float64)
        if initial_tf_values.shape != (7,) or not np.all(
            np.isfinite(initial_tf_values)
        ):
            raise ValueError("initial_tf must be a finite Warp transform")
        direction_W = np.asarray(motion_direction_W, dtype=np.float64)
        if direction_W.shape != (3,) or not np.all(np.isfinite(direction_W)):
            raise ValueError("motion_direction_W must be a finite 3-vector")
        direction_norm = float(np.linalg.norm(direction_W))
        require_positive("motion_direction_W norm", direction_norm)
        direction_W /= direction_norm
        require_positive("approach_speed_m_s", approach_speed_m_s)

        targets_n = np.asarray(target_forces_n, dtype=np.float64)
        if (
            targets_n.ndim != 1
            or targets_n.size < 1
            or targets_n.size > _FORCE_CHECKPOINT_CAPACITY
        ):
            raise ValueError("one to four force targets are supported")
        if not np.all(np.isfinite(targets_n)) or np.any(targets_n <= 0.0):
            raise ValueError("force targets must be finite and positive")
        if np.any(np.diff(targets_n) <= 0.0):
            raise ValueError("force targets must be strictly increasing")

        device = self.fingertip_model.model.device
        self._initial_translation_W_m = wp.array(
            [wp.vec3(*initial_tf_values[:3])], dtype=wp.vec3, device=device
        )
        self._initial_rotation = wp.array(
            [wp.quat(*initial_tf_values[3:])], dtype=wp.quat, device=device
        )
        self._motion_direction_W = wp.vec3(*direction_W)
        self._indenter_body_index = indenter.body_index
        self._approach_speed_m_s = float(approach_speed_m_s)
        self._target_count = int(targets_n.size)
        self._motion_float = wp.zeros(_MOTION_FLOAT_COUNT, dtype=float, device=device)
        initial_int = np.zeros(_MOTION_INT_COUNT, dtype=np.int32)
        initial_int[_MOTION_EVENT_SLOT] = -1
        self._motion_int = wp.array(initial_int, dtype=wp.int32, device=device)
        self._target_forces_n = wp.array(
            targets_n.astype(np.float32),
            dtype=float,
            device=device,
        )
        self._checkpoint_float = wp.zeros(
            (self._target_count, _CHECKPOINT_FLOAT_COUNT),
            dtype=float,
            device=device,
        )
        self._checkpoint_step = wp.zeros(
            self._target_count, dtype=wp.int32, device=device
        )
        self._checkpoint_states = tuple(
            self.fingertip_model.model.state()
            for _ in range(_FORCE_CHECKPOINT_CAPACITY)
        )
        self._checkpoint_contacts = tuple(
            self.collision_pipeline.contacts()
            for _ in range(_FORCE_CHECKPOINT_CAPACITY)
        )

        # Let Newton finish lazy contact-buffer sizing before graph capture.
        self._launch_force_checkpoint_tick(self.state, self._next_state)
        self.state, self._next_state = self._next_state, self.state
        self.step_count = 1
        self.time_s = self.time_step_s
        self._has_step_result = True
        self._capture_force_checkpoint_graphs()

    def _require_force_checkpoints(self) -> None:
        if (
            self._motion_float is None
            or self._motion_int is None
            or self._target_forces_n is None
            or self._checkpoint_float is None
            or self._checkpoint_step is None
            or self._initial_translation_W_m is None
            or self._initial_rotation is None
            or self._motion_direction_W is None
            or self._indenter_body_index is None
            or self._approach_speed_m_s is None
        ):
            raise RuntimeError("force checkpoints are not configured")

    def _launch_force_checkpoint_snapshots(self, state_out: newton.State) -> None:
        """Copy a threshold-crossing tick into its device checkpoint slot."""
        self._require_force_checkpoints()
        if state_out.particle_q is None or state_out.particle_qd is None:
            raise RuntimeError("simulation state has no particle state")
        particle_outputs: list[wp.array] = []
        contact_outputs: list[wp.array] = []
        for checkpoint_state, checkpoint_contact in zip(
            self._checkpoint_states,
            self._checkpoint_contacts,
            strict=True,
        ):
            if checkpoint_state.particle_q is None or checkpoint_state.particle_qd is None:
                raise RuntimeError("checkpoint state has no particle state")
            particle_outputs.extend(
                [checkpoint_state.particle_q, checkpoint_state.particle_qd]
            )
            contact_outputs.extend(
                [
                    checkpoint_contact.soft_contact_count,
                    checkpoint_contact.soft_contact_shape,
                    checkpoint_contact.soft_contact_indices,
                    checkpoint_contact.soft_contact_barycentric,
                    checkpoint_contact.soft_contact_normal,
                    checkpoint_contact.soft_contact_body_pos,
                ]
            )
        wp.launch(
            _snapshot_force_checkpoint_particles,
            dim=self.fingertip_model.model.particle_count,
            inputs=[self._motion_int, state_out.particle_q, state_out.particle_qd, *particle_outputs],
            device=self.fingertip_model.model.device,
        )
        wp.launch(
            _snapshot_force_checkpoint_contacts,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self._motion_int,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_indices,
                self.contacts.soft_contact_barycentric,
                self.contacts.soft_contact_normal,
                self.contacts.soft_contact_body_pos,
                *contact_outputs,
            ],
            device=self.fingertip_model.model.device,
        )

    def _launch_force_checkpoint_tick(
        self,
        state_in: newton.State,
        state_out: newton.State,
    ) -> None:
        """Launch prescribed motion, physics, wrench, and checkpoint work."""
        self._require_force_checkpoints()
        if state_in.body_q is None or state_out.body_q is None:
            raise RuntimeError("simulation state has no rigid-body poses")
        wp.launch(
            _force_checkpoint_before_step,
            dim=1,
            inputs=[
                self._initial_translation_W_m,
                self._initial_rotation,
                self._motion_direction_W,
                self._indenter_body_index,
                self._approach_speed_m_s,
                self.time_step_s,
                self._motion_float,
                self._motion_int,
                state_in.body_q,
                state_out.body_q,
            ],
            device=self.fingertip_model.model.device,
        )
        self._launch_step_device_operations(state_in, state_out)
        wp.launch(
            _project_body_reaction_force,
            dim=1,
            inputs=[
                self._body_wrenches,
                self._indenter_body_index,
                self._motion_direction_W,
            ],
            outputs=[self._reaction_force_n],
            device=self.fingertip_model.model.device,
        )
        wp.copy(self._motion_float, self._reaction_force_n, count=1)
        wp.launch(
            _force_checkpoint_after_step,
            dim=1,
            inputs=[
                self.time_step_s,
                self._target_count,
                self._target_forces_n,
                self._approach_speed_m_s,
                self._motion_float,
                self._motion_int,
                self._checkpoint_float,
                self._checkpoint_step,
            ],
            device=self.fingertip_model.model.device,
        )
        self._launch_force_checkpoint_snapshots(state_out)

    def _capture_force_checkpoint_graphs(self) -> None:
        """Capture two-tick graphs that return to their starting parity."""
        wp.synchronize_device(self.fingertip_model.model.device)
        self._validate_cuda_graph_capture_ready()
        allocation_signature = self._cuda_graph_contact_allocation_signature()
        with wp.ScopedCapture(
            device=self.fingertip_model.model.device,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as capture_a:
            self._launch_force_checkpoint_tick(self._state_a, self._state_b)
            self._launch_force_checkpoint_tick(self._state_b, self._state_a)
        self._checkpoint_graph_a = capture_a.graph
        if allocation_signature != self._cuda_graph_contact_allocation_signature():
            raise RuntimeError("checkpoint graph A replaced Newton contact storage")

        with wp.ScopedCapture(
            device=self.fingertip_model.model.device,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as capture_b:
            self._launch_force_checkpoint_tick(self._state_b, self._state_a)
            self._launch_force_checkpoint_tick(self._state_a, self._state_b)
        self._checkpoint_graph_b = capture_b.graph
        if allocation_signature != self._cuda_graph_contact_allocation_signature():
            raise RuntimeError("checkpoint graph B replaced Newton contact storage")

    def _launch_force_checkpoints(
        self,
        tick_count: int,
        *,
        stream: wp.Stream | None = None,
    ) -> None:
        """Enqueue GPU-resident threshold-checkpoint ticks."""
        self._require_force_checkpoints()
        if tick_count < 1:
            raise ValueError("tick_count must be positive")
        if stream is not None and stream.device != self.fingertip_model.model.device:
            raise ValueError("checkpoint stream belongs to another device")
        remaining = tick_count
        while remaining >= 2:
            if self.state is self._state_a:
                graph = self._checkpoint_graph_a
            elif self.state is self._state_b:
                graph = self._checkpoint_graph_b
            else:
                raise RuntimeError("live state is not an A/B state")
            if graph is None:
                raise RuntimeError("checkpoint graph is unavailable")
            wp.capture_launch(graph, stream=stream)
            self._cuda_graph_replay_count += 1
            self.step_count += 2
            remaining -= 2
        if remaining:
            with wp.ScopedStream(stream, sync_enter=False):
                self._launch_force_checkpoint_tick(self.state, self._next_state)
            self.state, self._next_state = self._next_state, self.state
            self.step_count += 1
        self.time_s = self.step_count * self.time_step_s
        self._has_step_result = True
        self._checkpoint_host_intervention_count += 1

    def _force_checkpoint_status(
        self,
        *,
        stream: wp.Stream | None = None,
    ) -> tuple[int, bool, int]:
        """Synchronize one stream and return compact checkpoint status."""
        self._require_force_checkpoints()
        if stream is not None:
            wp.synchronize_stream(stream)
        self._checkpoint_host_sync_count += 1
        status = self._motion_int.numpy()
        return (
            int(status[_MOTION_CHECKPOINT_COUNT]),
            bool(status[_MOTION_FINISHED]),
            int(status[_MOTION_ERROR]),
        )

    def _force_checkpoint(self, index: int) -> dict[str, float | int]:
        """Read one compact threshold-crossing record."""
        self._require_force_checkpoints()
        if index < 0 or index >= self._target_count:
            raise ValueError("checkpoint index is outside the target schedule")
        checkpoint_float = self._checkpoint_float.numpy()[index]
        checkpoint_step = int(self._checkpoint_step.numpy()[index])
        self._checkpoint_host_sync_count += 2
        return {
            "reaction_force_n": float(checkpoint_float[_CHECKPOINT_FORCE]),
            "travel_m": float(checkpoint_float[_CHECKPOINT_TRAVEL]),
            "force_change_n": float(checkpoint_float[_CHECKPOINT_FORCE_CHANGE]),
            "force_threshold_n": float(checkpoint_float[_CHECKPOINT_THRESHOLD]),
            "reaction_force_rate_n_s": float(checkpoint_float[_CHECKPOINT_FORCE_RATE]),
            "indentation_rate_m_s": float(checkpoint_float[_CHECKPOINT_INDENTATION_RATE]),
            "force_overshoot_n": float(checkpoint_float[_CHECKPOINT_FORCE_OVERSHOOT]),
            "step_count": checkpoint_step,
            "simulation_time_s": checkpoint_step * self.time_step_s,
        }

    def _current_reaction_force_n(self) -> float:
        """Read the current projected force for a terminal error report."""
        self._require_force_checkpoints()
        value = float(self._motion_float.numpy()[_MOTION_REACTION_FORCE])
        self._checkpoint_host_sync_count += 1
        return value

    def _select_force_checkpoint(self, index: int) -> None:
        """Expose one exact saved checkpoint through the callback API."""
        if self._live_checkpoint_selection is not None:
            raise RuntimeError("a checkpoint is already selected")
        if index < 0 or index >= self._target_count:
            raise ValueError("checkpoint index is outside the target schedule")
        checkpoint_step = int(self._checkpoint_step.numpy()[index])
        self._checkpoint_host_sync_count += 1
        self._live_checkpoint_selection = (
            self.state,
            self.contacts,
            self.step_count,
            self.time_s,
        )
        self.state = self._checkpoint_states[index]
        self.contacts = self._checkpoint_contacts[index]
        self.step_count = checkpoint_step
        self.time_s = checkpoint_step * self.time_step_s

    def _restore_live_state(self) -> None:
        """Restore the live graph state after one checkpoint callback."""
        if self._live_checkpoint_selection is None:
            raise RuntimeError("no checkpoint is selected")
        self.state, self.contacts, self.step_count, self.time_s = (
            self._live_checkpoint_selection
        )
        self._live_checkpoint_selection = None

    @property
    def _force_checkpoint_performance(self) -> dict[str, float | int]:
        """Return graph replay and synchronization diagnostics."""
        interventions = self._checkpoint_host_intervention_count
        live_step_count = (
            self.step_count
            if self._live_checkpoint_selection is None
            else self._live_checkpoint_selection[2]
        )
        return {
            "graph_replay_count": self._cuda_graph_replay_count,
            "host_intervention_count": interventions,
            "host_sync_count": self._checkpoint_host_sync_count,
            "average_ticks_per_host_intervention": (
                live_step_count / interventions if interventions else 0.0
            ),
        }

    def _validate_cuda_graph_capture_ready(self) -> None:
        """Verify that Newton's lazily sized contact storage is fixed."""
        soft_capacity = self.contacts.soft_contact_max
        if self.solver.body_particle_contact_penalty_k.shape[0] < soft_capacity:
            raise RuntimeError(
                "uncaptured warm-up did not size the body-particle contact state"
            )

        rigid_capacity = self.contacts.rigid_contact_max
        if self.solver.body_body_contact_penalty_k.shape[0] < rigid_capacity:
            raise RuntimeError(
                "uncaptured warm-up did not size the body-body contact state"
            )
        if self.solver.rigid_contact_history:
            contact_history = self.solver._prev_contact_lambda
            if (
                contact_history is None
                or contact_history.shape[0] < rigid_capacity
            ):
                raise RuntimeError(
                    "uncaptured warm-up did not size rigid contact history"
                )

    def _cuda_graph_contact_allocation_signature(self) -> tuple[int, ...]:
        """Return pointers for Newton contact storage captured by the graphs."""
        arrays = (
            self.solver.body_particle_contact_penalty_k,
            self.solver.body_particle_contact_material_ke,
            self.solver.body_particle_contact_material_kd,
            self.solver.body_particle_contact_material_mu,
            self.solver.body_body_contact_penalty_k,
            self.solver.body_body_contact_material_ke,
            self.solver.body_body_contact_material_kd,
            self.solver.body_body_contact_material_mu,
            self.solver.body_body_contact_lambda,
            self.solver.body_body_contact_C0,
            self.solver._prev_contact_lambda,
            self.solver._prev_contact_penalty_k,
            self.solver._prev_contact_normal,
        )
        pointers = []
        for array in arrays:
            if array is None or array.shape[0] == 0:
                pointers.append(0)
            elif array.ptr == 0:
                raise RuntimeError(
                    "uncaptured warm-up left contact storage unallocated"
                )
            else:
                pointers.append(int(array.ptr))
        return tuple(pointers)

    def step(self) -> None:
        """Advance one global tick and record its rigid-body wrenches."""
        self._launch_step_device_operations(self.state, self._next_state)
        self.state, self._next_state = self._next_state, self.state
        self.step_count += 1
        self.time_s = self.step_count * self.time_step_s
        self._has_step_result = True

    def indenter_reaction_force(
        self,
        indenter: Indenter,
        *,
        motion_direction_W: wp.vec3,
    ) -> float:
        """Return the last tick's force opposing world-frame motion."""
        if not isinstance(indenter, Indenter):
            raise TypeError("indenter must be an Indenter")
        body_count = self.fingertip_model.model.body_count
        if indenter.body_index < 0 or indenter.body_index >= body_count:
            raise ValueError("indenter body index is outside this simulation")
        if not self._has_step_result:
            raise RuntimeError(
                "reaction force is unavailable before the first step"
            )

        direction_W = np.asarray(motion_direction_W, dtype=np.float64)
        if direction_W.shape != (3,) or not np.all(np.isfinite(direction_W)):
            raise ValueError("motion_direction_W must be a finite 3-vector")
        direction_norm = float(np.linalg.norm(direction_W))
        require_positive("motion_direction_W norm", direction_norm)
        direction_W = direction_W / direction_norm

        wp.launch(
            _project_body_reaction_force,
            dim=1,
            inputs=[
                self._body_wrenches,
                indenter.body_index,
                wp.vec3(*direction_W),
            ],
            outputs=[self._reaction_force_n],
            device=self.fingertip_model.model.device,
        )
        reaction_force_n = float(self._reaction_force_n.numpy()[0])
        if not np.isfinite(reaction_force_n):
            raise RuntimeError(
                "simulation step produced a non-finite body wrench"
            )
        return reaction_force_n

    def maximum_active_particle_speed_m_s(self) -> float:
        """Return the maximum speed among active silicone particles."""
        particle_qd = self.state.particle_qd
        particle_flags = self.fingertip_model.model.particle_flags
        if particle_qd is None or particle_flags is None:
            raise RuntimeError("simulation has no silicone particle velocity")

        self._maximum_particle_speed_m_s.zero_()
        self._nonfinite_particle_velocity_count.zero_()
        wp.launch(
            _measure_active_particle_speed,
            dim=self.fingertip_model.model.particle_count,
            inputs=[
                particle_qd,
                particle_flags,
                int(newton.ParticleFlags.ACTIVE),
            ],
            outputs=[
                self._maximum_particle_speed_m_s,
                self._nonfinite_particle_velocity_count,
            ],
            device=self.fingertip_model.model.device,
        )
        if int(self._nonfinite_particle_velocity_count.numpy()[0]) != 0:
            raise RuntimeError(
                "simulation step produced a non-finite particle velocity"
            )
        return float(self._maximum_particle_speed_m_s.numpy()[0])


__all__ = ["LumoSimulation"]
