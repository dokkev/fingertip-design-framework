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
_DEFAULT_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6
_VBD_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 2048
_FORCE_SERVO_CHECKPOINT_CAPACITY = 4

_SERVO_REACTION_FORCE = 0
_SERVO_PREVIOUS_FORCE = 1
_SERVO_TRAVEL = 2
_SERVO_COMMANDED_DISPLACEMENT = 3
_SERVO_VELOCITY = 4
_SERVO_FLOAT_COUNT = 5

_SERVO_TARGET_INDEX = 0
_SERVO_DWELL_COUNT = 1
_SERVO_CHECKPOINT_COUNT = 2
_SERVO_FINISHED = 3
_SERVO_ERROR = 4
_SERVO_STEP_COUNT = 5
_SERVO_EVENT_SLOT = 6
_SERVO_INT_COUNT = 7

_CHECKPOINT_FORCE = 0
_CHECKPOINT_TRAVEL = 1
_CHECKPOINT_FORCE_CHANGE = 2
_CHECKPOINT_FORCE_REFERENCE = 3
_CHECKPOINT_FORCE_RATE = 4
_CHECKPOINT_INDENTATION_RATE = 5
_CHECKPOINT_SERVO_ERROR = 6
_CHECKPOINT_WINDOW_START_FORCE = 7
_CHECKPOINT_WINDOW_START_TRAVEL = 8
_CHECKPOINT_FLOAT_COUNT = 9


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
def _force_servo_before_step(
    initial_translation_W_m: wp.array(dtype=wp.vec3),
    initial_rotation: wp.array(dtype=wp.quat),
    motion_direction_W: wp.vec3,
    indenter_body_index: int,
    approach_speed_m_s: float,
    force_gain_m_s_n: float,
    time_step_s: float,
    target_forces_n: wp.array(dtype=float),
    servo_float: wp.array(dtype=float),
    servo_int: wp.array(dtype=wp.int32),
    body_q_a: wp.array(dtype=wp.transform),
    body_q_b: wp.array(dtype=wp.transform),
):
    """Apply the production proportional servo before one physics tick."""
    if servo_int[_SERVO_FINISHED] != 0 or servo_int[_SERVO_ERROR] != 0:
        servo_float[_SERVO_COMMANDED_DISPLACEMENT] = 0.0
        servo_float[_SERVO_VELOCITY] = 0.0
    else:
        target_index = servo_int[_SERVO_TARGET_INDEX]
        force_error_n = (
            target_forces_n[target_index]
            - servo_float[_SERVO_REACTION_FORCE]
        )
        velocity_m_s = wp.max(
            -approach_speed_m_s,
            wp.min(
                approach_speed_m_s,
                force_gain_m_s_n * force_error_n,
            ),
        )
        commanded_displacement_m = velocity_m_s * time_step_s
        servo_float[_SERVO_VELOCITY] = velocity_m_s
        servo_float[_SERVO_COMMANDED_DISPLACEMENT] = commanded_displacement_m
        servo_float[_SERVO_TRAVEL] = (
            servo_float[_SERVO_TRAVEL] + commanded_displacement_m
        )

    translation_W_m = (
        initial_translation_W_m[0]
        + servo_float[_SERVO_TRAVEL] * motion_direction_W
    )
    pose = wp.transform(translation_W_m, initial_rotation[0])
    body_q_a[indenter_body_index] = pose
    body_q_b[indenter_body_index] = pose


@wp.kernel
def _force_servo_after_step(
    time_step_s: float,
    settle_ticks: int,
    settle_window_ticks: int,
    target_count: int,
    displacement_tolerance_m: float,
    target_forces_n: wp.array(dtype=float),
    force_tolerances_n: wp.array(dtype=float),
    servo_float: wp.array(dtype=float),
    servo_int: wp.array(dtype=wp.int32),
    checkpoint_float: wp.array2d(dtype=float),
    checkpoint_step: wp.array(dtype=wp.int32),
):
    """Update dwell/checkpoint state after the current reaction is measured."""
    servo_int[_SERVO_EVENT_SLOT] = -1
    servo_int[_SERVO_STEP_COUNT] = servo_int[_SERVO_STEP_COUNT] + 1

    reaction_force_n = servo_float[_SERVO_REACTION_FORCE]
    previous_force_n = servo_float[_SERVO_PREVIOUS_FORCE]
    if not wp.isfinite(reaction_force_n):
        servo_int[_SERVO_ERROR] = 1
        return
    servo_float[_SERVO_PREVIOUS_FORCE] = reaction_force_n

    if servo_int[_SERVO_FINISHED] != 0 or servo_int[_SERVO_ERROR] != 0:
        return

    target_index = servo_int[_SERVO_TARGET_INDEX]
    target_force_n = target_forces_n[target_index]
    force_tolerance_n = force_tolerances_n[target_index]
    commanded_displacement_m = servo_float[_SERVO_COMMANDED_DISPLACEMENT]
    force_is_settled = wp.abs(reaction_force_n - target_force_n) <= force_tolerance_n
    displacement_is_settled = (
        displacement_tolerance_m < 0.0
        or wp.abs(commanded_displacement_m) <= displacement_tolerance_m
    )
    if force_is_settled and displacement_is_settled:
        servo_int[_SERVO_DWELL_COUNT] = servo_int[_SERVO_DWELL_COUNT] + 1
    else:
        servo_int[_SERVO_DWELL_COUNT] = 0

    window_start_count = wp.max(1, settle_ticks - settle_window_ticks)
    if servo_int[_SERVO_DWELL_COUNT] == window_start_count:
        checkpoint_slot = servo_int[_SERVO_CHECKPOINT_COUNT]
        checkpoint_float[checkpoint_slot, _CHECKPOINT_WINDOW_START_FORCE] = (
            reaction_force_n
        )
        checkpoint_float[checkpoint_slot, _CHECKPOINT_WINDOW_START_TRAVEL] = (
            servo_float[_SERVO_TRAVEL]
        )

    if servo_int[_SERVO_DWELL_COUNT] < settle_ticks:
        return

    checkpoint_slot = servo_int[_SERVO_CHECKPOINT_COUNT]
    if checkpoint_slot >= target_count:
        servo_int[_SERVO_ERROR] = 2
        return

    signed_force_change_n = reaction_force_n - previous_force_n
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE] = reaction_force_n
    checkpoint_float[checkpoint_slot, _CHECKPOINT_TRAVEL] = servo_float[_SERVO_TRAVEL]
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE_CHANGE] = wp.abs(
        signed_force_change_n
    )
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE_REFERENCE] = target_force_n
    checkpoint_float[checkpoint_slot, _CHECKPOINT_FORCE_RATE] = (
        signed_force_change_n / time_step_s
    )
    checkpoint_float[checkpoint_slot, _CHECKPOINT_INDENTATION_RATE] = (
        servo_float[_SERVO_VELOCITY]
    )
    checkpoint_float[checkpoint_slot, _CHECKPOINT_SERVO_ERROR] = (
        target_force_n - reaction_force_n
    )
    checkpoint_step[checkpoint_slot] = servo_int[_SERVO_STEP_COUNT]
    servo_int[_SERVO_EVENT_SLOT] = checkpoint_slot
    servo_int[_SERVO_CHECKPOINT_COUNT] = checkpoint_slot + 1
    servo_int[_SERVO_DWELL_COUNT] = 0
    if checkpoint_slot + 1 == target_count:
        servo_int[_SERVO_FINISHED] = 1
    else:
        servo_int[_SERVO_TARGET_INDEX] = target_index + 1


@wp.kernel
def _snapshot_force_servo_particles(
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
    slot = event_slot[_SERVO_EVENT_SLOT]
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
def _snapshot_force_servo_contacts(
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
    slot = event_slot[_SERVO_EVENT_SLOT]
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
        carrier_contact_stiffness_n_m: float = (
            _DEFAULT_CARRIER_CONTACT_STIFFNESS_N_M
        ),
        use_cuda_graph: bool = False,
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
        require_positive(
            "carrier_contact_stiffness_n_m",
            carrier_contact_stiffness_n_m,
        )
        if not isinstance(use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")

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
                carrier_contact_stiffness_n_m=carrier_contact_stiffness_n_m,
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
        self._fingertip_pose = wp.transform_identity()
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
        if use_cuda_graph and not model.device.is_cuda:
            raise ValueError("use_cuda_graph requires a CUDA Newton model")
        self._use_cuda_graph = use_cuda_graph
        self._step_graph_ab: wp.Graph | None = None
        self._step_graph_ba: wp.Graph | None = None
        self._cuda_graph_replay_count = 0
        self._servo_graph_a: wp.Graph | None = None
        self._servo_graph_b: wp.Graph | None = None
        self._servo_float: wp.array | None = None
        self._servo_int: wp.array | None = None
        self._servo_target_forces_n: wp.array | None = None
        self._servo_force_tolerances_n: wp.array | None = None
        self._servo_checkpoint_float: wp.array | None = None
        self._servo_checkpoint_step: wp.array | None = None
        self._servo_checkpoint_states: tuple[newton.State, ...] = ()
        self._servo_checkpoint_contacts: tuple[newton.Contacts, ...] = ()
        self._servo_initial_translation_W_m: wp.array | None = None
        self._servo_initial_rotation: wp.array | None = None
        self._servo_motion_direction_W: wp.vec3 | None = None
        self._servo_indenter_body_index: int | None = None
        self._servo_approach_speed_m_s: float | None = None
        self._servo_force_gain_m_s_n: float | None = None
        self._servo_settle_ticks: int | None = None
        self._servo_settle_window_ticks: int | None = None
        self._servo_displacement_tolerance_m: float | None = None
        self._servo_target_count = 0
        self._servo_host_intervention_count = 0
        self._servo_host_sync_count = 0
        self._servo_live_selection: tuple[
            newton.State,
            newton.Contacts,
            int,
            float,
        ] | None = None
        self._has_step_result = False
        self.collision_pipeline.collide(self.state, self.contacts)

    def set_fingertip_pose(self, pose: wp.transform) -> None:
        """Set the fingertip pose held across subsequent simulation ticks."""
        if self._step_graph_ab is not None:
            raise RuntimeError(
                "the fingertip pose cannot change after CUDA graph capture"
            )
        self._fingertip_pose = pose

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
        self.fingertip_model.prepare_step(
            state_in,
            state_out,
            self._fingertip_pose,
        )
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

    def configure_force_servo(
        self,
        indenter: Indenter,
        *,
        initial_tf: wp.transform,
        motion_direction_W: wp.vec3,
        approach_speed_m_s: float,
        force_gain_m_s_n: float,
        target_forces_n: tuple[float, ...],
        force_tolerances_n: tuple[float, ...],
        settle_ticks: int,
        displacement_tolerance_m: float | None,
    ) -> None:
        """Prepare the production force servo for GPU graph replay."""
        if not self._use_cuda_graph:
            raise RuntimeError("GPU force servo requires use_cuda_graph=True")
        if self._servo_float is not None:
            raise RuntimeError("force servo has already been configured")
        if len(target_forces_n) != len(force_tolerances_n):
            raise ValueError("force targets and tolerances must have equal length")
        if (
            not target_forces_n
            or len(target_forces_n) > _FORCE_SERVO_CHECKPOINT_CAPACITY
        ):
            raise ValueError("GPU force servo supports one to four force targets")
        if settle_ticks < 1:
            raise ValueError("settle_ticks must be positive")
        if self._step_graph_ab is not None:
            raise RuntimeError("partial step graphs already exist")

        initial_tf_values = np.asarray(initial_tf, dtype=np.float64)
        direction_W = np.asarray(motion_direction_W, dtype=np.float64)
        direction_W /= np.linalg.norm(direction_W)
        device = self.fingertip_model.model.device
        self._servo_initial_translation_W_m = wp.array(
            [wp.vec3(*initial_tf_values[:3])],
            dtype=wp.vec3,
            device=device,
        )
        self._servo_initial_rotation = wp.array(
            [wp.quat(*initial_tf_values[3:])],
            dtype=wp.quat,
            device=device,
        )
        self._servo_motion_direction_W = wp.vec3(*direction_W)
        self._servo_indenter_body_index = indenter.body_index
        self._servo_approach_speed_m_s = float(approach_speed_m_s)
        self._servo_force_gain_m_s_n = float(force_gain_m_s_n)
        self._servo_settle_ticks = int(settle_ticks)
        self._servo_settle_window_ticks = min(
            int(settle_ticks),
            max(1, int(np.ceil(0.5 / self.time_step_s))),
        )
        self._servo_displacement_tolerance_m = (
            -1.0
            if displacement_tolerance_m is None
            else float(displacement_tolerance_m)
        )
        self._servo_target_count = len(target_forces_n)
        self._servo_float = wp.zeros(
            _SERVO_FLOAT_COUNT,
            dtype=float,
            device=device,
        )
        initial_int = np.zeros(_SERVO_INT_COUNT, dtype=np.int32)
        initial_int[_SERVO_EVENT_SLOT] = -1
        self._servo_int = wp.array(initial_int, dtype=wp.int32, device=device)
        self._servo_target_forces_n = wp.array(
            np.asarray(target_forces_n, dtype=np.float32),
            dtype=float,
            device=device,
        )
        self._servo_force_tolerances_n = wp.array(
            np.asarray(force_tolerances_n, dtype=np.float32),
            dtype=float,
            device=device,
        )
        self._servo_checkpoint_float = wp.zeros(
            (self._servo_target_count, _CHECKPOINT_FLOAT_COUNT),
            dtype=float,
            device=device,
        )
        self._servo_checkpoint_step = wp.zeros(
            self._servo_target_count,
            dtype=wp.int32,
            device=device,
        )
        self._servo_checkpoint_states = tuple(
            self.fingertip_model.model.state()
            for _ in range(_FORCE_SERVO_CHECKPOINT_CAPACITY)
        )
        self._servo_checkpoint_contacts = tuple(
            self.collision_pipeline.contacts()
            for _ in range(_FORCE_SERVO_CHECKPOINT_CAPACITY)
        )

        # The first uncaptured tick performs Newton's lazy contact-buffer sizing.
        self._launch_force_servo_tick(self.state, self._next_state)
        self.state, self._next_state = self._next_state, self.state
        self.step_count = 1
        self.time_s = self.time_step_s
        self._has_step_result = True
        self._capture_force_servo_graphs()

    def reset_force_servo(
        self,
        indenter: Indenter,
        *,
        initial_tf: wp.transform,
    ) -> None:
        """Restore one captured runtime to an independent reference state."""
        self._require_force_servo()
        if self._servo_live_selection is not None:
            raise RuntimeError("cannot reset while a checkpoint is selected")
        if indenter.body_index != self._servo_indenter_body_index:
            raise ValueError("reset indenter does not match the captured runtime")

        wp.synchronize_device(self.fingertip_model.model.device)
        reset_flags = (
            newton.StateFlags.BODY_Q
            | newton.StateFlags.BODY_QD
            | newton.StateFlags.PARTICLE_Q
            | newton.StateFlags.PARTICLE_QD
        )
        self.solver.reset(self._state_a, flags=reset_flags)
        self.solver.reset(self._state_b, flags=reset_flags)

        initial_tf_values = np.asarray(initial_tf, dtype=np.float32)
        self._servo_initial_translation_W_m.assign(
            initial_tf_values[None, :3]
        )
        self._servo_initial_rotation.assign(initial_tf_values[None, 3:])
        self.apply_indenter_pose(indenter, initial_tf)
        self.fingertip_model.prepare_step(
            self._state_a,
            self._state_b,
            self._fingertip_pose,
        )

        self.contacts.clear()
        self.collision_pipeline.reset_contact_matching()
        self._state_a.clear_forces()
        self._state_b.clear_forces()
        self._body_qd_before.zero_()
        self._body_wrenches.zero_()
        self._reaction_force_n.zero_()
        self._maximum_particle_speed_m_s.zero_()
        self._nonfinite_particle_velocity_count.zero_()
        self.solver.body_body_contact_overflow_max.zero_()
        self.solver.body_particle_contact_overflow_max.zero_()
        self._servo_float.zero_()
        reset_int = np.zeros(_SERVO_INT_COUNT, dtype=np.int32)
        reset_int[_SERVO_EVENT_SLOT] = -1
        self._servo_int.assign(reset_int)
        self._servo_checkpoint_float.zero_()
        self._servo_checkpoint_step.zero_()
        for checkpoint_contact in self._servo_checkpoint_contacts:
            checkpoint_contact.clear()

        self.state = self._state_a
        self._next_state = self._state_b
        self.step_count = 0
        self.time_s = 0.0
        self._cuda_graph_replay_count = 0
        self._servo_host_intervention_count = 0
        self._servo_host_sync_count = 0
        self._has_step_result = False
        self.collision_pipeline.collide(self.state, self.contacts)

    def _require_force_servo(self) -> None:
        if (
            self._servo_float is None
            or self._servo_int is None
            or self._servo_target_forces_n is None
            or self._servo_force_tolerances_n is None
            or self._servo_checkpoint_float is None
            or self._servo_checkpoint_step is None
            or self._servo_initial_translation_W_m is None
            or self._servo_initial_rotation is None
            or self._servo_motion_direction_W is None
            or self._servo_indenter_body_index is None
            or self._servo_approach_speed_m_s is None
            or self._servo_force_gain_m_s_n is None
            or self._servo_settle_ticks is None
            or self._servo_settle_window_ticks is None
            or self._servo_displacement_tolerance_m is None
        ):
            raise RuntimeError("GPU force servo is not configured")

    def _launch_force_servo_snapshots(self, state_out: newton.State) -> None:
        """Copy an accepted tick into its device-resident checkpoint slot."""
        self._require_force_servo()
        if state_out.particle_q is None or state_out.particle_qd is None:
            raise RuntimeError("simulation state has no particle state")
        particle_outputs: list[wp.array] = []
        contact_outputs: list[wp.array] = []
        for checkpoint_state, checkpoint_contact in zip(
            self._servo_checkpoint_states,
            self._servo_checkpoint_contacts,
            strict=True,
        ):
            if (
                checkpoint_state.particle_q is None
                or checkpoint_state.particle_qd is None
            ):
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
            _snapshot_force_servo_particles,
            dim=self.fingertip_model.model.particle_count,
            inputs=[
                self._servo_int,
                state_out.particle_q,
                state_out.particle_qd,
                *particle_outputs,
            ],
            device=self.fingertip_model.model.device,
        )
        wp.launch(
            _snapshot_force_servo_contacts,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self._servo_int,
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

    def _launch_force_servo_tick(
        self,
        state_in: newton.State,
        state_out: newton.State,
    ) -> None:
        """Launch one complete servo, physics, wrench, and checkpoint tick."""
        self._require_force_servo()
        if state_in.body_q is None or state_out.body_q is None:
            raise RuntimeError("simulation state has no rigid-body poses")
        wp.launch(
            _force_servo_before_step,
            dim=1,
            inputs=[
                self._servo_initial_translation_W_m,
                self._servo_initial_rotation,
                self._servo_motion_direction_W,
                self._servo_indenter_body_index,
                self._servo_approach_speed_m_s,
                self._servo_force_gain_m_s_n,
                self.time_step_s,
                self._servo_target_forces_n,
                self._servo_float,
                self._servo_int,
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
                self._servo_indenter_body_index,
                self._servo_motion_direction_W,
            ],
            outputs=[self._reaction_force_n],
            device=self.fingertip_model.model.device,
        )
        wp.copy(self._servo_float, self._reaction_force_n, count=1)
        wp.launch(
            _force_servo_after_step,
            dim=1,
            inputs=[
                self.time_step_s,
                self._servo_settle_ticks,
                self._servo_settle_window_ticks,
                self._servo_target_count,
                self._servo_displacement_tolerance_m,
                self._servo_target_forces_n,
                self._servo_force_tolerances_n,
                self._servo_float,
                self._servo_int,
                self._servo_checkpoint_float,
                self._servo_checkpoint_step,
            ],
            device=self.fingertip_model.model.device,
        )
        self._launch_force_servo_snapshots(state_out)

    def _capture_force_servo_graphs(self) -> None:
        """Capture two-tick graphs that return to their starting parity."""
        wp.synchronize_device(self.fingertip_model.model.device)
        self._validate_cuda_graph_capture_ready()
        allocation_signature = self._cuda_graph_contact_allocation_signature()
        with wp.ScopedCapture(
            device=self.fingertip_model.model.device,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as capture_a:
            self._launch_force_servo_tick(self._state_a, self._state_b)
            self._launch_force_servo_tick(self._state_b, self._state_a)
        self._servo_graph_a = capture_a.graph
        if allocation_signature != self._cuda_graph_contact_allocation_signature():
            raise RuntimeError("force-servo graph A replaced Newton contact storage")

        with wp.ScopedCapture(
            device=self.fingertip_model.model.device,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as capture_b:
            self._launch_force_servo_tick(self._state_b, self._state_a)
            self._launch_force_servo_tick(self._state_a, self._state_b)
        self._servo_graph_b = capture_b.graph
        if allocation_signature != self._cuda_graph_contact_allocation_signature():
            raise RuntimeError("force-servo graph B replaced Newton contact storage")

    def launch_force_servo(
        self,
        tick_count: int,
        *,
        stream: wp.Stream | None = None,
    ) -> None:
        """Enqueue device-resident servo ticks without a host readback."""
        self._require_force_servo()
        if tick_count < 1:
            raise ValueError("tick_count must be positive")
        if stream is not None and stream.device != self.fingertip_model.model.device:
            raise ValueError("force-servo stream belongs to another device")
        remaining = tick_count
        while remaining >= 2:
            if self.state is self._state_a:
                if self._servo_graph_a is None:
                    raise RuntimeError("force-servo graph A is unavailable")
                wp.capture_launch(self._servo_graph_a, stream=stream)
            elif self.state is self._state_b:
                if self._servo_graph_b is None:
                    raise RuntimeError("force-servo graph B is unavailable")
                wp.capture_launch(self._servo_graph_b, stream=stream)
            else:
                raise RuntimeError("live state is not an A/B state")
            self._cuda_graph_replay_count += 1
            self.step_count += 2
            remaining -= 2
        if remaining:
            with wp.ScopedStream(stream, sync_enter=False):
                self._launch_force_servo_tick(self.state, self._next_state)
            self.state, self._next_state = self._next_state, self.state
            self.step_count += 1
        self.time_s = self.step_count * self.time_step_s
        self._has_step_result = True
        self._servo_host_intervention_count += 1

    def force_servo_status(
        self,
        *,
        stream: wp.Stream | None = None,
    ) -> tuple[int, bool, int]:
        """Synchronize one servo stream and return its compact device status."""
        self._require_force_servo()
        if stream is not None:
            wp.synchronize_stream(stream)
        self._servo_host_sync_count += 1
        status = self._servo_int.numpy()
        return (
            int(status[_SERVO_CHECKPOINT_COUNT]),
            bool(status[_SERVO_FINISHED]),
            int(status[_SERVO_ERROR]),
        )

    def advance_force_servo(self, tick_count: int) -> tuple[int, bool, int]:
        """Replay device-resident servo ticks and return coarse host status."""
        self.launch_force_servo(tick_count)
        return self.force_servo_status()

    def force_servo_checkpoint(self, index: int) -> dict[str, float | int]:
        """Read one compact checkpoint record after device completion."""
        self._require_force_servo()
        if index < 0 or index >= self._servo_target_count:
            raise ValueError("checkpoint index is outside the target schedule")
        checkpoint_float = self._servo_checkpoint_float.numpy()[index]
        checkpoint_step = int(self._servo_checkpoint_step.numpy()[index])
        self._servo_host_sync_count += 2
        return {
            "reaction_force_n": float(checkpoint_float[_CHECKPOINT_FORCE]),
            "travel_m": float(checkpoint_float[_CHECKPOINT_TRAVEL]),
            "force_change_n": float(checkpoint_float[_CHECKPOINT_FORCE_CHANGE]),
            "force_reference_n": float(
                checkpoint_float[_CHECKPOINT_FORCE_REFERENCE]
            ),
            "reaction_force_rate_n_s": float(
                checkpoint_float[_CHECKPOINT_FORCE_RATE]
            ),
            "indentation_rate_m_s": float(
                checkpoint_float[_CHECKPOINT_INDENTATION_RATE]
            ),
            "servo_error_n": float(checkpoint_float[_CHECKPOINT_SERVO_ERROR]),
            "settle_window_force_drift_n": float(
                checkpoint_float[_CHECKPOINT_FORCE]
                - checkpoint_float[_CHECKPOINT_WINDOW_START_FORCE]
            ),
            "settle_window_indentation_drift_m": float(
                checkpoint_float[_CHECKPOINT_TRAVEL]
                - checkpoint_float[_CHECKPOINT_WINDOW_START_TRAVEL]
            ),
            "step_count": checkpoint_step,
            "simulation_time_s": checkpoint_step * self.time_step_s,
        }

    def force_servo_current_force_n(self) -> float:
        """Read the current projected force for a terminal error report."""
        self._require_force_servo()
        value = float(self._servo_float.numpy()[_SERVO_REACTION_FORCE])
        self._servo_host_sync_count += 1
        return value

    def select_force_servo_checkpoint(self, index: int) -> None:
        """Expose one exact saved checkpoint through the existing callback API."""
        if self._servo_live_selection is not None:
            raise RuntimeError("a checkpoint is already selected")
        if index < 0 or index >= self._servo_target_count:
            raise ValueError("checkpoint index is outside the target schedule")
        checkpoint_step = int(self._servo_checkpoint_step.numpy()[index])
        self._servo_host_sync_count += 1
        self._servo_live_selection = (
            self.state,
            self.contacts,
            self.step_count,
            self.time_s,
        )
        self.state = self._servo_checkpoint_states[index]
        self.contacts = self._servo_checkpoint_contacts[index]
        self.step_count = checkpoint_step
        self.time_s = checkpoint_step * self.time_step_s

    def restore_force_servo_live_state(self) -> None:
        """Restore the live graph state after one checkpoint callback."""
        if self._servo_live_selection is None:
            raise RuntimeError("no checkpoint is selected")
        self.state, self.contacts, self.step_count, self.time_s = (
            self._servo_live_selection
        )
        self._servo_live_selection = None

    @property
    def force_servo_performance(self) -> dict[str, float | int]:
        """Return graph replay and synchronization diagnostics."""
        interventions = self._servo_host_intervention_count
        live_step_count = (
            self.step_count
            if self._servo_live_selection is None
            else self._servo_live_selection[2]
        )
        return {
            "graph_replay_count": self._cuda_graph_replay_count,
            "host_intervention_count": interventions,
            "host_sync_count": self._servo_host_sync_count,
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

    def _capture_step_graphs(self) -> None:
        """Capture the two fixed ping-pong Newton step graphs."""
        wp.synchronize_device(self.fingertip_model.model.device)
        self._validate_cuda_graph_capture_ready()
        allocation_signature = self._cuda_graph_contact_allocation_signature()

        with wp.ScopedCapture(
            device=self.fingertip_model.model.device,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as capture_ab:
            self._launch_step_device_operations(self._state_a, self._state_b)
        self._step_graph_ab = capture_ab.graph
        if allocation_signature != self._cuda_graph_contact_allocation_signature():
            raise RuntimeError("graph_AB capture replaced Newton contact storage")

        with wp.ScopedCapture(
            device=self.fingertip_model.model.device,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as capture_ba:
            self._launch_step_device_operations(self._state_b, self._state_a)
        self._step_graph_ba = capture_ba.graph
        if allocation_signature != self._cuda_graph_contact_allocation_signature():
            raise RuntimeError("graph_BA capture replaced Newton contact storage")

    def step(self) -> None:
        """Advance one global tick and record its rigid-body wrenches."""
        if self._step_graph_ab is None:
            self._launch_step_device_operations(self.state, self._next_state)
        elif self.state is self._state_a and self._next_state is self._state_b:
            wp.capture_launch(self._step_graph_ab)
            self._cuda_graph_replay_count += 1
        elif self.state is self._state_b and self._next_state is self._state_a:
            wp.capture_launch(self._step_graph_ba)
            self._cuda_graph_replay_count += 1
        else:
            raise RuntimeError("simulation state buffers no longer form A/B pairs")

        self.state, self._next_state = self._next_state, self.state
        self.step_count += 1
        self.time_s = self.step_count * self.time_step_s
        self._has_step_result = True
        if self._use_cuda_graph and self._step_graph_ab is None:
            self._capture_step_graphs()

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
