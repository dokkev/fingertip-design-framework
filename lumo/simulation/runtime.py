"""Concrete runtime for one LUMO simulation."""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton.indenter import Indenter
from lumo.newton.model import build_fingertip_newton_model
from lumo.util.scalar_validation import require_nonnegative, require_positive


_DEFAULT_SOFT_CONTACT_MARGIN_M = 1.0e-4
_DEFAULT_ELEMENT_SIZE_MM = 1.0
_DEFAULT_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6
_VBD_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 2048


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
