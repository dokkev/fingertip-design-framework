"""Concrete runtime for one LUMO simulation."""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import make_fingertip_mesh
from lumo.newton.indenter import Indenter
from lumo.newton.model import build_fingertip_newton_model
from lumo.util.scalar_validation import require_nonnegative, require_positive


@wp.kernel
def _set_body_pose(
    pose: wp.transform,
    body_index: int,
    body_q: wp.array(dtype=wp.transform),
):
    body_q[body_index] = pose


class LumoSimulation:
    """Construct and own the runtime for one analytic LUMO fingertip."""

    def __init__(
        self,
        fingertip: Fingertip,
        *,
        builder: newton.ModelBuilder | None = None,
        sim_frequency: float,
        iterations: int = 10,
        soft_contact_margin_m: float = 0.0,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        if builder is not None and not isinstance(
            builder,
            newton.ModelBuilder,
        ):
            raise TypeError("builder must be a newton.ModelBuilder")
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

        self.fingertip = fingertip
        self.fingertip_mesh = make_fingertip_mesh(fingertip)
        self.fingertip_model = build_fingertip_newton_model(
            self.fingertip_mesh,
            builder=builder,
        )
        model = self.fingertip_model.model

        self.sim_frequency = float(sim_frequency)
        self.time_step_s = 1.0 / self.sim_frequency
        self.step_count = 0
        self.time_s = 0.0
        self._fingertip_pose = wp.transform_identity()
        self.solver = newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            particle_enable_self_contact=False,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            model,
            soft_contact_margin=soft_contact_margin_m,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state = model.state()
        self._next_state = model.state()
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
        self._has_step_result = False

    def set_fingertip_pose(self, pose: wp.transform) -> None:
        """Set the fingertip pose held across subsequent simulation ticks."""
        self._fingertip_pose = pose

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

    def step(self) -> None:
        """Advance one global tick and record its rigid-body wrenches."""
        self.fingertip_model.prepare_step(
            self.state,
            self._next_state,
            self._fingertip_pose,
        )
        if self.state.body_qd is None:
            raise RuntimeError("simulation state has no rigid-body velocities")
        wp.copy(self._body_qd_before, self.state.body_qd)
        state_before = self.state

        self.solver.coupling_notify_input_state_update(
            self.state,
            newton.StateFlags.BODY_Q,
            dt=self.time_step_s,
        )

        self.state.clear_forces()
        self.collision_pipeline.collide(self.state, self.contacts)
        self.solver.step(
            self.state,
            self._next_state,
            self.control,
            self.contacts,
            self.time_step_s,
        )
        self._body_wrenches.zero_()
        self.solver.coupling_harvest_proxy_wrenches(
            self._body_to_wrench,
            self._body_wrenches,
            body_qd_before=self._body_qd_before,
            state=state_before,
            state_out=self._next_state,
            contacts=self.contacts,
            dt=self.time_step_s,
        )
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
        direction_W /= direction_norm

        wrench_on_indenter_W = self._body_wrenches.numpy()[
            indenter.body_index
        ]
        if not np.all(np.isfinite(wrench_on_indenter_W)):
            raise RuntimeError(
                "simulation step produced a non-finite body wrench"
            )

        force_on_indenter_W = wrench_on_indenter_W[:3]
        return max(
            0.0,
            -float(np.dot(force_on_indenter_W, direction_W)),
        )


__all__ = ["LumoSimulation"]
