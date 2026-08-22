"""Concrete runtime for one LUMO simulation."""

from __future__ import annotations

import newton
import warp as wp

from lumo.newton.indenter import Indenter
from lumo.newton.model import FingertipNewtonModel
from lumo.util.scalar_validation import require_nonnegative, require_positive


@wp.kernel
def _set_body_pose(
    pose: wp.transform,
    body_index: int,
    body_q: wp.array(dtype=wp.transform),
):
    body_q[body_index] = pose


class LumoSimulation:
    """Own the Newton runtime and global clock for one LUMO simulation."""

    def __init__(
        self,
        fingertip_model: FingertipNewtonModel,
        *,
        time_step_s: float,
        iterations: int = 10,
        soft_contact_margin_m: float = 0.0,
    ) -> None:
        if not isinstance(fingertip_model, FingertipNewtonModel):
            raise TypeError("fingertip_model must be a FingertipNewtonModel")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")
        require_positive("time_step_s", time_step_s)
        require_nonnegative(
            "soft_contact_margin_m",
            soft_contact_margin_m,
        )

        self.fingertip_model = fingertip_model
        self.time_step_s = time_step_s
        self.time_s = 0.0
        self.solver = newton.solvers.SolverVBD(
            fingertip_model.model,
            iterations=iterations,
            particle_enable_self_contact=False,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            fingertip_model.model,
            soft_contact_margin=soft_contact_margin_m,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state = fingertip_model.model.state()
        self._next_state = fingertip_model.model.state()
        self.control = fingertip_model.model.control()
        self._body_poses_updated = False

    def apply_carrier_pose(self, pose: wp.transform) -> None:
        """Apply the prescribed carrier pose to both state buffers."""
        self.fingertip_model.prepare_step(
            self.state,
            self._next_state,
            pose,
        )
        self._body_poses_updated = True

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
        self._body_poses_updated = True

    def step(self) -> None:
        """Advance the complete Newton world by one fixed timestep."""
        if self._body_poses_updated:
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
        self.state, self._next_state = self._next_state, self.state
        self._body_poses_updated = False
        self.time_s += self.time_step_s


__all__ = ["LumoSimulation"]
