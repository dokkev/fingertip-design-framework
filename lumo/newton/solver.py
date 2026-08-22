"""SolverVBD execution for the current fingertip Newton model."""

from __future__ import annotations

import newton
import warp as wp

from lumo.newton.indenter import Indenter
from lumo.newton.model import FingertipNewtonModel
from lumo.util.scalar_validation import require_nonnegative, require_positive


class FingertipNewtonSolver:
    """Own fixed-step SolverVBD execution for one fingertip model."""

    def __init__(
        self,
        fingertip_model: FingertipNewtonModel,
        *,
        time_step_s: float,
        indenter: Indenter | None = None,
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
        if indenter is not None and not isinstance(indenter, Indenter):
            raise TypeError("indenter must be an Indenter or None")
        require_positive("time_step_s", time_step_s)
        require_nonnegative(
            "soft_contact_margin_m",
            soft_contact_margin_m,
        )

        self.fingertip_model = fingertip_model
        self.indenter = indenter
        self.time_step_s = time_step_s
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

    def step(
        self,
        *,
        carrier_pose: wp.transform,
        indenter_pose: wp.transform | None = None,
    ) -> None:
        """Advance VBD by the solver's fixed timestep."""
        if indenter_pose is not None:
            if self.indenter is None:
                raise ValueError(
                    "indenter_pose requires an indenter in the solver"
                )
            self.indenter.apply_pose(self.state, indenter_pose)
            self.indenter.apply_pose(self._next_state, indenter_pose)

        self.fingertip_model.prepare_step(
            self.state,
            self._next_state,
            carrier_pose,
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


__all__ = ["FingertipNewtonSolver"]
