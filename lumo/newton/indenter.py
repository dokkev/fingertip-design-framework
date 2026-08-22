"""Kinematic rigid indenters for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import newton
import numpy as np
import warp as wp

from lumo.util.scalar_validation import require_finite, require_positive

if TYPE_CHECKING:
    from lumo.newton.solver import FingertipNewtonSolver


@wp.kernel
def _set_indenter_pose(
    pose: wp.transform,
    body_index: int,
    body_q: wp.array(dtype=wp.transform),
):
    body_q[body_index] = pose


@dataclass(frozen=True)
class Indenter:
    """One kinematic rigid indenter in a Newton model."""

    body_index: int

    @classmethod
    def add_urdf(
        cls,
        builder: newton.ModelBuilder,
        urdf_path: str | Path,
        *,
        tf: wp.transform | None = None,
    ) -> Self:
        """Add one rigid URDF as a kinematic indenter."""
        path = Path(urdf_path)
        if path.suffix.lower() != ".urdf":
            raise ValueError("urdf_path must be a .urdf file")
        if not path.is_file():
            raise FileNotFoundError(path)
        if tf is None:
            tf = wp.transform_identity()

        body_start = builder.body_count
        builder.add_urdf(
            str(path),
            xform=tf,
            floating=True,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )

        if builder.body_count != body_start + 1:
            raise ValueError(
                "indenter URDF must describe one rigid body "
                "after fixed-joint collapse"
            )

        # Newton stores the free-root pose in joint_q during URDF import, but
        # newly created State objects are initialized from builder.body_q.
        # Keep both representations at the requested initial world pose.
        builder.body_q[body_start] = tf
        builder.body_flags[body_start] = int(newton.BodyFlags.KINEMATIC)
        return cls(body_index=body_start)

    @classmethod
    def add_mesh(
        cls,
        builder: newton.ModelBuilder,
        mesh: newton.Mesh,
        *,
        tf: wp.transform | None = None,
        cfg: newton.ModelBuilder.ShapeConfig | None = None,
    ) -> Self:
        """Add one prepared Newton mesh as a kinematic indenter."""
        if tf is None:
            tf = wp.transform_identity()

        body_index = builder.add_body(
            xform=tf,
            is_kinematic=True,
            label="indenter",
        )
        builder.add_shape_mesh(
            body_index,
            mesh=mesh,
            cfg=cfg,
            label="indenter",
        )
        return cls(body_index=body_index)

    def get_current_pose(self, state: newton.State) -> wp.transform:
        """Return the current pose, synchronizing rigid-body state to the host."""
        if state.body_q is None:
            raise ValueError("state must contain rigid-body poses")

        pose = state.body_q.numpy()[self.body_index]
        return wp.transform(
            wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])),
            wp.quat(
                float(pose[3]),
                float(pose[4]),
                float(pose[5]),
                float(pose[6]),
            ),
        )

    def apply_pose(
        self,
        state: newton.State,
        pose: wp.transform,
    ) -> None:
        """Apply one prescribed pose to this indenter in a Newton state."""
        if state.body_q is None:
            raise ValueError("state must contain rigid-body poses")

        wp.launch(
            _set_indenter_pose,
            dim=1,
            inputs=[pose, self.body_index],
            outputs=[state.body_q],
            device=state.body_q.device,
        )

    def move_until_force(
        self,
        fingertip_solver: FingertipNewtonSolver,
        *,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        f_des_n: float,
        carrier_pose: wp.transform,
        max_steps: int,
    ) -> float:
        """Translate until the opposing transient reaction reaches ``f_des_n``.

        ``dx_m``, ``dy_m``, and ``dz_m`` are the fixed world-frame translation
        applied per solver step. The returned force is the first reaction-force
        projection opposite that translation whose magnitude reaches the
        inclusive threshold.
        """
        require_finite("dx_m", dx_m)
        require_finite("dy_m", dy_m)
        require_finite("dz_m", dz_m)
        require_positive("f_des_n", f_des_n)
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            fingertip_solver.indenter is None
            or fingertip_solver.indenter.body_index != self.body_index
        ):
            raise ValueError(
                "fingertip_solver must be configured with this indenter"
            )

        translation_step_W = np.asarray(
            [dx_m, dy_m, dz_m],
            dtype=np.float64,
        )
        translation_step_m = float(np.linalg.norm(translation_step_W))
        if translation_step_m == 0.0:
            raise ValueError("indenter translation step must be non-zero")
        motion_direction_W = translation_step_W / translation_step_m

        current_pose = self.get_current_pose(fingertip_solver.state)
        position_W = np.asarray(
            [float(current_pose[i]) for i in range(3)],
            dtype=np.float64,
        )
        rotation_W = wp.quat(
            float(current_pose[3]),
            float(current_pose[4]),
            float(current_pose[5]),
            float(current_pose[6]),
        )

        model = fingertip_solver.fingertip_model.model
        body_to_wrench = np.full(model.body_count, -1, dtype=np.int32)
        body_to_wrench[self.body_index] = 0
        body_to_wrench_device = wp.array(
            body_to_wrench,
            dtype=wp.int32,
            device=model.device,
        )
        indenter_wrench = wp.zeros(
            1,
            dtype=wp.spatial_vector,
            device=model.device,
        )

        peak_reaction_force_n = 0.0
        for _ in range(max_steps):
            position_W += translation_step_W
            pose = wp.transform(
                wp.vec3(
                    float(position_W[0]),
                    float(position_W[1]),
                    float(position_W[2]),
                ),
                rotation_W,
            )

            # Preserve VBD's previous rigid pose before the prescribed update
            # so transient contact damping uses the indenter velocity.
            self.apply_pose(fingertip_solver.state, pose)
            fingertip_solver.solver.coupling_notify_input_state_update(
                fingertip_solver.state,
                newton.StateFlags.BODY_Q,
                dt=fingertip_solver.time_step_s,
            )
            if fingertip_solver.state.body_qd is None:
                raise RuntimeError(
                    "solver state has no rigid-body velocities"
                )
            body_qd_before = wp.clone(fingertip_solver.state.body_qd)
            state_before = fingertip_solver.state

            fingertip_solver.step(
                carrier_pose=carrier_pose,
                indenter_pose=pose,
            )

            # SolverVBD does not populate Contacts.force for rigid-soft
            # contact. Its public coupling hook exposes the contact-only body
            # wrench using the same force law as the solve.
            fingertip_solver.solver.coupling_harvest_proxy_wrenches(
                body_to_wrench_device,
                indenter_wrench,
                body_qd_before=body_qd_before,
                state=state_before,
                state_out=fingertip_solver.state,
                contacts=fingertip_solver.contacts,
                dt=fingertip_solver.time_step_s,
            )
            wrench_on_indenter_W = indenter_wrench.numpy()[0]
            if not np.all(np.isfinite(wrench_on_indenter_W)):
                raise RuntimeError(
                    "indenter contact produced a non-finite wrench"
                )

            force_on_indenter_W = wrench_on_indenter_W[:3]
            reaction_force_n = max(
                0.0,
                -float(np.dot(force_on_indenter_W, motion_direction_W)),
            )
            peak_reaction_force_n = max(
                peak_reaction_force_n,
                reaction_force_n,
            )
            if reaction_force_n >= f_des_n:
                return reaction_force_n

        raise RuntimeError(
            "indenter exhausted max_steps without reaching the transient "
            f"force threshold; peak force was {peak_reaction_force_n:.9e} N"
        )


__all__ = ["Indenter"]
