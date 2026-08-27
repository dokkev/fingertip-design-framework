"""Kinematic rigid indenters for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import newton
import warp as wp

from lumo.util.scalar_validation import require_nonnegative, require_positive


@dataclass(frozen=True)
class Indenter:
    """One kinematic rigid indenter in a Newton model."""

    body_index: int

    def __post_init__(self) -> None:
        if isinstance(self.body_index, bool) or not isinstance(
            self.body_index,
            int,
        ):
            raise TypeError("body_index must be an integer")
        if self.body_index < 0:
            raise ValueError("body_index must be non-negative")

    @classmethod
    def add_urdf(
        cls,
        builder: newton.ModelBuilder,
        urdf_path: str | Path,
        *,
        tf: wp.transform | None = None,
        contact_stiffness_n_m: float | None = None,
        contact_damping_n_s_m: float | None = None,
    ) -> Self:
        """Add one rigid URDF as a kinematic indenter."""
        path = Path(urdf_path)
        if path.suffix.lower() != ".urdf":
            raise ValueError("urdf_path must be a .urdf file")
        if not path.is_file():
            raise FileNotFoundError(path)
        if contact_stiffness_n_m is not None:
            require_positive("contact_stiffness_n_m", contact_stiffness_n_m)
        if contact_damping_n_s_m is not None:
            require_nonnegative(
                "contact_damping_n_s_m",
                contact_damping_n_s_m,
            )
        if tf is None:
            tf = wp.transform_identity()

        body_start = builder.body_count
        previous_stiffness_n_m = builder.default_shape_cfg.ke
        previous_damping_n_s_m = builder.default_shape_cfg.kd
        try:
            if contact_stiffness_n_m is not None:
                builder.default_shape_cfg.ke = float(
                    contact_stiffness_n_m
                )
            if contact_damping_n_s_m is not None:
                builder.default_shape_cfg.kd = float(
                    contact_damping_n_s_m
                )
            # Import a free root so its world pose can be prescribed
            # kinematically after fixed-joint collapse.
            builder.add_urdf(
                str(path),
                xform=tf,
                floating=True,
                collapse_fixed_joints=True,
                enable_self_collisions=False,
            )
        finally:
            builder.default_shape_cfg.ke = previous_stiffness_n_m
            builder.default_shape_cfg.kd = previous_damping_n_s_m

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

__all__ = ["Indenter"]
