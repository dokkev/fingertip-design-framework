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

    @classmethod
    def add_mesh(
        cls,
        builder: newton.ModelBuilder,
        mesh: newton.Mesh,
        *,
        tf: wp.transform | None = None,
        cfg: newton.ModelBuilder.ShapeConfig | None = None,
        contact_stiffness_n_m: float | None = None,
        contact_damping_n_s_m: float | None = None,
    ) -> Self:
        """Add one prepared Newton mesh as a kinematic indenter."""
        if contact_stiffness_n_m is not None:
            require_positive("contact_stiffness_n_m", contact_stiffness_n_m)
        if contact_damping_n_s_m is not None:
            require_nonnegative(
                "contact_damping_n_s_m",
                contact_damping_n_s_m,
            )
        if tf is None:
            tf = wp.transform_identity()
        shape_cfg = cfg
        if (
            contact_stiffness_n_m is not None
            or contact_damping_n_s_m is not None
        ):
            shape_cfg = (
                builder.default_shape_cfg.copy()
                if cfg is None
                else cfg.copy()
            )
            if contact_stiffness_n_m is not None:
                shape_cfg.ke = float(contact_stiffness_n_m)
            if contact_damping_n_s_m is not None:
                shape_cfg.kd = float(contact_damping_n_s_m)

        body_index = builder.add_body(
            xform=tf,
            is_kinematic=True,
            label="indenter",
        )
        builder.add_shape_mesh(
            body_index,
            mesh=mesh,
            cfg=shape_cfg,
            label="indenter",
        )
        return cls(body_index=body_index)


__all__ = ["Indenter"]
