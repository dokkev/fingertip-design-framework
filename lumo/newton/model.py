"""Newton model construction for the current fingertip boundary condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import newton
import warp as wp

from lumo.mesh import FingertipMesh

if TYPE_CHECKING:
    from collections.abc import Sequence


# VBD body-particle contact is penalty-based. This keeps the nominally rigid
# carrier below the validation's 10 micrometre penetration tolerance at 15 N.
_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6


@wp.kernel
def _set_body_pose(
    pose: wp.transform,
    body_index: int,
    body_q: wp.array(dtype=wp.transform),
):
    body_q[body_index] = pose


@wp.kernel
def _set_bonded_particles(
    pose: wp.transform,
    bonded_particle_indices: wp.array(dtype=wp.int32),
    bonded_local_positions: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    particle_index = bonded_particle_indices[index]
    particle_q[particle_index] = wp.transform_point(
        pose,
        bonded_local_positions[index],
    )
    particle_qd[particle_index] = wp.vec3(0.0)


@dataclass
class FingertipNewtonModel:
    """Newton model with a prescribed carrier and perfect silicone bond.

    ``bonded_particle_indices`` are global Newton particle indices.  Their
    reference positions are kept in ``bonded_local_positions`` so the bond
    can be applied without copying particle arrays to the host.
    """

    fingertip_mesh: FingertipMesh
    model: newton.Model
    carrier_body: int
    carrier_shape: int
    carrier_collision_shape: int
    bonded_particle_indices: wp.array
    bonded_local_positions: wp.array

    def apply_carrier_pose(
        self,
        state: newton.State,
        pose: wp.transform,
    ) -> None:
        """Apply one prescribed carrier pose to one Newton state."""
        if state.body_q is None:
            raise ValueError("state must contain rigid-body poses")
        if state.particle_q is None or state.particle_qd is None:
            raise ValueError("state must contain particle positions and velocities")

        wp.launch(
            _set_body_pose,
            dim=1,
            inputs=[pose, self.carrier_body],
            outputs=[state.body_q],
            device=self.model.device,
        )
        wp.launch(
            _set_bonded_particles,
            dim=self.bonded_particle_indices.shape[0],
            inputs=[
                pose,
                self.bonded_particle_indices,
                self.bonded_local_positions,
            ],
            outputs=[state.particle_q, state.particle_qd],
            device=self.model.device,
        )

    def prepare_step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        pose: wp.transform,
    ) -> None:
        """Update both state buffers before one solver step.

        The caller should invoke this before one global Newton step. Updating
        both buffers keeps the prescribed bond valid after the runtime swaps
        its state references.
        """
        self.apply_carrier_pose(state_in, pose)
        if state_out is not state_in:
            self.apply_carrier_pose(state_out, pose)


def _bonded_local_positions(
    particle_positions: Sequence[object],
    particle_indices: np.ndarray,
) -> np.ndarray:
    """Copy selected builder positions into one compact float32 array."""
    return np.asarray(
        [
            [
                float(particle_positions[index][axis])
                for axis in range(3)
            ]
            for index in particle_indices
        ],
        dtype=np.float32,
    )


def build_fingertip_newton_model(
    fingertip_mesh: FingertipMesh,
    *,
    builder: newton.ModelBuilder | None = None,
    gravity: float = 0.0,
    carrier_color: wp.vec3 | None = None,
    device: str | None = None,
) -> FingertipNewtonModel:
    """Build the first concrete Newton model for one fingertip mesh.

    The carrier is a kinematic rigid body at the identity pose. Its full mesh
    is visualization-only, while a second mesh exposes only the cavity-facing
    particle-contact surface and closes through the carrier interior. A caller
    may supply a builder already containing external scene bodies; this
    function adds the fingertip and finalizes that builder.
    """
    if not isinstance(fingertip_mesh, FingertipMesh):
        raise TypeError("fingertip_mesh must be a FingertipMesh")

    local_indices = np.asarray(
        fingertip_mesh.bonded_vertex_indices,
        dtype=np.int32,
    )
    if local_indices.size == 0:
        raise ValueError("fingertip mesh must contain bonded vertices")
    if np.any(local_indices >= fingertip_mesh.silicone.vertex_count):
        raise ValueError("bonded vertex index exceeds silicone vertex count")

    parameters = fingertip_mesh.fingertip.parameters
    material = parameters.viscoelastic

    if builder is None:
        builder = newton.ModelBuilder(gravity=gravity)
    elif gravity != 0.0:
        raise ValueError(
            "gravity must be configured on a caller-supplied builder"
        )

    particle_start = builder.particle_count
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        mesh=fingertip_mesh.silicone,
        density=material.density_kg_m3,
        k_mu=material.k_mu_pa,
        k_lambda=material.k_lambda_pa,
        k_damp=material.damping,
        particle_radius=0.0,
        label="fingertip_silicone",
    )

    global_indices = particle_start + local_indices
    active_flag = int(newton.ParticleFlags.ACTIVE)
    for particle_index in global_indices:
        # Newton's particle solver skips non-active particles. Keep their
        # physical mass unchanged; the bond update supplies their positions.
        builder.particle_flags[int(particle_index)] = (
            int(builder.particle_flags[int(particle_index)])
            & ~active_flag
        )

    carrier_body = builder.add_body(
        xform=wp.transform_identity(),
        mass=0.0,
        lock_inertia=True,
        is_kinematic=True,
        label="fingertip_carrier",
    )
    carrier_cfg = newton.ModelBuilder.ShapeConfig(
        collision_group=0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    carrier_shape = builder.add_shape_mesh(
        body=carrier_body,
        mesh=fingertip_mesh.carrier,
        cfg=carrier_cfg,
        color=carrier_color,
        label="fingertip_carrier_surface",
    )
    carrier_collision_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=_CARRIER_CONTACT_STIFFNESS_N_M,
        margin=0.0,
        is_solid=True,
        collision_group=1,
        has_shape_collision=False,
        has_particle_collision=True,
        is_visible=False,
    )
    carrier_collision_shape = builder.add_shape_mesh(
        body=carrier_body,
        mesh=fingertip_mesh.carrier_collision,
        cfg=carrier_collision_cfg,
        label="fingertip_carrier_collision_surface",
    )

    builder.color()
    model = builder.finalize(device=device, requires_grad=False)
    local_positions = _bonded_local_positions(
        builder.particle_q,
        global_indices,
    )

    return FingertipNewtonModel(
        fingertip_mesh=fingertip_mesh,
        model=model,
        carrier_body=carrier_body,
        carrier_shape=carrier_shape,
        carrier_collision_shape=carrier_collision_shape,
        bonded_particle_indices=wp.array(
            global_indices,
            dtype=wp.int32,
            device=model.device,
        ),
        bonded_local_positions=wp.array(
            local_positions,
            dtype=wp.vec3,
            device=model.device,
        ),
    )


__all__ = ["FingertipNewtonModel", "build_fingertip_newton_model"]
