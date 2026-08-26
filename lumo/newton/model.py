"""Newton model construction for the current fingertip boundary condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import newton
import warp as wp

from lumo.mesh import FingertipMesh
from lumo.util.scalar_validation import require_positive

if TYPE_CHECKING:
    from collections.abc import Sequence


# VBD rigid-soft contact is penalty-based. This is the validated carrier-contact
# stiffness used by the current model.
_DEFAULT_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6


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
    silicone_particle_start: int
    silicone_particle_count: int
    bonded_particle_indices: wp.array
    bonded_local_positions: wp.array

    def silicone_vertices(self, state: newton.State) -> np.ndarray:
        """Return current silicone positions in fingertip-mesh vertex order."""
        if state.particle_q is None:
            raise ValueError("state must contain particle positions")

        particle_stop = (
            self.silicone_particle_start + self.silicone_particle_count
        )
        particle_positions = state.particle_q.numpy()
        if particle_stop > particle_positions.shape[0]:
            raise ValueError("state does not contain the silicone particle range")

        vertices = np.ascontiguousarray(
            particle_positions[self.silicone_particle_start:particle_stop],
            dtype=np.float32,
        )
        expected_shape = (self.silicone_particle_count, 3)
        if vertices.shape != expected_shape:
            raise ValueError(
                f"silicone particle positions must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(vertices)):
            raise RuntimeError("silicone particle positions are not finite")
        return vertices

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
    carrier_contact_stiffness_n_m: float = (
        _DEFAULT_CARRIER_CONTACT_STIFFNESS_N_M
    ),
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
    require_positive(
        "carrier_contact_stiffness_n_m",
        carrier_contact_stiffness_n_m,
    )

    local_indices = np.asarray(
        fingertip_mesh.bonded_vertex_indices,
        dtype=np.int32,
    )
    if local_indices.size == 0:
        raise ValueError("fingertip mesh must contain bonded vertices")
    if np.any(local_indices >= fingertip_mesh.silicone.vertex_count):
        raise ValueError("bonded vertex index exceeds silicone vertex count")

    parameters = fingertip_mesh.fingertip.parameters
    material = parameters.mechanics

    if builder is None:
        builder = newton.ModelBuilder(
            gravity=wp.vec3(0.0, 0.0, 0.0),
        )

    particle_start = builder.particle_count
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        mesh=fingertip_mesh.silicone,
        density=material.density_kg_m3,
        k_mu=material.shear_modulus_pa,
        k_lambda=material.lame_lambda_pa,
        k_damp=material.damping_pa_s,
        particle_radius=0.0,
        label="fingertip_silicone",
    )
    particle_count = builder.particle_count - particle_start
    if particle_count != fingertip_mesh.silicone.vertex_count:
        raise RuntimeError(
            "Newton did not preserve the fingertip silicone vertex count"
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

    # Full-surface rigid/soft contact catches tetrahedral faces that can pass
    # between particle vertices.  Build the carrier proxy SDF once per mesh;
    # the collision pipeline consumes it after finalization.
    if fingertip_mesh.carrier_collision.sdf is None:
        fingertip_mesh.carrier_collision.build_sdf(
            max_resolution=256,
            margin=5.0e-4,
            narrow_band_range=(-1.0e-2, 1.0e-2),
            texture_format="float32",
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
    builder.add_shape_mesh(
        body=carrier_body,
        mesh=fingertip_mesh.carrier,
        cfg=carrier_cfg,
        label="fingertip_carrier_surface",
    )
    carrier_collision_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=carrier_contact_stiffness_n_m,
        margin=0.0,
        is_solid=True,
        collision_group=1,
        # Newton's full-surface soft contact path requires a provisioned SDF
        # on a participating mesh shape.  Rigid shape pairs are filtered
        # below; particle contact remains the only physical carrier contact.
        has_shape_collision=True,
        has_particle_collision=True,
        is_visible=False,
    )
    existing_shape_indices = tuple(range(builder.shape_count))
    carrier_collision_shape = builder.add_shape_mesh(
        body=carrier_body,
        mesh=fingertip_mesh.carrier_collision,
        cfg=carrier_collision_cfg,
        label="fingertip_carrier_collision_surface",
    )
    for shape_index in existing_shape_indices:
        builder.add_shape_collision_filter_pair(
            shape_index,
            carrier_collision_shape,
        )

    builder.color()
    model = builder.finalize(requires_grad=False)
    local_positions = _bonded_local_positions(
        builder.particle_q,
        global_indices,
    )

    return FingertipNewtonModel(
        fingertip_mesh=fingertip_mesh,
        model=model,
        carrier_body=carrier_body,
        silicone_particle_start=particle_start,
        silicone_particle_count=particle_count,
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
