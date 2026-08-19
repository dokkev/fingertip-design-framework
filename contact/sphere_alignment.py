"""Canonical sphere alignment from the authoritative fingertip crown frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechanics3d.indentation import RigidPose3D
from mesh.indenter import crown_frame_from_model
from mesh.rigid_object import RigidObjectMesh
from model.fingertip_model import FingertipModel


_IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)


def _sphere_radius_mm(sphere_mesh: RigidObjectMesh) -> float:
    if sphere_mesh.metadata.get("primitive") != "sphere":
        raise ValueError("sphere alignment requires a mesh from make_sphere_mesh")
    radius = float(sphere_mesh.metadata.get("radius_mm", float("nan")))
    norms = np.linalg.norm(sphere_mesh.vertices_mm, axis=1)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or not np.allclose(norms, radius, rtol=0.0, atol=1.0e-10)
    ):
        raise ValueError("sphere mesh must have a finite spherical radius")
    return radius


@dataclass(frozen=True)
class SphereAlignment:
    """Deterministic target frame and collision-free nominal sphere pose."""

    target_point_mm: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    approach_direction: tuple[float, float, float]
    nominal_pose: RigidPose3D
    radius_mm: float
    initial_gap_mm: float


def canonical_sphere_alignment(
    model: FingertipModel,
    sphere_mesh: RigidObjectMesh,
    *,
    initial_gap_mm: float = 0.25,
) -> SphereAlignment:
    """Align one sphere with the authoritative central compliant arc crown."""

    if not isinstance(model, FingertipModel):
        raise TypeError("model must be FingertipModel")
    if not isinstance(sphere_mesh, RigidObjectMesh):
        raise TypeError("sphere_mesh must be a RigidObjectMesh")
    gap = float(initial_gap_mm)
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError("initial_gap_mm must be finite and positive")

    radius = _sphere_radius_mm(sphere_mesh)
    frame = crown_frame_from_model(model)
    target = np.asarray((frame.point_mm[0], frame.point_mm[1], 0.0), dtype=float)
    normal = np.asarray(
        (frame.pad_outward_normal[0], frame.pad_outward_normal[1], 0.0),
        dtype=float,
    )
    normal /= np.linalg.norm(normal)
    approach = -normal
    center = target + (radius + gap) * normal

    if not np.all(np.isfinite(center)):
        raise ValueError("canonical sphere alignment produced non-finite coordinates")
    return SphereAlignment(
        target_point_mm=tuple(float(value) for value in target),
        outward_normal=tuple(float(value) for value in normal),
        approach_direction=tuple(float(value) for value in approach),
        nominal_pose=RigidPose3D(
            translation_mm=tuple(float(value) for value in center),
            quaternion_xyzw=_IDENTITY_QUATERNION,
        ),
        radius_mm=radius,
        initial_gap_mm=gap,
    )


__all__ = ["SphereAlignment", "canonical_sphere_alignment"]
