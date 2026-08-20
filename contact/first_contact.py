"""Neutral sphere first-contact search by clear/hit bracketing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from physics.indentation import RigidPose3D
from mesh.rigid_object import RigidObjectMesh
from model.solid import FingertipSolid


@dataclass(frozen=True)
class FingertipContactSurface:
    """Extruded compliant outer arc used by the sphere MVP predicate."""

    outer_compliant_arc: LineString
    z_min_mm: float
    z_max_mm: float

    def __post_init__(self) -> None:
        if not isinstance(self.outer_compliant_arc, LineString):
            raise TypeError("outer_compliant_arc must be a LineString")
        if self.outer_compliant_arc.is_empty or not self.outer_compliant_arc.is_valid:
            raise ValueError("outer_compliant_arc must be a valid non-empty line")
        z_min = float(self.z_min_mm)
        z_max = float(self.z_max_mm)
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
            raise ValueError("contact surface z bounds must be finite with min < max")
        object.__setattr__(self, "z_min_mm", z_min)
        object.__setattr__(self, "z_max_mm", z_max)


def make_outer_compliant_surface(solid: FingertipSolid) -> FingertipContactSurface:
    """Extract the authoritative 11 mm compliant outer arc from a solid."""

    if not isinstance(solid, FingertipSolid):
        raise TypeError("solid must be FingertipSolid")
    definition = next(
        (
            surface
            for surface in solid.surfaces
            if surface.name == "outer_compliant_arc"
        ),
        None,
    )
    if definition is None or not isinstance(definition.source_geometry, LineString):
        raise ValueError("FingertipSolid has no authoritative outer_compliant_arc")
    return FingertipContactSurface(
        outer_compliant_arc=definition.source_geometry,
        z_min_mm=solid.z_min_mm,
        z_max_mm=solid.z_max_mm,
    )


@dataclass(frozen=True)
class FirstContactSettings:
    """Deterministic geometric search and Newton spawn settings in millimetres."""

    coarse_step_mm: float
    tolerance_mm: float
    spawn_clearance_mm: float
    max_travel_mm: float

    def __post_init__(self) -> None:
        for name in (
            "coarse_step_mm",
            "tolerance_mm",
            "spawn_clearance_mm",
            "max_travel_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.coarse_step_mm <= self.tolerance_mm:
            raise ValueError("coarse_step_mm must be greater than tolerance_mm")


@dataclass(frozen=True)
class FirstContactResult:
    """Clear/hit certificate, refined first contact, and safe spawn pose."""

    clear_pose: RigidPose3D
    hit_pose: RigidPose3D
    contact_pose: RigidPose3D
    spawn_pose: RigidPose3D
    approach_direction: tuple[float, float, float]
    travel_to_contact_mm: float
    bracket_width_mm: float
    spawn_clearance_mm: float

    def __post_init__(self) -> None:
        direction = np.asarray(self.approach_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("approach_direction must be a finite unit vector")
        norm = float(np.linalg.norm(direction))
        if not math.isclose(
            norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError("approach_direction must be a finite unit vector")
        for name in (
            "travel_to_contact_mm",
            "bracket_width_mm",
            "spawn_clearance_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.bracket_width_mm <= 0.0:
            raise ValueError("bracket_width_mm must be positive")
        object.__setattr__(
            self,
            "approach_direction",
            tuple(float(value) for value in direction),
        )

    def pose_at_post_contact_travel(self, travel_mm: float) -> RigidPose3D:
        """Return ``T_first + travel_mm * approach_direction``."""

        travel = float(travel_mm)
        if not np.isfinite(travel) or travel < 0.0:
            raise ValueError("post-contact travel must be finite and non-negative")
        translation = np.asarray(self.contact_pose.translation_mm, dtype=float)
        translation += travel * np.asarray(self.approach_direction, dtype=float)
        return RigidPose3D(
            translation_mm=tuple(float(value) for value in translation),
            quaternion_xyzw=self.contact_pose.quaternion_xyzw,
        )


def _sphere_radius_mm(object_mesh: RigidObjectMesh) -> float:
    if object_mesh.metadata.get("primitive") != "sphere":
        raise ValueError("first-contact MVP requires a sphere mesh")
    radius = float(object_mesh.metadata.get("radius_mm", float("nan")))
    norms = np.linalg.norm(object_mesh.vertices_mm, axis=1)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or not np.allclose(norms, radius, rtol=0.0, atol=1.0e-10)
    ):
        raise ValueError("object mesh is not a valid centered sphere mesh")
    return radius


def intersects(
    fingertip_surface: FingertipContactSurface,
    object_mesh: RigidObjectMesh,
    object_pose: RigidPose3D,
) -> bool:
    """Return the geometry-only sphere/outer-arc intersection predicate.

    The sphere mesh supplies the validated radius. The compliant surface is
    the undeformed 2D outer arc extruded through the representative cell depth;
    no Newton contact margin, solver state, or contact count is consulted.
    """

    if not isinstance(fingertip_surface, FingertipContactSurface):
        raise TypeError("fingertip_surface must be FingertipContactSurface")
    if not isinstance(object_mesh, RigidObjectMesh):
        raise TypeError("object_mesh must be RigidObjectMesh")
    if not isinstance(object_pose, RigidPose3D):
        raise TypeError("object_pose must be RigidPose3D")

    radius = _sphere_radius_mm(object_mesh)
    center = np.asarray(object_pose.translation_mm, dtype=float)
    if center[2] + radius < fingertip_surface.z_min_mm:
        return False
    if center[2] - radius > fingertip_surface.z_max_mm:
        return False
    distance_mm = fingertip_surface.outer_compliant_arc.distance(
        Point(float(center[0]), float(center[1]))
    )
    return bool(distance_mm <= radius + 1.0e-12)


def _pose_at_travel(
    reference_pose: RigidPose3D,
    approach_direction: np.ndarray,
    travel_mm: float,
) -> RigidPose3D:
    translation = np.asarray(reference_pose.translation_mm, dtype=float)
    translation += float(travel_mm) * approach_direction
    return RigidPose3D(
        translation_mm=tuple(float(value) for value in translation),
        quaternion_xyzw=reference_pose.quaternion_xyzw,
    )


def find_first_contact(
    fingertip_surface: FingertipContactSurface,
    object_mesh: RigidObjectMesh,
    reference_pose: RigidPose3D,
    approach_direction: tuple[float, float, float],
    settings: FirstContactSettings,
) -> FirstContactResult:
    """Find a sphere's first contact by coarse stepping and bisection."""

    if not isinstance(settings, FirstContactSettings):
        raise TypeError("settings must be FirstContactSettings")
    direction = np.asarray(approach_direction, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("approach_direction must be a finite nonzero vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("approach_direction must be a finite nonzero vector")
    direction /= norm
    if intersects(fingertip_surface, object_mesh, reference_pose):
        raise ValueError("first-contact reference pose already intersects the fingertip")

    s_clear = 0.0
    s_hit = settings.coarse_step_mm
    while s_hit <= settings.max_travel_mm + 1.0e-12:
        candidate = _pose_at_travel(reference_pose, direction, s_hit)
        if intersects(fingertip_surface, object_mesh, candidate):
            break
        s_clear = s_hit
        s_hit += settings.coarse_step_mm
    else:
        raise RuntimeError(
            "sphere first-contact search exceeded max_travel_mm without a hit: "
            f"max_travel_mm={settings.max_travel_mm:g}"
        )

    while s_hit - s_clear > settings.tolerance_mm:
        midpoint = 0.5 * (s_clear + s_hit)
        candidate = _pose_at_travel(reference_pose, direction, midpoint)
        if intersects(fingertip_surface, object_mesh, candidate):
            s_hit = midpoint
        else:
            s_clear = midpoint

    s_first = 0.5 * (s_clear + s_hit)
    clear_pose = _pose_at_travel(reference_pose, direction, s_clear)
    hit_pose = _pose_at_travel(reference_pose, direction, s_hit)
    contact_pose = _pose_at_travel(reference_pose, direction, s_first)
    spawn_pose = _pose_at_travel(
        contact_pose,
        -direction,
        settings.spawn_clearance_mm,
    )
    if intersects(fingertip_surface, object_mesh, spawn_pose):
        raise RuntimeError(
            "spawn_clearance_mm is insufficient to produce a collision-free pose"
        )
    return FirstContactResult(
        clear_pose=clear_pose,
        hit_pose=hit_pose,
        contact_pose=contact_pose,
        spawn_pose=spawn_pose,
        approach_direction=tuple(float(value) for value in direction),
        travel_to_contact_mm=s_first,
        bracket_width_mm=s_hit - s_clear,
        spawn_clearance_mm=settings.spawn_clearance_mm,
    )


__all__ = [
    "FirstContactResult",
    "FirstContactSettings",
    "FingertipContactSurface",
    "find_first_contact",
    "intersects",
    "make_outer_compliant_surface",
]
