"""Neutral 11 mm periodic geometry for deterministic 3D transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from lumo.finger.fingertip import Fingertip
from lumo.ray_tracing.contracts.objects import ObjectBoundaryOptics


class Transport3DGeometryError(ValueError):
    """Raised when a periodic transport scene cannot be built safely."""


Full3DSurfaceProvenance = Literal[
    "actual_reference_3d_volume_state",
    "actual_deformed_3d_fea_surface",
    "actual_deformed_3d_vbd_surface",
    "actual_deformed_3d_volume_state",
]


AIR_INTERFACE = "AIR_INTERFACE"
OBJECT_CONTACT_INTERFACE = "OBJECT_CONTACT_INTERFACE"
CARRIER_CONTACT_INTERFACE = "CARRIER_CONTACT_INTERFACE"
INTERNAL_INTERFACE = "INTERNAL_INTERFACE"


@dataclass(frozen=True)
class TriangleSurface:
    """One OptiX-ready triangle set and its semantic metadata."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    boundary_edge_indices: np.ndarray | None = None
    external_surface: np.ndarray | None = None
    u_start: np.ndarray | None = None
    u_end: np.ndarray | None = None
    semantic_tags: tuple[str, ...] | None = None
    interface_tags: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, dtype=np.float32, copy=True)
        faces = np.array(self.faces, dtype=np.uint32, copy=True)
        normals = np.array(self.normals, dtype=np.float32, copy=True)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            raise Transport3DGeometryError("surface vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise Transport3DGeometryError("surface faces must have shape (F, 3)")
        if np.any(faces >= len(vertices)):
            raise Transport3DGeometryError("surface face index is out of range")
        if normals.shape != faces.shape:
            raise Transport3DGeometryError("surface normals must have shape (F, 3)")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
            raise Transport3DGeometryError("surface geometry must be finite")
        if np.any(np.linalg.norm(normals, axis=1) <= 0.0):
            raise Transport3DGeometryError("surface normals must be nonzero")
        geometric_cross = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        geometric_lengths = np.linalg.norm(geometric_cross, axis=1)
        if np.any(geometric_lengths <= 1.0e-12):
            raise Transport3DGeometryError("surface contains a degenerate triangle")
        geometric_normals = geometric_cross / geometric_lengths[:, None]
        supplied_lengths = np.linalg.norm(normals, axis=1)
        orientation_alignment = np.sum(
            geometric_normals * normals / supplied_lengths[:, None],
            axis=1,
        )
        if np.any(orientation_alignment <= 1.0 - 1.0e-5):
            raise Transport3DGeometryError(
                "surface triangle normals are not consistently oriented"
            )
        edge_directions: dict[tuple[int, int], list[int]] = {}
        for triangle in faces:
            for first, second in (
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[1]), int(triangle[2])),
                (int(triangle[2]), int(triangle[0])),
            ):
                key = min(first, second), max(first, second)
                edge_directions.setdefault(key, []).append(
                    1 if (first, second) == key else -1
                )
        if any(
            len(directions) > 2 or (len(directions) == 2 and sum(directions) != 0)
            for directions in edge_directions.values()
        ):
            raise Transport3DGeometryError(
                "surface triangle orientation is inconsistent across shared edges"
            )
        arrays = [vertices, faces, normals]
        optional = []
        for name, value, dtype in (
            ("boundary_edge_indices", self.boundary_edge_indices, np.int64),
            ("external_surface", self.external_surface, bool),
            ("u_start", self.u_start, float),
            ("u_end", self.u_end, float),
        ):
            if value is None:
                optional.append((name, None))
                continue
            array = np.array(value, dtype=dtype, copy=True)
            if array.ndim != 1 or len(array) != len(faces):
                raise Transport3DGeometryError(
                    f"{name} must have one value per triangle"
                )
            if dtype is not bool and not np.all(np.isfinite(array)):
                raise Transport3DGeometryError(f"{name} must be finite")
            array.setflags(write=False)
            optional.append((name, array))
        semantic_tags = None
        if self.semantic_tags is not None:
            semantic_tags = tuple(str(tag) for tag in self.semantic_tags)
            if len(semantic_tags) != len(faces):
                raise Transport3DGeometryError(
                    "semantic_tags must have one value per triangle"
                )
        interface_tags = None
        if self.interface_tags is not None:
            interface_tags = tuple(str(tag) for tag in self.interface_tags)
            if len(interface_tags) != len(faces):
                raise Transport3DGeometryError(
                    "interface_tags must have one value per triangle"
                )
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "normals", normals)
        for name, value in optional:
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_tags", semantic_tags)
        object.__setattr__(self, "interface_tags", interface_tags)


@dataclass(frozen=True)
class TransportGeometry:
    """All neutral surfaces and explicit state facts for one FULL_3D scene."""

    silicone: TriangleSurface
    rigid: TriangleSurface
    envelope: TriangleSurface
    depth_mm: float
    z_min_mm: float
    z_max_mm: float
    source_position_mm: tuple[float, float, float]
    source_medium: int
    full3d_surface_provenance: Full3DSurfaceProvenance
    reference_periodic_z_planes_mm: tuple[float, float]
    deformed_surface_z_extent_mm: tuple[float, float]
    deformed_surface_exceeds_reference_z_planes: bool
    carrier_contact_active: bool
    carrier_mapping_tolerance_mm: float | None = None
    indenter_optics: ObjectBoundaryOptics | None = None
    carrier_optics: ObjectBoundaryOptics | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.depth_mm) or self.depth_mm != 11.0:
            raise Transport3DGeometryError("the transport cell depth must be exactly 11 mm")
        if self.z_min_mm != -5.5 or self.z_max_mm != 5.5:
            raise Transport3DGeometryError("the transport cell must span exactly +/-5.5 mm")
        source = tuple(float(value) for value in self.source_position_mm)
        if source[2] != 0.0 or not np.all(np.isfinite(source)):
            raise Transport3DGeometryError("the single source must be at z=0")
        if self.source_medium not in (0, 1):
            raise Transport3DGeometryError("source_medium must be air=0 or silicone=1")
        if self.full3d_surface_provenance not in {
            "actual_reference_3d_volume_state",
            "actual_deformed_3d_fea_surface",
            "actual_deformed_3d_vbd_surface",
            "actual_deformed_3d_volume_state",
        }:
            raise Transport3DGeometryError("invalid full 3D surface provenance")
        planes = tuple(float(value) for value in self.reference_periodic_z_planes_mm)
        extent = tuple(float(value) for value in self.deformed_surface_z_extent_mm)
        if len(planes) != 2 or len(extent) != 2:
            raise Transport3DGeometryError("z planes and extent must contain two values")
        if not np.all(np.isfinite(planes + extent)):
            raise Transport3DGeometryError("z planes and extent must be finite")
        if planes[0] >= planes[1] or extent[0] > extent[1]:
            raise Transport3DGeometryError("z planes and extent must be ordered")
        object.__setattr__(self, "reference_periodic_z_planes_mm", planes)
        object.__setattr__(self, "deformed_surface_z_extent_mm", extent)
        if self.carrier_mapping_tolerance_mm is not None:
            tolerance = float(self.carrier_mapping_tolerance_mm)
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise Transport3DGeometryError(
                    "carrier_mapping_tolerance_mm must be finite and non-negative"
                )
            object.__setattr__(self, "carrier_mapping_tolerance_mm", tolerance)
        object.__setattr__(self, "source_position_mm", source)


def _surface_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    normals = np.cross(second - first, third - first)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise Transport3DGeometryError("triangle normal is degenerate")
    return normals / lengths[:, None]


def build_full3d_transport_geometry(
    tip: Fingertip,
    *,
    silicone: TriangleSurface,
    rigid: TriangleSurface,
    envelope: TriangleSurface,
    source_position_mm: tuple[float, float, float],
    source_medium: int,
    full3d_surface_provenance: Full3DSurfaceProvenance,
    carrier_optics: ObjectBoundaryOptics | None = None,
    carrier_mapping_tolerance_mm: float | None = None,
    depth_mm: float = 11.0,
) -> TransportGeometry:
    """Build transport geometry from a direct deformed 3D surface artifact.

    This constructor intentionally accepts direct 3D triangles. The distinction
    is the provenance guard against accidentally labelling an extrusion of 2D
    deformation as FULL_3D. The
    provenance guard accepts only direct FEA or direct VBD surface states.
    Full 3D field accumulation is performed from retained native path
    segments; no collapsed 2D optical domain is accepted here.
    """
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    if full3d_surface_provenance not in {
        "actual_reference_3d_volume_state",
        "actual_deformed_3d_fea_surface",
        "actual_deformed_3d_vbd_surface",
        "actual_deformed_3d_volume_state",
    }:
        raise Transport3DGeometryError(
            "full 3D surface provenance must identify a direct FEA or VBD "
            "deformed surface"
        )
    source = tuple(float(value) for value in source_position_mm)
    if len(source) != 3 or not np.all(np.isfinite(source)):
        raise Transport3DGeometryError("full 3D source position must be finite")
    if abs(source[2]) > 1.0e-9:
        raise Transport3DGeometryError("the representative full 3D source must be at z=0")
    if source_medium not in (0, 1):
        raise Transport3DGeometryError("source_medium must be air=0 or silicone=1")
    for name, surface in (
        ("silicone", silicone),
        ("rigid", rigid),
        ("envelope", envelope),
    ):
        if not isinstance(surface, TriangleSurface):
            raise TypeError(f"{name} must be a TriangleSurface")
        if name == "silicone" and surface.semantic_tags is None:
            raise Transport3DGeometryError(
                "full 3D silicone surface must preserve semantic surface tags"
            )
        if not np.all(np.isfinite(surface.vertices[:, 2])):
            raise Transport3DGeometryError(f"{name} surface has a non-finite longitudinal coordinate")
    surface_extent = (
        float(np.min(silicone.vertices[:, 2])),
        float(np.max(silicone.vertices[:, 2])),
    )
    exceeds_reference_planes = bool(
        np.min(silicone.vertices[:, 2]) < -depth_mm / 2.0 - 1.0e-9
        or np.max(silicone.vertices[:, 2]) > depth_mm / 2.0 + 1.0e-9
    )
    has_carrier_contact = any(
        tag == CARRIER_CONTACT_INTERFACE
        for tag in (silicone.interface_tags or ())
    )
    if has_carrier_contact and carrier_optics is None:
        raise Transport3DGeometryError(
            "carrier contact triangles require an explicit carrier optical boundary"
        )
    if has_carrier_contact and carrier_mapping_tolerance_mm is None:
        raise Transport3DGeometryError(
            "carrier contact triangles require an explicit mapping tolerance"
        )
    return TransportGeometry(
        silicone=silicone,
        rigid=rigid,
        envelope=envelope,
        depth_mm=depth_mm,
        z_min_mm=-depth_mm / 2.0,
        z_max_mm=depth_mm / 2.0,
        source_position_mm=source,
        source_medium=source_medium,
        full3d_surface_provenance=full3d_surface_provenance,
        reference_periodic_z_planes_mm=(-depth_mm / 2.0, depth_mm / 2.0),
        deformed_surface_z_extent_mm=surface_extent,
        deformed_surface_exceeds_reference_z_planes=exceeds_reference_planes,
        carrier_contact_active=has_carrier_contact,
        carrier_mapping_tolerance_mm=carrier_mapping_tolerance_mm,
        carrier_optics=carrier_optics,
    )


__all__ = [
    "AIR_INTERFACE",
    "CARRIER_CONTACT_INTERFACE",
    "TransportGeometry",
    "Full3DSurfaceProvenance",
    "INTERNAL_INTERFACE",
    "OBJECT_CONTACT_INTERFACE",
    "TriangleSurface",
    "Transport3DGeometryError",
    "build_full3d_transport_geometry",
]
