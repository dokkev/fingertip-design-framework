"""Solver-neutral closed triangle meshes for rigid 3D objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_AREA_TOLERANCE_MM2 = 1.0e-12
_VOLUME_TOLERANCE_MM3 = 1.0e-12
_QUATERNION_NORM_TOLERANCE = 1.0e-12


def _finite_tuple(
    value: tuple[float, ...] | list[float],
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(component) for component in array)


def _normalize(
    value: tuple[float, ...],
    *,
    tolerance: float,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= tolerance:
        raise ValueError(f"{name} must have a finite nonzero norm")
    return tuple(float(component) for component in array / norm)


def _readonly_array(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RigidPose3D:
    """Solver-independent rigid pose in repository millimetres.

    The quaternion uses ``xyzw`` order.  This record belongs beside the
    neutral rigid-object mesh so contact registration does not depend on the
    mechanics package.
    """

    translation_mm: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        translation = _finite_tuple(
            self.translation_mm,
            length=3,
            name="translation_mm",
        )
        quaternion = _finite_tuple(
            self.quaternion_xyzw,
            length=4,
            name="quaternion_xyzw",
        )
        quaternion = _normalize(
            quaternion,
            tolerance=_QUATERNION_NORM_TOLERANCE,
            name="quaternion_xyzw",
        )
        object.__setattr__(self, "translation_mm", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)


def _validate_closed_triangle_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.shape[0] < 4 or faces.shape[0] < 4:
        raise ValueError("rigid object mesh must contain at least four vertices and faces")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("rigid object faces contain an out-of-range vertex index")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    ):
        raise ValueError("rigid object faces must contain three distinct vertex indices")

    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    cross = np.cross(second - first, third - first)
    area_twice = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(area_twice)) or np.any(area_twice <= _AREA_TOLERANCE_MM2):
        raise ValueError("rigid object faces must have nonzero finite area")

    edge_directions: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (min(start, end), max(start, end))
            direction = 1 if (start, end) == edge else -1
            edge_directions.setdefault(edge, []).append(direction)
    for edge, directions in edge_directions.items():
        if len(directions) != 2 or sorted(directions) != [-1, 1]:
            raise ValueError(
                "rigid object mesh must be a closed manifold with consistently wound faces; "
                f"edge {edge!r} has directions {directions!r}"
            )

    signed_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                first,
                np.cross(second, third),
            )
        )
        / 6.0
    )
    scale = max(float(np.max(np.ptp(vertices, axis=0))), 1.0)
    volume_tolerance = _VOLUME_TOLERANCE_MM3 * scale**3
    if not np.isfinite(signed_volume) or signed_volume <= volume_tolerance:
        raise ValueError("rigid object mesh must have positive outward signed volume")


@dataclass(frozen=True)
class RigidObjectMesh:
    """Immutable millimetre triangle mesh for a closed rigid object.

    The mesh is intentionally independent of Newton, Warp, CUDA, OptiX, and
    mechanics solver state.  Positive signed volume defines the outward
    winding convention used by the repository-facing contract.
    """

    vertices_mm: np.ndarray
    faces: np.ndarray
    name: str = "rigid_object"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_vertices = np.asarray(self.vertices_mm)
        raw_faces = np.asarray(self.faces)
        if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3:
            raise ValueError("vertices_mm must have shape (N, 3)")
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("faces must contain integer triangle indices")
        vertices = np.asarray(raw_vertices, dtype=np.float64)
        faces = np.asarray(raw_faces, dtype=np.int64)
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_mm must contain only finite values")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata = dict(self.metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("metadata keys must be non-empty strings")
        _validate_closed_triangle_mesh(vertices, faces)
        object.__setattr__(self, "vertices_mm", _readonly_array(vertices, dtype=np.float64))
        object.__setattr__(self, "faces", _readonly_array(faces, dtype=np.int64))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def bounds_mm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return componentwise minimum and maximum bounds in millimetres."""

        return (
            tuple(float(value) for value in np.min(self.vertices_mm, axis=0)),
            tuple(float(value) for value in np.max(self.vertices_mm, axis=0)),
        )


def _orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    signed_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                vertices[faces[:, 0]],
                np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
            )
        )
        / 6.0
    )
    if signed_volume < 0.0:
        faces = faces[:, [0, 2, 1]]
    return faces


def make_sphere_mesh(radius_mm: float, subdivisions: int = 2) -> RigidObjectMesh:
    """Create a deterministic outward-wound icosphere centered at the origin."""

    radius = float(radius_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    if isinstance(subdivisions, bool) or int(subdivisions) != subdivisions or subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    subdivisions = int(subdivisions)

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    faces = _orient_faces_outward(vertices, faces)

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        vertex_list = vertices.tolist()

        def midpoint(first: int, second: int) -> int:
            key = min(first, second), max(first, second)
            if key not in midpoint_cache:
                point = (vertices[first] + vertices[second]) * 0.5
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(vertex_list)
                vertex_list.append(point.tolist())
            return midpoint_cache[key]

        refined: list[tuple[int, int, int]] = []
        for first, second, third in faces:
            first_second = midpoint(int(first), int(second))
            second_third = midpoint(int(second), int(third))
            third_first = midpoint(int(third), int(first))
            refined.extend(
                (
                    (int(first), first_second, third_first),
                    (int(second), second_third, first_second),
                    (int(third), third_first, second_third),
                    (first_second, second_third, third_first),
                )
            )
        vertices = np.asarray(vertex_list, dtype=np.float64)
        faces = np.asarray(refined, dtype=np.int64)

    vertices *= radius
    return RigidObjectMesh(
        vertices_mm=vertices,
        faces=faces,
        name=f"sphere_r{radius:g}_sub{subdivisions}",
        metadata={"primitive": "sphere", "radius_mm": radius, "subdivisions": subdivisions},
    )


def make_cylinder_mesh(
    radius_mm: float,
    height_mm: float,
    radial_segments: int = 24,
) -> RigidObjectMesh:
    """Create a closed cylinder centered at the origin with its axis on +z."""

    radius = float(radius_mm)
    height = float(height_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("height_mm must be finite and positive")
    if isinstance(radial_segments, bool) or int(radial_segments) != radial_segments or radial_segments < 3:
        raise ValueError("radial_segments must be an integer of at least three")
    radial_segments = int(radial_segments)

    angles = 2.0 * np.pi * np.arange(radial_segments, dtype=np.float64) / radial_segments
    cosines = np.cos(angles)
    sines = np.sin(angles)
    z_bottom = -0.5 * height
    z_top = 0.5 * height
    vertices = [
        (radius * float(cosines[index]), radius * float(sines[index]), z_bottom)
        for index in range(radial_segments)
    ]
    vertices.extend(
        (radius * float(cosines[index]), radius * float(sines[index]), z_top)
        for index in range(radial_segments)
    )
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, z_bottom))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, z_top))

    faces: list[tuple[int, int, int]] = []
    for index in range(radial_segments):
        next_index = (index + 1) % radial_segments
        top_index = radial_segments + index
        next_top_index = radial_segments + next_index
        faces.extend(
            (
                (index, next_index, next_top_index),
                (index, next_top_index, top_index),
                (bottom_center, next_index, index),
                (top_center, top_index, next_top_index),
            )
        )

    return RigidObjectMesh(
        vertices_mm=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name=f"cylinder_r{radius:g}_h{height:g}_n{radial_segments}",
        metadata={
            "primitive": "cylinder",
            "radius_mm": radius,
            "height_mm": height,
            "radial_segments": radial_segments,
        },
    )


def make_box_mesh(size_x_mm: float, size_y_mm: float, size_z_mm: float) -> RigidObjectMesh:
    """Create a closed outward-wound box centered at the origin."""

    sizes = np.asarray((size_x_mm, size_y_mm, size_z_mm), dtype=float)
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("box dimensions must be finite and positive")
    half_x, half_y, half_z = (float(value) * 0.5 for value in sizes)
    vertices = np.asarray(
        [
            (-half_x, -half_y, -half_z),
            (half_x, -half_y, -half_z),
            (half_x, half_y, -half_z),
            (-half_x, half_y, -half_z),
            (-half_x, -half_y, half_z),
            (half_x, -half_y, half_z),
            (half_x, half_y, half_z),
            (-half_x, half_y, half_z),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 6, 2),
            (3, 7, 6),
            (1, 2, 6),
            (1, 6, 5),
            (0, 4, 7),
            (0, 7, 3),
        ],
        dtype=np.int64,
    )
    return RigidObjectMesh(
        vertices_mm=vertices,
        faces=faces,
        name=f"box_{sizes[0]:g}x{sizes[1]:g}x{sizes[2]:g}",
        metadata={"primitive": "box", "size_mm": tuple(float(value) for value in sizes)},
    )


def make_cube_mesh(size_mm: float) -> RigidObjectMesh:
    """Create a cube by calling the generic box generator."""

    size = float(size_mm)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("size_mm must be finite and positive")
    return make_box_mesh(size, size, size)


__all__ = [
    "RigidPose3D",
    "RigidObjectMesh",
    "make_box_mesh",
    "make_cube_mesh",
    "make_cylinder_mesh",
    "make_sphere_mesh",
]
