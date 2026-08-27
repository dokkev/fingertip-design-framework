"""Generate a 3D silicone mesh.

Coordinate convention of the returned Newton mesh:

    X: cross-section lateral direction
    Y: fingertip longitudinal / extrusion direction
    Z: contact-normal direction

Gmsh uses a temporary XY-profile/Z-extrusion frame internally. The
conversion to the canonical LUMO frame happens before the mesh is returned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lumo.fingertip.bonding_interface import BondingInterface
from lumo.fingertip.fingertip import Silicone

if TYPE_CHECKING:
    import newton


_MM_TO_M = 1.0e-3
_BOND_VERTEX_TOLERANCE_MM = 1.0e-5


def _to_lumo_frame(vertices: np.ndarray) -> np.ndarray:
    """Convert Gmsh coordinates to the canonical LUMO frame.

    Gmsh constructs the profile in the XY plane and extrudes along Z.
    The outer fingertip surface lies toward -Z in the LUMO frame, so
    indentation proceeds approximately in +Z.
    """
    rotated = np.empty_like(vertices)
    rotated[:, 0] = vertices[:, 0]
    rotated[:, 1] = -vertices[:, 2]
    rotated[:, 2] = vertices[:, 1]
    return rotated


def _find_bonded_vertex_indices(
    bonding_interface: BondingInterface,
    vertices_mm: np.ndarray,
    *,
    y_bounds_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Find silicone vertices on the analytic carrier-bond interfaces."""
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise ValueError("silicone vertices must have shape (N, 3)")

    points = vertices_mm[:, (0, 2)]
    bonded_indices: list[np.ndarray] = []

    for boundary in (bonding_interface.left, bonding_interface.right):
        distances_squared = np.full(points.shape[0], np.inf)

        for start, end in zip(
            boundary[:-1],
            boundary[1:],
            strict=True,
        ):
            delta = np.asarray(end, dtype=np.float64) - start
            relative = points - np.asarray(start, dtype=np.float64)
            length_squared = float(np.dot(delta, delta))

            if length_squared == 0.0:
                segment_distances_squared = np.sum(relative * relative, axis=1)
            else:
                parameters = np.sum(relative * delta, axis=1) / length_squared
                parameters = np.clip(parameters, 0.0, 1.0)
                closest = np.asarray(start) + parameters[:, None] * delta
                difference = points - closest
                segment_distances_squared = np.sum(
                    difference * difference,
                    axis=1,
                )

            distances_squared = np.minimum(
                distances_squared,
                segment_distances_squared,
            )

        matches = np.flatnonzero(
            distances_squared <= _BOND_VERTEX_TOLERANCE_MM**2
        )
        if y_bounds_mm is not None:
            lower_y_mm, upper_y_mm = y_bounds_mm
            matches = matches[
                (vertices_mm[matches, 1] >= lower_y_mm - _BOND_VERTEX_TOLERANCE_MM)
                & (
                    vertices_mm[matches, 1]
                    <= upper_y_mm + _BOND_VERTEX_TOLERANCE_MM
                )
            ]
        if matches.size == 0:
            raise RuntimeError(
                "silicone mesh contains no vertices on a carrier-bond "
                "interface"
            )
        bonded_indices.append(matches)

    return np.unique(np.concatenate(bonded_indices)).astype(np.int32)


def _make_silicone_mesh(
    silicone: Silicone,
    bonding_interface: BondingInterface,
    *,
    active_y_bounds_mm: tuple[float, float],
    distal_end_cap_length_mm: float,
    element_size_mm: float = 1.0,
) -> tuple["newton.TetMesh", np.ndarray]:
    """Mesh the active body and solid distal end-cap as one volume."""
    if not isinstance(silicone, Silicone):
        raise TypeError("silicone must be a Silicone geometry")
    if not isinstance(bonding_interface, BondingInterface):
        raise TypeError("bonding_interface must be a BondingInterface")
    lower_y_mm, upper_y_mm = active_y_bounds_mm
    if not lower_y_mm < upper_y_mm:
        raise ValueError("active_y_bounds_mm must be strictly increasing")
    if distal_end_cap_length_mm <= 0.0:
        raise ValueError("distal_end_cap_length_mm must be positive")

    try:
        import gmsh
        import newton
    except ImportError as exc:
        raise RuntimeError(
            "silicone meshing requires gmsh and newton"
        ) from exc

    # Gmsh extrusion Z maps to negative LUMO Y. The active section therefore
    # starts at its distal +Y face and extrudes toward the proximal -Y face;
    # the solid closure extrudes in the opposite direction from that interface.
    distal_gmsh_z_mm = -upper_y_mm
    active_depth_mm = upper_y_mm - lower_y_mm

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lumo_silicone")

        active_surface = _build_cross_section(
            gmsh,
            silicone,
            z_mm=distal_gmsh_z_mm,
        )
        active_extrusion = gmsh.model.occ.extrude(
            [(2, active_surface)],
            0.0,
            0.0,
            active_depth_mm,
        )
        active_volumes = [
            tag for dimension, tag in active_extrusion if dimension == 3
        ]
        if len(active_volumes) != 1:
            raise RuntimeError("active silicone must create one volume")

        end_cap_surface = _build_solid_end_cap_cross_section(
            gmsh,
            silicone,
            z_mm=distal_gmsh_z_mm,
        )
        end_cap_extrusion = gmsh.model.occ.extrude(
            [(2, end_cap_surface)],
            0.0,
            0.0,
            -distal_end_cap_length_mm,
        )
        end_cap_volumes = [
            tag for dimension, tag in end_cap_extrusion if dimension == 3
        ]
        if len(end_cap_volumes) != 1:
            raise RuntimeError("distal silicone end-cap must create one volume")

        fused, _ = gmsh.model.occ.fuse(
            [(3, active_volumes[0])],
            [(3, end_cap_volumes[0])],
        )
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        volume_tags = [tag for dimension, tag in fused if dimension == 3]
        if len(volume_tags) != 1:
            raise RuntimeError(
                "active silicone and distal end-cap must fuse into one volume"
            )

        vertices_mm, tetrahedra, bonded_vertex_indices = _mesh_volume(
            gmsh,
            volume_tags[0],
            bonding_interface,
            element_size_mm=element_size_mm,
            bonded_y_bounds_mm=active_y_bounds_mm,
        )
        dorsal_bond_indices = np.flatnonzero(
            (
                vertices_mm[:, 1]
                >= upper_y_mm - _BOND_VERTEX_TOLERANCE_MM
            )
            & (
                vertices_mm[:, 1]
                <= upper_y_mm
                + distal_end_cap_length_mm
                + _BOND_VERTEX_TOLERANCE_MM
            )
            & (
                np.abs(vertices_mm[:, 2] - silicone.bond_top_z_mm)
                <= _BOND_VERTEX_TOLERANCE_MM
            )
        )
        if dorsal_bond_indices.size == 0:
            raise RuntimeError(
                "distal end-cap contains no dorsal bond vertices"
            )
        bonded_vertex_indices = np.unique(
            np.concatenate((bonded_vertex_indices, dorsal_bond_indices))
        ).astype(np.int32)
    finally:
        gmsh.finalize()

    return (
        newton.TetMesh(
            vertices=vertices_mm * _MM_TO_M,
            tet_indices=tetrahedra.reshape(-1),
        ),
        bonded_vertex_indices,
    )


def _mesh_volume(
    gmsh,
    volume_tag: int,
    bonding_interface: BondingInterface,
    *,
    element_size_mm: float,
    bonded_y_bounds_mm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one TET4 volume and recover its kinematic bond vertices."""
    gmsh.option.setNumber("Mesh.MeshSizeMin", element_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", element_size_mm)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)

    vertices_mm, node_index = _extract_vertices(gmsh)
    tetrahedra = _extract_tetrahedra(gmsh, volume_tag, node_index)
    vertices_mm = _to_lumo_frame(vertices_mm)
    bonded_vertex_indices = _find_bonded_vertex_indices(
        bonding_interface,
        vertices_mm,
        y_bounds_mm=bonded_y_bounds_mm,
    )
    return vertices_mm, tetrahedra, bonded_vertex_indices


def _build_cross_section(
    gmsh,
    silicone: Silicone,
    *,
    z_mm: float,
) -> int:
    """Construct the analytic 2D silicone section in Gmsh OCC."""

    occ = gmsh.model.occ
    void_left = silicone.void_left
    void_right = silicone.void_right
    void_bottom = silicone.void_bottom
    cutout_left_x = void_left[0][0]
    cutout_right_x = void_right[0][0]
    cutout_bottom_y = void_bottom[0][1]

    pad = _build_outer_cross_section(gmsh, silicone, z_mm=z_mm)

    # Remove the internal stem clearance.
    cutout = occ.addRectangle(
        cutout_left_x,
        cutout_bottom_y,
        z_mm,
        cutout_right_x - cutout_left_x,
        -cutout_bottom_y,
    )

    pad, _ = occ.cut(
        [(2, pad)],
        [(2, cutout)],
    )

    surfaces = [
        tag
        for dimension, tag in pad
        if dimension == 2
    ]

    if len(surfaces) != 1:
        raise RuntimeError(
            "silicone cross-section must form one connected surface"
        )

    return surfaces[0]


def _build_outer_cross_section(
    gmsh,
    silicone: Silicone,
    *,
    z_mm: float,
) -> int:
    """Construct the silicone outer profile without the stem/void cutout."""
    occ = gmsh.model.occ
    half_width = silicone.half_width_mm
    ellipse_center_z = silicone.ellipse_center_z_mm
    horizontal_axis, vertical_axis = silicone.semiellipse_axes_mm

    # Flat compliant region.
    flat = occ.addRectangle(
        -half_width,
        ellipse_center_z,
        z_mm,
        2.0 * half_width,
        -ellipse_center_z,
    )

    # Exact elliptical disk.
    ellipse = _add_ellipse_disk(
        occ,
        x=0.0,
        y=ellipse_center_z,
        z=z_mm,
        radius_x=horizontal_axis,
        radius_y=vertical_axis,
    )

    # Retain only the lower half.
    lower_clip = occ.addRectangle(
        -half_width,
        ellipse_center_z - vertical_axis,
        z_mm,
        2.0 * half_width,
        vertical_axis,
    )

    lower_ellipse, _ = occ.intersect(
        [(2, ellipse)],
        [(2, lower_clip)],
    )

    # Compliant bond extensions.
    left_extension = silicone.bond_extension_left
    left_bond = occ.addRectangle(
        left_extension[0][0],
        left_extension[0][1],
        z_mm,
        left_extension[1][0] - left_extension[0][0],
        left_extension[2][1] - left_extension[1][1],
    )

    right_extension = silicone.bond_extension_right
    right_bond = occ.addRectangle(
        right_extension[0][0],
        right_extension[0][1],
        z_mm,
        right_extension[1][0] - right_extension[0][0],
        right_extension[2][1] - right_extension[1][1],
    )

    pad, _ = occ.fuse(
        [(2, flat), *lower_ellipse],
        [(2, left_bond), (2, right_bond)],
    )

    surfaces = [
        tag
        for dimension, tag in pad
        if dimension == 2
    ]

    if len(surfaces) != 1:
        raise RuntimeError(
            "silicone outer cross-section must form one connected surface"
        )

    return surfaces[0]


def _build_solid_end_cap_cross_section(
    gmsh,
    silicone: Silicone,
    *,
    z_mm: float,
) -> int:
    """Construct the distal silicone fill below the dorsal carrier plate."""
    occ = gmsh.model.occ
    outer = _build_outer_cross_section(gmsh, silicone, z_mm=z_mm)
    upper_fill = occ.addRectangle(
        -silicone.half_width_mm,
        0.0,
        z_mm,
        2.0 * silicone.half_width_mm,
        silicone.bond_top_z_mm,
    )
    fused, _ = occ.fuse(
        [(2, outer)],
        [(2, upper_fill)],
    )
    surfaces = [tag for dimension, tag in fused if dimension == 2]
    if len(surfaces) != 1:
        raise RuntimeError(
            "solid end-cap cross-section must form one surface"
        )
    return surfaces[0]


def _add_ellipse_disk(
    occ,
    *,
    x: float,
    y: float,
    z: float,
    radius_x: float,
    radius_y: float,
) -> int:
    """Create an axis-aligned elliptical disk with either aspect ratio."""
    if radius_x >= radius_y:
        return occ.addDisk(x, y, z, radius_x, radius_y)

    return occ.addDisk(
        x,
        y,
        z,
        radius_y,
        radius_x,
        zAxis=[0.0, 0.0, 1.0],
        xAxis=[0.0, 1.0, 0.0],
    )


def _extract_vertices(
    gmsh,
) -> tuple[np.ndarray, dict[int, int]]:
    """Extract Gmsh nodes and map node tags to local indices."""

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()

    vertices = np.asarray(
        coordinates,
        dtype=np.float32,
    ).reshape(-1, 3)

    node_index = {
        int(tag): index
        for index, tag in enumerate(node_tags)
    }

    return vertices, node_index


def _extract_tetrahedra(
    gmsh,
    volume_tag: int,
    node_index: dict[int, int],
) -> np.ndarray:
    """Extract first-order tetrahedra with zero-based local indices."""

    element_types, _, connectivity_groups = (
        gmsh.model.mesh.getElements(3, volume_tag)
    )

    tetrahedra = []

    for element_type, connectivity in zip(
        element_types,
        connectivity_groups,
        strict=True,
    ):
        (
            name,
            dimension,
            order,
            node_count,
            _,
            _,
        ) = gmsh.model.mesh.getElementProperties(element_type)

        if dimension != 3:
            continue

        if order != 1 or node_count != 4:
            raise RuntimeError(
                "silicone mesh must contain only TET4 elements, "
                f"got {name}"
            )

        connectivity = np.asarray(
            connectivity,
            dtype=np.int64,
        ).reshape(-1, 4)

        local_indices = np.asarray(
            [
                [
                    node_index[int(node_tag)]
                    for node_tag in tetrahedron
                ]
                for tetrahedron in connectivity
            ],
            dtype=np.int32,
        )

        tetrahedra.append(local_indices)

    if not tetrahedra:
        raise RuntimeError(
            "Gmsh generated no tetrahedral elements"
        )

    return np.concatenate(tetrahedra, axis=0)


__all__ = []
