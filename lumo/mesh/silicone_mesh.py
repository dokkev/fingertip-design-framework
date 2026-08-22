"""Generate a 3D silicone mesh.

Coordinate convention of the returned Newton mesh:

    X: cross-section lateral direction
    Y: fingertip width / extrusion direction
    Z: contact-normal direction

Gmsh uses a temporary XY-profile/Z-extrusion frame internally. The
conversion to the canonical LUMO frame happens before the mesh is returned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lumo.fingertip.fingertip import Silicone

if TYPE_CHECKING:
    import newton


_MM_TO_M = 1.0e-3


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


def _make_silicone_mesh(
    silicone: Silicone,
    *,
    extrusion_depth_mm: float = 11.0,
    element_size_mm: float = 1.0,
) -> "newton.TetMesh":
    """Extrude analytic silicone geometry into a Newton TetMesh."""
    if not isinstance(silicone, Silicone):
        raise TypeError("silicone must be a Silicone geometry")

    if extrusion_depth_mm <= 0.0:
        raise ValueError("extrusion_depth_mm must be positive")

    if element_size_mm <= 0.0:
        raise ValueError("element_size_mm must be positive")

    try:
        import gmsh
        import newton
    except ImportError as exc:
        raise RuntimeError(
            "silicone meshing requires gmsh and newton"
        ) from exc

    gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lumo_silicone")

        surface_tag = _build_cross_section(
            gmsh,
            silicone,
            z_mm=-0.5 * extrusion_depth_mm,
        )

        extrusion = gmsh.model.occ.extrude(
            [(2, surface_tag)],
            0.0,
            0.0,
            extrusion_depth_mm,
        )

        gmsh.model.occ.synchronize()

        volume_tags = [
            tag
            for dimension, tag in extrusion
            if dimension == 3
        ]

        if len(volume_tags) != 1:
            raise RuntimeError(
                "silicone extrusion must create exactly one volume"
            )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            element_size_mm,
        )
        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            element_size_mm,
        )
        gmsh.option.setNumber(
            "Mesh.ElementOrder",
            1,
        )

        gmsh.model.mesh.generate(3)

        vertices_mm, node_index = _extract_vertices(gmsh)
        tetrahedra = _extract_tetrahedra(
            gmsh,
            volume_tags[0],
            node_index,
        )
        vertices_mm = _to_lumo_frame(vertices_mm)

    finally:
        gmsh.finalize()

    # Silicone/Gmsh geometry is expressed in mm.
    # Newton TetMesh uses SI units.
    vertices_m = vertices_mm * _MM_TO_M

    return newton.TetMesh(
        vertices=vertices_m,
        tet_indices=tetrahedra.reshape(-1),
    )


def _build_cross_section(
    gmsh,
    silicone: Silicone,
    *,
    z_mm: float,
) -> int:
    """Construct the analytic 2D silicone section in Gmsh OCC."""

    occ = gmsh.model.occ

    half_width = silicone.half_width_mm
    ellipse_center_z = silicone.ellipse_center_z_mm
    horizontal_axis, vertical_axis = silicone.semiellipse_axes_mm
    void_left = silicone.void_left
    void_right = silicone.void_right
    void_bottom = silicone.void_bottom
    cutout_left_x = void_left[0][0]
    cutout_right_x = void_right[0][0]
    cutout_bottom_y = void_bottom[0][1]

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

    # Remove the internal stem clearance.
    cutout = occ.addRectangle(
        cutout_left_x,
        cutout_bottom_y,
        z_mm,
        cutout_right_x - cutout_left_x,
        -cutout_bottom_y,
    )

    pad, _ = occ.cut(
        pad,
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
