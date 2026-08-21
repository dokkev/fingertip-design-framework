"""Generate a Newton tetrahedral mesh from a LUMO fingertip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lumo.fingertip.fingertip import Fingertip

if TYPE_CHECKING:
    import newton


_MM_TO_M = 1.0e-3


def make_fingertip_mesh(
    fingertip: Fingertip,
    *,
    extrusion_depth_mm: float = 11.0,
    element_size_mm: float = 1.0,
) -> "newton.TetMesh":
    """Extrude the fingertip cross-section and generate a Newton TetMesh."""

    if extrusion_depth_mm <= 0.0:
        raise ValueError("extrusion_depth_mm must be positive")

    if element_size_mm <= 0.0:
        raise ValueError("element_size_mm must be positive")

    try:
        import gmsh
        import newton
    except ImportError as exc:
        raise RuntimeError(
            "fingertip meshing requires gmsh and newton"
        ) from exc

    gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lumo_fingertip")

        surface_tag = _build_cross_section(
            gmsh,
            fingertip,
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
                "fingertip extrusion must create exactly one volume"
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

    finally:
        gmsh.finalize()

    # Fingertip/Gmsh geometry is expressed in mm.
    # Newton TetMesh uses SI units.
    vertices_m = vertices_mm * _MM_TO_M

    return newton.TetMesh(
        vertices=vertices_m,
        tet_indices=tetrahedra.reshape(-1),
    )


def _build_cross_section(
    gmsh,
    fingertip: Fingertip,
    *,
    z_mm: float,
) -> int:
    """Construct the compliant 2D fingertip section in Gmsh OCC."""

    geometry = fingertip.parameters.geometry
    occ = gmsh.model.occ

    half_width = fingertip.half_width_mm
    ellipse_center_y = fingertip.ellipse_center_y_mm
    cutout_half_width = fingertip.cutout_half_width_mm

    cutout_bottom_y = -(
        geometry.stem_height_mm
        + geometry.void_height_mm
    )

    # Flat compliant region.
    flat = occ.addRectangle(
        -half_width,
        -geometry.flat_pad_height_mm,
        z_mm,
        2.0 * half_width,
        geometry.flat_pad_height_mm,
    )

    # Exact elliptical disk.
    ellipse = _add_ellipse_disk(
        occ,
        x=0.0,
        y=ellipse_center_y,
        z=z_mm,
        radius_x=half_width,
        radius_y=geometry.semiellipse_height_mm,
    )

    # Retain only the lower half.
    lower_clip = occ.addRectangle(
        -half_width,
        ellipse_center_y - geometry.semiellipse_height_mm,
        z_mm,
        2.0 * half_width,
        geometry.semiellipse_height_mm,
    )

    lower_ellipse, _ = occ.intersect(
        [(2, ellipse)],
        [(2, lower_clip)],
    )

    # Compliant bond extensions.
    left_bond = occ.addRectangle(
        -half_width,
        0.0,
        z_mm,
        geometry.bond_extension_width_mm,
        geometry.bond_extension_height_mm,
    )

    right_bond = occ.addRectangle(
        half_width - geometry.bond_extension_width_mm,
        0.0,
        z_mm,
        geometry.bond_extension_width_mm,
        geometry.bond_extension_height_mm,
    )

    pad, _ = occ.fuse(
        [(2, flat), *lower_ellipse],
        [(2, left_bond), (2, right_bond)],
    )

    # Remove the internal stem clearance.
    cutout = occ.addRectangle(
        -cutout_half_width,
        cutout_bottom_y,
        z_mm,
        2.0 * cutout_half_width,
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
            "fingertip cross-section must form one connected surface"
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
                "fingertip mesh must contain only TET4 elements, "
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


__all__ = ["make_fingertip_mesh"]
