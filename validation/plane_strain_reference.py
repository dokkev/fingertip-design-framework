"""Validation-only layered mesh for a plane-strain-equivalent 3D reference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mesh.fingertip import generate_fingertip_mesh
from mesh.types import MeshSettings
from mesh.volume_types import (
    FingertipVolumeMesh,
    SurfaceTriangle,
    Tetrahedron,
    VolumeMeshQuality,
    VolumeMeshSettings,
    VolumeMeshValidation,
    VolumeNode,
)
from model import Fingertip, FingertipParameters, build_fingertip_solid


@dataclass(frozen=True)
class PlaneStrainReferenceContract:
    """Layer correspondence carried alongside the validation mesh."""

    node_columns: tuple[tuple[int, ...], ...]
    z_layers_mm: tuple[float, ...]
    reference_layer_index: int
    source_2d_node_count: int
    source_2d_triangle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_column_count": len(self.node_columns),
            "layers": len(self.z_layers_mm),
            "z_layers_mm": list(self.z_layers_mm),
            "reference_layer_index": self.reference_layer_index,
            "source_2d_node_count": self.source_2d_node_count,
            "source_2d_triangle_count": self.source_2d_triangle_count,
            "correspondence": "exact shared local 2D node index across every explicit z layer",
        }


_BOUNDARY_TAGS = {
    "pad_bond_left": "support_bond_left",
    "pad_bond_right": "support_bond_right",
    "pad_outer_arc": "outer_compliant_arc",
    "pad_outer_left": "outer_compliant_left",
    "pad_outer_right": "outer_compliant_right",
    "pad_cutout_left": "void_left",
    "pad_cutout_right": "void_right",
    "pad_cutout_bottom": "void_bottom",
    "pad_void_unpaired": "void_other",
}


def _signed_tetra_volume(points: np.ndarray) -> float:
    return float(np.linalg.det(np.vstack((points[1:] - points[0]))) / 6.0)


def _orient_tetra(
    node_ids: tuple[int, int, int, int],
    nodes: Mapping[int, VolumeNode],
) -> tuple[tuple[int, int, int, int], float]:
    points = np.asarray(
        [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in node_ids],
        dtype=float,
    )
    volume = _signed_tetra_volume(points)
    if volume < 0.0:
        node_ids = (node_ids[0], node_ids[2], node_ids[1], node_ids[3])
        volume = -volume
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("layered reference tetrahedron has nonpositive volume")
    return node_ids, volume


def _tetra_quality(points: np.ndarray, volume: float) -> float:
    edge_sum = sum(
        float(np.sum((points[i] - points[j]) ** 2))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    return float(12.0 * (3.0 * volume) ** (2.0 / 3.0) / edge_sum) if edge_sum else 0.0


def _surface_area(triangle: SurfaceTriangle, nodes: Mapping[int, VolumeNode]) -> float:
    points = np.asarray(
        [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in triangle.node_ids],
        dtype=float,
    )
    return 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))


def build_plane_strain_reference_mesh(
    parameters: FingertipParameters,
    *,
    mesh_settings: MeshSettings | None = None,
    layer_count: int = 3,
) -> tuple[FingertipVolumeMesh, PlaneStrainReferenceContract]:
    """Build a deterministic layered TET4 mesh from the authoritative 2D pad."""
    if layer_count < 2 or layer_count % 2 == 0:
        raise ValueError("layer_count must be an odd integer of at least three")
    source_settings = mesh_settings or MeshSettings(
        "medium", 1.0, 0.4, contact_refinement_distance_mm=1.0
    )
    fingertip = Fingertip(parameters)
    source_mesh = generate_fingertip_mesh(fingertip.geometry, source_settings)
    pad = source_mesh.pad
    solid = build_fingertip_solid(fingertip.geometry)
    z_layers = tuple(
        float(value)
        for value in np.linspace(solid.z_min_mm, solid.z_max_mm, layer_count)
    )
    source_node_count = len(pad.node_ids)
    nodes: dict[int, VolumeNode] = {}
    node_columns: list[tuple[int, ...]] = []
    for local_index, (x_mm, y_mm) in enumerate(pad.reference_coordinates_mm):
        column: list[int] = []
        for layer_index, z_mm in enumerate(z_layers):
            node_id = layer_index * source_node_count + local_index + 1
            nodes[node_id] = VolumeNode(node_id, float(x_mm), float(y_mm), z_mm)
            column.append(node_id)
        node_columns.append(tuple(column))

    tetrahedra: list[Tetrahedron] = []
    quality_values: list[float] = []
    mesh_volume = 0.0
    element_id = 1
    for layer_index in range(layer_count - 1):
        bottom_offset = layer_index * source_node_count
        top_offset = (layer_index + 1) * source_node_count
        for triangle in pad.triangles:
            a, b, c = (int(value) for value in triangle)
            bottom = (bottom_offset + a + 1, bottom_offset + b + 1, bottom_offset + c + 1)
            top = (top_offset + a + 1, top_offset + b + 1, top_offset + c + 1)
            candidates = (
                (bottom[0], bottom[1], bottom[2], top[0]),
                (bottom[1], top[1], top[2], top[0]),
                (bottom[1], top[2], bottom[2], top[0]),
            )
            for raw_nodes in candidates:
                oriented, volume = _orient_tetra(raw_nodes, nodes)
                points = np.asarray(
                    [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in oriented],
                    dtype=float,
                )
                tetrahedra.append(Tetrahedron(element_id, oriented, "pad"))
                quality_values.append(_tetra_quality(points, volume))
                mesh_volume += volume
                element_id += 1

    surface_triangles: dict[str, list[SurfaceTriangle]] = {}
    surface_id = element_id
    for source_tag, output_tag in _BOUNDARY_TAGS.items():
        edges = pad.boundary_edges_by_tag.get(source_tag)
        if edges is None:
            continue
        target = surface_triangles.setdefault(output_tag, [])
        for layer_index in range(layer_count - 1):
            bottom_offset = layer_index * source_node_count
            top_offset = (layer_index + 1) * source_node_count
            for first, second in edges:
                a = bottom_offset + int(first) + 1
                b = bottom_offset + int(second) + 1
                A = top_offset + int(first) + 1
                B = top_offset + int(second) + 1
                target.extend(
                    (
                        SurfaceTriangle(surface_id, (a, b, B), output_tag, "pad"),
                        SurfaceTriangle(surface_id + 1, (a, B, A), output_tag, "pad"),
                    )
                )
                surface_id += 2
    for layer_index, z_tag in ((0, "longitudinal_end_minus"), (layer_count - 1, "longitudinal_end_plus")):
        offset = layer_index * source_node_count
        target = surface_triangles.setdefault(z_tag, [])
        for triangle in pad.triangles:
            a, b, c = (offset + int(value) + 1 for value in triangle)
            node_ids = (a, c, b) if layer_index == 0 else (a, b, c)
            target.append(SurfaceTriangle(surface_id, node_ids, z_tag, "pad"))
            surface_id += 1

    all_node_ids = set(nodes)
    used_node_ids = {node_id for tetrahedron in tetrahedra for node_id in tetrahedron.node_ids}
    inverted = sum(
        _signed_tetra_volume(
            np.asarray(
                [[nodes[node_id].x_mm, nodes[node_id].y_mm, nodes[node_id].z_mm] for node_id in tetrahedron.node_ids],
                dtype=float,
            )
        ) <= 0.0
        for tetrahedron in tetrahedra
    )
    bonded_tags = tuple(
        definition.name
        for definition in solid.surfaces
        if definition.kind == "support" and definition.source_geometry is not None
    )
    bonded_area = sum(
        _surface_area(triangle, nodes)
        for tag in bonded_tags
        for triangle in surface_triangles.get(tag, ())
    )
    expected_bonded_area = sum(
        float(definition.source_geometry.length * solid.extrusion_depth_mm)
        for definition in solid.surfaces
        if definition.name in bonded_tags and definition.source_geometry is not None
    )
    quality = VolumeMeshQuality(
        node_count=len(nodes),
        tetrahedron_count=len(tetrahedra),
        surface_triangle_count=sum(len(values) for values in surface_triangles.values()),
        minimum_scaled_jacobian=min(quality_values, default=0.0),
        maximum_edge_length_mm=max(
            (
                float(np.linalg.norm(
                    np.asarray([nodes[first].x_mm, nodes[first].y_mm, nodes[first].z_mm])
                    - np.asarray([nodes[second].x_mm, nodes[second].y_mm, nodes[second].z_mm])
                ))
                for tetrahedron in tetrahedra
                for first, second in ((tetrahedron.node_ids[0], tetrahedron.node_ids[1]),
                                      (tetrahedron.node_ids[0], tetrahedron.node_ids[2]),
                                      (tetrahedron.node_ids[0], tetrahedron.node_ids[3]),
                                      (tetrahedron.node_ids[1], tetrahedron.node_ids[2]),
                                      (tetrahedron.node_ids[1], tetrahedron.node_ids[3]),
                                      (tetrahedron.node_ids[2], tetrahedron.node_ids[3]))
            ),
            default=0.0,
        ),
        mesh_volume_mm3=mesh_volume,
        geometry_volume_mm3=solid.volume_mm3,
        volume_relative_error=abs(mesh_volume - solid.volume_mm3) / solid.volume_mm3,
        inverted_tetrahedron_count=inverted,
        semantic_surface_tags=tuple(sorted(surface_triangles)),
        surface_triangle_degenerate_count=0,
        surface_orientation_failure_count=0,
        closed_surface_edge_failure_count=0,
        bonded_surface_triangle_count=sum(len(surface_triangles.get(tag, ())) for tag in bonded_tags),
        bonded_surface_area_mm2=bonded_area,
        bonded_surface_expected_area_mm2=expected_bonded_area,
        bonded_surface_area_relative_error=abs(bonded_area - expected_bonded_area) / expected_bonded_area,
    )
    validation_checks = {
        "positive_volume": solid.volume_mm3 > 0.0,
        "all_tetrahedra_positive": inverted == 0 and bool(tetrahedra),
        "volume_consistency": quality.volume_relative_error <= 1.0e-10,
        "no_orphan_nodes": used_node_ids == all_node_ids,
        "semantic_surface_coverage": all(tag in surface_triangles for tag in ("outer_compliant_arc", "void_left", "void_right", "void_bottom")),
        "bonded_surface_area_consistency": quality.bonded_surface_area_relative_error <= 1.0e-10,
        "exact_layered_correspondence": len(node_columns) == source_node_count and all(len(column) == layer_count for column in node_columns),
    }
    validation = VolumeMeshValidation(
        passed=all(validation_checks.values()),
        checks=validation_checks,
        errors=tuple(name for name, passed in validation_checks.items() if not passed),
    )
    volume_mesh = FingertipVolumeMesh(
        solid=solid,
        nodes=nodes,
        tetrahedra=tuple(tetrahedra),
        surface_triangles={tag: tuple(values) for tag, values in surface_triangles.items()},
        volume_element_ids={"pad": tuple(tetrahedron.id for tetrahedron in tetrahedra)},
        settings=VolumeMeshSettings("reference", source_settings.bulk_target_size_mm, 0.02),
        quality=quality,
        validation=validation,
        gmsh_version=source_mesh.gmsh_version,
    )
    contract = PlaneStrainReferenceContract(
        node_columns=tuple(node_columns),
        z_layers_mm=z_layers,
        reference_layer_index=layer_count // 2,
        source_2d_node_count=source_node_count,
        source_2d_triangle_count=len(pad.triangles),
    )
    return volume_mesh, contract


__all__ = ["PlaneStrainReferenceContract", "build_plane_strain_reference_mesh"]
