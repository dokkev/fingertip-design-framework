"""Classify reference pad-mesh boundaries against analytic model semantics."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from model.fingertip_sensor_model import FingertipSensorModel
from optics.geometry.pad_mesh_template import PadMeshTemplate2D


class OpticalBoundaryClassificationError(ValueError):
    """Raised when reference mesh boundaries cannot be tagged uniquely."""


_NAMED_PAD_BOUNDARY_TAGS = (
    "pad_bond_left",
    "pad_bond_right",
    "pad_outer_left",
    "pad_outer_right",
    "pad_outer_arc",
    "pad_cutout_left",
    "pad_cutout_right",
    "pad_cutout_bottom",
)


def _covers_complete_edge(
    reference_geometry: BaseGeometry,
    edge: LineString,
    *,
    tolerance_mm: float,
) -> bool:
    return bool(reference_geometry.buffer(tolerance_mm).covers(edge))


def tag_reference_pad_boundaries(
    sensor_model: FingertipSensorModel,
    template: PadMeshTemplate2D,
) -> PadMeshTemplate2D:
    """Return a copy whose reference boundary has one semantic pad tag each."""
    geometry = sensor_model.geometry
    scale = max(
        1.0,
        *(abs(value) for value in geometry.outer_pad_geometry.bounds),
    )
    tolerance = max(
        8.0 * geometry.parameters.geometry_tolerance,
        64.0 * np.finfo(float).eps * scale,
    )
    analytic_segments = geometry.boundaries.segments
    material_boundary = geometry.pad_material_geometry.boundary
    external_boundary = geometry.outer_pad_geometry.boundary
    edges_by_tag: dict[str, list[tuple[int, int]]] = {
        tag: [] for tag in _NAMED_PAD_BOUNDARY_TAGS
    }
    edges_by_tag["pad_void_unpaired"] = []

    for first_value, second_value in template.boundary_edges:
        first = int(first_value)
        second = int(second_value)
        edge = LineString(
            [
                template.reference_coordinates_mm[first],
                template.reference_coordinates_mm[second],
            ]
        )
        matches = [
            tag
            for tag in _NAMED_PAD_BOUNDARY_TAGS
            if _covers_complete_edge(
                analytic_segments[tag].geometry,
                edge,
                tolerance_mm=tolerance,
            )
        ]
        if len(matches) > 1:
            global_ids = (
                int(template.node_ids[first]),
                int(template.node_ids[second]),
            )
            raise OpticalBoundaryClassificationError(
                f"pad boundary edge {global_ids} with coordinates "
                f"{tuple(edge.coords)} matches multiple tags: {sorted(matches)}"
            )
        if matches:
            edges_by_tag[matches[0]].append((first, second))
            continue

        on_material_boundary = _covers_complete_edge(
            material_boundary,
            edge,
            tolerance_mm=tolerance,
        )
        on_external_boundary = _covers_complete_edge(
            external_boundary,
            edge,
            tolerance_mm=tolerance,
        )
        if on_material_boundary and not on_external_boundary:
            edges_by_tag["pad_void_unpaired"].append((first, second))
            continue
        global_ids = (
            int(template.node_ids[first]),
            int(template.node_ids[second]),
        )
        raise OpticalBoundaryClassificationError(
            f"pad boundary edge {global_ids} with coordinates "
            f"{tuple(edge.coords)} cannot be classified safely"
        )

    populated = {
        tag: np.asarray(edges, dtype=np.int64)
        for tag, edges in edges_by_tag.items()
        if edges
    }
    try:
        return PadMeshTemplate2D(
            node_ids=template.node_ids,
            reference_coordinates_mm=template.reference_coordinates_mm,
            triangles=template.triangles,
            boundary_edges=template.boundary_edges,
            boundary_edges_by_tag=populated,
        )
    except ValueError as exc:
        raise OpticalBoundaryClassificationError(
            f"classified pad boundary partition is invalid: {exc}"
        ) from exc
