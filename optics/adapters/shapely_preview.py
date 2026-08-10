"""Dependency-light analytic preview topology for optical examples."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

from model.fingertip_sensor_model import FingertipSensorModel
from optics.adapters.semantic_boundaries import tag_reference_pad_boundaries
from optics.geometry.pad_mesh_template import PadMeshTemplate2D


def build_preview_pad_mesh_template(
    sensor_model: FingertipSensorModel,
) -> PadMeshTemplate2D:
    """Triangulate analytic pad material for undeformed preview rendering."""
    material = sensor_model.geometry.pad_material_geometry
    retained: list[Polygon] = []
    for candidate in triangulate(material):
        if material.covers(candidate):
            retained.append(candidate)
    if not retained:
        raise ValueError("analytic pad material produced no preview triangles")

    triangle_coordinates = [
        tuple(
            (float(point[0]), float(point[1]))
            for point in polygon.exterior.coords[:3]
        )
        for polygon in retained
    ]
    unique_coordinates = sorted(
        {point for triangle in triangle_coordinates for point in triangle}
    )
    coordinate_to_id = {
        coordinate: index + 1
        for index, coordinate in enumerate(unique_coordinates)
    }
    connectivity = sorted(
        tuple(coordinate_to_id[point] for point in triangle)
        for triangle in triangle_coordinates
    )
    template = PadMeshTemplate2D.from_arrays(
        node_ids=np.arange(1, len(unique_coordinates) + 1, dtype=np.int64),
        reference_coordinates_mm=np.asarray(unique_coordinates, dtype=float),
        element_connectivity_node_ids=np.asarray(connectivity, dtype=np.int64),
    )
    return tag_reference_pad_boundaries(sensor_model, template)
