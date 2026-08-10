"""Convert persisted or in-memory FEA fields to neutral optical state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from optics.geometry.deformation_state import (
    InvalidPadDeformationState,
    PadDeformationState2D,
    PadField2D,
)
from optics.geometry.pad_mesh_template import (
    InvalidPadMeshTemplate,
    PadMeshTemplate2D,
)

if TYPE_CHECKING:
    from mesh.types import FingertipMesh
    from model.fingertip_sensor_model import FingertipSensorModel


class OpticalFieldAdapterError(ValueError):
    """Raised when an FEA field cannot satisfy the optical-state contract."""


_NPZ_KEYS = (
    "node_ids",
    "reference_coordinates_mm",
    "element_connectivity_node_ids",
    "displacement_mm",
)


def build_pad_field_from_arrays(
    *,
    node_ids: np.ndarray,
    reference_coordinates_mm: np.ndarray,
    element_connectivity_node_ids: np.ndarray,
    displacement_mm: np.ndarray,
    boundary_edge_node_ids_by_tag: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PadField2D:
    """Build the canonical optical pad field from plain arrays."""
    try:
        template = PadMeshTemplate2D.from_arrays(
            node_ids=node_ids,
            reference_coordinates_mm=reference_coordinates_mm,
            element_connectivity_node_ids=element_connectivity_node_ids,
            boundary_edge_node_ids_by_tag=boundary_edge_node_ids_by_tag,
        )
        state = PadDeformationState2D(
            displacement_mm=displacement_mm,
            metadata=metadata or {},
        )
        return PadField2D(template=template, state=state)
    except (InvalidPadMeshTemplate, InvalidPadDeformationState) as exc:
        raise OpticalFieldAdapterError(f"invalid optical pad field: {exc}") from exc


def load_pad_field_npz(
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    sensor_model: FingertipSensorModel | None = None,
) -> PadField2D:
    """Load deformation arrays and optionally classify reference boundaries."""
    source = Path(path).expanduser()
    try:
        with np.load(source, allow_pickle=False) as payload:
            missing = [key for key in _NPZ_KEYS if key not in payload]
            if missing:
                raise OpticalFieldAdapterError(
                    "deformation NPZ is missing required arrays: "
                    + ", ".join(missing)
                )
            arrays = {key: np.array(payload[key], copy=True) for key in _NPZ_KEYS}
    except OpticalFieldAdapterError:
        raise
    except (OSError, ValueError) as exc:
        raise OpticalFieldAdapterError(
            f"could not load deformation NPZ '{source}': {exc}"
        ) from exc
    field = build_pad_field_from_arrays(metadata=metadata, **arrays)
    if sensor_model is None:
        return field
    from optics.adapters.semantic_boundaries import tag_reference_pad_boundaries

    tagged_template = tag_reference_pad_boundaries(
        sensor_model,
        field.template,
    )
    return PadField2D(template=tagged_template, state=field.state)


def build_pad_field_from_mesh_and_displacements(
    mesh: FingertipMesh,
    displacements: Mapping[int, Sequence[float]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PadField2D:
    """Adapt an in-memory fingertip mesh and nodal displacement mapping."""
    if not mesh.pad_elements:
        raise OpticalFieldAdapterError("mesh has no pad elements")
    pad_node_ids = sorted(
        {
            int(node_id)
            for element in mesh.pad_elements
            for node_id in element.node_ids
        }
    )
    try:
        coordinates = np.asarray(
            [
                [mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm]
                for node_id in pad_node_ids
            ],
            dtype=float,
        )
    except KeyError as exc:
        raise OpticalFieldAdapterError(
            f"pad element references missing mesh node {exc.args[0]}"
        ) from exc
    try:
        displacement_xy = np.asarray(
            [
                [displacements[node_id][0], displacements[node_id][1]]
                for node_id in pad_node_ids
            ],
            dtype=float,
        )
    except KeyError as exc:
        raise OpticalFieldAdapterError(
            f"displacement mapping is missing pad node {exc.args[0]}"
        ) from exc
    except (IndexError, TypeError, ValueError) as exc:
        raise OpticalFieldAdapterError(
            "each pad displacement must provide finite x-y components"
        ) from exc
    connectivity = np.asarray(
        [element.node_ids for element in mesh.pad_elements],
        dtype=np.int64,
    )
    pad_node_id_set = set(pad_node_ids)
    boundary_edge_node_ids_by_tag = {
        tag: np.asarray(selected_edges, dtype=np.int64)
        for tag, edges in mesh.boundary_edges.items()
        if (
            selected_edges := [
                edge.node_ids
                for edge in edges
                if edge.domain == "pad"
                and set(edge.node_ids).issubset(pad_node_id_set)
            ]
        )
    }
    return build_pad_field_from_arrays(
        node_ids=np.asarray(pad_node_ids, dtype=np.int64),
        reference_coordinates_mm=coordinates,
        element_connectivity_node_ids=connectivity,
        displacement_mm=displacement_xy,
        boundary_edge_node_ids_by_tag=boundary_edge_node_ids_by_tag,
        metadata=metadata,
    )
