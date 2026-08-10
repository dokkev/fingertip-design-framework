"""Convert external displacement artifacts to neutral pad mesh views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mesh import FingertipMesh, InvalidPadMesh, PadMesh


class OpticalFieldAdapterError(ValueError):
    """Raised when an external field cannot define a valid pad mesh view."""


_REQUIRED_NPZ_KEYS = (
    "node_ids",
    "reference_coordinates_mm",
    "element_connectivity_node_ids",
    "displacement_mm",
)
_BOUNDARY_KEY_PREFIX = "boundary_edge_node_ids__"


def build_pad_mesh_from_arrays(
    *,
    node_ids: np.ndarray,
    reference_coordinates_mm: np.ndarray,
    element_connectivity_node_ids: np.ndarray,
    displacement_mm: np.ndarray,
    boundary_edge_node_ids_by_tag: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Build one loaded pad mesh from an external array boundary."""
    if not boundary_edge_node_ids_by_tag:
        raise OpticalFieldAdapterError(
            "external optical pad arrays must include semantic boundary edges"
        )
    try:
        mesh = PadMesh.from_arrays(
            node_ids=node_ids,
            reference_coordinates_mm=reference_coordinates_mm,
            element_connectivity_node_ids=element_connectivity_node_ids,
            boundary_edge_node_ids_by_tag=boundary_edge_node_ids_by_tag,
        )
        return mesh.deformed(displacement_mm, metadata=metadata)
    except (InvalidPadMesh, ValueError) as exc:
        raise OpticalFieldAdapterError(f"invalid optical pad mesh: {exc}") from exc


def load_pad_mesh_npz(
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Load a deformed pad mesh, preserving stored semantic boundaries."""
    source = Path(path).expanduser()
    try:
        with np.load(source, allow_pickle=False) as payload:
            missing = [key for key in _REQUIRED_NPZ_KEYS if key not in payload]
            if missing:
                raise OpticalFieldAdapterError(
                    "deformation NPZ is missing required arrays: "
                    + ", ".join(missing)
                )
            arrays = {
                key: np.array(payload[key], copy=True)
                for key in _REQUIRED_NPZ_KEYS
            }
            boundaries = {
                key.removeprefix(_BOUNDARY_KEY_PREFIX): np.array(
                    payload[key],
                    copy=True,
                )
                for key in payload.files
                if key.startswith(_BOUNDARY_KEY_PREFIX)
            }
    except OpticalFieldAdapterError:
        raise
    except (OSError, ValueError) as exc:
        raise OpticalFieldAdapterError(
            f"could not load deformation NPZ '{source}': {exc}"
        ) from exc
    return build_pad_mesh_from_arrays(
        boundary_edge_node_ids_by_tag=boundaries or None,
        metadata=metadata,
        **arrays,
    )


def deformed_pad_mesh_from_nodal_displacements(
    mesh: FingertipMesh,
    displacements: Mapping[int, Sequence[float]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Map a neutral FEM displacement mapping onto the mesh's pad nodes."""
    pad_mesh = mesh.pad
    try:
        displacement = np.asarray(
            [
                [
                    displacements[int(node_id)][0],
                    displacements[int(node_id)][1],
                ]
                for node_id in pad_mesh.node_ids
            ],
            dtype=float,
        )
        return pad_mesh.deformed(displacement, metadata=metadata)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OpticalFieldAdapterError(
            "nodal displacements do not define a valid pad mesh view"
        ) from exc


__all__ = [
    "OpticalFieldAdapterError",
    "build_pad_mesh_from_arrays",
    "deformed_pad_mesh_from_nodal_displacements",
    "load_pad_mesh_npz",
]
