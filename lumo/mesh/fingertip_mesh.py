"""Discretized fingertip geometry for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lumo.fingertip.fingertip import Fingertip
from lumo.util.scalar_validation import require_positive

from .carrier_mesh import _make_carrier_collision_mesh, _make_carrier_mesh
from .silicone_mesh import _make_silicone_mesh

if TYPE_CHECKING:
    import newton


@dataclass(frozen=True)
class FingertipMesh:
    """Newton meshes produced from one analytic fingertip assembly."""

    fingertip: Fingertip
    silicone: "newton.TetMesh"
    carrier: "newton.Mesh"
    carrier_collision: "newton.Mesh"
    bonded_vertex_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(
            self.bonded_vertex_indices,
            dtype=np.int32,
        )
        if indices.ndim != 1:
            raise ValueError("bonded_vertex_indices must be one-dimensional")
        if np.any(indices < 0):
            raise ValueError("bonded_vertex_indices must be non-negative")

        indices = np.unique(indices)
        indices.setflags(write=False)
        object.__setattr__(self, "bonded_vertex_indices", indices)


def make_fingertip_mesh(
    fingertip: Fingertip,
    *,
    extrusion_depth_mm: float = 11.0,
    element_size_mm: float = 1.0,
) -> FingertipMesh:
    """Discretize the silicone and carrier of one fingertip assembly."""
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")

    require_positive("extrusion_depth_mm", extrusion_depth_mm)
    require_positive("element_size_mm", element_size_mm)

    silicone, bonded_vertex_indices = _make_silicone_mesh(
        fingertip.silicone,
        extrusion_depth_mm=extrusion_depth_mm,
        element_size_mm=element_size_mm,
    )
    carrier = _make_carrier_mesh(
        fingertip.carrier,
        extrusion_depth_mm=extrusion_depth_mm,
    )
    carrier_collision = _make_carrier_collision_mesh(
        fingertip.carrier,
        fingertip.silicone,
        extrusion_depth_mm=extrusion_depth_mm,
    )

    return FingertipMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=carrier,
        carrier_collision=carrier_collision,
        bonded_vertex_indices=bonded_vertex_indices,
    )


__all__ = ["FingertipMesh", "make_fingertip_mesh"]
