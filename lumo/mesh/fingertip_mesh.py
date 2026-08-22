"""Discretized fingertip geometry for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lumo.fingertip.fingertip import Fingertip
from lumo.util.scalar_validation import require_positive

from .carrier_mesh import _make_carrier_mesh
from .silicone_mesh import _make_silicone_mesh

if TYPE_CHECKING:
    import newton


@dataclass(frozen=True)
class FingertipMesh:
    """Newton meshes produced from one analytic fingertip assembly."""

    fingertip: Fingertip
    silicone: "newton.TetMesh"
    carrier: "newton.Mesh"


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

    silicone = _make_silicone_mesh(
        fingertip.silicone,
        extrusion_depth_mm=extrusion_depth_mm,
        element_size_mm=element_size_mm,
    )
    carrier = _make_carrier_mesh(
        fingertip.carrier,
        extrusion_depth_mm=extrusion_depth_mm,
    )

    return FingertipMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=carrier,
    )


__all__ = ["FingertipMesh", "make_fingertip_mesh"]
