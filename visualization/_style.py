"""Central visual vocabulary for physical and scalar figure elements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureStyle:
    """Small, explicit style table shared by all scientific renderers."""

    silicone_face: str = "#C7E8D2"
    silicone_edge: str = "#4E9270"
    rigid_face: str = "#D9DCDF"
    rigid_edge: str = "#7B8288"
    void_face: str = "#F7B4AE"
    void_edge: str = "#C9473D"
    bonded_interface_face: str = "#F4E04D"
    bonded_interface_edge: str = "#C49A00"
    contact_edge: str = "#D95F02"
    indenter_face: str = "#B8C2CC"
    indenter_edge: str = "#495057"
    led_face: str = "#F6C453"
    led_edge: str = "#9A6700"
    source_face: str = "#E63946"
    source_edge: str = "#4A1018"
    silicone_ray: str = "#FFF3B0"
    air_ray: str = "#8BE9FD"
    masked_cell: str = "#FFFFFF"
    debug_overlay: str = "#FFFFFF"
    mesh_edge: str = "#56616A"
    node_face: str = "#263238"
    mechanics_vectors: str = "#111111"


STYLE = FigureStyle()
MECHANICS_CMAP = "viridis"
OPTICS_CMAP = "magma"


__all__ = ["FigureStyle", "MECHANICS_CMAP", "OPTICS_CMAP", "STYLE"]
