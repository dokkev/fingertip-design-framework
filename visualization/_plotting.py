"""Small private Matplotlib helpers shared by the plotting functions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import PathPatch
from shapely.geometry import MultiPolygon, Polygon

from model.fingertip_model import PolygonalGeometry
from visualization._style import STYLE

PAD_COLOR = STYLE.silicone_face
PAD_EDGE = STYLE.silicone_edge
ALUMINUM_COLOR = STYLE.rigid_face
ALUMINUM_EDGE = STYLE.rigid_edge
VOID_COLOR = STYLE.void_face
VOID_EDGE = STYLE.void_edge
PAD_CONTACT_COLOR = STYLE.contact_edge
BONDED_INTERFACE_COLOR = STYLE.bonded_interface_face
BONDED_INTERFACE_EDGE = STYLE.bonded_interface_edge
LED_COLOR = STYLE.led_face
LED_EDGE = STYLE.led_edge
LIGHT_SOURCE_COLOR = STYLE.source_face
LIGHT_SOURCE_EDGE = STYLE.source_edge


def add_polygonal_patches(
    ax: Axes,
    geometry: PolygonalGeometry,
    *,
    facecolor: str | tuple[float, ...],
    edgecolor: str,
    linewidth: float,
    label: str,
    zorder: int,
    linestyle: str = "-",
    hatch: str | None = None,
    alpha: float = 1.0,
) -> None:
    for index, polygon in enumerate(iter_polygons(geometry)):
        ax.add_patch(
            PathPatch(
                polygon_to_path(polygon),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                hatch=hatch,
                alpha=alpha,
                label=label if index == 0 else None,
                zorder=zorder,
            )
        )


def iter_polygons(geometry: PolygonalGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return geometry.geoms
    return ()


def polygon_to_path(polygon: Polygon) -> MatplotlibPath:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in (polygon.exterior, *polygon.interiors):
        ring_vertices = [(float(x), float(y)) for x, y in ring.coords]
        vertices.extend(ring_vertices)
        codes.extend(
            [MatplotlibPath.MOVETO]
            + [MatplotlibPath.LINETO] * (len(ring_vertices) - 2)
            + [MatplotlibPath.CLOSEPOLY]
        )
    return MatplotlibPath(np.asarray(vertices, dtype=float), codes)
