"""Small private Matplotlib helpers shared by the plotting functions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import PathPatch
from shapely.geometry import MultiPolygon, Polygon

from model.fingertip_model import PolygonalGeometry

PAD_COLOR = "#C7E8D2"
PAD_EDGE = "#4E9270"
ALUMINUM_COLOR = "#D9DCDF"
ALUMINUM_EDGE = "#7B8288"
VOID_COLOR = "#F7B4AE"
VOID_EDGE = "#C9473D"
PAD_CONTACT_COLOR = "#D95F02"
BONDED_INTERFACE_COLOR = "#F4E04D"
BONDED_INTERFACE_EDGE = "#C49A00"
LED_COLOR = "#F6C453"
LED_EDGE = "#9A6700"
LIGHT_SOURCE_COLOR = "#E63946"
LIGHT_SOURCE_EDGE = "#4A1018"


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
