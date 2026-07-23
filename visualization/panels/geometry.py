"""Geometry, deformation heatmap, vector, and contact panels."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.tri as mtri
import numpy as np

from visualization.data import MeshData, ObservationChain, ScientificFigureError
from visualization.theme import FigureTheme, ScalePolicy
from visualization.transforms import (
    SelectedTransferState,
    deterministic_spatial_subsample,
)

class MeshPanel:
    """Render exact undeformed mesh topology without altering scientific data."""

    def render(
        self,
        axis: plt.Axes,
        mesh: MeshData,
        theme: FigureTheme,
        *,
        linewidth: float = 0.08,
        alpha: float = 0.18,
    ) -> dict[str, Any]:
        coordinates = mesh.coordinate_by_node_id
        polygons = [
            [coordinates[int(node_id)] for node_id in connectivity]
            for connectivity in mesh.element_connectivity
        ]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors="none",
                edgecolors=theme.mesh_color,
                linewidths=linewidth,
                alpha=alpha,
                zorder=1,
            )
        )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        return {
            "component": "MeshPanel",
            "mesh_id": mesh.mesh_id,
            "node_count": len(mesh.node_ids),
            "element_count": len(mesh.element_ids),
            "units": mesh.units,
            "aspect": "equal",
        }


class DeformedMeshPanel:
    """Render deformed topology only when a full nodal field is available."""

    def render(
        self,
        axis: plt.Axes,
        mesh: MeshData,
        displacement_by_node_id: Mapping[int, Sequence[float]],
        theme: FigureTheme,
        scale_policy: ScalePolicy,
    ) -> dict[str, Any]:
        if set(displacement_by_node_id) != set(mesh.node_ids):
            raise ScientificFigureError(
                "DeformedMeshPanel requires displacement for every mesh node"
            )
        coordinates = {
            node_id: mesh.coordinate_by_node_id[node_id]
            + scale_policy.deformation_scale
            * np.asarray(displacement_by_node_id[node_id], dtype=float)
            for node_id in mesh.node_ids
        }
        if not np.isfinite(np.asarray(list(coordinates.values()))).all():
            raise ScientificFigureError("deformed mesh coordinates are non-finite")
        polygons = [
            [coordinates[int(node_id)] for node_id in connectivity]
            for connectivity in mesh.element_connectivity
        ]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors="#B7E1EA",
                edgecolors=theme.mesh_color,
                linewidths=0.10,
                alpha=0.55,
                zorder=2,
            )
        )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        return {
            "component": "DeformedMeshPanel",
            "deformation_scale": scale_policy.deformation_scale,
            "full_nodal_field": True,
        }


class NodalDisplacementMagnitudePanel:
    """Render nodal ``|u|`` on the actual T3 mesh in its deformed position."""

    def render(
        self,
        axis: plt.Axes,
        mesh: MeshData,
        displacement_by_node_id: Mapping[int, Sequence[float]],
        theme: FigureTheme,
        scale_policy: ScalePolicy,
        norm: colors.Normalize,
    ) -> dict[str, Any]:
        if set(displacement_by_node_id) != set(mesh.node_ids):
            raise ScientificFigureError(
                "heatmap requires displacement for every pad mesh node"
            )
        displacement = np.asarray(
            [displacement_by_node_id[node_id] for node_id in mesh.node_ids],
            dtype=float,
        )
        if displacement.shape != mesh.node_coordinates.shape:
            raise ScientificFigureError("full nodal displacement shape is invalid")
        deformed = (
            mesh.node_coordinates
            + scale_policy.deformation_scale * displacement
        )
        magnitude = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(deformed).all() or not np.isfinite(magnitude).all():
            raise ScientificFigureError("heatmap coordinates or values are non-finite")
        node_index = {
            node_id: index for index, node_id in enumerate(mesh.node_ids)
        }
        triangle_indices = np.asarray(
            [
                [node_index[int(node_id)] for node_id in connectivity]
                for connectivity in mesh.element_connectivity
            ],
            dtype=int,
        )
        triangulation = mtri.Triangulation(
            deformed[:, 0], deformed[:, 1], triangle_indices
        )
        image = axis.tripcolor(
            triangulation,
            magnitude,
            shading="gouraud",
            cmap=theme.magnitude_colormap,
            norm=norm,
            zorder=2,
        )
        axis.triplot(
            triangulation,
            color=theme.mesh_color,
            linewidth=0.05,
            alpha=0.16,
            zorder=3,
        )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        return {
            "component": "NodalDisplacementMagnitudePanel",
            "image": image,
            "represented_scalar": "nodal displacement magnitude |u|",
            "units": "mm",
            "interpolation": "linear T3 nodal interpolation (gouraud)",
            "geometry_configuration": "deformed",
            "deformation_scale": scale_policy.deformation_scale,
            "full_pad_field": True,
            "carrier_included": False,
            "indenter_included": False,
        }


class DisplacementVectorPanel:
    """Render deterministic sparse arrows representing actual u=[ux,uy]."""

    def render(
        self,
        axis: plt.Axes,
        coordinates: np.ndarray,
        displacement: np.ndarray,
        theme: FigureTheme,
        scale_policy: ScalePolicy,
        *,
        maximum_arrows: int,
        color: str = "#262626",
        anchor_configuration: str = "reference",
    ) -> tuple[dict[str, Any], np.ndarray]:
        points = np.asarray(coordinates, dtype=float)
        vectors = np.asarray(displacement, dtype=float)
        if points.shape != vectors.shape or points.ndim != 2 or points.shape[1] != 2:
            raise ScientificFigureError("vector panel requires matching [n,2] arrays")
        magnitudes = np.linalg.norm(vectors, axis=1)
        eligible = np.flatnonzero(magnitudes >= scale_policy.arrow_minimum_mm)
        if not len(eligible):
            raise ScientificFigureError("no displacement vector passes the threshold")
        local = deterministic_spatial_subsample(
            points[eligible], maximum_count=maximum_arrows
        )
        selected = eligible[local]
        axis.quiver(
            points[selected, 0],
            points[selected, 1],
            vectors[selected, 0],
            vectors[selected, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0 / scale_policy.arrow_scale,
            color=color,
            width=0.0045,
            headwidth=3.8,
            headlength=5.0,
            zorder=8,
        )
        return (
            {
                "component": "DisplacementVectorPanel",
                "represented_vector": "physical displacement u=[u_x,u_y]",
                "selection": "deterministic spatial binning",
                "selected_count": int(len(selected)),
                "candidate_count": int(len(points)),
                "arrow_scale": scale_policy.arrow_scale,
                "minimum_magnitude_mm": scale_policy.arrow_minimum_mm,
                "normalized_arrows": False,
                "anchor_configuration": anchor_configuration,
            },
            selected,
        )


class ContactInputAnnotation:
    """Annotate the actual prescribed indentation direction and contact point."""

    def render(
        self,
        axis: plt.Axes,
        state: SelectedTransferState,
        *,
        length_mm: float = 3.0,
    ) -> dict[str, Any]:
        return self.render_at(
            axis,
            state.contact_point_mm,
            state.indentation_direction,
            length_mm=length_mm,
        )

    def render_at(
        self,
        axis: plt.Axes,
        contact_point_mm: Sequence[float] | None,
        indentation_direction: Sequence[float],
        *,
        length_mm: float = 3.0,
    ) -> dict[str, Any]:
        """Annotate an explicitly framed prescribed indentation vector."""
        if contact_point_mm is None:
            return {
                "component": "ContactInputAnnotation",
                "available": False,
            }
        point = np.asarray(contact_point_mm, dtype=float)
        direction = np.asarray(indentation_direction, dtype=float)
        if (
            point.shape != (2,)
            or direction.shape != (2,)
            or not np.isfinite(point).all()
            or not np.isfinite(direction).all()
            or not math.isclose(
                float(np.linalg.norm(direction)), 1.0, abs_tol=1.0e-10
            )
        ):
            raise ScientificFigureError(
                "contact annotation point/direction is invalid"
            )
        start = point - length_mm * direction
        axis.annotate(
            "",
            xy=point,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#111111",
                "lw": 2.0,
                "mutation_scale": 13,
            },
            zorder=12,
        )
        axis.scatter(*point, s=24, facecolor="white", edgecolor="black", zorder=13)
        return {
            "component": "ContactInputAnnotation",
            "available": True,
            "contact_point_mm": point.tolist(),
            "indentation_direction": direction.tolist(),
            "arrow_length_mm": length_mm,
            "represented_vector": "prescribed indentation direction, not displacement",
        }


class ObservationBoundaryOverlay:
    """Render independent reference/deformed chains without a center connection."""

    def render(
        self,
        axis: plt.Axes,
        chains: Mapping[str, ObservationChain],
        theme: FigureTheme,
        *,
        displacement_by_side: Mapping[str, np.ndarray] | None = None,
        values_by_side: Mapping[str, np.ndarray] | None = None,
        norm: colors.Normalize | None = None,
        scale_policy: ScalePolicy = ScalePolicy(),
    ) -> dict[str, Any]:
        for side in ("right", "left"):
            chain = chains[side]
            reference = chain.undeformed_coordinates
            axis.plot(
                reference[:, 0],
                reference[:, 1],
                color="#5F6368",
                lw=1.0,
                alpha=0.8,
                zorder=5,
            )
            if displacement_by_side is None:
                axis.plot(
                    reference[:, 0],
                    reference[:, 1],
                    color=theme.observation_colors[side],
                    lw=2.4,
                    label=f"{side} observation chain",
                    zorder=6,
                )
                continue
            displacement = np.asarray(displacement_by_side[side], dtype=float)
            if displacement.shape != reference.shape:
                raise ScientificFigureError("chain displacement shape is invalid")
            deformed = reference + scale_policy.deformation_scale * displacement
            segments = np.stack([deformed[:-1], deformed[1:]], axis=1)
            if values_by_side is not None and norm is not None:
                values = np.asarray(values_by_side[side], dtype=float)
                collection = LineCollection(
                    segments,
                    cmap=theme.signed_colormap,
                    norm=norm,
                    linewidths=3.0,
                    zorder=7,
                )
                collection.set_array(0.5 * (values[:-1] + values[1:]))
                axis.add_collection(collection)
            else:
                axis.plot(
                    deformed[:, 0],
                    deformed[:, 1],
                    color=theme.observation_colors[side],
                    lw=2.8,
                    zorder=7,
                )
        axis.set_aspect("equal", adjustable="box")
        return {
            "component": "ObservationBoundaryOverlay",
            "sides": ["right", "left"],
            "center_connected": False,
            "deformation_scale": scale_policy.deformation_scale,
            "contact_facing_surface_inferred": False,
        }

