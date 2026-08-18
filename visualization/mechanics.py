"""Shared mechanics draw layers and the public FEA plot wrapper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from mesh import FingertipMesh, PadMesh
from visualization._axes import apply_physical_axes, bounds_from_points
from visualization._style import MECHANICS_CMAP, STYLE


def _pad_view(mesh: Any) -> Any:
    if isinstance(mesh, FingertipMesh):
        return mesh.pad
    if isinstance(mesh, PadMesh):
        return mesh
    if not all(hasattr(mesh, name) for name in ("coordinates", "triangles")):
        raise TypeError("mesh must be a PadMesh or FingertipMesh")
    return mesh


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    pad = _pad_view(mesh)
    coordinates = np.asarray(pad.coordinates, dtype=float)
    triangles = np.asarray(pad.triangles, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("mesh.coordinates must have shape (N, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mesh.triangles must have shape (M, 3)")
    if np.any(triangles < 0) or np.any(triangles >= len(coordinates)):
        raise ValueError("mesh.triangles contains an invalid node index")
    return coordinates, triangles


def _carrier_arrays(mesh: FingertipMesh) -> tuple[np.ndarray, np.ndarray]:
    node_ids = sorted(
        {
            int(node_id)
            for element in mesh.carrier_elements
            for node_id in element.node_ids
        }
    )
    node_indices = {node_id: index for index, node_id in enumerate(node_ids)}
    coordinates = np.asarray(
        [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm] for node_id in node_ids],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [node_indices[int(node_id)] for node_id in element.node_ids]
            for element in mesh.carrier_elements
        ],
        dtype=np.int64,
    )
    return coordinates, triangles


def _validate_displacement(coordinates: np.ndarray, displacement: Any) -> np.ndarray:
    values = np.asarray(displacement, dtype=float)
    if values.shape != coordinates.shape:
        raise ValueError(
            "displacement must have the same shape as mesh.coordinates (N, 2)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("displacement must contain finite values")
    return values


def _validate_field(values: Any, *, name: str, size: int) -> np.ndarray:
    field = np.asarray(values, dtype=float)
    if field.shape != (size,):
        raise ValueError(f"{name} must contain one value per element")
    if not np.all(np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return field


def _arrow_indices(
    coordinates: np.ndarray,
    magnitudes: np.ndarray,
    *,
    minimum: float,
    maximum: int,
) -> np.ndarray:
    candidates = np.flatnonzero(magnitudes >= minimum)
    if len(candidates) <= maximum:
        return candidates
    order = np.lexsort((candidates, coordinates[candidates, 1], coordinates[candidates, 0]))
    ordered = candidates[order]
    selected = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
    return np.sort(ordered[selected])


def draw_rigid_carrier(
    ax: Axes,
    mesh: FingertipMesh,
    *,
    label: str = "Rigid link / stem",
) -> None:
    """Draw the neutral carrier topology carried by a full fingertip mesh."""
    coordinates, triangles = _carrier_arrays(mesh)
    if len(triangles):
        ax.add_collection(
            PolyCollection(
                coordinates[triangles],
                facecolors=STYLE.rigid_face,
                edgecolors=STYLE.rigid_edge,
                linewidths=0.35,
                label=label,
                zorder=1,
            )
        )


def draw_mesh(
    ax: Axes,
    mesh: PadMesh | FingertipMesh,
    *,
    coordinates: np.ndarray | None = None,
    show_nodes: bool = False,
    mesh_label: str = "T3 mesh",
    alpha: float = 1.0,
) -> None:
    """Draw pad T3 connectivity and semantic boundaries only."""
    reference_coordinates, triangles = _mesh_arrays(mesh)
    points = reference_coordinates if coordinates is None else np.asarray(coordinates, dtype=float)
    if points.shape != reference_coordinates.shape or not np.all(np.isfinite(points)):
        raise ValueError("display coordinates must match mesh.coordinates and be finite")
    ax.triplot(
        points[:, 0],
        points[:, 1],
        triangles,
        color=STYLE.mesh_edge,
        linewidth=0.45,
        alpha=alpha,
        label=mesh_label,
        zorder=2,
    )
    if show_nodes:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=10.0,
            color=STYLE.node_face,
            label="Nodes",
            zorder=3,
        )
    pad = _pad_view(mesh)
    tags = tuple(getattr(pad, "semantic_boundary_tags", ()))
    for tag in tags:
        edges = np.asarray(pad.boundary_edges_for(tag), dtype=np.int64)
        for index, edge in enumerate(edges):
            edge_points = points[edge]
            ax.plot(
                edge_points[:, 0],
                edge_points[:, 1],
                color=STYLE.contact_edge,
                linewidth=1.2,
                label=("Semantic boundary" if index == 0 and tag == tags[0] else "_nolegend_"),
                zorder=4,
            )


def draw_pad_outline(
    ax: Axes,
    mesh: PadMesh | FingertipMesh,
    *,
    label: str = "Deformed fingertip outline",
    linewidth: float = 1.0,
) -> None:
    """Draw the external pad boundary without drawing its interior mesh."""
    coordinates, _ = _mesh_arrays(mesh)
    pad = _pad_view(mesh)
    labeled = False
    for tag in ("pad_outer_left", "pad_outer_arc", "pad_outer_right"):
        if tag not in getattr(pad, "boundaries", {}):
            continue
        for edge in np.asarray(pad.boundaries[tag], dtype=np.int64):
            points = coordinates[edge]
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=STYLE.silicone_edge,
                linewidth=linewidth,
                label=label if not labeled else "_nolegend_",
                zorder=6,
            )
            labeled = True


def draw_contact_patch(
    ax: Axes,
    pose: Any | None,
    *,
    label: str = "Exact mechanical contact patch",
    linewidth: float = 2.4,
) -> None:
    """Draw only the exact contact boundary from a solved pose."""
    if pose is None or pose.contact_patch is None:
        return
    lines = pose.contact_patch.geoms if hasattr(pose.contact_patch, "geoms") else (pose.contact_patch,)
    for index, line in enumerate(lines):
        x, y = line.xy
        ax.plot(
            x,
            y,
            color=STYLE.contact_edge,
            linewidth=linewidth,
            label=label if index == 0 else "_nolegend_",
            zorder=10,
        )


def draw_indenter_pose(
    ax: Axes,
    pose: Any | None,
    *,
    show_contact_patch: bool = True,
) -> None:
    """Draw an already-solved indenter pose and its exact contact boundary."""
    if pose is None:
        return
    carrier_x, carrier_y = pose.carrier_geometry.exterior.xy
    ax.fill(
        carrier_x,
        carrier_y,
        facecolor=STYLE.indenter_face,
        edgecolor=STYLE.indenter_edge,
        alpha=0.30,
        label="Indenter pose",
        zorder=8,
    )
    if show_contact_patch:
        draw_contact_patch(ax, pose)


def draw_fea(
    ax: Axes,
    mesh: PadMesh | FingertipMesh,
    field: Any,
    *,
    field_name: str = "displacement",
    deformed_mesh: PadMesh | None = None,
    norm: Normalize | None = None,
    cmap: Any = None,
    deformation_scale: float = 1.0,
    show_vectors: bool = True,
    arrow_scale: float = 1.0,
    arrow_minimum_mm: float = 0.0,
    maximum_arrows: int = 80,
    show_rigid_structure: bool = True,
    pose: Any | None = None,
    show_contact_patch: bool = True,
    contact_point: Any | None = None,
    indentation_direction: Any | None = None,
) -> Any:
    """Draw either nodal displacement or element von Mises stress."""
    if field_name not in ("displacement", "von_mises"):
        raise ValueError("field_name must be 'displacement' or 'von_mises'")
    if not np.isfinite(deformation_scale) or deformation_scale <= 0.0:
        raise ValueError("deformation_scale must be finite and positive")
    if not np.isfinite(arrow_scale) or arrow_scale <= 0.0:
        raise ValueError("arrow_scale must be finite and positive")
    if arrow_minimum_mm < 0.0 or not np.isfinite(arrow_minimum_mm):
        raise ValueError("arrow_minimum_mm must be finite and nonnegative")
    if not isinstance(maximum_arrows, (int, np.integer)) or maximum_arrows < 1:
        raise ValueError("maximum_arrows must be a positive integer")

    reference_coordinates, triangles = _mesh_arrays(mesh)
    display_coordinates = reference_coordinates.copy()
    if field_name == "displacement":
        displacement = _validate_displacement(reference_coordinates, field)
        display_coordinates = (
            np.asarray(deformed_mesh.coordinates, dtype=float)
            if deformed_mesh is not None
            else reference_coordinates + deformation_scale * displacement
        )
        if display_coordinates.shape != reference_coordinates.shape:
            raise ValueError("deformed_mesh coordinates must match the reference mesh")
        if show_rigid_structure and isinstance(mesh, FingertipMesh):
            draw_rigid_carrier(ax, mesh)
        magnitude = np.linalg.norm(displacement, axis=1)
        if cmap is None:
            cmap = plt.get_cmap(MECHANICS_CMAP)
        if norm is not None:
            scalar = ax.tripcolor(
                display_coordinates[:, 0],
                display_coordinates[:, 1],
                triangles,
                magnitude,
                shading="gouraud",
                cmap=cmap,
                norm=norm,
                zorder=2,
            )
        else:
            scalar = ax.tripcolor(
                display_coordinates[:, 0],
                display_coordinates[:, 1],
                triangles,
                magnitude,
                shading="gouraud",
                cmap=cmap,
                vmin=0.0,
                vmax=max(float(np.max(magnitude)), 1.0e-15),
                zorder=2,
            )
        draw_mesh(ax, mesh, coordinates=display_coordinates, alpha=0.45)
        if show_vectors:
            selected = _arrow_indices(
                display_coordinates,
                magnitude,
                minimum=arrow_minimum_mm,
                maximum=int(maximum_arrows),
            )
            if len(selected):
                ax.quiver(
                    display_coordinates[selected, 0],
                    display_coordinates[selected, 1],
                    arrow_scale * displacement[selected, 0],
                    arrow_scale * displacement[selected, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=STYLE.mechanics_vectors,
                    width=0.0025,
                    label="Displacement u",
                    zorder=5,
                )
    else:
        stress = _validate_field(field, name="von_mises", size=len(triangles))
        if deformed_mesh is not None:
            display_coordinates = np.asarray(deformed_mesh.coordinates, dtype=float)
        if display_coordinates.shape != reference_coordinates.shape:
            raise ValueError("deformed_mesh coordinates must match the reference mesh")
        if show_rigid_structure and isinstance(mesh, FingertipMesh):
            draw_rigid_carrier(ax, mesh)
        if cmap is None:
            cmap = plt.get_cmap(MECHANICS_CMAP)
        if norm is None:
            norm = Normalize(vmin=0.0, vmax=max(float(np.max(stress)), 1.0e-15))
        scalar = ax.tripcolor(
            display_coordinates[:, 0],
            display_coordinates[:, 1],
            triangles,
            facecolors=stress,
            shading="flat",
            cmap=cmap,
            norm=norm,
            edgecolors=STYLE.mesh_edge,
            linewidth=0.18,
            alpha=0.88,
            zorder=2,
        )
        draw_mesh(ax, mesh, coordinates=display_coordinates, alpha=0.35)
    draw_indenter_pose(ax, pose, show_contact_patch=show_contact_patch)
    if contact_point is not None:
        contact = np.asarray(contact_point, dtype=float)
        if contact.shape != (2,) or not np.all(np.isfinite(contact)):
            raise ValueError("contact_point must be a finite 2-vector")
        ax.scatter(
            [contact[0]],
            [contact[1]],
            marker="x",
            color=STYLE.contact_edge,
            s=48.0,
            label="Contact point",
            zorder=11,
        )
        if indentation_direction is not None:
            direction = np.asarray(indentation_direction, dtype=float)
            if direction.shape != (2,) or not np.all(np.isfinite(direction)):
                raise ValueError("indentation_direction must be a finite 2-vector")
            magnitude = float(np.linalg.norm(direction))
            if magnitude <= 0.0:
                raise ValueError("indentation_direction must be nonzero")
            direction /= magnitude
            length = max(float(np.ptp(display_coordinates, axis=0).max()), 1.0) * 0.12
            ax.quiver(
                [contact[0]],
                [contact[1]],
                [length * direction[0]],
                [length * direction[1]],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=STYLE.contact_edge,
                width=0.003,
                label="Indentation direction",
                zorder=11,
            )
    return scalar


def plot_fea(
    mesh: PadMesh | FingertipMesh,
    field: Any,
    *,
    ax: Axes | None = None,
    field_name: str = "displacement",
    deformed_mesh: PadMesh | None = None,
    norm: Normalize | None = None,
    cmap: Any = None,
    deformation_scale: float = 1.0,
    show_vectors: bool = True,
    arrow_scale: float = 1.0,
    arrow_minimum_mm: float = 0.0,
    maximum_arrows: int = 80,
    show_rigid_structure: bool = True,
    pose: Any | None = None,
    show_contact_patch: bool = True,
    contact_point: Any | None = None,
    indentation_direction: Any | None = None,
    show_colorbar: bool = True,
    show_axes: bool = True,
    show_legend: bool = False,
    title: str | None = None,
) -> Axes:
    """Plot one FEA field with optional externally shared normalization."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 6.0))
    scalar = draw_fea(
        ax,
        mesh,
        field,
        field_name=field_name,
        deformed_mesh=deformed_mesh,
        norm=norm,
        cmap=cmap,
        deformation_scale=deformation_scale,
        show_vectors=show_vectors,
        arrow_scale=arrow_scale,
        arrow_minimum_mm=arrow_minimum_mm,
        maximum_arrows=maximum_arrows,
        show_rigid_structure=show_rigid_structure,
        pose=pose,
        show_contact_patch=show_contact_patch,
        contact_point=contact_point,
        indentation_direction=indentation_direction,
    )
    reference_coordinates, _ = _mesh_arrays(mesh)
    points = [reference_coordinates]
    if deformed_mesh is not None:
        points.append(np.asarray(deformed_mesh.coordinates, dtype=float))
    if isinstance(mesh, FingertipMesh) and show_rigid_structure:
        points.append(_carrier_arrays(mesh)[0])
    if pose is not None:
        x, y = pose.carrier_geometry.exterior.xy
        points.append(np.column_stack((x, y)))
    apply_physical_axes(ax, bounds_from_points(*points), show_axes=show_axes)
    ax.set_title(title or ("Nodal displacement" if field_name == "displacement" else "Von Mises stress"))
    if show_colorbar:
        label = "Displacement magnitude |u| [mm]" if field_name == "displacement" else "Cauchy von Mises stress [MPa]"
        ax.figure.colorbar(scalar, ax=ax, fraction=0.046, pad=0.04).set_label(label)
    if show_legend:
        ax.legend(loc="upper center", fontsize=8, ncol=2)
    return ax


__all__ = ["draw_fea", "draw_mesh", "plot_fea"]
