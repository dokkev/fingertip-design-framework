"""Solver-neutral 3D fingertip mesh and state visualization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from mesh import FingertipVolumeMesh, FingertipVolumeState
from visualization._style import STYLE


_DEFAULT_ELEVATION = 24.0
_DEFAULT_AZIMUTH = -60.0
_AXIS_MARGIN_FRACTION = 0.05
_MINIMUM_SPAN_MM = 1.0e-6
_DISPLACEMENT_CMAP = "viridis"
_SEMANTIC_COLORS = {
    "outer_compliant": STYLE.silicone_face,
    "void": STYLE.void_face,
    "support": STYLE.bonded_interface_face,
    "longitudinal_end": "#AEB8C2",
    "contact": STYLE.contact_edge,
}


def _surface_definitions(volume_mesh: FingertipVolumeMesh) -> dict[str, Any]:
    definitions = {
        definition.name: definition for definition in volume_mesh.solid.surfaces
    }
    unknown = set(volume_mesh.surface_triangles) - set(definitions)
    if unknown:
        raise ValueError(
            "volume mesh contains semantic surface families absent from the solid: "
            + repr(sorted(unknown))
        )
    return definitions


def _selected_surface_tags(
    volume_mesh: FingertipVolumeMesh,
    surface_tags: Iterable[str] | None,
    *,
    show_support: bool,
) -> tuple[str, ...]:
    definitions = _surface_definitions(volume_mesh)
    available = tuple(sorted(volume_mesh.surface_triangles))
    if surface_tags is None:
        selected = tuple(
            tag
            for tag in available
            if show_support or definitions[tag].kind != "support"
        )
    else:
        if isinstance(surface_tags, str):
            raise TypeError("surface_tags must be an iterable of tag strings, not a string")
        selected_list: list[str] = []
        for tag in surface_tags:
            if not isinstance(tag, str) or not tag:
                raise ValueError("surface_tags must contain non-empty strings")
            if tag not in volume_mesh.surface_triangles:
                raise KeyError(f"unknown volume surface tag: {tag!r}")
            if tag not in selected_list:
                selected_list.append(tag)
        selected = tuple(selected_list)
    if not selected:
        raise ValueError("surface_tags selected no semantic surface families")
    return selected


def _canonical_coordinates(volume_mesh: FingertipVolumeMesh) -> tuple[tuple[int, ...], np.ndarray]:
    node_ids = tuple(sorted(volume_mesh.nodes))
    coordinates = np.asarray(
        [
            [
                volume_mesh.nodes[node_id].x_mm,
                volume_mesh.nodes[node_id].y_mm,
                volume_mesh.nodes[node_id].z_mm,
            ]
            for node_id in node_ids
        ],
        dtype=float,
    )
    if coordinates.shape != (len(node_ids), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("volume mesh coordinates must be finite with shape (N, 3)")
    return node_ids, coordinates


def _surface_arrays(
    volume_mesh: FingertipVolumeMesh,
    surface_tags: tuple[str, ...],
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    node_ids, _ = _canonical_coordinates(volume_mesh)
    canonical_index = {node_id: index for index, node_id in enumerate(node_ids)}
    surface_node_ids = tuple(
        sorted(
            {
                int(node_id)
                for tag in surface_tags
                for triangle in volume_mesh.surface_triangles[tag]
                for node_id in triangle.node_ids
            }
        )
    )
    surface_index = {node_id: index for index, node_id in enumerate(surface_node_ids)}
    faces: list[list[int]] = []
    tags: list[str] = []
    for tag in surface_tags:
        for triangle in volume_mesh.surface_triangles[tag]:
            if triangle.semantic_tag != tag:
                raise ValueError(f"surface triangle semantic tag mismatch for {tag!r}")
            if any(node_id not in canonical_index for node_id in triangle.node_ids):
                raise ValueError(f"surface {tag!r} references an unknown canonical node")
            faces.append([surface_index[int(node_id)] for node_id in triangle.node_ids])
            tags.append(tag)
    face_array = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(
        [coordinates[canonical_index[node_id]] for node_id in surface_node_ids],
        dtype=float,
    )
    if face_array.ndim != 2 or face_array.shape[1] != 3 or not len(face_array):
        raise ValueError("selected volume surfaces must contain triangle faces")
    return vertices, face_array, tuple(tags)


def _new_3d_axes(ax: Axes | None) -> Axes:
    if ax is not None:
        if not hasattr(ax, "add_collection3d") or not hasattr(ax, "set_box_aspect"):
            raise TypeError("ax must be a Matplotlib 3D axes")
        return ax
    figure = plt.figure(figsize=(8.0, 6.5))
    return figure.add_subplot(111, projection="3d")


def _surface_colors(
    tags: tuple[str, ...],
    definitions: dict[str, Any],
) -> list[str]:
    return [
        _SEMANTIC_COLORS.get(definitions[tag].kind, STYLE.mesh_edge)
        for tag in tags
    ]


def _draw_surface(
    ax: Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    tags: tuple[str, ...],
    definitions: dict[str, Any],
    *,
    facecolors: Any,
    show_edges: bool,
    label: str,
) -> Poly3DCollection:
    polygons = [vertices[face] for face in faces]
    edgecolors = [
        STYLE.bonded_interface_edge
        if definitions[tag].kind == "support"
        else STYLE.mesh_edge
        for tag in tags
    ]
    collection = Poly3DCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=edgecolors if show_edges else "none",
        linewidths=0.35 if show_edges else 0.0,
        alpha=0.82,
        label=label,
    )
    ax.add_collection3d(collection)
    return collection


def _draw_reference_wireframe(
    ax: Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Line3DCollection:
    segments = np.asarray(
        [
            [vertices[first], vertices[second]]
            for face in faces
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        ],
        dtype=float,
    )
    collection = Line3DCollection(
        segments,
        colors=STYLE.rigid_edge,
        linewidths=0.45,
        alpha=0.35,
        label="Reference surface",
    )
    ax.add_collection3d(collection)
    return collection


def _apply_3d_axes(
    ax: Axes,
    point_sets: Iterable[np.ndarray],
    *,
    elev: float,
    azim: float,
    title: str,
) -> None:
    arrays = [np.asarray(points, dtype=float) for points in point_sets if len(points)]
    if not arrays or any(array.ndim != 2 or array.shape[1] != 3 for array in arrays):
        raise ValueError("3D display geometry must contain finite point arrays of shape (N, 3)")
    points = np.vstack(arrays)
    if not np.all(np.isfinite(points)):
        raise ValueError("3D display geometry must be finite")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    spans = maximum - minimum
    span = max(float(np.max(spans)), _MINIMUM_SPAN_MM)
    margins = np.maximum(_AXIS_MARGIN_FRACTION * span, _MINIMUM_SPAN_MM)
    lower = minimum - margins
    upper = maximum + margins
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_box_aspect(tuple(float(value) for value in upper - lower))
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(title)


def _validate_view(elev: float, azim: float) -> tuple[float, float]:
    values = (float(elev), float(azim))
    if not np.all(np.isfinite(values)):
        raise ValueError("elev and azim must be finite")
    return values


def _draw_nodes(
    ax: Axes,
    coordinates: np.ndarray,
    *,
    indices: np.ndarray | None = None,
    label: str,
    color: str,
    size: float,
) -> Any:
    selected = coordinates if indices is None else coordinates[indices]
    return ax.scatter(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        s=size,
        color=color,
        depthshade=False,
        label=label,
    )


def draw_volume_mesh(
    ax: Axes,
    volume_mesh: FingertipVolumeMesh,
    *,
    surface_tags: Iterable[str] | None = None,
    show_edges: bool = True,
    show_nodes: bool = False,
    show_support: bool = False,
) -> None:
    """Draw selected semantic surfaces of a canonical undeformed volume mesh."""
    if not isinstance(volume_mesh, FingertipVolumeMesh):
        raise TypeError("volume_mesh must be FingertipVolumeMesh")
    tags = _selected_surface_tags(volume_mesh, surface_tags, show_support=show_support)
    definitions = _surface_definitions(volume_mesh)
    _, coordinates = _canonical_coordinates(volume_mesh)
    vertices, faces, face_tags = _surface_arrays(volume_mesh, tags, coordinates)
    _draw_surface(
        ax,
        vertices,
        faces,
        face_tags,
        definitions,
        facecolors=_surface_colors(face_tags, definitions),
        show_edges=show_edges,
        label="Semantic volume surface",
    )
    if show_nodes:
        _draw_nodes(
            ax,
            coordinates,
            label="Volume nodes",
            color=STYLE.node_face,
            size=5.0,
        )


def _validate_state_field(field: str) -> None:
    if field not in {"displacement", "semantic"}:
        raise ValueError("field must be 'displacement' or 'semantic'")


def _state_face_values(state: FingertipVolumeState, face_tags: tuple[str, ...], faces: np.ndarray) -> np.ndarray:
    displacement_magnitude = np.linalg.norm(state.displacement_mm, axis=1)
    node_ids = tuple(sorted(state.volume_mesh.nodes))
    canonical_index = {node_id: index for index, node_id in enumerate(node_ids)}
    surface_node_ids = tuple(
        sorted(
            {
                int(node_id)
                for tag in face_tags
                for triangle in state.surface_triangles[tag]
                for node_id in triangle.node_ids
            }
        )
    )
    values: list[float] = []
    for tag, face in zip(face_tags, faces):
        # The selected semantic rows are emitted in the same deterministic
        # order as _surface_arrays; use the face's source-local nodes only for
        # the nodal scalar lookup, never to define geometry coordinates.
        source_ids = tuple(surface_node_ids[int(index)] for index in face)
        values.append(
            float(np.mean([displacement_magnitude[canonical_index[node_id]] for node_id in source_ids]))
        )
    return np.asarray(values, dtype=float)


def draw_volume_state(
    ax: Axes,
    state: FingertipVolumeState,
    *,
    field: str = "displacement",
    surface_tags: Iterable[str] | None = None,
    show_reference: bool = True,
    deformation_scale: float = 1.0,
    show_edges: bool = True,
    show_support: bool = False,
    highlight_vertex_indices: Iterable[int] | None = None,
    norm: Normalize | None = None,
    cmap: Any = None,
) -> Poly3DCollection | None:
    """Draw one neutral deformed volume state without solver inspection."""
    if not isinstance(state, FingertipVolumeState):
        raise TypeError("state must be FingertipVolumeState")
    _validate_state_field(field)
    if not np.isfinite(deformation_scale) or deformation_scale <= 0.0:
        raise ValueError("deformation_scale must be finite and positive")
    tags = _selected_surface_tags(state.volume_mesh, surface_tags, show_support=show_support)
    definitions = _surface_definitions(state.volume_mesh)
    node_ids, reference_coordinates = _canonical_coordinates(state.volume_mesh)
    display_coordinates = reference_coordinates + deformation_scale * state.displacement_mm
    vertices, faces, face_tags = _surface_arrays(
        state.volume_mesh,
        tags,
        display_coordinates,
    )
    if field == "displacement":
        face_values = _state_face_values(state, face_tags, faces)
        if not np.all(np.isfinite(face_values)) or np.any(face_values < 0.0):
            raise ValueError("displacement face values must be finite and nonnegative")
        if cmap is None:
            cmap = plt.get_cmap(_DISPLACEMENT_CMAP)
        if norm is None:
            norm = Normalize(vmin=0.0, vmax=max(float(np.max(face_values)), 1.0e-15))
        facecolors = cmap(norm(face_values))
        collection = _draw_surface(
            ax,
            vertices,
            faces,
            face_tags,
            definitions,
            facecolors=facecolors,
            show_edges=show_edges,
            label="Deformed volume surface",
        )
        collection.set_array(face_values)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
    else:
        _draw_surface(
            ax,
            vertices,
            faces,
            face_tags,
            definitions,
            facecolors=_surface_colors(face_tags, definitions),
            show_edges=show_edges,
            label="Semantic deformed surface",
        )

    if show_reference:
        reference_vertices, reference_faces, _ = _surface_arrays(
            state.volume_mesh,
            tags,
            reference_coordinates,
        )
        _draw_reference_wireframe(ax, reference_vertices, reference_faces)
    if highlight_vertex_indices is not None:
        raw_indices = np.asarray(tuple(highlight_vertex_indices))
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("highlight_vertex_indices must contain integer canonical local indices")
        indices = np.asarray(raw_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(node_ids)):
            raise ValueError("highlight_vertex_indices contain an out-of-range canonical index")
        _draw_nodes(
            ax,
            display_coordinates,
            indices=indices,
            label="Highlighted vertices",
            color=STYLE.contact_edge,
            size=24.0,
        )
    return collection if field == "displacement" else None


def plot_volume_mesh(
    volume_mesh: FingertipVolumeMesh,
    *,
    surface_tags: Iterable[str] | None = None,
    show_edges: bool = True,
    show_nodes: bool = False,
    show_support: bool = False,
    ax: Axes | None = None,
    elev: float = _DEFAULT_ELEVATION,
    azim: float = _DEFAULT_AZIMUTH,
    title: str | None = None,
) -> Axes:
    """Plot the semantic exterior shell of one canonical volume mesh."""
    axis = _new_3d_axes(ax)
    elev, azim = _validate_view(elev, azim)
    draw_volume_mesh(
        axis,
        volume_mesh,
        surface_tags=surface_tags,
        show_edges=show_edges,
        show_nodes=show_nodes,
        show_support=show_support,
    )
    _, coordinates = _canonical_coordinates(volume_mesh)
    _apply_3d_axes(
        axis,
        (coordinates,),
        elev=elev,
        azim=azim,
        title=title or "Fingertip volume mesh",
    )
    return axis


def plot_volume_state(
    state: FingertipVolumeState,
    *,
    field: str = "displacement",
    surface_tags: Iterable[str] | None = None,
    show_reference: bool = True,
    deformation_scale: float = 1.0,
    show_edges: bool = True,
    show_support: bool = False,
    highlight_vertex_indices: Iterable[int] | None = None,
    norm: Normalize | None = None,
    cmap: Any = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    elev: float = _DEFAULT_ELEVATION,
    azim: float = _DEFAULT_AZIMUTH,
    title: str | None = None,
) -> Axes:
    """Plot one canonical deformed volume state with optional reference overlay."""
    axis = _new_3d_axes(ax)
    elev, azim = _validate_view(elev, azim)
    collection = draw_volume_state(
        axis,
        state,
        field=field,
        surface_tags=surface_tags,
        show_reference=show_reference,
        deformation_scale=deformation_scale,
        show_edges=show_edges,
        show_support=show_support,
        highlight_vertex_indices=highlight_vertex_indices,
        norm=norm,
        cmap=cmap,
    )
    _, reference_coordinates = _canonical_coordinates(state.volume_mesh)
    display_coordinates = reference_coordinates + deformation_scale * state.displacement_mm
    _apply_3d_axes(
        axis,
        (reference_coordinates, display_coordinates),
        elev=elev,
        azim=azim,
        title=title or "Fingertip volume state",
    )
    if colorbar and field == "displacement":
        if collection is None:
            raise RuntimeError("displacement plotting did not create a scalar mappable")
        colorbar_artist = axis.figure.colorbar(collection, ax=axis, pad=0.08)
        colorbar_artist.set_label("displacement magnitude [mm]")
    return axis


__all__ = [
    "draw_volume_mesh",
    "draw_volume_state",
    "plot_volume_mesh",
    "plot_volume_state",
]
