"""Solver-independent Phase 4I indentation measurements and comparisons."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point

from fem.indenter_fixture import CrownFrame, Vector2
from fem.mesh_types import BoundaryEdge, FingertipMesh
from model.fingertip_model import FingertipModel


class IndentationPostprocessError(RuntimeError):
    """Raised when a requested measurement has no valid geometric support."""


def _normalized(vector: Vector2) -> Vector2:
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 0.0:
        raise IndentationPostprocessError("cannot normalize a zero-length vector")
    return vector[0] / length, vector[1] / length


def unique_projected_reaction(
    reactions: Mapping[int, Sequence[float]],
    node_ids: Iterable[int],
    direction: Vector2,
) -> float:
    """Sum each node once and project its reaction onto ``direction``."""
    unit_direction = _normalized(direction)
    total = 0.0
    for node_id in sorted(set(node_ids)):
        reaction = reactions[node_id]
        total += float(reaction[0]) * unit_direction[0] + float(reaction[1]) * unit_direction[1]
    return total


def compressive_indenter_reaction(
    reactions: Mapping[int, Sequence[float]],
    node_ids: Iterable[int],
    loading_direction: Vector2,
) -> float:
    """Return positive actuator compression in Kratos' REACTION convention.

    For the prescribed indenter DOFs Kratos 10.3 reports ``REACTION`` along
    the imposed loading direction, while the fixed support reaction has the
    opposite sign.  Both signed projections remain available to the force
    equilibrium calculation; this helper only names the positive indenter
    magnitude used by the load curve.
    """
    return unique_projected_reaction(
        reactions, node_ids, loading_direction
    )


def relative_force_equilibrium_error(
    indenter_signed_reaction: float,
    support_signed_reaction: float,
    force_floor_n: float,
) -> float:
    """Return the normalized residual of the signed constrained reactions."""
    if not math.isfinite(force_floor_n) or force_floor_n <= 0.0:
        raise ValueError("force_floor_n must be finite and positive")
    return abs(indenter_signed_reaction + support_signed_reaction) / max(
        abs(indenter_signed_reaction), force_floor_n
    )


def ordered_boundary_node_ids(
    mesh: FingertipMesh,
    boundary_tag: str,
    source_geometry: LineString,
) -> tuple[int, ...]:
    """Order an open Line2 chain according to the source Shapely line."""
    edges = mesh.boundary_edges[boundary_tag]
    if not edges:
        raise IndentationPostprocessError(f"boundary {boundary_tag!r} is empty")
    adjacency: dict[int, list[int]] = {}
    for edge in edges:
        first, second = edge.node_ids
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    endpoints = [node_id for node_id, neighbours in adjacency.items() if len(neighbours) == 1]
    if len(endpoints) != 2 or any(len(neighbours) > 2 for neighbours in adjacency.values()):
        raise IndentationPostprocessError(
            f"boundary {boundary_tag!r} is not one connected open Line2 chain"
        )
    source_start = Point(source_geometry.coords[0])
    start = min(
        endpoints,
        key=lambda node_id: source_start.distance(
            Point(mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm)
        ),
    )
    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = [node_id for node_id in adjacency[current] if node_id != previous]
        if not candidates:
            break
        if len(candidates) != 1:
            raise IndentationPostprocessError(
                f"boundary {boundary_tag!r} branches at node {current}"
            )
        previous, current = current, candidates[0]
        ordered.append(current)
    if len(ordered) != len(adjacency):
        raise IndentationPostprocessError(
            f"boundary {boundary_tag!r} contains disconnected edges"
        )
    return tuple(ordered)


def contact_width_metrics(
    mesh: FingertipMesh,
    active_node_ids: Iterable[int],
    crown_tangent: Vector2,
) -> dict[str, Any]:
    """Measure active chord width and fully active source-edge arc length."""
    active = set(active_node_ids)
    pad_ids = {
        node_id
        for edge in mesh.boundary_edges["pad_outer_arc"]
        for node_id in edge.node_ids
    }
    active.intersection_update(pad_ids)
    tangent = _normalized(crown_tangent)
    projections = [
        mesh.nodes[node_id].x_mm * tangent[0]
        + mesh.nodes[node_id].y_mm * tangent[1]
        for node_id in active
    ]
    chord = max(projections) - min(projections) if len(projections) >= 2 else 0.0
    arc_length = 0.0
    active_edge_count = 0
    for edge in mesh.boundary_edges["pad_outer_arc"]:
        if set(edge.node_ids).issubset(active):
            first, second = (mesh.nodes[node_id] for node_id in edge.node_ids)
            arc_length += math.hypot(second.x_mm - first.x_mm, second.y_mm - first.y_mm)
            active_edge_count += 1
    return {
        "active_node_count": len(active),
        "active_edge_count": active_edge_count,
        "chord_width_mm": chord,
        "arc_length_mm": arc_length,
        "active_definition": "PadOuterArc node ACTIVE flag",
        "active_edge_definition": "source edge with both endpoint nodes ACTIVE",
    }


def _outward_normal_at_boundary_node(
    model: FingertipModel,
    point: Vector2,
    tangent: Vector2,
) -> Vector2:
    candidates = ((-tangent[1], tangent[0]), (tangent[1], -tangent[0]))
    probe = max(1.0e-4, 1000.0 * model.parameters.geometry_tolerance)
    outside = [
        candidate
        for candidate in candidates
        if not model.pad_material_geometry.covers(
            Point(point[0] + probe * candidate[0], point[1] + probe * candidate[1])
        )
    ]
    if len(outside) == 1:
        return _normalized(outside[0])
    # The two diameter endpoints are geometric corners where both infinitesimal
    # tangent-normal probes can lie outside.  Resolve only that endpoint case
    # with the vector from a guaranteed interior representative point.
    interior = model.pad_material_geometry.representative_point()
    radial = (point[0] - interior.x, point[1] - interior.y)
    selected = max(
        candidates,
        key=lambda candidate: candidate[0] * radial[0] + candidate[1] * radial[1],
    )
    if selected[0] * radial[0] + selected[1] * radial[1] <= 0.0:
        raise IndentationPostprocessError("pad normal is ambiguous on PadOuterArc")
    return _normalized(selected)


def extract_outer_arc_profile(
    model: FingertipModel,
    mesh: FingertipMesh,
    displacements: Mapping[int, Sequence[float]],
    crown_frame: CrownFrame,
) -> list[dict[str, Any]]:
    """Extract the complete ordered PadOuterArc displacement profile."""
    ordered_ids = ordered_boundary_node_ids(
        mesh,
        "pad_outer_arc",
        model.boundaries.segments["pad_outer_arc"].geometry,
    )
    points = [(mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm) for node_id in ordered_ids]
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    if cumulative[-1] <= 0.0:
        raise IndentationPostprocessError("PadOuterArc has zero length")
    profile: list[dict[str, Any]] = []
    for index, (node_id, point) in enumerate(zip(ordered_ids, points)):
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        tangent = _normalized((after[0] - before[0], after[1] - before[1]))
        outward = _outward_normal_at_boundary_node(model, point, tangent)
        displacement = displacements[node_id]
        ux, uy = float(displacement[0]), float(displacement[1])
        tangent_coordinate = (
            (point[0] - crown_frame.point_mm[0]) * crown_frame.tangent[0]
            + (point[1] - crown_frame.point_mm[1]) * crown_frame.tangent[1]
        )
        profile.append(
            {
                "node_id": node_id,
                "reference_x_mm": point[0],
                "reference_y_mm": point[1],
                "normalized_arc_coordinate": cumulative[index] / cumulative[-1],
                "tangent_coordinate_from_crown_mm": tangent_coordinate,
                "side": (
                    "crown"
                    if abs(tangent_coordinate) <= mesh.settings.classification_tolerance_mm
                    else ("left" if point[0] < crown_frame.point_mm[0] else "right")
                ),
                "ux_mm": ux,
                "uy_mm": uy,
                "local_normal_displacement_mm": ux * outward[0] + uy * outward[1],
                "local_tangential_displacement_mm": ux * tangent[0] + uy * tangent[1],
                "deformed_x_mm": point[0] + ux,
                "deformed_y_mm": point[1] + uy,
                "reference_tangent": list(tangent),
                "reference_outward_normal": list(outward),
            }
        )
    return profile


def interpolate_profile(
    profile: Sequence[Mapping[str, Any]],
    common_coordinates: Sequence[float],
) -> dict[str, list[float]]:
    """Interpolate normal/tangential profile fields on a common arc grid."""
    source = np.asarray(
        [float(record["normalized_arc_coordinate"]) for record in profile]
    )
    common = np.asarray(common_coordinates, dtype=float)
    if source.size < 2 or np.any(np.diff(source) <= 0.0):
        raise IndentationPostprocessError("profile arc coordinates must increase")
    if np.any(common < 0.0) or np.any(common > 1.0) or np.any(np.diff(common) < 0.0):
        raise IndentationPostprocessError("common arc coordinates must be sorted in [0, 1]")
    result = {"normalized_arc_coordinate": common.tolist()}
    for output_name, source_name in (
        ("normal_displacement_mm", "local_normal_displacement_mm"),
        ("tangential_displacement_mm", "local_tangential_displacement_mm"),
    ):
        values = np.asarray([float(record[source_name]) for record in profile])
        result[output_name] = np.interp(common, source, values).tolist()
    return result


def profile_error_metrics(
    first: Sequence[float],
    reference: Sequence[float],
    absolute_displacement_floor_mm: float,
) -> dict[str, float]:
    """Return relative L2 and maximum absolute errors with an explicit floor."""
    first_values = np.asarray(first, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if first_values.shape != reference_values.shape or first_values.size == 0:
        raise IndentationPostprocessError("profile arrays must have one equal nonzero shape")
    if not math.isfinite(absolute_displacement_floor_mm) or absolute_displacement_floor_mm <= 0.0:
        raise ValueError("absolute_displacement_floor_mm must be finite and positive")
    difference = first_values - reference_values
    denominator = max(
        float(np.linalg.norm(reference_values)),
        absolute_displacement_floor_mm * math.sqrt(reference_values.size),
    )
    return {
        "relative_l2_error": float(np.linalg.norm(difference)) / denominator,
        "maximum_absolute_error_mm": float(np.max(np.abs(difference))),
        "absolute_displacement_floor_mm": absolute_displacement_floor_mm,
    }


def _deformation_gradient(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    reference_edges = np.column_stack((reference[1] - reference[0], reference[2] - reference[0]))
    determinant = float(np.linalg.det(reference_edges))
    if abs(determinant) <= 1.0e-15:
        raise IndentationPostprocessError("pad T3 has zero reference area")
    current_edges = np.column_stack((current[1] - current[0], current[2] - current[0]))
    return current_edges @ np.linalg.inv(reference_edges)


def pad_strain_det_f_statistics(
    mesh: FingertipMesh,
    displacements: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    """Measure affine-T3 Green--Lagrange strain and det(F) on pad elements."""
    determinant_records: list[tuple[float, int, Vector2]] = []
    principal_records: list[tuple[float, int, Vector2]] = []
    component_records: list[tuple[float, int, Vector2]] = []
    for element in mesh.pad_elements:
        reference = np.asarray(
            [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm] for node_id in element.node_ids],
            dtype=float,
        )
        current = reference + np.asarray(
            [[float(displacements[node_id][0]), float(displacements[node_id][1])] for node_id in element.node_ids],
            dtype=float,
        )
        deformation_gradient = _deformation_gradient(reference, current)
        determinant_f = float(np.linalg.det(deformation_gradient))
        green_lagrange = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(2))
        maximum_principal = float(np.max(np.linalg.eigvalsh(green_lagrange)))
        maximum_component = float(np.max(np.abs(green_lagrange)))
        centroid = (float(np.mean(reference[:, 0])), float(np.mean(reference[:, 1])))
        determinant_records.append((determinant_f, element.id, centroid))
        principal_records.append((maximum_principal, element.id, centroid))
        component_records.append((maximum_component, element.id, centroid))
    if not determinant_records:
        raise IndentationPostprocessError("mesh has no pad T3 elements")
    minimum_det = min(determinant_records, key=lambda item: item[0])
    maximum_det = max(determinant_records, key=lambda item: item[0])
    maximum_principal = max(principal_records, key=lambda item: item[0])
    maximum_component = max(component_records, key=lambda item: item[0])
    values = [record[0] for record in determinant_records]
    return {
        "source": "affine T3 nodal kinematics at the element integration point",
        "rigid_domains_excluded": True,
        "pad_element_count": len(determinant_records),
        "all_finite": all(math.isfinite(value) for value in values)
        and math.isfinite(maximum_principal[0])
        and math.isfinite(maximum_component[0]),
        "det_f": {
            "min": minimum_det[0],
            "max": maximum_det[0],
            "nonpositive_count": sum(value <= 0.0 for value in values),
            "minimum_element_id": minimum_det[1],
            "minimum_reference_coordinate_mm": list(minimum_det[2]),
            "maximum_element_id": maximum_det[1],
            "maximum_reference_coordinate_mm": list(maximum_det[2]),
        },
        "maximum_principal_green_lagrange_strain": {
            "value": maximum_principal[0],
            "element_id": maximum_principal[1],
            "reference_coordinate_mm": list(maximum_principal[2]),
        },
        "maximum_absolute_green_lagrange_component": {
            "value": maximum_component[0],
            "element_id": maximum_component[1],
            "reference_coordinate_mm": list(maximum_component[2]),
        },
    }


def unstructured_volumetric_oscillation(
    mesh: FingertipMesh,
    values_by_node_id: Mapping[int, float],
) -> dict[str, Any]:
    """Detect mesh-scale residuals against each pad node's neighbour mean."""
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in mesh.domain_node_ids["pad"]}
    for element in mesh.pad_elements:
        for node_id in element.node_ids:
            adjacency[node_id].update(other for other in element.node_ids if other != node_id)
    values = [float(values_by_node_id[node_id]) for node_id in adjacency]
    residuals = [
        float(values_by_node_id[node_id])
        - sum(float(values_by_node_id[neighbour]) for neighbour in neighbours) / len(neighbours)
        for node_id, neighbours in adjacency.items()
        if neighbours
    ]
    mean = sum(values) / len(values)
    centered_rms = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    residual_rms = math.sqrt(sum(value * value for value in residuals) / max(len(residuals), 1))
    ratio = residual_rms / max(centered_rms, 1.0e-15)
    return {
        "method": "unstructured nodal neighbour-mean residual",
        "neighbor_residual_rms_ratio": ratio,
        "limit": 1.0,
        "pass": all(math.isfinite(value) for value in values) and ratio <= 1.0,
    }


def signed_geometric_gap_statistics(
    slave_positions: Mapping[int, Vector2],
    slave_normals: Mapping[int, Vector2],
    master_edges: Sequence[BoundaryEdge] | Sequence[Any],
    master_positions: Mapping[int, Vector2],
) -> dict[str, Any]:
    """Project closest-master vectors on the slave outward normal."""
    lines = [
        LineString([master_positions[node_id] for node_id in edge.node_ids])
        for edge in master_edges
    ]
    if not lines or not slave_positions:
        return {
            "available": False,
            "reason": "empty slave nodes or master edges",
            "maximum_penetration_mm": None,
        }
    gaps: list[float] = []
    for node_id, position in slave_positions.items():
        point = Point(position)
        line = min(lines, key=point.distance)
        nearest_distance = line.project(point)
        nearest = line.interpolate(nearest_distance)
        normal = _normalized(slave_normals[node_id])
        gaps.append(
            (nearest.x - position[0]) * normal[0]
            + (nearest.y - position[1]) * normal[1]
        )
    return {
        "available": True,
        "method": "closest master point projected on runtime slave normal",
        "count": len(gaps),
        "min_signed_gap_mm": min(gaps),
        "max_signed_gap_mm": max(gaps),
        "mean_signed_gap_mm": sum(gaps) / len(gaps),
        "maximum_penetration_mm": max(0.0, -min(gaps)),
        "finite": all(math.isfinite(value) for value in gaps),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_history_csv(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "step",
        "pseudo_time",
        "prescribed_indenter_travel_mm",
        "achieved_indentation_mm",
        "indenter_normal_reaction_n",
        "support_signed_reaction_along_loading_n",
        "force_equilibrium_error",
        "nonlinear_iterations",
        "solver_converged",
        "active_set_converged",
        "external_active_condition_count",
        "internal_left_active_condition_count",
        "internal_right_active_condition_count",
        "internal_bottom_active_condition_count",
        "external_weighted_gap_min",
        "external_weighted_gap_mean",
        "external_contact_chord_width_mm",
        "external_contact_arc_length_mm",
        "maximum_principal_green_lagrange_strain",
        "minimum_pad_det_f",
        "maximum_pad_displacement_mm",
        "maximum_contact_penetration_mm",
        "solve_wall_clock_seconds",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in history:
            groups = point["contact_groups"]
            external = groups["external_pad_indenter"]
            writer.writerow(
                {
                    "step": point["step"],
                    "pseudo_time": point["pseudo_time"],
                    "prescribed_indenter_travel_mm": point[
                        "prescribed_indenter_travel_mm"
                    ],
                    "achieved_indentation_mm": point["achieved_indentation_mm"],
                    "indenter_normal_reaction_n": point[
                        "indenter_normal_reaction_n"
                    ],
                    "support_signed_reaction_along_loading_n": point[
                        "support_signed_reaction_along_loading_n"
                    ],
                    "force_equilibrium_error": point["force_equilibrium_error"],
                    "nonlinear_iterations": point["nonlinear_iterations"],
                    "solver_converged": point["solver_converged"],
                    "active_set_converged": point["active_set_converged"],
                    "external_active_condition_count": external[
                        "active_condition_count"
                    ],
                    "internal_left_active_condition_count": groups[
                        "internal_left"
                    ]["active_condition_count"],
                    "internal_right_active_condition_count": groups[
                        "internal_right"
                    ]["active_condition_count"],
                    "internal_bottom_active_condition_count": groups[
                        "internal_bottom"
                    ]["active_condition_count"],
                    "external_weighted_gap_min": external["weighted_gap"]["min"],
                    "external_weighted_gap_mean": external["weighted_gap"]["mean"],
                    "external_contact_chord_width_mm": point[
                        "external_contact_width"
                    ]["chord_width_mm"],
                    "external_contact_arc_length_mm": point[
                        "external_contact_width"
                    ]["arc_length_mm"],
                    "maximum_principal_green_lagrange_strain": point[
                        "pad_strain_det_f"
                    ]["maximum_principal_green_lagrange_strain"]["value"],
                    "minimum_pad_det_f": point["pad_strain_det_f"]["det_f"]["min"],
                    "maximum_pad_displacement_mm": point[
                        "maximum_pad_displacement_mm"
                    ],
                    "maximum_contact_penetration_mm": max(
                        float(group["signed_geometric_gap"].get("maximum_penetration_mm") or 0.0)
                        for group in groups.values()
                    ),
                    "solve_wall_clock_seconds": point["solve_wall_clock_seconds"],
                }
            )


def _write_profile_csv(path: Path, profile: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "node_id",
        "reference_x_mm",
        "reference_y_mm",
        "normalized_arc_coordinate",
        "tangent_coordinate_from_crown_mm",
        "side",
        "ux_mm",
        "uy_mm",
        "local_normal_displacement_mm",
        "local_tangential_displacement_mm",
        "deformed_x_mm",
        "deformed_y_mm",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in profile:
            writer.writerow({field: record[field] for field in fields})


def _save_history_plots(
    result: Mapping[str, Any],
    plots_directory: Path,
) -> None:
    import matplotlib.pyplot as plt

    history = result.get("history", [])
    if not history:
        return
    plots_directory.mkdir(parents=True, exist_ok=True)
    indentation = [point["achieved_indentation_mm"] for point in history]

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        indentation,
        [point["indenter_normal_reaction_n"] for point in history],
        marker="o",
        markersize=2.5,
    )
    axis.set(xlabel="Indentation [mm]", ylabel="Normal reaction [N]", title="Reaction–indentation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_directory / "reaction_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        indentation,
        [point["external_contact_width"]["chord_width_mm"] for point in history],
        label="Chord width",
    )
    axis.plot(
        indentation,
        [point["external_contact_width"]["arc_length_mm"] for point in history],
        label="Arc length",
    )
    axis.set(xlabel="Indentation [mm]", ylabel="Contact extent [mm]", title="External contact width")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_directory / "contact_width.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for group_name in (
        "external_pad_indenter",
        "internal_left",
        "internal_right",
        "internal_bottom",
    ):
        axis.plot(
            indentation,
            [point["contact_groups"][group_name]["active_condition_count"] for point in history],
            label=group_name,
        )
    axis.set(xlabel="Indentation [mm]", ylabel="ACTIVE generated conditions", title="Contact groups")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_directory / "contact_groups.png", dpi=180)
    plt.close(figure)

    figure, first_axis = plt.subplots(figsize=(6.8, 4.4))
    second_axis = first_axis.twinx()
    first_axis.plot(
        indentation,
        [point["pad_strain_det_f"]["maximum_principal_green_lagrange_strain"]["value"] for point in history],
        color="#B2182B",
        label="Maximum principal strain",
    )
    second_axis.plot(
        indentation,
        [point["pad_strain_det_f"]["det_f"]["min"] for point in history],
        color="#2166AC",
        label="Minimum det(F)",
    )
    first_axis.set(xlabel="Indentation [mm]", ylabel="Green–Lagrange strain", title="Pad strain and det(F)")
    second_axis.set_ylabel("Minimum det(F)")
    first_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_directory / "strain_detf.png", dpi=180)
    plt.close(figure)


def _save_outer_profile_plot(snapshots: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not snapshots:
        return
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    for key, snapshot in sorted(snapshots.items(), key=lambda item: float(item[0])):
        profile = snapshot["profile"]
        coordinate = [record["normalized_arc_coordinate"] for record in profile]
        axes[0].plot(
            coordinate,
            [record["local_normal_displacement_mm"] for record in profile],
            label=f"{float(key):g} mm",
        )
        axes[1].plot(
            coordinate,
            [record["local_tangential_displacement_mm"] for record in profile],
            label=f"{float(key):g} mm",
        )
    axes[0].set_ylabel("Normal displacement [mm]")
    axes[1].set_ylabel("Tangential displacement [mm]")
    axes[1].set_xlabel("Normalized reference PadOuterArc coordinate")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Pad outer-arc displacement profiles")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_deformed_mesh_plot(
    artifacts: Any,
    snapshot: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    mesh = artifacts.mesh
    displacements = snapshot["displacements"]
    figure, axis = plt.subplots(figsize=(8.2, 8.2))

    undeformed_pad = [
        [(mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm) for node_id in element.node_ids]
        for element in mesh.pad_elements
    ]
    deformed_pad = [
        [
            (
                mesh.nodes[node_id].x_mm + displacements[node_id][0],
                mesh.nodes[node_id].y_mm + displacements[node_id][1],
            )
            for node_id in element.node_ids
        ]
        for element in mesh.pad_elements
    ]
    axis.add_collection(
        PolyCollection(undeformed_pad, facecolors="none", edgecolors="#9E9E9E", linewidths=0.08, alpha=0.35)
    )
    axis.add_collection(
        PolyCollection(deformed_pad, facecolors="#9ED7E5", edgecolors="#3B7C8C", linewidths=0.08, alpha=0.62)
    )

    carrier = [
        [
            (
                mesh.nodes[node_id].x_mm + displacements[node_id][0],
                mesh.nodes[node_id].y_mm + displacements[node_id][1],
            )
            for node_id in element.node_ids
        ]
        for element in mesh.carrier_elements
    ]
    axis.add_collection(
        PolyCollection(carrier, facecolors="#747B84", edgecolors="#444A50", linewidths=0.08, alpha=0.8)
    )

    node_map = artifacts.indenter_topology.local_to_global_node_id
    indenter = [
        [
            (
                artifacts.indenter_mesh.nodes[local_id].x_mm + displacements[node_map[local_id]][0],
                artifacts.indenter_mesh.nodes[local_id].y_mm + displacements[node_map[local_id]][1],
            )
            for local_id in element.node_ids
        ]
        for element in artifacts.indenter_mesh.elements
    ]
    axis.add_collection(
        PolyCollection(indenter, facecolors="#D0D0D0", edgecolors="#555555", linewidths=0.1, alpha=0.9)
    )

    active_external = snapshot["active_external_node_ids"]
    if active_external:
        axis.scatter(
            [mesh.nodes[node_id].x_mm + displacements[node_id][0] for node_id in active_external],
            [mesh.nodes[node_id].y_mm + displacements[node_id][1] for node_id in active_external],
            s=12,
            color="#D73027",
            label="ACTIVE external",
            zorder=8,
        )
    internal_colors = {
        "internal_left": "#7B3294",
        "internal_right": "#008837",
        "internal_bottom": "#F46D43",
    }
    for name, node_ids in snapshot["active_internal_node_ids"].items():
        if node_ids:
            axis.scatter(
                [mesh.nodes[node_id].x_mm + displacements[node_id][0] for node_id in node_ids],
                [mesh.nodes[node_id].y_mm + displacements[node_id][1] for node_id in node_ids],
                s=10,
                color=internal_colors[name],
                label=f"ACTIVE {name}",
                zorder=8,
            )
    statistics = snapshot["pad_strain_det_f"]
    strain_point = statistics["maximum_principal_green_lagrange_strain"]["reference_coordinate_mm"]
    det_point = statistics["det_f"]["minimum_reference_coordinate_mm"]
    axis.scatter(*strain_point, marker="*", s=90, color="#B2182B", label="Maximum strain location", zorder=9)
    axis.scatter(*det_point, marker="X", s=55, color="#2166AC", label="Minimum det(F) location", zorder=9)

    all_points = [point for polygon in (*deformed_pad, *carrier, *indenter) for point in polygon]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    padding = 0.04 * max(max(x_values) - min(x_values), max(y_values) - min(y_values))
    axis.set_xlim(min(x_values) - padding, max(x_values) + padding)
    axis.set_ylim(min(y_values) - padding, max(y_values) + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set(xlabel="x [mm]", ylabel="y [mm]", title=f"Phase 4I at {snapshot['depth_mm']:g} mm — displacement scale 1×")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_indentation_case_outputs(
    result: Mapping[str, Any],
    artifacts: Any | None,
    output_directory: str | Path,
) -> dict[str, str]:
    """Write one case's JSON, CSV, and available diagnostic PNG files."""
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    history_path = directory / "history.csv"
    _write_json(result_path, result)
    _write_history_csv(history_path, result.get("history", []))
    outputs = {"result": str(result_path), "history": str(history_path)}
    if artifacts is None:
        return outputs
    profiles_directory = directory / "profiles"
    plots_directory = directory / "plots"
    for key, snapshot in artifacts.snapshots.items():
        label = str(key).replace(".", "p")
        profile_path = profiles_directory / f"profile_{label}.csv"
        _write_profile_csv(profile_path, snapshot["profile"])
        outputs[f"profile_{label}"] = str(profile_path)
        deformed_path = plots_directory / f"deformed_mesh_{label}.png"
        _save_deformed_mesh_plot(artifacts, snapshot, deformed_path)
        outputs[f"deformed_mesh_{label}"] = str(deformed_path)
    _save_history_plots(result, plots_directory)
    _save_outer_profile_plot(
        artifacts.snapshots, plots_directory / "outer_arc_profiles.png"
    )
    if artifacts.snapshots:
        final_key = max(artifacts.snapshots, key=float)
        final_path = directory / "deformed_mesh.png"
        _save_deformed_mesh_plot(artifacts, artifacts.snapshots[final_key], final_path)
        outputs["deformed_mesh"] = str(final_path)
    return outputs
