"""Solver-independent Phase 4I indentation measurements and comparisons."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point

from fem.kratos_adapter import import_kratos
from mesh.indenter import CrownFrame, Vector2
from mesh.types import BoundaryEdge, FingertipMesh
from model.fingertip_model import FingertipModel


class IndentationPostprocessError(RuntimeError):
    """Raised when a requested measurement has no valid geometric support."""


def scalar_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Summarize one finite scalar field without solver-specific objects."""
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "finite": bool(values)
        and all(math.isfinite(float(value)) for value in values),
    }


def failure_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Summarize a failed iterate without serializing NaN or infinity."""
    finite_values = [
        float(value) for value in values if math.isfinite(float(value))
    ]
    return {
        "count": len(values),
        "finite_count": len(finite_values),
        "nonfinite_count": len(values) - len(finite_values),
        "min_finite": min(finite_values) if finite_values else None,
        "max_finite": max(finite_values) if finite_values else None,
        "mean_finite": (
            sum(finite_values) / len(finite_values)
            if finite_values
            else None
        ),
        "all_finite": len(finite_values) == len(values),
    }


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

def extract_nodal_fields(model_part: Any, node_ids: Sequence[int]) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    KM, _, _, _ = import_kratos()
    displacements: dict[int, tuple[float, float]] = {}
    reactions: dict[int, tuple[float, float]] = {}
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        reaction = node.GetSolutionStepValue(KM.REACTION)
        displacements[node_id] = (float(displacement[0]), float(displacement[1]))
        reactions[node_id] = (float(reaction[0]), float(reaction[1]))
    return displacements, reactions


def finite_field_failures(model_part: Any, pad_node_ids: Sequence[int]) -> list[str]:
    KM, CSMA, _, _ = import_kratos()
    failures: list[str] = []
    pad_ids = set(pad_node_ids)
    for node in model_part.Nodes:
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        reaction = node.GetSolutionStepValue(KM.REACTION)
        if not all(math.isfinite(float(displacement[index])) for index in range(2)):
            failures.append(f"node_{node.Id}_DISPLACEMENT")
        if not all(math.isfinite(float(reaction[index])) for index in range(2)):
            failures.append(f"node_{node.Id}_REACTION")
        if node.Id in pad_ids and not math.isfinite(
            float(node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
        ):
            failures.append(f"node_{node.Id}_VOLUMETRIC_STRAIN")
        if node.Is(KM.SLAVE):
            for variable in (
                CSMA.WEIGHTED_GAP,
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
            ):
                if node.SolutionStepsDataHas(variable) and not math.isfinite(
                    float(node.GetSolutionStepValue(variable))
                ):
                    failures.append(f"node_{node.Id}_{variable.Name()}")
    return failures


def rigid_domain_validation(
    model_part: Any,
    node_ids: Sequence[int],
    element_ids: Sequence[int],
    prescribed_displacement: Sequence[float],
) -> dict[str, Any]:
    KM, _, _, _ = import_kratos()
    maximum_translation_error = 0.0
    for node_id in node_ids:
        displacement = model_part.Nodes[node_id].GetSolutionStepValue(KM.DISPLACEMENT)
        maximum_translation_error = max(
            maximum_translation_error,
            math.hypot(
                float(displacement[0]) - float(prescribed_displacement[0]),
                float(displacement[1]) - float(prescribed_displacement[1]),
            ),
        )
    maximum_strain = 0.0
    maximum_det_error = 0.0
    nonpositive_det_count = 0
    for element_id in element_ids:
        geometry = model_part.Elements[element_id].GetGeometry()
        reference = [
            (float(node.X0), float(node.Y0)) for node in geometry
        ]
        current = [(float(node.X), float(node.Y)) for node in geometry]
        reference_edges = (
            (reference[1][0] - reference[0][0], reference[2][0] - reference[0][0]),
            (reference[1][1] - reference[0][1], reference[2][1] - reference[0][1]),
        )
        current_edges = (
            (current[1][0] - current[0][0], current[2][0] - current[0][0]),
            (current[1][1] - current[0][1], current[2][1] - current[0][1]),
        )
        determinant_reference = (
            reference_edges[0][0] * reference_edges[1][1]
            - reference_edges[0][1] * reference_edges[1][0]
        )
        inverse = (
            (reference_edges[1][1] / determinant_reference, -reference_edges[0][1] / determinant_reference),
            (-reference_edges[1][0] / determinant_reference, reference_edges[0][0] / determinant_reference),
        )
        deformation_gradient = (
            (
                current_edges[0][0] * inverse[0][0] + current_edges[0][1] * inverse[1][0],
                current_edges[0][0] * inverse[0][1] + current_edges[0][1] * inverse[1][1],
            ),
            (
                current_edges[1][0] * inverse[0][0] + current_edges[1][1] * inverse[1][0],
                current_edges[1][0] * inverse[0][1] + current_edges[1][1] * inverse[1][1],
            ),
        )
        determinant_f = (
            deformation_gradient[0][0] * deformation_gradient[1][1]
            - deformation_gradient[0][1] * deformation_gradient[1][0]
        )
        c00 = deformation_gradient[0][0] ** 2 + deformation_gradient[1][0] ** 2
        c01 = deformation_gradient[0][0] * deformation_gradient[0][1] + deformation_gradient[1][0] * deformation_gradient[1][1]
        c11 = deformation_gradient[0][1] ** 2 + deformation_gradient[1][1] ** 2
        strain_norm = math.sqrt((0.5 * (c00 - 1.0)) ** 2 + 2.0 * (0.5 * c01) ** 2 + (0.5 * (c11 - 1.0)) ** 2)
        maximum_strain = max(maximum_strain, strain_norm)
        maximum_det_error = max(maximum_det_error, abs(determinant_f - 1.0))
        nonpositive_det_count += determinant_f <= 0.0
    tolerance = 1.0e-10
    return {
        "node_count": len(node_ids),
        "element_count": len(element_ids),
        "maximum_translation_error_mm": maximum_translation_error,
        "maximum_green_lagrange_strain_norm": maximum_strain,
        "maximum_abs_det_f_error_from_one": maximum_det_error,
        "nonpositive_det_f_count": nonpositive_det_count,
        "numerical_tolerance": tolerance,
        "strain_energy_interpretation": "negligible because F=I and strain is numerical zero",
        "pass": maximum_translation_error <= tolerance
        and maximum_strain <= tolerance
        and maximum_det_error <= tolerance
        and nonpositive_det_count == 0,
    }


def curve_acceptance(curve: Sequence[Mapping[str, Any]], force_tolerance_n: float) -> dict[str, Any]:
    active_points = [
        point
        for point in curve
        if point["contact_groups"]["external_pad_indenter"]["active_condition_count"] > 0
        and point["indenter_normal_reaction_n"] > force_tolerance_n
    ]
    if len(active_points) < 2:
        return {
            "first_load_bearing_step": None,
            "monotonic": False,
            "smooth": False,
            "reason": "fewer than two load-bearing contact steps",
        }
    reactions = [float(point["indenter_normal_reaction_n"]) for point in active_points]
    allowed_decrease = max(0.02 * reactions[-1], force_tolerance_n)
    monotonic = all(
        second >= first - allowed_decrease
        for first, second in zip(reactions, reactions[1:])
    )
    second_differences = [
        abs(reactions[index + 1] - 2.0 * reactions[index] + reactions[index - 1])
        for index in range(1, len(reactions) - 1)
    ]
    normalized_second = max(second_differences, default=0.0) / max(
        reactions[-1], force_tolerance_n
    )
    return {
        "first_load_bearing_step": int(active_points[0]["step"]),
        "allowed_reaction_decrease_n": allowed_decrease,
        "monotonic": monotonic,
        "normalized_max_second_difference": normalized_second,
        "smoothness_limit": 0.15,
        "smooth": monotonic and normalized_second <= 0.15,
    }
