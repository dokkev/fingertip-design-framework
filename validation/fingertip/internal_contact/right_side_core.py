"""Phase 4I-E left/right contact orientation and endpoint assembly audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
import traceback
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from mesh.fingertip import generate_fingertip_mesh
from fem.indentation import (
    IndentationSettings,
    run_indentation_case,
)
from validation.fingertip.internal_contact.diagnostics import (
    FIRST_STEP_TRAVEL_MM,
    build_diagnostic_context,
    contact_condition_records,
    dof_records,
    runtime_contract,
)
from fem.kratos_adapter import import_kratos, set_indenter_travel
from mesh.types import (
    BoundaryEdge,
    FingertipMesh,
    MeshLevel,
    mesh_settings_for_level,
)
from validation.fingertip.internal_contact.sparse import analyze_sparse_system
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


Side = Literal["left", "right"]


@dataclass(frozen=True)
class OrientationVariant:
    """Diagnostic-only right-side Line2 node-ordering controls."""

    name: str
    reverse_slave: bool
    reverse_master: bool

    @property
    def reversed_boundary_tags(self) -> tuple[str, ...]:
        tags: list[str] = []
        if self.reverse_slave:
            tags.append("pad_cutout_right")
        if self.reverse_master:
            tags.append("stem_right")
        return tuple(tags)


ORIENTATION_VARIANTS = {
    "R00": OrientationVariant("R00", False, False),
    "R10": OrientationVariant("R10", True, False),
    "R01": OrientationVariant("R01", False, True),
    "R11": OrientationVariant("R11", True, True),
}


SURFACE_TAGS = {
    "left": ("pad_cutout_left", "stem_left"),
    "right": ("pad_cutout_right", "stem_right"),
}

SURFACE_MODEL_PARTS = {
    "pad_cutout_left": "PadCutoutLeft",
    "pad_cutout_right": "PadCutoutRight",
    "stem_left": "StemLeft",
    "stem_right": "StemRight",
}


class RightSideAuditError(RuntimeError):
    """Raised when an orientation or mirror contract cannot be resolved."""


def reverse_boundary_condition_ordering(
    mesh: FingertipMesh,
    boundary_tags: Sequence[str],
) -> FingertipMesh:
    """Reverse selected Line2 connectivities without creating mesh entities."""
    requested = tuple(boundary_tags)
    unknown = sorted(set(requested) - set(mesh.boundary_edges))
    if unknown:
        raise ValueError(
            "unknown boundary orientation tag(s): " + ", ".join(unknown)
        )
    boundaries = dict(mesh.boundary_edges)
    for tag in requested:
        boundaries[tag] = tuple(
            BoundaryEdge(
                tuple(reversed(edge.node_ids)),
                edge.domain,
            )
            for edge in boundaries[tag]
        )
    result = replace(mesh, boundary_edges=boundaries)
    if result.quality.node_count != mesh.quality.node_count:
        raise RightSideAuditError("orientation edit changed the root node count")
    if len(result.elements) != len(mesh.elements):
        raise RightSideAuditError(
            "orientation edit changed volume element connectivity"
        )
    if any(
        result.elements[index].node_ids != mesh.elements[index].node_ids
        for index in range(len(mesh.elements))
    ):
        raise RightSideAuditError(
            "orientation edit changed volume element connectivity"
        )
    return result


def mesh_for_orientation_variant(
    mesh: FingertipMesh, variant: OrientationVariant
) -> FingertipMesh:
    """Return an immutable diagnostic mesh for one R00/R10/R01/R11 case."""
    return reverse_boundary_condition_ordering(
        mesh, variant.reversed_boundary_tags
    )


def _normalized(vector: Sequence[float]) -> tuple[float, float]:
    length = math.hypot(float(vector[0]), float(vector[1]))
    if not math.isfinite(length) or length <= 0.0:
        raise RightSideAuditError("cannot normalize a zero vector")
    return float(vector[0]) / length, float(vector[1]) / length


def _ordered_normal(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float]:
    tangent = _normalized(
        (float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
    )
    return tangent[1], -tangent[0]


def _expected_physical_normal(
    fingertip_model: FingertipModel,
    domain: str,
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float]:
    """Select the normal pointing from material into the physical void."""
    from shapely.geometry import Point

    material = (
        fingertip_model.pad_material_geometry
        if domain == "pad"
        else fingertip_model.link_geometry
    )
    tangent = _normalized(
        (float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
    )
    candidates = (
        (tangent[1], -tangent[0]),
        (-tangent[1], tangent[0]),
    )
    midpoint = (
        0.5 * (float(first[0]) + float(second[0])),
        0.5 * (float(first[1]) + float(second[1])),
    )
    probe = max(
        1.0e-4, 1000.0 * fingertip_model.parameters.geometry_tolerance
    )
    outside = [
        normal
        for normal in candidates
        if not material.covers(
            Point(
                midpoint[0] + probe * normal[0],
                midpoint[1] + probe * normal[1],
            )
        )
    ]
    if len(outside) != 1:
        raise RightSideAuditError(
            "physical outward normal is ambiguous at a Line2 midpoint"
        )
    return outside[0]


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return float(first[0]) * float(second[0]) + float(first[1]) * float(
        second[1]
    )


def boundary_orientation_contract(
    mesh: FingertipMesh,
    fingertip_model: FingertipModel,
    boundary_tags: Sequence[str] = (
        "pad_cutout_left",
        "pad_cutout_right",
        "stem_left",
        "stem_right",
    ),
) -> dict[str, Any]:
    """Inspect source ordering against a Shapely-derived physical normal."""
    result: dict[str, Any] = {}
    for tag in boundary_tags:
        records: list[dict[str, Any]] = []
        for index, edge in enumerate(mesh.boundary_edges[tag]):
            first = mesh.nodes[edge.node_ids[0]]
            second = mesh.nodes[edge.node_ids[1]]
            first_point = (first.x_mm, first.y_mm)
            second_point = (second.x_mm, second.y_mm)
            tangent = _normalized(
                (
                    second.x_mm - first.x_mm,
                    second.y_mm - first.y_mm,
                )
            )
            ordered = _ordered_normal(first_point, second_point)
            expected = _expected_physical_normal(
                fingertip_model,
                edge.domain,
                first_point,
                second_point,
            )
            records.append(
                {
                    "edge_index": index,
                    "node_ids": list(edge.node_ids),
                    "node_coordinates_mm": [
                        list(first_point),
                        list(second_point),
                    ],
                    "oriented_tangent": list(tangent),
                    "ordering_normal": list(ordered),
                    "expected_physical_normal": list(expected),
                    "normal_dot": _dot(ordered, expected),
                }
            )
        result[tag] = {
            "edge_count": len(records),
            "records": records,
            "all_ordering_normals_physical": all(
                record["normal_dot"] > 1.0 - 1.0e-12
                for record in records
            ),
        }
    return {
        "surfaces": result,
        "all_ordering_normals_physical": all(
            value["all_ordering_normals_physical"]
            for value in result.values()
        ),
    }


def _coordinate_key(
    x: float, y: float, tolerance: float
) -> tuple[int, int]:
    return round(x / tolerance), round(y / tolerance)


def mirror_node_mapping(
    mesh: FingertipMesh,
    left_tag: str,
    right_tag: str,
) -> list[dict[str, Any]]:
    """Map semantic boundary nodes through ``(x,y) -> (-x,y)``."""
    tolerance = mesh.settings.classification_tolerance_mm
    right_ids = {
        node_id
        for edge in mesh.boundary_edges[right_tag]
        for node_id in edge.node_ids
    }
    right_by_coordinate = {
        _coordinate_key(
            mesh.nodes[node_id].x_mm,
            mesh.nodes[node_id].y_mm,
            tolerance,
        ): node_id
        for node_id in right_ids
    }
    left_ids = sorted(
        {
            node_id
            for edge in mesh.boundary_edges[left_tag]
            for node_id in edge.node_ids
        },
        key=lambda node_id: (
            mesh.nodes[node_id].y_mm,
            mesh.nodes[node_id].x_mm,
        ),
    )
    records: list[dict[str, Any]] = []
    for left_id in left_ids:
        left = mesh.nodes[left_id]
        key = _coordinate_key(-left.x_mm, left.y_mm, tolerance)
        if key not in right_by_coordinate:
            raise RightSideAuditError(
                f"no reflected node for {left_tag} node {left_id}"
            )
        right_id = right_by_coordinate[key]
        right = mesh.nodes[right_id]
        records.append(
            {
                "left_node_id": left_id,
                "right_node_id": right_id,
                "left_coordinate_mm": [left.x_mm, left.y_mm],
                "right_coordinate_mm": [right.x_mm, right.y_mm],
                "reflection_error_mm": math.hypot(
                    right.x_mm + left.x_mm,
                    right.y_mm - left.y_mm,
                ),
            }
        )
    if len(records) != len(right_ids):
        raise RightSideAuditError(
            f"{left_tag}/{right_tag} reflected node counts differ"
        )
    return records


def mirror_condition_mapping(
    mesh: FingertipMesh,
    left_tag: str,
    right_tag: str,
) -> list[dict[str, Any]]:
    """Map Line2 geometries by reflected endpoint coordinates."""
    tolerance = mesh.settings.classification_tolerance_mm

    def signature(edge: BoundaryEdge, reflect_x: bool) -> tuple[Any, ...]:
        points = []
        for node_id in edge.node_ids:
            node = mesh.nodes[node_id]
            x = -node.x_mm if reflect_x else node.x_mm
            points.append(_coordinate_key(x, node.y_mm, tolerance))
        return tuple(sorted(points))

    right_by_signature = {
        signature(edge, False): (index, edge)
        for index, edge in enumerate(mesh.boundary_edges[right_tag])
    }
    records: list[dict[str, Any]] = []
    for left_index, left_edge in enumerate(mesh.boundary_edges[left_tag]):
        reflected = signature(left_edge, True)
        if reflected not in right_by_signature:
            raise RightSideAuditError(
                f"no reflected condition for {left_tag} edge {left_index}"
            )
        right_index, right_edge = right_by_signature[reflected]
        records.append(
            {
                "left_edge_index": left_index,
                "right_edge_index": right_index,
                "left_node_ids": list(left_edge.node_ids),
                "right_node_ids": list(right_edge.node_ids),
                "geometry_reflection_matches": True,
            }
        )
    if len(records) != len(mesh.boundary_edges[right_tag]):
        raise RightSideAuditError(
            f"{left_tag}/{right_tag} reflected condition counts differ"
        )
    return records


def left_right_mirror_contract(
    mesh: FingertipMesh, fingertip_model: FingertipModel
) -> dict[str, Any]:
    """Build the source-mesh oracle independent of Kratos flags/search."""
    orientation = boundary_orientation_contract(mesh, fingertip_model)
    pad_nodes = mirror_node_mapping(
        mesh, "pad_cutout_left", "pad_cutout_right"
    )
    stem_nodes = mirror_node_mapping(mesh, "stem_left", "stem_right")
    pad_conditions = mirror_condition_mapping(
        mesh, "pad_cutout_left", "pad_cutout_right"
    )
    stem_conditions = mirror_condition_mapping(
        mesh, "stem_left", "stem_right"
    )
    normal_checks: list[dict[str, Any]] = []
    for left_tag, right_tag in (
        ("pad_cutout_left", "pad_cutout_right"),
        ("stem_left", "stem_right"),
    ):
        left_records = orientation["surfaces"][left_tag]["records"]
        right_records = orientation["surfaces"][right_tag]["records"]
        mapping = mirror_condition_mapping(mesh, left_tag, right_tag)
        for pair in mapping:
            left = left_records[pair["left_edge_index"]]
            right = right_records[pair["right_edge_index"]]
            expected = (
                -float(left["expected_physical_normal"][0]),
                float(left["expected_physical_normal"][1]),
            )
            actual = right["expected_physical_normal"]
            normal_checks.append(
                {
                    "left_tag": left_tag,
                    "right_tag": right_tag,
                    **pair,
                    "reflected_left_expected_normal": list(expected),
                    "right_expected_normal": actual,
                    "normal_error": math.hypot(
                        actual[0] - expected[0],
                        actual[1] - expected[1],
                    ),
                }
            )
    return {
        "pad_node_mapping": pad_nodes,
        "stem_node_mapping": stem_nodes,
        "pad_condition_mapping": pad_conditions,
        "stem_condition_mapping": stem_conditions,
        "normal_reflection_checks": normal_checks,
        "checks": {
            "pad_nodes_mirror": all(
                record["reflection_error_mm"]
                <= mesh.settings.classification_tolerance_mm
                for record in pad_nodes
            ),
            "stem_nodes_mirror": all(
                record["reflection_error_mm"]
                <= mesh.settings.classification_tolerance_mm
                for record in stem_nodes
            ),
            "pad_conditions_mirror": all(
                record["geometry_reflection_matches"]
                for record in pad_conditions
            ),
            "stem_conditions_mirror": all(
                record["geometry_reflection_matches"]
                for record in stem_conditions
            ),
            "physical_normals_reflect": all(
                record["normal_error"] <= 1.0e-12
                for record in normal_checks
            ),
            "source_ordering_is_physical": orientation[
                "all_ordering_normals_physical"
            ],
        },
    }


def _condition_record(
    condition: Any,
    model_part: Any,
    fingertip_model: FingertipModel,
    domain: str,
) -> dict[str, Any]:
    KM, _, _, _ = import_kratos()
    geometry = condition.GetGeometry()
    first = geometry[0]
    second = geometry[1]
    first_point = (float(first.X0), float(first.Y0))
    second_point = (float(second.X0), float(second.Y0))
    tangent = _normalized(
        (
            second_point[0] - first_point[0],
            second_point[1] - first_point[1],
        )
    )
    ordered = _ordered_normal(first_point, second_point)
    expected = _expected_physical_normal(
        fingertip_model, domain, first_point, second_point
    )
    return {
        "condition_id": condition.Id,
        "connectivity": [first.Id, second.Id],
        "node_coordinates_mm": [list(first_point), list(second_point)],
        "oriented_tangent": list(tangent),
        "condition_ordering_normal": list(ordered),
        "expected_physical_normal": list(expected),
        "normal_dot": _dot(ordered, expected),
        "flags": {
            "MASTER": bool(condition.Is(KM.MASTER)),
            "SLAVE": bool(condition.Is(KM.SLAVE)),
            "ACTIVE": bool(condition.Is(KM.ACTIVE)),
        },
    }


def _safe_scalar(node: Any, variable: Any) -> float | None:
    if not node.SolutionStepsDataHas(variable):
        return None
    value = float(node.GetSolutionStepValue(variable))
    return value if math.isfinite(value) else None


def _surface_stage_snapshot(
    model_part: Any,
    fingertip_model: FingertipModel,
    stage: str,
) -> dict[str, Any]:
    KM, CSMA, _, _ = import_kratos()
    surfaces: dict[str, Any] = {}
    for tag in (
        "pad_cutout_left",
        "pad_cutout_right",
        "stem_left",
        "stem_right",
    ):
        name = SURFACE_MODEL_PARTS[tag]
        part = model_part.GetSubModelPart(name)
        domain = "pad" if tag.startswith("pad_") else "rigid_carrier"
        conditions = [
            _condition_record(
                condition, model_part, fingertip_model, domain
            )
            for condition in part.Conditions
        ]
        expected = (
            conditions[0]["expected_physical_normal"]
            if conditions
            else None
        )
        nodes: list[dict[str, Any]] = []
        for node in sorted(part.Nodes, key=lambda item: (item.Y0, item.X0)):
            normal = (
                [
                    float(node.GetSolutionStepValue(KM.NORMAL)[component])
                    for component in range(2)
                ]
                if node.SolutionStepsDataHas(KM.NORMAL)
                else None
            )
            lm_dof = None
            if node.HasDofFor(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            ):
                dof = node.GetDof(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                )
                lm_dof = {
                    "equation_id": int(dof.EquationId),
                    "fixed": bool(dof.IsFixed()),
                }
            nodes.append(
                {
                    "node_id": node.Id,
                    "reference_coordinate_mm": [
                        float(node.X0),
                        float(node.Y0),
                    ],
                    "nodal_normal": normal,
                    "expected_physical_normal": expected,
                    "normal_dot": (
                        _dot(normal, expected)
                        if normal is not None and expected is not None
                        else None
                    ),
                    "nodal_h": _safe_scalar(node, KM.NODAL_H),
                    "weighted_gap": _safe_scalar(
                        node, CSMA.WEIGHTED_GAP
                    ),
                    "lagrange_multiplier_contact_pressure": _safe_scalar(
                        node,
                        CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
                    ),
                    "lm_dof": lm_dof,
                    "flags": {
                        "MASTER": bool(node.Is(KM.MASTER)),
                        "SLAVE": bool(node.Is(KM.SLAVE)),
                        "ACTIVE": bool(node.Is(KM.ACTIVE)),
                    },
                }
            )
        upper = max(nodes, key=lambda record: record["reference_coordinate_mm"][1])
        surfaces[tag] = {
            "submodelpart": name,
            "conditions": conditions,
            "nodes": nodes,
            "upper_endpoint": upper,
            "all_condition_ordering_normals_physical": all(
                record["normal_dot"] > 1.0 - 1.0e-12
                for record in conditions
            ),
        }
    return {"stage": stage, "surfaces": surfaces}


def _projection_on_segment(
    point: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, Any]:
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    denominator = dx * dx + dy * dy
    if denominator <= 0.0:
        return {
            "success": False,
            "reason": "zero_length_master_segment",
        }
    parameter = (
        (float(point[0]) - float(first[0])) * dx
        + (float(point[1]) - float(first[1])) * dy
    ) / denominator
    projection = (
        float(first[0]) + parameter * dx,
        float(first[1]) + parameter * dy,
    )
    return {
        "success": -1.0e-12 <= parameter <= 1.0 + 1.0e-12,
        "parametric_coordinate": parameter,
        "projection_point_mm": list(projection),
        "distance_mm": math.hypot(
            float(point[0]) - projection[0],
            float(point[1]) - projection[1],
        ),
    }


def _pairing_projection_records(
    context: Any, pair_index: int, endpoint_node_id: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    computing = context.model[
        f"Structure.ComputingContact.ComputingContactSub{pair_index}"
    ]
    for condition in computing.Conditions:
        geometry = condition.GetGeometry()
        slave = geometry.GetGeometryPart(0)
        master = geometry.GetGeometryPart(1)
        slave_ids = [node.Id for node in slave]
        if endpoint_node_id not in slave_ids:
            continue
        master_points = [
            (float(node.X0), float(node.Y0)) for node in master
        ]
        endpoint = context.model_part.Nodes[endpoint_node_id]
        projection = _projection_on_segment(
            (endpoint.X0, endpoint.Y0),
            master_points[0],
            master_points[1],
        )
        records.append(
            {
                "generated_condition_id": condition.Id,
                "slave_node_ids": slave_ids,
                "master_node_ids": [node.Id for node in master],
                "master_node_coordinates_mm": [
                    list(point) for point in master_points
                ],
                "active": bool(condition.Is(import_kratos()[0].ACTIVE)),
                "endpoint_projection": projection,
            }
        )
    return records


def _local_lm_contributors(
    context: Any,
    pair_index: int,
    endpoint_node_id: int,
    lm_equation_id: int,
) -> list[dict[str, Any]]:
    KM, CSMA, _, _ = import_kratos()
    records: list[dict[str, Any]] = []
    computing = context.model[
        f"Structure.ComputingContact.ComputingContactSub{pair_index}"
    ]
    for condition in computing.Conditions:
        dofs = condition.GetDofList(context.model_part.ProcessInfo)
        local_rows = [
            index
            for index, dof in enumerate(dofs)
            if dof.Id() == endpoint_node_id
            and dof.GetVariable().Name()
            == CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
        ]
        if not local_rows:
            continue
        lhs = KM.Matrix()
        rhs = KM.Vector()
        condition.CalculateLocalSystem(
            lhs, rhs, context.model_part.ProcessInfo
        )
        array = np.asarray(
            [
                [float(lhs[row, column]) for column in range(lhs.Size2())]
                for row in range(lhs.Size1())
            ],
            dtype=float,
        )
        equation_ids = [
            int(value)
            for value in condition.EquationIdVector(
                context.model_part.ProcessInfo
            )
        ]
        dof_records = [
            {
                "node_id": int(dof.Id()),
                "variable": dof.GetVariable().Name(),
                "equation_id": equation_ids[index],
                "fixed": bool(dof.IsFixed()),
            }
            for index, dof in enumerate(dofs)
        ]
        for row in local_rows:
            free_columns = [
                index
                for index, dof in enumerate(dofs)
                if not dof.IsFixed()
            ]
            records.append(
                {
                    "condition_id": condition.Id,
                    "local_lm_row": row,
                    "global_lm_equation_id": lm_equation_id,
                    "local_row_norm_all_columns": float(
                        np.linalg.norm(array[row, :])
                    ),
                    "local_row_norm_free_columns": float(
                        np.linalg.norm(array[row, free_columns])
                    )
                    if free_columns
                    else 0.0,
                    "local_row_values": [
                        float(value) for value in array[row, :]
                    ],
                    "local_dofs": dof_records,
                }
            )
    return records


def endpoint_id(model_part: Any, side: Side, slave: bool) -> int:
    tag = (
        f"pad_cutout_{side}" if slave else f"stem_{side}"
    )
    part = model_part.GetSubModelPart(SURFACE_MODEL_PARTS[tag])
    return max(part.Nodes, key=lambda node: (node.Y0, -abs(node.X0))).Id


def _solver_endpoint_state(
    solve_result: Mapping[str, Any],
    group_name: str,
    endpoint_node_id: int,
) -> dict[str, Any]:
    if solve_result.get("history"):
        group = solve_result["history"][-1]["contact_groups"][group_name]
    else:
        group = (
            solve_result.get("failure_step_diagnostics", {})
            .get("contact_groups", {})
            .get(group_name, {})
        )
    state = next(
        (
            record
            for record in group.get("slave_nodal_state", [])
            if int(record["node_id"]) == endpoint_node_id
        ),
        None,
    )
    return {
        "available": state is not None,
        "state": state,
        "group_active_condition_count": group.get(
            "active_condition_count"
        ),
        "group_generated_condition_count": group.get(
            "generated_condition_count"
        ),
    }


def audit_side_orientation(
    side: Side,
    mesh_level: MeshLevel = "medium",
    variant: OrientationVariant | None = None,
    pre_solve_callback: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit all requested stages and run the unchanged first-step solver."""
    if side == "left" and variant is not None:
        raise ValueError("orientation variants apply only to the right side")
    resolved_variant = variant or OrientationVariant("L00", False, False)
    configuration = f"{side}_only"
    case = "C-left" if side == "left" else "C-right"
    fingertip_model = FingertipModel(FingertipParameters())
    base_mesh = generate_fingertip_mesh(
        fingertip_model, mesh_settings_for_level(mesh_level)
    )
    mesh = (
        mesh_for_orientation_variant(base_mesh, resolved_variant)
        if side == "right"
        else base_mesh
    )
    orientation = boundary_orientation_contract(mesh, fingertip_model)
    orientation_edit_scope = {
        "reversed_boundary_tags": list(
            resolved_variant.reversed_boundary_tags
        ),
        "root_node_count_before": base_mesh.quality.node_count,
        "root_node_count_after": mesh.quality.node_count,
        "root_condition_count_before": sum(
            len(edges) for edges in base_mesh.boundary_edges.values()
        ),
        "root_condition_count_after": sum(
            len(edges) for edges in mesh.boundary_edges.values()
        ),
        "volume_connectivity_unchanged": all(
            first.node_ids == second.node_ids
            for first, second in zip(base_mesh.elements, mesh.elements)
        ),
        "duplicate_boundary_connectivity_count": sum(
            len(edges)
            - len(
                {
                    tuple(sorted(edge.node_ids))
                    for edge in edges
                }
            )
            for edges in mesh.boundary_edges.values()
        ),
    }
    stages: dict[str, Any] = {}

    def capture_before(
        model_part: Any,
        _base_topology: Any,
        _indenter_topology: Any,
        _mesh: Any,
        _fixture: Any,
    ) -> None:
        stages["before_contact_process"] = _surface_stage_snapshot(
            model_part, fingertip_model, "before_contact_process"
        )

    context: Any | None = None
    initialized_step = False
    start = time.perf_counter()
    try:
        context = build_diagnostic_context(
            mesh_level,
            configuration,
            mesh_override=mesh,
            before_initialize=capture_before,
        )
        KM, CSMA, _, _ = import_kratos()
        stages["after_execute_initialize"] = _surface_stage_snapshot(
            context.model_part,
            fingertip_model,
            "after_execute_initialize",
        )
        runtime = runtime_contract(context)
        solver = context.analysis._GetSolver()
        context.analysis.time = solver.AdvanceInTime(context.analysis.time)
        set_indenter_travel(
            context.model_part,
            context.indenter_topology.node_ids,
            context.fixture,
            FIRST_STEP_TRAVEL_MM,
        )
        context.analysis.InitializeSolutionStep()
        initialized_step = True
        solver.Predict()
        stages["after_contact_search"] = _surface_stage_snapshot(
            context.model_part,
            fingertip_model,
            "after_contact_search",
        )
        endpoint_node_id = endpoint_id(
            context.model_part, side, slave=True
        )
        pairing = _pairing_projection_records(
            context, 1, endpoint_node_id
        )

        strategy = solver._GetSolutionStrategy()
        builder = solver._GetBuilderAndSolver()
        scheme = solver._GetScheme()
        dof_set = builder.GetDofSet()
        (
            dof_rows,
            equation_map,
            assembled_dofs,
            dof_summary,
        ) = dof_records(context, dof_set)
        matrix = strategy.GetSystemMatrix()
        rhs = strategy.GetSystemVector()
        increment = KM.Vector(matrix.Size1())
        builder.Build(
            scheme, solver.GetComputingModelPart(), matrix, rhs
        )
        builder.ApplyDirichletConditions(
            scheme,
            solver.GetComputingModelPart(),
            matrix,
            increment,
            rhs,
        )
        import KratosMultiphysics.scipy_conversion_tools as conversion

        csr = conversion.to_csr(matrix)
        rhs_array = np.asarray([rhs[index] for index in range(len(rhs))])
        matrix_diagnostics = analyze_sparse_system(
            csr, rhs_array, equation_map
        )
        lm_name = CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
        lm_dof = assembled_dofs[(endpoint_node_id, lm_name)]
        equation_id = int(lm_dof["equation_id"])
        row = csr.getrow(equation_id)
        contributors = _local_lm_contributors(
            context, 1, endpoint_node_id, equation_id
        )
        stages["after_first_newton_assembly"] = _surface_stage_snapshot(
            context.model_part,
            fingertip_model,
            "after_first_newton_assembly",
        )
        contact_records, pair_purity = contact_condition_records(
            context
        )
        diagnostic = {
            "runtime_contact_contract": runtime,
            "dof_summary": dof_summary,
            "matrix_diagnostics": matrix_diagnostics,
            "pair_purity": pair_purity,
            "contact_records": contact_records,
            "endpoint_assembly": {
                "side": side,
                "slave_endpoint_node_id": endpoint_node_id,
                "slave_endpoint_reference_coordinate_mm": [
                    float(context.model_part.Nodes[endpoint_node_id].X0),
                    float(context.model_part.Nodes[endpoint_node_id].Y0),
                ],
                "lm_equation_id": equation_id,
                "lm_fixed": bool(lm_dof["fixed"]),
                "global_tangent_row_norm": float(
                    np.linalg.norm(row.data)
                ),
                "global_tangent_row_nnz": int(row.nnz),
                "near_zero_tolerance": matrix_diagnostics[
                    "near_zero_row_tolerance"
                ],
                "near_zero": float(np.linalg.norm(row.data))
                < matrix_diagnostics["near_zero_row_tolerance"],
                "pairing_projection": pairing,
                "local_condition_contributors": contributors,
            },
        }
    except Exception as exception:
        return (
            {
                "phase": "4I-E",
                "side": side,
                "variant": asdict(resolved_variant),
                "status": "FAIL",
                "failure_reason": "diagnostic_exception",
                "exception": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
                "orientation_edit_scope": orientation_edit_scope,
                "source_orientation_contract": orientation,
                "stage_snapshots": stages,
                "wall_clock_seconds": time.perf_counter() - start,
            },
            [],
            [],
        )
    finally:
        if context is not None:
            try:
                if initialized_step:
                    context.analysis.FinalizeSolutionStep()
                context.analysis.Finalize()
            except Exception:
                pass

    relevant_tags = SURFACE_TAGS[side]
    relevant_ordering_physical = all(
        orientation["surfaces"][tag][
            "all_ordering_normals_physical"
        ]
        for tag in relevant_tags
    )
    if pre_solve_callback is not None:
        pre_solve_callback(
            {
                "phase": "4I-E",
                "case": case,
                "side": side,
                "variant": asdict(resolved_variant),
                "mesh_level": mesh_level,
                "status": "PENDING_SOLVE",
                "orientation_edit_scope": orientation_edit_scope,
                "source_orientation_contract": orientation,
                "stage_snapshots": stages,
                "diagnostic": diagnostic,
            },
            dof_rows,
            contact_records,
        )

    solve_result, _ = run_indentation_case(
        fingertip_model,
        mesh_level,
        IndentationSettings(FIRST_STEP_TRAVEL_MM, 1),
        internal_contact_configuration=configuration,
        mesh_override=mesh,
    )
    group_name = f"internal_{side}"
    solve_endpoint = _solver_endpoint_state(
        solve_result, group_name, diagnostic["endpoint_assembly"][
            "slave_endpoint_node_id"
        ]
    )
    stages["after_solve"] = {
        "stage": "after_solve",
        "solver_converged": solve_result.get("solve_status") == "PASS",
        "endpoint": solve_endpoint,
    }
    history = solve_result.get("history", [])
    point = history[-1] if history else None
    internal_group = (
        point.get("contact_groups", {}).get(group_name, {})
        if point
        else solve_result.get("failure_step_diagnostics", {})
        .get("contact_groups", {})
        .get(group_name, {})
    )
    reaction = (
        float(point["indenter_normal_reaction_n"]) if point else None
    )
    det_f = (
        point["pad_strain_det_f"]["det_f"]["min"] if point else None
    )
    checks = {
        "source_ordering_normals_physical": relevant_ordering_physical,
        "runtime_pair_purity": diagnostic["pair_purity"][
            "all_generated_conditions_pair_pure"
        ],
        "right_internal_contact_active": (
            int(internal_group.get("active_condition_count", 0)) > 0
        ),
        "upper_endpoint_projection_success": bool(
            diagnostic["endpoint_assembly"]["pairing_projection"]
        )
        and all(
            record["endpoint_projection"]["success"]
            for record in diagnostic["endpoint_assembly"][
                "pairing_projection"
            ]
        ),
        "first_step_solver_converged": solve_result.get("solve_status")
        == "PASS",
        "finite_positive_reaction": reaction is not None
        and math.isfinite(reaction)
        and reaction > 0.0,
        "finite_displacement_and_volumetric_strain": bool(
            point and point.get("finite_fields")
        ),
        "positive_det_f": det_f is not None
        and math.isfinite(float(det_f))
        and float(det_f) > 0.0,
        "upper_endpoint_lm_row_not_near_zero": not diagnostic[
            "endpoint_assembly"
        ]["near_zero"],
        "surface_crossing_absent": bool(
            internal_group.get("penetration_pass", False)
        ),
    }
    status_checks = (
        checks
        if side == "right"
        else {
            key: value
            for key, value in checks.items()
            if key != "upper_endpoint_lm_row_not_near_zero"
        }
    )
    result = {
        "phase": "4I-E",
        "case": case,
        "side": side,
        "variant": asdict(resolved_variant),
        "mesh_level": mesh_level,
        "first_step_travel_mm": FIRST_STEP_TRAVEL_MM,
        "status": "PASS" if all(status_checks.values()) else "FAIL",
        "orientation_edit_scope": orientation_edit_scope,
        "source_orientation_contract": orientation,
        "stage_snapshots": stages,
        "diagnostic": diagnostic,
        "solve_result": solve_result,
        "solve_endpoint_state": solve_endpoint,
        "acceptance_checks": checks,
        "wall_clock_seconds": time.perf_counter() - start,
    }
    return result, dof_rows, contact_records


def common_audit_mesh(
    mesh_level: MeshLevel = "medium",
) -> tuple[FingertipModel, FingertipMesh]:
    """Generate the one immutable source mesh used by mirror unit contracts."""
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(
        model, mesh_settings_for_level(mesh_level)
    )
    return model, mesh
