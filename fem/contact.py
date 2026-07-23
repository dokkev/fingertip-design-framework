"""Solver-facing internal-contact topology for Phase 4I-D.

The semantic left, bottom, and right submodel parts remain untouched.  The
continuous-U option creates aggregate memberships that reuse their existing
nodes and Line2 conditions.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from fem.kratos_adapter import (
    IndenterKratosTopology,
    KratosAdapterError,
    import_kratos,
)
from fem.results import (
    failure_statistics,
    scalar_statistics,
    signed_geometric_gap_statistics,
)
from mesh.types import BoundaryEdge, FingertipMesh


PAD_U_SEGMENTS = (
    "PadCutoutLeft",
    "PadCutoutBottom",
    "PadCutoutRight",
)
STEM_U_SEGMENTS = (
    "StemLeft",
    "StemBottom",
    "StemRight",
)
PAD_U_AGGREGATE = "PadInternalU"
STEM_U_AGGREGATE = "StemInternalU"


class InternalContactTopologyError(RuntimeError):
    """Raised when semantic U-boundaries cannot form one reusable aggregate."""


def _entity_ids(container: Any) -> tuple[int, ...]:
    return tuple(sorted(entity.Id for entity in container))


def _union_membership(
    model_part: Any,
    aggregate_name: str,
    semantic_names: Sequence[str],
) -> dict[str, Any]:
    if model_part.HasSubModelPart(aggregate_name):
        raise InternalContactTopologyError(
            f"aggregate submodel part {aggregate_name!r} already exists"
        )
    node_ids: set[int] = set()
    condition_ids: set[int] = set()
    semantic: dict[str, dict[str, list[int]]] = {}
    for name in semantic_names:
        if not model_part.HasSubModelPart(name):
            raise InternalContactTopologyError(
                f"missing semantic submodel part {name!r}"
            )
        part = model_part.GetSubModelPart(name)
        local_nodes = _entity_ids(part.Nodes)
        local_conditions = _entity_ids(part.Conditions)
        if not local_nodes or not local_conditions:
            raise InternalContactTopologyError(
                f"semantic submodel part {name!r} is empty"
            )
        node_ids.update(local_nodes)
        condition_ids.update(local_conditions)
        semantic[name] = {
            "node_ids": list(local_nodes),
            "condition_ids": list(local_conditions),
        }
    aggregate = model_part.CreateSubModelPart(aggregate_name)
    aggregate.AddNodes(sorted(node_ids))
    aggregate.AddConditions(sorted(condition_ids))
    return {
        "name": aggregate_name,
        "semantic_order": list(semantic_names),
        "node_ids": sorted(node_ids),
        "condition_ids": sorted(condition_ids),
        "semantic_membership": semantic,
    }


def _connectivity(condition: Any) -> tuple[int, ...]:
    return tuple(node.Id for node in condition.GetGeometry())


def _aggregate_contract(
    model_part: Any,
    membership: dict[str, Any],
) -> dict[str, Any]:
    aggregate = model_part.GetSubModelPart(membership["name"])
    condition_ids = _entity_ids(aggregate.Conditions)
    connectivities = [
        _connectivity(model_part.Conditions[condition_id])
        for condition_id in condition_ids
    ]
    signatures = [tuple(sorted(connectivity)) for connectivity in connectivities]
    adjacency: dict[int, set[int]] = {}
    for first, second in connectivities:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    endpoints = sorted(
        node_id for node_id, neighbours in adjacency.items() if len(neighbours) == 1
    )
    branch_nodes = sorted(
        node_id for node_id, neighbours in adjacency.items() if len(neighbours) > 2
    )
    return {
        **membership,
        "aggregate_node_ids": list(_entity_ids(aggregate.Nodes)),
        "aggregate_condition_ids": list(condition_ids),
        "condition_connectivity": [list(value) for value in connectivities],
        "duplicate_condition_connectivity_count": len(signatures)
        - len(set(signatures)),
        "chain_endpoint_node_ids": endpoints,
        "branch_node_ids": branch_nodes,
        "one_connected_open_chain": len(endpoints) == 2
        and not branch_nodes
        and len(adjacency) == len(membership["node_ids"]),
    }


def create_continuous_u_submodel_parts(model_part: Any) -> dict[str, Any]:
    """Create PadInternalU and StemInternalU without creating root entities."""
    root_nodes_before = model_part.NumberOfNodes()
    root_conditions_before = model_part.NumberOfConditions()
    pad = _union_membership(
        model_part, PAD_U_AGGREGATE, PAD_U_SEGMENTS
    )
    stem = _union_membership(
        model_part, STEM_U_AGGREGATE, STEM_U_SEGMENTS
    )
    root_nodes_after = model_part.NumberOfNodes()
    root_conditions_after = model_part.NumberOfConditions()
    result = {
        "pad": _aggregate_contract(model_part, pad),
        "stem": _aggregate_contract(model_part, stem),
        "root_counts": {
            "nodes_before": root_nodes_before,
            "nodes_after": root_nodes_after,
            "conditions_before": root_conditions_before,
            "conditions_after": root_conditions_after,
        },
    }
    result["checks"] = {
        "root_node_count_unchanged": root_nodes_before == root_nodes_after,
        "root_condition_count_unchanged": (
            root_conditions_before == root_conditions_after
        ),
        "pad_reuses_exact_semantic_conditions": set(
            result["pad"]["condition_ids"]
        )
        == set(result["pad"]["aggregate_condition_ids"]),
        "stem_reuses_exact_semantic_conditions": set(
            result["stem"]["condition_ids"]
        )
        == set(result["stem"]["aggregate_condition_ids"]),
        "pad_has_no_duplicate_connectivity": (
            result["pad"]["duplicate_condition_connectivity_count"] == 0
        ),
        "stem_has_no_duplicate_connectivity": (
            result["stem"]["duplicate_condition_connectivity_count"] == 0
        ),
        "pad_is_one_open_chain": result["pad"]["one_connected_open_chain"],
        "stem_is_one_open_chain": result["stem"]["one_connected_open_chain"],
    }
    if not all(result["checks"].values()):
        failures = [
            name for name, passed in result["checks"].items() if not passed
        ]
        raise InternalContactTopologyError(
            "continuous-U aggregate contract failed: " + ", ".join(failures)
        )
    return result


def u_corner_node_ids(model_part: Any) -> dict[str, int]:
    """Return the two pad and two stem lower U-corners by membership."""
    intersections = {
        "pad_left_bottom": ("PadCutoutLeft", "PadCutoutBottom"),
        "pad_bottom_right": ("PadCutoutBottom", "PadCutoutRight"),
        "stem_left_bottom": ("StemLeft", "StemBottom"),
        "stem_bottom_right": ("StemBottom", "StemRight"),
    }
    result: dict[str, int] = {}
    for label, (first_name, second_name) in intersections.items():
        first = {
            node.Id
            for node in model_part.GetSubModelPart(first_name).Nodes
        }
        second = {
            node.Id
            for node in model_part.GetSubModelPart(second_name).Nodes
        }
        shared = first.intersection(second)
        if len(shared) != 1:
            raise InternalContactTopologyError(
                f"{label} must contain exactly one shared node, got "
                f"{sorted(shared)}"
            )
        result[label] = next(iter(shared))
    if len(set(result.values())) != 4:
        raise InternalContactTopologyError(
            "pad/stem U-corners must use four distinct physical node IDs"
        )
    return result

def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def failed_contact_group_diagnostics(
    model: Any,
    model_part: Any,
    contact_groups: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Capture indexed ALM state without geometric operations on a NaN iterate."""
    KM, CSMA, _, _ = import_kratos()
    groups: dict[str, Any] = {}
    for index, (group_name, slave_name, _) in enumerate(contact_groups):
        slave = model_part.GetSubModelPart(slave_name)
        computing = model[
            f"Structure.ComputingContact.ComputingContactSub{index}"
        ]
        conditions = list(computing.Conditions)
        groups[group_name] = {
            "pair_index": index,
            "generated_condition_count": len(conditions),
            "active_condition_count": sum(
                condition.Is(KM.ACTIVE) for condition in conditions
            ),
            "active_condition_ids": sorted(
                condition.Id for condition in conditions if condition.Is(KM.ACTIVE)
            ),
            "active_slave_node_ids": sorted(
                node.Id for node in slave.Nodes if node.Is(KM.ACTIVE)
            ),
            "weighted_gap": failure_statistics(
                [
                    float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP))
                    for node in slave.Nodes
                ]
            ),
            "lagrange_multiplier_contact_pressure": failure_statistics(
                [
                    float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    )
                    for node in slave.Nodes
                ]
            ),
            "slave_nodal_state": [
                {
                    "node_id": node.Id,
                    "active": bool(node.Is(KM.ACTIVE)),
                    "weighted_gap": _finite_or_none(
                        node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)
                    ),
                    "lagrange_multiplier_contact_pressure": _finite_or_none(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    ),
                    "normal": [
                        _finite_or_none(
                            node.GetSolutionStepValue(KM.NORMAL)[component]
                        )
                        for component in range(2)
                    ],
                }
                for node in slave.Nodes
            ],
        }
    return groups


def runtime_contact_contract(
    model: Any,
    model_part: Any,
    contact_groups: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Verify indexed contact pairs using Kratos-owned submodel parts."""
    KM, _, _, _ = import_kratos()
    groups: dict[str, Any] = {}
    all_checks: list[bool] = []
    for index, (group_name, slave_name, master_name) in enumerate(
        contact_groups
    ):
        slave = model_part.GetSubModelPart(slave_name)
        master = model_part.GetSubModelPart(master_name)
        contact_path = f"Structure.Contact.ContactSub{index}"
        computing_path = f"Structure.ComputingContact.ComputingContactSub{index}"
        if not model.HasModelPart(contact_path) or not model.HasModelPart(computing_path):
            raise KratosAdapterError(
                f"Kratos did not create indexed contact model parts for {group_name}"
            )
        contact_subpart = model[contact_path]
        source_condition_ids = {
            condition.Id for condition in slave.Conditions
        }.union(condition.Id for condition in master.Conditions)
        runtime_condition_ids = {condition.Id for condition in contact_subpart.Conditions}
        source_connectivity = {
            tuple(sorted(node.Id for node in condition.GetGeometry()))
            for condition in (*list(slave.Conditions), *list(master.Conditions))
        }
        runtime_connectivity = {
            tuple(sorted(node.Id for node in condition.GetGeometry()))
            for condition in contact_subpart.Conditions
        }
        slave_role = all(node.Is(KM.SLAVE) and not node.Is(KM.MASTER) for node in slave.Nodes)
        master_role = all(node.Is(KM.MASTER) and not node.Is(KM.SLAVE) for node in master.Nodes)
        pair_membership = source_connectivity == runtime_connectivity
        nodal_h_slave = [float(node.GetSolutionStepValue(KM.NODAL_H)) for node in slave.Nodes]
        nodal_h_master = [float(node.GetSolutionStepValue(KM.NODAL_H)) for node in master.Nodes]
        group_checks = {
            "slave_runtime_role": slave_role,
            "master_runtime_role": master_role,
            "contact_submodelpart_matches_exact_pair": pair_membership,
            "slave_nodal_h_positive_finite": bool(nodal_h_slave)
            and all(math.isfinite(value) and value > 0.0 for value in nodal_h_slave),
            "master_nodal_h_positive_finite": bool(nodal_h_master)
            and all(math.isfinite(value) and value > 0.0 for value in nodal_h_master),
        }
        all_checks.extend(group_checks.values())
        groups[group_name] = {
            "pair_index": index,
            "slave": slave_name,
            "master": master_name,
            "contact_submodelpart": contact_path,
            "computing_contact_submodelpart": computing_path,
            "source_condition_ids": sorted(source_condition_ids),
            "contact_submodelpart_condition_ids": sorted(runtime_condition_ids),
            "source_and_runtime_condition_ids_equal": (
                source_condition_ids == runtime_condition_ids
            ),
            "condition_identity_contract": (
                "exact connectivity and node IDs; ALMContactProcess may "
                "renumber generated interface conditions when pair indexes "
                "are not the original three-pair ordering"
            ),
            "slave_node_count": slave.NumberOfNodes(),
            "master_node_count": master.NumberOfNodes(),
            "slave_node_flags": {
                "SLAVE": sum(node.Is(KM.SLAVE) for node in slave.Nodes),
                "MASTER": sum(node.Is(KM.MASTER) for node in slave.Nodes),
            },
            "master_node_flags": {
                "SLAVE": sum(node.Is(KM.SLAVE) for node in master.Nodes),
                "MASTER": sum(node.Is(KM.MASTER) for node in master.Nodes),
            },
            "slave_condition_flags": {
                "SLAVE": sum(condition.Is(KM.SLAVE) for condition in slave.Conditions),
                "MASTER": sum(condition.Is(KM.MASTER) for condition in slave.Conditions),
            },
            "master_condition_flags": {
                "SLAVE": sum(condition.Is(KM.SLAVE) for condition in master.Conditions),
                "MASTER": sum(condition.Is(KM.MASTER) for condition in master.Conditions),
            },
            "slave_mean_runtime_normal": [
                sum(float(node.GetSolutionStepValue(KM.NORMAL)[component]) for node in slave.Nodes)
                / max(slave.NumberOfNodes(), 1)
                for component in range(2)
            ],
            "master_mean_runtime_normal": [
                sum(float(node.GetSolutionStepValue(KM.NORMAL)[component]) for node in master.Nodes)
                / max(master.NumberOfNodes(), 1)
                for component in range(2)
            ],
            "slave_nodal_h": scalar_statistics(nodal_h_slave),
            "master_nodal_h": scalar_statistics(nodal_h_master),
            "checks": group_checks,
        }
    return {
        "group_identification_method": (
            "Kratos ContactSubN/ComputingContactSubN API; no coordinate inference"
        ),
        "groups": groups,
        "all_group_contracts_pass": all(all_checks),
        "initial_penalty": float(model_part.ProcessInfo[KM.INITIAL_PENALTY]),
        "scale_factor": float(model_part.ProcessInfo[KM.SCALE_FACTOR]),
    }



def _edge_positions(model_part: Any, edges: Sequence[BoundaryEdge]) -> dict[int, tuple[float, float]]:
    ids = {node_id for edge in edges for node_id in edge.node_ids}
    return {node_id: (float(model_part.Nodes[node_id].X), float(model_part.Nodes[node_id].Y)) for node_id in ids}


def _master_edges_for_group(
    group_name: str,
    mesh: FingertipMesh,
    indenter_topology: IndenterKratosTopology,
) -> tuple[BoundaryEdge, ...]:
    if group_name == "external_pad_indenter":
        return indenter_topology.contact_edges
    tags = {
        "internal_left": "stem_left",
        "internal_right": "stem_right",
        "internal_bottom": "stem_bottom",
    }
    if group_name == "internal_u":
        return tuple(
            edge
            for tag in ("stem_left", "stem_bottom", "stem_right")
            for edge in mesh.boundary_edges[tag]
        )
    return mesh.boundary_edges[tags[group_name]]


def contact_group_step_metrics(
    model: Any,
    model_part: Any,
    mesh: FingertipMesh,
    indenter_topology: IndenterKratosTopology,
    contact_group_definitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    KM, CSMA, _, _ = import_kratos()
    groups: dict[str, Any] = {}
    for index, (group_name, slave_name, _) in enumerate(
        contact_group_definitions
    ):
        slave = model_part.GetSubModelPart(slave_name)
        computing_path = f"Structure.ComputingContact.ComputingContactSub{index}"
        computing = model[computing_path]
        conditions = list(computing.Conditions)
        active_conditions = [condition for condition in conditions if condition.Is(KM.ACTIVE)]
        active_slave_ids = sorted(node.Id for node in slave.Nodes if node.Is(KM.ACTIVE))
        weighted_gaps = [float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)) for node in slave.Nodes]
        pressures = [
            float(node.GetSolutionStepValue(CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE))
            for node in slave.Nodes
        ]
        normals = {
            node.Id: (
                float(node.GetSolutionStepValue(KM.NORMAL)[0]),
                float(node.GetSolutionStepValue(KM.NORMAL)[1]),
            )
            for node in slave.Nodes
        }
        geometric_gap_node_ids = (
            set(active_slave_ids)
            if group_name == "external_pad_indenter"
            else {node.Id for node in slave.Nodes}
        )
        slave_positions = {
            node.Id: (float(node.X), float(node.Y))
            for node in slave.Nodes
            if node.Id in geometric_gap_node_ids
        }
        gap_normals = {
            node_id: normal
            for node_id, normal in normals.items()
            if node_id in geometric_gap_node_ids
        }
        master_edges = _master_edges_for_group(group_name, mesh, indenter_topology)
        geometric_gap = signed_geometric_gap_statistics(
            slave_positions,
            gap_normals,
            master_edges,
            _edge_positions(model_part, master_edges),
        )
        nodal_h = [float(node.GetSolutionStepValue(KM.NODAL_H)) for node in slave.Nodes]
        local_size = sum(nodal_h) / len(nodal_h)
        penetration_tolerance = max(0.001, 0.05 * local_size)
        groups[group_name] = {
            "pair_index": index,
            "computing_contact_submodelpart": computing_path,
            "generated_condition_count": len(conditions),
            "active_condition_count": len(active_conditions),
            "active_condition_ids": sorted(condition.Id for condition in active_conditions),
            "active_slave_node_ids": active_slave_ids,
            "weighted_gap": scalar_statistics(weighted_gaps),
            "lagrange_multiplier_contact_pressure": scalar_statistics(pressures),
            "signed_geometric_gap": geometric_gap,
            "geometric_gap_node_scope": (
                "active slave nodes"
                if group_name == "external_pad_indenter"
                else "all source slave nodes"
            ),
            "local_contact_mesh_size_mm": local_size,
            "penetration_tolerance_mm": penetration_tolerance,
            "penetration_pass": bool(geometric_gap.get("available"))
            and bool(geometric_gap.get("finite"))
            and float(geometric_gap["maximum_penetration_mm"]) <= penetration_tolerance,
            "slave_nodal_state": [
                {
                    "node_id": node.Id,
                    "active": bool(node.Is(KM.ACTIVE)),
                    "weighted_gap": float(
                        node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)
                    ),
                    "lagrange_multiplier_contact_pressure": float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    ),
                    "normal": [
                        float(
                            node.GetSolutionStepValue(KM.NORMAL)[component]
                        )
                        for component in range(2)
                    ],
                }
                for node in slave.Nodes
            ],
        }
        if group_name == "internal_u":
            semantic_regions: dict[str, Any] = {}
            for region, submodel_part_name in (
                ("left", "PadCutoutLeft"),
                ("bottom", "PadCutoutBottom"),
                ("right", "PadCutoutRight"),
            ):
                semantic = model_part.GetSubModelPart(submodel_part_name)
                region_gaps = [
                    float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP))
                    for node in semantic.Nodes
                ]
                region_pressures = [
                    float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    )
                    for node in semantic.Nodes
                ]
                semantic_regions[region] = {
                    "source_submodelpart": submodel_part_name,
                    "node_count": semantic.NumberOfNodes(),
                    "active_node_ids": sorted(
                        node.Id for node in semantic.Nodes if node.Is(KM.ACTIVE)
                    ),
                    "active_node_count": sum(
                        node.Is(KM.ACTIVE) for node in semantic.Nodes
                    ),
                    "weighted_gap": scalar_statistics(region_gaps),
                    "lagrange_multiplier_contact_pressure": scalar_statistics(
                        region_pressures
                    ),
                }
            groups[group_name]["semantic_regions"] = semantic_regions
    return groups
