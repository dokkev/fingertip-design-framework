"""Solver-facing internal-contact topology for Phase 4I-D.

The semantic left, bottom, and right submodel parts remain untouched.  The
continuous-U option creates aggregate memberships that reuse their existing
nodes and Line2 conditions.
"""

from __future__ import annotations

from typing import Any, Sequence


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
