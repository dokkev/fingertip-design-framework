"""Pure bookkeeping and metrics for the Phase 4J no-void baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mesh.types import FingertipMesh
from model.fingertip_model import FingertipModel
from validation.common.io import strict_read_json


def completed_case_result(path: Path, case_name: str) -> dict[str, Any] | None:
    """Return a valid completed case artifact, otherwise request a rerun."""
    if not path.is_file():
        return None
    try:
        result = strict_read_json(path)
    except (ValueError, OSError):
        return None
    if (
        result.get("phase") != "4J"
        or result.get("case_name") != case_name
        or result.get("status") not in {"PASS", "FAIL", "TIMEOUT", "SKIPPED"}
    ):
        return None
    if result["status"] == "PASS" and (
        not isinstance(result.get("history"), list)
        or not result["history"]
        or not (path.parent / "history.csv").is_file()
    ):
        return None
    return result


def no_void_geometry_contract(model: FingertipModel) -> dict[str, Any]:
    """Record why the existing default model is the bounded 4J reference."""
    summary = model.summary()
    return {
        "configuration_source": "FingertipParameters defaults",
        "void_width_mm": model.parameters.void_width,
        "void_height_mm": model.parameters.void_height,
        "void_classification": model.classify_void(),
        "void_geometry_is_none": model.void_geometry is None,
        "void_area_mm2": summary["void_area"],
        "removed_material_area_mm2": summary["removed_material_area"],
        "internal_contact_configuration": "none",
        "geometry_redesigned": False,
        "pass": (
            model.parameters.void_width == 0.0
            and model.parameters.void_height == 0.0
            and model.void_geometry is None
            and float(summary["void_area"]) == 0.0
        ),
    }


def reaction_work_proxy(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Integrate the reaction curve without mislabeling it as strain energy."""
    points = [(0.0, 0.0)]
    points.extend(
        (
            float(point["achieved_indentation_mm"]),
            float(point["indenter_normal_reaction_n"]),
        )
        for point in history
    )
    work = sum(
        0.5 * (first[1] + second[1]) * (second[0] - first[0])
        for first, second in zip(points, points[1:])
    )
    return {
        "available": bool(history) and math.isfinite(work),
        "value_n_mm": work if math.isfinite(work) else None,
        "interpretation": (
            "trapezoidal external reaction work proxy; not reported as "
            "Kratos element STRAIN_ENERGY"
        ),
    }


def achieved_contact_centroid(
    mesh: FingertipMesh,
    active_slave_node_ids: Sequence[int],
) -> dict[str, Any]:
    """Compute the reference-space centroid of the active pad contact nodes."""
    active = [
        mesh.nodes[node_id]
        for node_id in active_slave_node_ids
        if node_id in mesh.nodes
    ]
    if not active:
        return {
            "available": False,
            "reference_centroid_mm": None,
            "active_slave_node_count": 0,
        }
    return {
        "available": True,
        "reference_centroid_mm": [
            sum(node.x_mm for node in active) / len(active),
            sum(node.y_mm for node in active) / len(active),
        ],
        "active_slave_node_count": len(active),
        "coordinate_interpretation": (
            "reference coordinates of final active PadOuterArc slave nodes"
        ),
    }


def append_or_replace_case(
    run_state: dict[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Update one named checkpoint while preserving deterministic queue order."""
    cases = list(run_state.setdefault("cases", []))
    cases = [
        existing
        for existing in cases
        if existing.get("case_name") != record.get("case_name")
    ]
    cases.append(dict(record))
    order = {"J0": 0, "J1": 1, "J2": 2, "J3": 3, "J4": 4}
    cases.sort(key=lambda item: order.get(str(item.get("case_name")), 99))
    run_state["cases"] = cases
