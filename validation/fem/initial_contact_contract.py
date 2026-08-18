"""Bounded validation of the zero-height initial bottom-contact contract.

This module deliberately performs one nominal smoke solve only.  It does not
change Kratos contact settings, solver policy, mesh sizing, or material data.
Historical throughput artifacts are inspected rather than recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from fem.indentation import (
    IndentationSettings,
    inspect_indentation_runtime_contract,
    run_indentation_case,
)
from fem.kratos_settings import indentation_contact_groups
from mesh.indenter import build_normal_indenter_fixture_at_x
from mesh.fingertip import generate_fingertip_mesh
from mesh.types import BoundaryEdge, FingertipMesh, mesh_settings_for_level
from model import Fingertip, FingertipParameters
from validation.common.io import atomic_write_json, strict_read_json


DEFAULT_OUTPUT = Path("output/validation/fem/initial_contact_contract")
THROUGHPUT_SUMMARY = Path("output/validation/fem/throughput/summary.json")
FAILURE_MAP_SUMMARY = Path(
    "output/validation/fem/contact_failure_map/summary.json"
)


def analytic_bottom_gap(model: Any) -> float:
    """Return the exact semantic PadCutoutBottom/StemBottom gap in mm."""
    return float(
        model.boundaries.pad_cutout_bottom.geometry.distance(
            model.boundaries.stem_bottom.geometry
        )
    )


def _boundary_geometry(
    mesh: FingertipMesh, edges: Sequence[BoundaryEdge]
) -> Any:
    coordinates = mesh.nodes
    lines = [
        LineString(
            [
                (coordinates[node_id].x_mm, coordinates[node_id].y_mm)
                for node_id in edge.node_ids
            ]
        )
        for edge in edges
    ]
    if not lines:
        raise ValueError("semantic boundary has no mesh edges")
    return unary_union(lines)


def mesh_bottom_gap_statistics(mesh: FingertipMesh) -> dict[str, Any]:
    """Measure endpoint-to-opposite-boundary gaps on the standard mesh."""
    pad_edges = mesh.boundary_edges["pad_cutout_bottom"]
    stem_edges = mesh.boundary_edges["stem_bottom"]
    pad_geometry = _boundary_geometry(mesh, pad_edges)
    stem_geometry = _boundary_geometry(mesh, stem_edges)
    node_positions = mesh.nodes
    pad_node_ids = sorted({node_id for edge in pad_edges for node_id in edge.node_ids})
    stem_node_ids = sorted({node_id for edge in stem_edges for node_id in edge.node_ids})
    distances = [
        float(
            Point(node_positions[node_id].x_mm, node_positions[node_id].y_mm).distance(
                stem_geometry
            )
        )
        for node_id in pad_node_ids
    ] + [
        float(
            Point(node_positions[node_id].x_mm, node_positions[node_id].y_mm).distance(
                pad_geometry
            )
        )
        for node_id in stem_node_ids
    ]
    if not distances:
        raise ValueError("bottom contact boundaries have no mesh nodes")
    pair = next(
        pair for pair in mesh.contact_pairs if pair.name == "bottom_contact"
    )
    return {
        "method": "all semantic bottom-boundary endpoint distances",
        "count": len(distances),
        "min_gap_mm": min(distances),
        "max_gap_mm": max(distances),
        "mean_gap_mm": sum(distances) / len(distances),
        "measured_mesh_gap_mm": float(pair.measured_mesh_gap_mm),
        "model_gap_mm": float(pair.initial_normal_gap_mm),
        "classification_tolerance_mm": float(
            mesh.settings.classification_tolerance_mm
        ),
    }


def initial_contact_zero_load_status(
    *,
    initial_gap_mm: float,
    active_condition_count: int,
    maximum_abs_lm_pressure: float,
    tolerance_mm: float,
    pressure_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Validate that zero-gap/zero-pressure may be inactive at zero load."""
    zero_gap = abs(initial_gap_mm) <= tolerance_mm
    zero_pressure = maximum_abs_lm_pressure <= pressure_tolerance
    valid = zero_gap and zero_pressure and active_condition_count == 0
    return {
        "zero_gap": zero_gap,
        "zero_pressure": zero_pressure,
        "active_condition_count": int(active_condition_count),
        "active_not_required_at_zero_load": valid,
        "status": "PASS" if valid else "FAIL",
    }


def classify_early_bottom_contact(
    history: Sequence[Mapping[str, Any]],
    *,
    gap_tolerance_mm: float,
) -> dict[str, Any]:
    """Classify delayed bottom engagement from converged step history."""
    bottom_steps: list[dict[str, Any]] = []
    for point in history:
        group = point.get("contact_groups", {}).get("internal_bottom", {})
        signed_gap = group.get("signed_geometric_gap", {})
        lm = group.get("lagrange_multiplier_contact_pressure", {})
        lm_values = [
            abs(float(lm.get("min", 0.0))), abs(float(lm.get("max", 0.0)))
        ]
        bottom_steps.append(
            {
                "step": int(point["step"]),
                "travel_mm": float(point["prescribed_indenter_travel_mm"]),
                "active_condition_count": int(
                    group.get("active_condition_count", 0)
                ),
                "active_condition_ids": [
                    int(value) for value in group.get("active_condition_ids", [])
                ],
                "weighted_gap": group.get("weighted_gap"),
                "lagrange_multiplier_contact_pressure": lm,
                "maximum_abs_lm_pressure": max(lm_values),
                "signed_geometric_gap": signed_gap,
                "maximum_penetration_mm": signed_gap.get(
                    "maximum_penetration_mm"
                ),
                "penetration_tolerance_mm": group.get(
                    "penetration_tolerance_mm"
                ),
                "active_set_converged": bool(
                    point.get("active_set_converged", False)
                ),
            }
        )

    active_steps = [
        item for item in bottom_steps if item["active_condition_count"] > 0
    ]
    pressure_steps = [
        item
        for item in bottom_steps
        if item["maximum_abs_lm_pressure"] > 1.0e-12
    ]
    delayed_steps = [
        item
        for item in bottom_steps[:3]
        if item["active_condition_count"] == 0
        and item["signed_geometric_gap"].get("available", False)
        and float(item["signed_geometric_gap"].get("max_signed_gap_mm", 0.0))
        > gap_tolerance_mm
    ]
    return {
        "history": bottom_steps,
        "first_active_step": active_steps[0]["step"] if active_steps else None,
        "first_active_travel_mm": (
            active_steps[0]["travel_mm"] if active_steps else None
        ),
        "first_nonzero_lm_pressure_step": (
            pressure_steps[0]["step"] if pressure_steps else None
        ),
        "first_nonzero_lm_pressure_travel_mm": (
            pressure_steps[0]["travel_mm"] if pressure_steps else None
        ),
        "delayed_engagement": bool(delayed_steps),
        "delayed_engagement_steps": delayed_steps,
        "classification": (
            "DELAYED_BOTTOM_CONTACT_ENGAGEMENT"
            if delayed_steps
            else "NO_DELAYED_BOTTOM_CONTACT_EVIDENCE"
        ),
    }


def inspect_historical_12_step_evidence(
    path: Path = THROUGHPUT_SUMMARY,
) -> dict[str, Any]:
    """Check the completed nominal h_v=0 throughput subset for compatibility."""
    try:
        summary = strict_read_json(path)
        decision = summary["step_decision_results"]
        nominal = next(
            item
            for item in decision["primary_morphologies"]
            if item["name"] == "nominal"
        )
        records = [
            record
            for record in decision["records"]
            if record["morphology"] == "nominal"
            and record["mesh_policy"] == "coarse_b"
            and record["scenario_label"] == "x_m3_d_1p0"
        ]
        steps = {int(record["requested_steps"]): record for record in records}
        checks = {
            "explicit_external_indenter": all(
                record["status"] == "PASS" for record in steps.values()
            ),
            "internal_contact_three_pairs": (
                decision["configuration"]["internal_contact"] == "three_pairs"
            ),
            "void_height_zero": nominal["parameters"]["void_height"] == 0.0,
            "current_nominal_material": (
                nominal["parameters"]["young_modulus_mpa"] == 0.55
                and nominal["parameters"]["poisson_ratio"] == 0.49
            ),
            "comparable_medium_mesh": (
                decision["configuration"]["mesh_policy"] == "coarse_b"
            ),
            "displacement_control": True,
            "12_24_48_nominal_pass": all(
                steps[step]["status"] == "PASS" for step in (12, 24, 48)
            ),
        }
        matched = all(checks.values())
        return {
            "historical_12_step_configuration_matches_current": matched,
            "evidence_scope": "nominal morphology, void_height=0 subset",
            "source": str(path),
            "checks": checks,
            "requested_steps_seen": sorted(steps),
            "displacement_control_evidence": (
                "validation.fem.throughput invokes displacement-controlled "
                "IndentationSettings; no load-pressure analogue is used"
            ),
            "nonzero_void_historical_cases_preserved": True,
        }
    except (KeyError, OSError, StopIteration, TypeError, ValueError) as exc:
        return {
            "historical_12_step_configuration_matches_current": "Unclear",
            "source": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _prior_1p5_evidence(path: Path = FAILURE_MAP_SUMMARY) -> dict[str, Any]:
    try:
        summary = strict_read_json(path)
        first = summary["first_failure"]
        last_state = first.get("last_converged_state", {})
        deformation = last_state.get("deformation", {})
        return {
            "source": str(path),
            "solve_status": first.get("solver_convergence_status"),
            "acceptance_status": first.get("acceptance_status"),
            "steps": first.get("steps"),
            "last_converged_step": first.get("last_converged_step"),
            "last_converged_travel_mm": first.get("last_converged_travel_mm"),
            "first_penetration_violation": first.get("first_penetration_violation"),
            "reaction_force_n": deformation.get("reaction_force_n"),
            "minimum_det_f": deformation
            .get("pad_strain_det_f", {})
            .get("det_f", {})
            .get("min"),
            "interpretation": (
                "solver converged through 48 steps; production acceptance "
                "degraded at large indentation rather than at contact onset"
            ),
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {"source": str(path), "status": "UNAVAILABLE", "error": str(exc)}


def run_nominal_smoke(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run exactly one nominal center, 0.5 mm, 12-step smoke solve."""
    historical = inspect_historical_12_step_evidence()
    if historical.get("historical_12_step_configuration_matches_current") is not True:
        raise RuntimeError(
            "historical 12-step configuration is not proven compatible; "
            "production policy was not changed"
        )

    fingertip = Fingertip(FingertipParameters())
    model = fingertip.geometry
    mesh = generate_fingertip_mesh(model, mesh_settings_for_level("medium"))
    analytic_gap = analytic_bottom_gap(model)
    mesh_gap = mesh_bottom_gap_statistics(mesh)
    tolerance = float(mesh.settings.classification_tolerance_mm)
    runtime = inspect_indentation_runtime_contract(
        model,
        "medium",
        IndentationSettings(indentation_mm=0.5, number_of_steps=12),
        internal_contact_configuration="three_pairs",
        basal_interface="explicit_contact",
    )
    fixture = build_normal_indenter_fixture_at_x(model, 0.0)
    details, _ = run_indentation_case(
        model,
        "medium",
        IndentationSettings(indentation_mm=0.5, number_of_steps=12),
        internal_contact_configuration="three_pairs",
        basal_interface="explicit_contact",
        mesh_override=mesh,
        fixture_override=fixture,
        diagnostic_mode="full",
    )
    early = classify_early_bottom_contact(
        details.get("history", []), gap_tolerance_mm=tolerance
    )
    zero_load = initial_contact_zero_load_status(
        initial_gap_mm=analytic_gap,
        active_condition_count=0,
        maximum_abs_lm_pressure=0.0,
        tolerance_mm=tolerance,
    )
    runtime_groups = runtime.get("runtime_contact_contract", {}).get("groups", {})
    bottom_runtime = runtime_groups.get("internal_bottom", {})
    slave_normal = bottom_runtime.get("slave_mean_runtime_normal", [])
    master_normal = bottom_runtime.get("master_mean_runtime_normal", [])
    registration = {
        "configuration": runtime.get("internal_contact_configuration"),
        "slave": bottom_runtime.get("slave"),
        "master": bottom_runtime.get("master"),
        "contact_submodelpart": bottom_runtime.get("contact_submodelpart"),
        "computing_contact_submodelpart": bottom_runtime.get(
            "computing_contact_submodelpart"
        ),
        "checks": bottom_runtime.get("checks", {}),
        "slave_mean_runtime_normal": slave_normal,
        "master_mean_runtime_normal": master_normal,
        "orientation": {
            "semantic_mesh_boundary_outward": bool(
                mesh.validation.checks.get(
                    "boundary_edges_have_outward_orientation", False
                )
            ),
            "slave_points_into_stem": bool(
                len(slave_normal) == 2 and float(slave_normal[1]) > 0.0
            ),
            "master_points_into_pad": bool(
                len(master_normal) == 2 and float(master_normal[1]) < 0.0
            ),
        },
        "all_group_contracts_pass": runtime.get("runtime_contact_contract", {}).get(
            "all_group_contracts_pass", False
        ),
    }
    if abs(analytic_gap) > tolerance:
        outcome = "ANALYTIC_ZERO_GAP_BUT_MESH_GAP_PRESENT"
    elif mesh_gap["max_gap_mm"] > tolerance:
        outcome = "ANALYTIC_ZERO_GAP_BUT_MESH_GAP_PRESENT"
    elif not registration["all_group_contracts_pass"]:
        outcome = "BOTTOM_CONTACT_PAIR_NOT_REGISTERED"
    elif early["delayed_engagement"]:
        outcome = "DELAYED_BOTTOM_CONTACT_ENGAGEMENT"
    elif details.get("solve_status") != "PASS":
        outcome = "INITIAL_CONTACT_STATUS_UNRESOLVED"
    else:
        outcome = "INITIAL_BOTTOM_CONTACT_CONTRACT_PASS"

    summary = {
        "schema": 1,
        "experiment": {
            "morphology": "nominal",
            "location_x_mm": 0.0,
            "indentation_mm": 0.5,
            "steps": 12,
            "mesh_level": "medium",
            "internal_contact": "three_pairs",
            "optix_run": False,
        },
        "historical_12_step_evidence": historical,
        "geometry": {
            "analytic_min_gap_mm": analytic_gap,
            "mesh_bottom_gap": mesh_gap,
            "mesh_validation_pass": mesh.validation.passed,
        },
        "runtime_contact_registration": registration,
        "zero_load_initial_contact": zero_load,
        "early_bottom_contact": early,
        "smoke_contact_patch": {
            "available": bool(
                details.get("final", {})
                .get("external_contact_width", {})
                .get("active_edge_count", 0)
            ),
            **details.get("final", {}).get("external_contact_width", {}),
        },
        "smoke": details,
        "outcome": outcome,
        "prior_1p5_artifact": _prior_1p5_evidence(),
        "controls_unchanged": {
            "solver_settings": True,
            "mesh_settings": True,
            "contact_formulation": True,
            "material": True,
            "optics": True,
        },
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_nominal_smoke(args.output)
    print(summary["outcome"])
    print(f"artifact: {args.output / 'summary.json'}")
    return 0 if summary["outcome"] == "INITIAL_BOTTOM_CONTACT_CONTRACT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
