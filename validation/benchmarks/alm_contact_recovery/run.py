#!/usr/bin/env python3
"""Diagnose and retry the Phase 3 ALM contact benchmark.

The accepted Phase 3 JSON is an immutable input to this recovery study.  The
script first reproduces and instruments the 48-step fine-mesh failure, then
uses runtime MASTER/SLAVE flags to choose the direction required by Phase 3R
and runs the same three meshes with 96 common displacement increments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import KratosMultiphysics as KM
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)

import validation.benchmarks.localized_contact.run as phase3


BASELINE_STEPS = 48
RECOVERY_STEPS = 96
BASELINE_FINE_FAILURE_STEP = 23
BASELINE_SLAVE_SURFACE = "IndenterSurface"
RECOVERY_REQUIRED_SLAVE_SURFACE = "BlockTop"
ENERGY_NEGLIGIBLE_TOLERANCE_N_MM = 1.0e-12
CARRIER_STRAIN_TOLERANCE = 1.0e-12
CARRIER_DET_F_TOLERANCE = 1.0e-12
PRESSURE_ROUGHNESS_LIMIT = 0.5
MEDIUM_FINE_REACTION_TOLERANCE = 0.10
MEDIUM_STEP_REFINEMENT_TOLERANCE = 0.01


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--baseline", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)

    diagnosis_parser = subparsers.add_parser("diagnose-fine")
    diagnosis_parser.add_argument("--output", required=True, type=Path)

    recovery_diagnosis_parser = subparsers.add_parser("diagnose96")
    recovery_diagnosis_parser.add_argument(
        "--mesh-level", required=True, choices=phase3.MESH_LEVELS
    )
    recovery_diagnosis_parser.add_argument(
        "--slave-surface",
        required=True,
        choices=("BlockTop", "IndenterSurface"),
    )
    recovery_diagnosis_parser.add_argument(
        "--target-step", required=True, type=int
    )
    recovery_diagnosis_parser.add_argument("--output", required=True, type=Path)

    case_parser = subparsers.add_parser("case96")
    case_parser.add_argument(
        "--mesh-level", required=True, choices=phase3.MESH_LEVELS
    )
    case_parser.add_argument(
        "--slave-surface",
        required=True,
        choices=("BlockTop", "IndenterSurface"),
    )
    case_parser.add_argument("--output", required=True, type=Path)

    medium_48_parser = subparsers.add_parser("case48-medium")
    medium_48_parser.add_argument(
        "--slave-surface",
        required=True,
        choices=("BlockTop", "IndenterSurface"),
    )
    medium_48_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_parameters(number_of_steps: int, slave_surface: str) -> KM.Parameters:
    parameters = phase3._project_parameters()
    parameters["problem_data"]["end_time"].SetDouble(float(number_of_steps))
    parameters["processes"]["contact_process_list"][0]["Parameters"][
        "assume_master_slave"
    ]["0"][0].SetString(slave_surface)
    return parameters


def _coordinate_range(nodes: list[Any]) -> dict[str, Any]:
    if not nodes:
        return {"count": 0, "x0": None, "y0": None, "x": None, "y": None}
    return {
        "count": len(nodes),
        "x0": [min(node.X0 for node in nodes), max(node.X0 for node in nodes)],
        "y0": [min(node.Y0 for node in nodes), max(node.Y0 for node in nodes)],
        "x": [min(node.X for node in nodes), max(node.X for node in nodes)],
        "y": [min(node.Y for node in nodes), max(node.Y for node in nodes)],
    }


def _scalar_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    finite = all(math.isfinite(value) for value in values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "finite": finite,
    }


def _matching_contact_conditions(
    contact_part: KM.ModelPart | None, surface_node_ids: set[int]
) -> list[Any]:
    if contact_part is None:
        return []
    matching = []
    for condition in contact_part.Conditions:
        geometry_node_ids = {node.Id for node in condition.GetGeometry()}
        if geometry_node_ids and geometry_node_ids.issubset(surface_node_ids):
            matching.append(condition)
    return matching


def _surface_runtime_contract(
    model_part: KM.ModelPart,
    contact_part: KM.ModelPart | None,
    surface_name: str,
    node_ids: list[int],
) -> dict[str, Any]:
    nodes = [model_part.Nodes[node_id] for node_id in node_ids]
    conditions = _matching_contact_conditions(contact_part, set(node_ids))
    nodal_h = [
        float(node.GetSolutionStepValue(KM.NODAL_H)) for node in nodes
    ]
    return {
        "surface": surface_name,
        "node_ids": sorted(node_ids),
        "node_coordinates": _coordinate_range(nodes),
        "node_flags": {
            "MASTER": sum(node.Is(KM.MASTER) for node in nodes),
            "SLAVE": sum(node.Is(KM.SLAVE) for node in nodes),
            "ACTIVE": sum(node.Is(KM.ACTIVE) for node in nodes),
        },
        "contact_condition_ids": sorted(condition.Id for condition in conditions),
        "condition_coordinates": _coordinate_range(
            [node for condition in conditions for node in condition.GetGeometry()]
        ),
        "condition_flags": {
            "MASTER": sum(condition.Is(KM.MASTER) for condition in conditions),
            "SLAVE": sum(condition.Is(KM.SLAVE) for condition in conditions),
            "ACTIVE": sum(condition.Is(KM.ACTIVE) for condition in conditions),
        },
        "multiplier_storage": {
            "historical_variable_count": sum(
                node.SolutionStepsDataHas(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                )
                for node in nodes
            ),
            "dof_count": sum(
                node.HasDofFor(CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE)
                for node in nodes
            ),
        },
        "nodal_h": _scalar_statistics(nodal_h),
    }


def _classify_runtime_surfaces(
    model: KM.Model,
    model_part: KM.ModelPart,
    mesh_data: dict[str, Any],
) -> dict[str, Any]:
    contact_part = phase3._contact_model_part(model, "Structure.Contact")
    surfaces = {
        "BlockTop": _surface_runtime_contract(
            model_part,
            contact_part,
            "BlockTop",
            mesh_data["top_node_ids"],
        ),
        "IndenterSurface": _surface_runtime_contract(
            model_part,
            contact_part,
            "IndenterSurface",
            mesh_data["indenter_surface_node_ids"],
        ),
    }
    slave_surfaces = [
        name
        for name, data in surfaces.items()
        if data["node_flags"]["SLAVE"] > 0
        and data["node_flags"]["MASTER"] == 0
    ]
    master_surfaces = [
        name
        for name, data in surfaces.items()
        if data["node_flags"]["MASTER"] > 0
        and data["node_flags"]["SLAVE"] == 0
    ]
    process_info = model_part.ProcessInfo
    return {
        "surfaces": surfaces,
        "actual_slave_surface": (
            slave_surfaces[0] if len(slave_surfaces) == 1 else None
        ),
        "actual_master_surface": (
            master_surfaces[0] if len(master_surfaces) == 1 else None
        ),
        "slave_surface_candidates": slave_surfaces,
        "master_surface_candidates": master_surfaces,
        "multiplier_interpretation": (
            "The historical variable and DOF are allocated on both surfaces; "
            "runtime SLAVE flags determine the ALM multiplier owner."
        ),
        "initial_penalty": float(process_info[KM.INITIAL_PENALTY]),
        "scale_factor": float(process_info[KM.SCALE_FACTOR]),
    }


def _vector_l2_norm(vector: Any) -> float:
    return math.sqrt(sum(float(vector[index]) ** 2 for index in range(len(vector))))


def _node_iteration_values(node: Any) -> dict[str, Any]:
    normal = node.GetSolutionStepValue(KM.NORMAL)
    return {
        "weighted_gap": float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)),
        "lagrange_multiplier_contact_pressure": float(
            node.GetSolutionStepValue(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
        ),
        "augmented_normal_contact_pressure": float(
            node.GetValue(CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE)
        ),
        "nodal_area": float(node.GetValue(KM.NODAL_AREA)),
        "nodal_h": float(node.GetSolutionStepValue(KM.NODAL_H)),
        "normal": [float(normal[index]) for index in range(3)],
    }


def _integrated_slave_contact_reaction_y(
    model_part: KM.ModelPart, slave_node_ids: list[int]
) -> float:
    reaction_y = 0.0
    for node_id in slave_node_ids:
        node = model_part.Nodes[node_id]
        pressure = float(
            node.GetSolutionStepValue(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
        )
        nodal_area = float(node.GetValue(KM.NODAL_AREA))
        normal_y = float(node.GetSolutionStepValue(KM.NORMAL)[1])
        reaction_y += pressure * nodal_area * normal_y
    return reaction_y


def _capture_iteration_state(
    model_part: KM.ModelPart,
    strategy: Any,
    indenter_node_ids: list[int],
    iteration: int,
    previous_active_ids: set[int],
    criterion_converged: bool,
) -> tuple[dict[str, Any], set[int]]:
    actual_slave_ids = sorted(
        node.Id for node in model_part.Nodes if node.Is(KM.SLAVE)
    )
    active_ids = {
        node_id
        for node_id in actual_slave_ids
        if model_part.Nodes[node_id].Is(KM.ACTIVE)
    }
    changed_ids = sorted(previous_active_ids.symmetric_difference(active_ids))
    residual = strategy.GetSystemVector()
    record = {
        "step": int(model_part.ProcessInfo[KM.STEP]),
        "iteration": iteration,
        "active_slave_node_ids": sorted(active_ids),
        "changed_slave_node_ids": changed_ids,
        "global_residual_l2_norm": _vector_l2_norm(residual),
        "criterion_converged": criterion_converged,
        "active_set_converged": bool(
            model_part.ProcessInfo[CSMA.ACTIVE_SET_CONVERGED]
        ),
        "stored_indenter_reaction_y_n": phase3._reaction_sum(
            model_part, indenter_node_ids
        ),
        "lm_integrated_slave_contact_reaction_y_n": (
            _integrated_slave_contact_reaction_y(model_part, actual_slave_ids)
        ),
        "transition_node_values": {
            str(node_id): _node_iteration_values(model_part.Nodes[node_id])
            for node_id in changed_ids
        },
    }
    return record, active_ids


def _solve_one_update_at_a_time(
    solver: Any,
    model_part: KM.ModelPart,
    indenter_node_ids: list[int],
    total_iteration_limit: int,
) -> tuple[bool, list[dict[str, Any]]]:
    strategy = solver._GetSolutionStrategy()
    original_limit = int(strategy.GetMaxIterationNumber())
    previous_active_ids = {
        node.Id
        for node in model_part.Nodes
        if node.Is(KM.SLAVE) and node.Is(KM.ACTIVE)
    }
    records: list[dict[str, Any]] = []
    converged = False
    strategy.SetMaxIterationNumber(1)
    try:
        for iteration in range(1, total_iteration_limit + 1):
            converged = bool(solver.SolveSolutionStep())
            record, previous_active_ids = _capture_iteration_state(
                model_part,
                strategy,
                indenter_node_ids,
                iteration,
                previous_active_ids,
                converged,
            )
            records.append(record)
            if converged:
                break
    finally:
        strategy.SetMaxIterationNumber(original_limit)
        model_part.ProcessInfo[KM.NL_ITERATION_NUMBER] = len(records)
    return converged, records


def _active_set_cycle_analysis(
    iteration_records: list[dict[str, Any]], target_step: int
) -> dict[str, Any]:
    records = [record for record in iteration_records if record["step"] == target_step]
    signatures = [tuple(record["active_slave_node_ids"]) for record in records]
    unique: dict[tuple[int, ...], list[int]] = {}
    for record, signature in zip(records, signatures):
        unique.setdefault(signature, []).append(record["iteration"])

    alternating_events = [
        index
        for index in range(2, len(signatures))
        if signatures[index] == signatures[index - 2]
        and signatures[index] != signatures[index - 1]
    ]
    alternating_suffix_start = None
    alternating_states: list[list[int]] = []
    for start in range(max(0, len(signatures) - 20), len(signatures) - 3):
        first = signatures[start]
        second = signatures[start + 1]
        if first == second:
            continue
        if all(
            signatures[index] == (first if (index - start) % 2 == 0 else second)
            for index in range(start, len(signatures))
        ):
            alternating_suffix_start = records[start]["iteration"]
            alternating_states = [list(first), list(second)]
            break

    periodic_cycle: dict[str, Any] | None = None
    for period in range(2, min(16, len(signatures) // 2 + 1)):
        for start in range(0, len(signatures) - 2 * period + 1):
            if all(
                signatures[index]
                == signatures[start + (index - start) % period]
                for index in range(start, len(signatures))
            ):
                periodic_cycle = {
                    "start_iteration": records[start]["iteration"],
                    "period": period,
                    "state_sequence": [
                        list(signature)
                        for signature in signatures[start : start + period]
                    ],
                    "covered_iteration_count": len(signatures) - start,
                    "complete_repetitions": (len(signatures) - start) // period,
                }
                break
        if periodic_cycle is not None:
            break

    two_state_cycle = (
        periodic_cycle is not None
        and periodic_cycle["period"] == 2
        and not records[-1]["criterion_converged"]
    )
    return {
        "target_step": target_step,
        "iteration_count": len(records),
        "unique_active_sets": [
            {"active_slave_node_ids": list(signature), "iterations": iterations}
            for signature, iterations in unique.items()
        ],
        "period_two_repeat_event_count": len(alternating_events),
        "period_two_repeat_iterations": [
            records[index]["iteration"] for index in alternating_events
        ],
        "alternating_tail_start_iteration": alternating_suffix_start,
        "alternating_tail_states": alternating_states,
        "periodic_cycle": periodic_cycle,
        "periodic_cycle_detected": (
            periodic_cycle is not None and not records[-1]["criterion_converged"]
        ),
        "two_state_cycle_detected": two_state_cycle,
        "cycle_classification": (
            "two_state_cycle"
            if two_state_cycle
            else (
                f"period_{periodic_cycle['period']}_multi_state_cycle"
                if periodic_cycle is not None
                else "no_periodic_cycle_detected"
            )
        ),
    }


def _contact_field_statistics(
    model_part: KM.ModelPart, slave_node_ids: list[int]
) -> dict[str, Any]:
    weighted_gap = [
        float(model_part.Nodes[node_id].GetSolutionStepValue(CSMA.WEIGHTED_GAP))
        for node_id in slave_node_ids
    ]
    pressure = [
        float(
            model_part.Nodes[node_id].GetSolutionStepValue(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
        )
        for node_id in slave_node_ids
    ]
    augmented = [
        float(
            model_part.Nodes[node_id].GetValue(
                CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE
            )
        )
        for node_id in slave_node_ids
    ]
    return {
        "weighted_gap": _scalar_statistics(weighted_gap),
        "lagrange_multiplier_contact_pressure": _scalar_statistics(pressure),
        "augmented_normal_contact_pressure": _scalar_statistics(augmented),
        "finite": all(
            math.isfinite(value)
            for values in (weighted_gap, pressure, augmented)
            for value in values
        ),
    }


def _matrix_2x2_deformation_gradient(geometry: Any) -> list[list[float]]:
    first, second, third = geometry[0], geometry[1], geometry[2]
    reference = [
        [second.X0 - first.X0, third.X0 - first.X0],
        [second.Y0 - first.Y0, third.Y0 - first.Y0],
    ]
    current = [
        [second.X - first.X, third.X - first.X],
        [second.Y - first.Y, third.Y - first.Y],
    ]
    determinant = reference[0][0] * reference[1][1] - reference[0][1] * reference[1][0]
    if abs(determinant) <= 1.0e-15:
        raise RuntimeError("Carrier element has zero reference area")
    inverse = [
        [reference[1][1] / determinant, -reference[0][1] / determinant],
        [-reference[1][0] / determinant, reference[0][0] / determinant],
    ]
    return [
        [
            current[row][0] * inverse[0][column]
            + current[row][1] * inverse[1][column]
            for column in range(2)
        ]
        for row in range(2)
    ]


def _carrier_validation(
    model_part: KM.ModelPart, mesh_data: dict[str, Any], divisions: int
) -> dict[str, Any]:
    determinant_values: list[float] = []
    strain_norms: list[float] = []
    hyperelastic_energy = 0.0
    internal_rhs_norms: list[float] = []
    validation_errors: list[str] = []
    shear_modulus = phase3.YOUNG_MODULUS_MPA / (
        2.0 * (1.0 + phase3.POISSON_RATIO)
    )
    lame_lambda = (
        phase3.YOUNG_MODULUS_MPA
        * phase3.POISSON_RATIO
        / (
            (1.0 + phase3.POISSON_RATIO)
            * (1.0 - 2.0 * phase3.POISSON_RATIO)
        )
    )
    for element_id in mesh_data["indenter_carrier_element_ids"]:
        element = model_part.Elements[element_id]
        geometry = element.GetGeometry()
        deformation_gradient = _matrix_2x2_deformation_gradient(geometry)
        determinant_f = (
            deformation_gradient[0][0] * deformation_gradient[1][1]
            - deformation_gradient[0][1] * deformation_gradient[1][0]
        )
        c00 = deformation_gradient[0][0] ** 2 + deformation_gradient[1][0] ** 2
        c01 = (
            deformation_gradient[0][0] * deformation_gradient[0][1]
            + deformation_gradient[1][0] * deformation_gradient[1][1]
        )
        c11 = deformation_gradient[0][1] ** 2 + deformation_gradient[1][1] ** 2
        strain_components = (0.5 * (c00 - 1.0), 0.5 * c01, 0.5 * (c11 - 1.0))
        strain_norm = math.sqrt(
            strain_components[0] ** 2
            + 2.0 * strain_components[1] ** 2
            + strain_components[2] ** 2
        )
        reference_area = 0.5 * abs(
            phase3._triangle_determinant(
                [(node.X0, node.Y0) for node in geometry]
            )
        )
        if determinant_f > 0.0:
            log_j = math.log(determinant_f)
            plane_strain_i1 = c00 + c11 + 1.0
            energy_density = (
                0.5 * shear_modulus * (plane_strain_i1 - 3.0)
                - shear_modulus * log_j
                + 0.5 * lame_lambda * log_j**2
            )
            hyperelastic_energy += (
                energy_density
                * reference_area
                * phase3.INDENTER_CARRIER_THICKNESS_MM
            )
        else:
            hyperelastic_energy = math.inf
        determinant_values.append(determinant_f)
        strain_norms.append(strain_norm)
        try:
            rhs = KM.Vector()
            element.CalculateRightHandSide(rhs, model_part.ProcessInfo)
            internal_rhs_norms.append(_vector_l2_norm(rhs))
        except Exception as exception:
            validation_errors.append(
                f"element_{element_id}_rhs: {type(exception).__name__}: {exception}"
            )

    maximum_strain = max(strain_norms) if strain_norms else None
    maximum_det_f_error = max(
        (abs(value - 1.0) for value in determinant_values), default=None
    )
    maximum_internal_rhs = max(internal_rhs_norms) if internal_rhs_norms else None
    energy_negligible = (
        math.isfinite(hyperelastic_energy)
        and abs(hyperelastic_energy) <= ENERGY_NEGLIGIBLE_TOLERANCE_N_MM
    )
    return {
        "element": "TotalLagrangianElement2D3N kinematic carrier only",
        "mesh_rule": {
            "segments_equal_block_parametric_divisions": divisions,
            "elements_per_segment": 2,
            "thickness_mm": phase3.INDENTER_CARRIER_THICKNESS_MM,
            "case_specific_tuning": False,
        },
        "element_count": len(mesh_data["indenter_carrier_element_ids"]),
        "det_f": {
            "min": min(determinant_values),
            "max": max(determinant_values),
            "maximum_abs_error_from_one": maximum_det_f_error,
            "negative_count": sum(value <= 0.0 for value in determinant_values),
        },
        "maximum_green_lagrange_strain_frobenius_norm": maximum_strain,
        "maximum_element_internal_rhs_l2_norm": maximum_internal_rhs,
        "strain_energy": {
            "method": (
                "compressible Neo-Hookean plane-strain potential evaluated "
                "from the affine element deformation gradient"
            ),
            "kratos_strain_energy_api_used": False,
            "value_n_mm": hyperelastic_energy,
            "negligible_tolerance_n_mm": ENERGY_NEGLIGIBLE_TOLERANCE_N_MM,
            "negligible": energy_negligible,
        },
        "validation_errors": validation_errors,
        "block_statistics_exclusion": {
            "carrier_elements_disjoint_from_block_elements": set(
                mesh_data["indenter_carrier_element_ids"]
            ).isdisjoint(mesh_data["block_element_ids"]),
            "block_det_f_element_count": len(mesh_data["block_element_ids"]),
        },
        "pass": (
            maximum_strain is not None
            and maximum_strain <= CARRIER_STRAIN_TOLERANCE
            and maximum_det_f_error is not None
            and maximum_det_f_error <= CARRIER_DET_F_TOLERANCE
            and maximum_internal_rhs is not None
            and maximum_internal_rhs <= ENERGY_NEGLIGIBLE_TOLERANCE_N_MM
            and energy_negligible
            and not validation_errors
            and set(mesh_data["indenter_carrier_element_ids"]).isdisjoint(
                mesh_data["block_element_ids"]
            )
        ),
    }


def _surface_nonzero_multiplier_nodes(
    model_part: KM.ModelPart, mesh_data: dict[str, Any]
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for name, node_ids in (
        ("BlockTop", mesh_data["top_node_ids"]),
        ("IndenterSurface", mesh_data["indenter_surface_node_ids"]),
    ):
        result[name] = [
            node_id
            for node_id in node_ids
            if abs(
                float(
                    model_part.Nodes[node_id].GetSolutionStepValue(
                        CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                    )
                )
            )
            > phase3.LOAD_BEARING_CONTACT_TOLERANCE
        ]
    return result


def _run_model(
    mesh_level: str,
    number_of_steps: int,
    slave_surface: str,
    detailed_step: int | None,
) -> dict[str, Any]:
    divisions = phase3.MESH_LEVELS[mesh_level]
    result: dict[str, Any] = {
        "mesh_level": mesh_level,
        "number_of_steps": number_of_steps,
        "requested_slave_surface": slave_surface,
        "status": "FAIL",
        "curve": [],
    }
    start = time.perf_counter()
    analysis: StructuralMechanicsAnalysis | None = None
    initialized = False
    iteration_records: list[dict[str, Any]] = []
    try:
        model = KM.Model()
        parameters = _project_parameters(number_of_steps, slave_surface)
        analysis = StructuralMechanicsAnalysis(model, parameters)
        model_part = model["Structure"]
        phase3._create_block_mesh(model_part, divisions)
        phase3._configure_material(model_part)
        mesh_data = phase3._create_submodel_parts_and_indenter(
            model_part, divisions
        )
        analysis.Initialize()
        initialized = True
        phase3._fix_kinematic_dofs(model_part, mesh_data)

        runtime_contract = _classify_runtime_surfaces(
            model, model_part, mesh_data
        )
        result["runtime_contact_contract"] = runtime_contract
        actual_slave_surface = runtime_contract["actual_slave_surface"]
        if actual_slave_surface is None:
            raise RuntimeError("Runtime MASTER/SLAVE surface classification is ambiguous")
        actual_slave_node_ids = (
            mesh_data["top_node_ids"]
            if actual_slave_surface == "BlockTop"
            else mesh_data["indenter_surface_node_ids"]
        )
        result["mesh"] = {
            "divisions": divisions,
            "block_nodes": len(mesh_data["block_node_ids"]),
            "block_elements": len(mesh_data["block_element_ids"]),
            "indenter_surface_segments": mesh_data["number_of_arc_segments"],
            "carrier_elements": len(mesh_data["indenter_carrier_element_ids"]),
        }

        solver = analysis._GetSolver()
        solve_time = 0.0
        for step in range(1, number_of_steps + 1):
            prescribed_motion = (
                phase3.FINAL_PRESCRIBED_MOTION_MM * step / number_of_steps
            )
            analysis.time = solver.AdvanceInTime(analysis.time)
            phase3._set_indenter_motion(
                model_part, mesh_data["indenter_node_ids"], prescribed_motion
            )
            analysis.InitializeSolutionStep()
            solver.Predict()
            solve_start = time.perf_counter()
            if detailed_step is not None and step == detailed_step:
                solver_converged, captured_records = _solve_one_update_at_a_time(
                    solver,
                    model_part,
                    mesh_data["indenter_node_ids"],
                    phase3.MAXIMUM_NEWTON_ITERATIONS,
                )
                iteration_records.extend(captured_records)
            else:
                solver_converged = bool(solver.SolveSolutionStep())
            step_solve_time = time.perf_counter() - solve_start
            solve_time += step_solve_time
            analysis.FinalizeSolutionStep()

            field_failures = phase3.finite_field_failures(model_part)
            reaction_y = phase3._reaction_sum(
                model_part, mesh_data["indenter_node_ids"]
            )
            contact_fields = _contact_field_statistics(
                model_part, actual_slave_node_ids
            )
            deformation = phase3._measure_deformation(
                model_part, mesh_data["block_element_ids"]
            )
            volumetric_strain = phase3._volumetric_statistics(
                model_part, mesh_data["block_node_ids"]
            )
            volumetric_oscillation = phase3._checkerboard_metrics(
                model_part, mesh_data["block_node_ids"], divisions
            )
            contact_oscillation = phase3._contact_oscillation_metrics(
                model_part, actual_slave_node_ids
            )
            point = {
                "step": step,
                "prescribed_motion_mm": prescribed_motion,
                "indentation_after_gap_mm": max(
                    0.0, prescribed_motion - phase3.INITIAL_GAP_MM
                ),
                "solver_converged": solver_converged,
                "nonlinear_iterations": int(
                    model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]
                ),
                "active_set_converged": bool(
                    model_part.ProcessInfo[CSMA.ACTIVE_SET_CONVERGED]
                ),
                "reaction_y_n": reaction_y,
                "reaction_magnitude_n": abs(reaction_y),
                "finite_solution_fields": not field_failures,
                "contact_fields": contact_fields,
                "solve_wall_clock_seconds": step_solve_time,
                **deformation,
                "volumetric_strain": volumetric_strain,
                "volumetric_strain_oscillation": volumetric_oscillation,
                "contact_oscillation": contact_oscillation,
            }
            if field_failures:
                point["non_finite_fields"] = field_failures[:30]
            result["curve"].append(point)

            if not solver_converged:
                result["failure_reason"] = "nonlinear_solver_did_not_converge"
                break
            if field_failures or not contact_fields["finite"]:
                result["failure_reason"] = "non_finite_field"
                break
            if deformation["det_f"]["negative_count"] > 0:
                result["failure_reason"] = "non_positive_det_f"
                break

        result["iteration_records"] = iteration_records
        result["iteration_cycle_by_step"] = {
            str(step): _active_set_cycle_analysis(iteration_records, step)
            for step in sorted({record["step"] for record in iteration_records})
        }
        result["unresolved_periodic_cycle_steps"] = [
            int(step)
            for step, data in result["iteration_cycle_by_step"].items()
            if data["periodic_cycle_detected"]
        ]
        result["surface_nonzero_multiplier_node_ids"] = (
            _surface_nonzero_multiplier_nodes(model_part, mesh_data)
        )
        result["carrier_validation"] = _carrier_validation(
            model_part, mesh_data, divisions
        )
        result["solve_wall_clock_seconds"] = solve_time

        completed = len(result["curve"]) == number_of_steps and all(
            point["solver_converged"]
            and point["active_set_converged"]
            and point["finite_solution_fields"]
            and point["contact_fields"]["finite"]
            and point["det_f"]["negative_count"] == 0
            for point in result["curve"]
        )
        converged_points = [
            point for point in result["curve"] if point["solver_converged"]
        ]
        result["maximum_nonlinear_iterations"] = max(
            (point["nonlinear_iterations"] for point in result["curve"]),
            default=0,
        )
        result["minimum_det_f"] = min(
            (point["det_f"]["min"] for point in result["curve"]),
            default=None,
        )
        result["all_fields_finite"] = all(
            point["finite_solution_fields"] and point["contact_fields"]["finite"]
            for point in result["curve"]
        )
        result["all_det_f_positive"] = all(
            point["det_f"]["negative_count"] == 0 for point in result["curve"]
        )
        if converged_points:
            result["last_converged"] = converged_points[-1]

        load_bearing_step = next(
            (
                point["step"]
                for point in converged_points
                if point["reaction_magnitude_n"]
                > phase3.LOAD_BEARING_CONTACT_TOLERANCE
            ),
            None,
        )
        result["first_load_bearing_step"] = load_bearing_step
        if load_bearing_step is not None:
            result["curve_smoothness"] = phase3._curve_metrics(
                [
                    {
                        "step": point["step"],
                        "indenter_reaction_magnitude_n": point[
                            "reaction_magnitude_n"
                        ],
                    }
                    for point in converged_points
                ],
                load_bearing_step,
            )

        if completed:
            result["final"] = result["curve"][-1]
            final = result["final"]
            result["solve_status"] = "PASS"
            case_checks = {
                "target_indentation_reached": (
                    final["indentation_after_gap_mm"]
                    >= phase3.TARGET_INDENTATION_MM - 1.0e-12
                ),
                "all_fields_finite": result["all_fields_finite"],
                "all_det_f_positive": result["all_det_f_positive"],
                "force_curve_smooth_and_monotonic": result.get(
                    "curve_smoothness", {}
                ).get("smooth", False),
                "final_pressure_roughness_below_0_5": final[
                    "contact_oscillation"
                ].get("pressure_roughness_ratio", math.inf)
                < PRESSURE_ROUGHNESS_LIMIT,
                "final_pressure_sign_consistent": final[
                    "contact_oscillation"
                ].get("consistent_pressure_sign", True),
                "final_volumetric_checkerboard_absent": final[
                    "volumetric_strain_oscillation"
                ]["pass"],
                "active_set_converged_every_step": all(
                    point["active_set_converged"] for point in result["curve"]
                ),
                "no_unresolved_periodic_active_set_cycle": not result[
                    "unresolved_periodic_cycle_steps"
                ],
                "carrier_validation": result["carrier_validation"]["pass"],
            }
            result["case_acceptance_checks"] = case_checks
            result["contact_profile_auxiliary_diagnostic"] = {
                "phase3_pressure_and_gap_shape_check_pass": final[
                    "contact_oscillation"
                ]["pass"],
                "note": (
                    "The Phase 3R numerical acceptance threshold is the final "
                    "pressure roughness ratio; weighted-gap slope changes are "
                    "retained as an auxiliary diagnostic."
                ),
            }
            case_acceptance = all(case_checks.values())
            result["status"] = "PASS" if case_acceptance else "FAIL"
            if not case_acceptance:
                result["failure_reason"] = "case_acceptance_checks_failed"
                result["failed_case_acceptance_checks"] = [
                    name for name, passed in case_checks.items() if not passed
                ]
        else:
            result["solve_status"] = "FAIL"
    except Exception as exception:
        result["failure_reason"] = "exception"
        result["exception"] = f"{type(exception).__name__}: {exception}"
    finally:
        if initialized and analysis is not None:
            try:
                analysis.Finalize()
            except Exception as exception:
                result["finalize_exception"] = (
                    f"{type(exception).__name__}: {exception}"
                )
    result["case_wall_clock_seconds"] = time.perf_counter() - start
    return result


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / abs(second) if second != 0.0 else math.inf


def _run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1200,
    )


def _load_subprocess_result(
    output_path: Path,
    completed: subprocess.CompletedProcess[str],
    fallback_id: str,
) -> dict[str, Any]:
    if output_path.is_file():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "case_id": fallback_id,
            "status": "FAIL",
            "failure_reason": "subprocess_failed_without_output",
        }
    result["process_exit_code"] = completed.returncode
    if completed.stdout or completed.stderr:
        result["process_output_tail"] = (
            completed.stdout + completed.stderr
        )[-6000:]
    return result


def _phase3r_analysis(
    baseline: dict[str, Any],
    medium_48_recovery: dict[str, Any] | None,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_level = {case["mesh_level"]: case for case in cases}
    all_solves_pass = all(
        by_level[level].get("solve_status") == "PASS"
        for level in phase3.MESH_LEVELS
    )
    all_case_acceptance_pass = all(
        by_level[level].get("status") == "PASS" for level in phase3.MESH_LEVELS
    )
    medium_fine_difference = None
    medium_refinement_difference = None
    original_direction_medium_difference = None
    if all_solves_pass:
        medium_reaction = by_level["medium"]["final"]["reaction_magnitude_n"]
        fine_reaction = by_level["fine"]["final"]["reaction_magnitude_n"]
        medium_fine_difference = _relative_difference(
            medium_reaction, fine_reaction
        )
        baseline_medium_reaction = next(
            case for case in baseline["cases"] if case["mesh_level"] == "medium"
        )["final"]["indenter_reaction_magnitude_n"]
        original_direction_medium_difference = _relative_difference(
            medium_reaction, baseline_medium_reaction
        )
        if (
            medium_48_recovery is not None
            and medium_48_recovery.get("solve_status") == "PASS"
        ):
            medium_refinement_difference = _relative_difference(
                medium_reaction,
                medium_48_recovery["final"]["reaction_magnitude_n"],
            )
    acceptance = (
        all_case_acceptance_pass
        and medium_fine_difference is not None
        and medium_fine_difference < MEDIUM_FINE_REACTION_TOLERANCE
        and medium_refinement_difference is not None
        and medium_refinement_difference < MEDIUM_STEP_REFINEMENT_TOLERANCE
    )
    return {
        "all_96_step_solves_pass": all_solves_pass,
        "all_96_step_case_acceptance_pass": all_case_acceptance_pass,
        "final_pressure_roughness_by_mesh": {
            level: by_level[level].get("final", {})
            .get("contact_oscillation", {})
            .get("pressure_roughness_ratio")
            for level in phase3.MESH_LEVELS
        },
        "medium_fine_final_reaction_relative_difference": medium_fine_difference,
        "medium_fine_threshold": MEDIUM_FINE_REACTION_TOLERANCE,
        "medium_48_96_final_reaction_relative_difference": (
            medium_refinement_difference
        ),
        "medium_48_96_comparison_uses_same_master_slave_direction": True,
        "original_baseline_48_to_recovery_96_medium_reaction_relative_difference": (
            original_direction_medium_difference
        ),
        "original_baseline_comparison_note": (
            "The accepted 48-step baseline used the opposite runtime "
            "MASTER/SLAVE direction, so this value is not a pure step-size metric."
        ),
        "medium_step_refinement_threshold": MEDIUM_STEP_REFINEMENT_TOLERANCE,
        "phase3r_acceptance_pass": acceptance,
        "mixed_solid_formulation_status": "ADOPT",
        "kratos_2d_alm_contact_status": "ADOPT" if acceptance else "NO_ADOPTION",
        "next_blocker_if_failed": (
            None
            if acceptance
            else "evaluate penalty mortar or a different contact solver"
        ),
    }


def _run_all(baseline_path: Path, output_path: Path) -> int:
    baseline_path = baseline_path.resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline JSON does not exist: {baseline_path}")
    baseline_hash_before = _file_sha256(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("phase3_acceptance_pass") is not False:
        raise RuntimeError("Expected the accepted 48-step Phase 3 FAIL baseline")

    commands: list[list[str]] = []
    start = time.perf_counter()
    recovery_failure_diagnosis: dict[str, Any] | None = None
    medium_48_recovery: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="lit-phase3r-") as temporary:
        temporary_path = Path(temporary)
        diagnosis_path = temporary_path / "fine_diagnosis.json"
        diagnosis_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "diagnose-fine",
            "--output",
            str(diagnosis_path),
        ]
        commands.append(diagnosis_command)
        diagnosis_completed = _run_subprocess(diagnosis_command)
        diagnosis = _load_subprocess_result(
            diagnosis_path, diagnosis_completed, "fine_48_step_diagnosis"
        )

        actual_slave = diagnosis.get("runtime_contact_contract", {}).get(
            "actual_slave_surface"
        )
        if actual_slave == "IndenterSurface":
            recovery_slave = RECOVERY_REQUIRED_SLAVE_SURFACE
        elif actual_slave == "BlockTop":
            recovery_slave = "BlockTop"
        else:
            recovery_slave = None

        cases: list[dict[str, Any]] = []
        if recovery_slave is not None:
            medium_48_path = temporary_path / "medium_48_corrected_direction.json"
            medium_48_command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "case48-medium",
                "--slave-surface",
                recovery_slave,
                "--output",
                str(medium_48_path),
            ]
            commands.append(medium_48_command)
            medium_48_completed = _run_subprocess(medium_48_command)
            medium_48_recovery = _load_subprocess_result(
                medium_48_path,
                medium_48_completed,
                "phase3r_48_medium_corrected_direction",
            )

            for mesh_level in phase3.MESH_LEVELS:
                case_path = temporary_path / f"{mesh_level}_96.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "case96",
                    "--mesh-level",
                    mesh_level,
                    "--slave-surface",
                    recovery_slave,
                    "--output",
                    str(case_path),
                ]
                commands.append(command)
                completed = _run_subprocess(command)
                cases.append(
                    _load_subprocess_result(
                        case_path,
                        completed,
                        f"phase3r_96_{mesh_level}",
                    )
                )

            fine_case = next(
                (case for case in cases if case.get("mesh_level") == "fine"),
                None,
            )
            if (
                fine_case is not None
                and fine_case.get("failure_reason")
                == "nonlinear_solver_did_not_converge"
                and fine_case.get("curve")
            ):
                failure_step = int(fine_case["curve"][-1]["step"])
                recovery_diagnosis_path = (
                    temporary_path / "fine_96_failure_diagnosis.json"
                )
                recovery_diagnosis_command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "diagnose96",
                    "--mesh-level",
                    "fine",
                    "--slave-surface",
                    recovery_slave,
                    "--target-step",
                    str(failure_step),
                    "--output",
                    str(recovery_diagnosis_path),
                ]
                commands.append(recovery_diagnosis_command)
                recovery_diagnosis_completed = _run_subprocess(
                    recovery_diagnosis_command
                )
                recovery_failure_diagnosis = _load_subprocess_result(
                    recovery_diagnosis_path,
                    recovery_diagnosis_completed,
                    "fine_96_step_failure_diagnosis",
                )

    baseline_hash_after = _file_sha256(baseline_path)
    analysis = _phase3r_analysis(
        baseline, medium_48_recovery, cases
    ) if cases else {
        "phase3r_acceptance_pass": False,
        "mixed_solid_formulation_status": "ADOPT",
        "kratos_2d_alm_contact_status": "NO_ADOPTION",
        "next_blocker_if_failed": "runtime MASTER/SLAVE classification failed",
    }
    output = {
        "phase": "3R",
        "kratos_version": KM.Kernel.Version(),
        "python_executable": sys.executable,
        "baseline_preservation": {
            "path": str(baseline_path),
            "sha256_before": baseline_hash_before,
            "sha256_after": baseline_hash_after,
            "unchanged": baseline_hash_before == baseline_hash_after,
        },
        "configuration": {
            "baseline_steps": BASELINE_STEPS,
            "recovery_steps": RECOVERY_STEPS,
            "final_prescribed_motion_mm": phase3.FINAL_PRESCRIBED_MOTION_MM,
            "target_indentation_mm": phase3.TARGET_INDENTATION_MM,
            "element": phase3.ELEMENT_NAME,
            "constitutive_law": phase3.CONSTITUTIVE_LAW_NAME,
            "poisson_ratio": phase3.POISSON_RATIO,
            "contact": phase3.MORTAR_TYPE,
            "contact_parameters": (
                "Kratos 10.3 ALM penalty/search defaults; identical for all meshes"
            ),
            "maximum_newton_iterations": phase3.MAXIMUM_NEWTON_ITERATIONS,
            "convergence_tolerances": {
                "relative": phase3.RELATIVE_TOLERANCE,
                "absolute": phase3.ABSOLUTE_TOLERANCE,
                "contact_residual_relative": 1.0e-4,
                "contact_residual_absolute": 1.0e-9,
            },
            "runtime_selected_recovery_slave_surface": recovery_slave,
        },
        "fine_48_step_diagnosis": diagnosis,
        "medium_48_step_corrected_direction": medium_48_recovery,
        "recovery_cases_96_step": cases,
        "fine_96_step_failure_diagnosis": recovery_failure_diagnosis,
        "analysis": analysis,
        "commands": commands,
        "total_wall_clock_seconds": time.perf_counter() - start,
    }
    _write_json(output_path, output)
    return 0


def main() -> int:
    arguments = _parse_arguments()
    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    if arguments.command == "diagnose-fine":
        result = _run_model(
            "fine",
            BASELINE_STEPS,
            BASELINE_SLAVE_SURFACE,
            BASELINE_FINE_FAILURE_STEP,
        )
        result["cycle_analysis_step_23"] = _active_set_cycle_analysis(
            result.get("iteration_records", []), BASELINE_FINE_FAILURE_STEP
        )
        _write_json(arguments.output, result)
        return 0
    if arguments.command == "diagnose96":
        result = _run_model(
            arguments.mesh_level,
            RECOVERY_STEPS,
            arguments.slave_surface,
            arguments.target_step,
        )
        result["cycle_analysis_target_step"] = _active_set_cycle_analysis(
            result.get("iteration_records", []), arguments.target_step
        )
        _write_json(arguments.output, result)
        return 0
    if arguments.command == "case96":
        result = _run_model(
            arguments.mesh_level,
            RECOVERY_STEPS,
            arguments.slave_surface,
            None,
        )
        _write_json(arguments.output, result)
        return 0
    if arguments.command == "case48-medium":
        result = _run_model(
            "medium",
            BASELINE_STEPS,
            arguments.slave_surface,
            None,
        )
        _write_json(arguments.output, result)
        return 0
    return _run_all(arguments.baseline, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
