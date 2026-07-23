#!/usr/bin/env python3
"""Combine the validated ALM contact and mixed hyperelastic solid baselines.

This is a localized indentation benchmark, not the LIT fingertip geometry.  Each
mesh case is executed in a fresh subprocess and creates a fresh Kratos Model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import KratosMultiphysics as KM
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
import KratosMultiphysics.ConstitutiveLawsApplication as CLA
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)


ELEMENT_NAME = "TotalLagrangianMixedVolumetricStrainElement2D3N"
CONSTITUTIVE_LAW_NAME = "HyperElasticPlaneStrain2DLaw"
CONTACT_PROCESS_NAME = "ALMContactProcess"
MORTAR_TYPE = "ALMContactFrictionless"

MESH_LEVELS = {"coarse": 8, "medium": 16, "fine": 32}
WIDTH_MM = 10.0
HEIGHT_MM = 5.0
THICKNESS_MM = 1.0
YOUNG_MODULUS_MPA = 1.0
POISSON_RATIO = 0.49
INDENTER_RADIUS_MM = 2.0
INDENTER_HALF_SPAN_MM = 2.0
INDENTER_CARRIER_THICKNESS_MM = 0.5
INITIAL_GAP_MM = 0.12
FINAL_PRESCRIBED_MOTION_MM = 0.62
TARGET_INDENTATION_MM = FINAL_PRESCRIBED_MOTION_MM - INITIAL_GAP_MM
NUMBER_OF_STEPS = 48

RELATIVE_TOLERANCE = 1.0e-6
ABSOLUTE_TOLERANCE = 1.0e-9
MAXIMUM_NEWTON_ITERATIONS = 35
MESH_REACTION_RELATIVE_TOLERANCE = 0.10
SMOOTHNESS_SECOND_DIFFERENCE_LIMIT = 0.15
CHECKERBOARD_MODE_LIMIT = 0.35
CHECKERBOARD_RESIDUAL_LIMIT = 1.0
PRESSURE_ROUGHNESS_LIMIT = 0.50
AREA_RATIO_ABSOLUTE_TOLERANCE = 1.0e-10
LOAD_BEARING_CONTACT_TOLERANCE = 1.0e-10

OFFICIAL_CONTACT_SOURCE = (
    "https://github.com/KratosMultiphysics/Kratos/tree/v10.3.0/"
    "applications/ContactStructuralMechanicsApplication/tests/"
    "ALM_frictionless_contact_test_2D"
)
OFFICIAL_MIXED_SOURCE = (
    "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
    "applications/StructuralMechanicsApplication/custom_elements/solid_elements/"
    "total_lagrangian_mixed_volumetric_strain_element.h"
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", required=True, type=Path)
    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("--mesh-level", choices=MESH_LEVELS, required=True)
    case_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _project_parameters() -> KM.Parameters:
    return KM.Parameters(
        f"""{{
            "problem_data": {{
                "problem_name": "phase3_localized_indentation",
                "parallel_type": "OpenMP",
                "start_time": 0.0,
                "end_time": {NUMBER_OF_STEPS}.0,
                "echo_level": 0
            }},
            "solver_settings": {{
                "model_part_name": "Structure",
                "domain_size": 2,
                "solver_type": "Static",
                "echo_level": 0,
                "analysis_type": "non_linear",
                "model_import_settings": {{
                    "input_type": "use_input_model_part"
                }},
                "material_import_settings": {{
                    "materials_filename": ""
                }},
                "time_stepping": {{
                    "time_step": 1.0
                }},
                "volumetric_strain_dofs": true,
                "contact_settings": {{
                    "mortar_type": "{MORTAR_TYPE}",
                    "ensure_contact": false,
                    "silent_strategy": true,
                    "simplified_semi_smooth_newton": false,
                    "fancy_convergence_criterion": false,
                    "print_convergence_criterion": false
                }},
                "clear_storage": true,
                "reform_dofs_at_each_step": true,
                "compute_reactions": true,
                "move_mesh_flag": true,
                "convergence_criterion": "contact_residual_criterion",
                "displacement_relative_tolerance": {RELATIVE_TOLERANCE},
                "displacement_absolute_tolerance": {ABSOLUTE_TOLERANCE},
                "residual_relative_tolerance": {RELATIVE_TOLERANCE},
                "residual_absolute_tolerance": {ABSOLUTE_TOLERANCE},
                "max_iteration": {MAXIMUM_NEWTON_ITERATIONS},
                "builder_and_solver_settings": {{
                    "type": "block",
                    "advanced_settings": {{}}
                }},
                "solving_strategy_settings": {{
                    "type": "newton_raphson",
                    "advanced_settings": {{}}
                }},
                "linear_solver_settings": {{
                    "solver_type": "skyline_lu_factorization"
                }}
            }},
            "processes": {{
                "contact_process_list": [{{
                    "python_module": "alm_contact_process",
                    "kratos_module": "KratosMultiphysics.ContactStructuralMechanicsApplication",
                    "process_name": "ALMContactProcess",
                    "Parameters": {{
                        "model_part_name": "Structure",
                        "assume_master_slave": {{"0": ["IndenterSurface"]}},
                        "contact_model_part": {{
                            "0": ["BlockTop", "IndenterSurface"]
                        }},
                        "contact_type": "Frictionless"
                    }}
                }}]
            }}
        }}"""
    )


def _create_block_mesh(model_part: KM.ModelPart, divisions: int) -> None:
    geometry = KM.Quadrilateral2D4(
        KM.Node(1, 0.0, 0.0, 0.0),
        KM.Node(2, 0.0, HEIGHT_MM, 0.0),
        KM.Node(3, WIDTH_MM, HEIGHT_MM, 0.0),
        KM.Node(4, WIDTH_MM, 0.0, 0.0),
    )
    settings = KM.Parameters(
        """{
            "number_of_divisions": 1,
            "create_skin_sub_model_part": false,
            "element_name": ""
        }"""
    )
    settings["number_of_divisions"].SetInt(divisions)
    settings["element_name"].SetString(ELEMENT_NAME)
    KM.StructuredMeshGeneratorProcess(geometry, model_part, settings).Execute()


def _configure_material(model_part: KM.ModelPart) -> None:
    properties = model_part.Properties[0]
    properties[KM.YOUNG_MODULUS] = YOUNG_MODULUS_MPA
    properties[KM.POISSON_RATIO] = POISSON_RATIO
    properties[KM.THICKNESS] = THICKNESS_MM
    properties[KM.DENSITY] = 1.0
    properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElasticPlaneStrain2DLaw()


def _create_submodel_parts_and_indenter(
    model_part: KM.ModelPart, divisions: int
) -> dict[str, Any]:
    tolerance = 1.0e-12
    block_node_ids = [node.Id for node in model_part.Nodes]
    block_element_ids = [element.Id for element in model_part.Elements]
    bottom_node_ids = [
        node.Id for node in model_part.Nodes if abs(node.Y0) <= tolerance
    ]
    top_node_ids = [
        node.Id
        for node in model_part.Nodes
        if abs(node.Y0 - HEIGHT_MM) <= tolerance
    ]

    solid_domain = model_part.CreateSubModelPart("SolidDomain")
    solid_domain.AddNodes(block_node_ids)
    solid_domain.AddElements(block_element_ids)
    bottom = model_part.CreateSubModelPart("Bottom")
    bottom.AddNodes(bottom_node_ids)
    block_top = model_part.CreateSubModelPart("BlockTop")
    block_top.AddNodes(top_node_ids)

    indenter = model_part.CreateSubModelPart("IndenterSurface")
    carrier = model_part.CreateSubModelPart("IndenterCarrier")
    motion = model_part.CreateSubModelPart("IndenterMotion")
    number_of_arc_segments = divisions
    first_node_id = max(block_node_ids) + 1
    indenter_surface_node_ids: list[int] = []
    indenter_carrier_node_ids: list[int] = []
    center_x = 0.5 * WIDTH_MM
    center_y = HEIGHT_MM + INITIAL_GAP_MM + INDENTER_RADIUS_MM
    for index in range(number_of_arc_segments + 1):
        local_x = -INDENTER_HALF_SPAN_MM + (
            2.0 * INDENTER_HALF_SPAN_MM * index / number_of_arc_segments
        )
        under_root = max(0.0, INDENTER_RADIUS_MM**2 - local_x**2)
        x = center_x + local_x
        y = center_y - math.sqrt(under_root)
        node_id = first_node_id + index
        model_part.CreateNewNode(node_id, x, y, 0.0)
        indenter_surface_node_ids.append(node_id)
        carrier_node_id = first_node_id + number_of_arc_segments + 1 + index
        model_part.CreateNewNode(
            carrier_node_id,
            x,
            y + INDENTER_CARRIER_THICKNESS_MM,
            0.0,
        )
        indenter_carrier_node_ids.append(carrier_node_id)

    first_carrier_element_id = max(block_element_ids) + 1
    indenter_carrier_element_ids: list[int] = []
    for index in range(number_of_arc_segments):
        first_element_id = first_carrier_element_id + 2 * index
        surface_left = indenter_surface_node_ids[index]
        surface_right = indenter_surface_node_ids[index + 1]
        carrier_left = indenter_carrier_node_ids[index]
        carrier_right = indenter_carrier_node_ids[index + 1]
        # The two T3s are only a kinematic carrier for the prescribed rigid
        # motion.  Attaching elements to the contact surface lets Kratos
        # compute a finite master NODAL_H for mortar search/ALM scaling.
        model_part.CreateNewElement(
            "TotalLagrangianElement2D3N",
            first_element_id,
            [surface_left, surface_right, carrier_right],
            model_part.Properties[0],
        )
        model_part.CreateNewElement(
            "TotalLagrangianElement2D3N",
            first_element_id + 1,
            [surface_left, carrier_right, carrier_left],
            model_part.Properties[0],
        )
        indenter_carrier_element_ids.extend(
            [first_element_id, first_element_id + 1]
        )

    first_condition_id = 1
    indenter_condition_ids: list[int] = []
    for index in range(number_of_arc_segments):
        condition_id = first_condition_id + index
        # Left-to-right ordering gives a downward normal for a 2D line.
        model_part.CreateNewCondition(
            "LineCondition2D2N",
            condition_id,
            [
                indenter_surface_node_ids[index],
                indenter_surface_node_ids[index + 1],
            ],
            model_part.Properties[0],
        )
        indenter_condition_ids.append(condition_id)
    all_indenter_node_ids = (
        indenter_surface_node_ids + indenter_carrier_node_ids
    )
    indenter.AddNodes(indenter_surface_node_ids)
    indenter.AddConditions(indenter_condition_ids)
    carrier.AddNodes(all_indenter_node_ids)
    carrier.AddElements(indenter_carrier_element_ids)
    motion.AddNodes(all_indenter_node_ids)

    for node in model_part.Nodes:
        node.SetSolutionStepValue(KM.VOLUMETRIC_STRAIN, 0.0)

    return {
        "block_node_ids": block_node_ids,
        "block_element_ids": block_element_ids,
        "bottom_node_ids": bottom_node_ids,
        "top_node_ids": top_node_ids,
        "indenter_node_ids": all_indenter_node_ids,
        "indenter_surface_node_ids": indenter_surface_node_ids,
        "indenter_carrier_node_ids": indenter_carrier_node_ids,
        "indenter_carrier_element_ids": indenter_carrier_element_ids,
        "indenter_condition_ids": indenter_condition_ids,
        "number_of_arc_segments": number_of_arc_segments,
    }


def _fix_kinematic_dofs(
    model_part: KM.ModelPart, mesh_data: dict[str, Any]
) -> None:
    for node_id in mesh_data["bottom_node_ids"]:
        node = model_part.Nodes[node_id]
        node.Fix(KM.DISPLACEMENT_X)
        node.Fix(KM.DISPLACEMENT_Y)
        node.Fix(KM.DISPLACEMENT_Z)
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)
    for node_id in mesh_data["indenter_node_ids"]:
        node = model_part.Nodes[node_id]
        node.Fix(KM.DISPLACEMENT_X)
        node.Fix(KM.DISPLACEMENT_Y)
        node.Fix(KM.DISPLACEMENT_Z)
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)


def _set_indenter_motion(
    model_part: KM.ModelPart,
    indenter_node_ids: list[int],
    prescribed_motion_mm: float,
) -> None:
    displacement_y = -prescribed_motion_mm
    for node_id in indenter_node_ids:
        node = model_part.Nodes[node_id]
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displacement_y)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)
        # Contact search is executed before the solver predictor.  Keep the
        # prescribed master geometry at its current target position.
        node.X = node.X0
        node.Y = node.Y0 + displacement_y
        node.Z = node.Z0


def _triangle_determinant(points: list[tuple[float, float]]) -> float:
    first, second, third = points
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        third[0] - first[0]
    ) * (second[1] - first[1])


def _measure_deformation(
    model_part: KM.ModelPart, element_ids: list[int]
) -> dict[str, Any]:
    determinant_values: list[float] = []
    initial_area = 0.0
    deformed_area = 0.0
    weighted_sum = 0.0
    for element_id in element_ids:
        geometry = model_part.Elements[element_id].GetGeometry()
        initial_points = [(node.X0, node.Y0) for node in geometry]
        deformed_points = [(node.X, node.Y) for node in geometry]
        initial_determinant = _triangle_determinant(initial_points)
        deformed_determinant = _triangle_determinant(deformed_points)
        if abs(initial_determinant) <= 1.0e-15:
            raise RuntimeError(f"Element {element_id} has zero reference area")
        det_f = deformed_determinant / initial_determinant
        reference_area = 0.5 * abs(initial_determinant)
        current_area = 0.5 * abs(deformed_determinant)
        determinant_values.append(det_f)
        initial_area += reference_area
        deformed_area += current_area
        weighted_sum += reference_area * det_f
    if not determinant_values:
        raise RuntimeError("No solid deformation values were produced")
    weighted_mean = weighted_sum / initial_area
    area_ratio = deformed_area / initial_area
    return {
        "det_f": {
            "source": "affine T3 nodal kinematics",
            "min": min(determinant_values),
            "max": max(determinant_values),
            "area_weighted_mean": weighted_mean,
            "negative_count": sum(value <= 0.0 for value in determinant_values),
        },
        "deformed_area_ratio": area_ratio,
        "det_f_area_ratio_abs_difference": abs(weighted_mean - area_ratio),
    }


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _nodal_h_statistics(
    model_part: KM.ModelPart, node_ids: list[int]
) -> dict[str, Any]:
    values = [
        float(model_part.Nodes[node_id].GetSolutionStepValue(KM.NODAL_H))
        for node_id in node_ids
    ]
    finite = all(math.isfinite(value) for value in values)
    return {
        "min": min(values),
        "max": max(values),
        "finite": finite,
        "positive": finite and all(value > 0.0 for value in values),
    }


def _finite_field_failures(model_part: KM.ModelPart) -> list[str]:
    failures: list[str] = []
    for node in model_part.Nodes:
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        reaction = node.GetSolutionStepValue(KM.REACTION)
        if not all(math.isfinite(displacement[index]) for index in range(2)):
            failures.append(f"node_{node.Id}_DISPLACEMENT")
        if not all(math.isfinite(reaction[index]) for index in range(2)):
            failures.append(f"node_{node.Id}_REACTION")
        volumetric_strain = float(
            node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN)
        )
        if not math.isfinite(volumetric_strain):
            failures.append(f"node_{node.Id}_VOLUMETRIC_STRAIN")
    return failures


def _condition_snapshot(model_part: KM.ModelPart | None) -> dict[str, Any]:
    if model_part is None:
        return {
            "count": 0,
            "active_count": 0,
            "slave_count": 0,
            "master_count": 0,
            "names": [],
        }
    conditions = list(model_part.Conditions)
    return {
        "count": len(conditions),
        "active_count": sum(condition.Is(KM.ACTIVE) for condition in conditions),
        "slave_count": sum(condition.Is(KM.SLAVE) for condition in conditions),
        "master_count": sum(condition.Is(KM.MASTER) for condition in conditions),
        "names": sorted({condition.Info() for condition in conditions}),
    }


def _contact_model_part(model: KM.Model, name: str) -> KM.ModelPart | None:
    return model[name] if model.HasModelPart(name) else None


def _nodal_scalar_values(
    model_part: KM.ModelPart, node_ids: list[int], variable: Any, historical: bool
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        if historical:
            if node.SolutionStepsDataHas(variable):
                values[str(node_id)] = _finite_float(
                    node.GetSolutionStepValue(variable)
                )
        elif node.Has(variable):
            values[str(node_id)] = _finite_float(node.GetValue(variable))
    return values


def _contact_fields(
    model_part: KM.ModelPart, contact_node_ids: list[int]
) -> dict[str, Any]:
    weighted_gap = _nodal_scalar_values(
        model_part, contact_node_ids, CSMA.WEIGHTED_GAP, historical=True
    )
    multiplier_pressure = _nodal_scalar_values(
        model_part,
        contact_node_ids,
        CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
        historical=True,
    )
    augmented_pressure = _nodal_scalar_values(
        model_part,
        contact_node_ids,
        CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE,
        historical=False,
    )
    normal_gap_variable = getattr(CSMA, "NORMAL_GAP", None)
    normal_gap = (
        _nodal_scalar_values(
            model_part, contact_node_ids, normal_gap_variable, historical=False
        )
        if normal_gap_variable is not None
        else {}
    )
    all_values = [
        value
        for field in (
            weighted_gap,
            multiplier_pressure,
            augmented_pressure,
            normal_gap,
        )
        for value in field.values()
    ]
    return {
        "weighted_gap": weighted_gap,
        "lagrange_multiplier_contact_pressure": multiplier_pressure,
        "augmented_normal_contact_pressure": augmented_pressure,
        "normal_gap": normal_gap,
        "finite": all(value is not None for value in all_values),
    }


def _reaction_sum(model_part: KM.ModelPart, node_ids: list[int]) -> float:
    return sum(
        float(model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION_Y))
        for node_id in node_ids
    )


def _volumetric_statistics(
    model_part: KM.ModelPart, node_ids: list[int]
) -> dict[str, float]:
    values = [
        float(model_part.Nodes[node_id].GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
        for node_id in node_ids
    ]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "max_abs": max(abs(value) for value in values),
    }


def _checkerboard_metrics(
    model_part: KM.ModelPart, block_node_ids: list[int], divisions: int
) -> dict[str, Any]:
    grid: dict[tuple[int, int], float] = {}
    for node_id in block_node_ids:
        node = model_part.Nodes[node_id]
        i = int(round(node.X0 / WIDTH_MM * divisions))
        j = int(round(node.Y0 / HEIGHT_MM * divisions))
        grid[(i, j)] = float(node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
    values = list(grid.values())
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    l1 = sum(abs(value) for value in centered)
    checkerboard_projection = abs(
        sum(((-1.0) ** (i + j)) * (value - mean) for (i, j), value in grid.items())
    )
    checkerboard_ratio = checkerboard_projection / max(l1, 1.0e-15)

    residuals: list[float] = []
    for (i, j), value in grid.items():
        neighbours = [
            grid[key]
            for key in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
            if key in grid
        ]
        if neighbours:
            residuals.append(value - sum(neighbours) / len(neighbours))
    centered_rms = math.sqrt(
        sum(value * value for value in centered) / max(len(centered), 1)
    )
    residual_rms = math.sqrt(
        sum(value * value for value in residuals) / max(len(residuals), 1)
    )
    residual_ratio = residual_rms / max(centered_rms, 1.0e-15)
    return {
        "checkerboard_mode_ratio": checkerboard_ratio,
        "neighbor_residual_rms_ratio": residual_ratio,
        "checkerboard_limit": CHECKERBOARD_MODE_LIMIT,
        "residual_limit": CHECKERBOARD_RESIDUAL_LIMIT,
        "pass": checkerboard_ratio <= CHECKERBOARD_MODE_LIMIT
        and residual_ratio <= CHECKERBOARD_RESIDUAL_LIMIT,
    }


def _sign_changes(values: list[float], tolerance: float) -> int:
    signs: list[int] = []
    for value in values:
        if value > tolerance:
            signs.append(1)
        elif value < -tolerance:
            signs.append(-1)
    return sum(first != second for first, second in zip(signs, signs[1:]))


def _contact_oscillation_metrics(
    model_part: KM.ModelPart, top_node_ids: list[int]
) -> dict[str, Any]:
    records: list[tuple[float, float, float, bool]] = []
    for node_id in top_node_ids:
        node = model_part.Nodes[node_id]
        pressure = float(
            node.GetSolutionStepValue(CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE)
        )
        gap = float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP))
        records.append((node.X0, pressure, gap, node.Is(KM.ACTIVE)))
    records.sort(key=lambda item: item[0])
    active_records = [record for record in records if record[3]]
    if len(active_records) < 3:
        return {
            "active_node_count": len(active_records),
            "pressure_slope_sign_changes": 0,
            "pressure_roughness_ratio": 0.0,
            "gap_slope_sign_changes": 0,
            "pass": bool(active_records),
        }
    pressures = [abs(record[1]) for record in active_records]
    gaps = [record[2] for record in active_records]
    pressure_scale = max(max(pressures), 1.0e-15)
    pressure_differences = [
        second - first for first, second in zip(pressures, pressures[1:])
    ]
    pressure_sign_changes = _sign_changes(
        pressure_differences, 1.0e-8 * pressure_scale
    )
    pressure_residuals = [
        pressures[index]
        - 0.5 * (pressures[index - 1] + pressures[index + 1])
        for index in range(1, len(pressures) - 1)
    ]
    pressure_roughness = math.sqrt(
        sum(value * value for value in pressure_residuals)
        / max(len(pressure_residuals), 1)
    ) / pressure_scale
    gap_scale = max(max(abs(value) for value in gaps), 1.0e-15)
    gap_differences = [second - first for first, second in zip(gaps, gaps[1:])]
    gap_sign_changes = _sign_changes(gap_differences, 1.0e-8 * gap_scale)
    consistent_pressure_sign = not (
        any(record[1] > 1.0e-10 * pressure_scale for record in active_records)
        and any(record[1] < -1.0e-10 * pressure_scale for record in active_records)
    )
    return {
        "active_node_count": len(active_records),
        "pressure_slope_sign_changes": pressure_sign_changes,
        "pressure_roughness_ratio": pressure_roughness,
        "pressure_roughness_limit": PRESSURE_ROUGHNESS_LIMIT,
        "gap_slope_sign_changes": gap_sign_changes,
        "consistent_pressure_sign": consistent_pressure_sign,
        "pass": pressure_sign_changes <= 2
        and pressure_roughness <= PRESSURE_ROUGHNESS_LIMIT
        and gap_sign_changes <= 2
        and consistent_pressure_sign,
    }


def _curve_metrics(curve: list[dict[str, Any]], first_active_step: int) -> dict[str, Any]:
    active_curve = [
        point for point in curve if point["step"] >= first_active_step
    ]
    reactions = [point["indenter_reaction_magnitude_n"] for point in active_curve]
    if len(reactions) < 3 or reactions[-1] <= 0.0:
        return {
            "monotonic_non_decreasing": False,
            "normalized_max_second_difference": None,
            "smooth": False,
        }
    tolerance = 1.0e-8 * max(reactions[-1], 1.0)
    monotonic = all(
        second + tolerance >= first
        for first, second in zip(reactions, reactions[1:])
    )
    second_differences = [
        abs(reactions[index + 1] - 2.0 * reactions[index] + reactions[index - 1])
        for index in range(1, len(reactions) - 1)
    ]
    normalized = max(second_differences) / reactions[-1]
    return {
        "monotonic_non_decreasing": monotonic,
        "normalized_max_second_difference": normalized,
        "smoothness_limit": SMOOTHNESS_SECOND_DIFFERENCE_LIMIT,
        "smooth": monotonic and normalized <= SMOOTHNESS_SECOND_DIFFERENCE_LIMIT,
    }


def _maximum_absolute_contact_pressure(point: dict[str, Any]) -> float:
    pressure_values = point["contact_fields"][
        "lagrange_multiplier_contact_pressure"
    ].values()
    finite_values = [abs(value) for value in pressure_values if value is not None]
    return max(finite_values, default=0.0)


def _run_case(mesh_level: str) -> dict[str, Any]:
    divisions = MESH_LEVELS[mesh_level]
    result: dict[str, Any] = {
        "case_id": f"phase3_localized_indentation__{mesh_level}",
        "mesh_level": mesh_level,
        "status": "FAIL",
        "curve": [],
    }
    start = time.perf_counter()
    analysis: StructuralMechanicsAnalysis | None = None
    initialized = False
    try:
        model = KM.Model()
        parameters = _project_parameters()
        analysis = StructuralMechanicsAnalysis(model, parameters)
        model_part = model["Structure"]
        _create_block_mesh(model_part, divisions)
        _configure_material(model_part)
        mesh_data = _create_submodel_parts_and_indenter(model_part, divisions)
        result["mesh"] = {
            "parametric_divisions_x": divisions,
            "parametric_divisions_y": divisions,
            "number_of_solid_nodes": len(mesh_data["block_node_ids"]),
            "number_of_solid_elements": len(mesh_data["block_element_ids"]),
            "number_of_indenter_segments": mesh_data["number_of_arc_segments"],
            "number_of_kinematic_carrier_elements": len(
                mesh_data["indenter_carrier_element_ids"]
            ),
            "surface_spacing_mm": WIDTH_MM / divisions,
            "vertical_spacing_mm": HEIGHT_MM / divisions,
        }
        analysis.Initialize()
        initialized = True
        _fix_kinematic_dofs(model_part, mesh_data)

        contact_part = _contact_model_part(model, "Structure.Contact")
        computing_contact = _contact_model_part(model, "Structure.ComputingContact")
        contact_node_ids = (
            [node.Id for node in contact_part.Nodes]
            if contact_part is not None
            else sorted(
                set(
                    mesh_data["top_node_ids"]
                    + mesh_data["indenter_surface_node_ids"]
                )
            )
        )
        result["initial_contact_interface"] = _condition_snapshot(contact_part)
        result["initial_computing_contact"] = _condition_snapshot(computing_contact)
        result["initial_nodal_h"] = {
            "block_top": _nodal_h_statistics(
                model_part, mesh_data["top_node_ids"]
            ),
            "indenter_surface": _nodal_h_statistics(
                model_part, mesh_data["indenter_surface_node_ids"]
            ),
        }
        if not all(
            statistics["positive"]
            for statistics in result["initial_nodal_h"].values()
        ):
            raise RuntimeError(
                "Contact surface NODAL_H must be finite and positive"
            )
        result["initial_normals"] = {
            "block_top_mean_y": sum(
                model_part.Nodes[node_id].GetSolutionStepValue(KM.NORMAL)[1]
                for node_id in mesh_data["top_node_ids"]
            )
            / len(mesh_data["top_node_ids"]),
            "indenter_mean_y": sum(
                model_part.Nodes[node_id].GetSolutionStepValue(KM.NORMAL)[1]
                for node_id in mesh_data["indenter_surface_node_ids"]
            )
            / len(mesh_data["indenter_surface_node_ids"]),
        }

        solver = analysis._GetSolver()
        solve_time = 0.0
        for step in range(1, NUMBER_OF_STEPS + 1):
            prescribed_motion = FINAL_PRESCRIBED_MOTION_MM * step / NUMBER_OF_STEPS
            analysis.time = solver.AdvanceInTime(analysis.time)
            _set_indenter_motion(
                model_part, mesh_data["indenter_node_ids"], prescribed_motion
            )
            analysis.InitializeSolutionStep()
            solver.Predict()
            solve_start = time.perf_counter()
            solver_converged = bool(solver.SolveSolutionStep())
            step_solve_time = time.perf_counter() - solve_start
            solve_time += step_solve_time
            analysis.FinalizeSolutionStep()

            field_failures = _finite_field_failures(model_part)
            computing_contact = _contact_model_part(
                model, "Structure.ComputingContact"
            )
            contact_snapshot = _condition_snapshot(computing_contact)
            contact_fields = _contact_fields(model_part, contact_node_ids)
            reaction_y = _reaction_sum(
                model_part, mesh_data["indenter_node_ids"]
            )
            point: dict[str, Any] = {
                "step": step,
                "time": float(analysis.time),
                "prescribed_motion_mm": prescribed_motion,
                "indentation_after_gap_mm": max(
                    0.0, prescribed_motion - INITIAL_GAP_MM
                ),
                "solver_converged": solver_converged,
                "nonlinear_iterations": int(
                    model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]
                ),
                "active_set_converged": bool(
                    model_part.ProcessInfo[CSMA.ACTIVE_SET_CONVERGED]
                ),
                "solve_wall_clock_seconds": step_solve_time,
                "finite_solution_fields": not field_failures,
                "finite_contact_fields": contact_fields["finite"],
                "contact_conditions": contact_snapshot,
                "contact_fields": contact_fields,
                "indenter_reaction_y_n": reaction_y,
                "indenter_reaction_magnitude_n": abs(reaction_y),
            }
            if field_failures:
                point["non_finite_fields"] = field_failures[:30]
            if (
                not field_failures
                and contact_fields["finite"]
                and math.isfinite(reaction_y)
            ):
                point.update(
                    _measure_deformation(model_part, mesh_data["block_element_ids"])
                )
                point["volumetric_strain"] = _volumetric_statistics(
                    model_part, mesh_data["block_node_ids"]
                )
                point["volumetric_strain_oscillation"] = _checkerboard_metrics(
                    model_part, mesh_data["block_node_ids"], divisions
                )
                point["contact_oscillation"] = _contact_oscillation_metrics(
                    model_part, mesh_data["indenter_surface_node_ids"]
                )
            result["curve"].append(point)

            if not solver_converged:
                result["failure_reason"] = "nonlinear_solver_did_not_converge"
                break
            if field_failures:
                result["failure_reason"] = "non_finite_solution_fields"
                break
            if not contact_fields["finite"]:
                result["failure_reason"] = "non_finite_contact_fields"
                break
            if not math.isfinite(reaction_y):
                result["failure_reason"] = "non_finite_indenter_reaction"
                break
            if point["det_f"]["negative_count"] > 0:
                result["failure_reason"] = "non_positive_det_f"
                break

        first_candidate_active_step = next(
            (
                point["step"]
                for point in result["curve"]
                if point["contact_conditions"]["active_count"] > 0
            ),
            None,
        )
        first_load_bearing_step = next(
            (
                point["step"]
                for point in result["curve"]
                if point["indenter_reaction_magnitude_n"]
                > LOAD_BEARING_CONTACT_TOLERANCE
                and _maximum_absolute_contact_pressure(point)
                > LOAD_BEARING_CONTACT_TOLERANCE
            ),
            None,
        )
        result["first_candidate_active_step"] = first_candidate_active_step
        result["first_load_bearing_contact_step"] = first_load_bearing_step
        result["first_load_bearing_prescribed_motion_mm"] = (
            result["curve"][first_load_bearing_step - 1]["prescribed_motion_mm"]
            if first_load_bearing_step is not None
            else None
        )
        previous_motion = (
            result["curve"][first_load_bearing_step - 2]["prescribed_motion_mm"]
            if first_load_bearing_step is not None and first_load_bearing_step > 1
            else 0.0
        )
        result["contact_activation_brackets_initial_gap"] = (
            first_load_bearing_step is not None
            and previous_motion < INITIAL_GAP_MM
            <= result["first_load_bearing_prescribed_motion_mm"]
        )
        converged_points = [
            point for point in result["curve"] if point["solver_converged"]
        ]
        measured_points = [
            point for point in result["curve"] if "det_f" in point
        ]
        result["maximum_nonlinear_iterations"] = max(
            (point["nonlinear_iterations"] for point in result["curve"]),
            default=0,
        )
        result["all_recorded_solution_and_contact_fields_finite"] = all(
            point["finite_solution_fields"] and point["finite_contact_fields"]
            for point in result["curve"]
        )
        result["all_measured_states_have_positive_det_f"] = bool(
            measured_points
        ) and all(
            point["det_f"]["negative_count"] == 0 for point in measured_points
        )
        result["minimum_measured_det_f"] = min(
            (point["det_f"]["min"] for point in measured_points),
            default=None,
        )
        result["candidate_condition_count_monotonic"] = all(
            second["contact_conditions"]["active_count"]
            >= first["contact_conditions"]["active_count"]
            for first, second in zip(result["curve"], result["curve"][1:])
        )
        if first_load_bearing_step is not None and converged_points:
            result["converged_curve_smoothness"] = _curve_metrics(
                converged_points, first_load_bearing_step
            )
        if converged_points:
            last_converged = converged_points[-1]
            result["last_converged_state"] = {
                key: last_converged[key]
                for key in (
                    "step",
                    "prescribed_motion_mm",
                    "indentation_after_gap_mm",
                    "nonlinear_iterations",
                    "active_set_converged",
                    "indenter_reaction_magnitude_n",
                    "det_f",
                    "deformed_area_ratio",
                    "volumetric_strain",
                    "volumetric_strain_oscillation",
                    "contact_oscillation",
                )
            }

        completed = len(result["curve"]) == NUMBER_OF_STEPS and all(
            point["solver_converged"]
            and point["finite_solution_fields"]
            and point["finite_contact_fields"]
            and point["active_set_converged"]
            and point["det_f"]["negative_count"] == 0
            for point in result["curve"]
        )
        if completed:
            result["final"] = result["curve"][-1]
            result["active_condition_count_monotonic"] = all(
                second["contact_conditions"]["active_count"]
                >= first["contact_conditions"]["active_count"]
                for first, second in zip(result["curve"], result["curve"][1:])
            )
            result["maximum_det_f_area_ratio_abs_difference"] = max(
                point["det_f_area_ratio_abs_difference"]
                for point in result["curve"]
            )
            result["area_consistency_pass"] = (
                result["maximum_det_f_area_ratio_abs_difference"]
                <= AREA_RATIO_ABSOLUTE_TOLERANCE
            )
            if first_load_bearing_step is not None:
                result["curve_smoothness"] = _curve_metrics(
                    result["curve"], first_load_bearing_step
                )
            else:
                result["curve_smoothness"] = {
                    "smooth": False,
                    "reason": "Contact never activated",
                }
            result["volumetric_strain_oscillation"] = result["final"][
                "volumetric_strain_oscillation"
            ]
            result["contact_oscillation"] = result["final"][
                "contact_oscillation"
            ]
            acceptance = (
                first_load_bearing_step is not None
                and result["contact_activation_brackets_initial_gap"]
                and result["curve_smoothness"]["smooth"]
                and result["active_condition_count_monotonic"]
                and result["area_consistency_pass"]
                and result["volumetric_strain_oscillation"]["pass"]
                and result["contact_oscillation"]["pass"]
            )
            result["status"] = "PASS" if acceptance else "FAIL"
            if not acceptance:
                result["failure_reason"] = "case_acceptance_checks_failed"
        result["solve_wall_clock_seconds"] = solve_time
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


def _relative_difference(first: float, reference: float) -> float:
    if reference == 0.0:
        return math.inf
    return abs(first - reference) / abs(reference)


def _analyze_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_level = {case["mesh_level"]: case for case in cases}
    all_pass = all(by_level[level]["status"] == "PASS" for level in MESH_LEVELS)
    converged_by_level = {
        level: [
            point
            for point in by_level[level].get("curve", [])
            if point.get("solver_converged", False)
        ]
        for level in MESH_LEVELS
    }
    if all(converged_by_level.values()):
        last_common_step = min(
            points[-1]["step"] for points in converged_by_level.values()
        )
        common_points = {
            level: next(
                point
                for point in converged_by_level[level]
                if point["step"] == last_common_step
            )
            for level in MESH_LEVELS
        }
        common_reactions = {
            level: common_points[level]["indenter_reaction_magnitude_n"]
            for level in MESH_LEVELS
        }
        partial_force_comparison = {
            "available": True,
            "last_common_converged_step": last_common_step,
            "prescribed_motion_mm": common_points["fine"][
                "prescribed_motion_mm"
            ],
            "indentation_after_gap_mm": common_points["fine"][
                "indentation_after_gap_mm"
            ],
            "reaction_n": common_reactions,
            "medium_fine_relative_difference": _relative_difference(
                common_reactions["medium"], common_reactions["fine"]
            ),
        }
    else:
        partial_force_comparison = {
            "available": False,
            "reason": "No common converged step exists across all meshes",
        }

    if all_pass:
        final_reactions = {
            level: by_level[level]["final"]["indenter_reaction_magnitude_n"]
            for level in MESH_LEVELS
        }
        medium_fine_difference = _relative_difference(
            final_reactions["medium"], final_reactions["fine"]
        )
        force_convergence = {
            "available": True,
            "final_reaction_n": final_reactions,
            "medium_fine_relative_difference": medium_fine_difference,
            "medium_fine_pass": (
                medium_fine_difference < MESH_REACTION_RELATIVE_TOLERANCE
            ),
        }
    else:
        force_convergence = {
            "available": False,
            "medium_fine_pass": False,
            "reason": "At least one mesh case failed",
        }
    acceptance = all_pass and force_convergence["medium_fine_pass"]
    return {
        "all_mesh_cases_pass": all_pass,
        "force_convergence": force_convergence,
        "partial_force_comparison": partial_force_comparison,
        "phase3_acceptance_pass": acceptance,
        "recommendation": (
            "retain_mixed_hyperelastic_alm_contact_stack"
            if acceptance
            else "do_not_advance_contact_stack"
        ),
        "acceptance_thresholds": {
            "medium_fine_final_reaction_relative_difference": (
                MESH_REACTION_RELATIVE_TOLERANCE
            ),
            "normalized_force_curve_second_difference": (
                SMOOTHNESS_SECOND_DIFFERENCE_LIMIT
            ),
            "volumetric_checkerboard_mode_ratio": CHECKERBOARD_MODE_LIMIT,
            "volumetric_neighbor_residual_rms_ratio": (
                CHECKERBOARD_RESIDUAL_LIMIT
            ),
            "contact_pressure_roughness_ratio": PRESSURE_ROUGHNESS_LIMIT,
        },
    }


def _run_case_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
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
        timeout=600,
    )


def _run_all(output_path: Path) -> int:
    start = time.perf_counter()
    cases: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(prefix="lit-phase3-cases-") as temporary:
        temporary_path = Path(temporary)
        for mesh_level in MESH_LEVELS:
            case_output = temporary_path / f"{mesh_level}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "case",
                "--mesh-level",
                mesh_level,
                "--output",
                str(case_output),
            ]
            commands.append(command)
            completed = _run_case_subprocess(command)
            if case_output.is_file():
                case = json.loads(case_output.read_text(encoding="utf-8"))
                case["process_exit_code"] = completed.returncode
                if completed.stdout or completed.stderr:
                    case["process_output_tail"] = (
                        completed.stdout + completed.stderr
                    )[-4000:]
            else:
                case = {
                    "case_id": f"phase3_localized_indentation__{mesh_level}",
                    "mesh_level": mesh_level,
                    "status": "FAIL",
                    "failure_reason": "case_process_failed_without_output",
                    "process_exit_code": completed.returncode,
                    "process_output_tail": (
                        completed.stdout + completed.stderr
                    )[-4000:],
                }
            cases.append(case)

    analysis = _analyze_cases(cases)
    output = {
        "phase": "3",
        "benchmark_execution_status": "COMPLETE",
        "phase3_acceptance_pass": analysis["phase3_acceptance_pass"],
        "kratos_version": KM.Kernel.Version(),
        "python_executable": sys.executable,
        "configuration": {
            "element": ELEMENT_NAME,
            "constitutive_law": CONSTITUTIVE_LAW_NAME,
            "poisson_ratio": POISSON_RATIO,
            "contact_process": CONTACT_PROCESS_NAME,
            "mortar_type": MORTAR_TYPE,
            "units": {"length": "mm", "force": "N", "stress": "MPa"},
            "block": {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM},
            "indenter": {
                "radius_mm": INDENTER_RADIUS_MM,
                "half_span_mm": INDENTER_HALF_SPAN_MM,
                "kinematic_carrier_thickness_mm": (
                    INDENTER_CARRIER_THICKNESS_MM
                ),
                "kinematic_carrier_element": "TotalLagrangianElement2D3N",
                "initial_gap_mm": INITIAL_GAP_MM,
                "final_prescribed_motion_mm": FINAL_PRESCRIBED_MOTION_MM,
                "target_indentation_after_gap_mm": TARGET_INDENTATION_MM,
            },
            "steps": NUMBER_OF_STEPS,
            "mesh_divisions": MESH_LEVELS,
            "nonlinear_solver": {
                "strategy": "standard_newton_raphson_contact_strategy",
                "convergence_criterion": "contact_residual_criterion",
                "displacement_and_other_dof_relative_tolerance": (
                    RELATIVE_TOLERANCE
                ),
                "displacement_and_other_dof_absolute_tolerance": (
                    ABSOLUTE_TOLERANCE
                ),
                "contact_residual_relative_tolerance": 1.0e-4,
                "contact_residual_absolute_tolerance": 1.0e-9,
                "maximum_iterations": MAXIMUM_NEWTON_ITERATIONS,
                "simplified_semi_smooth_newton": False,
            },
            "contact_parameter_policy": (
                "Kratos 10.3 ALMContactProcess defaults, identical for every mesh"
            ),
            "contact_pair": {
                "slave": "rounded IndenterSurface",
                "master": "deformable BlockTop",
                "selection_basis": (
                    "Kratos official 2D Hertz contact examples designate the "
                    "upper curved body as slave"
                ),
            },
            "boundary_conditions": {
                "block_bottom": "DISPLACEMENT_X/Y fixed",
                "block_lateral": "free",
                "indenter": "all nodes kinematically constrained; prescribed Y motion",
                "volumetric_strain": "initial 0.0 and free on every node",
                "displacement_z": "DOF retained for runtime Check",
            },
        },
        "official_sources": {
            "phase1_contact": OFFICIAL_CONTACT_SOURCE,
            "mixed_element": OFFICIAL_MIXED_SOURCE,
        },
        "cases": cases,
        "analysis": analysis,
        "commands": commands,
        "total_wall_clock_seconds": time.perf_counter() - start,
    }
    _write_json(output_path, output)
    return 0


def main() -> int:
    arguments = _parse_arguments()
    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    if arguments.command == "case":
        result = _run_case(arguments.mesh_level)
        _write_json(arguments.output, result)
        return 0
    return _run_all(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
