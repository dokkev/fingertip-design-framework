#!/usr/bin/env python3
"""Run the Phase 2B mixed-volumetric-strain locking benchmark.

The fingertip geometry and contact stack are deliberately out of scope.  A
small runtime-contract patch is executed first.  Compression cases are only
launched when that patch produces a finite solution.
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
import KratosMultiphysics.ConstitutiveLawsApplication as CLA
import KratosMultiphysics.StructuralMechanicsApplication as SMA


FORMULATIONS = {
    "displacement_tl_t3": {
        "element_name": "TotalLagrangianElement2D3N",
        "constitutive_law": "HyperElasticPlaneStrain2DLaw",
        "formulation_type": "pure displacement",
        "nodal_unknowns": ["DISPLACEMENT_X", "DISPLACEMENT_Y"],
        "official_source": (
            "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
            "applications/StructuralMechanicsApplication/custom_elements/"
            "solid_elements/total_lagrangian.cpp"
        ),
    },
    "mixed_volumetric_strain_tl_t3": {
        "element_name": (
            "TotalLagrangianMixedVolumetricStrainElement2D3N"
        ),
        "constitutive_law": "HyperElasticPlaneStrain2DLaw",
        "formulation_type": "mixed nodal displacement-volumetric strain",
        "nodal_unknowns": [
            "DISPLACEMENT_X",
            "DISPLACEMENT_Y",
            "VOLUMETRIC_STRAIN",
        ],
        "official_source": (
            "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
            "applications/StructuralMechanicsApplication/custom_elements/"
            "solid_elements/total_lagrangian_mixed_volumetric_strain_element.h"
        ),
        "official_test": (
            "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
            "applications/StructuralMechanicsApplication/tests/cpp_tests/"
            "test_total_lagrangian_mixed_volumetric_strain_element.cpp"
        ),
    },
}

POISSON_RATIOS = (0.45, 0.49, 0.499)
MESH_LEVELS = {"coarse": 4, "medium": 8, "fine": 16}
WIDTH_MM = 10.0
HEIGHT_MM = 10.0
THICKNESS_MM = 1.0
YOUNG_MODULUS_MPA = 1.0
NUMBER_OF_STEPS = 30
STEP_COMPRESSION = 0.01
TARGET_STEPS = (10, 20, 30)
RELATIVE_TOLERANCE = 1.0e-6
ABSOLUTE_DISPLACEMENT_TOLERANCE_MM = 1.0e-9
ABSOLUTE_VOLUMETRIC_STRAIN_TOLERANCE = 1.0e-9
MAXIMUM_NEWTON_ITERATIONS = 30
AREA_RATIO_ABSOLUTE_TOLERANCE = 1.0e-10
MESH_REACTION_RELATIVE_TOLERANCE = 0.05
SMOOTHNESS_SECOND_DIFFERENCE_LIMIT = 0.05


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", required=True, type=Path)

    patch_parser = subparsers.add_parser("patch")
    patch_parser.add_argument("--output", required=True, type=Path)

    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("--formulation", required=True, choices=FORMULATIONS)
    case_parser.add_argument("--poisson-ratio", required=True, type=float)
    case_parser.add_argument("--mesh-level", required=True, choices=MESH_LEVELS)
    case_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_mixed(formulation: str) -> bool:
    return formulation == "mixed_volumetric_strain_tl_t3"


def _set_buffer(model_part: KM.ModelPart) -> None:
    model_part.SetBufferSize(3)
    model_part.ProcessInfo[KM.DELTA_TIME] = 1.0
    model_part.ProcessInfo[KM.TIME] = -3.0
    model_part.ProcessInfo[KM.STEP] = -2
    for time_value in (-2.0, -1.0, 0.0):
        model_part.ProcessInfo[KM.STEP] += 1
        model_part.CloneTimeStep(time_value)


def _create_structured_triangular_mesh(
    model_part: KM.ModelPart, element_name: str, divisions: int
) -> None:
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
    settings["element_name"].SetString(element_name)
    KM.StructuredMeshGeneratorProcess(geometry, model_part, settings).Execute()


def _build_model(
    formulation: str, poisson_ratio: float, divisions: int
) -> tuple[KM.ModelPart, list[int], list[int], dict[str, Any]]:
    definition = FORMULATIONS[formulation]
    model = KM.Model()
    model_part = model.CreateModelPart("compression_block", 1)
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 2
    model_part.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    model_part.AddNodalSolutionStepVariable(KM.REACTION)
    if _is_mixed(formulation):
        model_part.AddNodalSolutionStepVariable(KM.VOLUMETRIC_STRAIN)
        model_part.AddNodalSolutionStepVariable(SMA.REACTION_STRAIN)

    _create_structured_triangular_mesh(
        model_part, definition["element_name"], divisions
    )

    # StructuredMeshGeneratorProcess assigns Properties 0 to every element.
    properties = model_part.Properties[0]
    properties[KM.YOUNG_MODULUS] = YOUNG_MODULUS_MPA
    properties[KM.POISSON_RATIO] = poisson_ratio
    properties[KM.THICKNESS] = THICKNESS_MM
    properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElasticPlaneStrain2DLaw()

    for dof, reaction in (
        (KM.DISPLACEMENT_X, KM.REACTION_X),
        (KM.DISPLACEMENT_Y, KM.REACTION_Y),
        (KM.DISPLACEMENT_Z, KM.REACTION_Z),
    ):
        KM.VariableUtils().AddDof(dof, reaction, model_part)
    if _is_mixed(formulation):
        KM.VariableUtils().AddDof(
            KM.VOLUMETRIC_STRAIN, SMA.REACTION_STRAIN, model_part
        )

    coordinate_tolerance = 1.0e-12
    bottom_node_ids: list[int] = []
    top_node_ids: list[int] = []
    bottom_anchor_id: int | None = None
    for node in model_part.Nodes:
        if _is_mixed(formulation):
            node.SetSolutionStepValue(KM.VOLUMETRIC_STRAIN, 0.0)
        if abs(node.Y0) <= coordinate_tolerance:
            bottom_node_ids.append(node.Id)
            node.Fix(KM.DISPLACEMENT_Y)
            node.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)
            if abs(node.X0) <= coordinate_tolerance:
                bottom_anchor_id = node.Id
        if abs(node.Y0 - HEIGHT_MM) <= coordinate_tolerance:
            top_node_ids.append(node.Id)
            node.Fix(KM.DISPLACEMENT_Y)
    if bottom_anchor_id is None:
        raise RuntimeError("No bottom-left anchor node was generated")
    anchor = model_part.Nodes[bottom_anchor_id]
    anchor.Fix(KM.DISPLACEMENT_X)
    anchor.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)

    _set_buffer(model_part)

    law = properties[KM.CONSTITUTIVE_LAW]
    law_features = KM.ConstitutiveLawFeatures()
    law.GetLawFeatures(law_features)
    law_options = law_features.GetOptions()
    metadata: dict[str, Any] = {
        "formulation": formulation,
        "formulation_type": definition["formulation_type"],
        "element_name": definition["element_name"],
        "constitutive_law": definition["constitutive_law"],
        "official_source": definition["official_source"],
        "required_nodal_variables": ["DISPLACEMENT", "REACTION"],
        "runtime_check_required_displacement_dofs": [
            "DISPLACEMENT_X",
            "DISPLACEMENT_Y",
            "DISPLACEMENT_Z",
        ],
        "required_properties": [
            "YOUNG_MODULUS",
            "POISSON_RATIO",
            "CONSTITUTIVE_LAW",
        ],
        "benchmark_properties": [
            "YOUNG_MODULUS",
            "POISSON_RATIO",
            "THICKNESS",
            "CONSTITUTIVE_LAW",
        ],
        "nodal_unknowns": definition["nodal_unknowns"],
        "law_features": {
            "plane_strain": law_options.Is(KM.ConstitutiveLaw.PLANE_STRAIN_LAW),
            "finite_strains": law_options.Is(KM.ConstitutiveLaw.FINITE_STRAINS),
            "strain_measures": [
                str(measure) for measure in law_features.GetStrainMeasures()
            ],
            "stress_measure": str(law.GetStressMeasure()),
        },
        "boundary_conditions": {
            "bottom_vertical_fixed_node_count": len(bottom_node_ids),
            "bottom_horizontal_anchor_node_id": bottom_anchor_id,
            "bottom_horizontal_fixed_node_count": 1,
            "top_vertical_prescribed_node_count": len(top_node_ids),
            "top_horizontal_fixed_node_count": 0,
            "lateral_horizontal_fixed_node_count": 0,
        },
        "mesh_generation": (
            "StructuredMeshGeneratorProcess; two T3 elements per structured "
            "square cell"
        ),
        "reaction_computation_requested": True,
    }
    if _is_mixed(formulation):
        initial_values = [
            float(node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
            for node in model_part.Nodes
        ]
        metadata.update(
            {
                "official_test": definition["official_test"],
                "required_nodal_variables": [
                    "DISPLACEMENT",
                    "REACTION",
                    "VOLUMETRIC_STRAIN",
                    "REACTION_STRAIN",
                ],
                "volumetric_strain_dof": True,
                "volumetric_strain_reaction": "REACTION_STRAIN",
                "required_dofs_from_element_specifications": [
                    "DISPLACEMENT_X",
                    "DISPLACEMENT_Y",
                    "VOLUMETRIC_STRAIN",
                ],
                "assembled_element_dofs": [
                    "DISPLACEMENT_X",
                    "DISPLACEMENT_Y",
                    "VOLUMETRIC_STRAIN",
                ],
                "runtime_check_dof_discrepancy": (
                    "The 2D element assembles X/Y/VOLUMETRIC_STRAIN and its "
                    "specifications list those DOFs, but SolidElementCheck also "
                    "requires a DISPLACEMENT_Z DOF on every node."
                ),
                "volumetric_strain_initial_value": {
                    "min": min(initial_values),
                    "max": max(initial_values),
                },
                "volumetric_strain_boundary_treatment": (
                    "No VOLUMETRIC_STRAIN DOF is fixed; all nodal values are "
                    "initialized to zero and solved as free unknowns."
                ),
                "deformation_gradient_output": (
                    "Element integration-point DEFORMATION_GRADIENT returns no "
                    "values in this build; F is evaluated from affine T3 nodal "
                    "kinematics without modifying Kratos."
                ),
            }
        )
    return model_part, bottom_node_ids, top_node_ids, metadata


def _create_strategy(model_part: KM.ModelPart, formulation: str) -> Any:
    linear_solver = KM.SkylineLUFactorizationSolver()
    builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    if _is_mixed(formulation):
        criterion = KM.MixedGenericCriteria(
            [
                (
                    KM.DISPLACEMENT,
                    RELATIVE_TOLERANCE,
                    ABSOLUTE_DISPLACEMENT_TOLERANCE_MM,
                ),
                (
                    KM.VOLUMETRIC_STRAIN,
                    RELATIVE_TOLERANCE,
                    ABSOLUTE_VOLUMETRIC_STRAIN_TOLERANCE,
                ),
            ]
        )
    else:
        criterion = KM.DisplacementCriteria(
            RELATIVE_TOLERANCE, ABSOLUTE_DISPLACEMENT_TOLERANCE_MM
        )
    criterion.SetEchoLevel(0)
    strategy = KM.ResidualBasedNewtonRaphsonStrategy(
        model_part,
        scheme,
        criterion,
        builder_and_solver,
        MAXIMUM_NEWTON_ITERATIONS,
        True,
        False,
        True,
    )
    strategy.SetEchoLevel(0)
    return strategy


def _triangle_determinant(points: list[tuple[float, float]]) -> float:
    first, second, third = points
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        third[0] - first[0]
    ) * (second[1] - first[1])


def _measure_deformation(model_part: KM.ModelPart) -> dict[str, Any]:
    determinant_values: list[float] = []
    weighted_determinant_sum = 0.0
    initial_area = 0.0
    deformed_area = 0.0
    maximum_coordinate_error = 0.0
    for node in model_part.Nodes:
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        maximum_coordinate_error = max(
            maximum_coordinate_error,
            abs(node.X - (node.X0 + displacement[0])),
            abs(node.Y - (node.Y0 + displacement[1])),
        )

    for element in model_part.Elements:
        geometry = element.GetGeometry()
        initial_points = [(node.X0, node.Y0) for node in geometry]
        deformed_points = [(node.X, node.Y) for node in geometry]
        initial_determinant = _triangle_determinant(initial_points)
        deformed_determinant = _triangle_determinant(deformed_points)
        if abs(initial_determinant) <= 1.0e-15:
            raise RuntimeError(f"Element {element.Id} has zero reference area")
        det_f = deformed_determinant / initial_determinant
        element_initial_area = 0.5 * abs(initial_determinant)
        element_deformed_area = 0.5 * abs(deformed_determinant)
        determinant_values.append(det_f)
        initial_area += element_initial_area
        deformed_area += element_deformed_area
        weighted_determinant_sum += element_initial_area * det_f

    if not determinant_values or initial_area <= 0.0:
        raise RuntimeError("No deformation measurements were produced")
    if not all(math.isfinite(value) for value in determinant_values):
        raise FloatingPointError("Non-finite deformation gradient determinant")
    determinant_mean = weighted_determinant_sum / initial_area
    area_ratio = deformed_area / initial_area
    return {
        "det_f": {
            "source": "affine T3 nodal kinematics",
            "min": min(determinant_values),
            "max": max(determinant_values),
            "area_weighted_mean": determinant_mean,
            "number_of_elements": len(determinant_values),
        },
        "initial_area_mm2": initial_area,
        "deformed_area_mm2": deformed_area,
        "deformed_area_ratio": area_ratio,
        "det_f_area_ratio_abs_difference": abs(determinant_mean - area_ratio),
        "mesh_coordinate_consistency_max_abs_mm": maximum_coordinate_error,
    }


def _volumetric_strain_statistics(model_part: KM.ModelPart) -> dict[str, float]:
    values = [
        float(node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
        for node in model_part.Nodes
    ]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "max_abs": max(abs(value) for value in values),
    }


def _non_finite_solution_fields(
    model_part: KM.ModelPart, formulation: str
) -> list[str]:
    failures: list[str] = []
    for node in model_part.Nodes:
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        reaction = node.GetSolutionStepValue(KM.REACTION)
        if not all(math.isfinite(displacement[index]) for index in range(2)):
            failures.append(f"node_{node.Id}_DISPLACEMENT")
        if not all(math.isfinite(reaction[index]) for index in range(2)):
            failures.append(f"node_{node.Id}_REACTION")
        if _is_mixed(formulation):
            if not math.isfinite(
                float(node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
            ):
                failures.append(f"node_{node.Id}_VOLUMETRIC_STRAIN")
            if not math.isfinite(
                float(node.GetSolutionStepValue(SMA.REACTION_STRAIN))
            ):
                failures.append(f"node_{node.Id}_REACTION_STRAIN")
    return failures


def _sum_top_reaction(
    model_part: KM.ModelPart, top_node_ids: list[int]
) -> float:
    return sum(
        float(model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION_Y))
        for node_id in top_node_ids
    )


def _curve_smoothness(curve: list[dict[str, Any]]) -> dict[str, Any]:
    reactions = [point["compression_reaction_magnitude_n"] for point in curve]
    if len(reactions) < 3 or reactions[-1] <= 0.0:
        return {
            "monotonic_non_decreasing": False,
            "normalized_max_second_difference": None,
            "smooth": False,
        }
    monotonic_tolerance = 1.0e-10 * reactions[-1]
    monotonic = all(
        second + monotonic_tolerance >= first
        for first, second in zip(reactions, reactions[1:])
    )
    second_differences = [
        abs(reactions[index + 1] - 2.0 * reactions[index] + reactions[index - 1])
        for index in range(1, len(reactions) - 1)
    ]
    normalized_maximum = max(second_differences) / reactions[-1]
    return {
        "monotonic_non_decreasing": monotonic,
        "normalized_max_second_difference": normalized_maximum,
        "smooth": monotonic
        and normalized_maximum <= SMOOTHNESS_SECOND_DIFFERENCE_LIMIT,
    }


def _initialize_runtime(
    model_part: KM.ModelPart, formulation: str
) -> tuple[Any, dict[str, Any]]:
    element_initialize_start = time.perf_counter()
    for element in model_part.Elements:
        element.Initialize(model_part.ProcessInfo)
    element_initialize_time = time.perf_counter() - element_initialize_start
    strategy = _create_strategy(model_part, formulation)
    strategy_check_status = int(strategy.Check())
    strategy.Initialize()
    return strategy, {
        "element_initialize_status": "PASS",
        "element_initialize_wall_clock_seconds": element_initialize_time,
        "strategy_check_status": strategy_check_status,
        "strategy_initialize_status": "PASS",
    }


def _solve_step(
    model_part: KM.ModelPart,
    strategy: Any,
    formulation: str,
    top_node_ids: list[int],
    step: int,
) -> dict[str, Any]:
    model_part.CloneTimeStep(float(step))
    model_part.ProcessInfo[KM.STEP] = step
    compression = STEP_COMPRESSION * step
    prescribed_displacement = -compression * HEIGHT_MM
    for node_id in top_node_ids:
        model_part.Nodes[node_id].SetSolutionStepValue(
            KM.DISPLACEMENT_Y, prescribed_displacement
        )

    strategy.InitializeSolutionStep()
    strategy.Predict()
    solve_start = time.perf_counter()
    solver_converged = bool(strategy.SolveSolutionStep())
    solve_time = time.perf_counter() - solve_start
    strategy.FinalizeSolutionStep()

    non_finite_fields = _non_finite_solution_fields(model_part, formulation)
    point: dict[str, Any] = {
        "step": step,
        "nominal_compression": compression,
        "prescribed_displacement_y_mm": prescribed_displacement,
        "solver_converged": solver_converged,
        "nonlinear_iterations": int(model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]),
        "solve_wall_clock_seconds": solve_time,
        "finite_solution_fields": not non_finite_fields,
    }
    if non_finite_fields:
        point["non_finite_fields"] = non_finite_fields[:20]
        return point
    if not solver_converged:
        return point

    deformation = _measure_deformation(model_part)
    reaction_y = _sum_top_reaction(model_part, top_node_ids)
    point.update(deformation)
    point["reaction_y_n"] = reaction_y
    point["compression_reaction_magnitude_n"] = abs(reaction_y)
    point["finite_reaction"] = math.isfinite(reaction_y)
    point["finite_det_f"] = all(
        math.isfinite(point["det_f"][key])
        for key in ("min", "max", "area_weighted_mean")
    )
    if _is_mixed(formulation):
        point["volumetric_strain"] = _volumetric_strain_statistics(model_part)
        point["finite_volumetric_strain"] = all(
            math.isfinite(value)
            for value in point["volumetric_strain"].values()
        )
    return point


def _run_patch() -> dict[str, Any]:
    formulation = "mixed_volumetric_strain_tl_t3"
    result: dict[str, Any] = {
        "patch_type": "one structured cell split into two T3 elements",
        "status": "FAIL",
        "poisson_ratio": 0.49,
        "first_step_nominal_compression": STEP_COMPRESSION,
    }
    start = time.perf_counter()
    strategy = None
    try:
        model_part, _, top_node_ids, metadata = _build_model(
            formulation, 0.49, 1
        )
        result["mesh"] = {
            "number_of_nodes": model_part.NumberOfNodes(),
            "number_of_elements": model_part.NumberOfElements(),
        }
        result["runtime_contract"] = metadata
        strategy, initialization = _initialize_runtime(model_part, formulation)
        result["runtime_initialization"] = initialization
        point = _solve_step(
            model_part, strategy, formulation, top_node_ids, step=1
        )
        result["first_step"] = point
        valid = (
            point["solver_converged"]
            and point["finite_solution_fields"]
            and point.get("finite_reaction", False)
            and point.get("finite_det_f", False)
            and point.get("finite_volumetric_strain", False)
        )
        if valid:
            result["status"] = "PASS"
        else:
            result["failure_reason"] = "first_nonlinear_step_invalid"
    except Exception as exception:
        result["failure_reason"] = "exception"
        result["exception"] = f"{type(exception).__name__}: {exception}"
    finally:
        if strategy is not None:
            strategy.Clear()
    result["wall_clock_seconds"] = time.perf_counter() - start
    return result


def _run_case(
    formulation: str, poisson_ratio: float, mesh_level: str
) -> dict[str, Any]:
    divisions = MESH_LEVELS[mesh_level]
    result: dict[str, Any] = {
        "case_id": f"{formulation}__nu_{poisson_ratio}__{mesh_level}",
        "formulation": formulation,
        "poisson_ratio": poisson_ratio,
        "mesh_level": mesh_level,
        "mesh": {
            "divisions_x": divisions,
            "divisions_y": divisions,
            "characteristic_cell_size_mm": WIDTH_MM / divisions,
            "number_of_elements": 2 * divisions * divisions,
            "number_of_nodes": (divisions + 1) * (divisions + 1),
        },
        "status": "FAIL",
        "curve": [],
    }
    case_start = time.perf_counter()
    strategy = None
    try:
        model_part, _, top_node_ids, metadata = _build_model(
            formulation, poisson_ratio, divisions
        )
        result["formulation_metadata"] = metadata
        strategy, initialization = _initialize_runtime(model_part, formulation)
        result["runtime_initialization"] = initialization

        for step in range(1, NUMBER_OF_STEPS + 1):
            point = _solve_step(
                model_part, strategy, formulation, top_node_ids, step
            )
            result["curve"].append(point)
            if not point["finite_solution_fields"]:
                result["failure_reason"] = "non_finite_solution_fields"
                break
            if not point["solver_converged"]:
                result["failure_reason"] = "nonlinear_solver_did_not_converge"
                break
            if not point.get("finite_reaction", False):
                result["failure_reason"] = "non_finite_reaction"
                break
            if not point.get("finite_det_f", False):
                result["failure_reason"] = "non_finite_det_f"
                break
            if _is_mixed(formulation) and not point.get(
                "finite_volumetric_strain", False
            ):
                result["failure_reason"] = "non_finite_volumetric_strain"
                break

        completed = len(result["curve"]) == NUMBER_OF_STEPS and all(
            point["solver_converged"] and point["finite_solution_fields"]
            for point in result["curve"]
        )
        if completed:
            result["status"] = "PASS"
            result["final"] = result["curve"][-1]
            result["target_samples"] = {
                str(step): result["curve"][step - 1] for step in TARGET_STEPS
            }
            result["curve_smoothness"] = _curve_smoothness(result["curve"])
            result["maximum_det_f_area_ratio_abs_difference"] = max(
                point["det_f_area_ratio_abs_difference"]
                for point in result["curve"]
            )
            result["area_consistency_pass"] = (
                result["maximum_det_f_area_ratio_abs_difference"]
                <= AREA_RATIO_ABSOLUTE_TOLERANCE
            )
    except Exception as exception:
        result["failure_reason"] = "exception"
        result["exception"] = f"{type(exception).__name__}: {exception}"
    finally:
        if strategy is not None:
            strategy.Clear()
    result["solve_wall_clock_seconds"] = sum(
        point.get("solve_wall_clock_seconds", 0.0) for point in result["curve"]
    )
    result["case_wall_clock_seconds"] = time.perf_counter() - case_start
    return result


def _relative_difference(first: float, reference: float) -> float:
    if reference == 0.0:
        return math.inf
    return abs(first - reference) / abs(reference)


def _analyze_results(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (case["formulation"], case["poisson_ratio"], case["mesh_level"]): case
        for case in cases
    }
    convergence: dict[str, Any] = {}
    for formulation in FORMULATIONS:
        formulation_analysis: dict[str, Any] = {}
        for poisson_ratio in POISSON_RATIOS:
            ordered = [
                by_key[(formulation, poisson_ratio, level)]
                for level in MESH_LEVELS
            ]
            entry: dict[str, Any] = {
                "case_status": {
                    level: case["status"]
                    for level, case in zip(MESH_LEVELS, ordered)
                }
            }
            if all(case["status"] == "PASS" for case in ordered):
                reactions = [
                    case["final"]["compression_reaction_magnitude_n"]
                    for case in ordered
                ]
                target_differences: dict[str, float] = {}
                for target in TARGET_STEPS:
                    medium = ordered[1]["target_samples"][str(target)][
                        "compression_reaction_magnitude_n"
                    ]
                    fine = ordered[2]["target_samples"][str(target)][
                        "compression_reaction_magnitude_n"
                    ]
                    target_differences[str(target)] = _relative_difference(
                        medium, fine
                    )
                entry.update(
                    {
                        "final_reaction_n": {
                            level: reaction
                            for level, reaction in zip(MESH_LEVELS, reactions)
                        },
                        "medium_fine_reaction_relative_difference_by_target_step": (
                            target_differences
                        ),
                        "medium_fine_final_reaction_relative_difference": (
                            target_differences[str(TARGET_STEPS[-1])]
                        ),
                        "coarse_fine_final_reaction_relative_difference": (
                            _relative_difference(reactions[0], reactions[2])
                        ),
                        "medium_fine_final_reaction_pass": (
                            target_differences[str(TARGET_STEPS[-1])]
                            < MESH_REACTION_RELATIVE_TOLERANCE
                        ),
                        "all_curves_smooth": all(
                            case["curve_smoothness"]["smooth"]
                            for case in ordered
                        ),
                        "all_area_consistent": all(
                            case["area_consistency_pass"] for case in ordered
                        ),
                    }
                )
            else:
                entry["medium_fine_final_reaction_pass"] = False
                entry["failure_reason"] = "At least one mesh case failed"
            formulation_analysis[str(poisson_ratio)] = entry
        convergence[formulation] = formulation_analysis

    displacement = convergence["displacement_tl_t3"]["0.49"]
    mixed = convergence["mixed_volumetric_strain_tl_t3"]["0.49"]
    sensitivity_comparison: dict[str, Any]
    if (
        "medium_fine_final_reaction_relative_difference" in displacement
        and "medium_fine_final_reaction_relative_difference" in mixed
    ):
        displacement_sensitivity = displacement[
            "medium_fine_final_reaction_relative_difference"
        ]
        mixed_sensitivity = mixed["medium_fine_final_reaction_relative_difference"]
        if displacement_sensitivity > 0.0:
            relative_reduction = (
                displacement_sensitivity - mixed_sensitivity
            ) / displacement_sensitivity
        else:
            relative_reduction = 0.0
        sensitivity_comparison = {
            "available": True,
            "comparison_mode": "relative_reaction_mesh_sensitivity",
            "displacement_medium_fine_relative_difference": (
                displacement_sensitivity
            ),
            "mixed_medium_fine_relative_difference": mixed_sensitivity,
            "mixed_relative_sensitivity_reduction": relative_reduction,
            "meaningful_reduction": (
                mixed_sensitivity < MESH_REACTION_RELATIVE_TOLERANCE
                and displacement_sensitivity >= MESH_REACTION_RELATIVE_TOLERANCE
            ),
        }
    elif (
        mixed.get("medium_fine_final_reaction_pass", False)
        and mixed.get("all_curves_smooth", False)
        and mixed.get("all_area_consistent", False)
    ):
        failed_displacement_cases = [
            by_key[("displacement_tl_t3", 0.49, level)]
            for level in MESH_LEVELS
            if by_key[("displacement_tl_t3", 0.49, level)]["status"] != "PASS"
        ]
        sensitivity_comparison = {
            "available": True,
            "comparison_mode": "fine_mesh_finite_solution_robustness",
            "quantitative_relative_reduction_available": False,
            "displacement_failed_cases": [
                {
                    "mesh_level": case["mesh_level"],
                    "failure_reason": case.get("failure_reason"),
                    "last_completed_step": len(case.get("curve", [])),
                }
                for case in failed_displacement_cases
            ],
            "mixed_mesh_series_status": {
                level: by_key[
                    ("mixed_volumetric_strain_tl_t3", 0.49, level)
                ]["status"]
                for level in MESH_LEVELS
            },
            "meaningful_reduction": bool(failed_displacement_cases),
            "reason": (
                "The displacement formulation loses finite solutions on a "
                "refined mesh, while all mixed meshes complete with a "
                "sub-5% medium/fine reaction difference."
            ),
        }
    else:
        sensitivity_comparison = {
            "available": False,
            "reason": "One or both nu=0.49 formulation series failed",
        }

    mixed_nu_049_cases = [
        by_key[("mixed_volumetric_strain_tl_t3", 0.49, level)]
        for level in MESH_LEVELS
    ]
    mixed_nu_049_acceptance = (
        all(case["status"] == "PASS" for case in mixed_nu_049_cases)
        and mixed.get("medium_fine_final_reaction_pass", False)
        and mixed.get("all_curves_smooth", False)
        and mixed.get("all_area_consistent", False)
        and sensitivity_comparison.get("meaningful_reduction", False)
    )
    if mixed_nu_049_acceptance:
        recommendation = {
            "decision": "adopt_mixed_volumetric_strain_tl_t3",
            "element": FORMULATIONS["mixed_volumetric_strain_tl_t3"][
                "element_name"
            ],
            "constitutive_law": "HyperElasticPlaneStrain2DLaw",
            "reason": "The mixed element satisfies every nu=0.49 acceptance check.",
        }
    else:
        recommendation = {
            "decision": "end_kratos_element_search_review_other_solver",
            "reason": (
                "The remaining Kratos mixed candidate did not satisfy every "
                "nu=0.49 Phase 2B acceptance check."
            ),
        }

    return {
        "mesh_convergence": convergence,
        "nu_0_49_sensitivity_comparison": sensitivity_comparison,
        "nu_0_49_mixed_acceptance": mixed_nu_049_acceptance,
        "recommendation": recommendation,
        "acceptance_thresholds": {
            "medium_fine_reaction_relative_difference": (
                MESH_REACTION_RELATIVE_TOLERANCE
            ),
            "det_f_area_ratio_absolute_difference": (
                AREA_RATIO_ABSOLUTE_TOLERANCE
            ),
            "normalized_curve_second_difference": (
                SMOOTHNESS_SECOND_DIFFERENCE_LIMIT
            ),
        },
    }


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
        timeout=300,
    )


def _failed_subprocess_result(
    identifier: str, completed: subprocess.CompletedProcess[str]
) -> dict[str, Any]:
    return {
        "case_id": identifier,
        "status": "FAIL",
        "failure_reason": "subprocess_failed_without_output",
        "process_exit_code": completed.returncode,
        "process_output_tail": (completed.stdout + completed.stderr)[-4000:],
    }


def _run_all(output_path: Path) -> int:
    start = time.perf_counter()
    commands: list[list[str]] = []
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lit-phase2b-cases-") as temporary:
        temporary_path = Path(temporary)
        patch_output = temporary_path / "runtime-patch.json"
        patch_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "patch",
            "--output",
            str(patch_output),
        ]
        commands.append(patch_command)
        patch_completed = _run_subprocess(patch_command)
        if patch_output.is_file():
            runtime_patch = json.loads(patch_output.read_text(encoding="utf-8"))
            runtime_patch["process_exit_code"] = patch_completed.returncode
        else:
            runtime_patch = _failed_subprocess_result(
                "mixed_runtime_patch", patch_completed
            )

        if runtime_patch["status"] == "PASS":
            for formulation in FORMULATIONS:
                for poisson_ratio in POISSON_RATIOS:
                    for mesh_level in MESH_LEVELS:
                        case_output = temporary_path / (
                            f"{formulation}-{poisson_ratio}-{mesh_level}.json"
                        )
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "case",
                            "--formulation",
                            formulation,
                            "--poisson-ratio",
                            str(poisson_ratio),
                            "--mesh-level",
                            mesh_level,
                            "--output",
                            str(case_output),
                        ]
                        commands.append(command)
                        completed = _run_subprocess(command)
                        if case_output.is_file():
                            case = json.loads(
                                case_output.read_text(encoding="utf-8")
                            )
                            case["process_exit_code"] = completed.returncode
                        else:
                            case = _failed_subprocess_result(
                                (
                                    f"{formulation}__nu_{poisson_ratio}__"
                                    f"{mesh_level}"
                                ),
                                completed,
                            )
                            case.update(
                                {
                                    "formulation": formulation,
                                    "poisson_ratio": poisson_ratio,
                                    "mesh_level": mesh_level,
                                }
                            )
                        cases.append(case)

    if runtime_patch["status"] == "PASS" and len(cases) == 18:
        analysis = _analyze_results(cases)
        acceptance_pass = analysis["nu_0_49_mixed_acceptance"]
        execution_status = "COMPLETE"
    else:
        analysis = {
            "recommendation": {
                "decision": "end_kratos_element_search_review_other_solver",
                "reason": "The mandatory mixed-element runtime patch failed.",
            }
        }
        acceptance_pass = False
        execution_status = "STOPPED_AFTER_PATCH_FAILURE"

    output = {
        "phase": "2B",
        "benchmark_execution_status": execution_status,
        "phase2b_acceptance_pass": acceptance_pass,
        "formulation_selection_status": (
            "ADOPT_MIXED_VOLUMETRIC_STRAIN_TL_T3"
            if acceptance_pass
            else "NO_ADOPTION"
        ),
        "kratos_version": KM.Kernel.Version(),
        "python_executable": sys.executable,
        "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "physical_configuration": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "thickness_mm": THICKNESS_MM,
            "young_modulus_mpa": YOUNG_MODULUS_MPA,
            "number_of_steps": NUMBER_OF_STEPS,
            "compression_increment": STEP_COMPRESSION,
            "poisson_ratios": list(POISSON_RATIOS),
            "mesh_divisions": MESH_LEVELS,
        },
        "boundary_conditions": {
            "bottom": (
                "DISPLACEMENT_Y fixed on all nodes; DISPLACEMENT_X fixed only "
                "at the bottom-left anchor"
            ),
            "top": "DISPLACEMENT_Y prescribed; DISPLACEMENT_X free",
            "lateral": "DISPLACEMENT_X free",
            "volumetric_strain": "initialized to zero and free at every node",
        },
        "phase2a_boundary_condition_difference": {
            "phase2a": "DISPLACEMENT_X and DISPLACEMENT_Y fixed on every bottom node",
            "phase2b": (
                "Only bottom DISPLACEMENT_Y plus one bottom-node "
                "DISPLACEMENT_X are fixed"
            ),
            "comparison_action": (
                "Phase 2A results are unchanged. A T3 displacement control "
                "series is rerun with exactly the Phase 2B boundary conditions."
            ),
        },
        "runtime_patch": runtime_patch,
        "cases": cases,
        "analysis": analysis,
        "commands": commands,
        "total_wall_clock_seconds": time.perf_counter() - start,
    }
    _write_json(output_path, output)
    return 0


def main() -> int:
    arguments = _parse_arguments()
    if arguments.command == "patch":
        result = _run_patch()
        _write_json(arguments.output, result)
        return 0
    if arguments.command == "case":
        result = _run_case(
            arguments.formulation,
            arguments.poisson_ratio,
            arguments.mesh_level,
        )
        _write_json(arguments.output, result)
        return 0
    return _run_all(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
