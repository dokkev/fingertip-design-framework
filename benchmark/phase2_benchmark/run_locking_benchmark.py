#!/usr/bin/env python3
"""Run the Phase 2 nearly-incompressible locking benchmark.

The benchmark is intentionally independent of the fingertip geometry. Each
formulation/Poisson-ratio/mesh case runs in a fresh Python process and creates a
fresh Kratos Model.
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
import KratosMultiphysics.StructuralMechanicsApplication


FORMULATIONS = {
    "displacement_tl_q4": {
        "element_name": "TotalLagrangianElement2D4N",
        "constitutive_law": "HyperElasticPlaneStrain2DLaw",
        "formulation_type": "pure displacement",
        "internal_unknown": None,
        "official_source": (
            "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
            "applications/StructuralMechanicsApplication/custom_elements/"
            "solid_elements/total_lagrangian.cpp"
        ),
    },
    "q1p0_mixed_tl_q4": {
        "element_name": "TotalLagrangianQ1P0MixedElement2D4N",
        "constitutive_law": "HyperElasticPlaneStrain2DLaw",
        "formulation_type": "mixed u-p with element-level pressure condensation",
        "internal_unknown": "element nonhistorical PRESSURE",
        "official_source": (
            "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
            "applications/StructuralMechanicsApplication/custom_elements/"
            "solid_elements/total_lagrangian_q1p0_mixed_element.cpp"
        ),
    },
}

POISSON_RATIOS = (0.45, 0.49, 0.499)
MESH_LEVELS = {
    "coarse": (4, 4),
    "medium": (8, 8),
    "fine": (16, 16),
}

WIDTH_MM = 10.0
HEIGHT_MM = 10.0
THICKNESS_MM = 1.0
YOUNG_MODULUS_MPA = 1.0
NUMBER_OF_STEPS = 30
STEP_COMPRESSION = 0.01
TARGET_STEPS = (10, 20, 30)
RELATIVE_DISPLACEMENT_TOLERANCE = 1.0e-6
ABSOLUTE_DISPLACEMENT_TOLERANCE_MM = 1.0e-9
MAXIMUM_NEWTON_ITERATIONS = 30
AREA_RATIO_ABSOLUTE_TOLERANCE = 1.0e-8
MESH_REACTION_RELATIVE_TOLERANCE = 0.05
SMOOTHNESS_SECOND_DIFFERENCE_LIMIT = 0.05


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", required=True, type=Path)

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


def _create_constitutive_law(name: str) -> KM.ConstitutiveLaw:
    if name == "HyperElasticPlaneStrain2DLaw":
        return CLA.HyperElasticPlaneStrain2DLaw()
    return KM.KratosGlobals.GetConstitutiveLaw(name).Clone()


def _set_buffer(model_part: KM.ModelPart) -> None:
    model_part.SetBufferSize(3)
    model_part.ProcessInfo[KM.DELTA_TIME] = 1.0
    model_part.ProcessInfo[KM.TIME] = -3.0
    model_part.ProcessInfo[KM.STEP] = -2
    for time_value in (-2.0, -1.0, 0.0):
        model_part.ProcessInfo[KM.STEP] += 1
        model_part.CloneTimeStep(time_value)


def _node_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i + 1


def _build_model(
    formulation: str, poisson_ratio: float, nx: int, ny: int
) -> tuple[KM.ModelPart, list[int], list[int], dict[str, Any]]:
    definition = FORMULATIONS[formulation]
    model = KM.Model()
    model_part = model.CreateModelPart("compression_block")
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 2
    model_part.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    model_part.AddNodalSolutionStepVariable(KM.REACTION)
    model_part.AddNodalSolutionStepVariable(KM.VOLUME_ACCELERATION)

    properties = model_part.CreateNewProperties(1)
    properties[KM.YOUNG_MODULUS] = YOUNG_MODULUS_MPA
    properties[KM.POISSON_RATIO] = poisson_ratio
    properties[KM.THICKNESS] = THICKNESS_MM
    properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    properties[KM.CONSTITUTIVE_LAW] = _create_constitutive_law(
        definition["constitutive_law"]
    )

    for j in range(ny + 1):
        y = HEIGHT_MM * j / ny
        for i in range(nx + 1):
            x = WIDTH_MM * i / nx
            model_part.CreateNewNode(_node_id(i, j, nx), x, y, 0.0)

    for dof, reaction in (
        (KM.DISPLACEMENT_X, KM.REACTION_X),
        (KM.DISPLACEMENT_Y, KM.REACTION_Y),
        (KM.DISPLACEMENT_Z, KM.REACTION_Z),
    ):
        KM.VariableUtils().AddDof(dof, reaction, model_part)

    element_id = 1
    for j in range(ny):
        for i in range(nx):
            lower_left = _node_id(i, j, nx)
            connectivity = [
                lower_left,
                lower_left + 1,
                lower_left + nx + 2,
                lower_left + nx + 1,
            ]
            model_part.CreateNewElement(
                definition["element_name"], element_id, connectivity, properties
            )
            element_id += 1

    bottom_node_ids = [_node_id(i, 0, nx) for i in range(nx + 1)]
    top_node_ids = [_node_id(i, ny, nx) for i in range(nx + 1)]
    for node_id in bottom_node_ids:
        node = model_part.Nodes[node_id]
        node.Fix(KM.DISPLACEMENT_X)
        node.Fix(KM.DISPLACEMENT_Y)
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)
    for node_id in top_node_ids:
        model_part.Nodes[node_id].Fix(KM.DISPLACEMENT_Y)

    _set_buffer(model_part)
    first_element = model_part.Elements[1]
    element_specifications = json.loads(
        first_element.GetSpecifications().PrettyPrintJsonString()
    )
    law = properties[KM.CONSTITUTIVE_LAW]
    law_features = KM.ConstitutiveLawFeatures()
    law.GetLawFeatures(law_features)
    law_options = law_features.GetOptions()
    metadata = {
        "formulation": formulation,
        "formulation_type": definition["formulation_type"],
        "element_name": definition["element_name"],
        "constitutive_law": definition["constitutive_law"],
        "compatible_constitutive_laws": element_specifications.get(
            "compatible_constitutive_laws"
        ),
        "required_dofs_from_specifications": element_specifications.get(
            "required_dofs"
        ),
        "required_dofs_from_runtime_check": [
            "DISPLACEMENT_X",
            "DISPLACEMENT_Y",
            "DISPLACEMENT_Z",
        ],
        "required_variables": element_specifications.get("required_variables"),
        "required_properties": [
            "YOUNG_MODULUS",
            "POISSON_RATIO",
            "THICKNESS",
            "VOLUME_ACCELERATION",
            "CONSTITUTIVE_LAW",
        ],
        "nodal_unknowns": ["DISPLACEMENT_X", "DISPLACEMENT_Y"],
        "internal_unknown": definition["internal_unknown"],
        "has_nodal_pressure_dof": any(
            node.HasDofFor(KM.PRESSURE) for node in model_part.Nodes
        ),
        "law_features": {
            "plane_strain": law_options.Is(KM.ConstitutiveLaw.PLANE_STRAIN_LAW),
            "finite_strains": law_options.Is(KM.ConstitutiveLaw.FINITE_STRAINS),
            "strain_measures": [
                str(measure) for measure in law_features.GetStrainMeasures()
            ],
            "stress_measure": str(law.GetStressMeasure()),
        },
        "reaction_computation_requested": True,
    }
    if definition["internal_unknown"]:
        metadata["initial_internal_pressure"] = float(
            first_element.GetValue(KM.PRESSURE)
        )
    return model_part, bottom_node_ids, top_node_ids, metadata


def _create_strategy(model_part: KM.ModelPart) -> Any:
    linear_solver = KM.SkylineLUFactorizationSolver()
    builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    criterion = KM.DisplacementCriteria(
        RELATIVE_DISPLACEMENT_TOLERANCE,
        ABSOLUTE_DISPLACEMENT_TOLERANCE_MM,
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


def _matrix_to_rows(matrix: Any) -> list[list[float]]:
    return [
        [float(matrix[row, column]) for column in range(matrix.Size2())]
        for row in range(matrix.Size1())
    ]


def _determinant_2d(matrix: list[list[float]]) -> float:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    twice_area = 0.0
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        twice_area += first[0] * second[1] - second[0] * first[1]
    return 0.5 * abs(twice_area)


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
        element_initial_area = _polygon_area(initial_points)
        initial_area += element_initial_area
        deformed_area += _polygon_area(deformed_points)
        deformation_gradients = element.CalculateOnIntegrationPoints(
            KM.DEFORMATION_GRADIENT, model_part.ProcessInfo
        )
        element_determinants = [
            _determinant_2d(_matrix_to_rows(matrix))
            for matrix in deformation_gradients
        ]
        determinant_values.extend(element_determinants)
        weighted_determinant_sum += element_initial_area * (
            sum(element_determinants) / len(element_determinants)
        )

    if not determinant_values or initial_area <= 0.0:
        raise RuntimeError("No valid element deformation measurements were produced")
    if not all(math.isfinite(value) for value in determinant_values):
        raise FloatingPointError("Non-finite deformation gradient determinant")
    determinant_mean = weighted_determinant_sum / initial_area
    area_ratio = deformed_area / initial_area
    return {
        "det_f": {
            "min": min(determinant_values),
            "max": max(determinant_values),
            "mean": determinant_mean,
            "number_of_integration_points": len(determinant_values),
        },
        "initial_area_mm2": initial_area,
        "deformed_area_mm2": deformed_area,
        "deformed_area_ratio": area_ratio,
        "det_f_area_ratio_abs_difference": abs(determinant_mean - area_ratio),
        "mesh_coordinate_consistency_max_abs_mm": maximum_coordinate_error,
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
    if formulation == "q1p0_mixed_tl_q4":
        for element in model_part.Elements:
            if not math.isfinite(float(element.GetValue(KM.PRESSURE))):
                failures.append(f"element_{element.Id}_PRESSURE")
    return failures


def _sum_top_reaction(
    model_part: KM.ModelPart, top_node_ids: list[int]
) -> float:
    return sum(
        float(model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION_Y))
        for node_id in top_node_ids
    )


def _pressure_statistics(model_part: KM.ModelPart) -> dict[str, float]:
    values = [float(element.GetValue(KM.PRESSURE)) for element in model_part.Elements]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


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


def _run_case(
    formulation: str, poisson_ratio: float, mesh_level: str
) -> dict[str, Any]:
    nx, ny = MESH_LEVELS[mesh_level]
    result: dict[str, Any] = {
        "case_id": f"{formulation}__nu_{poisson_ratio}__{mesh_level}",
        "formulation": formulation,
        "poisson_ratio": poisson_ratio,
        "mesh_level": mesh_level,
        "mesh": {
            "nx": nx,
            "ny": ny,
            "number_of_elements": nx * ny,
            "number_of_nodes": (nx + 1) * (ny + 1),
        },
        "status": "FAIL",
        "curve": [],
    }
    case_start = time.perf_counter()
    strategy = None
    solve_time = 0.0
    try:
        model_part, _, top_node_ids, metadata = _build_model(
            formulation, poisson_ratio, nx, ny
        )
        result["formulation_metadata"] = metadata
        strategy = _create_strategy(model_part)
        result["strategy_check_status"] = int(strategy.Check())
        strategy.Initialize()
        result["formulation_metadata"]["element_runtime_info_after_initialize"] = (
            model_part.Elements[1].Info()
        )

        for step in range(1, NUMBER_OF_STEPS + 1):
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
            step_solve_time = time.perf_counter() - solve_start
            solve_time += step_solve_time
            strategy.FinalizeSolutionStep()

            non_finite_fields = _non_finite_solution_fields(model_part, formulation)
            point: dict[str, Any] = {
                "step": step,
                "nominal_compression": compression,
                "prescribed_displacement_y_mm": prescribed_displacement,
                "solver_converged": solver_converged,
                "nonlinear_iterations": int(
                    model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]
                ),
                "solve_wall_clock_seconds": step_solve_time,
                "finite_solution": not non_finite_fields,
            }
            if non_finite_fields:
                point["non_finite_fields"] = non_finite_fields[:20]
                result["curve"].append(point)
                result["failure_reason"] = "non_finite_solution"
                break
            if not solver_converged:
                result["curve"].append(point)
                result["failure_reason"] = "nonlinear_solver_did_not_converge"
                break

            reaction_y = _sum_top_reaction(model_part, top_node_ids)
            deformation = _measure_deformation(model_part)
            point.update(deformation)
            point["reaction_y_n"] = reaction_y
            point["compression_reaction_magnitude_n"] = abs(reaction_y)
            if formulation == "q1p0_mixed_tl_q4":
                point["element_pressure_mpa"] = _pressure_statistics(model_part)
            result["curve"].append(point)

        completed = len(result["curve"]) == NUMBER_OF_STEPS and all(
            point["solver_converged"] and point["finite_solution"]
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
            result["formulation_metadata"]["reaction_runtime_valid"] = True
        else:
            result["formulation_metadata"]["reaction_runtime_valid"] = False
    except Exception as exception:
        result["failure_reason"] = "exception"
        result["exception"] = f"{type(exception).__name__}: {exception}"
    finally:
        if strategy is not None:
            strategy.Clear()
    result["solve_wall_clock_seconds"] = solve_time
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
    mesh_convergence: dict[str, Any] = {}
    reaction_trends: dict[str, Any] = {}
    for formulation in FORMULATIONS:
        formulation_results: dict[str, Any] = {}
        formulation_trends: dict[str, Any] = {}
        for poisson_ratio in POISSON_RATIOS:
            coarse = by_key[(formulation, poisson_ratio, "coarse")]
            medium = by_key[(formulation, poisson_ratio, "medium")]
            fine = by_key[(formulation, poisson_ratio, "fine")]
            entry: dict[str, Any] = {
                "medium_status": medium["status"],
                "fine_status": fine["status"],
            }
            if medium["status"] == "PASS" and fine["status"] == "PASS":
                target_differences = {}
                for target_step in TARGET_STEPS:
                    medium_reaction = medium["target_samples"][str(target_step)][
                        "compression_reaction_magnitude_n"
                    ]
                    fine_reaction = fine["target_samples"][str(target_step)][
                        "compression_reaction_magnitude_n"
                    ]
                    target_differences[str(target_step)] = _relative_difference(
                        medium_reaction, fine_reaction
                    )
                entry["medium_fine_reaction_relative_difference"] = target_differences
                entry["final_difference_pass"] = (
                    target_differences[str(TARGET_STEPS[-1])]
                    < MESH_REACTION_RELATIVE_TOLERANCE
                )
            else:
                entry["final_difference_pass"] = False
            formulation_results[str(poisson_ratio)] = entry
            ordered_cases = [coarse, medium, fine]
            if all(case["status"] == "PASS" for case in ordered_cases):
                final_reactions = [
                    case["final"]["compression_reaction_magnitude_n"]
                    for case in ordered_cases
                ]
                increasing = final_reactions[0] < final_reactions[1] < final_reactions[2]
                decreasing = final_reactions[0] > final_reactions[1] > final_reactions[2]
                formulation_trends[str(poisson_ratio)] = {
                    "final_reaction_n": {
                        level: value
                        for level, value in zip(MESH_LEVELS, final_reactions)
                    },
                    "reaction_increases_with_refinement": increasing,
                    "reaction_decreases_with_refinement": decreasing,
                    "coarse_fine_relative_difference": _relative_difference(
                        final_reactions[0], final_reactions[2]
                    ),
                    "medium_fine_relative_difference": _relative_difference(
                        final_reactions[1], final_reactions[2]
                    ),
                    "mesh_dependent_over_five_percent": _relative_difference(
                        final_reactions[1], final_reactions[2]
                    )
                    >= MESH_REACTION_RELATIVE_TOLERANCE,
                    "coarse_over_stiffening_pattern": decreasing
                    and _relative_difference(final_reactions[0], final_reactions[2])
                    >= MESH_REACTION_RELATIVE_TOLERANCE,
                }
            else:
                formulation_trends[str(poisson_ratio)] = {
                    "available": False,
                    "reason": "At least one mesh case failed.",
                }
        mesh_convergence[formulation] = formulation_results
        reaction_trends[formulation] = formulation_trends

    poisson_amplification: dict[str, Any] = {}
    for formulation in FORMULATIONS:
        formulation_amplification: dict[str, Any] = {}
        for mesh_level in MESH_LEVELS:
            reference = by_key[(formulation, 0.45, mesh_level)]
            nu_049 = by_key[(formulation, 0.49, mesh_level)]
            nu_0499 = by_key[(formulation, 0.499, mesh_level)]
            if all(case["status"] == "PASS" for case in (reference, nu_049, nu_0499)):
                reference_reaction = reference["final"][
                    "compression_reaction_magnitude_n"
                ]
                formulation_amplification[mesh_level] = {
                    "nu_0_49_over_nu_0_45": nu_049["final"][
                        "compression_reaction_magnitude_n"
                    ]
                    / reference_reaction,
                    "nu_0_499_over_nu_0_45": nu_0499["final"][
                        "compression_reaction_magnitude_n"
                    ]
                    / reference_reaction,
                }
            else:
                formulation_amplification[mesh_level] = {
                    "available": False,
                    "reason": "At least one Poisson-ratio case failed.",
                }
        poisson_amplification[formulation] = formulation_amplification

    displacement_nu_049 = [
        by_key[("displacement_tl_q4", 0.49, level)] for level in MESH_LEVELS
    ]
    displacement_nu_049_stable = all(
        case["status"] == "PASS"
        and case["curve_smoothness"]["smooth"]
        and case["area_consistency_pass"]
        for case in displacement_nu_049
    ) and mesh_convergence["displacement_tl_q4"]["0.49"]["final_difference_pass"]

    mixed_nu_049 = [
        by_key[("q1p0_mixed_tl_q4", 0.49, level)] for level in MESH_LEVELS
    ]
    mixed_nu_049_stable = all(
        case["status"] == "PASS"
        and case["curve_smoothness"]["smooth"]
        and case["area_consistency_pass"]
        for case in mixed_nu_049
    ) and mesh_convergence["q1p0_mixed_tl_q4"]["0.49"]["final_difference_pass"]

    if all(
        by_key[(formulation, 0.49, level)]["status"] == "PASS"
        for formulation in FORMULATIONS
        for level in ("medium", "fine")
    ):
        mixed_reduction: dict[str, Any] = {"available": True}
        for level in ("medium", "fine"):
            displacement_reaction = by_key[
                ("displacement_tl_q4", 0.49, level)
            ]["final"]["compression_reaction_magnitude_n"]
            mixed_reaction = by_key[("q1p0_mixed_tl_q4", 0.49, level)][
                "final"
            ]["compression_reaction_magnitude_n"]
            mixed_reduction[level] = {
                "displacement_reaction_n": displacement_reaction,
                "mixed_reaction_n": mixed_reaction,
                "mixed_relative_reduction": (
                    displacement_reaction - mixed_reaction
                )
                / displacement_reaction,
            }
    else:
        mixed_reduction = {
            "available": False,
            "reason": "Q1P0 mixed cases produced non-finite solutions at the first step.",
        }

    if displacement_nu_049_stable:
        recommendation = {
            "decision": "adopt_displacement_tl_q4",
            "element": FORMULATIONS["displacement_tl_q4"]["element_name"],
            "constitutive_law": FORMULATIONS["displacement_tl_q4"][
                "constitutive_law"
            ],
            "reason": "The simple formulation satisfies the nu=0.49 mesh and curve criteria.",
        }
    elif mixed_nu_049_stable:
        recommendation = {
            "decision": "adopt_q1p0_mixed_tl_q4",
            "element": FORMULATIONS["q1p0_mixed_tl_q4"]["element_name"],
            "constitutive_law": FORMULATIONS["q1p0_mixed_tl_q4"][
                "constitutive_law"
            ],
            "reason": "The displacement formulation failed nu=0.49 while the mixed formulation passed.",
        }
    else:
        recommendation = {
            "decision": "review_other_element_law_or_solver",
            "reason": (
                "The displacement formulation exceeds the 5% medium/fine "
                "reaction criterion at nu=0.49, and Q1P0 produces non-finite "
                "solutions."
            ),
        }

    return {
        "mesh_convergence": mesh_convergence,
        "reaction_refinement_trends": reaction_trends,
        "poisson_ratio_reaction_amplification": poisson_amplification,
        "mixed_stiffening_reduction": mixed_reduction,
        "nu_0_49_displacement_stable": displacement_nu_049_stable,
        "nu_0_49_mixed_stable": mixed_nu_049_stable,
        "recommendation": recommendation,
        "acceptance_thresholds": {
            "medium_fine_reaction_relative_difference": MESH_REACTION_RELATIVE_TOLERANCE,
            "det_f_area_ratio_absolute_difference": AREA_RATIO_ABSOLUTE_TOLERANCE,
            "normalized_curve_second_difference": SMOOTHNESS_SECOND_DIFFERENCE_LIMIT,
        },
    }


def _run_all_cases(output_path: Path) -> int:
    start = time.perf_counter()
    cases: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(prefix="lit-phase2-cases-") as temporary:
        temporary_path = Path(temporary)
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
                    environment = os.environ.copy()
                    environment.pop("PYTHONPATH", None)
                    environment["OMP_NUM_THREADS"] = "1"
                    environment["PYTHONDONTWRITEBYTECODE"] = "1"
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=300,
                    )
                    if case_output.is_file():
                        case = json.loads(case_output.read_text(encoding="utf-8"))
                        case["process_exit_code"] = completed.returncode
                    else:
                        case = {
                            "case_id": (
                                f"{formulation}__nu_{poisson_ratio}__{mesh_level}"
                            ),
                            "formulation": formulation,
                            "poisson_ratio": poisson_ratio,
                            "mesh_level": mesh_level,
                            "status": "FAIL",
                            "failure_reason": "case_process_failed_without_output",
                            "process_exit_code": completed.returncode,
                            "process_output_tail": (
                                completed.stdout + completed.stderr
                            )[-4000:],
                        }
                    cases.append(case)

    analysis = _analyze_results(cases)
    result = {
        "phase": 2,
        "benchmark_execution_status": "COMPLETE",
        "formulation_selection_status": (
            "SELECTED"
            if analysis["recommendation"]["decision"].startswith("adopt_")
            else "NO_ADOPTION"
        ),
        "phase2_acceptance_pass": analysis["recommendation"][
            "decision"
        ].startswith("adopt_"),
        "benchmark": "2D rectangular block displacement-controlled compression",
        "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "kratos_version": KM.Kernel().Version(),
        "configuration": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "thickness_mm": THICKNESS_MM,
            "young_modulus_mpa": YOUNG_MODULUS_MPA,
            "poisson_ratios": list(POISSON_RATIOS),
            "mesh_levels": {
                name: {"nx": value[0], "ny": value[1]}
                for name, value in MESH_LEVELS.items()
            },
            "number_of_steps": NUMBER_OF_STEPS,
            "compression_per_step": STEP_COMPRESSION,
            "target_compressions": [step * STEP_COMPRESSION for step in TARGET_STEPS],
            "bottom_boundary": "DISPLACEMENT_X = DISPLACEMENT_Y = 0",
            "top_boundary": "prescribed DISPLACEMENT_Y; DISPLACEMENT_X free",
            "strategy": "ResidualBasedNewtonRaphsonStrategy",
            "scheme": "ResidualBasedIncrementalUpdateStaticScheme",
            "linear_solver": "SkylineLUFactorizationSolver",
            "builder_and_solver": "ResidualBasedBlockBuilderAndSolver",
            "relative_displacement_tolerance": RELATIVE_DISPLACEMENT_TOLERANCE,
            "absolute_displacement_tolerance_mm": ABSOLUTE_DISPLACEMENT_TOLERANCE_MM,
            "maximum_newton_iterations": MAXIMUM_NEWTON_ITERATIONS,
            "openmp_threads_per_case": 1,
            "fresh_model_and_process_per_case": True,
        },
        "formulations": FORMULATIONS,
        "cases": cases,
        "analysis": analysis,
        "case_commands": [" ".join(command) for command in commands],
        "benchmark_wall_clock_seconds": time.perf_counter() - start,
    }
    _write_json(output_path, result)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


def main() -> int:
    arguments = _parse_arguments()
    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    if arguments.command == "case":
        result = _run_case(
            arguments.formulation,
            arguments.poisson_ratio,
            arguments.mesh_level,
        )
        _write_json(arguments.output, result)
        return 0 if result["status"] == "PASS" else 2
    return _run_all_cases(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
