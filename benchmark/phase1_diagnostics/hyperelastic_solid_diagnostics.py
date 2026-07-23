#!/usr/bin/env python3
"""Minimal instrumented TL hyperelastic solid based on a Kratos v10.3 test.

This is a custom diagnostic benchmark built from official Kratos components; it
is not an official Kratos example. Geometry, material, boundary conditions, load
steps, and solver settings follow the TL half of
``TestPatchTestLargeStrain._compare_TL_UL_2D_triangle``.

Official source:
https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/applications/StructuralMechanicsApplication/tests/test_patch_test_large_strain.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import KratosMultiphysics as KM
import KratosMultiphysics.ConstitutiveLawsApplication as CLA
import KratosMultiphysics.StructuralMechanicsApplication as SMA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _set_buffer(model_part: KM.ModelPart) -> float:
    buffer_size = 3
    model_part.SetBufferSize(buffer_size)
    model_part.ProcessInfo[KM.DELTA_TIME] = 1.0
    delta_time = model_part.ProcessInfo[KM.DELTA_TIME]
    time = model_part.ProcessInfo[KM.TIME] - delta_time * buffer_size
    model_part.ProcessInfo[KM.TIME] = time
    step = -buffer_size + 1
    for _ in range(buffer_size):
        step += 1
        time += delta_time
        model_part.ProcessInfo[KM.STEP] = step
        model_part.CloneTimeStep(time)
    return float(delta_time)


def _create_strategy(model_part: KM.ModelPart) -> Any:
    linear_solver = KM.SkylineLUFactorizationSolver()
    builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    criterion = KM.DisplacementCriteria(1.0e-10, 1.0e-20)
    criterion.SetEchoLevel(0)
    strategy = KM.ResidualBasedNewtonRaphsonStrategy(
        model_part,
        scheme,
        criterion,
        builder_and_solver,
        20,
        True,
        True,
        True,
    )
    strategy.SetEchoLevel(0)
    strategy.SetUseOldStiffnessInFirstIterationFlag(False)
    return strategy


def _to_json(value: Any) -> Any:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        pass
    else:
        return scalar if math.isfinite(scalar) else None
    if hasattr(value, "Size1") and hasattr(value, "Size2"):
        return [
            [_to_json(value[row, column]) for column in range(value.Size2())]
            for row in range(value.Size1())
        ]
    try:
        return [_to_json(item) for item in value]
    except TypeError:
        return str(value)


def _integration_point_output(
    element: KM.Element, variable: Any, process_info: KM.ProcessInfo
) -> dict[str, Any]:
    try:
        values = element.CalculateOnIntegrationPoints(variable, process_info)
    except Exception as exception:
        return {"supported": False, "error": f"{type(exception).__name__}: {exception}"}
    converted = _to_json(values)
    return {"supported": True, "values": converted}


def _kinematic_determinant_f(model_part: KM.ModelPart) -> list[float]:
    """Compute det(F) at the 2x2 Gauss points from current nodal kinematics."""
    inverse_sqrt_three = 1.0 / math.sqrt(3.0)
    gauss_points = (
        (-inverse_sqrt_three, -inverse_sqrt_three),
        (inverse_sqrt_three, -inverse_sqrt_three),
        (inverse_sqrt_three, inverse_sqrt_three),
        (-inverse_sqrt_three, inverse_sqrt_three),
    )
    nodes = [model_part.Nodes[node_id] for node_id in (1, 2, 3, 4)]
    determinants: list[float] = []
    for xi, eta in gauss_points:
        derivatives = (
            (-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)),
            (0.25 * (1.0 - eta), -0.25 * (1.0 + xi)),
            (0.25 * (1.0 + eta), 0.25 * (1.0 + xi)),
            (-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)),
        )
        reference_jacobian = [[0.0, 0.0], [0.0, 0.0]]
        current_jacobian = [[0.0, 0.0], [0.0, 0.0]]
        for node, (d_n_d_xi, d_n_d_eta) in zip(nodes, derivatives):
            displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
            current_x = node.X0 + displacement[0]
            current_y = node.Y0 + displacement[1]
            reference_jacobian[0][0] += node.X0 * d_n_d_xi
            reference_jacobian[0][1] += node.X0 * d_n_d_eta
            reference_jacobian[1][0] += node.Y0 * d_n_d_xi
            reference_jacobian[1][1] += node.Y0 * d_n_d_eta
            current_jacobian[0][0] += current_x * d_n_d_xi
            current_jacobian[0][1] += current_x * d_n_d_eta
            current_jacobian[1][0] += current_y * d_n_d_xi
            current_jacobian[1][1] += current_y * d_n_d_eta
        reference_determinant = (
            reference_jacobian[0][0] * reference_jacobian[1][1]
            - reference_jacobian[0][1] * reference_jacobian[1][0]
        )
        current_determinant = (
            current_jacobian[0][0] * current_jacobian[1][1]
            - current_jacobian[0][1] * current_jacobian[1][0]
        )
        determinants.append(current_determinant / reference_determinant)
    return determinants


def _matrix_determinants(matrices: list[list[list[float]]]) -> list[float]:
    return [
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        for matrix in matrices
    ]


def _sum_reaction(model_part: KM.ModelPart) -> dict[str, float]:
    total = [0.0, 0.0, 0.0]
    for node in model_part.Nodes:
        value = node.GetSolutionStepValue(KM.REACTION)
        for index in range(3):
            total[index] += value[index]
    return {"x": total[0], "y": total[1], "z": total[2]}


def _build_model() -> tuple[KM.ModelPart, KM.Element]:
    model = KM.Model()
    model_part = model.CreateModelPart("tl_solid_part")
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 2
    model_part.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    model_part.AddNodalSolutionStepVariable(KM.REACTION)
    model_part.AddNodalSolutionStepVariable(KM.VOLUME_ACCELERATION)

    properties = model_part.GetProperties()[1]
    properties[KM.YOUNG_MODULUS] = 210.0e9
    properties[KM.POISSON_RATIO] = 0.3
    properties[KM.THICKNESS] = 1.0
    properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElasticPlaneStrain2DLaw()

    model_part.CreateNewNode(1, 0.0, 0.0, 0.0)
    model_part.CreateNewNode(2, 1.0, 0.0, 0.0)
    model_part.CreateNewNode(3, 1.0, 1.0, 0.0)
    model_part.CreateNewNode(4, 0.0, 1.0, 0.0)

    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, model_part)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, model_part)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, model_part)

    fixed = model_part.CreateSubModelPart("BoundaryConditions")
    fixed.AddNodes([1, 2])
    for node in fixed.Nodes:
        node.Fix(KM.DISPLACEMENT_X)
        node.Fix(KM.DISPLACEMENT_Y)
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)

    loaded = model_part.CreateSubModelPart("LoadConditions")
    loaded.AddNodes([3, 4])
    element = model_part.CreateNewElement(
        "TotalLagrangianElement2D4N", 1, [1, 2, 3, 4], properties
    )
    model_part.CreateNewCondition("LineLoadCondition2D2N", 1, [3, 4], properties)
    return model_part, element


def main() -> int:
    args = _parse_args()
    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    model_part, element = _build_model()
    delta_time = _set_buffer(model_part)
    strategy = _create_strategy(model_part)
    strategy.Check()
    strategy.Initialize()

    law = model_part.GetProperties()[1][KM.CONSTITUTIVE_LAW]
    element_specifications = json.loads(element.GetSpecifications().PrettyPrintJsonString())
    law_features = KM.ConstitutiveLawFeatures()
    law.GetLawFeatures(law_features)
    law_options = law_features.GetOptions()
    steps: list[dict[str, Any]] = []
    time = float(model_part.ProcessInfo[KM.TIME])
    try:
        for step in range(1, 4):
            time += step * delta_time
            model_part.CloneTimeStep(time)
            prescribed_displacement = step * 5.0e-1
            for node in model_part.GetSubModelPart("LoadConditions").Nodes:
                node.Fix(KM.DISPLACEMENT_X)
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, prescribed_displacement)

            strategy.InitializeSolutionStep()
            strategy.Predict()
            converged = bool(strategy.SolveSolutionStep())
            strategy.FinalizeSolutionStep()

            determinant_f = _integration_point_output(
                element, KM.DETERMINANT_F, model_part.ProcessInfo
            )
            determinant_f["meaningful_at_nonzero_deformation"] = bool(
                determinant_f["supported"]
                and any(abs(value) > 1.0e-14 for value in determinant_f["values"])
            )
            deformation_gradient = _integration_point_output(
                element, KM.DEFORMATION_GRADIENT, model_part.ProcessInfo
            )
            deformation_gradient_determinants = _matrix_determinants(
                deformation_gradient["values"]
            )
            kinematic_determinants = _kinematic_determinant_f(model_part)
            outputs = {
                "determinant_f": determinant_f,
                "deformation_gradient": deformation_gradient,
                "deformation_gradient_determinants": {
                    "supported": True,
                    "values": deformation_gradient_determinants,
                },
                "kinematic_determinant_f": {
                    "supported": True,
                    "values": kinematic_determinants,
                    "method": "det(current isoparametric Jacobian) / det(reference Jacobian)",
                    "max_abs_difference_from_deformation_gradient": max(
                        abs(first - second)
                        for first, second in zip(
                            kinematic_determinants, deformation_gradient_determinants
                        )
                    ),
                },
                "green_lagrange_strain": _integration_point_output(
                    element, KM.GREEN_LAGRANGE_STRAIN_VECTOR, model_part.ProcessInfo
                ),
                "pk2_stress": _integration_point_output(
                    element, KM.PK2_STRESS_VECTOR, model_part.ProcessInfo
                ),
                "strain_energy": _integration_point_output(
                    element, KM.STRAIN_ENERGY, model_part.ProcessInfo
                ),
                "internal_energy": _integration_point_output(
                    element, KM.INTERNAL_ENERGY, model_part.ProcessInfo
                ),
            }
            steps.append(
                {
                    "step": step,
                    "time": time,
                    "prescribed_displacement_x": prescribed_displacement,
                    "converged": converged,
                    "nonlinear_iterations": int(
                        model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]
                    ),
                    "reaction_fixed_boundary": _sum_reaction(
                        model_part.GetSubModelPart("BoundaryConditions")
                    ),
                    "reaction_prescribed_boundary": _sum_reaction(
                        model_part.GetSubModelPart("LoadConditions")
                    ),
                    "outputs": outputs,
                }
            )
    finally:
        strategy.Clear()

    result = {
        "diagnostic_status": "PASS" if steps and all(s["converged"] for s in steps) else "FAIL",
        "kratos_version": KM.Kernel().Version(),
        "classification": "custom diagnostic benchmark using official Kratos components",
        "official_upstream": (
            "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
            "applications/StructuralMechanicsApplication/tests/"
            "test_patch_test_large_strain.py"
        ),
        "official_method_basis": "TestPatchTestLargeStrain._compare_TL_UL_2D_triangle",
        "diagnostic_changes": [
            "kept only the Total Lagrangian model from the official TL/UL comparison",
            "added programmatic convergence, reaction, and integration-point output capture",
        ],
        "configuration": {
            "domain_size": 2,
            "element_registered_name": "TotalLagrangianElement2D4N",
            "element_runtime_info": element.Info(),
            "constitutive_law_registered_name": "HyperElasticPlaneStrain2DLaw",
            "constitutive_law_python_type": type(law).__name__,
            "constitutive_law_runtime_info": law.Info(),
            "constitutive_law_working_space_dimension": law.WorkingSpaceDimension(),
            "constitutive_law_strain_size": law.GetStrainSize(),
            "constitutive_law_features": {
                "plane_strain": law_options.Is(KM.ConstitutiveLaw.PLANE_STRAIN_LAW),
                "plane_stress": law_options.Is(KM.ConstitutiveLaw.PLANE_STRESS_LAW),
                "finite_strains": law_options.Is(KM.ConstitutiveLaw.FINITE_STRAINS),
                "infinitesimal_strains": law_options.Is(
                    KM.ConstitutiveLaw.INFINITESIMAL_STRAINS
                ),
                "isotropic": law_options.Is(KM.ConstitutiveLaw.ISOTROPIC),
                "strain_measures": [
                    str(measure) for measure in law_features.GetStrainMeasures()
                ],
                "stress_measure": str(law.GetStressMeasure()),
            },
            "element_specifications": element_specifications,
            "properties": {
                "YOUNG_MODULUS": model_part.GetProperties()[1][KM.YOUNG_MODULUS],
                "POISSON_RATIO": model_part.GetProperties()[1][KM.POISSON_RATIO],
                "THICKNESS": model_part.GetProperties()[1][KM.THICKNESS],
                "VOLUME_ACCELERATION": _to_json(
                    model_part.GetProperties()[1][KM.VOLUME_ACCELERATION]
                ),
            },
            "nodal_variables": ["DISPLACEMENT", "REACTION", "VOLUME_ACCELERATION"],
            "dofs": ["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"],
            "scheme": "ResidualBasedIncrementalUpdateStaticScheme",
            "strategy": "ResidualBasedNewtonRaphsonStrategy",
            "linear_solver": "SkylineLUFactorizationSolver",
            "builder_and_solver": "ResidualBasedBlockBuilderAndSolver",
            "criterion": "DisplacementCriteria(relative=1e-10, absolute=1e-20)",
            "max_iterations": 20,
            "compute_reactions": True,
            "reform_dofs_at_each_step": True,
            "move_mesh": True,
        },
        "steps": steps,
        "final_converged": bool(steps and steps[-1]["converged"]),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["diagnostic_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
