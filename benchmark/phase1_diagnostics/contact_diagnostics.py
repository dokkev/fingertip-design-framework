#!/usr/bin/env python3
"""Instrument the official Kratos v10.3.0 2D ALM contact fixture.

This is a diagnostic wrapper, not an official Kratos test. It reads the official
fixture in-place from a Kratos v10.3.0 source checkout and changes only
``solver_settings.compute_reactions`` in memory.

Official source:
https://github.com/KratosMultiphysics/Kratos/tree/v10.3.0/applications/ContactStructuralMechanicsApplication/tests/ALM_frictionless_contact_test_2D
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import KratosMultiphysics as KM
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
import KratosMultiphysics.ConstitutiveLawsApplication
import KratosMultiphysics.StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)


OFFICIAL_RELATIVE_PARAMETERS = Path(
    "applications/ContactStructuralMechanicsApplication/tests/"
    "ALM_frictionless_contact_test_2D/hyper_simple_patch_test_parameters.json"
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _model_part_names(model_part: KM.ModelPart) -> list[str]:
    names = [model_part.FullName()]
    for child in model_part.SubModelParts:
        names.extend(_model_part_names(child))
    return names


def _condition_snapshot(model_part: KM.ModelPart) -> dict[str, Any]:
    conditions = list(model_part.Conditions)
    return {
        "count": len(conditions),
        "names": sorted({condition.Info() for condition in conditions}),
        "active_count": sum(condition.Is(KM.ACTIVE) for condition in conditions),
        "slave_count": sum(condition.Is(KM.SLAVE) for condition in conditions),
        "master_count": sum(condition.Is(KM.MASTER) for condition in conditions),
        "slip_count": sum(condition.Is(KM.SLIP) for condition in conditions),
    }


def _nodal_values(nodes: list[KM.Node], variable: Any) -> dict[str, Any]:
    historical: dict[str, float | None] = {}
    nonhistorical: dict[str, float | None] = {}
    for node in nodes:
        if node.SolutionStepsDataHas(variable):
            historical[str(node.Id)] = _finite_float(node.GetSolutionStepValue(variable))
        if node.Has(variable):
            nonhistorical[str(node.Id)] = _finite_float(node.GetValue(variable))
    return {
        "historical": historical,
        "nonhistorical": nonhistorical,
        "historical_available": bool(historical),
        "nonhistorical_available": bool(nonhistorical),
    }


def _condition_values(
    conditions: list[KM.Condition], variable: Any, process_info: KM.ProcessInfo
) -> dict[str, Any]:
    stored: dict[str, float | None] = {}
    integration_point: dict[str, list[float | None]] = {}
    for condition in conditions:
        if condition.Has(variable):
            stored[str(condition.Id)] = _finite_float(condition.GetValue(variable))
        try:
            values = condition.CalculateOnIntegrationPoints(variable, process_info)
        except Exception:  # Variable support is what this diagnostic probes.
            continue
        converted = [_finite_float(value) for value in values]
        if converted:
            integration_point[str(condition.Id)] = converted
    return {
        "stored": stored,
        "integration_point": integration_point,
        "stored_available": bool(stored),
        "integration_point_available": bool(integration_point),
    }


def _reaction_sum(model_part: KM.ModelPart) -> dict[str, float]:
    reaction = KM.Vector(3)
    reaction[0] = 0.0
    reaction[1] = 0.0
    reaction[2] = 0.0
    for node in model_part.Nodes:
        nodal_reaction = node.GetSolutionStepValue(KM.REACTION)
        reaction[0] += nodal_reaction[0]
        reaction[1] += nodal_reaction[1]
        reaction[2] += nodal_reaction[2]
    return {"x": float(reaction[0]), "y": float(reaction[1])}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kratos-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parameters_path = (args.kratos_root / OFFICIAL_RELATIVE_PARAMETERS).resolve()
    tests_directory = parameters_path.parents[1]
    if not parameters_path.is_file():
        raise FileNotFoundError(parameters_path)

    parameters = KM.Parameters(parameters_path.read_text(encoding="utf-8"))
    solver_settings = parameters["solver_settings"]
    if solver_settings.Has("compute_reactions"):
        solver_settings["compute_reactions"].SetBool(True)
    else:
        solver_settings.AddEmptyValue("compute_reactions").SetBool(True)

    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(model, parameters)
    previous_directory = Path.cwd()
    result: dict[str, Any] = {}
    try:
        os.chdir(tests_directory)
        analysis.Initialize()

        structure = model["Structure"]
        contact = model["Structure.Contact"]
        computing_contact = model["Structure.ComputingContact"]
        contact_nodes = list(contact.Nodes)
        root_nodes = list(structure.Nodes)
        initial_interface_conditions = _condition_snapshot(contact)
        initial_computing_conditions = _condition_snapshot(computing_contact)

        first_node = root_nodes[0]
        variable_and_dof_status = {
            "historical_variables": {
                "NORMAL": first_node.SolutionStepsDataHas(KM.NORMAL),
                "NODAL_H": first_node.SolutionStepsDataHas(KM.NODAL_H),
                "WEIGHTED_GAP": first_node.SolutionStepsDataHas(CSMA.WEIGHTED_GAP),
                "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE": first_node.SolutionStepsDataHas(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                ),
                "WEIGHTED_SCALAR_RESIDUAL": first_node.SolutionStepsDataHas(
                    CSMA.WEIGHTED_SCALAR_RESIDUAL
                ),
            },
            "dofs": {
                "DISPLACEMENT_X": first_node.HasDofFor(KM.DISPLACEMENT_X),
                "DISPLACEMENT_Y": first_node.HasDofFor(KM.DISPLACEMENT_Y),
                "DISPLACEMENT_Z": first_node.HasDofFor(KM.DISPLACEMENT_Z),
                "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE": first_node.HasDofFor(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                ),
            },
        }

        steps: list[dict[str, Any]] = []
        while analysis.KeepAdvancingSolutionLoop():
            analysis.time = analysis._AdvanceTime()
            time = analysis.time
            analysis.InitializeSolutionStep()
            solver = analysis._GetSolver()
            solver.Predict()
            converged = bool(solver.SolveSolutionStep())
            analysis.FinalizeSolutionStep()

            conditions = list(computing_contact.Conditions)
            snapshot = _condition_snapshot(computing_contact)
            step_record = {
                "step": int(structure.ProcessInfo[KM.STEP]),
                "time": float(time),
                "converged": converged,
                "nonlinear_iterations": int(structure.ProcessInfo[KM.NL_ITERATION_NUMBER]),
                "conditions": snapshot,
                "interface_conditions": _condition_snapshot(contact),
                "weighted_gap": _nodal_values(contact_nodes, CSMA.WEIGHTED_GAP),
                "lagrange_multiplier_contact_pressure": _nodal_values(
                    contact_nodes, CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                ),
                "augmented_normal_contact_pressure": _nodal_values(
                    contact_nodes, CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE
                ),
                "reactions": {
                    "prescribed_load_boundary": _reaction_sum(
                        model["Structure.IMPOSE_DISP_Auto1"]
                    ),
                    "fixed_lower_boundary": _reaction_sum(
                        model["Structure.DISPLACEMENT_Displacement_Auto2"]
                    ),
                    "whole_model": _reaction_sum(structure),
                },
            }
            normal_gap = getattr(CSMA, "NORMAL_GAP", None)
            if normal_gap is None:
                step_record["normal_gap"] = {"registered": False}
            else:
                step_record["normal_gap"] = {
                    "registered": True,
                    "nodal": _nodal_values(contact_nodes, normal_gap),
                    "condition": _condition_values(
                        conditions, normal_gap, structure.ProcessInfo
                    ),
                }
            steps.append(step_record)
            analysis.OutputSolutionStep()

        result = {
            "diagnostic_status": "PASS" if steps and all(s["converged"] for s in steps) else "FAIL",
            "kratos_version": KM.Kernel().Version(),
            "official_parameters": str(parameters_path),
            "official_upstream": (
                "https://github.com/KratosMultiphysics/Kratos/blob/v10.3.0/"
                + OFFICIAL_RELATIVE_PARAMETERS.as_posix()
            ),
            "diagnostic_changes": ["solver_settings.compute_reactions = true (in memory)"],
            "process_class": "ALMContactProcess",
            "mortar_type": "ALMContactFrictionless",
            "element_registered_name": "SmallDisplacementElement2D4N",
            "constitutive_law_registered_name": "LinearElasticPlaneStrain2DLaw",
            "contact_condition_registered_name": (
                "ALMFrictionlessMortarContactCondition2D2N"
            ),
            "model_parts": _model_part_names(structure),
            "initial_interface_conditions": initial_interface_conditions,
            "initial_contact_conditions": initial_computing_conditions,
            "variables_and_dofs": variable_and_dof_status,
            "steps": steps,
            "first_active_step": next(
                (s["step"] for s in steps if s["conditions"]["active_count"] > 0), None
            ),
            "final_converged": bool(steps and steps[-1]["converged"]),
        }
    finally:
        try:
            analysis.Finalize()
        finally:
            os.chdir(previous_directory)

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["diagnostic_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
