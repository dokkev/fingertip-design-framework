"""Isolated 3D ALM/MortarContactCondition3D3N contact harness."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from fem.solid3d import (
    SolidFEASettings,
    create_surface_condition,
    import_kratos,
    parameters_for_settings,
    properties_for_model_part,
    validate_contact_triangles,
)


def _child() -> int:
    KM, _, CLA, _ = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

    settings = SolidFEASettings(number_of_steps=1, indentation_mm=0.001, external_contact=True)
    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(model, parameters_for_settings(settings))
    model_part = model["Structure"]
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 3
    properties = properties_for_model_part(model_part, 1)
    properties[KM.YOUNG_MODULUS] = 1.0e3
    properties[KM.POISSON_RATIO] = 0.3
    properties[KM.DENSITY] = 1.0
    properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElastic3DLaw()

    body_nodes = {
        1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0), 4: (0.0, 1.0, 0.0),
        5: (0.0, 0.0, 1.0), 6: (1.0, 0.0, 1.0),
        7: (1.0, 1.0, 1.0), 8: (0.0, 1.0, 1.0),
    }
    # The slave face is the top of a valid cube and points outward in +z.
    # The master is the same planar patch at a strictly positive gap.
    master_nodes = {
        9: (0.0, 0.0, 1.0 + 1.0e-4), 10: (1.0, 0.0, 1.0 + 1.0e-4),
        11: (1.0, 1.0, 1.0 + 1.0e-4), 12: (0.0, 1.0, 1.0 + 1.0e-4),
    }
    for node_id, coordinates in {**body_nodes, **master_nodes}.items():
        model_part.CreateNewNode(node_id, *coordinates)
    for element_id, node_ids in enumerate(
        ((1, 2, 4, 5), (2, 3, 4, 7), (2, 4, 5, 7), (2, 5, 6, 7), (4, 5, 7, 8)),
        start=1,
    ):
        model_part.CreateNewElement(
            "TotalLagrangianMixedVolumetricStrainElement3D4N", element_id, list(node_ids), properties
        )
    slave_conditions = ((1001, (5, 6, 7)), (1002, (5, 7, 8)))
    master_conditions = ((1101, (9, 11, 10)), (1102, (9, 12, 11)))
    for condition_id, node_ids in (*slave_conditions, *master_conditions):
        create_surface_condition(model_part, properties, condition_id, node_ids)
    coordinates = {**body_nodes, **master_nodes}
    slave_report = validate_contact_triangles(
        slave_conditions, coordinates,
        expected_normal=(0.0, 0.0, 1.0),
        contact_assignment=("PadOuterArc", "IndenterContactArc"),
    )
    master_report = validate_contact_triangles(
        master_conditions, coordinates,
        expected_normal=(0.0, 0.0, -1.0),
        contact_assignment=("IndenterContactArc", "PadOuterArc"),
    )
    if not slave_report.passed or not master_report.passed:
        print(json.dumps({"status": "FAIL", "reason": "contact preflight failed", "slave": slave_report.__dict__, "master": master_report.__dict__}, default=str))
        return 2
    _add_part(model_part, "PadOuterArc", (5, 6, 7, 8), (1001, 1002))
    _add_part(model_part, "IndenterContactArc", (9, 10, 11, 12), (1101, 1102))
    analysis.Initialize()
    for node in model_part.Nodes:
        if node.Id > 4:
            continue
        for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)
    solver = analysis._GetSolver()
    analysis.time = solver.AdvanceInTime(analysis.time)
    analysis.InitializeSolutionStep()
    generated_parts = sorted(
        str(name)
        for name in model.GetModelPartNames()
        if ".ComputingContact.ComputingContactSub" in str(name)
    )
    if not generated_parts:
        analysis.FinalizeSolutionStep()
        analysis.Finalize()
        print(json.dumps({"status": "FAIL", "reason": "ALM generated no contact submodel part", "slave": slave_report.__dict__, "master": master_report.__dict__}))
        return 3
    converged = bool(solver.SolveSolutionStep())
    analysis.FinalizeSolutionStep()
    analysis.Finalize()
    print(json.dumps({
        "status": "PASS" if converged else "FAIL",
        "converged": converged,
        "generated_contact_parts": generated_parts,
        "slave": slave_report.__dict__,
        "master": master_report.__dict__,
        "configuration": {
            "element": "TotalLagrangianMixedVolumetricStrainElement3D4N",
            "constitutive_law": "HyperElastic3DLaw",
            "condition": "SurfaceCondition3D3N",
            "mortar_family": "MortarContactCondition3D3N",
            "process": "ALMContactProcess",
            "positive_initial_gap_mm": 1.0e-4,
            "master_slave": {"slave": "PadOuterArc", "master": "IndenterContactArc"},
        },
    }))
    return 0 if converged else 4


def _add_part(model_part: Any, name: str, node_ids: tuple[int, ...], condition_ids: tuple[int, ...]) -> None:
    part = model_part.CreateSubModelPart(name)
    part.AddNodes(list(node_ids))
    part.AddConditions(list(condition_ids))


def run_minimal_contact_harness(timeout_seconds: float = 120.0) -> dict[str, Any]:
    """Run the crash-prone child process and retain raw logs for provenance."""
    completed = subprocess.run(
        [sys.executable, "-m", "fem.contact3d_harness", "--child"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    payload: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    status = payload.get("status") if payload is not None else "FAIL"
    if completed.returncode != 0:
        status = "FAIL"
    return {
        "status": status,
        "return_code": completed.returncode,
        "classification": "adapter_contact_setup" if status != "PASS" else "minimal_harness_pass",
        "payload": payload,
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> None:
    if "--child" in sys.argv[1:]:
        raise SystemExit(_child())
    print(json.dumps(run_minimal_contact_harness(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
