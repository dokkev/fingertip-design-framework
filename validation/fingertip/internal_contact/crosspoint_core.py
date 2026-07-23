"""Phase 4I-G contact/Dirichlet crosspoint algebra and source audit.

The minimal patch is intentionally separate from the fingertip geometry.  It
uses the adopted mixed T3 solid, the installed Kratos ALM process, a flat rigid
master, and a deformable slave whose left or right contact endpoint has fully
prescribed displacement.  No correction is applied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from fem.kratos_adapter import import_kratos
from fem.kratos_settings import (
    ABSOLUTE_TOLERANCE,
    CONSTITUTIVE_LAW,
    MAXIMUM_NEWTON_ITERATIONS,
    MIXED_PAD_ELEMENT,
    MORTAR_TYPE,
    POISSON_RATIO,
    RELATIVE_TOLERANCE,
    THICKNESS_MM,
    YOUNG_MODULUS_MPA,
)
from validation.fingertip.internal_contact.sparse import analyze_sparse_system


CrosspointSide = Literal["left", "right"]

PATCH_WIDTH_MM = 2.0
PATCH_HEIGHT_MM = 1.0
MASTER_THICKNESS_MM = 0.1
PRESCRIBED_PENETRATION_MM = 1.0e-4
LM_ROW_ABSOLUTE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class CrosspointRuleInput:
    """Topology/fixity data used by a prospective multiplier restriction."""

    node_id: int
    displacement_x_fixed: bool
    displacement_y_fixed: bool
    incident_active_contact_condition_count: int


def contact_coupled_free_primal_dof_count(
    displacement_x_fixed: bool,
    displacement_y_fixed: bool,
) -> int:
    """Count free in-plane trace unknowns without inspecting coordinates/IDs."""
    return int(not displacement_x_fixed) + int(not displacement_y_fixed)


def fully_prescribed_contact_crosspoint(
    record: CrosspointRuleInput,
) -> bool:
    """Identify the G2 topology rule without any node-ID or side convention."""
    return (
        record.incident_active_contact_condition_count > 0
        and contact_coupled_free_primal_dof_count(
            record.displacement_x_fixed,
            record.displacement_y_fixed,
        )
        == 0
    )


def _project_parameters(number_of_steps: int = 1) -> Any:
    KM, _, _, _ = import_kratos()
    return KM.Parameters(
        f"""{{
            "problem_data": {{
                "problem_name": "phase4ig_crosspoint_patch",
                "parallel_type": "OpenMP",
                "start_time": 0.0,
                "end_time": {float(number_of_steps)},
                "echo_level": 0
            }},
            "solver_settings": {{
                "model_part_name": "Structure",
                "domain_size": 2,
                "solver_type": "Static",
                "echo_level": 0,
                "analysis_type": "non_linear",
                "model_import_settings": {{"input_type": "use_input_model_part"}},
                "material_import_settings": {{"materials_filename": ""}},
                "time_stepping": {{"time_step": 1.0}},
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
                    "kratos_module":
                        "KratosMultiphysics.ContactStructuralMechanicsApplication",
                    "process_name": "ALMContactProcess",
                    "Parameters": {{
                        "model_part_name": "Structure",
                        "assume_master_slave": {{"0": ["PatchTop"]}},
                        "contact_model_part": {{
                            "0": ["PatchTop", "FlatMaster"]
                        }},
                        "contact_type": "Frictionless"
                    }}
                }}]
            }}
        }}"""
    )


def _create_patch_mesh(model_part: Any, divisions: int) -> dict[str, Any]:
    KM, _, CLA, _ = import_kratos()
    if divisions < 2:
        raise ValueError("crosspoint patch needs at least two divisions")
    geometry = KM.Quadrilateral2D4(
        KM.Node(1, 0.0, 0.0, 0.0),
        KM.Node(2, 0.0, PATCH_HEIGHT_MM, 0.0),
        KM.Node(3, PATCH_WIDTH_MM, PATCH_HEIGHT_MM, 0.0),
        KM.Node(4, PATCH_WIDTH_MM, 0.0, 0.0),
    )
    generator_settings = KM.Parameters(
        """{
            "number_of_divisions": 2,
            "create_skin_sub_model_part": false,
            "element_name":
                "TotalLagrangianMixedVolumetricStrainElement2D3N"
        }"""
    )
    generator_settings["number_of_divisions"].SetInt(divisions)
    KM.StructuredMeshGeneratorProcess(
        geometry, model_part, generator_settings
    ).Execute()

    properties = model_part.Properties[0]
    properties[KM.YOUNG_MODULUS] = YOUNG_MODULUS_MPA
    properties[KM.POISSON_RATIO] = POISSON_RATIO
    properties[KM.THICKNESS] = THICKNESS_MM
    properties[KM.DENSITY] = 1.0
    properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElasticPlaneStrain2DLaw()

    tolerance = 1.0e-12
    block_node_ids = sorted(node.Id for node in model_part.Nodes)
    block_element_ids = sorted(element.Id for element in model_part.Elements)
    top_node_ids = [
        node.Id
        for node in sorted(model_part.Nodes, key=lambda item: item.X0)
        if abs(float(node.Y0) - PATCH_HEIGHT_MM) <= tolerance
    ]
    bottom_node_ids = [
        node.Id
        for node in model_part.Nodes
        if abs(float(node.Y0)) <= tolerance
    ]
    solid = model_part.CreateSubModelPart("PatchSolid")
    solid.AddNodes(block_node_ids)
    solid.AddElements(block_element_ids)
    top = model_part.CreateSubModelPart("PatchTop")
    top.AddNodes(top_node_ids)

    first_master_node_id = max(block_node_ids) + 1
    lower_master_ids: list[int] = []
    upper_master_ids: list[int] = []
    for index, block_node_id in enumerate(top_node_ids):
        x = float(model_part.Nodes[block_node_id].X0)
        lower_id = first_master_node_id + index
        upper_id = first_master_node_id + len(top_node_ids) + index
        model_part.CreateNewNode(lower_id, x, PATCH_HEIGHT_MM, 0.0)
        model_part.CreateNewNode(
            upper_id,
            x,
            PATCH_HEIGHT_MM + MASTER_THICKNESS_MM,
            0.0,
        )
        lower_master_ids.append(lower_id)
        upper_master_ids.append(upper_id)

    next_element_id = max(block_element_ids) + 1
    master_element_ids: list[int] = []
    for index in range(len(lower_master_ids) - 1):
        lower_left = lower_master_ids[index]
        lower_right = lower_master_ids[index + 1]
        upper_left = upper_master_ids[index]
        upper_right = upper_master_ids[index + 1]
        model_part.CreateNewElement(
            "TotalLagrangianElement2D3N",
            next_element_id,
            [lower_left, lower_right, upper_right],
            properties,
        )
        model_part.CreateNewElement(
            "TotalLagrangianElement2D3N",
            next_element_id + 1,
            [lower_left, upper_right, upper_left],
            properties,
        )
        master_element_ids.extend([next_element_id, next_element_id + 1])
        next_element_id += 2

    next_condition_id = 1
    master_condition_ids: list[int] = []
    for index in range(len(lower_master_ids) - 1):
        model_part.CreateNewCondition(
            "LineCondition2D2N",
            next_condition_id,
            [lower_master_ids[index], lower_master_ids[index + 1]],
            properties,
        )
        master_condition_ids.append(next_condition_id)
        next_condition_id += 1

    flat_master = model_part.CreateSubModelPart("FlatMaster")
    flat_master.AddNodes(lower_master_ids)
    flat_master.AddConditions(master_condition_ids)
    master_carrier = model_part.CreateSubModelPart("MasterCarrier")
    master_carrier.AddNodes(lower_master_ids + upper_master_ids)
    master_carrier.AddElements(master_element_ids)
    return {
        "block_node_ids": block_node_ids,
        "block_element_ids": block_element_ids,
        "top_node_ids": top_node_ids,
        "bottom_node_ids": bottom_node_ids,
        "master_node_ids": lower_master_ids + upper_master_ids,
        "lower_master_node_ids": lower_master_ids,
        "master_element_ids": master_element_ids,
    }


def _fix_patch(
    model_part: Any,
    mesh: Mapping[str, Sequence[int]],
    side: CrosspointSide,
) -> int:
    KM, _, _, _ = import_kratos()
    for node in model_part.Nodes:
        node.SetSolutionStepValue(KM.VOLUMETRIC_STRAIN, 0.0)
    for node_id in mesh["bottom_node_ids"]:
        node = model_part.Nodes[node_id]
        for variable in (
            KM.DISPLACEMENT_X,
            KM.DISPLACEMENT_Y,
            KM.DISPLACEMENT_Z,
        ):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)
    for node_id in mesh["master_node_ids"]:
        node = model_part.Nodes[node_id]
        for variable in (
            KM.DISPLACEMENT_X,
            KM.DISPLACEMENT_Y,
            KM.DISPLACEMENT_Z,
        ):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)
    endpoint_id = (
        mesh["top_node_ids"][0]
        if side == "left"
        else mesh["top_node_ids"][-1]
    )
    endpoint = model_part.Nodes[endpoint_id]
    for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y):
        endpoint.Fix(variable)
        endpoint.SetSolutionStepValue(variable, 0.0)
    return int(endpoint_id)


def _move_master(
    model_part: Any,
    node_ids: Sequence[int],
) -> None:
    KM, _, _, _ = import_kratos()
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        node.SetSolutionStepValue(
            KM.DISPLACEMENT_Y, -PRESCRIBED_PENETRATION_MM
        )
        node.Y = node.Y0 - PRESCRIBED_PENETRATION_MM


def _condition_rows(
    computing_contact: Any,
    endpoint_id: int,
    process_info: Any,
) -> list[dict[str, Any]]:
    KM, CSMA, _, _ = import_kratos()
    rows: list[dict[str, Any]] = []
    for condition in computing_contact.Conditions:
        geometry = condition.GetGeometry()
        slave = geometry.GetGeometryPart(0)
        if endpoint_id not in [node.Id for node in slave]:
            continue
        dofs = list(condition.GetDofList(process_info))
        equation_ids = [
            int(value) for value in condition.EquationIdVector(process_info)
        ]
        lm_rows = [
            index
            for index, dof in enumerate(dofs)
            if int(dof.Id()) == endpoint_id
            and dof.GetVariable().Name()
            == CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
        ]
        lhs = KM.Matrix()
        rhs = KM.Vector()
        condition.CalculateLocalSystem(lhs, rhs, process_info)
        for row_index in lm_rows:
            values = [
                float(lhs[row_index, column])
                for column in range(lhs.Size2())
            ]
            free_indices = [
                index for index, dof in enumerate(dofs) if not dof.IsFixed()
            ]
            fixed_indices = [
                index for index, dof in enumerate(dofs) if dof.IsFixed()
            ]
            diagonal = values[row_index]
            rows.append(
                {
                    "condition_id": int(condition.Id),
                    "condition_active": bool(condition.Is(KM.ACTIVE)),
                    "slave_node_ids": [node.Id for node in slave],
                    "master_node_ids": [
                        node.Id for node in geometry.GetGeometryPart(1)
                    ],
                    "endpoint_node_active": bool(
                        model_part_node(slave, endpoint_id).Is(KM.ACTIVE)
                    ),
                    "row_norm_all_columns": float(np.linalg.norm(values)),
                    "row_norm_free_columns": float(
                        np.linalg.norm(
                            [values[index] for index in free_indices]
                        )
                    ),
                    "row_norm_fixed_columns": float(
                        np.linalg.norm(
                            [values[index] for index in fixed_indices]
                        )
                    ),
                    "lm_diagonal": diagonal,
                    "local_dofs": [
                        {
                            "node_id": int(dof.Id()),
                            "variable": dof.GetVariable().Name(),
                            "equation_id": equation_ids[index],
                            "fixed": bool(dof.IsFixed()),
                        }
                        for index, dof in enumerate(dofs)
                    ],
                }
            )
    return rows


def model_part_node(geometry: Any, node_id: int) -> Any:
    """Return an exact node from a small geometry."""
    return next(node for node in geometry if int(node.Id) == node_id)


def run_crosspoint_patch(
    side: CrosspointSide,
    divisions: int,
) -> dict[str, Any]:
    """Build and assemble one fresh mirrored crosspoint patch."""
    KM, CSMA, _, _ = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
        StructuralMechanicsAnalysis,
    )

    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(model, _project_parameters())
    model_part = model["Structure"]
    mesh = _create_patch_mesh(model_part, divisions)
    initialized = False
    step_initialized = False
    try:
        analysis.Initialize()
        initialized = True
        endpoint_id = _fix_patch(model_part, mesh, side)
        endpoint = model_part.Nodes[endpoint_id]
        solver = analysis._GetSolver()
        analysis.time = solver.AdvanceInTime(analysis.time)
        _move_master(model_part, mesh["master_node_ids"])
        analysis.ApplyBoundaryConditions()
        computing_contact = model[
            "Structure.ComputingContact.ComputingContactSub0"
        ]
        incident_conditions = [
            condition
            for condition in computing_contact.Conditions
            if endpoint_id
            in [
                node.Id
                for node in condition.GetGeometry().GetGeometryPart(0)
            ]
        ]
        analysis.ChangeMaterialProperties()
        solver.InitializeSolutionStep()
        step_initialized = True
        solver.Predict()

        strategy = solver._GetSolutionStrategy()
        builder = solver._GetBuilderAndSolver()
        scheme = solver._GetScheme()
        matrix = strategy.GetSystemMatrix()
        rhs = strategy.GetSystemVector()
        increment = strategy.GetSolutionVector()
        builder.Build(
            scheme, solver.GetComputingModelPart(), matrix, rhs
        )
        import KratosMultiphysics.scipy_conversion_tools as conversion

        pre_dirichlet_csr = conversion.to_csr(matrix).copy()
        builder.ApplyDirichletConditions(
            scheme,
            solver.GetComputingModelPart(),
            matrix,
            increment,
            rhs,
        )
        post_dirichlet_csr = conversion.to_csr(matrix)
        dof_set = builder.GetDofSet()
        equation_map: dict[int, dict[str, Any]] = {}
        dof_rows: list[dict[str, Any]] = []
        for dof in dof_set:
            record = {
                "node_id": int(dof.Id()),
                "variable": dof.GetVariable().Name(),
                "equation_id": int(dof.EquationId),
                "fixed": bool(dof.IsFixed()),
            }
            equation_map[int(dof.EquationId)] = record
            dof_rows.append(record)
        matrix_diagnostic = analyze_sparse_system(
            post_dirichlet_csr,
            np.asarray([float(rhs[index]) for index in range(len(rhs))]),
            equation_map,
        )
        lm_dof = endpoint.GetDof(
            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
        )
        lm_equation_id = int(lm_dof.EquationId)
        pre_lm_row = pre_dirichlet_csr.getrow(lm_equation_id)
        post_lm_row = post_dirichlet_csr.getrow(lm_equation_id)
        fixed_equations = {
            int(dof.EquationId) for dof in dof_set if dof.IsFixed()
        }
        pre_free_values = [
            value
            for column, value in zip(pre_lm_row.indices, pre_lm_row.data)
            if int(column) not in fixed_equations
        ]
        pre_fixed_values = [
            value
            for column, value in zip(pre_lm_row.indices, pre_lm_row.data)
            if int(column) in fixed_equations
        ]
        post_free_values = [
            value
            for column, value in zip(post_lm_row.indices, post_lm_row.data)
            if int(column) not in fixed_equations
        ]
        local_rows = _condition_rows(
            computing_contact, endpoint_id, model_part.ProcessInfo
        )
        adjacent_top_ids = [
            node_id
            for node_id in mesh["top_node_ids"]
            if node_id != endpoint_id
        ]
        adjacent_id = min(
            adjacent_top_ids,
            key=lambda node_id: abs(
                float(model_part.Nodes[node_id].X0)
                - float(endpoint.X0)
            ),
        )
        adjacent = model_part.Nodes[adjacent_id]
        rule_input = CrosspointRuleInput(
            node_id=endpoint_id,
            displacement_x_fixed=endpoint.GetDof(
                KM.DISPLACEMENT_X
            ).IsFixed(),
            displacement_y_fixed=endpoint.GetDof(
                KM.DISPLACEMENT_Y
            ).IsFixed(),
            incident_active_contact_condition_count=sum(
                condition.Is(KM.ACTIVE)
                for condition in incident_conditions
            ),
        )
        return {
            "side": side,
            "divisions": divisions,
            "element": MIXED_PAD_ELEMENT,
            "constitutive_law": CONSTITUTIVE_LAW,
            "poisson_ratio": POISSON_RATIO,
            "mortar_type": MORTAR_TYPE,
            "endpoint_node_id": endpoint_id,
            "endpoint_coordinate_mm": [
                float(endpoint.X0),
                float(endpoint.Y0),
            ],
            "adjacent_interior_node_id": adjacent_id,
            "adjacent_interior_coordinate_mm": [
                float(adjacent.X0),
                float(adjacent.Y0),
            ],
            "endpoint_displacement_fixity": {
                "x": rule_input.displacement_x_fixed,
                "y": rule_input.displacement_y_fixed,
            },
            "endpoint_free_primal_dof_count": (
                contact_coupled_free_primal_dof_count(
                    rule_input.displacement_x_fixed,
                    rule_input.displacement_y_fixed,
                )
            ),
            "adjacent_displacement_fixity": {
                "x": adjacent.GetDof(KM.DISPLACEMENT_X).IsFixed(),
                "y": adjacent.GetDof(KM.DISPLACEMENT_Y).IsFixed(),
            },
            "fully_prescribed_crosspoint_rule": (
                fully_prescribed_contact_crosspoint(rule_input)
            ),
            "endpoint_node_flags": {
                "ACTIVE": bool(endpoint.Is(KM.ACTIVE)),
                "SLAVE": bool(endpoint.Is(KM.SLAVE)),
                "MASTER": bool(endpoint.Is(KM.MASTER)),
            },
            "incident_generated_condition_count": len(incident_conditions),
            "incident_active_condition_count": sum(
                condition.Is(KM.ACTIVE)
                for condition in incident_conditions
            ),
            "contact_lm_basis_support": local_rows,
            "dof_set": dof_rows,
            "lm_equation_id": lm_equation_id,
            "lm_fixed": bool(lm_dof.IsFixed()),
            "pre_dirichlet_lm_row": {
                "columns": [int(value) for value in pre_lm_row.indices],
                "values": [float(value) for value in pre_lm_row.data],
            },
            "pre_dirichlet_lm_row_norm": float(
                np.linalg.norm(pre_lm_row.data)
            ),
            "pre_dirichlet_lm_free_column_norm": float(
                np.linalg.norm(pre_free_values)
            ),
            "pre_dirichlet_lm_fixed_column_norm": float(
                np.linalg.norm(pre_fixed_values)
            ),
            "post_dirichlet_lm_row": {
                "columns": [int(value) for value in post_lm_row.indices],
                "values": [float(value) for value in post_lm_row.data],
            },
            "post_dirichlet_lm_row_norm": float(
                np.linalg.norm(post_lm_row.data)
            ),
            "post_dirichlet_lm_free_column_norm": float(
                np.linalg.norm(post_free_values)
            ),
            "post_dirichlet_lm_diagonal": float(
                post_dirichlet_csr[lm_equation_id, lm_equation_id]
            ),
            "lm_rhs_entry": float(rhs[lm_equation_id]),
            "near_zero_lm_row": float(np.linalg.norm(post_lm_row.data))
            < LM_ROW_ABSOLUTE_TOLERANCE,
            "matrix_diagnostics": matrix_diagnostic,
            "normal_gap": (
                float(endpoint.GetValue(CSMA.NORMAL_GAP))
                if endpoint.Has(CSMA.NORMAL_GAP)
                else None
            ),
            "weighted_gap": float(
                endpoint.GetSolutionStepValue(CSMA.WEIGHTED_GAP)
            ),
            "endpoint_normal": [
                float(endpoint.GetSolutionStepValue(KM.NORMAL)[component])
                for component in range(2)
            ],
            "lagrange_multiplier_contact_pressure": float(
                endpoint.GetSolutionStepValue(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                )
            ),
            "prescribed_penetration_mm": PRESCRIBED_PENETRATION_MM,
        }
    finally:
        if step_initialized:
            try:
                analysis.FinalizeSolutionStep()
            except Exception:
                pass
        if initialized:
            analysis.Finalize()


def candidate_assessment() -> list[dict[str, Any]]:
    """Return the bounded G1/G2 gate without applying prohibited mutations."""
    return [
        {
            "candidate": "G1",
            "name": "official Kratos crosspoint treatment",
            "status": "UNAVAILABLE",
            "implemented": False,
            "reason": (
                "No v10.3 process/setting omits or condenses a slave pressure "
                "LM whose coupled displacement trace is fully prescribed."
            ),
        },
        {
            "candidate": "G2",
            "name": "application-level multiplier trace restriction",
            "status": "INCONCLUSIVE",
            "implemented": False,
            "topology_rule_identified": True,
            "reason": (
                "The topology/fixity rule is ID-independent, but the installed "
                "condition unconditionally returns one pressure LM per slave "
                "node. Python exposes no pre-DOF basis restriction or complete "
                "condition condensation hook; forcing LM fixity or changing "
                "ACTIVE after construction is prohibited."
            ),
        },
    ]


def source_audit() -> dict[str, Any]:
    """Describe the exact installed Kratos multiplier-space contract."""
    commit = "14ee273e97af403622699e797ea5fa356b1a7e60"
    root = (
        "https://github.com/KratosMultiphysics/Kratos/blob/"
        f"{commit}/"
    )
    return {
        "kernel": "10.3.0-14ee273e",
        "commit": commit,
        "official_crosspoint_treatment_found": False,
        "lm_dof_addition": {
            "file": (
                "applications/ContactStructuralMechanicsApplication/"
                "python_scripts/auxiliary_methods_solvers.py"
            ),
            "function": "AuxiliaryAddDofs",
            "behavior": (
                "VariableUtils.AddDof adds scalar contact-pressure LM to the "
                "entire main ModelPart for ALMContactFrictionless"
            ),
            "url": root
            + (
                "applications/ContactStructuralMechanicsApplication/"
                "python_scripts/auxiliary_methods_solvers.py"
            ),
        },
        "lm_basis_and_condition_dofs": {
            "file": (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_conditions/"
                "ALM_frictionless_mortar_contact_condition.cpp"
            ),
            "class": (
                "AugmentedLagrangianMethodFrictionlessMortarContactCondition"
            ),
            "methods": ["EquationIdVector", "GetDofList"],
            "behavior": (
                "Every slave geometry node contributes one "
                "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE equation and DOF; "
                "displacement fixity is not consulted."
            ),
            "url": root
            + (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_conditions/"
                "ALM_frictionless_mortar_contact_condition.cpp"
            ),
        },
        "active_inactive_algebra": {
            "file": (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_conditions/"
                "ALM_frictionless_mortar_contact_condition.cpp"
            ),
            "behavior": (
                "Inactive slave nodes receive scale_factor^2/penalty on the "
                "LM diagonal; active slave nodes use mortar coupling and no "
                "equivalent diagonal stabilization."
            ),
        },
        "dirichlet_elimination": {
            "file": (
                "kratos/solving_strategies/builder_and_solvers/"
                "residualbased_block_builder_and_solver.h"
            ),
            "method": "ApplyDirichletConditions",
            "behavior": (
                "Fixed displacement rows/columns are eliminated after entity "
                "assembly. Free contact LM rows remain in the global system."
            ),
            "url": root
            + (
                "kratos/solving_strategies/builder_and_solvers/"
                "residualbased_block_builder_and_solver.h"
            ),
        },
        "contact_builder_isolated_node_handling": {
            "file": (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_strategies/custom_builder_and_solvers/"
                "contact_residualbased_block_builder_and_solver.h"
            ),
            "methods": ["FixIsolatedNodes", "FreeIsolatedNodes"],
            "behavior": (
                "Temporarily fixes LM only for nodes marked ISOLATED by all "
                "incident generated conditions. It does not inspect whether "
                "the contact-coupled displacement trace is Dirichlet."
            ),
            "url": root
            + (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_strategies/custom_builder_and_solvers/"
                "contact_residualbased_block_builder_and_solver.h"
            ),
        },
        "supported_omission_or_condensation": {
            "lm_dof_omission": False,
            "static_condensation_for_ALM_contact_pressure": False,
            "boundary_trace_restriction_setting": False,
            "python_pre_dof_basis_hook": False,
        },
        "literature_comment": {
            "file": (
                "applications/ContactStructuralMechanicsApplication/"
                "symbolic_generation/ALM_frictionless_mortar_condition/"
                "alm_frictionless_mortar_contact_condition.tex"
            ),
            "behavior": (
                "The bundled derivation states that the discrete multiplier "
                "space choice is decisive for stability, but supplies no "
                "contact/Dirichlet endpoint restriction rule."
            ),
        },
    }
