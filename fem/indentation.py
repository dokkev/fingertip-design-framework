"""Phase 4I central indentation using the Phase 4M mesh and Kratos stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from mesh.indenter import (
    IndenterFixture,
    IndenterMesh,
    IndenterSettings,
    build_indenter_fixture,
    generate_indenter_mesh,
)
from fem.contact import (
    contact_group_step_metrics,
    failed_contact_group_diagnostics,
    runtime_contact_contract,
    create_continuous_u_submodel_parts,
)
from fem.results import (
    curve_acceptance,
    finite_field_failures,
    extract_nodal_fields,
    rigid_domain_validation,
    compressive_indenter_reaction,
    contact_width_metrics,
    extract_outer_arc_profile,
    failure_statistics,
    pad_strain_det_f_statistics,
    relative_force_equilibrium_error,
    scalar_statistics,
    signed_geometric_gap_statistics,
    unique_projected_reaction,
    unstructured_volumetric_oscillation,
)
from mesh.fingertip import generate_fingertip_mesh
from fem.kratos_adapter import (
    IndenterKratosTopology,
    KratosAdapterError,
    KratosTopology,
    import_kratos,
    apply_indentation_constraints,
    populate_kratos_model_part,
    populate_indenter_model_part,
    set_indenter_travel,
)
from fem.kratos_settings import (
    CARRIER_ELEMENT,
    CONSTITUTIVE_LAW,
    MAXIMUM_NEWTON_ITERATIONS,
    MIXED_PAD_ELEMENT,
    MORTAR_TYPE,
    POISSON_RATIO,
    RELATIVE_TOLERANCE,
    ABSOLUTE_TOLERANCE,
    THICKNESS_MM,
    YOUNG_MODULUS_MPA,
    build_indentation_project_parameters_json,
    indentation_contact_groups,
    validate_internal_contact_configuration,
)
from mesh.types import BoundaryEdge, FingertipMesh, MeshLevel, mesh_settings_for_level
from model.fingertip_model import FingertipModel


class InvalidIndentationSettings(ValueError):
    """Raised when a nonlinear loading request is internally inconsistent."""


@dataclass(frozen=True)
class IndentationSettings:
    """Common displacement-controlled Phase 4I loading settings."""

    indentation_mm: float
    number_of_steps: int
    force_floor_n: float = 1.0e-8
    numerical_force_tolerance_n: float = 1.0e-8
    profile_displacement_floor_mm: float = 1.0e-5

    def __post_init__(self) -> None:
        values = {
            "indentation_mm": self.indentation_mm,
            "force_floor_n": self.force_floor_n,
            "numerical_force_tolerance_n": self.numerical_force_tolerance_n,
            "profile_displacement_floor_mm": self.profile_displacement_floor_mm,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise InvalidIndentationSettings("indentation settings must be finite")
        if self.indentation_mm <= 0.0:
            raise InvalidIndentationSettings("indentation_mm must be positive")
        if (
            not isinstance(self.number_of_steps, int)
            or isinstance(self.number_of_steps, bool)
            or self.number_of_steps <= 0
        ):
            raise InvalidIndentationSettings("number_of_steps must be a positive integer")
        for name in (
            "force_floor_n",
            "numerical_force_tolerance_n",
            "profile_displacement_floor_mm",
        ):
            if values[name] <= 0.0:
                raise InvalidIndentationSettings(f"{name} must be positive")

    @property
    def capture_depths_mm(self) -> tuple[float, ...]:
        requested = [depth for depth in (0.5, 1.0, 1.5) if depth <= self.indentation_mm + 1.0e-12]
        if not requested or abs(requested[-1] - self.indentation_mm) > 1.0e-12:
            requested.append(self.indentation_mm)
        return tuple(requested)

    def capture_step(self, depth_mm: float) -> int:
        exact = depth_mm / self.indentation_mm * self.number_of_steps
        step = int(round(exact))
        if step < 1 or step > self.number_of_steps or abs(step - exact) > 1.0e-10:
            raise InvalidIndentationSettings(
                f"depth {depth_mm:g} mm is not an exact solution step for "
                f"{self.indentation_mm:g} mm / {self.number_of_steps} steps"
            )
        return step



@dataclass
class IndentationArtifacts:
    """In-memory fields needed for CSV and PNG output, excluded from JSON."""

    mesh: FingertipMesh
    fixture: IndenterFixture
    indenter_mesh: IndenterMesh
    indenter_topology: IndenterKratosTopology
    snapshots: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ConvergedIndentationStep:
    """Read-only view exposed immediately after one converged solution step."""

    model: Any
    model_part: Any
    fingertip_model: FingertipModel
    mesh: FingertipMesh
    fixture: IndenterFixture
    base_topology: KratosTopology
    indenter_topology: IndenterKratosTopology
    settings: IndentationSettings
    displacements: Mapping[int, Sequence[float]]
    reactions: Mapping[int, Sequence[float]]
    result_point: Mapping[str, Any]
    elapsed_case_seconds: float


ConvergedStepObserver = Callable[
    [ConvergedIndentationStep], Mapping[str, Any] | None
]



def inspect_indentation_runtime_contract(
    fingertip_model: FingertipModel,
    mesh_level: MeshLevel,
    settings: IndentationSettings,
    indenter_settings: IndenterSettings | None = None,
    internal_contact_configuration: str = "three_pairs",
) -> dict[str, Any]:
    """Initialize and inspect Phase 4I without entering a nonlinear step."""
    KM, CSMA, _, _ = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
        StructuralMechanicsAnalysis,
    )

    analysis: Any | None = None
    initialized = False
    start = time.perf_counter()
    try:
        configuration = validate_internal_contact_configuration(
            internal_contact_configuration
        )
        contact_groups = indentation_contact_groups(configuration)
        mesh = generate_fingertip_mesh(
            fingertip_model, mesh_settings_for_level(mesh_level)
        )
        fixture = build_indenter_fixture(fingertip_model, indenter_settings)
        indenter_mesh = generate_indenter_mesh(
            fixture, mesh.settings.contact_boundary_target_size_mm
        )
        model = KM.Model()
        analysis = StructuralMechanicsAnalysis(
            model,
            KM.Parameters(
                build_indentation_project_parameters_json(
                    settings.number_of_steps, configuration
                )
            ),
        )
        model_part = model["Structure"]
        base_topology = populate_kratos_model_part(model_part, mesh)
        indenter_topology = populate_indenter_model_part(
            model_part, indenter_mesh, base_topology
        )
        aggregate_contract = (
            create_continuous_u_submodel_parts(model_part)
            if configuration == "continuous_u"
            else None
        )
        analysis.Initialize()
        initialized = True
        apply_indentation_constraints(
            model_part, base_topology, indenter_topology
        )
        runtime = runtime_contact_contract(
            model, model_part, contact_groups
        )
        model_part_names = sorted(str(name) for name in model.GetModelPartNames())
        indexed_contact_paths = [
            name
            for name in model_part_names
            if ".Contact.ContactSub" in name
            or ".ComputingContact.ComputingContactSub" in name
        ]
        internal_surface_names = (
            "PadCutoutLeft",
            "PadCutoutRight",
            "PadCutoutBottom",
            "StemLeft",
            "StemRight",
            "StemBottom",
        )
        internal_surface_node_ids = {
            node.Id
            for name in internal_surface_names
            for node in model_part.GetSubModelPart(name).Nodes
        }
        global_lm_node_ids = {
            node.Id
            for node in model_part.Nodes
            if node.HasDofFor(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
        }
        return {
            "status": "PASS" if runtime["all_group_contracts_pass"] else "FAIL",
            "mesh_level": mesh_level,
            "kratos_version": KM.Kernel.Version(),
            "mesh_validation_pass": mesh.validation.passed,
            "mesh_counts": {
                "nodes": mesh.quality.node_count,
                "elements": mesh.quality.t3_element_count,
                "pad_outer_arc_edges": len(mesh.boundary_edges["pad_outer_arc"]),
                "indenter_nodes": len(indenter_topology.node_ids),
                "indenter_elements": len(indenter_topology.element_ids),
                "indenter_contact_edges": len(indenter_topology.contact_edges),
            },
            "pad_indenter_node_ids_disjoint": set(base_topology.pad_node_ids).isdisjoint(
                indenter_topology.node_ids
            ),
            "fixture": fixture.to_dict(),
            "internal_contact_configuration": configuration,
            "contact_groups": [list(group) for group in contact_groups],
            "continuous_u_aggregate_contract": aggregate_contract,
            "runtime_contact_contract": runtime,
            "contact_process_count": len(contact_groups),
            "indexed_contact_model_part_paths": indexed_contact_paths,
            "internal_contact_registration": {
                "registered_group_names": [
                    group_name
                    for group_name, _, _ in contact_groups
                    if group_name.startswith("internal_")
                ],
                "internal_contact_submodel_parts_present": [
                    name
                    for name in indexed_contact_paths
                    if not name.endswith("Sub0")
                ],
                "internal_source_semantic_parts_retained": list(
                    internal_surface_names
                ),
                "internal_source_nodes_with_root_level_lm_dof": len(
                    internal_surface_node_ids.intersection(global_lm_node_ids)
                ),
                "root_level_lm_dof_explanation": (
                    "AuxiliaryAddDofs adds the ALM pressure DOF to every root "
                    "node even for one external pair. These dormant DOFs are "
                    "not evidence of an internal contact process or assembly."
                ),
            },
            "internal_contact_lm_boundary_treatment": "Kratos process default; no manual LM pressure constraints",
            "strategy_check": int(
                analysis._GetSolver()._GetSolutionStrategy().Check()
            ),
            "wall_clock_seconds": time.perf_counter() - start,
        }
    finally:
        if initialized and analysis is not None:
            analysis.Finalize()


def run_indentation_case(
    fingertip_model: FingertipModel,
    mesh_level: MeshLevel,
    settings: IndentationSettings,
    indenter_settings: IndenterSettings | None = None,
    internal_contact_configuration: str = "three_pairs",
    mesh_override: FingertipMesh | None = None,
    fixture_override: IndenterFixture | None = None,
    converged_step_observer: ConvergedStepObserver | None = None,
) -> tuple[dict[str, Any], IndentationArtifacts | None]:
    """Run one fresh-model Phase 4I case and retain failure diagnostics."""
    KM, CSMA, _, _ = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
        StructuralMechanicsAnalysis,
    )

    result: dict[str, Any] = {
        "phase": "4I",
        "mesh_level": mesh_level,
        "status": "FAIL",
        "solve_status": "FAIL",
        "history": [],
    }
    artifacts: IndentationArtifacts | None = None
    analysis: Any | None = None
    initialized = False
    start = time.perf_counter()
    solve_time = 0.0
    try:
        configuration = validate_internal_contact_configuration(
            internal_contact_configuration
        )
        contact_group_definitions = indentation_contact_groups(configuration)
        mesh = (
            mesh_override
            if mesh_override is not None
            else generate_fingertip_mesh(
                fingertip_model, mesh_settings_for_level(mesh_level)
            )
        )
        if mesh.settings.level != mesh_level:
            raise InvalidIndentationSettings(
                "mesh_override level must match mesh_level"
            )
        if fixture_override is not None and indenter_settings is not None:
            raise InvalidIndentationSettings(
                "fixture_override and indenter_settings are mutually exclusive"
            )
        fixture = (
            fixture_override
            if fixture_override is not None
            else build_indenter_fixture(fingertip_model, indenter_settings)
        )
        indenter_mesh = generate_indenter_mesh(
            fixture, mesh.settings.contact_boundary_target_size_mm
        )
        model = KM.Model()
        parameters = KM.Parameters(
            build_indentation_project_parameters_json(
                settings.number_of_steps, configuration
            )
        )
        analysis = StructuralMechanicsAnalysis(model, parameters)
        model_part = model["Structure"]
        base_topology = populate_kratos_model_part(model_part, mesh)
        indenter_topology = populate_indenter_model_part(
            model_part, indenter_mesh, base_topology
        )
        aggregate_contract = (
            create_continuous_u_submodel_parts(model_part)
            if configuration == "continuous_u"
            else None
        )
        analysis.Initialize()
        initialized = True
        apply_indentation_constraints(
            model_part, base_topology, indenter_topology
        )
        runtime_contact = runtime_contact_contract(
            model, model_part, contact_group_definitions
        )
        if not runtime_contact["all_group_contracts_pass"]:
            raise KratosAdapterError("one or more indexed contact group contracts failed")
        strategy_check = int(analysis._GetSolver()._GetSolutionStrategy().Check())
        if strategy_check != 0:
            raise KratosAdapterError(f"solution strategy Check returned {strategy_check}")

        support_node_ids = set(base_topology.carrier_node_ids)
        for name in ("PadBondLeft", "PadBondRight"):
            support_node_ids.update(
                node.Id for node in model_part.GetSubModelPart(name).Nodes
            )
        all_node_ids = tuple(node.Id for node in model_part.Nodes)
        capture_steps = {
            settings.capture_step(depth): depth for depth in settings.capture_depths_mm
        }
        snapshots: dict[str, dict[str, Any]] = {}
        result.update(
            {
                "kratos_version": KM.Kernel.Version(),
                "configuration": {
                    "mesh_level": mesh_level,
                    "internal_contact_configuration": configuration,
                    "contact_groups": [
                        list(group) for group in contact_group_definitions
                    ],
                    "mesh_settings": asdict(mesh.settings),
                    "fingertip_parameters": asdict(fingertip_model.parameters),
                    "indenter": fixture.to_dict(),
                    "indentation": asdict(settings),
                    "element": MIXED_PAD_ELEMENT,
                    "carrier_element": CARRIER_ELEMENT,
                    "constitutive_law": CONSTITUTIVE_LAW,
                    "young_modulus_mpa": YOUNG_MODULUS_MPA,
                    "young_modulus_role": "placeholder, not calibrated silicone",
                    "poisson_ratio": POISSON_RATIO,
                    "thickness_mm": THICKNESS_MM,
                    "mortar_type": MORTAR_TYPE,
                    "contact_process": "ALMContactProcess",
                    "contact_parameter_policy": "Kratos 10.3 defaults, identical for medium/fine",
                    "linear_solver": "skyline_lu_factorization, identical for medium/fine",
                    "convergence_tolerance": {
                        "relative": RELATIVE_TOLERANCE,
                        "absolute": ABSOLUTE_TOLERANCE,
                    },
                    "maximum_newton_iterations": MAXIMUM_NEWTON_ITERATIONS,
                    "iteration_level_active_set_capture": {
                        "available": False,
                        "reason": (
                            "Kratos strategy exposes converged-step NL_ITERATION_NUMBER "
                            "and ACTIVE_SET_CONVERGED; no stable observer callback is "
                            "available without replacing the solve loop"
                        ),
                    },
                },
                "mesh": {
                    "pad_nodes": len(base_topology.pad_node_ids),
                    "pad_elements": len(base_topology.pad_element_ids),
                    "fixed_carrier_nodes": len(base_topology.carrier_node_ids),
                    "fixed_carrier_elements": len(base_topology.carrier_element_ids),
                    "indenter_nodes": len(indenter_topology.node_ids),
                    "indenter_elements": len(indenter_topology.element_ids),
                    "indenter_contact_edges": len(indenter_topology.contact_edges),
                    "indenter_minimum_triangle_angle_degrees": indenter_mesh.minimum_triangle_angle_degrees,
                    "indenter_maximum_contact_edge_length_mm": indenter_mesh.maximum_contact_edge_length_mm,
                    "pad_indenter_node_ids_disjoint": set(base_topology.pad_node_ids).isdisjoint(indenter_topology.node_ids),
                },
                "runtime_contact_contract": runtime_contact,
                "continuous_u_aggregate_contract": aggregate_contract,
                "internal_contact_lm_boundary_treatment": (
                    "Kratos process default; no manual LM pressure constraints"
                ),
                "strategy_check": strategy_check,
                "capture_steps": {str(step): depth for step, depth in capture_steps.items()},
            }
        )
        solver = analysis._GetSolver()
        for step in range(1, settings.number_of_steps + 1):
            travel = settings.indentation_mm * step / settings.number_of_steps
            analysis.time = solver.AdvanceInTime(analysis.time)
            set_indenter_travel(
                model_part, indenter_topology.node_ids, fixture, travel
            )
            analysis.InitializeSolutionStep()
            solver.Predict()
            step_start = time.perf_counter()
            solver_converged = bool(solver.SolveSolutionStep())
            step_time = time.perf_counter() - step_start
            solve_time += step_time
            analysis.FinalizeSolutionStep()

            if not solver_converged:
                _, failed_reactions = extract_nodal_fields(model_part, all_node_ids)
                reaction_values = [
                    component
                    for reaction in failed_reactions.values()
                    for component in reaction
                ]
                result["failure_reason"] = "nonlinear_solver_did_not_converge"
                result["failure_step"] = step
                result["failure_step_diagnostics"] = {
                    "step": step,
                    "pseudo_time": float(analysis.time),
                    "prescribed_indenter_travel_mm": travel,
                    "nonlinear_iterations": int(
                        model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]
                    ),
                    "active_set_converged": bool(
                        model_part.ProcessInfo[CSMA.ACTIVE_SET_CONVERGED]
                    ),
                    "solve_wall_clock_seconds": step_time,
                    "finite_field_failures": finite_field_failures(
                        model_part, base_topology.pad_node_ids
                    )[:100],
                    "reaction_components": failure_statistics(reaction_values),
                    "contact_groups": failed_contact_group_diagnostics(
                        model, model_part, contact_group_definitions
                    ),
                    "det_f": {
                        "available": False,
                        "reason": (
                            "failed iterate contains non-finite nodal fields; "
                            "no physical det(F) metric is reported"
                        ),
                    },
                }
                break

            if "assembled_contact_lm_contract" not in result:
                assembled_lm_node_ids = sorted(
                    {
                        int(dof.Id())
                        for index, _ in enumerate(
                            contact_group_definitions
                        )
                        for condition in model[
                            "Structure.ComputingContact."
                            f"ComputingContactSub{index}"
                        ].Conditions
                        for dof in condition.GetDofList(
                            model_part.ProcessInfo
                        )
                        if dof.GetVariable().Name()
                        == (
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
                        )
                    }
                )
                internal_source_ids = {
                    node.Id
                    for name in (
                        "PadCutoutLeft",
                        "PadCutoutRight",
                        "PadCutoutBottom",
                        "StemLeft",
                        "StemRight",
                        "StemBottom",
                    )
                    for node in model_part.GetSubModelPart(name).Nodes
                }
                external_slave_ids = {
                    node.Id
                    for node in model_part.GetSubModelPart("PadOuterArc").Nodes
                }
                result["assembled_contact_lm_contract"] = {
                    "assembled_lm_node_ids": assembled_lm_node_ids,
                    "assembled_lm_node_count": len(assembled_lm_node_ids),
                    "identification_method": (
                        "LM DOFs returned by generated ComputingContact "
                        "condition GetDofList"
                    ),
                    "external_slave_lm_node_count": len(
                        set(assembled_lm_node_ids).intersection(
                            external_slave_ids
                        )
                    ),
                    "internal_exclusive_lm_node_ids": sorted(
                        set(assembled_lm_node_ids)
                        .intersection(internal_source_ids)
                        .difference(external_slave_ids)
                    ),
                    "no_internal_contact_lm_assembly": not (
                        set(assembled_lm_node_ids)
                        .intersection(internal_source_ids)
                        .difference(external_slave_ids)
                    ),
                }

            displacements, reactions = extract_nodal_fields(model_part, all_node_ids)
            field_failures = finite_field_failures(
                model_part, base_topology.pad_node_ids
            )
            pad_statistics = pad_strain_det_f_statistics(mesh, displacements)
            volumetric_values = {
                node_id: float(
                    model_part.Nodes[node_id].GetSolutionStepValue(
                        KM.VOLUMETRIC_STRAIN
                    )
                )
                for node_id in base_topology.pad_node_ids
            }
            volumetric_statistics = {
                **scalar_statistics(list(volumetric_values.values())),
                "max_abs": max(abs(value) for value in volumetric_values.values()),
            }
            volumetric_oscillation = unstructured_volumetric_oscillation(
                mesh, volumetric_values
            )
            contact_groups = contact_group_step_metrics(
                model,
                model_part,
                mesh,
                indenter_topology,
                contact_group_definitions,
            )
            loading_direction = fixture.frame.loading_direction
            indenter_signed = unique_projected_reaction(
                reactions, indenter_topology.node_ids, loading_direction
            )
            support_signed = unique_projected_reaction(
                reactions, support_node_ids, loading_direction
            )
            indenter_normal_reaction = compressive_indenter_reaction(
                reactions, indenter_topology.node_ids, loading_direction
            )
            contact_width = contact_width_metrics(
                mesh,
                contact_groups["external_pad_indenter"]["active_slave_node_ids"],
                fixture.frame.tangent,
            )
            achieved_displacements = [
                displacements[node_id][0] * loading_direction[0]
                + displacements[node_id][1] * loading_direction[1]
                for node_id in indenter_topology.node_ids
            ]
            pad_displacement_values = [
                math.hypot(*displacements[node_id])
                for node_id in base_topology.pad_node_ids
            ]
            rigid_validation = rigid_domain_validation(
                model_part,
                indenter_topology.node_ids,
                indenter_topology.element_ids,
                fixture.displacement_for_travel(travel),
            )
            point: dict[str, Any] = {
                "step": step,
                "pseudo_time": float(analysis.time),
                "prescribed_indenter_travel_mm": travel,
                "achieved_indentation_mm": sum(achieved_displacements) / len(achieved_displacements),
                "indenter_signed_reaction_along_loading_n": indenter_signed,
                "indenter_normal_reaction_n": indenter_normal_reaction,
                "support_signed_reaction_along_loading_n": support_signed,
                "force_equilibrium_error": relative_force_equilibrium_error(
                    indenter_signed, support_signed, settings.force_floor_n
                ),
                "nonlinear_iterations": int(model_part.ProcessInfo[KM.NL_ITERATION_NUMBER]),
                "solver_converged": solver_converged,
                "active_set_converged": bool(model_part.ProcessInfo[CSMA.ACTIVE_SET_CONVERGED]),
                "solve_wall_clock_seconds": step_time,
                "finite_fields": not field_failures and pad_statistics["all_finite"],
                "contact_groups": contact_groups,
                "external_contact_width": contact_width,
                "pad_strain_det_f": pad_statistics,
                "volumetric_strain": volumetric_statistics,
                "volumetric_strain_oscillation": volumetric_oscillation,
                "maximum_pad_displacement_mm": max(pad_displacement_values),
                "rigid_indenter_validation": rigid_validation,
            }
            if field_failures:
                point["non_finite_fields"] = field_failures[:50]
            if converged_step_observer is not None:
                observer_value = converged_step_observer(
                    ConvergedIndentationStep(
                        model=model,
                        model_part=model_part,
                        fingertip_model=fingertip_model,
                        mesh=mesh,
                        fixture=fixture,
                        base_topology=base_topology,
                        indenter_topology=indenter_topology,
                        settings=settings,
                        displacements=displacements,
                        reactions=reactions,
                        result_point=point,
                        elapsed_case_seconds=time.perf_counter() - start,
                    )
                )
                if observer_value is not None:
                    point["converged_step_observation"] = dict(observer_value)
            result["history"].append(point)

            if step in capture_steps:
                depth = capture_steps[step]
                profile = extract_outer_arc_profile(
                    fingertip_model,
                    mesh,
                    displacements,
                    fixture.frame,
                )
                snapshots[f"{depth:g}"] = {
                    "depth_mm": depth,
                    "step": step,
                    "displacements": displacements,
                    "profile": profile,
                    "active_external_node_ids": contact_groups[
                        "external_pad_indenter"
                    ]["active_slave_node_ids"],
                    "active_internal_node_ids": {
                        name: data["active_slave_node_ids"]
                        for name, data in contact_groups.items()
                        if name != "external_pad_indenter"
                    },
                    "pad_strain_det_f": pad_statistics,
                }

            failure_reason = None
            if field_failures or not pad_statistics["all_finite"]:
                failure_reason = "non_finite_field"
            elif pad_statistics["det_f"]["nonpositive_count"] > 0:
                failure_reason = "nonpositive_pad_det_f"
            elif not rigid_validation["pass"]:
                failure_reason = "rigid_indenter_deformed"
            if failure_reason is not None:
                result["failure_reason"] = failure_reason
                result["failure_step"] = step
                break

        result["solve_wall_clock_seconds"] = solve_time
        curve_checks = curve_acceptance(
            result["history"], settings.numerical_force_tolerance_n
        )
        result["curve_diagnostics"] = curve_checks
        completed = len(result["history"]) == settings.number_of_steps
        all_fields_finite = completed and all(point["finite_fields"] for point in result["history"])
        all_det_f_positive = completed and all(
            point["pad_strain_det_f"]["det_f"]["nonpositive_count"] == 0
            for point in result["history"]
        )
        all_active_sets_converged = completed and all(
            point["active_set_converged"] for point in result["history"]
        )
        external_active = completed and result["history"][-1]["contact_groups"][
            "external_pad_indenter"
        ]["active_condition_count"] > 0
        all_penetrations_pass = completed and all(
            group["penetration_pass"]
            for point in result["history"]
            for group in point["contact_groups"].values()
        )
        contact_force_points = [
            point
            for point in result["history"]
            if point["indenter_normal_reaction_n"]
            > settings.numerical_force_tolerance_n
        ]
        equilibrium_pass = bool(contact_force_points) and all(
            point["force_equilibrium_error"] < 0.02
            for point in contact_force_points
        )
        volumetric_pass = completed and all(
            point["volumetric_strain_oscillation"]["pass"]
            for point in result["history"]
        )
        target_reached = completed and abs(
            result["history"][-1]["achieved_indentation_mm"] - settings.indentation_mm
        ) <= 1.0e-9
        case_checks = {
            "target_displacement_reached": target_reached,
            "external_contact_active": external_active,
            "all_fields_finite": all_fields_finite,
            "all_pad_det_f_positive": all_det_f_positive,
            "force_curve_smooth_and_monotonic": curve_checks.get("smooth", False),
            "force_equilibrium_below_2_percent": equilibrium_pass,
            "contact_penetration_within_tolerance": all_penetrations_pass,
            "volumetric_checkerboard_absent": volumetric_pass,
            "active_set_converged_every_step": all_active_sets_converged,
            "rigid_indenter_remained_rigid": completed
            and all(point["rigid_indenter_validation"]["pass"] for point in result["history"]),
        }
        result["case_acceptance_checks"] = case_checks
        result["solve_status"] = "PASS" if completed else "FAIL"
        result["status"] = "PASS" if all(case_checks.values()) else "FAIL"
        if completed and result["status"] == "FAIL" and "failure_reason" not in result:
            result["failure_reason"] = "case_acceptance_checks_failed"
        if result["history"]:
            result["final"] = result["history"][-1]
            result["maximum_nonlinear_iterations"] = max(
                point["nonlinear_iterations"] for point in result["history"]
            )
            result["minimum_pad_det_f"] = min(
                point["pad_strain_det_f"]["det_f"]["min"]
                for point in result["history"]
            )
            result["maximum_pad_strain"] = max(
                point["pad_strain_det_f"]["maximum_principal_green_lagrange_strain"]["value"]
                for point in result["history"]
            )
            result["maximum_penetration_mm"] = max(
                float(group["signed_geometric_gap"].get("maximum_penetration_mm") or 0.0)
                for point in result["history"]
                for group in point["contact_groups"].values()
            )
            result["maximum_force_equilibrium_error"] = max(
                point["force_equilibrium_error"] for point in contact_force_points
            ) if contact_force_points else None
        artifacts = IndentationArtifacts(
            mesh=mesh,
            fixture=fixture,
            indenter_mesh=indenter_mesh,
            indenter_topology=indenter_topology,
            snapshots=snapshots,
        )
    except Exception as exception:
        result["failure_reason"] = "exception"
        result["exception"] = f"{type(exception).__name__}: {exception}"
        result["traceback"] = traceback.format_exc()
    finally:
        if initialized and analysis is not None:
            try:
                analysis.Finalize()
            except Exception as exception:
                result["finalize_exception"] = f"{type(exception).__name__}: {exception}"
    result["case_wall_clock_seconds"] = time.perf_counter() - start
    return result, artifacts
