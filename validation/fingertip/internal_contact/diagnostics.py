"""Phase 4I-D isolation, contact-purity, DOF, and tangent diagnostics.

The diagnostic assembly and the production nonlinear solve intentionally use
different fresh ``Kratos.Model`` instances.  Building the tangent here must
not mutate the state used to decide whether the original Skyline solve
converges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from mesh.fingertip import generate_fingertip_mesh
from fem.indentation import (
    IndentationSettings,
    run_indentation_case,
)
from mesh.indenter import (
    build_indenter_fixture,
    generate_indenter_mesh,
)
from fem.contact import (
    PAD_U_AGGREGATE,
    PAD_U_SEGMENTS,
    STEM_U_AGGREGATE,
    STEM_U_SEGMENTS,
    create_continuous_u_submodel_parts,
    u_corner_node_ids,
)
from fem.kratos_adapter import (
    apply_indentation_constraints,
    import_kratos,
    populate_indenter_model_part,
    populate_kratos_model_part,
    set_indenter_travel,
)
from fem.results import extract_nodal_fields
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
    build_indentation_project_parameters_json,
    indentation_contact_groups,
    validate_internal_contact_configuration,
)
from mesh.types import MeshLevel, mesh_settings_for_level
from validation.fingertip.internal_contact.sparse import analyze_sparse_system
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


FIRST_STEP_TRAVEL_MM = 0.25 / 48.0

CASE_CONFIGURATIONS = {
    "A": "none",
    "B": "bottom_only",
    "C": "sides_separate",
    "D": "three_pairs",
    "E": "continuous_u",
    "C-left": "left_only",
    "C-right": "right_only",
}

CASE_DIRECTORY_NAMES = {
    "A": "case_a_external_only",
    "B": "case_b_bottom_only",
    "C": "case_c_sides_separate",
    "D": "case_d_three_pairs",
    "E": "case_e_continuous_u",
    "C-left": "case_c_left_only",
    "C-right": "case_c_right_only",
}

SEMANTIC_SURFACES = (
    "PadOuterArc",
    "IndenterContactArc",
    *PAD_U_SEGMENTS,
    *STEM_U_SEGMENTS,
)


@dataclass
class DiagnosticContext:
    """Objects owned by one fresh diagnostic-only Kratos model."""

    configuration: str
    groups: tuple[tuple[str, str, str], ...]
    fingertip_model: Any
    mesh: Any
    fixture: Any
    indenter_mesh: Any
    model: Any
    analysis: Any
    model_part: Any
    base_topology: Any
    indenter_topology: Any
    aggregate_contract: dict[str, Any] | None
    counts_before_initialize: dict[str, int]
    counts_after_initialize: dict[str, int]


def configuration_for_case(case: str) -> str:
    """Return the explicit internal-contact configuration for a case label."""
    try:
        return CASE_CONFIGURATIONS[case]
    except KeyError as exception:
        raise ValueError(
            f"unsupported diagnostic case {case!r}; expected one of "
            f"{', '.join(CASE_CONFIGURATIONS)}"
        ) from exception


def _entity_counts(model_part: Any) -> dict[str, int]:
    return {
        "nodes": int(model_part.NumberOfNodes()),
        "elements": int(model_part.NumberOfElements()),
        "conditions": int(model_part.NumberOfConditions()),
    }


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "all_finite": len(finite) == len(values),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
    }


def build_diagnostic_context(
    mesh_level: MeshLevel,
    configuration: str,
    number_of_steps: int = 1,
    mesh_override: Any | None = None,
    before_initialize: Any | None = None,
) -> DiagnosticContext:
    KM, _, _, _ = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
        StructuralMechanicsAnalysis,
    )

    validated = validate_internal_contact_configuration(configuration)
    fingertip_model = FingertipModel(FingertipParameters())
    mesh = (
        mesh_override
        if mesh_override is not None
        else generate_fingertip_mesh(
            fingertip_model, mesh_settings_for_level(mesh_level)
        )
    )
    if mesh.settings.level != mesh_level:
        raise ValueError("mesh_override level must match mesh_level")
    fixture = build_indenter_fixture(fingertip_model)
    indenter_mesh = generate_indenter_mesh(
        fixture, mesh.settings.contact_boundary_target_size_mm
    )
    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(
        model,
        KM.Parameters(
            build_indentation_project_parameters_json(
                number_of_steps, validated
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
        if validated == "continuous_u"
        else None
    )
    counts_before_initialize = _entity_counts(model_part)
    if before_initialize is not None:
        before_initialize(
            model_part,
            base_topology,
            indenter_topology,
            mesh,
            fixture,
        )
    analysis.Initialize()
    apply_indentation_constraints(
        model_part, base_topology, indenter_topology
    )
    return DiagnosticContext(
        configuration=validated,
        groups=indentation_contact_groups(validated),
        fingertip_model=fingertip_model,
        mesh=mesh,
        fixture=fixture,
        indenter_mesh=indenter_mesh,
        model=model,
        analysis=analysis,
        model_part=model_part,
        base_topology=base_topology,
        indenter_topology=indenter_topology,
        aggregate_contract=aggregate_contract,
        counts_before_initialize=counts_before_initialize,
        counts_after_initialize=_entity_counts(model_part),
    )


def _model_part_ids(model_part: Any) -> dict[str, list[int]]:
    return {
        "node_ids": sorted(node.Id for node in model_part.Nodes),
        "condition_ids": sorted(
            condition.Id for condition in model_part.Conditions
        ),
    }


def _coordinates(nodes: Any) -> dict[str, Any]:
    values = list(nodes)
    return {
        "count": len(values),
        "x_range_mm": (
            [min(float(node.X0) for node in values), max(float(node.X0) for node in values)]
            if values
            else None
        ),
        "y_range_mm": (
            [min(float(node.Y0) for node in values), max(float(node.Y0) for node in values)]
            if values
            else None
        ),
    }


def runtime_contract(context: DiagnosticContext) -> dict[str, Any]:
    KM, CSMA, _, _ = import_kratos()
    groups: dict[str, Any] = {}
    selected_source_conditions: set[int] = set()
    for index, (name, slave_name, master_name) in enumerate(context.groups):
        slave = context.model_part.GetSubModelPart(slave_name)
        master = context.model_part.GetSubModelPart(master_name)
        contact_path = f"Structure.Contact.ContactSub{index}"
        computing_path = (
            f"Structure.ComputingContact.ComputingContactSub{index}"
        )
        contact = context.model[contact_path]
        source_ids = {
            condition.Id for condition in slave.Conditions
        }.union(condition.Id for condition in master.Conditions)
        selected_source_conditions.update(source_ids)
        runtime_ids = {condition.Id for condition in contact.Conditions}
        source_connectivity = {
            tuple(sorted(_connectivity(condition.GetGeometry())))
            for condition in (*list(slave.Conditions), *list(master.Conditions))
        }
        runtime_connectivity = {
            tuple(sorted(_connectivity(condition.GetGeometry())))
            for condition in contact.Conditions
        }
        slave_h = [
            float(node.GetSolutionStepValue(KM.NODAL_H))
            for node in slave.Nodes
        ]
        master_h = [
            float(node.GetSolutionStepValue(KM.NODAL_H))
            for node in master.Nodes
        ]
        checks = {
            "source_pair_exactly_populates_contact_subpart": (
                source_connectivity == runtime_connectivity
            ),
            "all_slave_nodes_flagged_slave": all(
                node.Is(KM.SLAVE) for node in slave.Nodes
            ),
            "all_master_nodes_flagged_master": all(
                node.Is(KM.MASTER) for node in master.Nodes
            ),
            "slave_nodal_h_positive_finite": bool(slave_h)
            and all(math.isfinite(value) and value > 0.0 for value in slave_h),
            "master_nodal_h_positive_finite": bool(master_h)
            and all(math.isfinite(value) and value > 0.0 for value in master_h),
        }
        groups[name] = {
            "index": index,
            "slave": slave_name,
            "master": master_name,
            "contact_submodelpart": contact_path,
            "computing_contact_submodelpart": computing_path,
            "slave_membership": _model_part_ids(slave),
            "master_membership": _model_part_ids(master),
            "slave_coordinates": _coordinates(slave.Nodes),
            "master_coordinates": _coordinates(master.Nodes),
            "source_condition_ids": sorted(source_ids),
            "contact_submodelpart_condition_ids": sorted(runtime_ids),
            "source_and_runtime_condition_ids_equal": source_ids == runtime_ids,
            "condition_identity_contract": (
                "Exact Line2 connectivity/node identity. Kratos may renumber "
                "interface conditions according to ContactSub index."
            ),
            "slave_node_flags": {
                "SLAVE": sum(node.Is(KM.SLAVE) for node in slave.Nodes),
                "MASTER": sum(node.Is(KM.MASTER) for node in slave.Nodes),
            },
            "master_node_flags": {
                "SLAVE": sum(node.Is(KM.SLAVE) for node in master.Nodes),
                "MASTER": sum(node.Is(KM.MASTER) for node in master.Nodes),
            },
            "slave_condition_flags": {
                "SLAVE": sum(
                    condition.Is(KM.SLAVE)
                    for condition in slave.Conditions
                ),
                "MASTER": sum(
                    condition.Is(KM.MASTER)
                    for condition in slave.Conditions
                ),
            },
            "master_condition_flags": {
                "SLAVE": sum(
                    condition.Is(KM.SLAVE)
                    for condition in master.Conditions
                ),
                "MASTER": sum(
                    condition.Is(KM.MASTER)
                    for condition in master.Conditions
                ),
            },
            "slave_nodal_h": _statistics(slave_h),
            "master_nodal_h": _statistics(master_h),
            "slave_lm_field": _statistics(
                [
                    float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    )
                    for node in slave.Nodes
                ]
            ),
            "checks": checks,
        }

    unused: dict[str, Any] = {}
    selected_names = {
        surface for _, slave, master in context.groups for surface in (slave, master)
    }
    if PAD_U_AGGREGATE in selected_names:
        selected_names.update(PAD_U_SEGMENTS)
    if STEM_U_AGGREGATE in selected_names:
        selected_names.update(STEM_U_SEGMENTS)
    for name in (*PAD_U_SEGMENTS, *STEM_U_SEGMENTS):
        part = context.model_part.GetSubModelPart(name)
        condition_ids = {condition.Id for condition in part.Conditions}
        if name not in selected_names:
            unused[name] = {
                "condition_ids": sorted(condition_ids),
                "condition_ids_in_any_contact_pair": sorted(
                    condition_ids.intersection(selected_source_conditions)
                ),
                "condition_slave_flag_count": sum(
                    condition.Is(KM.SLAVE)
                    for condition in part.Conditions
                ),
                "condition_master_flag_count": sum(
                    condition.Is(KM.MASTER)
                    for condition in part.Conditions
                ),
                "condition_excluded": not condition_ids.intersection(
                    selected_source_conditions
                )
                and not any(
                    condition.Is(KM.SLAVE) or condition.Is(KM.MASTER)
                    for condition in part.Conditions
                ),
                "nodal_flag_note": (
                    "Nodes shared by a selected and an unused semantic segment "
                    "may inherit the selected segment's role; exclusion is "
                    "therefore asserted on source conditions and assembly."
                ),
            }
    all_group_checks = [
        value
        for group in groups.values()
        for value in group["checks"].values()
    ]
    return {
        "groups": groups,
        "all_group_contracts_pass": all(all_group_checks),
        "unused_internal_surfaces": unused,
        "all_unused_internal_conditions_excluded": all(
            record["condition_excluded"] for record in unused.values()
        ),
        "initial_penalty": float(
            context.model_part.ProcessInfo[KM.INITIAL_PENALTY]
        ),
        "scale_factor": float(
            context.model_part.ProcessInfo[KM.SCALE_FACTOR]
        ),
        "global_lm_dof_contract": (
            "Kratos 10.3 AuxiliaryAddDofs adds the scalar ALM pressure DOF to "
            "the root ModelPart. Unused surfaces are verified by condition "
            "flags, ContactSub membership, and assembled-DOF participation."
        ),
    }


def _connectivity(geometry: Any) -> tuple[int, ...]:
    return tuple(node.Id for node in geometry)


def source_condition_maps(
    context: DiagnosticContext,
) -> tuple[
    dict[str, dict[tuple[int, ...], int]],
    dict[int, list[str]],
]:
    by_surface: dict[str, dict[tuple[int, ...], int]] = {}
    semantics: dict[int, list[str]] = {}
    surface_names = list(SEMANTIC_SURFACES)
    for aggregate_name in (PAD_U_AGGREGATE, STEM_U_AGGREGATE):
        if context.model_part.HasSubModelPart(aggregate_name):
            surface_names.append(aggregate_name)
    for name in surface_names:
        part = context.model_part.GetSubModelPart(name)
        by_surface[name] = {}
        for condition in part.Conditions:
            signature = tuple(sorted(_connectivity(condition.GetGeometry())))
            by_surface[name][signature] = condition.Id
            semantics.setdefault(condition.Id, []).append(name)
    return by_surface, semantics


def _mean_nodal_vector(geometry: Any, variable: Any) -> list[float]:
    nodes = list(geometry)
    if not nodes:
        return [0.0, 0.0]
    return [
        sum(
            float(node.GetSolutionStepValue(variable)[component])
            for node in nodes
        )
        / len(nodes)
        for component in range(2)
    ]


def _mean_nodal_scalar(geometry: Any, variable: Any) -> float | None:
    nodes = list(geometry)
    values = [
        float(node.GetSolutionStepValue(variable)) for node in nodes
    ]
    return sum(values) / len(values) if values else None


def _semantic_key(names: Sequence[str]) -> str | None:
    for name in names:
        lowered = name.lower()
        for key in ("left", "bottom", "right", "outerarc", "contactarc"):
            if key in lowered:
                return key
    return None


def contact_condition_records(
    context: DiagnosticContext,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read CouplingGeometry parts 0/1; never infer parents from coordinates."""
    KM, CSMA, _, _ = import_kratos()
    by_surface, condition_semantics = source_condition_maps(context)
    records: list[dict[str, Any]] = []
    group_summary: dict[str, Any] = {}
    generated_ids_seen: list[int] = []
    for index, (group_name, slave_name, master_name) in enumerate(
        context.groups
    ):
        computing = context.model[
            f"Structure.ComputingContact.ComputingContactSub{index}"
        ]
        group_records: list[dict[str, Any]] = []
        for condition in computing.Conditions:
            geometry = condition.GetGeometry()
            slave_geometry = geometry.GetGeometryPart(0)
            master_geometry = geometry.GetGeometryPart(1)
            slave_signature = tuple(
                sorted(_connectivity(slave_geometry))
            )
            master_signature = tuple(
                sorted(_connectivity(master_geometry))
            )
            slave_source_id = by_surface[slave_name].get(slave_signature)
            master_source_id = by_surface[master_name].get(master_signature)
            slave_semantic = (
                condition_semantics.get(slave_source_id, [])
                if slave_source_id is not None
                else []
            )
            master_semantic = (
                condition_semantics.get(master_source_id, [])
                if master_source_id is not None
                else []
            )
            semantic_match = (
                True
                if group_name == "external_pad_indenter"
                else _semantic_key(slave_semantic)
                == _semantic_key(master_semantic)
            )
            record = {
                "contact_process_index": index,
                "contact_group": group_name,
                "generated_condition_id": condition.Id,
                "generated_condition_info": condition.Info().split(" #", 1)[0],
                "active": bool(condition.Is(KM.ACTIVE)),
                "source_slave_submodelpart": slave_name,
                "source_master_submodelpart": master_name,
                "slave_source_condition_id": slave_source_id,
                "master_source_condition_id": master_source_id,
                "slave_node_ids": list(_connectivity(slave_geometry)),
                "master_node_ids": list(_connectivity(master_geometry)),
                "slave_semantic_regions": slave_semantic,
                "master_semantic_regions": master_semantic,
                "slave_normal": _mean_nodal_vector(
                    slave_geometry, KM.NORMAL
                ),
                "master_normal": _mean_nodal_vector(
                    master_geometry, KM.NORMAL
                ),
                "weighted_gap_mean": _mean_nodal_scalar(
                    slave_geometry, CSMA.WEIGHTED_GAP
                ),
                "lagrange_multiplier_contact_pressure_mean": (
                    _mean_nodal_scalar(
                        slave_geometry,
                        CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
                    )
                ),
                "source_pair_resolved": (
                    slave_source_id is not None
                    and master_source_id is not None
                ),
                "semantic_region_match": semantic_match,
                "pair_pure": (
                    slave_source_id is not None
                    and master_source_id is not None
                    and semantic_match
                ),
                "parent_resolution_method": (
                    "CouplingGeometry.GetGeometryPart(0/1) connectivity "
                    "matched to exact source SubModelPart conditions"
                ),
            }
            records.append(record)
            group_records.append(record)
            generated_ids_seen.append(condition.Id)
        group_summary[group_name] = {
            "generated_condition_count": len(group_records),
            "active_condition_count": sum(
                record["active"] for record in group_records
            ),
            "all_source_pairs_resolved": all(
                record["source_pair_resolved"] for record in group_records
            ),
            "all_conditions_pair_pure": all(
                record["pair_pure"] for record in group_records
            ),
        }
    return records, {
        "groups": group_summary,
        "generated_condition_ids_unique_across_processes": (
            len(generated_ids_seen) == len(set(generated_ids_seen))
        ),
        "all_generated_conditions_pair_pure": all(
            record["pair_pure"] for record in records
        ),
        "generated_condition_parent_api": "available",
        "generated_condition_parent_api_limit": (
            "Kratos exposes the paired slave/master Line2 geometries, not a "
            "direct Python property containing the original condition IDs; "
            "IDs are recovered by exact connectivity within declared sources."
        ),
    }


def _line_normal(condition: Any) -> list[float]:
    geometry = condition.GetGeometry()
    first = geometry[0]
    second = geometry[1]
    dx = float(second.X0 - first.X0)
    dy = float(second.Y0 - first.Y0)
    length = math.hypot(dx, dy)
    return [dy / length, -dx / length]


def _corner_contract(
    context: DiagnosticContext,
    assembled_dofs: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    KM, CSMA, _, _ = import_kratos()
    corner_ids = u_corner_node_ids(context.model_part)
    slave_names = [slave for _, slave, _ in context.groups]
    output: dict[str, Any] = {}
    for label, node_id in corner_ids.items():
        node = context.model_part.Nodes[node_id]
        semantic_memberships = [
            name
            for name in (*PAD_U_SEGMENTS, *STEM_U_SEGMENTS)
            if node_id
            in {
                member.Id
                for member in context.model_part.GetSubModelPart(name).Nodes
            }
        ]
        incident = []
        signatures: list[tuple[int, ...]] = []
        for name in semantic_memberships:
            for condition in context.model_part.GetSubModelPart(name).Conditions:
                connectivity = _connectivity(condition.GetGeometry())
                if node_id not in connectivity:
                    continue
                signature = tuple(sorted(connectivity))
                if signature in signatures:
                    continue
                signatures.append(signature)
                incident.append(
                    {
                        "condition_id": condition.Id,
                        "connectivity": list(connectivity),
                        "semantic_segment": name,
                        "normal": _line_normal(condition),
                        "MASTER": bool(condition.Is(KM.MASTER)),
                        "SLAVE": bool(condition.Is(KM.SLAVE)),
                    }
                )
        aggregate_membership = [
            name
            for name in (PAD_U_AGGREGATE, STEM_U_AGGREGATE)
            if context.model_part.HasSubModelPart(name)
            and node_id
            in {
                member.Id
                for member in context.model_part.GetSubModelPart(name).Nodes
            }
        ]
        lm_key = (
            node_id,
            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name(),
        )
        lm = dict(assembled_dofs.get(lm_key, {}))
        if node.HasDofFor(CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE):
            nodal_dof = node.GetDof(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
            lm.update(
                {
                    "name": CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name(),
                    "node_has_dof": True,
                    "equation_id": int(nodal_dof.EquationId),
                    "fixed": bool(nodal_dof.IsFixed()),
                    "assembled": lm_key in assembled_dofs,
                }
            )
        output[label] = {
            "node_id": node_id,
            "reference_coordinate_mm": [
                float(node.X0),
                float(node.Y0),
                float(node.Z0),
            ],
            "semantic_segment_membership": semantic_memberships,
            "aggregate_u_membership": aggregate_membership,
            "incident_conditions": sorted(
                incident, key=lambda record: record["condition_id"]
            ),
            "incident_condition_count": len(incident),
            "duplicate_incident_connectivity_count": (
                len(signatures) - len(set(signatures))
            ),
            "nodal_normal": [
                float(node.GetSolutionStepValue(KM.NORMAL)[index])
                for index in range(3)
            ],
            "nodal_h": float(node.GetSolutionStepValue(KM.NODAL_H)),
            "flags": {
                "MASTER": bool(node.Is(KM.MASTER)),
                "SLAVE": bool(node.Is(KM.SLAVE)),
                "ACTIVE": bool(node.Is(KM.ACTIVE)),
            },
            "contact_related_dofs": [lm],
            "contact_process_registration_count": sum(
                context.model_part.GetSubModelPart(name).HasNode(node_id)
                for name in slave_names
            ),
        }
    return {
        "corners": output,
        "four_corner_node_ids_unique": len(set(corner_ids.values())) == 4,
        "all_incident_connectivity_unique": all(
            record["duplicate_incident_connectivity_count"] == 0
            for record in output.values()
        ),
    }


def dof_records(
    context: DiagnosticContext,
    dof_set: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[tuple[int, str], dict[str, Any]],
    dict[str, Any],
]:
    KM, CSMA, _, _ = import_kratos()
    assembled: dict[tuple[int, str], dict[str, Any]] = {}
    equation_map: dict[int, dict[str, Any]] = {}
    equation_ids: list[int] = []
    for dof in dof_set:
        variable_name = dof.GetVariable().Name()
        node_id = int(dof.Id())
        equation_id = int(dof.EquationId)
        node = context.model_part.Nodes[node_id]
        record = {
            "node_id": node_id,
            "variable": variable_name,
            "equation_id": equation_id,
            "fixed": bool(dof.IsFixed()),
            "reference_x_mm": float(node.X0),
            "reference_y_mm": float(node.Y0),
            "assembled": True,
        }
        assembled[(node_id, variable_name)] = record
        equation_map[equation_id] = record
        equation_ids.append(equation_id)

    variables = (
        ("displacement", KM.DISPLACEMENT_X),
        ("displacement", KM.DISPLACEMENT_Y),
        ("displacement", KM.DISPLACEMENT_Z),
        ("volumetric_strain", KM.VOLUMETRIC_STRAIN),
        (
            "contact_lm",
            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
        ),
    )
    rows: list[dict[str, Any]] = []
    for node in context.model_part.Nodes:
        for category, variable in variables:
            if not node.HasDofFor(variable):
                continue
            dof = node.GetDof(variable)
            key = (node.Id, variable.Name())
            rows.append(
                {
                    "node_id": node.Id,
                    "reference_x_mm": float(node.X0),
                    "reference_y_mm": float(node.Y0),
                    "category": category,
                    "variable": variable.Name(),
                    "equation_id": int(dof.EquationId),
                    "fixed": bool(dof.IsFixed()),
                    "assembled": key in assembled,
                }
            )
    assembled_rows = list(assembled.values())
    contact_name = CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
    group_lm: dict[str, Any] = {}
    for group_name, slave_name, _ in context.groups:
        node_ids = {
            node.Id
            for node in context.model_part.GetSubModelPart(slave_name).Nodes
        }
        group_lm[group_name] = {
            "source_slave_node_count": len(node_ids),
            "global_nodal_lm_dof_count": sum(
                context.model_part.Nodes[node_id].HasDofFor(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                )
                for node_id in node_ids
            ),
            "assembled_lm_dof_count": sum(
                (node_id, contact_name) in assembled for node_id in node_ids
            ),
        }
    summary = {
        "assembled_total_dof_count": len(assembled_rows),
        "assembled_free_dof_count": sum(
            not record["fixed"] for record in assembled_rows
        ),
        "assembled_fixed_dof_count": sum(
            record["fixed"] for record in assembled_rows
        ),
        "assembled_displacement_dof_count": sum(
            record["variable"].startswith("DISPLACEMENT_")
            for record in assembled_rows
        ),
        "assembled_volumetric_strain_dof_count": sum(
            record["variable"] == KM.VOLUMETRIC_STRAIN.Name()
            for record in assembled_rows
        ),
        "assembled_contact_lm_dof_count": sum(
            record["variable"] == contact_name for record in assembled_rows
        ),
        "global_nodal_contact_lm_dof_count": sum(
            node.HasDofFor(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
            for node in context.model_part.Nodes
        ),
        "contact_group_lm_dof_counts": group_lm,
        "duplicate_equation_ids": sorted(
            {
                identifier
                for identifier in equation_ids
                if equation_ids.count(identifier) > 1
            }
        ),
        "fixed_contact_unknowns": [
            record
            for record in assembled_rows
            if record["variable"] == contact_name and record["fixed"]
        ],
    }
    return rows, equation_map, assembled, summary


def _add_unused_surface_assembly_contract(
    context: DiagnosticContext,
    runtime: dict[str, Any],
    assembled_dofs: Mapping[tuple[int, str], Mapping[str, Any]],
) -> bool:
    """Distinguish Kratos' global nodal DOF addition from DofSet assembly."""
    _, CSMA, _, _ = import_kratos()
    lm_name = CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
    selected_slave_nodes = {
        node.Id
        for _, slave_name, _ in context.groups
        for node in context.model_part.GetSubModelPart(slave_name).Nodes
    }
    checks: list[bool] = []
    for name, record in runtime["unused_internal_surfaces"].items():
        part_nodes = {
            node.Id
            for node in context.model_part.GetSubModelPart(name).Nodes
        }
        exclusive_nodes = sorted(part_nodes - selected_slave_nodes)
        assembled = sorted(
            node_id
            for node_id in exclusive_nodes
            if (node_id, lm_name) in assembled_dofs
        )
        record.update(
            {
                "node_ids_exclusive_of_selected_slave_surfaces": (
                    exclusive_nodes
                ),
                "exclusive_node_assembled_lm_dof_ids": assembled,
                "exclusive_nodes_excluded_from_assembled_lm_dofset": (
                    not assembled
                ),
            }
        )
        checks.append(not assembled)
    passed = all(checks)
    runtime[
        "all_unused_exclusive_nodes_excluded_from_assembled_lm_dofset"
    ] = passed
    return passed


def assemble_first_step_diagnostics(
    case: str,
    mesh_level: MeshLevel = "medium",
    mesh_override: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble one tangent in a fresh model and retain sparse evidence."""
    configuration = configuration_for_case(case)
    context: DiagnosticContext | None = None
    initialized_step = False
    start = time.perf_counter()
    try:
        context = build_diagnostic_context(
            mesh_level, configuration, mesh_override=mesh_override
        )
        KM, _, _, _ = import_kratos()
        runtime = runtime_contract(context)
        solver = context.analysis._GetSolver()
        strategy_check_before_search = int(
            solver._GetSolutionStrategy().Check()
        )
        context.analysis.time = solver.AdvanceInTime(context.analysis.time)
        set_indenter_travel(
            context.model_part,
            context.indenter_topology.node_ids,
            context.fixture,
            FIRST_STEP_TRAVEL_MM,
        )
        context.analysis.InitializeSolutionStep()
        initialized_step = True
        solver.Predict()
        strategy = solver._GetSolutionStrategy()
        builder = solver._GetBuilderAndSolver()
        scheme = solver._GetScheme()
        dof_set = builder.GetDofSet()
        (
            dof_rows,
            equation_map,
            assembled_dofs,
            dof_summary,
        ) = dof_records(context, dof_set)
        unused_assembly_pass = _add_unused_surface_assembly_contract(
            context, runtime, assembled_dofs
        )

        matrix = strategy.GetSystemMatrix()
        rhs = strategy.GetSystemVector()
        increment = KM.Vector(matrix.Size1())
        builder.Build(
            scheme, solver.GetComputingModelPart(), matrix, rhs
        )
        builder.ApplyDirichletConditions(
            scheme,
            solver.GetComputingModelPart(),
            matrix,
            increment,
            rhs,
        )
        import KratosMultiphysics.scipy_conversion_tools as conversion

        csr = conversion.to_csr(matrix)
        rhs_array = np.asarray([rhs[index] for index in range(len(rhs))])
        matrix_diagnostics = analyze_sparse_system(
            csr, rhs_array, equation_map
        )
        row_norms = np.sqrt(
            np.asarray(csr.multiply(csr).sum(axis=1)).reshape(-1)
        )
        free_without_contribution = [
            {
                **equation_map[equation_id],
                "row_norm": float(row_norms[equation_id]),
            }
            for equation_id in sorted(equation_map)
            if not equation_map[equation_id]["fixed"]
            and row_norms[equation_id] == 0.0
        ]
        dof_summary["free_dofs_without_assembly_contribution"] = (
            free_without_contribution
        )

        contact_records, pair_purity = contact_condition_records(context)
        corner = _corner_contract(context, assembled_dofs)
        counts_after_search = _entity_counts(context.model_part)
        result = {
            "phase": "4I-D",
            "case": case,
            "configuration": configuration,
            "mesh_level": mesh_level,
            "status": "PASS"
            if runtime["all_group_contracts_pass"]
            and runtime["all_unused_internal_conditions_excluded"]
            and unused_assembly_pass
            and pair_purity["all_generated_conditions_pair_pure"]
            else "FAIL",
            "kratos_version": KM.Kernel.Version(),
            "first_step_travel_mm": FIRST_STEP_TRAVEL_MM,
            "fresh_model": True,
            "counts": {
                "before_initialize": context.counts_before_initialize,
                "after_initialize": context.counts_after_initialize,
                "after_first_search": counts_after_search,
            },
            "runtime_contact_contract": runtime,
            "dof_summary": dof_summary,
            "matrix_diagnostics": matrix_diagnostics,
            "contact_pair_purity": pair_purity,
            "corner_contract": corner,
            "continuous_u_aggregate_contract": (
                context.aggregate_contract
            ),
            "strategy_check_before_contact_search": (
                strategy_check_before_search
            ),
            "strategy_check_after_contact_search": {
                "available": False,
                "reason": (
                    "The Kratos 10.3 strategy Check() traverses generated "
                    "mortar conditions whose pair geometry has already been "
                    "consumed/reset by this diagnostic assembly."
                ),
            },
            "diagnostic_assembly_note": (
                "The first tangent/RHS was assembled in a dedicated fresh "
                "model and was not used for the production Skyline verdict."
            ),
            "wall_clock_seconds": time.perf_counter() - start,
        }
        return result, dof_rows, contact_records
    except Exception as exception:
        return (
            {
                "phase": "4I-D",
                "case": case,
                "configuration": configuration,
                "mesh_level": mesh_level,
                "status": "FAIL",
                "failure_reason": "diagnostic_assembly_exception",
                "exception": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
                "wall_clock_seconds": time.perf_counter() - start,
            },
            [],
            [],
        )
    finally:
        if context is not None:
            try:
                if initialized_step:
                    context.analysis.FinalizeSolutionStep()
                context.analysis.Finalize()
            except Exception:
                pass


def evaluate_first_step_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the Phase 4I-D one-step criteria, not full-curve criteria."""
    history = list(result.get("history", []))
    point = history[-1] if len(history) == 1 else None
    external = (
        point.get("contact_groups", {}).get("external_pad_indenter", {})
        if point
        else {}
    )
    reaction = (
        float(point.get("indenter_normal_reaction_n", math.nan))
        if point
        else math.nan
    )
    det_f_min = (
        point.get("pad_strain_det_f", {}).get("det_f", {}).get("min")
        if point
        else None
    )
    checks = {
        "one_step_completed": result.get("solve_status") == "PASS"
        and len(history) == 1,
        "external_contact_active": (
            int(external.get("active_condition_count", 0)) > 0
        ),
        "finite_positive_reaction": math.isfinite(reaction)
        and reaction > 0.0,
        "finite_displacement_and_volumetric_strain": bool(
            point and point.get("finite_fields")
        ),
        "positive_det_f": (
            det_f_min is not None
            and math.isfinite(float(det_f_min))
            and float(det_f_min) > 0.0
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "acceptance_checks": checks,
        "nonlinear_iterations": (
            point.get("nonlinear_iterations")
            if point
            else result.get("failure_step_diagnostics", {}).get(
                "nonlinear_iterations"
            )
        ),
        "failed_iteration": (
            result.get("failure_step_diagnostics", {}).get(
                "nonlinear_iterations"
            )
            if result.get("solve_status") != "PASS"
            else None
        ),
        "solver_converged": result.get("solve_status") == "PASS",
        "external_generated_condition_count": (
            external.get("generated_condition_count")
            if point
            else result.get("failure_step_diagnostics", {})
            .get("contact_groups", {})
            .get("external_pad_indenter", {})
            .get("generated_condition_count")
        ),
        "external_active_condition_count": (
            external.get("active_condition_count")
            if point
            else result.get("failure_step_diagnostics", {})
            .get("contact_groups", {})
            .get("external_pad_indenter", {})
            .get("active_condition_count")
        ),
        "reaction_n": reaction if math.isfinite(reaction) else None,
        "det_f_min": det_f_min,
        "underlying_result": dict(result),
    }


def run_first_step_case(
    case: str,
    mesh_level: MeshLevel = "medium",
) -> dict[str, Any]:
    """Run the original nonlinear strategy for exactly the Trial increment."""
    configuration = configuration_for_case(case)
    result, _ = run_indentation_case(
        FingertipModel(FingertipParameters()),
        mesh_level,
        IndentationSettings(
            indentation_mm=FIRST_STEP_TRAVEL_MM,
            number_of_steps=1,
        ),
        internal_contact_configuration=configuration,
    )
    evaluated = evaluate_first_step_result(result)
    evaluated.update(
        {
            "phase": "4I-D",
            "case": case,
            "configuration": configuration,
            "mesh_level": mesh_level,
            "requested_travel_mm": FIRST_STEP_TRAVEL_MM,
        }
    )
    return evaluated


def run_continuous_u_full_trial(
    mesh_level: MeshLevel = "medium",
    indentation_mm: float = 0.25,
    steps: int = 48,
) -> dict[str, Any]:
    """Run the gated continuous-U Trial in another fresh model."""
    result, _ = run_indentation_case(
        FingertipModel(FingertipParameters()),
        mesh_level,
        IndentationSettings(
            indentation_mm=indentation_mm,
            number_of_steps=steps,
        ),
        internal_contact_configuration="continuous_u",
    )
    result["phase"] = "4I-D"
    result["diagnostic_case"] = "E"
    result["phase4i_status_note"] = (
        "Phase 4I remains incomplete even if this 0.25 mm Trial passes."
    )
    return result


def common_settings(mesh_level: MeshLevel) -> dict[str, Any]:
    """Serialize the settings held invariant across isolation cases."""
    return {
        "mesh_level": mesh_level,
        "geometry": "default zero-clearance FingertipModel",
        "first_step_travel_mm": FIRST_STEP_TRAVEL_MM,
        "trial_total_indentation_mm": 0.25,
        "trial_step_count": 48,
        "element": MIXED_PAD_ELEMENT,
        "constitutive_law": CONSTITUTIVE_LAW,
        "young_modulus_mpa": YOUNG_MODULUS_MPA,
        "poisson_ratio": POISSON_RATIO,
        "thickness_mm": THICKNESS_MM,
        "mortar_type": MORTAR_TYPE,
        "contact_process": "ALMContactProcess",
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "maximum_newton_iterations": MAXIMUM_NEWTON_ITERATIONS,
        "linear_solver": "skyline_lu_factorization",
        "parameter_tuning": "none",
    }
