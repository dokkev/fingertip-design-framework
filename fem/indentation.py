"""Phase 4I central indentation using the Phase 4M mesh and Kratos stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from fem.indenter_fixture import (
    IndenterBoundaryEdge,
    IndenterFixture,
    IndenterMesh,
    IndenterSettings,
    build_indenter_fixture,
    generate_indenter_mesh,
)
from fem.internal_contact_configuration import (
    create_continuous_u_submodel_parts,
)
from fem.indentation_postprocess import (
    compressive_indenter_reaction,
    contact_width_metrics,
    extract_outer_arc_profile,
    pad_strain_det_f_statistics,
    relative_force_equilibrium_error,
    signed_geometric_gap_statistics,
    unique_projected_reaction,
    unstructured_volumetric_oscillation,
)
from fem.fingertip_mesher import generate_fingertip_mesh
from fem.kratos_adapter import (
    KratosAdapterError,
    KratosTopology,
    _import_kratos,
    apply_initialization_constraints,
    populate_kratos_model_part,
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
from fem.mesh_types import BoundaryEdge, FingertipMesh, MeshLevel, mesh_settings_for_level
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


@dataclass(frozen=True)
class IndenterKratosTopology:
    """Global Kratos IDs assigned to the separately generated fixture mesh."""

    node_ids: tuple[int, ...]
    element_ids: tuple[int, ...]
    contact_condition_ids: tuple[int, ...]
    remainder_condition_ids: tuple[int, ...]
    contact_node_ids: tuple[int, ...]
    local_to_global_node_id: dict[int, int]
    contact_edges: tuple[BoundaryEdge, ...]
    remainder_edges: tuple[BoundaryEdge, ...]


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


def _next_id(entities: Any) -> int:
    identifiers = [entity.Id for entity in entities]
    return max(identifiers, default=0) + 1


def _add_submodel_part(
    model_part: Any,
    name: str,
    node_ids: Sequence[int],
    element_ids: Sequence[int] = (),
    condition_ids: Sequence[int] = (),
) -> Any:
    submodel_part = model_part.CreateSubModelPart(name)
    if node_ids:
        submodel_part.AddNodes(list(node_ids))
    if element_ids:
        submodel_part.AddElements(list(element_ids))
    if condition_ids:
        submodel_part.AddConditions(list(condition_ids))
    return submodel_part


def populate_indenter_model_part(
    model_part: Any,
    indenter_mesh: IndenterMesh,
    base_topology: KratosTopology,
) -> IndenterKratosTopology:
    """Append the independent carrier with globally disjoint node IDs."""
    KM, _, CLA, _ = _import_kratos()
    properties_id = 3
    properties = (
        model_part.Properties[properties_id]
        if model_part.HasProperties(properties_id)
        else model_part.CreateNewProperties(properties_id)
    )
    properties[KM.YOUNG_MODULUS] = YOUNG_MODULUS_MPA
    properties[KM.POISSON_RATIO] = POISSON_RATIO
    properties[KM.THICKNESS] = THICKNESS_MM
    properties[KM.DENSITY] = 1.0
    properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElasticPlaneStrain2DLaw()

    first_node_id = _next_id(model_part.Nodes)
    node_map = {
        local_id: first_node_id + index
        for index, local_id in enumerate(sorted(indenter_mesh.nodes))
    }
    for local_id in sorted(indenter_mesh.nodes):
        node = indenter_mesh.nodes[local_id]
        model_part.CreateNewNode(node_map[local_id], node.x_mm, node.y_mm, 0.0)

    first_element_id = _next_id(model_part.Elements)
    element_ids: list[int] = []
    for index, element in enumerate(indenter_mesh.elements):
        element_id = first_element_id + index
        model_part.CreateNewElement(
            CARRIER_ELEMENT,
            element_id,
            [node_map[node_id] for node_id in element.node_ids],
            properties,
        )
        element_ids.append(element_id)

    first_condition_id = _next_id(model_part.Conditions)
    contact_condition_ids: list[int] = []
    remainder_condition_ids: list[int] = []
    for edge_group, identifiers in (
        (indenter_mesh.contact_edges, contact_condition_ids),
        (indenter_mesh.remainder_edges, remainder_condition_ids),
    ):
        for edge in edge_group:
            condition_id = first_condition_id + len(contact_condition_ids) + len(remainder_condition_ids)
            model_part.CreateNewCondition(
                "LineCondition2D2N",
                condition_id,
                [node_map[node_id] for node_id in edge.node_ids],
                properties,
            )
            identifiers.append(condition_id)

    node_ids = tuple(sorted(node_map.values()))
    contact_node_ids = tuple(sorted(node_map[node_id] for node_id in indenter_mesh.contact_node_ids))
    contact_edges = tuple(
        BoundaryEdge(tuple(node_map[node_id] for node_id in edge.node_ids), "rigid_carrier")
        for edge in indenter_mesh.contact_edges
    )
    remainder_edges = tuple(
        BoundaryEdge(tuple(node_map[node_id] for node_id in edge.node_ids), "rigid_carrier")
        for edge in indenter_mesh.remainder_edges
    )
    _add_submodel_part(model_part, "RigidIndenter", node_ids, element_ids)
    _add_submodel_part(
        model_part,
        "IndenterContactArc",
        contact_node_ids,
        condition_ids=contact_condition_ids,
    )
    remainder_node_ids = tuple(
        sorted({node_id for edge in remainder_edges for node_id in edge.node_ids})
    )
    _add_submodel_part(
        model_part,
        "IndenterOuterRemainder",
        remainder_node_ids,
        condition_ids=remainder_condition_ids,
    )
    _add_submodel_part(model_part, "IndenterRigidMotion", node_ids)
    pad_outer = model_part.GetSubModelPart("PadOuterArc")
    external_nodes = tuple(
        sorted({node.Id for node in pad_outer.Nodes}.union(contact_node_ids))
    )
    external_conditions = tuple(
        sorted(
            {condition.Id for condition in pad_outer.Conditions}.union(
                contact_condition_ids
            )
        )
    )
    _add_submodel_part(
        model_part,
        "ExternalContact",
        external_nodes,
        condition_ids=external_conditions,
    )
    if not set(node_ids).isdisjoint(
        set(base_topology.pad_node_ids).union(base_topology.carrier_node_ids)
    ):
        raise KratosAdapterError("pad/link and indenter node IDs are not disjoint")
    return IndenterKratosTopology(
        node_ids=node_ids,
        element_ids=tuple(element_ids),
        contact_condition_ids=tuple(contact_condition_ids),
        remainder_condition_ids=tuple(remainder_condition_ids),
        contact_node_ids=contact_node_ids,
        local_to_global_node_id=node_map,
        contact_edges=contact_edges,
        remainder_edges=remainder_edges,
    )


def apply_indentation_constraints(
    model_part: Any,
    base_topology: KratosTopology,
    indenter_topology: IndenterKratosTopology,
) -> None:
    """Fix the Phase 4M supports and constrain the indenter to translation."""
    KM, _, _, _ = _import_kratos()
    apply_initialization_constraints(model_part, base_topology)
    for node_id in indenter_topology.node_ids:
        node = model_part.Nodes[node_id]
        for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)


def set_indenter_travel(
    model_part: Any,
    node_ids: Sequence[int],
    fixture: IndenterFixture,
    travel_mm: float,
) -> None:
    """Apply one common translation before contact search and prediction."""
    KM, _, _, _ = _import_kratos()
    displacement = fixture.displacement_for_travel(travel_mm)
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, displacement[0])
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displacement[1])
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)
        node.X = node.X0 + displacement[0]
        node.Y = node.Y0 + displacement[1]
        node.Z = node.Z0


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "finite": bool(values) and all(math.isfinite(value) for value in values),
    }


def _failure_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Summarize a failed iterate without serializing NaN or infinity."""
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(values),
        "finite_count": len(finite_values),
        "nonfinite_count": len(values) - len(finite_values),
        "min_finite": min(finite_values) if finite_values else None,
        "max_finite": max(finite_values) if finite_values else None,
        "mean_finite": (
            sum(finite_values) / len(finite_values) if finite_values else None
        ),
        "all_finite": len(finite_values) == len(values),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _failed_contact_group_diagnostics(
    model: Any,
    model_part: Any,
    contact_groups: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Capture indexed ALM state without geometric operations on a NaN iterate."""
    KM, CSMA, _, _ = _import_kratos()
    groups: dict[str, Any] = {}
    for index, (group_name, slave_name, _) in enumerate(contact_groups):
        slave = model_part.GetSubModelPart(slave_name)
        computing = model[
            f"Structure.ComputingContact.ComputingContactSub{index}"
        ]
        conditions = list(computing.Conditions)
        groups[group_name] = {
            "pair_index": index,
            "generated_condition_count": len(conditions),
            "active_condition_count": sum(
                condition.Is(KM.ACTIVE) for condition in conditions
            ),
            "active_condition_ids": sorted(
                condition.Id for condition in conditions if condition.Is(KM.ACTIVE)
            ),
            "active_slave_node_ids": sorted(
                node.Id for node in slave.Nodes if node.Is(KM.ACTIVE)
            ),
            "weighted_gap": _failure_statistics(
                [
                    float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP))
                    for node in slave.Nodes
                ]
            ),
            "lagrange_multiplier_contact_pressure": _failure_statistics(
                [
                    float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    )
                    for node in slave.Nodes
                ]
            ),
            "slave_nodal_state": [
                {
                    "node_id": node.Id,
                    "active": bool(node.Is(KM.ACTIVE)),
                    "weighted_gap": _finite_or_none(
                        node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)
                    ),
                    "lagrange_multiplier_contact_pressure": _finite_or_none(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    ),
                    "normal": [
                        _finite_or_none(
                            node.GetSolutionStepValue(KM.NORMAL)[component]
                        )
                        for component in range(2)
                    ],
                }
                for node in slave.Nodes
            ],
        }
    return groups


def _runtime_contact_contract(
    model: Any,
    model_part: Any,
    contact_groups: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Verify indexed contact pairs using Kratos-owned submodel parts."""
    KM, _, _, _ = _import_kratos()
    groups: dict[str, Any] = {}
    all_checks: list[bool] = []
    for index, (group_name, slave_name, master_name) in enumerate(
        contact_groups
    ):
        slave = model_part.GetSubModelPart(slave_name)
        master = model_part.GetSubModelPart(master_name)
        contact_path = f"Structure.Contact.ContactSub{index}"
        computing_path = f"Structure.ComputingContact.ComputingContactSub{index}"
        if not model.HasModelPart(contact_path) or not model.HasModelPart(computing_path):
            raise KratosAdapterError(
                f"Kratos did not create indexed contact model parts for {group_name}"
            )
        contact_subpart = model[contact_path]
        source_condition_ids = {
            condition.Id for condition in slave.Conditions
        }.union(condition.Id for condition in master.Conditions)
        runtime_condition_ids = {condition.Id for condition in contact_subpart.Conditions}
        source_connectivity = {
            tuple(sorted(node.Id for node in condition.GetGeometry()))
            for condition in (*list(slave.Conditions), *list(master.Conditions))
        }
        runtime_connectivity = {
            tuple(sorted(node.Id for node in condition.GetGeometry()))
            for condition in contact_subpart.Conditions
        }
        slave_role = all(node.Is(KM.SLAVE) and not node.Is(KM.MASTER) for node in slave.Nodes)
        master_role = all(node.Is(KM.MASTER) and not node.Is(KM.SLAVE) for node in master.Nodes)
        pair_membership = source_connectivity == runtime_connectivity
        nodal_h_slave = [float(node.GetSolutionStepValue(KM.NODAL_H)) for node in slave.Nodes]
        nodal_h_master = [float(node.GetSolutionStepValue(KM.NODAL_H)) for node in master.Nodes]
        group_checks = {
            "slave_runtime_role": slave_role,
            "master_runtime_role": master_role,
            "contact_submodelpart_matches_exact_pair": pair_membership,
            "slave_nodal_h_positive_finite": bool(nodal_h_slave)
            and all(math.isfinite(value) and value > 0.0 for value in nodal_h_slave),
            "master_nodal_h_positive_finite": bool(nodal_h_master)
            and all(math.isfinite(value) and value > 0.0 for value in nodal_h_master),
        }
        all_checks.extend(group_checks.values())
        groups[group_name] = {
            "pair_index": index,
            "slave": slave_name,
            "master": master_name,
            "contact_submodelpart": contact_path,
            "computing_contact_submodelpart": computing_path,
            "source_condition_ids": sorted(source_condition_ids),
            "contact_submodelpart_condition_ids": sorted(runtime_condition_ids),
            "source_and_runtime_condition_ids_equal": (
                source_condition_ids == runtime_condition_ids
            ),
            "condition_identity_contract": (
                "exact connectivity and node IDs; ALMContactProcess may "
                "renumber generated interface conditions when pair indexes "
                "are not the original three-pair ordering"
            ),
            "slave_node_count": slave.NumberOfNodes(),
            "master_node_count": master.NumberOfNodes(),
            "slave_node_flags": {
                "SLAVE": sum(node.Is(KM.SLAVE) for node in slave.Nodes),
                "MASTER": sum(node.Is(KM.MASTER) for node in slave.Nodes),
            },
            "master_node_flags": {
                "SLAVE": sum(node.Is(KM.SLAVE) for node in master.Nodes),
                "MASTER": sum(node.Is(KM.MASTER) for node in master.Nodes),
            },
            "slave_condition_flags": {
                "SLAVE": sum(condition.Is(KM.SLAVE) for condition in slave.Conditions),
                "MASTER": sum(condition.Is(KM.MASTER) for condition in slave.Conditions),
            },
            "master_condition_flags": {
                "SLAVE": sum(condition.Is(KM.SLAVE) for condition in master.Conditions),
                "MASTER": sum(condition.Is(KM.MASTER) for condition in master.Conditions),
            },
            "slave_mean_runtime_normal": [
                sum(float(node.GetSolutionStepValue(KM.NORMAL)[component]) for node in slave.Nodes)
                / max(slave.NumberOfNodes(), 1)
                for component in range(2)
            ],
            "master_mean_runtime_normal": [
                sum(float(node.GetSolutionStepValue(KM.NORMAL)[component]) for node in master.Nodes)
                / max(master.NumberOfNodes(), 1)
                for component in range(2)
            ],
            "slave_nodal_h": _statistics(nodal_h_slave),
            "master_nodal_h": _statistics(nodal_h_master),
            "checks": group_checks,
        }
    return {
        "group_identification_method": (
            "Kratos ContactSubN/ComputingContactSubN API; no coordinate inference"
        ),
        "groups": groups,
        "all_group_contracts_pass": all(all_checks),
        "initial_penalty": float(model_part.ProcessInfo[KM.INITIAL_PENALTY]),
        "scale_factor": float(model_part.ProcessInfo[KM.SCALE_FACTOR]),
    }


def inspect_indentation_runtime_contract(
    fingertip_model: FingertipModel,
    mesh_level: MeshLevel,
    settings: IndentationSettings,
    indenter_settings: IndenterSettings | None = None,
    internal_contact_configuration: str = "three_pairs",
) -> dict[str, Any]:
    """Initialize and inspect Phase 4I without entering a nonlinear step."""
    KM, CSMA, _, _ = _import_kratos()
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
        runtime = _runtime_contact_contract(
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


def _nodal_fields(model_part: Any, node_ids: Sequence[int]) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    KM, _, _, _ = _import_kratos()
    displacements: dict[int, tuple[float, float]] = {}
    reactions: dict[int, tuple[float, float]] = {}
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        reaction = node.GetSolutionStepValue(KM.REACTION)
        displacements[node_id] = (float(displacement[0]), float(displacement[1]))
        reactions[node_id] = (float(reaction[0]), float(reaction[1]))
    return displacements, reactions


def _finite_field_failures(model_part: Any, pad_node_ids: Sequence[int]) -> list[str]:
    KM, CSMA, _, _ = _import_kratos()
    failures: list[str] = []
    pad_ids = set(pad_node_ids)
    for node in model_part.Nodes:
        displacement = node.GetSolutionStepValue(KM.DISPLACEMENT)
        reaction = node.GetSolutionStepValue(KM.REACTION)
        if not all(math.isfinite(float(displacement[index])) for index in range(2)):
            failures.append(f"node_{node.Id}_DISPLACEMENT")
        if not all(math.isfinite(float(reaction[index])) for index in range(2)):
            failures.append(f"node_{node.Id}_REACTION")
        if node.Id in pad_ids and not math.isfinite(
            float(node.GetSolutionStepValue(KM.VOLUMETRIC_STRAIN))
        ):
            failures.append(f"node_{node.Id}_VOLUMETRIC_STRAIN")
        if node.Is(KM.SLAVE):
            for variable in (
                CSMA.WEIGHTED_GAP,
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
            ):
                if node.SolutionStepsDataHas(variable) and not math.isfinite(
                    float(node.GetSolutionStepValue(variable))
                ):
                    failures.append(f"node_{node.Id}_{variable.Name()}")
    return failures


def _edge_positions(model_part: Any, edges: Sequence[BoundaryEdge]) -> dict[int, tuple[float, float]]:
    ids = {node_id for edge in edges for node_id in edge.node_ids}
    return {node_id: (float(model_part.Nodes[node_id].X), float(model_part.Nodes[node_id].Y)) for node_id in ids}


def _master_edges_for_group(
    group_name: str,
    mesh: FingertipMesh,
    indenter_topology: IndenterKratosTopology,
) -> tuple[BoundaryEdge, ...]:
    if group_name == "external_pad_indenter":
        return indenter_topology.contact_edges
    tags = {
        "internal_left": "stem_left",
        "internal_right": "stem_right",
        "internal_bottom": "stem_bottom",
    }
    if group_name == "internal_u":
        return tuple(
            edge
            for tag in ("stem_left", "stem_bottom", "stem_right")
            for edge in mesh.boundary_edges[tag]
        )
    return mesh.boundary_edges[tags[group_name]]


def _contact_group_step_metrics(
    model: Any,
    model_part: Any,
    mesh: FingertipMesh,
    indenter_topology: IndenterKratosTopology,
    contact_group_definitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    KM, CSMA, _, _ = _import_kratos()
    groups: dict[str, Any] = {}
    for index, (group_name, slave_name, _) in enumerate(
        contact_group_definitions
    ):
        slave = model_part.GetSubModelPart(slave_name)
        computing_path = f"Structure.ComputingContact.ComputingContactSub{index}"
        computing = model[computing_path]
        conditions = list(computing.Conditions)
        active_conditions = [condition for condition in conditions if condition.Is(KM.ACTIVE)]
        active_slave_ids = sorted(node.Id for node in slave.Nodes if node.Is(KM.ACTIVE))
        weighted_gaps = [float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)) for node in slave.Nodes]
        pressures = [
            float(node.GetSolutionStepValue(CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE))
            for node in slave.Nodes
        ]
        normals = {
            node.Id: (
                float(node.GetSolutionStepValue(KM.NORMAL)[0]),
                float(node.GetSolutionStepValue(KM.NORMAL)[1]),
            )
            for node in slave.Nodes
        }
        geometric_gap_node_ids = (
            set(active_slave_ids)
            if group_name == "external_pad_indenter"
            else {node.Id for node in slave.Nodes}
        )
        slave_positions = {
            node.Id: (float(node.X), float(node.Y))
            for node in slave.Nodes
            if node.Id in geometric_gap_node_ids
        }
        gap_normals = {
            node_id: normal
            for node_id, normal in normals.items()
            if node_id in geometric_gap_node_ids
        }
        master_edges = _master_edges_for_group(group_name, mesh, indenter_topology)
        geometric_gap = signed_geometric_gap_statistics(
            slave_positions,
            gap_normals,
            master_edges,
            _edge_positions(model_part, master_edges),
        )
        nodal_h = [float(node.GetSolutionStepValue(KM.NODAL_H)) for node in slave.Nodes]
        local_size = sum(nodal_h) / len(nodal_h)
        penetration_tolerance = max(0.001, 0.05 * local_size)
        groups[group_name] = {
            "pair_index": index,
            "computing_contact_submodelpart": computing_path,
            "generated_condition_count": len(conditions),
            "active_condition_count": len(active_conditions),
            "active_condition_ids": sorted(condition.Id for condition in active_conditions),
            "active_slave_node_ids": active_slave_ids,
            "weighted_gap": _statistics(weighted_gaps),
            "lagrange_multiplier_contact_pressure": _statistics(pressures),
            "signed_geometric_gap": geometric_gap,
            "geometric_gap_node_scope": (
                "active slave nodes"
                if group_name == "external_pad_indenter"
                else "all source slave nodes"
            ),
            "local_contact_mesh_size_mm": local_size,
            "penetration_tolerance_mm": penetration_tolerance,
            "penetration_pass": bool(geometric_gap.get("available"))
            and bool(geometric_gap.get("finite"))
            and float(geometric_gap["maximum_penetration_mm"]) <= penetration_tolerance,
            "slave_nodal_state": [
                {
                    "node_id": node.Id,
                    "active": bool(node.Is(KM.ACTIVE)),
                    "weighted_gap": float(
                        node.GetSolutionStepValue(CSMA.WEIGHTED_GAP)
                    ),
                    "lagrange_multiplier_contact_pressure": float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    ),
                    "normal": [
                        float(
                            node.GetSolutionStepValue(KM.NORMAL)[component]
                        )
                        for component in range(2)
                    ],
                }
                for node in slave.Nodes
            ],
        }
        if group_name == "internal_u":
            semantic_regions: dict[str, Any] = {}
            for region, submodel_part_name in (
                ("left", "PadCutoutLeft"),
                ("bottom", "PadCutoutBottom"),
                ("right", "PadCutoutRight"),
            ):
                semantic = model_part.GetSubModelPart(submodel_part_name)
                region_gaps = [
                    float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP))
                    for node in semantic.Nodes
                ]
                region_pressures = [
                    float(
                        node.GetSolutionStepValue(
                            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                        )
                    )
                    for node in semantic.Nodes
                ]
                semantic_regions[region] = {
                    "source_submodelpart": submodel_part_name,
                    "node_count": semantic.NumberOfNodes(),
                    "active_node_ids": sorted(
                        node.Id for node in semantic.Nodes if node.Is(KM.ACTIVE)
                    ),
                    "active_node_count": sum(
                        node.Is(KM.ACTIVE) for node in semantic.Nodes
                    ),
                    "weighted_gap": _statistics(region_gaps),
                    "lagrange_multiplier_contact_pressure": _statistics(
                        region_pressures
                    ),
                }
            groups[group_name]["semantic_regions"] = semantic_regions
    return groups


def _rigid_domain_validation(
    model_part: Any,
    node_ids: Sequence[int],
    element_ids: Sequence[int],
    prescribed_displacement: Sequence[float],
) -> dict[str, Any]:
    KM, _, _, _ = _import_kratos()
    maximum_translation_error = 0.0
    for node_id in node_ids:
        displacement = model_part.Nodes[node_id].GetSolutionStepValue(KM.DISPLACEMENT)
        maximum_translation_error = max(
            maximum_translation_error,
            math.hypot(
                float(displacement[0]) - float(prescribed_displacement[0]),
                float(displacement[1]) - float(prescribed_displacement[1]),
            ),
        )
    maximum_strain = 0.0
    maximum_det_error = 0.0
    nonpositive_det_count = 0
    for element_id in element_ids:
        geometry = model_part.Elements[element_id].GetGeometry()
        reference = [
            (float(node.X0), float(node.Y0)) for node in geometry
        ]
        current = [(float(node.X), float(node.Y)) for node in geometry]
        reference_edges = (
            (reference[1][0] - reference[0][0], reference[2][0] - reference[0][0]),
            (reference[1][1] - reference[0][1], reference[2][1] - reference[0][1]),
        )
        current_edges = (
            (current[1][0] - current[0][0], current[2][0] - current[0][0]),
            (current[1][1] - current[0][1], current[2][1] - current[0][1]),
        )
        determinant_reference = (
            reference_edges[0][0] * reference_edges[1][1]
            - reference_edges[0][1] * reference_edges[1][0]
        )
        inverse = (
            (reference_edges[1][1] / determinant_reference, -reference_edges[0][1] / determinant_reference),
            (-reference_edges[1][0] / determinant_reference, reference_edges[0][0] / determinant_reference),
        )
        deformation_gradient = (
            (
                current_edges[0][0] * inverse[0][0] + current_edges[0][1] * inverse[1][0],
                current_edges[0][0] * inverse[0][1] + current_edges[0][1] * inverse[1][1],
            ),
            (
                current_edges[1][0] * inverse[0][0] + current_edges[1][1] * inverse[1][0],
                current_edges[1][0] * inverse[0][1] + current_edges[1][1] * inverse[1][1],
            ),
        )
        determinant_f = (
            deformation_gradient[0][0] * deformation_gradient[1][1]
            - deformation_gradient[0][1] * deformation_gradient[1][0]
        )
        c00 = deformation_gradient[0][0] ** 2 + deformation_gradient[1][0] ** 2
        c01 = deformation_gradient[0][0] * deformation_gradient[0][1] + deformation_gradient[1][0] * deformation_gradient[1][1]
        c11 = deformation_gradient[0][1] ** 2 + deformation_gradient[1][1] ** 2
        strain_norm = math.sqrt((0.5 * (c00 - 1.0)) ** 2 + 2.0 * (0.5 * c01) ** 2 + (0.5 * (c11 - 1.0)) ** 2)
        maximum_strain = max(maximum_strain, strain_norm)
        maximum_det_error = max(maximum_det_error, abs(determinant_f - 1.0))
        nonpositive_det_count += determinant_f <= 0.0
    tolerance = 1.0e-10
    return {
        "node_count": len(node_ids),
        "element_count": len(element_ids),
        "maximum_translation_error_mm": maximum_translation_error,
        "maximum_green_lagrange_strain_norm": maximum_strain,
        "maximum_abs_det_f_error_from_one": maximum_det_error,
        "nonpositive_det_f_count": nonpositive_det_count,
        "numerical_tolerance": tolerance,
        "strain_energy_interpretation": "negligible because F=I and strain is numerical zero",
        "pass": maximum_translation_error <= tolerance
        and maximum_strain <= tolerance
        and maximum_det_error <= tolerance
        and nonpositive_det_count == 0,
    }


def _curve_acceptance(curve: Sequence[Mapping[str, Any]], force_tolerance_n: float) -> dict[str, Any]:
    active_points = [
        point
        for point in curve
        if point["contact_groups"]["external_pad_indenter"]["active_condition_count"] > 0
        and point["indenter_normal_reaction_n"] > force_tolerance_n
    ]
    if len(active_points) < 2:
        return {
            "first_load_bearing_step": None,
            "monotonic": False,
            "smooth": False,
            "reason": "fewer than two load-bearing contact steps",
        }
    reactions = [float(point["indenter_normal_reaction_n"]) for point in active_points]
    allowed_decrease = max(0.02 * reactions[-1], force_tolerance_n)
    monotonic = all(
        second >= first - allowed_decrease
        for first, second in zip(reactions, reactions[1:])
    )
    second_differences = [
        abs(reactions[index + 1] - 2.0 * reactions[index] + reactions[index - 1])
        for index in range(1, len(reactions) - 1)
    ]
    normalized_second = max(second_differences, default=0.0) / max(
        reactions[-1], force_tolerance_n
    )
    return {
        "first_load_bearing_step": int(active_points[0]["step"]),
        "allowed_reaction_decrease_n": allowed_decrease,
        "monotonic": monotonic,
        "normalized_max_second_difference": normalized_second,
        "smoothness_limit": 0.15,
        "smooth": monotonic and normalized_second <= 0.15,
    }


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
    KM, CSMA, _, _ = _import_kratos()
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
        runtime_contact = _runtime_contact_contract(
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
                _, failed_reactions = _nodal_fields(model_part, all_node_ids)
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
                    "finite_field_failures": _finite_field_failures(
                        model_part, base_topology.pad_node_ids
                    )[:100],
                    "reaction_components": _failure_statistics(reaction_values),
                    "contact_groups": _failed_contact_group_diagnostics(
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

            displacements, reactions = _nodal_fields(model_part, all_node_ids)
            field_failures = _finite_field_failures(
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
                **_statistics(list(volumetric_values.values())),
                "max_abs": max(abs(value) for value in volumetric_values.values()),
            }
            volumetric_oscillation = unstructured_volumetric_oscillation(
                mesh, volumetric_values
            )
            contact_groups = _contact_group_step_metrics(
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
            rigid_validation = _rigid_domain_validation(
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
        curve_acceptance = _curve_acceptance(
            result["history"], settings.numerical_force_tolerance_n
        )
        result["curve_diagnostics"] = curve_acceptance
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
            "force_curve_smooth_and_monotonic": curve_acceptance.get("smooth", False),
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
