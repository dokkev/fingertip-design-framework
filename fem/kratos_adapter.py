"""Adapt a validated ``FingertipMesh`` to the Phase 3R Kratos stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from fem.kratos_settings import (
    CARRIER_ELEMENT,
    CONSTITUTIVE_LAW,
    CONTACT_GROUPS,
    MIXED_PAD_ELEMENT,
    POISSON_RATIO,
    THICKNESS_MM,
    YOUNG_MODULUS_MPA,
    build_project_parameters_json,
)
from fem.mesh_types import BoundaryEdge, FingertipMesh


class KratosDependencyError(RuntimeError):
    """Raised when Phase 4M is run without the validated Kratos applications."""


class KratosAdapterError(RuntimeError):
    """Raised when mesh or runtime topology violates the Kratos contract."""


BOUNDARY_MODEL_PART_NAMES = {
    "pad_bond_left": "PadBondLeft",
    "pad_bond_right": "PadBondRight",
    "pad_outer_arc": "PadOuterArc",
    "pad_cutout_left": "PadCutoutLeft",
    "pad_cutout_right": "PadCutoutRight",
    "pad_cutout_bottom": "PadCutoutBottom",
    "stem_left": "StemLeft",
    "stem_right": "StemRight",
    "stem_bottom": "StemBottom",
    "pad_void_unpaired": "PadVoidUnpaired",
    "rigid_link_outer": "RigidLinkOuter",
    "rigid_bond_interface": "RigidBondInterface",
}


@dataclass(frozen=True)
class KratosTopology:
    """IDs created by the adapter for runtime inspection."""

    boundary_condition_ids: dict[str, tuple[int, ...]]
    pad_element_ids: tuple[int, ...]
    carrier_element_ids: tuple[int, ...]
    pad_node_ids: tuple[int, ...]
    carrier_node_ids: tuple[int, ...]


def _import_kratos() -> tuple[Any, Any, Any, Any]:
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
        import KratosMultiphysics.ConstitutiveLawsApplication as CLA
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
    except (ImportError, OSError) as exception:
        raise KratosDependencyError(
            "Phase 4M Kratos smoke testing requires KratosMultiphysics with "
            "StructuralMechanics, ConstitutiveLaws, and "
            "ContactStructuralMechanics applications."
        ) from exception
    return KM, CSMA, CLA, SMA


def _properties(model_part: Any, identifier: int) -> Any:
    if model_part.HasProperties(identifier):
        return model_part.Properties[identifier]
    return model_part.CreateNewProperties(identifier)


def _configure_properties(model_part: Any, KM: Any, CLA: Any) -> tuple[Any, Any]:
    pad_properties = _properties(model_part, 1)
    carrier_properties = _properties(model_part, 2)
    for properties in (pad_properties, carrier_properties):
        properties[KM.YOUNG_MODULUS] = YOUNG_MODULUS_MPA
        properties[KM.POISSON_RATIO] = POISSON_RATIO
        properties[KM.THICKNESS] = THICKNESS_MM
        properties[KM.DENSITY] = 1.0
        properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
        properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElasticPlaneStrain2DLaw()
    return pad_properties, carrier_properties


def _add_submodel_part(
    model_part: Any,
    name: str,
    node_ids: tuple[int, ...],
    element_ids: tuple[int, ...] = (),
    condition_ids: tuple[int, ...] = (),
) -> Any:
    submodel_part = model_part.CreateSubModelPart(name)
    if node_ids:
        submodel_part.AddNodes(list(node_ids))
    if element_ids:
        submodel_part.AddElements(list(element_ids))
    if condition_ids:
        submodel_part.AddConditions(list(condition_ids))
    return submodel_part


def populate_kratos_model_part(
    model_part: Any, mesh: FingertipMesh
) -> KratosTopology:
    """Create nodes, T3 elements, Line2 conditions, and semantic submodel parts."""
    if not mesh.validation.passed:
        raise KratosAdapterError(
            "refusing to adapt an invalid mesh: " + ", ".join(mesh.validation.errors)
        )
    KM, _, CLA, _ = _import_kratos()
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 2
    pad_properties, carrier_properties = _configure_properties(
        model_part, KM, CLA
    )
    for node in sorted(mesh.nodes.values(), key=lambda item: item.id):
        model_part.CreateNewNode(node.id, node.x_mm, node.y_mm, 0.0)
    for element in mesh.pad_elements:
        model_part.CreateNewElement(
            MIXED_PAD_ELEMENT,
            element.id,
            list(element.node_ids),
            pad_properties,
        )
    for element in mesh.carrier_elements:
        # Same purpose as the Phase 3R carrier: this standard TL solid only
        # supplies constrained rigid topology and finite characteristic length.
        model_part.CreateNewElement(
            CARRIER_ELEMENT,
            element.id,
            list(element.node_ids),
            carrier_properties,
        )

    condition_ids_by_tag: dict[str, tuple[int, ...]] = {}
    next_condition_id = 1
    for tag, edges in mesh.boundary_edges.items():
        property_object = (
            pad_properties if all(edge.domain == "pad" for edge in edges) else carrier_properties
        )
        condition_ids: list[int] = []
        for edge in edges:
            model_part.CreateNewCondition(
                "LineCondition2D2N",
                next_condition_id,
                list(edge.node_ids),
                property_object,
            )
            condition_ids.append(next_condition_id)
            next_condition_id += 1
        condition_ids_by_tag[tag] = tuple(condition_ids)

    pad_node_ids = mesh.domain_node_ids["pad"]
    carrier_node_ids = mesh.domain_node_ids["rigid_carrier"]
    pad_element_ids = mesh.domain_element_ids["pad"]
    carrier_element_ids = mesh.domain_element_ids["rigid_carrier"]
    _add_submodel_part(
        model_part, "PadDomain", pad_node_ids, pad_element_ids
    )
    _add_submodel_part(
        model_part,
        "RigidCarrier",
        carrier_node_ids,
        carrier_element_ids,
    )
    _add_submodel_part(model_part, "RigidMotion", carrier_node_ids)
    for tag, model_part_name in BOUNDARY_MODEL_PART_NAMES.items():
        edges = mesh.boundary_edges[tag]
        node_ids = tuple(
            sorted({node_id for edge in edges for node_id in edge.node_ids})
        )
        _add_submodel_part(
            model_part,
            model_part_name,
            node_ids,
            condition_ids=condition_ids_by_tag[tag],
        )
    return KratosTopology(
        boundary_condition_ids=condition_ids_by_tag,
        pad_element_ids=pad_element_ids,
        carrier_element_ids=carrier_element_ids,
        pad_node_ids=pad_node_ids,
        carrier_node_ids=carrier_node_ids,
    )


def apply_initialization_constraints(
    model_part: Any, topology: KratosTopology
) -> None:
    """Fix the pad bond and the entire kinematic carrier for the smoke model."""
    KM, _, _, _ = _import_kratos()
    for node in model_part.Nodes:
        node.SetSolutionStepValue(KM.VOLUMETRIC_STRAIN, 0.0)
    constrained_ids = set(topology.carrier_node_ids)
    for name in ("PadBondLeft", "PadBondRight"):
        constrained_ids.update(node.Id for node in model_part.GetSubModelPart(name).Nodes)
    for node_id in constrained_ids:
        node = model_part.Nodes[node_id]
        for variable in (
            KM.DISPLACEMENT_X,
            KM.DISPLACEMENT_Y,
            KM.DISPLACEMENT_Z,
        ):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)


def _coordinate_range(nodes: list[Any]) -> dict[str, Any]:
    return {
        "count": len(nodes),
        "x_mm": [min(node.X0 for node in nodes), max(node.X0 for node in nodes)],
        "y_mm": [min(node.Y0 for node in nodes), max(node.Y0 for node in nodes)],
    }


def _scalar_statistics(values: list[float]) -> dict[str, Any]:
    finite = bool(values) and all(math.isfinite(value) for value in values)
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "finite": finite,
        "positive": finite and all(value > 0.0 for value in values),
    }


def _normalized(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    return (vector[0] / length, vector[1] / length) if length else (0.0, 0.0)


def _mesh_edge_average_normal(
    edges: tuple[BoundaryEdge, ...], mesh: FingertipMesh
) -> tuple[float, float]:
    normal_x = 0.0
    normal_y = 0.0
    total_length = 0.0
    for edge in edges:
        first, second = (mesh.nodes[node_id] for node_id in edge.node_ids)
        dx = second.x_mm - first.x_mm
        dy = second.y_mm - first.y_mm
        length = math.hypot(dx, dy)
        normal_x += dy
        normal_y -= dx
        total_length += length
    if total_length == 0.0:
        return 0.0, 0.0
    return _normalized((normal_x / total_length, normal_y / total_length))


def _matching_contact_conditions(
    contact_part: Any, surface_node_ids: set[int]
) -> list[Any]:
    return [
        condition
        for condition in contact_part.Conditions
        if {node.Id for node in condition.GetGeometry()}.issubset(surface_node_ids)
    ]


def inspect_runtime_contact_contract(
    model: Any,
    model_part: Any,
    mesh: FingertipMesh,
) -> dict[str, Any]:
    """Inspect flags and fields after ``ALMContactProcess`` initialization."""
    KM, CSMA, _, _ = _import_kratos()
    if not model.HasModelPart("Structure.Contact"):
        raise KratosAdapterError("ALMContactProcess did not create Structure.Contact")
    contact_part = model["Structure.Contact"]
    computing_contact = (
        model["Structure.ComputingContact"]
        if model.HasModelPart("Structure.ComputingContact")
        else None
    )
    inverse_names = {value: key for key, value in BOUNDARY_MODEL_PART_NAMES.items()}
    surface_names = [name for group in CONTACT_GROUPS for name in group]
    surfaces: dict[str, Any] = {}
    for name in surface_names:
        submodel_part = model_part.GetSubModelPart(name)
        nodes = list(submodel_part.Nodes)
        node_ids = {node.Id for node in nodes}
        conditions = _matching_contact_conditions(contact_part, node_ids)
        nodal_h = [
            float(node.GetSolutionStepValue(KM.NODAL_H)) for node in nodes
        ]
        runtime_normal_values = [
            node.GetSolutionStepValue(KM.NORMAL) for node in nodes
        ]
        runtime_average = _normalized(
            (
                sum(float(normal[0]) for normal in runtime_normal_values),
                sum(float(normal[1]) for normal in runtime_normal_values),
            )
        )
        tag = inverse_names[name]
        expected_average = _mesh_edge_average_normal(
            mesh.boundary_edges[tag], mesh
        )
        normal_dot = sum(
            runtime_average[index] * expected_average[index] for index in range(2)
        )
        surfaces[name] = {
            "source_boundary_tag": tag,
            "node_count": len(nodes),
            "condition_count": len(conditions),
            "node_flags": {
                "MASTER": sum(node.Is(KM.MASTER) for node in nodes),
                "SLAVE": sum(node.Is(KM.SLAVE) for node in nodes),
            },
            "condition_flags": {
                "MASTER": sum(condition.Is(KM.MASTER) for condition in conditions),
                "SLAVE": sum(condition.Is(KM.SLAVE) for condition in conditions),
            },
            "condition_info": sorted({condition.Info() for condition in conditions}),
            "coordinates": _coordinate_range(nodes),
            "nodal_h": _scalar_statistics(nodal_h),
            "expected_outward_normal": list(expected_average),
            "runtime_nodal_normal_average": list(runtime_average),
            "runtime_expected_normal_dot": normal_dot,
        }

    zero_clearance_distinct = True
    for pair in mesh.contact_pairs:
        if pair.initial_normal_gap_mm <= mesh.settings.classification_tolerance_mm:
            pad_ids = {
                node_id
                for edge in mesh.boundary_edges[pair.pad_boundary_tag]
                for node_id in edge.node_ids
            }
            stem_ids = {
                node_id
                for edge in mesh.boundary_edges[pair.stem_boundary_tag]
                for node_id in edge.node_ids
            }
            zero_clearance_distinct = zero_clearance_distinct and pad_ids.isdisjoint(
                stem_ids
            )
    process_info = model_part.ProcessInfo
    pad_slave = all(
        surfaces[slave]["condition_flags"]["SLAVE"]
        == surfaces[slave]["condition_count"]
        and surfaces[slave]["condition_flags"]["MASTER"] == 0
        for slave, _ in CONTACT_GROUPS
    )
    stem_master = all(
        surfaces[master]["condition_flags"]["MASTER"]
        == surfaces[master]["condition_count"]
        and surfaces[master]["condition_flags"]["SLAVE"] == 0
        for _, master in CONTACT_GROUPS
    )
    nodal_h_valid = all(
        data["nodal_h"]["positive"] for data in surfaces.values()
    )
    normals_valid = all(
        data["runtime_expected_normal_dot"] > 0.7 for data in surfaces.values()
    )
    return {
        "surfaces": surfaces,
        "contact_model_part": {
            "condition_count": contact_part.NumberOfConditions(),
            "condition_info": sorted(
                {condition.Info().split(" #", 1)[0] for condition in contact_part.Conditions}
            ),
        },
        "computing_contact_model_part": {
            "exists": computing_contact is not None,
            "condition_count": (
                computing_contact.NumberOfConditions()
                if computing_contact is not None
                else 0
            ),
            "condition_info": (
                sorted(
                    {
                        condition.Info().split(" #", 1)[0]
                        for condition in computing_contact.Conditions
                    }
                )
                if computing_contact is not None
                else []
            ),
        },
        "initial_penalty": float(process_info[KM.INITIAL_PENALTY]),
        "scale_factor": float(process_info[KM.SCALE_FACTOR]),
        "zero_clearance_contact_node_ids_distinct": zero_clearance_distinct,
        "checks": {
            "all_pad_contact_conditions_are_slave": pad_slave,
            "all_stem_contact_conditions_are_master": stem_master,
            "all_contact_nodal_h_finite_and_positive": nodal_h_valid,
            "runtime_normals_match_mesh_outward_normals": normals_valid,
            "zero_clearance_contact_node_ids_distinct": zero_clearance_distinct,
        },
    }


def run_initialization_smoke(mesh: FingertipMesh) -> dict[str, Any]:
    """Initialize the mixed-solid/internal-ALM model without a nonlinear solve."""
    KM, _, _, _ = _import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
        StructuralMechanicsAnalysis,
    )

    model = KM.Model()
    parameters = KM.Parameters(build_project_parameters_json())
    analysis = StructuralMechanicsAnalysis(model, parameters)
    model_part = model["Structure"]
    topology = populate_kratos_model_part(model_part, mesh)
    initialized = False
    try:
        analysis.Initialize()
        initialized = True
        apply_initialization_constraints(model_part, topology)
        runtime = inspect_runtime_contact_contract(model, model_part, mesh)
        pad_element_info = sorted(
            {
                model_part.Elements[element_id].Info().split(" #", 1)[0]
                for element_id in topology.pad_element_ids
            }
        )
        carrier_element_info = sorted(
            {
                model_part.Elements[element_id].Info().split(" #", 1)[0]
                for element_id in topology.carrier_element_ids
            }
        )
        strategy_check = int(analysis._GetSolver()._GetSolutionStrategy().Check())
        pad_geometry_node_counts = sorted(
            {
                len(model_part.Elements[element_id].GetGeometry())
                for element_id in topology.pad_element_ids
            }
        )
        carrier_geometry_node_counts = sorted(
            {
                len(model_part.Elements[element_id].GetGeometry())
                for element_id in topology.carrier_element_ids
            }
        )
        submodel_parts = {
            name: {
                "nodes": model_part.GetSubModelPart(name).NumberOfNodes(),
                "elements": model_part.GetSubModelPart(name).NumberOfElements(),
                "conditions": model_part.GetSubModelPart(name).NumberOfConditions(),
            }
            for name in (
                "PadDomain",
                "RigidCarrier",
                "PadBondLeft",
                "PadBondRight",
                "PadOuterArc",
                "PadCutoutLeft",
                "PadCutoutRight",
                "PadCutoutBottom",
                "StemLeft",
                "StemRight",
                "StemBottom",
                "RigidMotion",
            )
        }
        element_checks = {
            "pad_element_is_mixed_t3": bool(topology.pad_element_ids)
            and pad_geometry_node_counts == [3]
            and strategy_check == 0,
            "carrier_element_is_standard_tl_t3": bool(
                topology.carrier_element_ids
            )
            and carrier_geometry_node_counts == [3]
            and strategy_check == 0,
        }
        acceptance_checks = {
            **element_checks,
            **runtime["checks"],
        }
        return {
            "status": "PASS" if all(acceptance_checks.values()) else "FAIL",
            "kratos_version": KM.Kernel.Version(),
            "initialization_succeeded": True,
            "mesh_level": mesh.settings.level,
            "material": {
                "element": MIXED_PAD_ELEMENT,
                "constitutive_law": CONSTITUTIVE_LAW,
                "plane_strain": True,
                "young_modulus_mpa": YOUNG_MODULUS_MPA,
                "young_modulus_role": "validation placeholder, not calibrated silicone",
                "poisson_ratio": POISSON_RATIO,
                "thickness_mm": THICKNESS_MM,
                "volumetric_strain_initial_value": 0.0,
            },
            "pad_element_info": pad_element_info,
            "carrier_element_info": carrier_element_info,
            "element_runtime_contract": {
                "pad_registered_creation_name": MIXED_PAD_ELEMENT,
                "pad_info": pad_element_info,
                "pad_geometry_node_counts": pad_geometry_node_counts,
                "carrier_registered_creation_name": CARRIER_ELEMENT,
                "carrier_info": carrier_element_info,
                "carrier_geometry_node_counts": carrier_geometry_node_counts,
                "strategy_check_return_value": strategy_check,
                "note": (
                    "Kratos Element.Info() reports the inherited C++ base "
                    "description; registered creation names are the explicit "
                    "CreateNewElement runtime inputs."
                ),
            },
            "topology": asdict(topology),
            "submodel_parts": submodel_parts,
            "runtime_contact_contract": runtime,
            "acceptance_checks": acceptance_checks,
        }
    finally:
        if initialized:
            analysis.Finalize()
