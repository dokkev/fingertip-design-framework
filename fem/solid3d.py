"""3D Kratos adapter for the authoritative compliant-pad volume mesh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import json
import math
import time
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from mesh.indenter import IndenterFixture
from mesh.volume_types import FingertipVolumeMesh, SurfaceTriangle


SolidFEAMode = Literal["3d_equivalent_reference", "production"]


class SolidFEAError(RuntimeError):
    """Raised when a 3D mechanics contract cannot be established."""


@dataclass(frozen=True)
class SolidFEASettings:
    """Explicit 3D mechanics mode and fidelity settings."""

    mode: SolidFEAMode = "production"
    number_of_steps: int = 12
    indentation_mm: float = 0.5
    reference_longitudinal_constraint: bool = False
    maximum_newton_iterations: int = 35
    external_contact: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("3d_equivalent_reference", "production"):
            raise ValueError("unsupported SolidFEA mode")
        if not isinstance(self.number_of_steps, int) or self.number_of_steps <= 0:
            raise ValueError("number_of_steps must be a positive integer")
        if not math.isfinite(self.indentation_mm) or self.indentation_mm <= 0.0:
            raise ValueError("indentation_mm must be finite and positive")
        if not isinstance(self.reference_longitudinal_constraint, bool):
            raise TypeError("reference_longitudinal_constraint must be bool")
        if not isinstance(self.maximum_newton_iterations, int) or self.maximum_newton_iterations <= 0:
            raise ValueError("maximum_newton_iterations must be positive")
        if not isinstance(self.external_contact, bool):
            raise TypeError("external_contact must be bool")


@dataclass(frozen=True)
class SolidFEAResult:
    """Neutral final state of one actual 3D volume mechanics solve."""

    volume_mesh: FingertipVolumeMesh
    reference_coordinates_mm: np.ndarray
    deformed_coordinates_mm: np.ndarray | None
    displacement_mm: np.ndarray | None
    reaction_force_n: float | None
    contact_state: dict[str, Any]
    configuration: dict[str, Any]
    converged: bool
    failure_message: str | None = None

    def __post_init__(self) -> None:
        reference = np.array(self.reference_coordinates_mm, dtype=float, copy=True)
        expected_shape = (len(self.volume_mesh.nodes), 3)
        if reference.shape != expected_shape or not np.all(np.isfinite(reference)):
            raise ValueError("reference_coordinates_mm must be finite with shape (N, 3)")
        reference.setflags(write=False)
        object.__setattr__(self, "reference_coordinates_mm", reference)
        if self.deformed_coordinates_mm is None or self.displacement_mm is None:
            if self.converged:
                raise ValueError("a converged solve requires deformed coordinates")
            return
        deformed = np.array(self.deformed_coordinates_mm, dtype=float, copy=True)
        displacement = np.array(self.displacement_mm, dtype=float, copy=True)
        if deformed.shape != expected_shape or displacement.shape != expected_shape:
            raise ValueError("3D FEA fields must have shape (N, 3)")
        if not np.all(np.isfinite(deformed)) or not np.all(np.isfinite(displacement)):
            raise ValueError("3D FEA fields must be finite")
        if not np.allclose(deformed - reference, displacement, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("deformed coordinates and displacement disagree")
        deformed.setflags(write=False)
        displacement.setflags(write=False)
        object.__setattr__(self, "deformed_coordinates_mm", deformed)
        object.__setattr__(self, "displacement_mm", displacement)


@dataclass(frozen=True)
class ContactSurfaceValidation:
    """Fail-closed geometry evidence for one mortar contact surface."""

    passed: bool
    triangle_count: int
    area_mm2: float
    normal_min_norm: float
    normal_max_norm: float
    connected_component_count: int
    boundary_edge_count: int
    orientation_conflict_count: int
    duplicate_triangle_count: int
    duplicate_condition_id_count: int
    checks: dict[str, bool]
    errors: tuple[str, ...]


def validate_contact_triangles(
    triangles: Sequence[tuple[int, tuple[int, int, int]] | SurfaceTriangle],
    coordinates: Mapping[int, Sequence[float]],
    *,
    expected_normal: Sequence[float] | None = None,
    contact_assignment: tuple[str, str] | None = None,
) -> ContactSurfaceValidation:
    """Validate every triangle before contact process initialization."""
    rows: list[tuple[int, tuple[int, int, int]]] = []
    for value in triangles:
        if isinstance(value, SurfaceTriangle):
            rows.append((value.id, value.node_ids))
        else:
            rows.append(value)
    duplicate_keys = [key for key, count in _counts(tuple(sorted(node_ids)) for _, node_ids in rows).items() if count > 1]
    duplicate_condition_ids = [key for key, count in _counts_1d(condition_id for condition_id, _ in rows).items() if count > 1]
    normals: list[np.ndarray] = []
    areas: list[float] = []
    invalid = 0
    finite_edges = True
    for _, node_ids in rows:
        if len(node_ids) != 3 or len(set(node_ids)) != 3 or any(node_id not in coordinates for node_id in node_ids):
            invalid += 1
            continue
        points = np.asarray([coordinates[node_id] for node_id in node_ids], dtype=float)
        if points.shape != (3, 3) or not np.all(np.isfinite(points)):
            invalid += 1
            continue
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        finite_edges &= bool(np.all(np.isfinite(points[1:] - points[0])))
        if not np.all(np.isfinite(normal)) or norm <= 1.0e-12:
            invalid += 1
            continue
        normals.append(normal)
        areas.append(0.5 * norm)
    edge_use: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(rows))}
    for index, (_, node_ids) in enumerate(rows):
        if len(node_ids) != 3 or len(set(node_ids)) != 3:
            continue
        directed = tuple((node_ids[i], node_ids[(i + 1) % 3]) for i in range(3))
        for first, second in directed:
            edge_use[tuple(sorted((first, second)))].append((index, 1 if first < second else -1))
    boundary_edges = 0
    orientation_conflicts = 0
    for uses in edge_use.values():
        if len(uses) == 1:
            boundary_edges += 1
        elif len(uses) == 2:
            if uses[0][1] == uses[1][1]:
                orientation_conflicts += 1
            adjacency[uses[0][0]].add(uses[1][0])
            adjacency[uses[1][0]].add(uses[0][0])
    components = 0
    unseen = set(adjacency)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbor in adjacency[stack.pop()] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
    normal_norms = [float(np.linalg.norm(normal)) for normal in normals]
    normal_direction_ok = True
    if expected_normal is not None and normals:
        direction = np.asarray(expected_normal, dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        normal_direction_ok = direction_norm > 0.0 and all(
            float(np.dot(normal, direction)) > 0.0 for normal in normals
        )
    assignment_ok = (
        contact_assignment is not None
        and len(contact_assignment) == 2
        and all(bool(name) for name in contact_assignment)
        and contact_assignment[0] != contact_assignment[1]
    )
    checks = {
        "finite_coordinates": invalid == 0,
        "finite_edge_vectors": finite_edges,
        "three_distinct_node_ids": invalid == 0,
        "nonzero_area": bool(areas) and min(areas) > 1.0e-12,
        "finite_nonzero_normals": bool(normals) and bool(np.isfinite(np.asarray(normals)).all()),
        "consistent_orientation": orientation_conflicts == 0 and normal_direction_ok,
        "no_duplicate_triangles": not duplicate_keys,
        "no_duplicate_conditions": not duplicate_condition_ids,
        "explicit_master_slave_assignment": assignment_ok,
    }
    errors = tuple(name for name, passed in checks.items() if not passed)
    return ContactSurfaceValidation(
        passed=not errors,
        triangle_count=len(rows),
        area_mm2=float(sum(areas)),
        normal_min_norm=min(normal_norms, default=0.0),
        normal_max_norm=max(normal_norms, default=0.0),
        connected_component_count=components,
        boundary_edge_count=boundary_edges,
        orientation_conflict_count=orientation_conflicts,
        duplicate_triangle_count=len(duplicate_keys),
        duplicate_condition_id_count=len(duplicate_condition_ids),
        checks=checks,
        errors=errors,
    )


def _counts(values: Iterable[tuple[int, int, int]]) -> dict[tuple[int, int, int], int]:
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def _counts_1d(values: Iterable[int]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def import_kratos() -> tuple[Any, Any, Any, Any]:
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
        import KratosMultiphysics.ConstitutiveLawsApplication as CLA
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
    except (ImportError, OSError) as exception:
        raise SolidFEAError("3D FEA requires Kratos Structural/Contact/Constitutive applications") from exception
    return KM, CSMA, CLA, SMA


def properties_for_model_part(model_part: Any, identifier: int) -> Any:
    return model_part.Properties[identifier] if model_part.HasProperties(identifier) else model_part.CreateNewProperties(identifier)


def _add_part(model_part: Any, name: str, node_ids: Sequence[int], condition_ids: Sequence[int] = (), element_ids: Sequence[int] = ()) -> Any:
    part = model_part.CreateSubModelPart(name)
    if node_ids:
        part.AddNodes(sorted(node_ids))
    if condition_ids:
        part.AddConditions(sorted(condition_ids))
    if element_ids:
        part.AddElements(sorted(element_ids))
    return part


def _signed_tetra_volume(points: np.ndarray) -> float:
    return float(np.linalg.det(np.vstack((points[1:] - points[0]))) / 6.0)


def _master_surface(
    fixture: IndenterFixture,
    *,
    first_node_id: int,
    first_condition_id: int,
    first_element_id: int,
    z_min_mm: float,
    z_max_mm: float,
    z_bands: int = 8,
) -> tuple[
    list[tuple[int, float, float, float]],
    list[tuple[int, tuple[int, int, int]]],
    list[tuple[int, tuple[int, int, int, int]]],
]:
    arc = list(fixture.contact_arc.coords)
    if len(arc) < 3:
        raise SolidFEAError("indenter contact arc has too few points")
    node_rows: list[list[int]] = []
    back_rows: list[list[int]] = []
    nodes: list[tuple[int, float, float, float]] = []
    node_id = first_node_id
    back_node_id = first_node_id + (z_bands + 1) * len(arc)
    thickness = fixture.settings.thickness_mm
    for z_index in range(z_bands + 1):
        z = z_min_mm + (z_max_mm - z_min_mm) * z_index / z_bands
        row: list[int] = []
        back_row: list[int] = []
        for x, y in arc:
            row.append(node_id)
            nodes.append((node_id, float(x), float(y), float(z)))
            node_id += 1
            radial_x = float(x - fixture.center_mm[0])
            radial_y = float(y - fixture.center_mm[1])
            radial_norm = math.hypot(radial_x, radial_y)
            if radial_norm <= 1.0e-12:
                raise SolidFEAError("indenter contact arc contains its carrier center")
            back_row.append(back_node_id)
            nodes.append(
                (
                    back_node_id,
                    float(x - thickness * radial_x / radial_norm),
                    float(y - thickness * radial_y / radial_norm),
                    float(z),
                )
            )
            back_node_id += 1
        node_rows.append(row)
        back_rows.append(back_row)
    conditions: list[tuple[int, tuple[int, int, int]]] = []
    elements: list[tuple[int, tuple[int, int, int, int]]] = []
    condition_id = first_condition_id
    element_id = first_element_id
    coordinates = {node_id: np.asarray((x, y, z), dtype=float) for node_id, x, y, z in nodes}
    for z_index in range(z_bands):
        for arc_index in range(len(arc) - 1):
            a = node_rows[z_index][arc_index]
            b = node_rows[z_index][arc_index + 1]
            c = node_rows[z_index + 1][arc_index + 1]
            d = node_rows[z_index + 1][arc_index]
            back_a = back_rows[z_index][arc_index]
            back_b = back_rows[z_index][arc_index + 1]
            back_c = back_rows[z_index + 1][arc_index + 1]
            back_d = back_rows[z_index + 1][arc_index]
            conditions.extend(((condition_id, (a, b, c)), (condition_id + 1, (a, c, d))))
            for raw_nodes in (
                (a, b, c, back_a),
                (b, c, back_c, back_a),
                (b, back_c, back_b, back_a),
                (c, d, back_d, back_a),
                (c, back_d, back_c, back_a),
            ):
                points = np.asarray([coordinates[node] for node in raw_nodes])
                if _signed_tetra_volume(points) < 0.0:
                    raw_nodes = (raw_nodes[0], raw_nodes[2], raw_nodes[1], raw_nodes[3])
                if _signed_tetra_volume(np.asarray([coordinates[node] for node in raw_nodes])) <= 0.0:
                    raise SolidFEAError("indenter carrier extrusion created a zero-volume tetrahedron")
                elements.append((element_id, raw_nodes))
                element_id += 1
            condition_id += 2
    return nodes, conditions, elements


def parameters_for_settings(settings: SolidFEASettings, *, use_mpc: bool = False) -> Any:
    KM, _, _, _ = import_kratos()
    pairs = [("PadOuterArc", "IndenterContactArc")] if settings.external_contact else []
    data = {
        "problem_data": {
            "problem_name": "fingertip_pad_3d_contact",
            "parallel_type": "OpenMP",
            "start_time": 0.0,
            "end_time": float(settings.number_of_steps),
            "echo_level": 0,
        },
        "solver_settings": {
            "model_part_name": "Structure",
            "domain_size": 3,
            "solver_type": "Static",
            "echo_level": 0,
            "analysis_type": "non_linear",
            "model_import_settings": {"input_type": "use_input_model_part"},
            "material_import_settings": {"materials_filename": ""},
            "time_stepping": {"time_step": 1.0},
            "volumetric_strain_dofs": True,
            "contact_settings": {
                "mortar_type": "ALMContactFrictionless",
                "ensure_contact": False,
                "silent_strategy": True,
                "simplified_semi_smooth_newton": False,
                "fancy_convergence_criterion": False,
                "print_convergence_criterion": False,
            },
            "clear_storage": True,
            "reform_dofs_at_each_step": True,
            "compute_reactions": True,
            "move_mesh_flag": True,
            "convergence_criterion": "contact_residual_criterion" if pairs else "residual_criterion",
            "displacement_relative_tolerance": 1.0e-6,
            "displacement_absolute_tolerance": 1.0e-9,
            "residual_relative_tolerance": 1.0e-6,
            "residual_absolute_tolerance": 1.0e-9,
            "max_iteration": settings.maximum_newton_iterations,
            "builder_and_solver_settings": {
                "type": "elimination" if use_mpc else "block",
                "advanced_settings": {},
            },
            "solving_strategy_settings": {"type": "newton_raphson", "advanced_settings": {}},
            "linear_solver_settings": {"solver_type": "skyline_lu_factorization"},
        },
        "processes": {
            "contact_process_list": [{
                "python_module": "alm_contact_process",
                "kratos_module": "KratosMultiphysics.ContactStructuralMechanicsApplication",
                "process_name": "ALMContactProcess",
                "Parameters": {
                    "model_part_name": "Structure",
                    "assume_master_slave": {str(index): [pair[0]] for index, pair in enumerate(pairs)},
                    "contact_model_part": {str(index): list(pair) for index, pair in enumerate(pairs)},
                    "contact_type": "Frictionless",
                },
            }] if pairs else []
        },
    }
    if not pairs:
        data["solver_settings"].pop("contact_settings")
    if use_mpc:
        data["solver_settings"]["multi_point_constraints_used"] = True
    return KM.Parameters(json.dumps(data))


def create_surface_condition(model_part: Any, properties: Any, condition_id: int, node_ids: Sequence[int]) -> None:
    if len(node_ids) != 3 or len(set(node_ids)) != 3:
        raise SolidFEAError(f"refusing invalid SurfaceCondition3D3N node list: {node_ids!r}")
    model_part.CreateNewCondition("SurfaceCondition3D3N", condition_id, list(node_ids), properties)


def _populate(
    model_part: Any,
    volume_mesh: FingertipVolumeMesh,
    fixture: IndenterFixture,
    *,
    external_contact: bool,
) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    KM, _, CLA, _ = import_kratos()
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 3
    pad_properties = properties_for_model_part(model_part, 1)
    pad_properties[KM.YOUNG_MODULUS] = volume_mesh.parameters.young_modulus_mpa
    pad_properties[KM.POISSON_RATIO] = volume_mesh.parameters.poisson_ratio
    pad_properties[KM.DENSITY] = 1.0
    pad_properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    pad_properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElastic3DLaw()
    for node in sorted(volume_mesh.nodes.values(), key=lambda value: value.id):
        model_part.CreateNewNode(node.id, node.x_mm, node.y_mm, node.z_mm)
    for tetrahedron in volume_mesh.tetrahedra:
        model_part.CreateNewElement(
            "TotalLagrangianMixedVolumetricStrainElement3D4N",
            tetrahedron.id,
            list(tetrahedron.node_ids),
            pad_properties,
        )

    condition_ids: dict[str, list[int]] = {"PadOuterArc": [], "IndenterContactArc": []}
    next_condition_id = max((triangle.id for values in volume_mesh.surface_triangles.values() for triangle in values), default=0) + 1
    if external_contact:
        pad_triangles = volume_mesh.surface_triangles["outer_compliant_arc"]
        coordinates = {node.id: (node.x_mm, node.y_mm, node.z_mm) for node in volume_mesh.nodes.values()}
        pad_report = validate_contact_triangles(
            pad_triangles,
            coordinates,
            contact_assignment=("PadOuterArc", "IndenterContactArc"),
        )
        if not pad_report.passed:
            raise SolidFEAError("pad contact surface failed preflight: " + ", ".join(pad_report.errors))
        for triangle in pad_triangles:
            create_surface_condition(model_part, pad_properties, triangle.id, triangle.node_ids)
            condition_ids["PadOuterArc"].append(triangle.id)

        master_nodes, master_conditions, master_elements = _master_surface(
            fixture,
            first_node_id=max(volume_mesh.nodes) + 1,
            first_condition_id=next_condition_id,
            first_element_id=max((tetrahedron.id for tetrahedron in volume_mesh.tetrahedra), default=0) + 1,
            z_min_mm=volume_mesh.solid.z_min_mm,
            z_max_mm=volume_mesh.solid.z_max_mm,
        )
        master_properties = properties_for_model_part(model_part, 2)
        master_properties[KM.YOUNG_MODULUS] = 1.0
        master_properties[KM.POISSON_RATIO] = 0.49
        master_properties[KM.DENSITY] = 1.0
        master_properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
        master_properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElastic3DLaw()
        master_coordinates: dict[int, tuple[float, float, float]] = {}
        for node_id, x, y, z in master_nodes:
            model_part.CreateNewNode(node_id, x, y, z)
            master_coordinates[node_id] = (x, y, z)
        master_report = validate_contact_triangles(
            master_conditions,
            master_coordinates,
            contact_assignment=("IndenterContactArc", "PadOuterArc"),
        )
        if not master_report.passed:
            raise SolidFEAError("master contact surface failed preflight: " + ", ".join(master_report.errors))
        for condition_id, node_ids in master_conditions:
            create_surface_condition(model_part, master_properties, condition_id, node_ids)
            condition_ids["IndenterContactArc"].append(condition_id)
        for element_id, node_ids in master_elements:
            model_part.CreateNewElement(
                "TotalLagrangianElement3D4N", element_id, list(node_ids), master_properties
            )
        _add_part(
            model_part,
            "IndenterContactArc",
            tuple(sorted({node_id for _, node_ids in master_conditions for node_id in node_ids})),
            condition_ids["IndenterContactArc"],
        )
        master_node_ids = tuple(sorted(master_coordinates))
        _add_part(
            model_part,
            "IndenterRigidCarrier",
            tuple(master_coordinates),
            element_ids=tuple(element_id for element_id, _ in master_elements),
        )
    else:
        master_node_ids = ()

    pad_contact_nodes = tuple(sorted({node_id for triangle in volume_mesh.surface_triangles["outer_compliant_arc"] for node_id in triangle.node_ids})) if external_contact else ()
    if external_contact:
        _add_part(model_part, "PadOuterArc", pad_contact_nodes, condition_ids=condition_ids["PadOuterArc"])
    bonded_tags = tuple(
        definition.name
        for definition in volume_mesh.solid.surfaces
        if definition.kind == "support" and definition.source_geometry is not None
    )
    support_node_ids = tuple(
        sorted(
            {
                node_id
                for tag in bonded_tags
                for triangle in volume_mesh.surface_triangles[tag]
                for node_id in triangle.node_ids
            }
        )
    )
    _add_part(model_part, "BondedSupport", support_node_ids)
    return {name: tuple(values) for name, values in condition_ids.items()}, master_node_ids, support_node_ids


def _apply_plane_strain_constraints(
    model_part: Any,
    node_columns: Sequence[Sequence[int]],
    reference_layer_index: int,
    fixed_node_ids: Sequence[int] = (),
) -> int:
    """Add exact in-plane equality constraints for a deterministic node map."""
    KM, _, _, _ = import_kratos()
    if not node_columns:
        raise SolidFEAError("plane-strain reference requires nonempty node columns")
    if any(
        len(column) <= reference_layer_index
        or len(set(column)) != len(column)
        for column in node_columns
    ):
        raise SolidFEAError("plane-strain node columns are not valid layered mappings")
    constraint_id = max((constraint.Id for constraint in model_part.MasterSlaveConstraints), default=0) + 1
    fixed_nodes = {int(node_id) for node_id in fixed_node_ids}
    count = 0
    for column in node_columns:
        # Every node in an authoritative bonded support column is already
        # prescribed to zero.  Avoid redundant fixed-DOF/MPC cycles, which
        # are not part of the plane-strain contract and are rejected by some
        # Kratos contact builders.
        if all(int(node_id) in fixed_nodes for node_id in column):
            continue
        master = model_part.Nodes[int(column[reference_layer_index])]
        for node_id in column:
            if int(node_id) == master.Id:
                continue
            slave = model_part.Nodes[int(node_id)]
            for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y):
                model_part.CreateNewMasterSlaveConstraint(
                    "LinearMasterSlaveConstraint",
                    constraint_id,
                    master,
                    variable,
                    slave,
                    variable,
                    1.0,
                    0.0,
                )
                constraint_id += 1
                count += 1
    return count


def _apply_constraints(
    model_part: Any,
    support_node_ids: Sequence[int],
    master_node_ids: Sequence[int],
    *,
    constrain_z: bool,
    reference_node_columns: Sequence[Sequence[int]] | None = None,
    reference_layer_index: int = 0,
) -> int:
    KM, _, _, _ = import_kratos()
    mpc_count = 0
    if reference_node_columns is not None:
        mpc_count = _apply_plane_strain_constraints(
            model_part,
            reference_node_columns,
            reference_layer_index,
            fixed_node_ids=support_node_ids,
        )
    for node_id in (*support_node_ids, *master_node_ids):
        node = model_part.Nodes[node_id]
        for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)
    if constrain_z:
        for node in model_part.Nodes:
            if node.Id not in support_node_ids and node.Id not in master_node_ids:
                node.Fix(KM.DISPLACEMENT_Z)
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)
    return mpc_count


def _move_master(model_part: Any, node_ids: Sequence[int], fixture: IndenterFixture, travel_mm: float) -> None:
    KM, _, _, _ = import_kratos()
    dx, dy = fixture.displacement_for_travel(travel_mm)
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        node.X = node.X0 + dx
        node.Y = node.Y0 + dy
        node.Z = node.Z0
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, dx)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, dy)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)


def localized_profile(distance_mm: float, radius_mm: float) -> float:
    """Return the fixed compact cosine profile used by the load-only path."""
    if distance_mm < 0.0 or distance_mm > radius_mm:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * distance_mm / radius_mm))


def _create_localized_surface_load_conditions(
    model_part: Any,
    volume_mesh: FingertipVolumeMesh,
    properties: Any,
    load_definition: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Create deterministic pressure conditions on the outer pad surface.

    This helper is deliberately separate from the mortar/contact population:
    it creates only prescribed Neumann conditions and never creates a contact
    pair or contact-search model part.
    """
    _, _, _, SMA = import_kratos()
    center_x = float(load_definition["center_x_mm"])
    center_z = float(load_definition["center_z_mm"])
    radius = float(load_definition["radius_mm"])
    if not all(math.isfinite(value) for value in (center_x, center_z, radius)) or radius <= 0.0:
        raise SolidFEAError("localized load center/radius must be finite and positive where required")
    outer_tags = tuple(
        definition.name
        for definition in volume_mesh.solid.surfaces
        if definition.kind == "outer_compliant" and definition.material_region == "pad"
    )
    source_triangles = tuple(
        triangle
        for tag in outer_tags
        for triangle in volume_mesh.surface_triangles.get(tag, ())
    )
    if not source_triangles:
        raise SolidFEAError("localized load requires semantic outer compliant surface triangles")
    coordinates = {
        node.id: np.asarray((node.x_mm, node.y_mm, node.z_mm), dtype=float)
        for node in volume_mesh.nodes.values()
    }
    next_condition_id = max((condition.Id for condition in model_part.Conditions), default=0) + 1
    conditions: list[Any] = []
    records: list[dict[str, Any]] = []
    for triangle in source_triangles:
        points = np.asarray([coordinates[node_id] for node_id in triangle.node_ids], dtype=float)
        centroid = np.mean(points, axis=0)
        distance = math.hypot(float(centroid[0]) - center_x, float(centroid[2]) - center_z)
        profile = localized_profile(distance, radius)
        if profile <= 0.0:
            continue
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        if not np.all(np.isfinite(normal)) or not math.isfinite(norm) or norm <= 1.0e-12:
            raise SolidFEAError(f"localized load triangle {triangle.id} has an invalid normal")
        inward = -normal / norm
        condition = model_part.CreateNewCondition(
            "SurfaceLoadCondition3D3N",
            next_condition_id,
            list(triangle.node_ids),
            properties,
        )
        condition.SetValue(SMA.SURFACE_LOAD, [0.0, 0.0, 0.0])
        conditions.append(condition)
        records.append(
            {
                "condition_id": int(next_condition_id),
                "surface_triangle_id": int(triangle.id),
                "node_ids": list(triangle.node_ids),
                "centroid_mm": centroid.tolist(),
                "distance_mm": float(distance),
                "profile_weight": float(profile),
                "inward_normal": inward.tolist(),
                "reference_area_mm2": 0.5 * norm,
            }
        )
        next_condition_id += 1
    if not conditions:
        raise SolidFEAError("localized load footprint selected no outer surface triangles")
    load_part = model_part.CreateSubModelPart("LocalizedLoad")
    # Add the exact condition/node topology. The conditions own the selected
    # footprint membership.
    load_node_ids = sorted({node.Id for condition in conditions for node in condition.GetGeometry()})
    if load_node_ids:
        load_part.AddNodes(load_node_ids)
    load_part.AddConditions([condition.Id for condition in conditions])
    return tuple(conditions), {
        "center_x_mm": center_x,
        "center_z_mm": center_z,
        "radius_mm": radius,
        "profile": str(load_definition.get("profile", "compact_cosine_radial")),
        "normalization": str(load_definition.get("normalization", "peak_pressure")),
        "surface_tags": list(outer_tags),
        "selected_triangle_count": len(records),
        "selected_condition_count": len(conditions),
        "selected_triangles": records,
    }


def _set_localized_surface_load(
    conditions: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    pressure_mpa: float,
    scale: float,
) -> np.ndarray:
    """Set one load-ramp value and return its reference resultant vector."""
    _, _, _, SMA = import_kratos()
    resultant = np.zeros(3, dtype=float)
    for condition, record in zip(conditions, records):
        magnitude = float(pressure_mpa) * float(scale) * float(record["profile_weight"])
        vector = magnitude * np.asarray(record["inward_normal"], dtype=float)
        condition.SetValue(SMA.SURFACE_LOAD, vector.tolist())
        resultant += float(record["reference_area_mm2"]) * vector
    return resultant


def _active_contact_diagnostics(
    model_part: Any,
    volume_mesh: FingertipVolumeMesh,
    fixture: IndenterFixture,
    pad_condition_ids: Sequence[int],
    active_condition_count: int,
    travel_mm: float,
) -> dict[str, Any]:
    """Check the final deformed contact state without changing the solve."""
    pad_triangles = volume_mesh.surface_triangles["outer_compliant_arc"]
    normal_norms: list[float] = []
    finite_normals = True
    for triangle in pad_triangles:
        points = np.asarray(
            [
                [
                    model_part.Nodes[node_id].X,
                    model_part.Nodes[node_id].Y,
                    model_part.Nodes[node_id].Z,
                ]
                for node_id in triangle.node_ids
            ],
            dtype=float,
        )
        if points.shape != (3, 3) or not np.all(np.isfinite(points)):
            finite_normals = False
            continue
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            finite_normals = False
        else:
            normal_norms.append(norm)

    dx, dy = fixture.displacement_for_travel(travel_mm)
    center = np.asarray(
        [fixture.center_mm[0] + dx, fixture.center_mm[1] + dy], dtype=float
    )
    contact_node_ids = sorted(
        {
            node_id
            for triangle in pad_triangles
            for node_id in triangle.node_ids
        }
    )
    clearances = np.asarray(
        [
            math.hypot(
                float(model_part.Nodes[node_id].X) - center[0],
                float(model_part.Nodes[node_id].Y) - center[1],
            )
            - fixture.settings.radius_mm
            for node_id in contact_node_ids
        ],
        dtype=float,
    )
    finite_clearance = bool(clearances.size) and bool(np.isfinite(clearances).all())
    max_penetration = float(max(0.0, -float(clearances.min()))) if finite_clearance else math.inf
    # ALM contact permits a finite penalty overlap.  For this focused active
    # state gate, require only that it remains finite and below the prescribed
    # travel; M4 owns the stricter, precommitted fidelity tolerance.
    penetration_tolerance_mm = max(float(travel_mm), 1.0e-12)
    generated_active = int(active_condition_count)
    checks = {
        "active_contact_conditions": generated_active > 0,
        "finite_nonzero_contact_normals": finite_normals and bool(normal_norms),
        "finite_contact_clearance": finite_clearance,
        "bounded_penetration": finite_clearance and max_penetration <= penetration_tolerance_mm,
        "source_contact_conditions_present": len(pad_condition_ids) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_condition_count": generated_active,
        "source_condition_count": len(pad_condition_ids),
        "normal_min_norm": min(normal_norms, default=0.0),
        "normal_max_norm": max(normal_norms, default=0.0),
        "min_clearance_mm": float(clearances.min()) if finite_clearance else None,
        "max_penetration_mm": max_penetration if finite_clearance else None,
        "penetration_tolerance_mm": penetration_tolerance_mm,
        "contact_node_count": len(contact_node_ids),
        "travel_mm": float(travel_mm),
    }


def solve_solid_3d(
    volume_mesh: FingertipVolumeMesh,
    fixture: IndenterFixture | None,
    settings: SolidFEASettings,
    *,
    reference_node_columns: Sequence[Sequence[int]] | None = None,
    reference_layer_index: int = 0,
    step_history: list[dict[str, Any]] | None = None,
    localized_load: Mapping[str, Any] | None = None,
) -> SolidFEAResult:
    """Solve the compliant-pad 3D model with contact or prescribed load.

    ``localized_load`` is a validation-only Neumann path.  It is mutually
    exclusive with external contact and does not alter the default production
    contact behavior.
    """
    if settings.mode == "3d_equivalent_reference" and reference_node_columns is None:
        raise SolidFEAError(
            "3d_equivalent_reference requires an explicit layered node-column map"
        )
    if settings.mode == "production" and reference_node_columns is not None:
        raise SolidFEAError("layered plane-strain constraints are validation-only")
    if localized_load is not None and settings.external_contact:
        raise SolidFEAError("localized load and external contact are mutually exclusive")
    if settings.external_contact and fixture is None:
        raise SolidFEAError("external contact requires an indenter fixture")
    if not volume_mesh.validation.passed:
        raise SolidFEAError("refusing invalid volume mesh: " + ", ".join(volume_mesh.validation.errors))
    KM, _, _, SMA = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

    node_order = tuple(sorted(volume_mesh.nodes))
    reference = np.asarray(
        [[volume_mesh.nodes[node_id].x_mm, volume_mesh.nodes[node_id].y_mm, volume_mesh.nodes[node_id].z_mm] for node_id in node_order],
        dtype=float,
    )
    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(
        model,
        parameters_for_settings(settings, use_mpc=reference_node_columns is not None),
    )
    model_part = model["Structure"]
    initialized = False
    try:
        condition_ids, master_node_ids, support_node_ids = _populate(
            model_part, volume_mesh, fixture, external_contact=settings.external_contact
        )
        localized_conditions: tuple[Any, ...] = ()
        localized_metadata: dict[str, Any] | None = None
        localized_records: tuple[Mapping[str, Any], ...] = ()
        if localized_load is not None:
            pad_properties = properties_for_model_part(model_part, 1)
            localized_conditions, localized_metadata = _create_localized_surface_load_conditions(
                model_part, volume_mesh, pad_properties, localized_load
            )
            localized_records = tuple(localized_metadata["selected_triangles"])
        analysis.Initialize()
        initialized = True
        # The public AnalysisStage owns DOF creation.  Add exact layered MPCs
        # after that initialization, then explicitly ask Kratos' supported
        # elimination strategy to rebuild its constraint-aware builder before
        # the first solution step.
        mpc_count = _apply_constraints(
            model_part,
            support_node_ids,
            master_node_ids,
            constrain_z=(settings.mode == "3d_equivalent_reference" or settings.reference_longitudinal_constraint),
            reference_node_columns=reference_node_columns,
            reference_layer_index=reference_layer_index,
        )
        solver = analysis._GetSolver()
        if reference_node_columns is not None:
            solver._GetSolutionStrategy()
        for step in range(1, settings.number_of_steps + 1):
            step_started = time.perf_counter()
            analysis.time = solver.AdvanceInTime(analysis.time)
            applied_resultant = np.zeros(3, dtype=float)
            if localized_load is not None:
                applied_resultant = _set_localized_surface_load(
                    localized_conditions,
                    localized_records,
                    float(localized_load["pressure_mpa"]),
                    step / settings.number_of_steps,
                )
            elif settings.external_contact:
                _move_master(model_part, master_node_ids, fixture, settings.indentation_mm * step / settings.number_of_steps)
            analysis.InitializeSolutionStep()
            solver.Predict()
            converged_step = bool(solver.SolveSolutionStep())
            analysis.FinalizeSolutionStep()
            if not converged_step:
                if step_history is not None:
                    step_history.append(
                        {
                            "step": step,
                            "load_ramp_fraction": step / settings.number_of_steps,
                            "converged": False,
                            "step_wall_seconds": time.perf_counter() - step_started,
                        }
                    )
                return SolidFEAResult(
                    volume_mesh, reference, None, None, None,
                    {name: {"condition_count": len(ids)} for name, ids in condition_ids.items()},
                    {
                        "mode": settings.mode,
                        "settings": asdict(settings),
                        "failed_step": step,
                        "plane_strain_mpc_count": mpc_count,
                    },
                    False,
                    f"3D nonlinear solver did not converge at step {step}",
                )
            if step_history is not None:
                generated_active = 0
                if model_part.HasSubModelPart("ComputingContact"):
                    generated = model_part.GetSubModelPart("ComputingContact")
                    generated_active = sum(
                        bool(condition.Is(KM.ACTIVE)) for condition in generated.Conditions
                    )
                reaction_vector_sum = np.zeros(3, dtype=float)
                for node_id in support_node_ids:
                    reaction_vector = np.asarray(
                        model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION),
                        dtype=float,
                    )
                    if not np.all(np.isfinite(reaction_vector)):
                        raise SolidFEAError("non-finite step-history support reaction")
                    reaction_vector_sum += reaction_vector
                if localized_load is not None:
                    load_scale = float(np.linalg.norm(applied_resultant))
                    step_history.append(
                        {
                            "step": step,
                            "load_ramp_fraction": step / settings.number_of_steps,
                            "pressure_mpa": float(localized_load["pressure_mpa"]),
                            "applied_resultant_n": applied_resultant.tolist(),
                            "applied_resultant_magnitude_n": load_scale,
                            "active_mortar_count": 0,
                            "reaction_force_n": float(np.linalg.norm(reaction_vector_sum)),
                            "load_balance_error_n": float(np.linalg.norm(reaction_vector_sum + applied_resultant)),
                            "converged": True,
                            "step_wall_seconds": time.perf_counter() - step_started,
                        }
                    )
                else:
                    loading_direction = np.asarray(
                        [fixture.frame.loading_direction[0], fixture.frame.loading_direction[1], 0.0],
                        dtype=float,
                    )
                    step_diagnostics = _active_contact_diagnostics(
                        model_part,
                        volume_mesh,
                        fixture,
                        condition_ids["PadOuterArc"],
                        generated_active,
                        settings.indentation_mm * step / settings.number_of_steps,
                    )
                    step_history.append(
                        {
                            "step": step,
                            "prescribed_travel_mm": settings.indentation_mm * step / settings.number_of_steps,
                            "active_mortar_count": generated_active,
                            "reaction_force_n": float(abs(np.dot(reaction_vector_sum, loading_direction))),
                            "minimum_gap_or_contact_clearance_mm": step_diagnostics["min_clearance_mm"],
                            "max_penetration_mm": step_diagnostics["max_penetration_mm"],
                            "converged": True,
                            "step_wall_seconds": time.perf_counter() - step_started,
                        }
                    )
        displacement = np.asarray(
            [[
                float(model_part.Nodes[node_id].GetSolutionStepValue(KM.DISPLACEMENT_X)),
                float(model_part.Nodes[node_id].GetSolutionStepValue(KM.DISPLACEMENT_Y)),
                float(model_part.Nodes[node_id].GetSolutionStepValue(KM.DISPLACEMENT_Z)),
            ] for node_id in node_order],
            dtype=float,
        )
        deformed = reference + displacement
        reaction_vector_sum = np.zeros(3, dtype=float)
        for node_id in support_node_ids:
            reaction_vector = np.asarray(model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION), dtype=float)
            if not np.all(np.isfinite(reaction_vector)):
                raise SolidFEAError("non-finite bonded-support reaction")
            reaction_vector_sum += reaction_vector
        actuator_reaction_vector_sum = np.zeros(3, dtype=float)
        for node_id in master_node_ids:
            reaction_vector = np.asarray(model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION), dtype=float)
            if not np.all(np.isfinite(reaction_vector)):
                raise SolidFEAError("non-finite indenter reaction")
            actuator_reaction_vector_sum += reaction_vector
        if localized_load is not None:
            final_applied_resultant = _set_localized_surface_load(
                localized_conditions,
                localized_records,
                float(localized_load["pressure_mpa"]),
                1.0,
            )
            loading_direction = final_applied_resultant / max(
                float(np.linalg.norm(final_applied_resultant)), 1.0e-30
            )
        else:
            loading_direction = np.asarray(
                [fixture.frame.loading_direction[0], fixture.frame.loading_direction[1], 0.0],
                dtype=float,
            )
            final_applied_resultant = np.zeros(3, dtype=float)
        reaction = float(abs(np.dot(reaction_vector_sum, loading_direction)))
        actuator_reaction = float(abs(np.dot(actuator_reaction_vector_sum, loading_direction)))
        force_equilibrium_error = (
            abs(actuator_reaction - reaction) / max(actuator_reaction, 1.0e-12)
            if master_node_ids
            else None
        )
        contact_state = {
            name: {"condition_count": len(ids), "active_condition_count": sum(bool(condition.Is(KM.ACTIVE)) for condition in model_part.GetSubModelPart(name).Conditions) if model_part.HasSubModelPart(name) else 0}
            for name, ids in condition_ids.items()
        }
        if localized_load is not None:
            contact_state["localized_load"] = {
                **(localized_metadata or {}),
                "pressure_mpa": float(localized_load["pressure_mpa"]),
                "final_applied_resultant_n": final_applied_resultant.tolist(),
                "final_applied_resultant_magnitude_n": float(np.linalg.norm(final_applied_resultant)),
                "selected_condition_count": len(localized_conditions),
            }
        if model_part.HasSubModelPart("ComputingContact"):
            generated = model_part.GetSubModelPart("ComputingContact")
            contact_state["generated_mortar"] = {
                "condition_count": generated.NumberOfConditions(),
                "active_condition_count": sum(bool(condition.Is(KM.ACTIVE)) for condition in generated.Conditions),
            }
        if settings.external_contact:
            generated_active = int(contact_state.get("generated_mortar", {}).get("active_condition_count", 0))
            contact_state["active_contact_diagnostics"] = _active_contact_diagnostics(
                model_part,
                volume_mesh,
                fixture,
                condition_ids["PadOuterArc"],
                generated_active,
                settings.indentation_mm,
            )
        contact_state["reaction_diagnostics"] = {
            "support_reaction_vector_sum_n": reaction_vector_sum.tolist(),
            "actuator_reaction_vector_sum_n": actuator_reaction_vector_sum.tolist(),
            "loading_direction_projection_n": reaction,
            "actuator_loading_direction_projection_n": actuator_reaction,
            "force_equilibrium_error": force_equilibrium_error,
            "finite": bool(np.isfinite(reaction_vector_sum).all()),
            "nonzero": reaction > 1.0e-12 if settings.external_contact else reaction >= 0.0,
        }
        if reference_node_columns is not None:
            displacement_by_node = {
                node_id: displacement[index]
                for index, node_id in enumerate(node_order)
            }
            ux_residuals: list[float] = []
            uy_residuals: list[float] = []
            uz_values: list[float] = []
            for column in reference_node_columns:
                reference_node = displacement_by_node[int(column[reference_layer_index])]
                for node_id in column:
                    value = displacement_by_node[int(node_id)]
                    ux_residuals.append(float(value[0] - reference_node[0]))
                    uy_residuals.append(float(value[1] - reference_node[1]))
                    uz_values.append(float(value[2]))
            contact_state["plane_strain_residuals"] = {
                "mpc_count": mpc_count,
                "reference_layer_index": reference_layer_index,
                "column_count": len(reference_node_columns),
                "max_abs_ux_mm": max(map(abs, ux_residuals), default=0.0),
                "max_abs_uy_mm": max(map(abs, uy_residuals), default=0.0),
                "max_abs_uz_mm": max(map(abs, uz_values), default=0.0),
                "rms_ux_mm": float(np.sqrt(np.mean(np.square(ux_residuals)))) if ux_residuals else 0.0,
                "rms_uy_mm": float(np.sqrt(np.mean(np.square(uy_residuals)))) if uy_residuals else 0.0,
                "rms_uz_mm": float(np.sqrt(np.mean(np.square(uz_values)))) if uz_values else 0.0,
            }
        return SolidFEAResult(
            volume_mesh,
            reference,
            deformed,
            displacement,
            reaction,
            contact_state,
            {
                "mode": settings.mode,
                "settings": asdict(settings),
                "element": "TotalLagrangianMixedVolumetricStrainElement3D4N",
                "constitutive_law": "HyperElastic3DLaw",
                "contact_condition": "SurfaceCondition3D3N",
                "bonded_support": {
                    "surface_tags": [
                        definition.name
                        for definition in volume_mesh.solid.surfaces
                        if definition.kind == "support" and definition.source_geometry is not None
                    ],
                    "constraint": "prescribed_zero_displacement_on_authoritative_surface_nodes",
                    "support_node_count": len(support_node_ids),
                },
                "plane_strain_reference": {
                    "enabled": reference_node_columns is not None,
                    "constraint": "LinearMasterSlaveConstraint ux/uy to reference z layer; uz fixed",
                    "mpc_count": mpc_count,
                    "reference_layer_index": reference_layer_index,
                    "column_count": len(reference_node_columns) if reference_node_columns is not None else 0,
                },
            },
            True,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exception:
        return SolidFEAResult(
            volume_mesh, reference, None, None, None, {},
            {"mode": settings.mode, "settings": asdict(settings)},
            False,
            f"3D FEA setup/solve error: {type(exception).__name__}: {exception}",
        )
    finally:
        if initialized:
            analysis.Finalize()


__all__ = [
    "ContactSurfaceValidation",
    "SolidFEAError",
    "SolidFEAResult",
    "SolidFEASettings",
    "create_surface_condition",
    "import_kratos",
    "localized_profile",
    "parameters_for_settings",
    "properties_for_model_part",
    "solve_solid_3d",
    "validate_contact_triangles",
]
