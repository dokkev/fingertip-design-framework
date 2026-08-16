"""Direct 3D Kratos mechanics on the neutral tetrahedral fingertip mesh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Literal

import numpy as np

from mesh.indenter import IndenterFixture
from mesh.volume_types import FingertipVolumeMesh


SolidFEAMode = Literal["3d_equivalent_reference", "production"]
SolidFEAContact = Literal["none", "three_pairs"]


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
    internal_contact: SolidFEAContact = "three_pairs"
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
        if self.internal_contact not in ("none", "three_pairs"):
            raise ValueError("internal_contact must be 'none' or 'three_pairs'")
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
        if not np.allclose(deformed - reference, displacement, rtol=0.0, atol=1.0e-10):
            raise ValueError("deformed coordinates and displacement disagree")
        deformed.setflags(write=False)
        displacement.setflags(write=False)
        object.__setattr__(self, "deformed_coordinates_mm", deformed)
        object.__setattr__(self, "displacement_mm", displacement)
        if self.reaction_force_n is not None and not math.isfinite(float(self.reaction_force_n)):
            raise ValueError("reaction_force_n must be finite")

    @property
    def morphology_fingerprint(self) -> str:
        """Return the source morphology fingerprint."""
        return self.volume_mesh.morphology_fingerprint


def _import_kratos() -> tuple[Any, Any, Any, Any]:
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
        import KratosMultiphysics.ConstitutiveLawsApplication as CLA
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
    except (ImportError, OSError) as exception:
        raise SolidFEAError("3D FEA requires Kratos Structural/Contact/Constitutive applications") from exception
    return KM, CSMA, CLA, SMA


def _properties(model_part: Any, identifier: int) -> Any:
    if model_part.HasProperties(identifier):
        return model_part.Properties[identifier]
    return model_part.CreateNewProperties(identifier)


def _add_part(
    model_part: Any,
    name: str,
    node_ids: set[int] | tuple[int, ...],
    condition_ids: set[int] | tuple[int, ...] = (),
    element_ids: set[int] | tuple[int, ...] = (),
) -> Any:
    part = model_part.CreateSubModelPart(name)
    if node_ids:
        part.AddNodes(sorted(node_ids))
    if condition_ids:
        part.AddConditions(sorted(condition_ids))
    if element_ids:
        part.AddElements(sorted(element_ids))
    return part


def _master_surface(
    fixture: IndenterFixture,
    *,
    first_node_id: int,
    first_condition_id: int,
    z_min_mm: float,
    z_max_mm: float,
    z_bands: int = 8,
) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, tuple[int, int, int]]]]:
    """Extrude the existing circular contact arc into one 3D master surface."""
    arc = list(fixture.contact_arc.coords)
    if len(arc) < 3:
        raise SolidFEAError("indenter contact arc has too few points")
    node_rows: list[list[int]] = []
    nodes: list[tuple[int, float, float, float]] = []
    node_id = first_node_id
    for z_index in range(z_bands + 1):
        z = z_min_mm + (z_max_mm - z_min_mm) * z_index / z_bands
        row: list[int] = []
        for x, y in arc:
            row.append(node_id)
            nodes.append((node_id, float(x), float(y), float(z)))
            node_id += 1
        node_rows.append(row)
    conditions: list[tuple[int, tuple[int, int, int]]] = []
    condition_id = first_condition_id
    for z_index in range(z_bands):
        for arc_index in range(len(arc) - 1):
            a = node_rows[z_index][arc_index]
            b = node_rows[z_index][arc_index + 1]
            c = node_rows[z_index + 1][arc_index + 1]
            d = node_rows[z_index + 1][arc_index]
            conditions.extend(((condition_id, (a, b, c)), (condition_id + 1, (a, c, d))))
            condition_id += 2
    return nodes, conditions


def _parameters(settings: SolidFEASettings) -> Any:
    KM, _, _, _ = _import_kratos()
    pairs: list[tuple[str, str]] = []
    if settings.external_contact:
        pairs.append(("PadOuterArc", "IndenterContactArc"))
    if settings.internal_contact == "three_pairs":
        pairs.extend(
            (
                ("PadVoidLeft", "StemContactLeft"),
                ("PadVoidRight", "StemContactRight"),
                ("PadVoidBottom", "StemContactBottom"),
            )
        )
    contact_model_part = {str(i): list(pair) for i, pair in enumerate(pairs)}
    data = {
        "problem_data": {
            "problem_name": "fingertip_3d_contact",
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
            "convergence_criterion": "contact_residual_criterion",
            "displacement_relative_tolerance": 1.0e-6,
            "displacement_absolute_tolerance": 1.0e-9,
            "residual_relative_tolerance": 1.0e-6,
            "residual_absolute_tolerance": 1.0e-9,
            "max_iteration": settings.maximum_newton_iterations,
            "builder_and_solver_settings": {"type": "block", "advanced_settings": {}},
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
                    "assume_master_slave": {str(i): [pair[0]] for i, pair in enumerate(pairs)},
                    "contact_model_part": contact_model_part,
                    "contact_type": "Frictionless",
                },
            }] if pairs else []
        },
    }
    if not pairs:
        data["solver_settings"].pop("contact_settings")
        data["solver_settings"]["convergence_criterion"] = "residual_criterion"
    return KM.Parameters(json.dumps(data))


def _populate(
    model_part: Any,
    volume_mesh: FingertipVolumeMesh,
    fixture: IndenterFixture,
    *,
    internal_contact: SolidFEAContact,
    external_contact: bool,
) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    KM, _, CLA, _ = _import_kratos()
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 3
    pad_properties = _properties(model_part, 1)
    rigid_properties = _properties(model_part, 2)
    for properties, young_modulus in (
        (pad_properties, volume_mesh.parameters.young_modulus_mpa),
        (rigid_properties, 1.0),
    ):
        properties[KM.YOUNG_MODULUS] = young_modulus
        properties[KM.POISSON_RATIO] = volume_mesh.parameters.poisson_ratio
        properties[KM.DENSITY] = 1.0
        properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
        properties[KM.CONSTITUTIVE_LAW] = CLA.HyperElastic3DLaw()
    for node in sorted(volume_mesh.nodes.values(), key=lambda value: value.id):
        model_part.CreateNewNode(node.id, node.x_mm, node.y_mm, node.z_mm)
    for tetrahedron in volume_mesh.tetrahedra:
        properties = pad_properties if tetrahedron.domain == "pad" else rigid_properties
        element_name = (
            "TotalLagrangianMixedVolumetricStrainElement3D4N"
            if tetrahedron.domain == "pad"
            else "TotalLagrangianElement3D4N"
        )
        model_part.CreateNewElement(element_name, tetrahedron.id, list(tetrahedron.node_ids), properties)

    surface_tags = {"PadOuterArc": "outer_compliant_arc"}
    if internal_contact == "three_pairs":
        surface_tags.update(
            {
                "PadVoidLeft": "void_left",
                "PadVoidRight": "void_right",
                "PadVoidBottom": "void_bottom",
                "StemContactLeft": "contact_left",
                "StemContactRight": "contact_right",
                "StemContactBottom": "contact_bottom",
            }
        )
    condition_ids: dict[str, list[int]] = {name: [] for name in (*surface_tags, "IndenterContactArc")}
    max_condition = max(
        (triangle.id for values in volume_mesh.surface_triangles.values() for triangle in values),
        default=0,
    )
    for part_name, tag in surface_tags.items():
        properties = rigid_properties if part_name.startswith("Stem") else pad_properties
        for triangle in volume_mesh.surface_triangles[tag]:
            model_part.CreateNewCondition(
                "SurfaceCondition3D3N", triangle.id, list(triangle.node_ids), properties
            )
            condition_ids[part_name].append(triangle.id)

    master_nodes, master_conditions = _master_surface(
        fixture,
        first_node_id=max(volume_mesh.nodes) + 1,
        first_condition_id=max_condition + 1,
        z_min_mm=volume_mesh.solid.z_min_mm,
        z_max_mm=volume_mesh.solid.z_max_mm,
    )
    master_node_ids: set[int] = set()
    master_properties = _properties(model_part, 3)
    master_properties[KM.YOUNG_MODULUS] = 1.0
    master_properties[KM.POISSON_RATIO] = 0.49
    master_properties[KM.DENSITY] = 1.0
    master_properties[KM.VOLUME_ACCELERATION] = [0.0, 0.0, 0.0]
    for node_id, x, y, z in master_nodes:
        model_part.CreateNewNode(node_id, x, y, z)
        master_node_ids.add(node_id)
    for condition_id, node_ids in master_conditions:
        model_part.CreateNewCondition(
            "SurfaceCondition3D3N", condition_id, list(node_ids), master_properties
        )
        condition_ids["IndenterContactArc"].append(condition_id)

    for part_name, tag in surface_tags.items():
        nodes = {
            node_id
            for triangle in volume_mesh.surface_triangles[tag]
            for node_id in triangle.node_ids
        }
        _add_part(model_part, part_name, nodes, condition_ids[part_name])
    _add_part(model_part, "IndenterContactArc", master_node_ids, condition_ids["IndenterContactArc"])

    carrier_element_ids = set(volume_mesh.volume_element_ids["rigid_carrier"])
    carrier_node_ids = {
        node.Id
        for element_id in carrier_element_ids
        for node in model_part.Elements[element_id].GetGeometry()
    }
    _add_part(model_part, "RigidCarrier", carrier_node_ids, element_ids=carrier_element_ids)
    _add_part(model_part, "RigidMotion", carrier_node_ids)
    support_node_ids = {
        node_id
        for tag in ("support_bond_left", "support_bond_right")
        for triangle in volume_mesh.surface_triangles[tag]
        for node_id in triangle.node_ids
    }
    return (
        {name: tuple(sorted(values)) for name, values in condition_ids.items()},
        tuple(sorted(carrier_node_ids)),
        tuple(sorted(master_node_ids)),
        tuple(sorted(support_node_ids)),
    )


def _support_tie_pairs(volume_mesh: FingertipVolumeMesh) -> tuple[tuple[int, int], ...]:
    """Return exact pad-to-rigid support pairs, or fail closed.

    Contact facets intentionally remain disconnected for zero-clearance
    contact.  The bonded pad/link interface therefore needs an explicit tie;
    silently omitting it would leave the compliant pad mechanically floating.
    """
    pairs: list[tuple[int, int]] = []
    for tag in ("support_bond_left", "support_bond_right"):
        pad_ids = {
            node_id
            for triangle in volume_mesh.surface_triangles[tag]
            if triangle.domain == "pad"
            for node_id in triangle.node_ids
        }
        rigid_ids = {
            node_id
            for triangle in volume_mesh.surface_triangles[tag]
            if triangle.domain == "rigid_carrier"
            for node_id in triangle.node_ids
        }
        key = lambda node_id: tuple(
            round(float(value), 8)
            for value in (
                volume_mesh.nodes[node_id].x_mm,
                volume_mesh.nodes[node_id].y_mm,
                volume_mesh.nodes[node_id].z_mm,
            )
        )
        rigid_by_key = {key(node_id): node_id for node_id in rigid_ids}
        for pad_id in sorted(pad_ids):
            rigid_id = rigid_by_key.get(key(pad_id))
            if rigid_id is None:
                raise SolidFEAError(
                    f"bonded interface {tag!r} is nonconforming: no exact rigid node for pad node {pad_id}"
                )
            pairs.append((rigid_id, pad_id))
    return tuple(sorted(set(pairs)))


def _create_tie_constraints(model_part: Any, pairs: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """Create explicit X/Y/Z linear ties at the bonded support interface."""
    KM, _, _, _ = _import_kratos()
    constraint_ids: list[int] = []
    next_id = max((constraint.Id for constraint in model_part.MasterSlaveConstraints), default=0) + 1
    for master_id, slave_id in pairs:
        master = model_part.Nodes[master_id]
        slave = model_part.Nodes[slave_id]
        for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
            model_part.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint",
                next_id,
                master,
                variable,
                slave,
                variable,
                1.0,
                0.0,
            )
            constraint_ids.append(next_id)
            next_id += 1
    return tuple(constraint_ids)


def _apply_constraints(
    model_part: Any,
    carrier_node_ids: tuple[int, ...],
    master_node_ids: tuple[int, ...],
    support_node_ids: tuple[int, ...],
    *,
    constrain_z: bool,
) -> None:
    KM, _, _, _ = _import_kratos()
    for node_id in (*carrier_node_ids, *master_node_ids, *support_node_ids):
        node = model_part.Nodes[node_id]
        for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
            node.Fix(variable)
            node.SetSolutionStepValue(variable, 0.0)
    if constrain_z:
        for node in model_part.Nodes:
            if node.Id not in carrier_node_ids and node.Id not in master_node_ids:
                node.Fix(KM.DISPLACEMENT_Z)
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)


def _move_master(model_part: Any, node_ids: tuple[int, ...], fixture: IndenterFixture, travel_mm: float) -> None:
    KM, _, _, _ = _import_kratos()
    dx, dy = fixture.displacement_for_travel(travel_mm)
    for node_id in node_ids:
        node = model_part.Nodes[node_id]
        node.X = node.X0 + dx
        node.Y = node.Y0 + dy
        node.Z = node.Z0
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, dx)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, dy)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)


def solve_solid_3d(
    volume_mesh: FingertipVolumeMesh,
    fixture: IndenterFixture,
    settings: SolidFEASettings,
) -> SolidFEAResult:
    """Solve nonlinear 3D ALM contact on TET4 volume elements."""
    if not volume_mesh.validation.passed:
        raise SolidFEAError("refusing invalid volume mesh: " + ", ".join(volume_mesh.validation.errors))
    KM, _, _, _ = _import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

    node_order = tuple(sorted(volume_mesh.nodes))
    reference = np.asarray(
        [[volume_mesh.nodes[node_id].x_mm, volume_mesh.nodes[node_id].y_mm, volume_mesh.nodes[node_id].z_mm] for node_id in node_order],
        dtype=float,
    )
    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(model, _parameters(settings))
    model_part = model["Structure"]
    try:
        condition_ids, carrier_node_ids, master_node_ids, support_node_ids = _populate(
            model_part,
            volume_mesh,
            fixture,
            internal_contact=settings.internal_contact,
            external_contact=settings.external_contact,
        )
        support_tie_pairs = _support_tie_pairs(volume_mesh)
        tie_constraint_ids = _create_tie_constraints(model_part, support_tie_pairs)
        analysis.Initialize()
        _apply_constraints(
            model_part,
            carrier_node_ids,
            master_node_ids,
            support_node_ids,
            constrain_z=(settings.mode == "3d_equivalent_reference" or settings.reference_longitudinal_constraint),
        )
        solver = analysis._GetSolver()
        failed_step: int | None = None
        for step in range(1, settings.number_of_steps + 1):
            analysis.time = solver.AdvanceInTime(analysis.time)
            _move_master(model_part, master_node_ids, fixture, settings.indentation_mm * step / settings.number_of_steps)
            analysis.InitializeSolutionStep()
            solver.Predict()
            converged = bool(solver.SolveSolutionStep())
            analysis.FinalizeSolutionStep()
            if not converged:
                failed_step = step
                break
        if failed_step is not None:
            return SolidFEAResult(
                volume_mesh, reference, None, None, None,
                {name: {"condition_count": len(ids)} for name, ids in condition_ids.items()},
                {"mode": settings.mode, "settings": asdict(settings), "failed_step": failed_step},
                False,
                f"3D nonlinear solver did not converge at step {failed_step}",
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
        loading = np.asarray((*fixture.frame.loading_direction, 0.0), dtype=float)
        reaction = 0.0
        for node_id in carrier_node_ids:
            reaction += float(np.dot(np.asarray(model_part.Nodes[node_id].GetSolutionStepValue(KM.REACTION), dtype=float), loading))
        contact_state = {
            name: {
                "condition_count": len(ids),
                "active_condition_count": sum(
                    bool(condition.Is(KM.ACTIVE))
                    for condition in model_part.GetSubModelPart(name).Conditions
                ) if model_part.HasSubModelPart(name) else 0,
            }
            for name, ids in condition_ids.items()
        }
        return SolidFEAResult(
            volume_mesh,
            reference,
            deformed,
            displacement,
            abs(reaction),
            contact_state,
            {
                "mode": settings.mode,
                "settings": asdict(settings),
                "element_pad": "TotalLagrangianMixedVolumetricStrainElement3D4N",
                "element_rigid": "TotalLagrangianElement3D4N",
                "constitutive_law": "HyperElastic3DLaw",
                "contact_condition": "SurfaceCondition3D3N",
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "bonded_interface": {
                    "constraint_count": len(tie_constraint_ids),
                    "pair_count": len(support_tie_pairs),
                    "contract": "explicit LinearMasterSlaveConstraint on exact support coordinates",
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
        analysis.Finalize()


__all__ = ["SolidFEAError", "SolidFEAResult", "SolidFEASettings", "solve_solid_3d"]
