"""Phase 4I-F search-pair and contact/bond crosspoint diagnostics.

The routines in this module deliberately reproduce Kratos' first nonlinear
step with the public strategy components.  They do not alter the production
contact configuration.  The only state mutation, ``ACTIVE=False`` in case
F02, is explicitly diagnostic-only and is applied after the normal search has
created the paired condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import traceback
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from fem.indentation_analysis import (
    _nodal_fields,
    set_indenter_travel,
)
from fem.indentation_postprocess import (
    compressive_indenter_reaction,
    pad_strain_det_f_statistics,
)
from fem.internal_contact_diagnostic import (
    FIRST_STEP_TRAVEL_MM,
    _build_context,
    _contact_condition_records,
    _dof_records,
    _runtime_contract,
    _source_condition_maps,
)
from fem.kratos_adapter import _import_kratos
from fem.kratos_settings import MAXIMUM_NEWTON_ITERATIONS
from fem.right_side_audit import _endpoint_id


Side = Literal["left", "right"]
Variant = Literal["F00", "F02", "L00"]

LOCAL_DOMAIN_TOLERANCE = 1.0e-12
EXACT_OVERLAP_TOLERANCE_MM = 1.0e-12


@dataclass(frozen=True)
class CausalVariant:
    """One executable diagnostic mutation or an unmodified control."""

    name: Variant
    side: Side
    force_invalid_pair_inactive: bool = False


CAUSAL_VARIANTS = {
    "F00": CausalVariant("F00", "right", False),
    "F02": CausalVariant("F02", "right", True),
    "L00": CausalVariant("L00", "left", False),
}


def line2_projection_local_domain(
    point: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float = LOCAL_DOMAIN_TOLERANCE,
) -> dict[str, Any]:
    """Project a point and apply the Line2 local-coordinate domain [-1, 1]."""
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("projection tolerance must be finite and non-negative")
    delta = np.asarray(second[:2], dtype=float) - np.asarray(
        first[:2], dtype=float
    )
    denominator = float(delta @ delta)
    if not math.isfinite(denominator) or denominator <= 0.0:
        return {
            "available": False,
            "reason": "zero_or_nonfinite_segment_length",
        }
    point_array = np.asarray(point[:2], dtype=float)
    first_array = np.asarray(first[:2], dtype=float)
    segment_fraction = float((point_array - first_array) @ delta / denominator)
    local_coordinate = 2.0 * segment_fraction - 1.0
    projection = first_array + segment_fraction * delta
    inside = abs(local_coordinate) <= 1.0 + tolerance
    return {
        "available": True,
        "segment_fraction": segment_fraction,
        "local_coordinate": local_coordinate,
        "local_domain": [-1.0, 1.0],
        "tolerance": tolerance,
        "inside_local_domain": inside,
        "projection_point_mm": [float(value) for value in projection],
        "distance_to_infinite_line_mm": float(
            np.linalg.norm(point_array - projection)
        ),
        "implementation": (
            "orthogonal projection converted by xi=2*t-1; acceptance matches "
            "Line2D2::IsInside abs(xi)<=1+tolerance"
        ),
        "python_geometry_is_inside_exposed": False,
    }


def _safe_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _node_scalar(node: Any, variable: Any) -> float | None:
    try:
        if node.SolutionStepsDataHas(variable):
            return _safe_float(node.GetSolutionStepValue(variable))
    except Exception:
        pass
    try:
        if node.Has(variable):
            return _safe_float(node.GetValue(variable))
    except Exception:
        pass
    return None


def _process_flag(process_info: Any, variable: Any) -> bool | None:
    try:
        if process_info.Has(variable):
            return bool(process_info[variable])
    except Exception:
        pass
    return None


def _source_condition(
    context: Any,
    source_name: str,
    connectivity: Sequence[int],
) -> Any | None:
    by_surface, _ = _source_condition_maps(context)
    identifier = by_surface[source_name].get(tuple(sorted(connectivity)))
    return (
        context.model_part.Conditions[identifier]
        if identifier is not None
        else None
    )


def _exact_overlap(
    context: Any,
    slave_name: str,
    master_name: str,
    slave_connectivity: Sequence[int],
    master_connectivity: Sequence[int],
) -> dict[str, Any]:
    KM, _, _, _ = _import_kratos()
    slave = _source_condition(context, slave_name, slave_connectivity)
    master = _source_condition(context, master_name, master_connectivity)
    if slave is None or master is None:
        return {
            "available": False,
            "reason": "source_condition_connectivity_not_resolved",
        }
    try:
        value = float(
            KM.ExactMortarIntegrationUtility2D2N(
                2
            ).TestGetExactAreaIntegration(slave, master)
        )
    except Exception as exception:
        return {
            "available": False,
            "reason": f"{type(exception).__name__}: {exception}",
        }
    return {
        "available": math.isfinite(value),
        "overlap_length_mm": value if math.isfinite(value) else None,
        "positive_overlap": (
            math.isfinite(value)
            and value > EXACT_OVERLAP_TOLERANCE_MM
        ),
        "zero_overlap_tolerance_mm": EXACT_OVERLAP_TOLERANCE_MM,
        "api": (
            "KratosMultiphysics.ExactMortarIntegrationUtility2D2N(2)."
            "TestGetExactAreaIntegration"
        ),
    }


def endpoint_pair_records(
    context: Any,
    pair_index: int,
    endpoint_node_id: int,
) -> list[dict[str, Any]]:
    """Resolve every generated condition incident to an endpoint."""
    KM, _, _, _ = _import_kratos()
    group_name, slave_name, master_name = context.groups[pair_index]
    computing = context.model[
        f"Structure.ComputingContact.ComputingContactSub{pair_index}"
    ]
    endpoint = context.model_part.Nodes[endpoint_node_id]
    point = [float(endpoint.X), float(endpoint.Y)]
    records: list[dict[str, Any]] = []
    for condition in computing.Conditions:
        geometry = condition.GetGeometry()
        slave = geometry.GetGeometryPart(0)
        master = geometry.GetGeometryPart(1)
        slave_ids = [node.Id for node in slave]
        if endpoint_node_id not in slave_ids:
            continue
        master_ids = [node.Id for node in master]
        master_points = [
            [float(node.X), float(node.Y)] for node in master
        ]
        projection = line2_projection_local_domain(
            point, master_points[0], master_points[1]
        )
        overlap = _exact_overlap(
            context,
            slave_name,
            master_name,
            slave_ids,
            master_ids,
        )
        records.append(
            {
                "contact_process_index": pair_index,
                "contact_group": group_name,
                "generated_condition_id": int(condition.Id),
                "condition_active": bool(condition.Is(KM.ACTIVE)),
                "slave_node_ids": slave_ids,
                "master_node_ids": master_ids,
                "master_node_coordinates_mm": master_points,
                "endpoint_projection": projection,
                "exact_overlap": overlap,
                "valid_endpoint_pair": bool(
                    projection.get("inside_local_domain")
                    and overlap.get("positive_overlap")
                ),
                "out_of_domain_extra_pair": bool(
                    not projection.get("inside_local_domain", False)
                ),
            }
        )
    return sorted(records, key=lambda record: record["generated_condition_id"])


def _dof_state(node: Any, variable: Any) -> dict[str, Any]:
    if not node.HasDofFor(variable):
        return {"present": False}
    dof = node.GetDof(variable)
    return {
        "present": True,
        "equation_id": int(dof.EquationId),
        "fixed": bool(dof.IsFixed()),
    }


def _local_lm_rows(
    context: Any,
    pair_index: int,
    endpoint_node_id: int,
) -> dict[str, Any]:
    KM, CSMA, _, _ = _import_kratos()
    computing = context.model[
        f"Structure.ComputingContact.ComputingContactSub{pair_index}"
    ]
    rows: list[dict[str, Any]] = []
    aggregate: dict[int, float] = {}
    valid_aggregate: dict[int, float] = {}
    for pair in endpoint_pair_records(context, pair_index, endpoint_node_id):
        condition = computing.Conditions[pair["generated_condition_id"]]
        dofs = list(condition.GetDofList(context.model_part.ProcessInfo))
        row_indices = [
            index
            for index, dof in enumerate(dofs)
            if int(dof.Id()) == endpoint_node_id
            and dof.GetVariable().Name()
            == CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
        ]
        if not row_indices:
            continue
        try:
            lhs = KM.Matrix()
            rhs = KM.Vector()
            condition.CalculateLocalSystem(
                lhs, rhs, context.model_part.ProcessInfo
            )
            equation_ids = [
                int(identifier)
                for identifier in condition.EquationIdVector(
                    context.model_part.ProcessInfo
                )
            ]
            local_dofs = [
                {
                    "node_id": int(dof.Id()),
                    "variable": dof.GetVariable().Name(),
                    "equation_id": equation_ids[index],
                    "fixed": bool(dof.IsFixed()),
                }
                for index, dof in enumerate(dofs)
            ]
            for row_index in row_indices:
                values = [
                    float(lhs[row_index, column])
                    for column in range(lhs.Size2())
                ]
                free_columns = [
                    index
                    for index, dof in enumerate(dofs)
                    if not dof.IsFixed()
                ]
                fixed_columns = [
                    index
                    for index, dof in enumerate(dofs)
                    if dof.IsFixed()
                ]
                lm_diagonal = next(
                    (
                        values[index]
                        for index, dof in enumerate(local_dofs)
                        if dof["node_id"] == endpoint_node_id
                        and dof["variable"]
                        == CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE.Name()
                    ),
                    None,
                )
                row = {
                    "condition_id": int(condition.Id),
                    "condition_active": bool(condition.Is(KM.ACTIVE)),
                    "valid_endpoint_pair": pair["valid_endpoint_pair"],
                    "out_of_domain_extra_pair": pair[
                        "out_of_domain_extra_pair"
                    ],
                    "local_lm_row_index": row_index,
                    "row_norm_all_columns": float(np.linalg.norm(values)),
                    "row_norm_free_columns": float(
                        np.linalg.norm(
                            [values[index] for index in free_columns]
                        )
                    ),
                    "row_norm_fixed_columns": float(
                        np.linalg.norm(
                            [values[index] for index in fixed_columns]
                        )
                    ),
                    "lm_diagonal": _safe_float(lm_diagonal),
                    "local_row_values": values,
                    "local_dofs": local_dofs,
                }
                rows.append(row)
                for equation_id, value in zip(equation_ids, values):
                    aggregate[equation_id] = (
                        aggregate.get(equation_id, 0.0) + value
                    )
                    if pair["valid_endpoint_pair"]:
                        valid_aggregate[equation_id] = (
                            valid_aggregate.get(equation_id, 0.0) + value
                        )
        except Exception as exception:
            rows.append(
                {
                    "condition_id": int(condition.Id),
                    "condition_active": bool(condition.Is(KM.ACTIVE)),
                    "valid_endpoint_pair": pair["valid_endpoint_pair"],
                    "available": False,
                    "reason": f"{type(exception).__name__}: {exception}",
                }
            )

    dof_by_equation: dict[int, Any] = {}
    for dof in context.analysis._GetSolver()._GetBuilderAndSolver().GetDofSet():
        dof_by_equation[int(dof.EquationId)] = dof

    def aggregate_stats(values: Mapping[int, float]) -> dict[str, Any]:
        all_values = list(values.values())
        free_values = [
            value
            for equation_id, value in values.items()
            if equation_id in dof_by_equation
            and not dof_by_equation[equation_id].IsFixed()
        ]
        fixed_values = [
            value
            for equation_id, value in values.items()
            if equation_id in dof_by_equation
            and dof_by_equation[equation_id].IsFixed()
        ]
        return {
            "row_norm_all_columns": float(np.linalg.norm(all_values)),
            "row_norm_free_columns": float(np.linalg.norm(free_values)),
            "row_norm_fixed_columns": float(np.linalg.norm(fixed_values)),
            "nonzero_equation_count": sum(
                abs(value) > 0.0 for value in all_values
            ),
        }

    return {
        "available": bool(rows),
        "condition_rows": rows,
        "aggregate_before_dirichlet": aggregate_stats(aggregate),
        "valid_pairs_only_before_dirichlet": aggregate_stats(valid_aggregate),
    }


def _global_lm_row(
    matrix: Any | None,
    rhs: Any | None,
    equation_id: int | None,
    fixed_equation_ids: set[int],
) -> dict[str, Any]:
    if matrix is None or rhs is None or equation_id is None:
        return {"available": False, "reason": "system_not_assembled"}
    if equation_id < 0 or equation_id >= matrix.Size1():
        return {"available": False, "reason": "equation_outside_system"}
    try:
        import KratosMultiphysics.scipy_conversion_tools as conversion

        row = conversion.to_csr(matrix).getrow(equation_id)
        free = [
            value
            for column, value in zip(row.indices, row.data)
            if int(column) not in fixed_equation_ids
        ]
        fixed = [
            value
            for column, value in zip(row.indices, row.data)
            if int(column) in fixed_equation_ids
        ]
        return {
            "available": True,
            "row_norm_all_columns": float(np.linalg.norm(row.data)),
            "row_norm_free_columns": float(np.linalg.norm(free)),
            "row_norm_fixed_columns": float(np.linalg.norm(fixed)),
            "nnz": int(row.nnz),
            "rhs_entry": _safe_float(rhs[equation_id]),
            "stage": "after_builder_dirichlet_elimination",
        }
    except Exception as exception:
        return {
            "available": False,
            "reason": f"{type(exception).__name__}: {exception}",
        }


def _memberships(context: Any, node_id: int) -> dict[str, Any]:
    names = (
        "PadCutoutLeft",
        "PadCutoutRight",
        "PadBondLeft",
        "PadBondRight",
        "RigidBondInterface",
        "RigidMotion",
        "StemLeft",
        "StemRight",
    )
    membership = {
        name: bool(
            context.model_part.HasSubModelPart(name)
            and context.model_part.GetSubModelPart(name).HasNode(node_id)
        )
        for name in names
    }
    node = context.model_part.Nodes[node_id]
    KM, _, _, _ = _import_kratos()
    return {
        "submodelparts": membership,
        "pad_internal_contact_boundary": (
            membership["PadCutoutLeft"]
            or membership["PadCutoutRight"]
        ),
        "pad_bond_boundary": (
            membership["PadBondLeft"] or membership["PadBondRight"]
        ),
        "stem_contact_boundary": (
            membership["StemLeft"] or membership["StemRight"]
        ),
        "tied_or_rigid_boundary_membership": (
            membership["RigidBondInterface"]
            or membership["RigidMotion"]
        ),
        "dirichlet_boundary": (
            node.GetDof(KM.DISPLACEMENT_X).IsFixed()
            or node.GetDof(KM.DISPLACEMENT_Y).IsFixed()
        ),
        "xy_free_primal_dof_count": sum(
            not node.GetDof(variable).IsFixed()
            for variable in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y)
        ),
    }


def endpoint_snapshot(
    context: Any,
    side: Side,
    stage: str,
    iteration: int | None,
    matrix: Any | None = None,
    rhs: Any | None = None,
    pair_index: int = 1,
) -> dict[str, Any]:
    """Serialize the endpoint state at one exact lifecycle point."""
    KM, CSMA, _, _ = _import_kratos()
    endpoint_id = _endpoint_id(context.model_part, side, slave=True)
    node = context.model_part.Nodes[endpoint_id]
    lm_dof = _dof_state(
        node, CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
    )
    dof_set = context.analysis._GetSolver()._GetBuilderAndSolver().GetDofSet()
    fixed_equation_ids = {
        int(dof.EquationId) for dof in dof_set if dof.IsFixed()
    }
    weighted_gap = _node_scalar(node, CSMA.WEIGHTED_GAP)
    lm_pressure = _node_scalar(
        node, CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
    )
    penalty = _safe_float(context.model_part.ProcessInfo[KM.INITIAL_PENALTY])
    scale = _safe_float(context.model_part.ProcessInfo[KM.SCALE_FACTOR])
    augmented = (
        scale * lm_pressure + penalty * weighted_gap
        if None not in (scale, lm_pressure, penalty, weighted_gap)
        else None
    )
    return {
        "variant": None,
        "side": side,
        "stage": stage,
        "iteration": iteration,
        "endpoint_node_id": endpoint_id,
        "reference_coordinate_mm": [float(node.X0), float(node.Y0)],
        "current_coordinate_mm": [float(node.X), float(node.Y)],
        "node_flags": {
            "ACTIVE": bool(node.Is(KM.ACTIVE)),
            "SLAVE": bool(node.Is(KM.SLAVE)),
            "MASTER": bool(node.Is(KM.MASTER)),
        },
        "lagrange_multiplier_contact_pressure": lm_pressure,
        "weighted_gap": weighted_gap,
        "normal_gap": _node_scalar(node, CSMA.NORMAL_GAP),
        "nodal_area": _node_scalar(node, KM.NODAL_AREA),
        "nodal_h": _node_scalar(node, KM.NODAL_H),
        "augmented_normal_contact_pressure_recomputed": _safe_float(
            augmented
        ),
        "active_set_rule": (
            "active iff SCALE_FACTOR*LM + INITIAL_PENALTY*WEIGHTED_GAP < 0"
        ),
        "process_flags": {
            "ACTIVE_SET_COMPUTED": _process_flag(
                context.model_part.ProcessInfo,
                CSMA.ACTIVE_SET_COMPUTED,
            ),
            "CONTACT_CONVERGED": _process_flag(
                context.model_part.ProcessInfo,
                CSMA.CONTACT_CONVERGED,
            ),
        },
        "dofs": {
            "DISPLACEMENT_X": _dof_state(node, KM.DISPLACEMENT_X),
            "DISPLACEMENT_Y": _dof_state(node, KM.DISPLACEMENT_Y),
            "VOLUMETRIC_STRAIN": _dof_state(
                node, KM.VOLUMETRIC_STRAIN
            ),
            "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE": lm_dof,
        },
        "crosspoint": _memberships(context, endpoint_id),
        "incident_generated_conditions": endpoint_pair_records(
            context, pair_index, endpoint_id
        ),
        "local_lm_assembly": _local_lm_rows(
            context, pair_index, endpoint_id
        ),
        "global_lm_assembly": _global_lm_row(
            matrix,
            rhs,
            lm_dof.get("equation_id"),
            fixed_equation_ids,
        ),
        "per_condition_weighted_gap_contribution": {
            "available": False,
            "reason": (
                "Kratos 10.3 Python does not expose a read-only per-condition "
                "WEIGHTED_GAP contribution; the process aggregates explicit "
                "condition contributions directly onto the slave node."
            ),
        },
    }


def _final_fields(context: Any, converged: bool) -> dict[str, Any]:
    KM, _, _, _ = _import_kratos()
    all_node_ids = [node.Id for node in context.model_part.Nodes]
    displacements, reactions = _nodal_fields(
        context.model_part, all_node_ids
    )
    displacement_values = [
        value for vector in displacements.values() for value in vector
    ]
    reaction_values = [
        value for vector in reactions.values() for value in vector
    ]
    try:
        det_f = pad_strain_det_f_statistics(
            context.mesh, displacements
        )["det_f"]
    except Exception as exception:
        det_f = {
            "available": False,
            "reason": f"{type(exception).__name__}: {exception}",
        }
    loading_direction = context.fixture.loading_direction
    reaction = compressive_indenter_reaction(
        reactions,
        context.indenter_topology.node_ids,
        loading_direction,
    )
    return {
        "converged": converged,
        "all_displacements_finite": all(
            math.isfinite(value) for value in displacement_values
        ),
        "all_reactions_finite": all(
            math.isfinite(value) for value in reaction_values
        ),
        "indenter_normal_reaction_n": _safe_float(reaction),
        "pad_det_f": det_f,
    }


def run_lifecycle_case(
    variant: CausalVariant,
    mesh_level: str = "medium",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one fresh first-step diagnostic with manual lifecycle capture."""
    KM, _, _, _ = _import_kratos()
    configuration = f"{variant.side}_only"
    snapshots: list[dict[str, Any]] = []
    context: Any | None = None
    initialized_step = False
    start = time.perf_counter()

    def capture_before(
        model_part: Any,
        _base_topology: Any,
        _indenter_topology: Any,
        _mesh: Any,
        _fixture: Any,
    ) -> None:
        endpoint_id = _endpoint_id(
            model_part, variant.side, slave=True
        )
        node = model_part.Nodes[endpoint_id]
        snapshots.append(
            {
                "variant": variant.name,
                "side": variant.side,
                "stage": "before_process_creation",
                "iteration": None,
                "endpoint_node_id": endpoint_id,
                "reference_coordinate_mm": [
                    float(node.X0),
                    float(node.Y0),
                ],
                "node_flags": {
                    "ACTIVE": bool(node.Is(KM.ACTIVE)),
                    "SLAVE": bool(node.Is(KM.SLAVE)),
                    "MASTER": bool(node.Is(KM.MASTER)),
                },
                "incident_generated_conditions": [],
            }
        )

    def capture(
        stage: str,
        iteration: int | None = None,
        matrix: Any | None = None,
        rhs: Any | None = None,
    ) -> None:
        snapshot = endpoint_snapshot(
            context,
            variant.side,
            stage,
            iteration,
            matrix,
            rhs,
        )
        snapshot["variant"] = variant.name
        snapshots.append(snapshot)

    result: dict[str, Any]
    dof_rows: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    exception_text: str | None = None
    converged = False
    iterations = 0
    invalid_ids: list[int] = []
    try:
        context = _build_context(
            mesh_level,
            configuration,
            before_initialize=capture_before,
        )
        capture("after_execute_initialize")
        runtime = _runtime_contract(context)
        solver = context.analysis._GetSolver()
        context.analysis.time = solver.AdvanceInTime(context.analysis.time)
        set_indenter_travel(
            context.model_part,
            context.indenter_topology.node_ids,
            context.fixture,
            FIRST_STEP_TRAVEL_MM,
        )

        context.analysis.ApplyBoundaryConditions()
        capture("after_contact_search")
        endpoint_id = _endpoint_id(
            context.model_part, variant.side, slave=True
        )
        pairs_after_search = endpoint_pair_records(
            context, 1, endpoint_id
        )
        invalid_ids = [
            int(record["generated_condition_id"])
            for record in pairs_after_search
            if record["out_of_domain_extra_pair"]
        ]
        if variant.force_invalid_pair_inactive:
            computing = context.model[
                "Structure.ComputingContact.ComputingContactSub1"
            ]
            for identifier in invalid_ids:
                computing.Conditions[identifier].Set(KM.ACTIVE, False)
            capture("after_diagnostic_invalid_pair_deactivation")

        context.analysis.ChangeMaterialProperties()
        solver.InitializeSolutionStep()
        initialized_step = True
        capture("after_initialize_solution_step")
        solver.Predict()
        capture("after_predict")

        strategy = solver._GetSolutionStrategy()
        builder = solver._GetBuilderAndSolver()
        scheme = solver._GetScheme()
        criterion = solver._GetConvergenceCriterion()
        computing_model_part = solver.GetComputingModelPart()
        dof_set = builder.GetDofSet()
        (
            dof_rows,
            _equation_map,
            _assembled_dofs,
            dof_summary,
        ) = _dof_records(context, dof_set)
        matrix = strategy.GetSystemMatrix()
        increment = strategy.GetSolutionVector()
        rhs = strategy.GetSystemVector()
        sparse_space = KM.UblasSparseSpace()

        for iteration in range(1, MAXIMUM_NEWTON_ITERATIONS + 1):
            iterations = iteration
            context.model_part.ProcessInfo[KM.NL_ITERATION_NUMBER] = iteration
            scheme.InitializeNonLinIteration(
                context.model_part, matrix, increment, rhs
            )
            criterion.InitializeNonLinearIteration(
                context.model_part, dof_set, matrix, increment, rhs
            )
            pre_converged = bool(
                criterion.PreCriteria(
                    context.model_part, dof_set, matrix, increment, rhs
                )
            )
            capture(
                "after_initialize_non_linear_iteration",
                iteration,
                matrix,
                rhs,
            )

            sparse_space.SetToZeroMatrix(matrix)
            sparse_space.SetToZeroVector(increment)
            sparse_space.SetToZeroVector(rhs)
            builder.BuildAndSolve(
                scheme,
                computing_model_part,
                matrix,
                increment,
                rhs,
            )
            capture("after_tangent_assembly", iteration, matrix, rhs)

            scheme.Update(
                context.model_part,
                dof_set,
                matrix,
                increment,
                rhs,
            )
            strategy.MoveMesh()
            scheme.FinalizeNonLinIteration(
                context.model_part, matrix, increment, rhs
            )
            criterion.FinalizeNonLinearIteration(
                context.model_part, dof_set, matrix, increment, rhs
            )
            capture(
                "after_finalize_non_linear_iteration",
                iteration,
                matrix,
                rhs,
            )

            post_converged = False
            if pre_converged:
                if iteration == 1:
                    criterion.InitializeSolutionStep(
                        context.model_part,
                        dof_set,
                        matrix,
                        increment,
                        rhs,
                    )
                if criterion.GetActualizeRHSflag():
                    sparse_space.SetToZeroVector(rhs)
                    builder.BuildRHS(scheme, computing_model_part, rhs)
                post_converged = bool(
                    criterion.PostCriteria(
                        context.model_part,
                        dof_set,
                        matrix,
                        increment,
                        rhs,
                    )
                )
            capture(
                "after_active_set_convergence_check",
                iteration,
                matrix,
                rhs,
            )
            if post_converged:
                converged = True
                break

        builder.CalculateReactions(
            scheme, computing_model_part, matrix, increment, rhs
        )
        capture("after_solve", iterations, matrix, rhs)
        pair_records, pair_purity = _contact_condition_records(context)
        final_fields = _final_fields(context, converged)
        result = {
            "phase": "4I-F",
            "variant": asdict(variant),
            "mesh_level": mesh_level,
            "status": "PASS" if converged else "FAIL",
            "solve_converged": converged,
            "newton_iterations": iterations,
            "failure_reason": (
                None if converged else "maximum_newton_iterations_exceeded"
            ),
            "diagnostic_mutation": {
                "invalid_pair_condition_ids": invalid_ids,
                "invalid_pair_forced_inactive": (
                    variant.force_invalid_pair_inactive
                ),
                "production_configuration_modified": False,
            },
            "runtime_contact_contract": runtime,
            "dof_summary": dof_summary,
            "pair_purity": pair_purity,
            "final_fields": final_fields,
            "wall_clock_seconds": time.perf_counter() - start,
        }
    except Exception as exception:
        exception_text = f"{type(exception).__name__}: {exception}"
        if context is not None:
            try:
                capture("after_solve_exception", iterations)
            except Exception:
                pass
        result = {
            "phase": "4I-F",
            "variant": asdict(variant),
            "mesh_level": mesh_level,
            "status": "FAIL",
            "solve_converged": False,
            "newton_iterations": iterations,
            "failure_reason": "diagnostic_or_solver_exception",
            "exception": exception_text,
            "traceback": traceback.format_exc(),
            "diagnostic_mutation": {
                "invalid_pair_condition_ids": invalid_ids,
                "invalid_pair_forced_inactive": (
                    variant.force_invalid_pair_inactive
                ),
                "production_configuration_modified": False,
            },
            "wall_clock_seconds": time.perf_counter() - start,
        }
    finally:
        if context is not None:
            try:
                if initialized_step:
                    context.analysis.FinalizeSolutionStep()
                context.analysis.Finalize()
            except Exception:
                pass

    return result, snapshots, pair_records


def unavailable_case_records() -> dict[str, dict[str, Any]]:
    """Document causal cases that cannot use a safe public Kratos API."""
    common = {
        "status": "UNAVAILABLE",
        "executed": False,
        "production_configuration_modified": False,
        "reason": (
            "ContactSearchProcess exposes no Python INDEX_MAP/pair-filter hook. "
            "Removing a generated condition alone would leave the C++ pairing "
            "map used by ClearMortarConditions inconsistent."
        ),
    }
    return {
        "F01": {
            **common,
            "requested_state": "invalid pair excluded before creation",
        },
        "F03": {
            **common,
            "requested_state": "valid generated pair only",
        },
        "symmetric_control": {
            **common,
            "status": "NOT_RUN",
            "requested_state": "symmetric invalid candidate insertion",
            "reason": (
                "No supported Python API inserts both a coupled mortar "
                "condition and the owning search INDEX_MAP entry atomically."
            ),
        },
    }


def source_trace() -> dict[str, Any]:
    """Return the exact Kratos 10.3 source/API contract used by the audit."""
    commit = "14ee273e97af403622699e797ea5fa356b1a7e60"
    root = (
        "https://github.com/KratosMultiphysics/Kratos/blob/"
        f"{commit}/"
    )
    return {
        "kratos_package_version": "10.3.x",
        "kernel_banner": "10.3.0-14ee273e",
        "upstream_commit": commit,
        "line2_local_domain": {
            "file": "kratos/geometries/line_2d_2.h",
            "class": "Line2D2",
            "method": "IsInside",
            "behavior": "accepts abs(local_xi) <= 1 + tolerance",
            "url": root + "kratos/geometries/line_2d_2.h",
            "python_geometry_is_inside_exposed": False,
        },
        "search_lifecycle": [
            {
                "file": (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_processes/base_contact_search_process.cpp"
                ),
                "class": "BaseContactSearchProcess",
                "methods": [
                    "ExecuteInitializeSolutionStep",
                    "UpdateMortarConditions",
                    "SearchUsingKDTree",
                    "CheckCondition",
                    "CheckPairing",
                    "CreateAuxiliaryConditions",
                    "AddPairing",
                    "ClearMortarConditions",
                ],
                "behavior": (
                    "KD-tree/OBB broad phase accepts candidate conditions; "
                    "CheckCondition rejects identity/normal/duplicates but does "
                    "not apply an endpoint Line2 IsInside test. AddPairing "
                    "creates the coupled condition and sets it ACTIVE."
                ),
                "url": root
                + (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_processes/base_contact_search_process.cpp"
                ),
            },
            {
                "file": (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_processes/advanced_contact_search_process.cpp"
                ),
                "class": "AdvancedContactSearchProcess",
                "methods": ["CheckPairing", "ComputeActiveInactiveNodes"],
                "behavior": (
                    "active/inactive state is computed at slave-node level "
                    "after gap accumulation, not as a per-pair exact-overlap flag"
                ),
                "url": root
                + (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_processes/advanced_contact_search_process.cpp"
                ),
            },
        ],
        "gap_and_active_set": [
            {
                "file": (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_utilities/active_set_utilities.cpp"
                ),
                "class": "ActiveSetUtilities",
                "method": "ComputeALMFrictionlessActiveSet",
                "behavior": (
                    "node augmented pressure is SCALE_FACTOR*LM + "
                    "INITIAL_PENALTY*WEIGHTED_GAP; negative means ACTIVE"
                ),
                "url": root
                + (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_utilities/active_set_utilities.cpp"
                ),
            },
            {
                "file": (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_strategies/custom_convergencecriterias/"
                    "base_mortar_criteria.h"
                ),
                "class": "BaseMortarConvergenceCriteria",
                "methods": [
                    "PostCriteria",
                    "FinalizeNonLinearIteration",
                ],
                "behavior": (
                    "WEIGHTED_GAP is reset and recomputed by explicit "
                    "contributions from ComputingContact conditions"
                ),
                "url": root
                + (
                    "applications/ContactStructuralMechanicsApplication/"
                    "custom_strategies/custom_convergencecriterias/"
                    "base_mortar_criteria.h"
                ),
            },
        ],
        "nonlinear_order": {
            "file": (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_strategies/custom_strategies/"
                "residualbased_newton_raphson_contact_strategy.h"
            ),
            "class": "ResidualBasedNewtonRaphsonContactStrategy",
            "method": "BaseSolveSolutionStep",
            "behavior": (
                "InitializeNonLinIteration -> BuildAndSolve -> UpdateDatabase "
                "-> FinalizeNonLinIteration -> PostCriteria(active set)"
            ),
            "url": root
            + (
                "applications/ContactStructuralMechanicsApplication/"
                "custom_strategies/custom_strategies/"
                "residualbased_newton_raphson_contact_strategy.h"
            ),
        },
        "python_api_limits": {
            "ContactSearchProcess_INDEX_MAP_exposed": False,
            "Geometry_IsInside_exposed": False,
            "safe_generated_condition_removal_hook": False,
            "safe_pair_insertion_hook": False,
            "exact_overlap_api": (
                "ExactMortarIntegrationUtility2D2N."
                "TestGetExactAreaIntegration"
            ),
        },
        "crosspoint_support_search": {
            "official_crosspoint_lm_boundary_treatment_found": False,
            "scope": (
                "ContactStructuralMechanicsApplication search, ALM active-set, "
                "mortar criteria, contact strategy, and ALM frictionless "
                "condition sources at the exact installed commit"
            ),
            "conclusion": (
                "No documented automatic elimination/stabilization rule for "
                "a slave LM at a contact/fully-Dirichlet crosspoint was found."
            ),
        },
    }
