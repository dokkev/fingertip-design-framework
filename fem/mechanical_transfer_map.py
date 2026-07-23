"""Phase 4K contact-to-observation deformation transfer measurements.

This module is intentionally a post-processing layer.  It does not create a
Kratos strategy, change contact parameters, or alter the adopted Phase 4J
material and mesh contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point

from fem.indentation_analysis import ConvergedIndentationStep
from fem.indentation_postprocess import IndentationPostprocessError
from fem.mesh_types import BoundaryEdge, FingertipMesh
from model.fingertip_model import FingertipModel

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class TransferMapSettings:
    """Coordinate and numerical conventions for the Phase 4K extractor."""

    n_observation_samples: int = 41
    observation_right_xi_start: float = 0.0
    observation_right_xi_end: float = 0.25
    observation_left_xi_start: float = 1.0
    observation_left_xi_end: float = 0.75
    force_closure_relative_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if (
            not isinstance(self.n_observation_samples, int)
            or isinstance(self.n_observation_samples, bool)
            or self.n_observation_samples < 2
        ):
            raise ValueError("n_observation_samples must be an integer of at least two")
        values = (
            self.observation_right_xi_start,
            self.observation_right_xi_end,
            self.observation_left_xi_start,
            self.observation_left_xi_end,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("observation xi bounds must be finite and in [0, 1]")
        if self.observation_right_xi_start >= self.observation_right_xi_end:
            raise ValueError("right observation coordinate must increase")
        if self.observation_left_xi_start <= self.observation_left_xi_end:
            raise ValueError("left observation coordinate must decrease")
        if (
            not math.isfinite(self.force_closure_relative_tolerance)
            or self.force_closure_relative_tolerance <= 0.0
        ):
            raise ValueError("force closure tolerance must be finite and positive")


@dataclass(frozen=True)
class ReferenceBoundaryChain:
    """One ordered open Line2 chain in the undeformed configuration."""

    node_ids: tuple[int, ...]
    points_mm: tuple[Vector2, ...]
    cumulative_length_mm: tuple[float, ...]
    total_length_mm: float

    @property
    def normalized_coordinates(self) -> tuple[float, ...]:
        return tuple(
            value / self.total_length_mm for value in self.cumulative_length_mm
        )


def order_open_edge_chain(
    edges: Sequence[tuple[int, int]],
    node_coordinates: Mapping[int, Sequence[float]],
    source_start: Sequence[float],
) -> tuple[int, ...]:
    """Order an undirected open edge chain independently of IDs and edge order."""
    if not edges:
        raise IndentationPostprocessError("open edge chain is empty")
    adjacency: dict[int, list[int]] = {}
    for first, second in edges:
        if first == second:
            raise IndentationPostprocessError("open edge chain contains a self edge")
        if first not in node_coordinates or second not in node_coordinates:
            raise IndentationPostprocessError("edge references an unknown node")
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    endpoints = [
        node_id for node_id, neighbours in adjacency.items() if len(neighbours) == 1
    ]
    if len(endpoints) != 2 or any(
        len(neighbours) > 2 for neighbours in adjacency.values()
    ):
        raise IndentationPostprocessError(
            "boundary is not one connected open Line2 chain"
        )
    start_xy = (float(source_start[0]), float(source_start[1]))
    start = min(
        endpoints,
        key=lambda node_id: math.dist(
            start_xy,
            (
                float(node_coordinates[node_id][0]),
                float(node_coordinates[node_id][1]),
            ),
        ),
    )
    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = [
            neighbour for neighbour in adjacency[current] if neighbour != previous
        ]
        if not candidates:
            break
        if len(candidates) != 1:
            raise IndentationPostprocessError(
                f"boundary branches at node {current}"
            )
        previous, current = current, candidates[0]
        ordered.append(current)
    if len(ordered) != len(adjacency):
        raise IndentationPostprocessError("boundary contains disconnected edges")
    return tuple(ordered)


def reference_outer_arc_chain(
    model: FingertipModel,
    mesh: FingertipMesh,
) -> ReferenceBoundaryChain:
    """Build the semantic PadOuterArc reference coordinate."""
    source = model.boundaries.segments["pad_outer_arc"].geometry
    coordinates = {
        node_id: (node.x_mm, node.y_mm) for node_id, node in mesh.nodes.items()
    }
    ordered = order_open_edge_chain(
        [edge.node_ids for edge in mesh.boundary_edges["pad_outer_arc"]],
        coordinates,
        source.coords[0],
    )
    points = tuple(
        (float(mesh.nodes[node_id].x_mm), float(mesh.nodes[node_id].y_mm))
        for node_id in ordered
    )
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    if cumulative[-1] <= 0.0:
        raise IndentationPostprocessError("PadOuterArc has zero reference length")
    return ReferenceBoundaryChain(
        node_ids=ordered,
        points_mm=points,
        cumulative_length_mm=tuple(cumulative),
        total_length_mm=cumulative[-1],
    )


def _normalized(vector: Sequence[float]) -> Vector2:
    length = math.hypot(float(vector[0]), float(vector[1]))
    if not math.isfinite(length) or length <= 0.0:
        raise IndentationPostprocessError("cannot normalize a zero vector")
    return float(vector[0]) / length, float(vector[1]) / length


def _outward_normal(
    model: FingertipModel,
    point: Vector2,
    tangent: Vector2,
) -> Vector2:
    candidates = ((-tangent[1], tangent[0]), (tangent[1], -tangent[0]))
    probe = max(1.0e-4, 1000.0 * model.parameters.geometry_tolerance)
    outside = [
        candidate
        for candidate in candidates
        if not model.pad_material_geometry.covers(
            Point(
                point[0] + probe * candidate[0],
                point[1] + probe * candidate[1],
            )
        )
    ]
    if len(outside) == 1:
        return _normalized(outside[0])
    interior = model.pad_material_geometry.representative_point()
    radial = (point[0] - interior.x, point[1] - interior.y)
    selected = max(
        candidates,
        key=lambda candidate: candidate[0] * radial[0]
        + candidate[1] * radial[1],
    )
    return _normalized(selected)


def _interpolation_support(
    chain: ReferenceBoundaryChain,
    normalized_coordinate: float,
) -> tuple[int, int, float, Vector2, Vector2]:
    if (
        not math.isfinite(normalized_coordinate)
        or normalized_coordinate < 0.0
        or normalized_coordinate > 1.0
    ):
        raise ValueError("normalized coordinate must lie in [0, 1]")
    target = normalized_coordinate * chain.total_length_mm
    cumulative = np.asarray(chain.cumulative_length_mm, dtype=float)
    segment = min(
        int(np.searchsorted(cumulative, target, side="right") - 1),
        len(chain.node_ids) - 2,
    )
    segment = max(0, segment)
    first_length = cumulative[segment]
    second_length = cumulative[segment + 1]
    weight = float((target - first_length) / (second_length - first_length))
    first = chain.points_mm[segment]
    second = chain.points_mm[segment + 1]
    point = (
        (1.0 - weight) * first[0] + weight * second[0],
        (1.0 - weight) * first[1] + weight * second[1],
    )
    tangent = _normalized((second[0] - first[0], second[1] - first[1]))
    return (
        chain.node_ids[segment],
        chain.node_ids[segment + 1],
        weight,
        point,
        tangent,
    )


def interpolate_linear_chain_field(
    chain: ReferenceBoundaryChain,
    values: Mapping[int, Sequence[float]],
    normalized_coordinate: float,
) -> tuple[Vector2, Vector2, Vector2]:
    """Interpolate a vector and return ``(value, point, native_tangent)``."""
    first, second, weight, point, tangent = _interpolation_support(
        chain, normalized_coordinate
    )
    value = (
        (1.0 - weight) * float(values[first][0])
        + weight * float(values[second][0]),
        (1.0 - weight) * float(values[first][1])
        + weight * float(values[second][1]),
    )
    return value, point, tangent


def observation_boundary_contract(
    settings: TransferMapSettings,
) -> dict[str, Any]:
    """Describe the bounded semantic rule used in the absence of named sides."""
    return {
        "source_named_boundary": "PadOuterArc",
        "source_geometry": "FingertipModel.boundaries.segments['pad_outer_arc']",
        "rule_type": "configured subranges of undeformed semantic arc coordinate",
        "right": {
            "full_arc_xi_at_eta_0": settings.observation_right_xi_start,
            "full_arc_xi_at_eta_1": settings.observation_right_xi_end,
        },
        "left": {
            "full_arc_xi_at_eta_0": settings.observation_left_xi_start,
            "full_arc_xi_at_eta_1": settings.observation_left_xi_end,
        },
        "eta_orientation": (
            "eta=0 at the corresponding bonded diameter endpoint and eta=1 "
            "toward the crown on both sides"
        ),
        "selection_rationale": (
            "outer quarter-arcs adjacent to the two bonded endpoints are the "
            "bounded camera-visible sidewall candidates; no node-ID or current-"
            "coordinate threshold is used"
        ),
        "sample_count_per_side": settings.n_observation_samples,
        "interpolation": "linear Line2 shape functions in reference arc length",
    }


def sample_observation_sidewalls(
    model: FingertipModel,
    chain: ReferenceBoundaryChain,
    displacements: Mapping[int, Sequence[float]],
    settings: TransferMapSettings,
) -> dict[str, list[dict[str, Any]]]:
    """Sample fixed material coordinates on the two observation sidewalls."""
    output: dict[str, list[dict[str, Any]]] = {}
    eta_values = np.linspace(0.0, 1.0, settings.n_observation_samples)
    bounds = {
        "right": (
            settings.observation_right_xi_start,
            settings.observation_right_xi_end,
        ),
        "left": (
            settings.observation_left_xi_start,
            settings.observation_left_xi_end,
        ),
    }
    for side, (start, end) in bounds.items():
        records: list[dict[str, Any]] = []
        orientation = 1.0 if end > start else -1.0
        for eta in eta_values:
            full_xi = start + float(eta) * (end - start)
            displacement, point, native_tangent = interpolate_linear_chain_field(
                chain, displacements, full_xi
            )
            tangent = (
                orientation * native_tangent[0],
                orientation * native_tangent[1],
            )
            outward = _outward_normal(model, point, tangent)
            ux, uy = displacement
            records.append(
                {
                    "side": side,
                    "eta": float(eta),
                    "full_outer_arc_xi": full_xi,
                    "reference_x_mm": point[0],
                    "reference_y_mm": point[1],
                    "reference_tangent_x": tangent[0],
                    "reference_tangent_y": tangent[1],
                    "reference_outward_normal_x": outward[0],
                    "reference_outward_normal_y": outward[1],
                    "ux_mm": ux,
                    "uy_mm": uy,
                    "u_normal_mm": ux * outward[0] + uy * outward[1],
                    "u_tangent_mm": ux * tangent[0] + uy * tangent[1],
                    "deformed_x_mm": point[0] + ux,
                    "deformed_y_mm": point[1] + uy,
                }
            )
        output[side] = records
    return output


def integrate_nodal_contact_distribution(
    xi: Sequence[float],
    pressure: Sequence[float],
    nodal_area: Sequence[float],
    active: Sequence[bool],
    pressure_tolerance: float,
    thickness_mm: float = 1.0,
    global_projection_factors: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Lump one nodal pressure field using the Kratos ALM debug convention."""
    arrays = [
        np.asarray(values)
        for values in (xi, pressure, nodal_area, active)
    ]
    projection = (
        np.ones_like(arrays[0], dtype=float)
        if global_projection_factors is None
        else np.asarray(global_projection_factors, dtype=float)
    )
    arrays.append(projection)
    if not arrays or any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("contact arrays must have one equal shape")
    if not math.isfinite(pressure_tolerance) or pressure_tolerance <= 0.0:
        raise ValueError("pressure_tolerance must be finite and positive")
    if not math.isfinite(thickness_mm) or thickness_mm <= 0.0:
        raise ValueError("thickness_mm must be finite and positive")
    xi_values = arrays[0].astype(float)
    pressure_values = arrays[1].astype(float)
    areas = arrays[2].astype(float)
    active_values = arrays[3].astype(bool)
    finite = (
        np.isfinite(xi_values)
        & np.isfinite(pressure_values)
        & np.isfinite(areas)
        & np.isfinite(projection)
    )
    compressive = active_values & finite & (pressure_values > pressure_tolerance)
    normal_weights = np.where(
        compressive,
        pressure_values * areas * thickness_mm,
        0.0,
    )
    projected_weights = normal_weights * projection
    force = float(np.sum(projected_weights))
    normal_magnitude = float(np.sum(normal_weights))
    length = float(np.sum(np.where(compressive, areas, 0.0)))
    centroid = (
        float(np.sum(normal_weights * xi_values) / normal_magnitude)
        if normal_magnitude > 0.0
        else None
    )
    return {
        "finite": bool(np.all(finite)),
        "compressive_node_count": int(np.count_nonzero(compressive)),
        "integrated_contact_resultant_n": force,
        "integrated_contact_normal_magnitude_n": normal_magnitude,
        "contact_length_mm": length,
        "xi_centroid": centroid,
        "weighting": (
            "sum(-AUGMENTED_NORMAL_CONTACT_PRESSURE * NODAL_AREA * "
            "plane_strain_thickness * (-slave_normal dot global_loading)) "
            "on ACTIVE slave nodes; centroid uses unprojected compressive "
            "normal-traction weights"
        ),
    }


def _node_full_arc_xi(
    chain: ReferenceBoundaryChain,
) -> dict[int, float]:
    return {
        node_id: cumulative / chain.total_length_mm
        for node_id, cumulative in zip(
            chain.node_ids, chain.cumulative_length_mm
        )
    }


def runtime_contact_storage_audit(
    step: ConvergedIndentationStep,
) -> dict[str, Any]:
    """Probe candidate variables on the actual external slave nodes."""
    import KratosMultiphysics as KM
    import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA

    slave = step.model_part.GetSubModelPart("PadOuterArc")
    candidates = (
        ("KM.DISPLACEMENT", KM.DISPLACEMENT),
        ("KM.REACTION", KM.REACTION),
        ("KM.CONTACT_FORCE", KM.CONTACT_FORCE),
        ("KM.CONTACT_PRESSURE", KM.CONTACT_PRESSURE),
        ("KM.NORMAL_CONTACT_STRESS", KM.NORMAL_CONTACT_STRESS),
        ("KM.CONTACT_NORMAL", KM.CONTACT_NORMAL),
        ("KM.NORMAL", KM.NORMAL),
        ("CSMA.NORMAL_GAP", CSMA.NORMAL_GAP),
        ("CSMA.WEIGHTED_GAP", CSMA.WEIGHTED_GAP),
        (
            "CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE",
            CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE,
        ),
        (
            "CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE",
            CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE,
        ),
        ("KM.NODAL_AREA", KM.NODAL_AREA),
    )
    rows: list[dict[str, Any]] = []
    for name, variable in candidates:
        historical = sum(
            bool(node.SolutionStepsDataHas(variable)) for node in slave.Nodes
        )
        nonhistorical = sum(bool(node.Has(variable)) for node in slave.Nodes)
        rows.append(
            {
                "variable": name,
                "slave_node_count": slave.NumberOfNodes(),
                "historical_node_count": historical,
                "nonhistorical_node_count": nonhistorical,
            }
        )
    return {
        "external_slave_model_part": "Structure.PadOuterArc",
        "runtime_slave_node_count": slave.NumberOfNodes(),
        "runtime_slave_active_count": sum(
            bool(node.Is(KM.ACTIVE)) for node in slave.Nodes
        ),
        "variables": rows,
        "selected_distribution": {
            "pressure": "CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE",
            "pressure_storage": "nodal non-historical",
            "quadrature_weight": "KM.NODAL_AREA",
            "quadrature_weight_storage": "nodal non-historical",
            "role_filter": "PadOuterArc SLAVE and ACTIVE flags",
            "compression_sign": "negative Kratos augmented pressure; p_plus=-value",
            "global_force_projection": (
                "-historical KM.NORMAL dot unchanged global loading direction"
            ),
        },
    }


def extract_contact_distribution(
    step: ConvergedIndentationStep,
    chain: ReferenceBoundaryChain,
    canonical_reaction_n: float,
    transfer_settings: TransferMapSettings,
) -> dict[str, Any]:
    """Extract a force-closure-checked external contact descriptor."""
    import KratosMultiphysics as KM
    import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA

    slave = step.model_part.GetSubModelPart("PadOuterArc")
    xi_by_node = _node_full_arc_xi(chain)
    # The threshold is explicit and scales the existing Phase 4J force
    # tolerance by the available 2D integration length and unit thickness.
    pressure_tolerance = (
        step.settings.numerical_force_tolerance_n
        / (chain.total_length_mm * 1.0)
    )
    rows: list[dict[str, Any]] = []
    for node in slave.Nodes:
        augmented = (
            float(node.GetValue(CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE))
            if node.Has(CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE)
            else math.nan
        )
        area = (
            float(node.GetValue(KM.NODAL_AREA))
            if node.Has(KM.NODAL_AREA)
            else math.nan
        )
        lm = (
            float(
                node.GetSolutionStepValue(
                    CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
                )
            )
            if node.SolutionStepsDataHas(
                CSMA.LAGRANGE_MULTIPLIER_CONTACT_PRESSURE
            )
            else math.nan
        )
        weighted_gap = (
            float(node.GetSolutionStepValue(CSMA.WEIGHTED_GAP))
            if node.SolutionStepsDataHas(CSMA.WEIGHTED_GAP)
            else math.nan
        )
        normal = node.GetSolutionStepValue(KM.NORMAL)
        projection_factor = max(
            -(
                float(normal[0]) * step.fixture.frame.loading_direction[0]
                + float(normal[1]) * step.fixture.frame.loading_direction[1]
            ),
            0.0,
        )
        rows.append(
            {
                "node_id": int(node.Id),
                "xi": xi_by_node[int(node.Id)],
                "active": bool(node.Is(KM.ACTIVE)),
                "augmented_normal_contact_pressure": augmented,
                "p_plus": max(-augmented, 0.0) if math.isfinite(augmented) else math.nan,
                "lagrange_multiplier_contact_pressure": lm,
                "weighted_gap": weighted_gap,
                "nodal_area": area,
                "slave_normal": [float(normal[0]), float(normal[1])],
                "global_loading_projection_factor": projection_factor,
            }
        )
    integrated = integrate_nodal_contact_distribution(
        [row["xi"] for row in rows],
        [row["p_plus"] for row in rows],
        [row["nodal_area"] for row in rows],
        [row["active"] for row in rows],
        pressure_tolerance,
        global_projection_factors=[
            row["global_loading_projection_factor"] for row in rows
        ],
    )
    force = float(integrated["integrated_contact_resultant_n"])
    force_floor = step.settings.force_floor_n
    force_error = (
        abs(force - canonical_reaction_n)
        / max(abs(canonical_reaction_n), force_floor)
    )
    load_bearing = canonical_reaction_n > step.settings.numerical_force_tolerance_n
    verified = (
        integrated["finite"]
        and load_bearing
        and force > step.settings.numerical_force_tolerance_n
        and force_error <= transfer_settings.force_closure_relative_tolerance
    )
    if not load_bearing:
        verification = "NOT_APPLICABLE_NO_LOAD_BEARING_CONTACT"
    elif verified:
        verification = "VERIFIED"
    else:
        verification = "UNVERIFIED_FORCE_CLOSURE"
    return {
        "active": bool(integrated["compressive_node_count"]),
        "pressure_tolerance": {
            "value": pressure_tolerance,
            "unit": "N/mm^2 under the repository mm-N-MPa convention",
            "formula": (
                "IndentationSettings.numerical_force_tolerance_n / "
                "(PadOuterArc reference length * 1 mm thickness)"
            ),
        },
        "canonical_reaction_n": canonical_reaction_n,
        "integrated_contact_resultant_n": force,
        "integrated_contact_normal_magnitude_n": integrated[
            "integrated_contact_normal_magnitude_n"
        ],
        "force_closure_relative_error": force_error,
        "force_closure_tolerance": (
            transfer_settings.force_closure_relative_tolerance
        ),
        "verification": verification,
        "xi_centroid": integrated["xi_centroid"] if verified else None,
        "contact_length_mm": (
            integrated["contact_length_mm"] if verified else None
        ),
        "candidate_xi_centroid": integrated["xi_centroid"],
        "candidate_contact_length_mm": integrated["contact_length_mm"],
        "compressive_node_count": integrated["compressive_node_count"],
        "distribution_finite": integrated["finite"],
        "integration_configuration": (
            "Kratos nodal NODAL_AREA lumping; source is the current runtime "
            "contact model rather than a reconstructed reference-edge integral; "
            "force closure uses the nodal slave normal projected onto the "
            "unchanged global loading direction"
        ),
        "plane_strain_interpretation": (
            "contact_length_mm is a 2D active length, not a 3D area"
        ),
        "nodal_distribution": rows,
    }


class CODTMStepRecorder:
    """Callable converged-step observer with cumulative reaction work."""

    def __init__(
        self,
        case_name: str,
        xi_cmd: float,
        settings: TransferMapSettings | None = None,
    ) -> None:
        if not math.isfinite(xi_cmd) or not 0.0 <= xi_cmd <= 1.0:
            raise ValueError("xi_cmd must be finite and lie in [0, 1]")
        self.case_name = case_name
        self.xi_cmd = xi_cmd
        self.settings = settings or TransferMapSettings()
        self.records: list[dict[str, Any]] = []
        self.storage_audit: dict[str, Any] | None = None
        self._chain: ReferenceBoundaryChain | None = None
        self._last_delta = 0.0
        self._last_force = 0.0
        self._work = 0.0
        self._max_iterations = 0

    @property
    def chain(self) -> ReferenceBoundaryChain:
        if self._chain is None:
            raise RuntimeError("recorder has not observed a step")
        return self._chain

    def __call__(
        self, step: ConvergedIndentationStep
    ) -> Mapping[str, Any]:
        if self._chain is None:
            self._chain = reference_outer_arc_chain(
                step.fingertip_model, step.mesh
            )
            self.storage_audit = runtime_contact_storage_audit(step)
        delta = float(step.result_point["achieved_indentation_mm"])
        force = float(step.result_point["indenter_normal_reaction_n"])
        self._work += 0.5 * (self._last_force + force) * (
            delta - self._last_delta
        )
        self._last_delta = delta
        self._last_force = force
        self._max_iterations = max(
            self._max_iterations,
            int(step.result_point["nonlinear_iterations"]),
        )
        sidewalls = sample_observation_sidewalls(
            step.fingertip_model,
            self.chain,
            step.displacements,
            self.settings,
        )
        contact = extract_contact_distribution(
            step,
            self.chain,
            force,
            self.settings,
        )
        det_f = step.result_point["pad_strain_det_f"]["det_f"]
        strain = step.result_point["pad_strain_det_f"][
            "maximum_principal_green_lagrange_strain"
        ]
        record = {
            "case_name": self.case_name,
            "step": int(step.result_point["step"]),
            "delta_n_mm": delta,
            "xi_cmd": self.xi_cmd,
            "canonical_normal_reaction_n": force,
            "contact": contact,
            "observation_sidewalls": sidewalls,
            "minimum_det_f": float(det_f["min"]),
            "minimum_det_f_element_id": int(det_f["minimum_element_id"]),
            "minimum_det_f_reference_coordinate_mm": list(
                det_f["minimum_reference_coordinate_mm"]
            ),
            "canonical_strain_metric": {
                "name": "maximum principal Green-Lagrange strain",
                "value": float(strain["value"]),
                "element_id": int(strain["element_id"]),
                "reference_coordinate_mm": list(
                    strain["reference_coordinate_mm"]
                ),
            },
            "nonlinear_iterations": int(
                step.result_point["nonlinear_iterations"]
            ),
            "maximum_nonlinear_iterations_to_step": self._max_iterations,
            "solver_converged": bool(step.result_point["solver_converged"]),
            "active_set_converged": bool(
                step.result_point["active_set_converged"]
            ),
            "finite_fields": bool(step.result_point["finite_fields"]),
            "external_reaction_work_n_mm": self._work,
            "solve_wall_clock_seconds": float(
                step.result_point["solve_wall_clock_seconds"]
            ),
            "elapsed_case_wall_clock_seconds": float(
                step.elapsed_case_seconds
            ),
            "displacement_reference": (
                "raw displacement from the undeformed configuration; "
                "delta_reference=0 mm"
            ),
        }
        self.records.append(record)
        # The case history carries a compact index; the full records are
        # written separately by the Phase 4K runner.
        return {
            "record_index": len(self.records) - 1,
            "xi_cmd": self.xi_cmd,
            "xi_centroid": contact["xi_centroid"],
            "contact_length_mm": contact["contact_length_mm"],
            "integrated_contact_resultant_n": contact[
                "integrated_contact_resultant_n"
            ],
            "force_closure_relative_error": contact[
                "force_closure_relative_error"
            ],
            "contact_distribution_verification": contact["verification"],
            "external_reaction_work_n_mm": self._work,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "xi_cmd": self.xi_cmd,
            "settings": asdict(self.settings),
            "observation_boundary": observation_boundary_contract(
                self.settings
            ),
            "contact_coordinate": {
                "source": "undeformed semantic PadOuterArc",
                "orientation": (
                    "xi=0 right bonded endpoint, xi=0.5 crown, "
                    "xi=1 left bonded endpoint"
                ),
                "chain_node_count": (
                    len(self.chain.node_ids) if self._chain is not None else None
                ),
                "reference_polyline_length_mm": (
                    self.chain.total_length_mm
                    if self._chain is not None
                    else None
                ),
            },
            "runtime_storage_audit": self.storage_audit,
        }


def strict_json_round_trip(value: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise the exact strict serialization policy used by Phase 4K."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def edge_source_from_mesh(
    mesh: FingertipMesh,
    tag: str,
) -> tuple[BoundaryEdge, ...]:
    """Small public adapter retained for checkpoint/extractor reuse."""
    return tuple(mesh.boundary_edges[tag])
