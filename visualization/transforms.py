"""Scientific transformations independent of Matplotlib rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from visualization.data import (
    ContactCase,
    DisplacementField,
    ObservationChain,
    ScientificFigureError,
    TransferSignature,
    VisualizationDataset,
)


SUPPORTED_QUANTITIES = {
    "raw_displacement",
    "secant_gain",
    "force_compliance",
    "tangent_gain",
}


@dataclass(frozen=True)
class SelectedTransferState:
    """One semantic state selected exactly or by valid delta interpolation."""

    design_id: str
    mesh_id: str
    case_id: str
    xi: float
    delta_mm: float
    reaction_force_n: float | None
    contact_point_mm: tuple[float, float] | None
    indentation_direction: tuple[float, float]
    values_by_side: Mapping[str, np.ndarray]
    eta_by_side: Mapping[str, np.ndarray]
    displacement_by_side: Mapping[str, np.ndarray]
    quantity: str
    units: str
    normalization: str
    selection_metadata: Mapping[str, object]


def symmetric_limits(
    states_by_design: Mapping[str, Sequence[SelectedTransferState]],
) -> tuple[float, float]:
    """Return one signed scale covering every compared state."""
    maximum = max(
        abs(float(value))
        for states in states_by_design.values()
        for state in states
        for side in ("left", "right")
        for value in state.values_by_side[side]
    )
    if maximum <= 0.0:
        raise ScientificFigureError(
            "cannot create a signed color scale from zero data"
        )
    return -maximum, maximum


def project_outward_displacement(
    displacement: DisplacementField,
    chains: Mapping[str, ObservationChain],
) -> dict[str, np.ndarray]:
    """Project actual u vectors onto declared reference outward normals."""
    point_index = {point_id: index for index, point_id in enumerate(displacement.point_ids)}
    projected: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        chain = chains[side]
        missing = [point_id for point_id in chain.point_ids if point_id not in point_index]
        if missing:
            raise ScientificFigureError(
                f"displacement field misses {len(missing)} {side} points"
            )
        indices = np.asarray([point_index[point_id] for point_id in chain.point_ids])
        vectors = displacement.nodal_displacement[indices]
        if not displacement.validity_mask[indices].all():
            raise ScientificFigureError("cannot project an invalid displacement sample")
        projected[side] = np.einsum("ij,ij->i", vectors, chain.outward_normals)
    return projected


def quantity_values(
    signature: TransferSignature,
    quantity: str,
    *,
    denominator_floor: float = 1.0e-12,
) -> tuple[np.ndarray, str, str]:
    """Return a supported scientific quantity with units and normalization."""
    if quantity not in SUPPORTED_QUANTITIES:
        raise ScientificFigureError(f"unsupported transfer quantity {quantity!r}")
    if not signature.validity_mask.all():
        raise ScientificFigureError("transfer quantity requested from invalid samples")
    if quantity == "raw_displacement":
        return np.array(signature.u_normal, copy=True), "mm", "none"
    if quantity == "secant_gain":
        if signature.delta_mm <= denominator_floor:
            raise ScientificFigureError("secant gain requires positive indentation")
        calculated = signature.u_normal / signature.delta_mm
        if signature.stored_secant_gain is not None and not np.allclose(
            calculated, signature.stored_secant_gain, rtol=1.0e-12, atol=1.0e-12
        ):
            raise ScientificFigureError("stored secant gain violates u_normal/delta")
        return calculated, "mm/mm", "u_normal / prescribed indentation"
    if quantity == "force_compliance":
        if (
            signature.reaction_force_n is None
            or abs(signature.reaction_force_n) <= denominator_floor
        ):
            raise ScientificFigureError(
                "force compliance requires a finite nonzero reaction"
            )
        return (
            signature.u_normal / signature.reaction_force_n,
            "mm/N",
            "u_normal / reaction force",
        )
    if signature.stored_tangent_gain is None:
        raise ScientificFigureError("stored tangent gain is unavailable")
    return (
        np.array(signature.stored_tangent_gain, copy=True),
        "mm/mm",
        "stored finite difference du_normal/ddelta",
    )


def _select_case(
    states: Sequence[ContactCase],
    target_delta_mm: float,
    interpolation_enabled: bool,
) -> tuple[ContactCase, dict[str, object], float | None]:
    if not states:
        raise ScientificFigureError("no contact states available")
    delta = np.asarray([state.delta_mm for state in states])
    validity = np.asarray([state.codtm_valid for state in states])
    force = np.asarray(
        [
            state.reaction_force_n
            if state.reaction_force_n is not None
            else float("nan")
            for state in states
        ]
    )
    try:
        selected = select_indentation(
            delta, delta[:, None], validity, target_delta_mm
        )
    except CODTMVisualizationError as exc:
        raise ScientificFigureError(str(exc)) from exc
    if not selected.exact and not interpolation_enabled:
        raise ScientificFigureError(
            "target indentation is absent and interpolation is disabled"
        )
    representative = states[selected.lower_step_index]
    if selected.exact:
        reaction = (
            float(force[selected.lower_step_index])
            if np.isfinite(force[selected.lower_step_index])
            else None
        )
    elif np.isfinite(force[[selected.lower_step_index, selected.upper_step_index]]).all():
        reaction = float(
            (1.0 - selected.interpolation_weight)
            * force[selected.lower_step_index]
            + selected.interpolation_weight * force[selected.upper_step_index]
        )
    else:
        reaction = None
    return representative, selected.metadata(), reaction


def _select_signature(
    states: Sequence[TransferSignature],
    target_delta_mm: float,
    quantity: str,
    interpolation_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], str, str]:
    if not states:
        raise ScientificFigureError("no transfer-signature states available")
    delta = np.asarray([state.delta_mm for state in states])
    validity = np.asarray([state.validity_mask.all() for state in states])
    transformed = []
    units = ""
    normalization = ""
    for state in states:
        values, current_units, current_normalization = quantity_values(
            state, quantity
        )
        transformed.append(values)
        if units and units != current_units:
            raise ScientificFigureError("quantity units changed along one case")
        units = current_units
        normalization = current_normalization
    try:
        selected = select_indentation(
            delta, np.asarray(transformed), validity, target_delta_mm
        )
    except CODTMVisualizationError as exc:
        raise ScientificFigureError(str(exc)) from exc
    if not selected.exact and not interpolation_enabled:
        raise ScientificFigureError(
            "target indentation is absent and interpolation is disabled"
        )
    reference_eta = states[selected.lower_step_index].eta
    upper_eta = states[selected.upper_step_index].eta
    if not np.array_equal(reference_eta, upper_eta):
        raise ScientificFigureError("eta grid changes across interpolation bracket")
    return selected.values, reference_eta, selected.metadata(), units, normalization


def select_transfer_state(
    dataset: VisualizationDataset,
    *,
    design_id: str,
    mesh_id: str,
    xi: float,
    delta_mm: float,
    quantity: str,
    interpolation_enabled: bool = False,
) -> SelectedTransferState:
    """Select the same physical state for case, signatures, and vector field."""
    cases = dataset.case_states(design_id, mesh_id, xi)
    case, case_selection, reaction = _select_case(
        cases, delta_mm, interpolation_enabled
    )
    values: dict[str, np.ndarray] = {}
    eta: dict[str, np.ndarray] = {}
    signature_selection: dict[str, object] = {}
    units = ""
    normalization = ""
    for side in ("left", "right"):
        (
            values[side],
            eta[side],
            side_selection,
            current_units,
            current_normalization,
        ) = _select_signature(
            dataset.signature_states(design_id, mesh_id, xi, side),
            delta_mm,
            quantity,
            interpolation_enabled,
        )
        signature_selection[side] = side_selection
        if units and units != current_units:
            raise ScientificFigureError("left/right quantity units disagree")
        units = current_units
        normalization = current_normalization
    if not bool(case_selection["selection"] == "exact"):
        raise ScientificFigureError(
            "vector-field interpolation is intentionally unavailable in this adapter"
        )
    field = dataset.displacement_state(design_id, case.case_id, case.step)
    point_index = {point_id: index for index, point_id in enumerate(field.point_ids)}
    displacement_by_side: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        chain = dataset.chain(design_id, mesh_id, side)
        indices = [point_index[point_id] for point_id in chain.point_ids]
        displacement_by_side[side] = field.nodal_displacement[indices]
    return SelectedTransferState(
        design_id=design_id,
        mesh_id=mesh_id,
        case_id=case.case_id,
        xi=xi,
        delta_mm=delta_mm,
        reaction_force_n=reaction,
        contact_point_mm=case.contact_point_mm,
        indentation_direction=case.indentation_direction,
        values_by_side=values,
        eta_by_side=eta,
        displacement_by_side=displacement_by_side,
        quantity=quantity,
        units=units,
        normalization=normalization,
        selection_metadata={
            "case": case_selection,
            "signatures": signature_selection,
            "interpolation_enabled": interpolation_enabled,
            "xi_interpolation": False,
        },
    )


def stack_state_signatures(
    states: Sequence[SelectedTransferState],
    side_order: Sequence[str] = ("left", "right"),
) -> tuple[np.ndarray, np.ndarray]:
    """Stack location signatures while preserving explicit semantic sides."""
    if not states:
        raise ScientificFigureError("at least one selected state is required")
    signatures = []
    eta = []
    for side in side_order:
        reference_eta = np.asarray(states[0].eta_by_side[side])
        if any(
            not np.array_equal(reference_eta, state.eta_by_side[side])
            for state in states[1:]
        ):
            raise ScientificFigureError("selected states do not share one eta grid")
        eta.append(reference_eta)
    for state in states:
        signatures.append(
            np.stack([state.values_by_side[side] for side in side_order])
        )
    return np.asarray(signatures), np.asarray(eta)


def raw_distance_metrics(
    states: Sequence[SelectedTransferState],
) -> np.ndarray:
    signatures, eta = stack_state_signatures(states)
    return location_distance_matrix(signatures, eta)


def normalized_shape_distance_metrics(
    states: Sequence[SelectedTransferState],
) -> np.ndarray:
    signatures, eta = stack_state_signatures(states)
    try:
        return shape_distance_matrix(signatures, eta)
    except CODTMVisualizationError as exc:
        raise ScientificFigureError(str(exc)) from exc


def mirror_side_swap(
    values_by_side: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Mirror a signature by swapping sides while preserving eta ordering."""
    if set(values_by_side) != {"left", "right"}:
        raise ScientificFigureError("mirror mapping requires left and right sides")
    left = np.asarray(values_by_side["left"], dtype=float)
    right = np.asarray(values_by_side["right"], dtype=float)
    if left.shape != right.shape or not np.isfinite([left, right]).all():
        raise ScientificFigureError("mirror signature arrays are invalid")
    return {"left": np.array(right, copy=True), "right": np.array(left, copy=True)}


def transfer_summary_metrics(
    states: Sequence[SelectedTransferState],
) -> dict[str, float | None]:
    """Return comparison-ready scalar summaries without an observability claim."""
    if not states:
        raise ScientificFigureError("cannot summarize an empty state collection")
    signatures, eta = stack_state_signatures(states)
    distances = location_distance_matrix(signatures, eta)
    off_diagonal = distances[~np.eye(len(states), dtype=bool)]
    magnitudes = np.sqrt(np.mean(signatures**2, axis=(1, 2)))
    reactions = np.asarray(
        [
            state.reaction_force_n
            if state.reaction_force_n is not None
            else float("nan")
            for state in states
        ]
    )
    return {
        "minimum_pairwise_distance": float(off_diagonal.min())
        if off_diagonal.size
        else None,
        "mean_pairwise_distance": float(off_diagonal.mean())
        if off_diagonal.size
        else None,
        "gain_magnitude_rms_mean": float(magnitudes.mean()),
        "gain_uniformity_coefficient_of_variation": (
            float(magnitudes.std() / magnitudes.mean())
            if magnitudes.mean() > 1.0e-14
            else None
        ),
        "reaction_force_min_n": float(np.nanmin(reactions))
        if np.isfinite(reactions).any()
        else None,
        "reaction_force_max_n": float(np.nanmax(reactions))
        if np.isfinite(reactions).any()
        else None,
    }


def deterministic_spatial_subsample(
    coordinates: np.ndarray,
    *,
    maximum_count: int,
) -> np.ndarray:
    """Select points by spatial bins, independent of node/sample ordering."""
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ScientificFigureError("vector coordinates must be finite [n,2]")
    if maximum_count < 1:
        raise ScientificFigureError("maximum vector count must be positive")
    if len(points) <= maximum_count:
        return np.lexsort((points[:, 1], points[:, 0]))
    span = np.ptp(points, axis=0)
    active_dimensions = max(1, int(np.count_nonzero(span > 1.0e-14)))
    bins_per_dimension = max(
        1, int(math.ceil(maximum_count ** (1.0 / active_dimensions)))
    )
    normalized = np.zeros_like(points)
    for dimension in range(2):
        if span[dimension] > 1.0e-14:
            normalized[:, dimension] = (
                points[:, dimension] - points[:, dimension].min()
            ) / span[dimension]
    bins = np.minimum(
        (normalized * bins_per_dimension).astype(int), bins_per_dimension - 1
    )
    candidates: list[tuple[tuple[int, int], int]] = []
    for bin_key in sorted({tuple(row) for row in bins}):
        members = np.flatnonzero(np.all(bins == bin_key, axis=1))
        center = (np.asarray(bin_key, dtype=float) + 0.5) / bins_per_dimension
        distances = np.linalg.norm(normalized[members] - center, axis=1)
        best_distance = float(distances.min())
        tied = members[np.isclose(distances, best_distance, atol=1.0e-15)]
        best = min(tied, key=lambda index: (points[index, 0], points[index, 1]))
        candidates.append((bin_key, int(best)))
    selected = [index for _, index in candidates]
    if len(selected) > maximum_count:
        chosen_positions = np.linspace(
            0, len(selected) - 1, maximum_count, dtype=int
        )
        selected = [selected[position] for position in chosen_positions]
    return np.asarray(
        sorted(selected, key=lambda index: (points[index, 0], points[index, 1])),
        dtype=int,
    )
class CODTMVisualizationError(RuntimeError):
    """Raised when an artifact or requested visualization is not valid."""


@dataclass(frozen=True)
class IndentationSelection:
    """An exact or linearly interpolated value at one indentation."""

    target_mm: float
    values: np.ndarray
    exact: bool
    lower_step_index: int
    upper_step_index: int
    lower_delta_mm: float
    upper_delta_mm: float
    interpolation_weight: float

    def metadata(self) -> dict[str, Any]:
        return {
            "target_mm": self.target_mm,
            "selection": "exact" if self.exact else "linear_interpolation",
            "lower_step": self.lower_step_index + 1,
            "upper_step": self.upper_step_index + 1,
            "lower_delta_mm": self.lower_delta_mm,
            "upper_delta_mm": self.upper_delta_mm,
            "interpolation_weight": self.interpolation_weight,
        }
def select_indentation(
    delta_mm: Sequence[float] | np.ndarray,
    values: np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
    target_mm: float,
    *,
    tolerance: float = 1.0e-10,
) -> IndentationSelection:
    """Select an exact step or interpolate between two valid finite steps."""
    delta = np.asarray(delta_mm, dtype=float)
    data = np.asarray(values, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if (
        delta.ndim != 1
        or data.shape[0] != delta.size
        or valid.shape != delta.shape
        or not math.isfinite(target_mm)
    ):
        raise CODTMVisualizationError("invalid indentation selection inputs")
    exact = np.flatnonzero(np.isclose(delta, target_mm, rtol=0.0, atol=tolerance))
    if exact.size:
        index = int(exact[0])
        if not valid[index] or not np.isfinite(data[index]).all():
            raise CODTMVisualizationError("exact indentation step is invalid")
        return IndentationSelection(
            target_mm=float(target_mm),
            values=np.array(data[index], copy=True),
            exact=True,
            lower_step_index=index,
            upper_step_index=index,
            lower_delta_mm=float(delta[index]),
            upper_delta_mm=float(delta[index]),
            interpolation_weight=0.0,
        )
    if target_mm < float(np.min(delta)) or target_mm > float(np.max(delta)):
        raise CODTMVisualizationError("indentation extrapolation is forbidden")
    upper = int(np.searchsorted(delta, target_mm, side="right"))
    lower = upper - 1
    if lower < 0 or upper >= delta.size or not delta[lower] < target_mm < delta[upper]:
        raise CODTMVisualizationError("indentation coordinates are not strictly ordered")
    if not valid[lower] or not valid[upper]:
        raise CODTMVisualizationError("cannot interpolate across an invalid step")
    if not np.isfinite(data[[lower, upper]]).all():
        raise CODTMVisualizationError("cannot interpolate across a non-finite step")
    weight = float((target_mm - delta[lower]) / (delta[upper] - delta[lower]))
    interpolated = (1.0 - weight) * data[lower] + weight * data[upper]
    return IndentationSelection(
        target_mm=float(target_mm),
        values=np.asarray(interpolated, dtype=float),
        exact=False,
        lower_step_index=lower,
        upper_step_index=upper,
        lower_delta_mm=float(delta[lower]),
        upper_delta_mm=float(delta[upper]),
        interpolation_weight=weight,
    )
def _sorted_side(
    values: np.ndarray,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(eta)
    return np.asarray(values)[..., order], np.asarray(eta)[order]


def signature_inner_product(
    first: np.ndarray,
    second: np.ndarray,
    eta_by_side: np.ndarray,
) -> float:
    """Integrate each side separately, then sum; the center gap has no length."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    eta = np.asarray(eta_by_side, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or eta.shape != a.shape:
        raise CODTMVisualizationError("signature/eta shapes must be [side, sample]")
    total = 0.0
    for side_index in range(a.shape[0]):
        difference_a, coordinate = _sorted_side(a[side_index], eta[side_index])
        difference_b, _ = _sorted_side(b[side_index], eta[side_index])
        total += float(np.trapezoid(difference_a * difference_b, coordinate))
    return total


def signature_norm(values: np.ndarray, eta_by_side: np.ndarray) -> float:
    norm_squared = signature_inner_product(values, values, eta_by_side)
    return math.sqrt(max(norm_squared, 0.0))


def location_distance_matrix(
    signatures: np.ndarray,
    eta_by_side: np.ndarray,
) -> np.ndarray:
    fields = np.asarray(signatures, dtype=float)
    if fields.ndim != 3:
        raise CODTMVisualizationError("signatures must have [location, side, sample]")
    count = fields.shape[0]
    result = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(first + 1, count):
            distance = signature_norm(fields[first] - fields[second], eta_by_side)
            result[first, second] = distance
            result[second, first] = distance
    return result


def shape_distance_matrix(
    signatures: np.ndarray,
    eta_by_side: np.ndarray,
    *,
    norm_floor: float = 1.0e-12,
) -> np.ndarray:
    fields = np.asarray(signatures, dtype=float)
    normalized: list[np.ndarray] = []
    for field in fields:
        norm = signature_norm(field, eta_by_side)
        if not math.isfinite(norm) or norm <= norm_floor:
            raise CODTMVisualizationError("shape normalization encountered zero norm")
        normalized.append(field / norm)
    return location_distance_matrix(np.asarray(normalized), eta_by_side)
