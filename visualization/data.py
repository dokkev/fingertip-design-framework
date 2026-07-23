"""Canonical semantic data for scientific figures."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


FRAMEWORK_VERSION = "1.0.0"


class ScientificFigureError(RuntimeError):
    """Raised when semantic figure data violates its declared contract."""


def finite_array(
    values: np.ndarray | Sequence[float],
    *,
    name: str,
    dimensions: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if dimensions is not None and array.ndim != dimensions:
        raise ScientificFigureError(
            f"{name} must have rank {dimensions}, got {array.ndim}"
        )
    if not np.isfinite(array).all():
        raise ScientificFigureError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class MeshData:
    """Physical mesh coordinates and exact element topology."""

    node_ids: tuple[int, ...]
    node_coordinates: np.ndarray
    element_ids: tuple[int, ...]
    element_connectivity: np.ndarray
    spatial_dimension: int
    mesh_id: str
    design_id: str
    units: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = finite_array(
            self.node_coordinates, name="node_coordinates", dimensions=2
        )
        connectivity = np.asarray(self.element_connectivity, dtype=int)
        if self.spatial_dimension != 2 or coordinates.shape != (
            len(self.node_ids),
            self.spatial_dimension,
        ):
            raise ScientificFigureError("MeshData coordinate shape is invalid")
        if connectivity.ndim != 2 or connectivity.shape[0] != len(
            self.element_ids
        ):
            raise ScientificFigureError("MeshData connectivity shape is invalid")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ScientificFigureError("MeshData node IDs are not unique")
        if not set(connectivity.ravel()).issubset(self.node_ids):
            raise ScientificFigureError("MeshData connectivity references unknown nodes")
        if self.units != "mm":
            raise ScientificFigureError("LIT Hand visualization mesh units must be mm")
        object.__setattr__(self, "node_coordinates", coordinates)
        object.__setattr__(self, "element_connectivity", connectivity)

    @property
    def coordinate_by_node_id(self) -> dict[int, np.ndarray]:
        return {
            node_id: self.node_coordinates[index]
            for index, node_id in enumerate(self.node_ids)
        }


@dataclass(frozen=True)
class DisplacementField:
    """Actual displacement vectors at explicitly identified sample points."""

    point_ids: tuple[str, ...]
    nodal_displacement: np.ndarray
    case_id: str
    step: int
    mesh_id: str
    design_id: str
    represented_configuration: str
    validity_mask: np.ndarray
    units: str
    location_kind: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = finite_array(
            self.nodal_displacement, name="nodal_displacement", dimensions=2
        )
        validity = np.asarray(self.validity_mask, dtype=bool)
        if values.shape != (len(self.point_ids), 2):
            raise ScientificFigureError("DisplacementField vector shape is invalid")
        if validity.shape != (len(self.point_ids),):
            raise ScientificFigureError("DisplacementField validity shape is invalid")
        if self.units != "mm":
            raise ScientificFigureError("displacement units must be mm")
        object.__setattr__(self, "nodal_displacement", values)
        object.__setattr__(self, "validity_mask", validity)


@dataclass(frozen=True)
class ObservationChain:
    """One independent, eta-ordered material sidewall chain."""

    side: str
    point_ids: tuple[str, ...]
    eta: np.ndarray
    undeformed_coordinates: np.ndarray
    outward_normals: np.ndarray
    mesh_id: str
    design_id: str
    units: str
    interpolation_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        eta = finite_array(self.eta, name="eta", dimensions=1)
        coordinates = finite_array(
            self.undeformed_coordinates,
            name="undeformed_coordinates",
            dimensions=2,
        )
        normals = finite_array(
            self.outward_normals, name="outward_normals", dimensions=2
        )
        count = len(self.point_ids)
        if self.side not in {"left", "right"}:
            raise ScientificFigureError("ObservationChain side must be left/right")
        if eta.shape != (count,) or coordinates.shape != (count, 2):
            raise ScientificFigureError("ObservationChain coordinate shape is invalid")
        if normals.shape != (count, 2):
            raise ScientificFigureError("ObservationChain normal shape is invalid")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-10):
            raise ScientificFigureError("ObservationChain normals must be unit vectors")
        if len(np.unique(eta)) != count or eta.min() < 0.0 or eta.max() > 1.0:
            raise ScientificFigureError("ObservationChain eta must be unique in [0,1]")
        if self.units != "mm":
            raise ScientificFigureError("ObservationChain units must be mm")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "undeformed_coordinates", coordinates)
        object.__setattr__(self, "outward_normals", normals)


@dataclass(frozen=True)
class ContactCase:
    """One converged or invalid state on a prescribed indentation path."""

    case_id: str
    design_id: str
    mesh_id: str
    step: int
    xi: float
    delta_mm: float
    reaction_force_n: float | None
    indentation_direction: tuple[float, float]
    contact_point_mm: tuple[float, float] | None
    convergence_state: str
    codtm_valid: bool
    descriptor_valid: bool
    source_artifact: str
    surface_x_mm: float | None = None

    def __post_init__(self) -> None:
        scalar_values = (self.xi, self.delta_mm, *self.indentation_direction)
        if not all(math.isfinite(value) for value in scalar_values):
            raise ScientificFigureError("ContactCase contains non-finite coordinates")
        if not 0.0 <= self.xi <= 1.0 or self.delta_mm < 0.0:
            raise ScientificFigureError("ContactCase xi/delta is outside its domain")
        if self.reaction_force_n is not None and not math.isfinite(
            self.reaction_force_n
        ):
            raise ScientificFigureError("ContactCase reaction is non-finite")
        direction_norm = math.hypot(*self.indentation_direction)
        if not math.isclose(direction_norm, 1.0, abs_tol=1.0e-10):
            raise ScientificFigureError("indentation direction must be unit length")
        if self.contact_point_mm is not None and not all(
            math.isfinite(value) for value in self.contact_point_mm
        ):
            raise ScientificFigureError("contact point is non-finite")
        if self.surface_x_mm is not None and not math.isfinite(self.surface_x_mm):
            raise ScientificFigureError("surface x coordinate is non-finite")


@dataclass(frozen=True)
class TransferSignature:
    """One side of a CODTM signature with explicit units and provenance."""

    design_id: str
    case_id: str
    mesh_id: str
    step: int
    side: str
    eta: np.ndarray
    u_normal: np.ndarray
    u_tangent: np.ndarray
    stored_secant_gain: np.ndarray | None
    stored_tangent_gain: np.ndarray | None
    delta_mm: float
    reaction_force_n: float | None
    normalization: str
    validity_mask: np.ndarray
    units: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        eta = finite_array(self.eta, name="signature eta", dimensions=1)
        normal = finite_array(
            self.u_normal, name="signature u_normal", dimensions=1
        )
        tangent = finite_array(
            self.u_tangent, name="signature u_tangent", dimensions=1
        )
        validity = np.asarray(self.validity_mask, dtype=bool)
        if eta.shape != normal.shape or normal.shape != tangent.shape:
            raise ScientificFigureError("TransferSignature arrays disagree")
        if validity.shape != eta.shape:
            raise ScientificFigureError("TransferSignature validity shape is invalid")
        if self.side not in {"left", "right"} or self.units != "mm":
            raise ScientificFigureError("TransferSignature side/units are invalid")
        for name in ("stored_secant_gain", "stored_tangent_gain"):
            optional = getattr(self, name)
            if optional is not None:
                optional_array = finite_array(
                    optional, name=name, dimensions=1
                )
                if optional_array.shape != eta.shape:
                    raise ScientificFigureError(f"{name} shape is invalid")
                object.__setattr__(self, name, optional_array)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "u_normal", normal)
        object.__setattr__(self, "u_tangent", tangent)
        object.__setattr__(self, "validity_mask", validity)


@dataclass
class VisualizationDataset:
    """Reusable semantic dataset consumed by figure builders."""

    meshes: dict[tuple[str, str], MeshData]
    observation_chains: dict[tuple[str, str, str], ObservationChain]
    contact_cases: tuple[ContactCase, ...]
    transfer_signatures: tuple[TransferSignature, ...]
    displacement_fields: tuple[DisplacementField, ...]
    source_artifacts: tuple[str, ...]
    source_checksums_sha256: dict[str, str]
    metadata: dict[str, Any]

    @property
    def design_ids(self) -> tuple[str, ...]:
        return tuple(sorted({case.design_id for case in self.contact_cases}))

    def mesh(self, design_id: str, mesh_id: str) -> MeshData:
        try:
            return self.meshes[(design_id, mesh_id)]
        except KeyError as exc:
            raise ScientificFigureError(
                f"mesh {design_id}/{mesh_id} is unavailable"
            ) from exc

    def chain(self, design_id: str, mesh_id: str, side: str) -> ObservationChain:
        try:
            return self.observation_chains[(design_id, mesh_id, side)]
        except KeyError as exc:
            raise ScientificFigureError(
                f"observation chain {design_id}/{mesh_id}/{side} is unavailable"
            ) from exc

    def case_states(
        self, design_id: str, mesh_id: str, xi: float
    ) -> tuple[ContactCase, ...]:
        states = [
            case
            for case in self.contact_cases
            if case.design_id == design_id
            and case.mesh_id == mesh_id
            and math.isclose(case.xi, xi, abs_tol=1.0e-12)
        ]
        return tuple(sorted(states, key=lambda case: (case.delta_mm, case.step)))

    def signature_states(
        self, design_id: str, mesh_id: str, xi: float, side: str
    ) -> tuple[TransferSignature, ...]:
        case_ids = {
            case.case_id for case in self.case_states(design_id, mesh_id, xi)
        }
        states = [
            signature
            for signature in self.transfer_signatures
            if signature.design_id == design_id
            and signature.mesh_id == mesh_id
            and signature.side == side
            and signature.case_id in case_ids
        ]
        return tuple(sorted(states, key=lambda state: (state.delta_mm, state.step)))

    def displacement_state(
        self, design_id: str, case_id: str, step: int
    ) -> DisplacementField:
        matches = [
            field
            for field in self.displacement_fields
            if field.design_id == design_id
            and field.case_id == case_id
            and field.step == step
        ]
        if len(matches) != 1:
            raise ScientificFigureError(
                f"expected one displacement state for {case_id}/step {step}"
            )
        return matches[0]




def merge_visualization_datasets(
    datasets: Sequence[VisualizationDataset],
) -> VisualizationDataset:
    """Combine independent designs while rejecting semantic identity collisions."""
    if not datasets:
        raise ScientificFigureError("at least one visualization dataset is required")
    meshes: dict[tuple[str, str], MeshData] = {}
    chains: dict[tuple[str, str, str], ObservationChain] = {}
    cases: list[ContactCase] = []
    signatures: list[TransferSignature] = []
    fields: list[DisplacementField] = []
    artifacts: list[str] = []
    checksums: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "framework_version": FRAMEWORK_VERSION,
        "adapters": [],
    }
    for dataset in datasets:
        for key, value in dataset.meshes.items():
            if key in meshes:
                raise ScientificFigureError(f"duplicate mesh identity {key}")
            meshes[key] = value
        for key, value in dataset.observation_chains.items():
            if key in chains:
                raise ScientificFigureError(f"duplicate chain identity {key}")
            chains[key] = value
        cases.extend(dataset.contact_cases)
        signatures.extend(dataset.transfer_signatures)
        fields.extend(dataset.displacement_fields)
        artifacts.extend(dataset.source_artifacts)
        for name, digest in dataset.source_checksums_sha256.items():
            qualified = f"{dataset.metadata.get('design_id', 'design')}:{name}"
            checksums[qualified] = digest
        metadata["adapters"].append(dataset.metadata)
    return VisualizationDataset(
        meshes=meshes,
        observation_chains=chains,
        contact_cases=tuple(cases),
        transfer_signatures=tuple(signatures),
        displacement_fields=tuple(fields),
        source_artifacts=tuple(artifacts),
        source_checksums_sha256=checksums,
        metadata=metadata,
    )
