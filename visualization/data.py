"""Canonical semantic data and repository adapters for scientific figures."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fem.codtm_visualization import (
    CODTMVisualizationError,
    descriptor_verified_mask,
    input_checksums,
    load_codtm_dataset,
)
from fem.fingertip_mesher import generate_fingertip_mesh
from fem.mechanical_transfer_map import (
    TransferMapSettings,
    reference_outer_arc_chain,
    sample_observation_sidewalls,
)
from fem.mesh_types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


FRAMEWORK_VERSION = "1.0.0"


class ScientificFigureError(RuntimeError):
    """Raised when semantic figure data violates its declared contract."""


def _finite_array(
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
        coordinates = _finite_array(
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
        values = _finite_array(
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
        eta = _finite_array(self.eta, name="eta", dimensions=1)
        coordinates = _finite_array(
            self.undeformed_coordinates,
            name="undeformed_coordinates",
            dimensions=2,
        )
        normals = _finite_array(
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
        eta = _finite_array(self.eta, name="signature eta", dimensions=1)
        normal = _finite_array(
            self.u_normal, name="signature u_normal", dimensions=1
        )
        tangent = _finite_array(
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
                optional_array = _finite_array(
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


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ScientificFigureError(f"invalid JSON artifact {path}") from exc


def _case_result_path(input_dir: Path, case_name: str) -> Path:
    mapping = {
        "center_medium": "k2_center_baseline/medium/result.json",
        "center_fine": "k2_center_baseline/fine/result.json",
        "medium_xi_0p20": "k3_medium_location_sweep/xi_0p20/result.json",
        "medium_xi_0p35": "k3_medium_location_sweep/xi_0p35/result.json",
        "medium_xi_0p65": "k3_medium_location_sweep/xi_0p65/result.json",
        "medium_xi_0p80": "k3_medium_location_sweep/xi_0p80/result.json",
        "fine_xi_0p20": "k4_fine_spot_checks/xi_0p20/result.json",
        "fine_xi_0p80": "k4_fine_spot_checks/xi_0p80/result.json",
    }
    try:
        return input_dir / mapping[case_name]
    except KeyError as exc:
        raise ScientificFigureError(
            f"no Phase 4K result mapping for case {case_name}"
        ) from exc


def _build_mesh_and_chains(
    input_dir: Path,
    design_id: str,
    mesh_id: str,
    case_name: str,
    reference_by_side: Mapping[str, np.ndarray],
) -> tuple[MeshData, dict[str, ObservationChain]]:
    result = _strict_json(_case_result_path(input_dir, case_name))
    parameters = FingertipParameters(**result["configuration"]["fingertip_parameters"])
    model = FingertipModel(parameters)
    mesh = generate_fingertip_mesh(model, mesh_settings_for_level(mesh_id))
    expected = result["mesh"]
    if (
        len(mesh.pad_elements) != int(expected["pad_elements"])
        or len(mesh.carrier_elements) != int(expected["fixed_carrier_elements"])
    ):
        raise ScientificFigureError(
            "deterministically reconstructed mesh does not match Phase 4K counts"
        )
    pad_node_ids = tuple(
        sorted({node_id for element in mesh.pad_elements for node_id in element.node_ids})
    )
    pad_node_set = set(pad_node_ids)
    coordinates = np.asarray(
        [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm] for node_id in pad_node_ids]
    )
    pad_elements = tuple(sorted(mesh.pad_elements, key=lambda item: item.id))
    mesh_data = MeshData(
        node_ids=pad_node_ids,
        node_coordinates=coordinates,
        element_ids=tuple(element.id for element in pad_elements),
        element_connectivity=np.asarray(
            [element.node_ids for element in pad_elements], dtype=int
        ),
        spatial_dimension=2,
        mesh_id=mesh_id,
        design_id=design_id,
        units="mm",
        provenance={
            "kind": "deterministic_reference_mesh_reconstruction",
            "source_geometry": "FingertipModel",
            "source_settings": result["configuration"]["mesh_settings"],
            "gmsh_version": mesh.gmsh_version,
            "phase4k_counts_matched": True,
            "fem_solve_performed": False,
            "pad_node_count": len(pad_node_set),
        },
    )
    reference_chain = reference_outer_arc_chain(model, mesh)
    zero = {node_id: (0.0, 0.0) for node_id in mesh.nodes}
    sampled = sample_observation_sidewalls(
        model, reference_chain, zero, TransferMapSettings()
    )
    chains: dict[str, ObservationChain] = {}
    for side in ("left", "right"):
        rows = sampled[side]
        adapter_reference = np.asarray(
            [[row["reference_x_mm"], row["reference_y_mm"]] for row in rows]
        )
        if not np.allclose(
            adapter_reference, reference_by_side[side], rtol=0.0, atol=1.0e-10
        ):
            raise ScientificFigureError(
                f"reconstructed {side} observation coordinate differs from Phase 4K"
            )
        point_ids = tuple(f"{side}:{index}" for index in range(len(rows)))
        chains[side] = ObservationChain(
            side=side,
            point_ids=point_ids,
            eta=np.asarray([row["eta"] for row in rows]),
            undeformed_coordinates=adapter_reference,
            outward_normals=np.asarray(
                [
                    [
                        row["reference_outward_normal_x"],
                        row["reference_outward_normal_y"],
                    ]
                    for row in rows
                ]
            ),
            mesh_id=mesh_id,
            design_id=design_id,
            units="mm",
            interpolation_metadata={
                "method": "linear Line2 shape functions in reference arc length",
                "source": "undeformed PadOuterArc",
                "sample_count": len(rows),
            },
        )
    return mesh_data, chains


def load_phase4k_visualization_dataset(
    input_dir: str | Path,
    *,
    design_id: str = "baseline",
    mesh_ids: Sequence[str] = ("medium",),
) -> VisualizationDataset:
    """Adapt immutable Phase 4K artifacts into the canonical semantic model."""
    root = Path(input_dir).resolve()
    try:
        source, audit = load_codtm_dataset(root)
    except CODTMVisualizationError as exc:
        raise ScientificFigureError(str(exc)) from exc
    requested_meshes = tuple(dict.fromkeys(str(mesh) for mesh in mesh_ids))
    if not requested_meshes or set(requested_meshes) - {"medium", "fine"}:
        raise ScientificFigureError("Phase 4K adapter supports medium/fine meshes")
    descriptor_mask = descriptor_verified_mask(source)
    u_normal = source.canonical_field("u_normal")
    u_tangent = source.canonical_field("u_tangent")
    u_xy = source.canonical_field("u_xy")
    secant = source.canonical_field("G_secant")
    tangent = source.canonical_field("G_tangent")
    delta = source.canonical_field("delta_n")
    force = source.canonical_field("F_n")
    valid = np.asarray(source.arrays["valid_mask"], dtype=bool)
    meshes: dict[tuple[str, str], MeshData] = {}
    chains: dict[tuple[str, str, str], ObservationChain] = {}
    contact_cases: list[ContactCase] = []
    signatures: list[TransferSignature] = []
    fields: list[DisplacementField] = []

    for mesh_id in requested_meshes:
        mesh_cases = source.cases_for_mesh(mesh_id)
        representative = min(
            mesh_cases, key=lambda case: (abs(case.xi_cmd - 0.5), case.name)
        )
        reference = {
            side: source.reference_xy[(representative.name, side)]
            for side in ("left", "right")
        }
        mesh_data, mesh_chains = _build_mesh_and_chains(
            root, design_id, mesh_id, representative.name, reference
        )
        meshes[(design_id, mesh_id)] = mesh_data
        for side, chain in mesh_chains.items():
            chains[(design_id, mesh_id, side)] = chain

        for case_record in mesh_cases:
            case_index = source.case_index(case_record.name)
            case_result_path = _case_result_path(root, case_record.name)
            result = _strict_json(case_result_path)
            indenter = result["configuration"]["indenter"]
            direction = tuple(float(value) for value in indenter["loading_direction"])
            contact_point = tuple(float(value) for value in indenter["crown_point_mm"])
            point_ids = tuple(
                point_id
                for side in source.side_order
                for point_id in mesh_chains[side].point_ids
            )
            for step_index in range(delta.shape[1]):
                step = step_index + 1
                codtm_valid = bool(valid[case_index, step_index])
                descriptor_valid = bool(descriptor_mask[case_index, step_index])
                contact_cases.append(
                    ContactCase(
                        case_id=case_record.name,
                        design_id=design_id,
                        mesh_id=mesh_id,
                        step=step,
                        xi=float(case_record.xi_cmd),
                        delta_mm=float(delta[case_index, step_index]),
                        reaction_force_n=float(force[case_index, step_index])
                        if np.isfinite(force[case_index, step_index])
                        else None,
                        indentation_direction=direction,
                        contact_point_mm=contact_point,
                        convergence_state="converged" if codtm_valid else "invalid",
                        codtm_valid=codtm_valid,
                        descriptor_valid=descriptor_valid,
                        source_artifact=str(case_result_path),
                    )
                )
                displacement_parts = []
                validity_parts = []
                for side in source.side_order:
                    side_index = source.side_index(side)
                    chain = mesh_chains[side]
                    signatures.append(
                        TransferSignature(
                            design_id=design_id,
                            case_id=case_record.name,
                            mesh_id=mesh_id,
                            step=step,
                            side=side,
                            eta=chain.eta,
                            u_normal=u_normal[case_index, step_index, side_index],
                            u_tangent=u_tangent[case_index, step_index, side_index],
                            stored_secant_gain=secant[
                                case_index, step_index, side_index
                            ],
                            stored_tangent_gain=tangent[
                                case_index, step_index, side_index
                            ],
                            delta_mm=float(delta[case_index, step_index]),
                            reaction_force_n=float(force[case_index, step_index]),
                            normalization="raw displacement",
                            validity_mask=np.full(len(chain.eta), codtm_valid),
                            units="mm",
                            provenance={
                                "source": "codtm_arrays.npz",
                                "represented_variable": "u_normal",
                                "outward_positive": True,
                            },
                        )
                    )
                    displacement_parts.append(
                        u_xy[case_index, step_index, side_index]
                    )
                    validity_parts.append(np.full(len(chain.eta), codtm_valid))
                fields.append(
                    DisplacementField(
                        point_ids=point_ids,
                        nodal_displacement=np.concatenate(displacement_parts, axis=0),
                        case_id=case_record.name,
                        step=step,
                        mesh_id=mesh_id,
                        design_id=design_id,
                        represented_configuration="reference observation-chain samples",
                        validity_mask=np.concatenate(validity_parts),
                        units="mm",
                        location_kind="observation_chain_sample",
                        provenance={
                            "source": "codtm_arrays.npz:u_xy",
                            "full_volume_nodal_field_available": False,
                            "internal_displacement_inference": False,
                        },
                    )
                )
    checksums = input_checksums(root)
    return VisualizationDataset(
        meshes=meshes,
        observation_chains=chains,
        contact_cases=tuple(contact_cases),
        transfer_signatures=tuple(signatures),
        displacement_fields=tuple(fields),
        source_artifacts=tuple(str(root / name) for name in checksums),
        source_checksums_sha256=checksums,
        metadata={
            "framework_version": FRAMEWORK_VERSION,
            "adapter": "phase4k",
            "phase4k_audit": audit,
            "design_id": design_id,
            "mesh_ids": list(requested_meshes),
            "units": {"length": "mm", "force": "N"},
            "coordinate_convention": {
                "primary": "(side, eta)",
                "eta": "0 bonded; 1 crownward independently on each side",
                "u_normal_positive": "outward",
                "zeta": "visualization-only; central region unsampled",
            },
        },
    )


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
