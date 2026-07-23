"""Phase 4K CODTM artifacts to canonical visualization data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fem.observables import (
    TransferMapSettings,
    reference_outer_arc_chain,
    sample_observation_sidewalls,
)
from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from visualization.adapters.phase4k import (
    descriptor_verified_mask,
    input_checksums,
    load_codtm_dataset,
)
from visualization.data import (
    FRAMEWORK_VERSION,
    ContactCase,
    DisplacementField,
    MeshData,
    ObservationChain,
    ScientificFigureError,
    TransferSignature,
    VisualizationDataset,
)
from visualization.transforms import CODTMVisualizationError

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
