"""Local-normal full-pad FEM artifacts to canonical visualization data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from visualization.data import (
    FRAMEWORK_VERSION,
    ContactCase,
    DisplacementField,
    MeshData,
    ScientificFigureError,
    VisualizationDataset,
    finite_array,
)


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ScientificFigureError(f"JSON root must be an object: {path}")
    return value

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_normal_indentation_visualization_dataset(
    input_dir: str | Path,
    *,
    design_id: str = "baseline",
    mesh_id: str = "medium",
) -> VisualizationDataset:
    """Load persisted full-pad fields from the local-normal three-case run."""
    root = Path(input_dir).resolve()
    manifest_path = root / "dataset_manifest.json"
    manifest = _strict_json(manifest_path)
    if (
        manifest.get("phase") != "normal_indentation_full_field"
        or manifest.get("status") != "PASS"
        or manifest.get("mesh_level") != mesh_id
    ):
        raise ScientificFigureError(
            "local-normal full-field manifest is missing, failed, or uses "
            "a different mesh"
        )
    case_records = manifest.get("cases")
    if not isinstance(case_records, list) or len(case_records) != 3:
        raise ScientificFigureError(
            "local-normal full-field manifest must contain exactly three cases"
        )

    mesh_data: MeshData | None = None
    contact_cases: list[ContactCase] = []
    fields: list[DisplacementField] = []
    artifacts = [str(manifest_path)]
    checksums = {"dataset_manifest.json": _sha256(manifest_path)}
    reference_node_ids: np.ndarray | None = None
    reference_coordinates: np.ndarray | None = None
    reference_element_ids: np.ndarray | None = None
    reference_connectivity: np.ndarray | None = None

    for record in case_records:
        if not isinstance(record, Mapping) or record.get("status") != "PASS":
            raise ScientificFigureError("local-normal full-field case is not PASS")
        result_path = root / str(record["result"])
        field_path = root / str(record["field"])
        if _sha256(result_path) != record.get("result_sha256"):
            raise ScientificFigureError(f"result checksum mismatch: {result_path}")
        if _sha256(field_path) != record.get("field_sha256"):
            raise ScientificFigureError(f"field checksum mismatch: {field_path}")
        result = _strict_json(result_path)
        if (
            result.get("phase") != "normal_indentation_full_field"
            or result.get("status") != "PASS"
            or result.get("solve_status") != "PASS"
        ):
            raise ScientificFigureError(f"invalid full-field result: {result_path}")
        with np.load(field_path, allow_pickle=False) as source:
            node_ids = np.asarray(source["node_ids"], dtype=np.int64)
            coordinates = finite_array(
                source["reference_coordinates_mm"],
                name="reference_coordinates_mm",
                dimensions=2,
            )
            element_ids = np.asarray(source["element_ids"], dtype=np.int64)
            connectivity = np.asarray(
                source["element_connectivity_node_ids"], dtype=np.int64
            )
            displacement = finite_array(
                source["displacement_mm"],
                name="displacement_mm",
                dimensions=2,
            )
            stored_magnitude = finite_array(
                source["displacement_magnitude_mm"],
                name="displacement_magnitude_mm",
                dimensions=1,
            )
        if (
            coordinates.shape != (len(node_ids), 2)
            or displacement.shape != coordinates.shape
            or stored_magnitude.shape != (len(node_ids),)
            or connectivity.shape != (len(element_ids), 3)
        ):
            raise ScientificFigureError(
                f"full-field array shape is invalid: {field_path}"
            )
        if not np.allclose(
            stored_magnitude,
            np.linalg.norm(displacement, axis=1),
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ScientificFigureError(
                f"stored displacement magnitude is inconsistent: {field_path}"
            )
        if reference_node_ids is None:
            reference_node_ids = node_ids
            reference_coordinates = coordinates
            reference_element_ids = element_ids
            reference_connectivity = connectivity
            mesh_data = MeshData(
                node_ids=tuple(int(value) for value in node_ids),
                node_coordinates=coordinates,
                element_ids=tuple(int(value) for value in element_ids),
                element_connectivity=connectivity,
                spatial_dimension=2,
                mesh_id=mesh_id,
                design_id=design_id,
                units="mm",
                provenance={
                    "kind": "persisted_full_pad_fem_mesh",
                    "source": str(field_path),
                    "fem_solve_performed": True,
                    "carrier_excluded": True,
                    "indenter_excluded": True,
                },
            )
        elif not (
            np.array_equal(node_ids, reference_node_ids)
            and np.array_equal(element_ids, reference_element_ids)
            and np.array_equal(connectivity, reference_connectivity)
            and np.array_equal(coordinates, reference_coordinates)
        ):
            raise ScientificFigureError(
                "the three local-normal cases do not share identical pad topology"
            )

        final = result["final"]
        indenter = result["configuration"]["indenter"]
        surface_x_mm = float(result["surface_x_command_mm"])
        actual_xi = float(result["actual_reference_xi"])
        step = int(final["step"])
        case_id = str(result["case_name"])
        contact_cases.append(
            ContactCase(
                case_id=case_id,
                design_id=design_id,
                mesh_id=mesh_id,
                step=step,
                xi=actual_xi,
                delta_mm=float(final["achieved_indentation_mm"]),
                reaction_force_n=float(final["indenter_normal_reaction_n"]),
                indentation_direction=tuple(
                    float(value) for value in indenter["loading_direction"]
                ),
                contact_point_mm=tuple(
                    float(value) for value in result["actual_surface_point_mm"]
                ),
                convergence_state="converged",
                codtm_valid=True,
                descriptor_valid=False,
                source_artifact=str(result_path),
                surface_x_mm=surface_x_mm,
            )
        )
        fields.append(
            DisplacementField(
                point_ids=tuple(str(int(value)) for value in node_ids),
                nodal_displacement=displacement,
                case_id=case_id,
                step=step,
                mesh_id=mesh_id,
                design_id=design_id,
                represented_configuration="full pad nodal FEM field",
                validity_mask=np.ones(len(node_ids), dtype=bool),
                units="mm",
                location_kind="mesh_node",
                provenance={
                    "source": str(field_path),
                    "full_volume_nodal_field_available": True,
                    "heatmap_quantity": "displacement magnitude |u|",
                    "carrier_excluded": True,
                    "indenter_excluded": True,
                },
            )
        )
        for path in (result_path, field_path):
            relative = str(path.relative_to(root))
            artifacts.append(str(path))
            checksums[relative] = _sha256(path)

    assert mesh_data is not None
    return VisualizationDataset(
        meshes={(design_id, mesh_id): mesh_data},
        observation_chains={},
        contact_cases=tuple(contact_cases),
        transfer_signatures=(),
        displacement_fields=tuple(fields),
        source_artifacts=tuple(artifacts),
        source_checksums_sha256=checksums,
        metadata={
            "framework_version": FRAMEWORK_VERSION,
            "adapter": "normal_indentation_full_field",
            "design_id": design_id,
            "mesh_ids": [mesh_id],
            "units": {"length": "mm", "force": "N"},
            "coordinate_convention": {
                "frame": "2D FingertipModel x-y frame",
                "contact_coordinate": "global surface x [mm]",
                "positive_travel": (
                    "local inward pad normal, equal to -pad_outward_normal"
                ),
                "displacement": "u=[u_x,u_y] in the model x-y frame [mm]",
                "heatmap": "nodal |u| [mm] on 1x deformed T3 geometry",
            },
            "source_manifest": manifest,
        },
    )
