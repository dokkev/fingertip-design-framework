from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from optics.transport3d import (
    LEGACY_UNIFIED_ARTIFACT_SCHEMA,
    Transport3DResult,
    UnifiedTransportResult,
    load_case_artifact,
    save_case_artifact,
)


def test_planar_summary_transposes_yx_accumulator_for_xy_public_axes() -> None:
    density_yx = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=float,
    )
    raw = Transport3DResult(
        source_position_mm=(0.0, -6.0, 0.0),
        source_mode="planar",
        extrusion_depth_mm=11.0,
        launched_ray_count=3,
        launched_weight=1.0,
        escaped_weight=1.0,
        absorbed_weight=0.0,
        terminated_weight=0.0,
        outgoing_surface_weight=1.0,
        surface_u_edges=np.asarray([0.0, 1.0]),
        surface_z_edges=np.asarray([-1.0, 1.0]),
        outgoing_surface_field=np.ones((1, 1), dtype=float),
        escape_positions_mm=np.asarray([[0.0, 0.0, 0.0]]),
        escape_directions=np.asarray([[0.0, 1.0, 0.0]]),
        escape_surface_normals=np.asarray([[0.0, -1.0, 0.0]]),
        escape_surface_u=np.asarray([0.5]),
        escape_surface_z=np.asarray([0.0]),
        escape_surface_tags=("pad_outer_arc",),
        escape_surface_primitive_indices=np.asarray([0]),
        escape_weights=np.asarray([1.0]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([1]),
        energy_balance_error=0.0,
        energy_balance_tolerance=1.0e-6,
        projected_x_edges_mm=np.asarray([0.0, 1.0, 2.0, 3.0]),
        projected_y_edges_mm=np.asarray([0.0, 1.0, 2.0]),
        projected_weighted_path_density=density_yx,
    )

    summary = UnifiedTransportResult.from_transport_result(
        raw,
        morphology_id="synthetic",
        morphology_fingerprint="morphology",
        mechanics_source="explicit_contact_fea",
        mechanics_dimension="2D",
        contact_state={},
        transport_configuration_fingerprint="configuration",
    )

    np.testing.assert_array_equal(summary.field, density_yx.T)
    np.testing.assert_array_equal(summary.field_axes[0], raw.projected_x_edges_mm)
    np.testing.assert_array_equal(summary.field_axes[1], raw.projected_y_edges_mm)


def test_unified_artifact_schema_marks_xy_and_reads_legacy_planar_orientation(
    tmp_path: Path,
) -> None:
    density_yx = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    raw = Transport3DResult(
        source_position_mm=(0.0, -6.0, 0.0),
        source_mode="planar",
        extrusion_depth_mm=11.0,
        launched_ray_count=3,
        launched_weight=1.0,
        escaped_weight=1.0,
        absorbed_weight=0.0,
        terminated_weight=0.0,
        outgoing_surface_weight=1.0,
        surface_u_edges=np.asarray([0.0, 1.0]),
        surface_z_edges=np.asarray([-1.0, 1.0]),
        outgoing_surface_field=np.ones((1, 1)),
        escape_positions_mm=np.asarray([[0.0, 0.0, 0.0]]),
        escape_directions=np.asarray([[0.0, 1.0, 0.0]]),
        escape_surface_normals=np.asarray([[0.0, -1.0, 0.0]]),
        escape_surface_u=np.asarray([0.5]),
        escape_surface_z=np.asarray([0.0]),
        escape_surface_tags=("pad_outer_arc",),
        escape_surface_primitive_indices=np.asarray([0]),
        escape_weights=np.asarray([1.0]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([1]),
        energy_balance_error=0.0,
        energy_balance_tolerance=1.0e-6,
        projected_x_edges_mm=np.asarray([0.0, 1.0, 2.0, 3.0]),
        projected_y_edges_mm=np.asarray([0.0, 1.0, 2.0]),
        projected_weighted_path_density=density_yx,
    )
    result = UnifiedTransportResult.from_transport_result(
        raw,
        morphology_id="synthetic",
        morphology_fingerprint="morphology",
        mechanics_source="explicit_contact_fea",
        mechanics_dimension="2D",
        contact_state={},
        transport_configuration_fingerprint="configuration",
    )
    contract = {
        "morphology_id": "synthetic",
        "morphology_parameters_fingerprint": "morphology",
        "mechanics_source": "explicit_contact_fea",
        "mechanics_dimension": "2D",
        "optical_mode": "PLANAR_2D",
        "ray_count": 3,
        "transport_configuration_fingerprint": "configuration",
    }
    path = tmp_path / "transport.json"
    save_case_artifact(path, result, contract)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["field_axis_order"] == "x,y"
    loaded = load_case_artifact(path, expected_contract=contract)
    np.testing.assert_array_equal(loaded.field, result.field)

    field_path = path.with_suffix(".npz")
    with field_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            field=result.field.T,
            axis_0=result.field_axes[0],
            axis_1=result.field_axes[1],
        )
    metadata["schema"] = LEGACY_UNIFIED_ARTIFACT_SCHEMA
    metadata.pop("field_axis_order")
    metadata["field_sha256"] = hashlib.sha256(field_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    legacy_loaded = load_case_artifact(path, expected_contract=contract)
    np.testing.assert_array_equal(legacy_loaded.field, result.field)
