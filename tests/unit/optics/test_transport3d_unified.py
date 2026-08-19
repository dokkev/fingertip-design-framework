from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

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
        escaped_weight=0.5,
        absorbed_weight=0.2,
        terminated_weight=0.1,
        outgoing_surface_weight=0.5,
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
        escape_weights=np.asarray([0.5]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([1]),
        energy_balance_error=0.0,
        energy_balance_tolerance=1.0e-6,
        object_absorbed_weight=0.1,
        object_transmitted_weight=0.1,
        object_interface_incident_weight=0.3,
        object_reflected_weight=0.1,
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
    assert loaded.object_absorbed_weight == result.object_absorbed_weight
    assert loaded.object_transmitted_weight == result.object_transmitted_weight
    assert loaded.object_interface_incident_weight == result.object_interface_incident_weight
    assert loaded.object_reflected_weight == result.object_reflected_weight

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


def test_full3d_artifact_records_and_validates_xyz_axis_order(tmp_path: Path) -> None:
    result = UnifiedTransportResult(
        morphology_id="synthetic-3d",
        morphology_fingerprint="morphology-3d",
        mechanics_source="explicit_contact_fea",
        mechanics_dimension="3D",
        contact_state={},
        optical_mode="FULL_3D",
        ray_count=3,
        transport_configuration_fingerprint="configuration-3d",
        field=np.ones((2, 3, 4), dtype=float),
        field_axes=(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        ),
        total_transport=1.0,
        launched_weight=1.0,
        escaped_weight=1.0,
        absorbed_weight=0.0,
        terminated_weight=0.0,
        valid_ray_count=3,
        terminated_ray_count=0,
        energy_balance_error=0.0,
        path_diagnostics={},
    )
    contract = {
        "morphology_id": "synthetic-3d",
        "morphology_parameters_fingerprint": "morphology-3d",
        "mechanics_source": "explicit_contact_fea",
        "mechanics_dimension": "3D",
        "optical_mode": "FULL_3D",
        "ray_count": 3,
        "transport_configuration_fingerprint": "configuration-3d",
    }
    path = tmp_path / "transport-3d.json"
    save_case_artifact(path, result, contract)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["field_axis_order"] == "x,y,z"
    loaded = load_case_artifact(path, expected_contract=contract)
    np.testing.assert_array_equal(loaded.field, result.field)
    np.testing.assert_array_equal(loaded.field_axes[2], result.field_axes[2])

    metadata["field_axis_order"] = "x,y"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="axis order"):
        load_case_artifact(path, expected_contract=contract)


def test_carrier_absorption_is_preserved_in_raw_and_unified_energy_channels(
    tmp_path: Path,
) -> None:
    raw = Transport3DResult(
        source_position_mm=(0.0, -6.0, 0.0),
        source_mode="full3d",
        extrusion_depth_mm=11.0,
        launched_ray_count=4,
        launched_weight=1.0,
        escaped_weight=0.4,
        absorbed_weight=0.05,
        terminated_weight=0.1,
        outgoing_surface_weight=0.4,
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
        escape_weights=np.asarray([0.4]),
        escape_primary_ray_indices=np.asarray([0]),
        escape_path_lengths_mm=np.asarray([1.0]),
        escape_interaction_counts=np.asarray([1]),
        energy_balance_error=0.0,
        energy_balance_tolerance=1.0e-6,
        object_absorbed_weight=0.05,
        object_interface_incident_weight=0.05,
        carrier_absorbed_weight=0.4,
        carrier_interface_incident_weight=0.4,
        internal_path_x_edges_mm=np.asarray([0.0, 1.0]),
        internal_path_y_edges_mm=np.asarray([0.0, 1.0]),
        internal_path_z_edges_mm=np.asarray([0.0, 1.0]),
        internal_weighted_path_density_3d=np.ones((1, 1, 1), dtype=float),
        internal_z_integrated_path_density=np.ones((1, 1), dtype=float),
    )
    result = UnifiedTransportResult.from_transport_result(
        raw,
        morphology_id="carrier-contact",
        morphology_fingerprint="morphology",
        mechanics_source="newton",
        mechanics_dimension="3D",
        contact_state={"carrier_contact": True},
        transport_configuration_fingerprint="carrier-absorber",
    )
    assert result.carrier_absorbed_weight == pytest.approx(0.4)
    assert result.carrier_interface_incident_weight == pytest.approx(0.4)
    assert result.object_absorbed_weight == pytest.approx(0.05)
    assert result.object_absorbed_weight != result.carrier_absorbed_weight

    contract = {
        "morphology_id": "carrier-contact",
        "morphology_parameters_fingerprint": "morphology",
        "mechanics_source": "newton",
        "mechanics_dimension": "3D",
        "optical_mode": "FULL_3D",
        "ray_count": 4,
        "transport_configuration_fingerprint": "carrier-absorber",
    }
    path = tmp_path / "carrier.json"
    save_case_artifact(path, result, contract)
    loaded = load_case_artifact(path, expected_contract=contract)
    assert loaded.carrier_absorbed_weight == pytest.approx(0.4)
    assert loaded.carrier_interface_incident_weight == pytest.approx(0.4)
    assert loaded.object_absorbed_weight == pytest.approx(0.05)
