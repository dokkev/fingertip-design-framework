from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optics.transport3d import (
    Transport3DResult,
    UnifiedTransportResult,
    load_case_artifact,
    save_case_artifact,
)


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
