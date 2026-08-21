from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from lumo.finger import Fingertip, FingertipParameters, LED, OpticalParameters
from lumo.ray_tracing.optical_mechanics import Transport3DResult
from lumo.optimization.optical_artifact import (
    energy_record,
    load_case_artifact,
    save_case_artifact,
)
from lumo.optimization.optical_contract import optical_physics_parameters
from lumo.optimization.optical_contract import DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE


def _result() -> Transport3DResult:
    return Transport3DResult(
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
        outgoing_surface_field=np.full((1, 1), 0.4, dtype=float),
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
        processed_sample_count=12,
        clipped_sample_count=2,
        represented_weighted_path_length_mm=3.5,
        clipped_weighted_path_length_mm=0.25,
        branch_cutoff_termination_count=1,
        branch_cutoff_termination_weight=0.1,
        object_absorbed_weight=0.05,
        object_interface_incident_weight=0.05,
        carrier_absorbed_weight=0.4,
        carrier_interface_incident_weight=0.4,
        field_x_edges_mm=np.asarray([0.0, 1.0, 2.0]),
        field_y_edges_mm=np.asarray([0.0, 1.0, 2.0, 3.0]),
        field_z_edges_mm=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        field_density_3d=np.ones((2, 3, 4), dtype=float),
    )


def test_full3d_artifact_records_and_validates_xyz_axis_order(tmp_path: Path) -> None:
    result = _result()
    contract = {
        "morphology_id": "synthetic-3d",
        "morphology_parameters_fingerprint": "morphology-3d",
        "mechanics_source": "explicit_contact_fea",
        "mechanics_dimension": "3D",
        "optical_mode": "FULL_3D",
        "ray_count": 4,
        "transport_configuration_fingerprint": "configuration-3d",
    }
    path = tmp_path / "transport-3d.json"
    save_case_artifact(path, result, contract)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["field_axis_order"] == "x,y,z"
    loaded = load_case_artifact(path, expected_contract=contract)
    np.testing.assert_array_equal(loaded.field, result.field)
    np.testing.assert_array_equal(loaded.field_axes[2], result.field_axes[2])
    assert loaded.outgoing_surface_weight == pytest.approx(0.4)
    assert loaded.processed_sample_count == 12
    assert loaded.clipped_sample_count == 2
    assert loaded.represented_weighted_path_length_mm == pytest.approx(3.5)
    assert loaded.clipped_weighted_path_length_mm == pytest.approx(0.25)

    metadata["field_axis_order"] = "x,y"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="axes"):
        load_case_artifact(path, expected_contract=contract)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result.pop("clipped_sample_count"),
            "missing required path-field diagnostics",
        ),
        (
            lambda result: result["path_diagnostics"]["path_field"].__setitem__(
                "clipped_sample_count", -1
            ),
            "non-negative integer",
        ),
        (
            lambda result: result["path_diagnostics"]["path_field"].__setitem__(
                "represented_weighted_path_length_mm", 9.0
            ),
            "disagrees",
        ),
        (
            lambda result: result["path_diagnostics"].__setitem__(
                "processed_weighted_path_length_mm", 99.0
            ),
            "represented plus clipped",
        ),
    ),
)
def test_v7_artifact_rejects_missing_negative_or_inconsistent_path_diagnostics(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    contract = {"morphology_id": "strict-v7", "mechanics_dimension": "3D"}
    path = tmp_path / "strict-v7.json"
    save_case_artifact(path, _result(), contract)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mutation(metadata["result"])
    path.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_case_artifact(path, expected_contract=contract)


def test_direct_result_keeps_carrier_energy_channels_distinct() -> None:
    result = _result()
    assert result.carrier_absorbed_weight == pytest.approx(0.4)
    assert result.object_absorbed_weight == pytest.approx(0.05)
    assert result.object_absorbed_weight != result.carrier_absorbed_weight


def test_energy_record_exposes_count_weight_and_fraction_for_each_termination_reason() -> None:
    record = energy_record(_result())

    for reason in (
        "periodic_wrap",
        "no_event",
        "branch_cutoff",
        "max_interaction",
        "segment_budget",
        "rigid_surface",
    ):
        assert f"{reason}_termination_count" in record
        weight = record[f"{reason}_termination_weight"]
        assert record[f"{reason}_termination_fraction"] == pytest.approx(
            weight / record["launched_weight"]
        )


def test_v7_round_trip_preserves_segment_budget_rejection_for_reassessment(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        branch_cutoff_termination_count=0,
        branch_cutoff_termination_weight=0.0,
        segment_budget_termination_count=1,
        segment_budget_termination_weight=0.1,
    )
    contract = {"morphology_id": "segment-round-trip", "mechanics_dimension": "3D"}
    path = tmp_path / "segment-round-trip.json"
    save_case_artifact(path, result, contract)

    loaded = load_case_artifact(path, expected_contract=contract)
    assessment = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.assess(loaded)

    assert loaded.segment_budget_termination_count == 1
    assert loaded.segment_budget_termination_weight == pytest.approx(0.1)
    assert assessment.accepted is False
    assert "segment_budget_termination" in assessment.failure_reasons


def test_optical_fingerprint_inputs_match_full3d_transport_inputs() -> None:
    tip = Fingertip(
        parameters=FingertipParameters(optical=OpticalParameters()),
        led=LED(relative_radiant_power=2.0, emission_half_angle_deg=60.0),
    )

    parameters = optical_physics_parameters(tip)

    assert parameters == {
        "refractive_index_air": 1.0,
        "refractive_index_silicone": 1.41,
        "absorption_per_mm": 0.02,
        "relative_radiant_power": 2.0,
        "emission_half_angle_deg": 60.0,
    }
