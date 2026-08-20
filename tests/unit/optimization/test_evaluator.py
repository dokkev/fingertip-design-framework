from __future__ import annotations

import json

import numpy as np
import pytest

from contact import CandidateContactError
from mesh.volume.mesh import VolumeMeshDependencyError
from model import FingertipParameters
from optimization.objectives import TrajectoryObservation, compute_trajectory_objective
from optimization.protocol import DEFAULT_TRAJECTORY_PROTOCOL, TrajectoryEvaluationProtocol
from optimization.evaluator import (
    Lumo3DTrajectoryEvaluator,
    _objective_failure,
    create_lumo3d_trajectory_study,
)
import optimization.evaluator as evaluator_module


def test_default_checkpoint_state_identities_are_unique() -> None:
    states = DEFAULT_TRAJECTORY_PROTOCOL.checkpoint_states()
    assert len(states) == 18
    assert len(set(states)) == 18


def test_custom_protocol_is_consumed_by_study_without_evaluator_changes(tmp_path) -> None:
    protocol = TrajectoryEvaluationProtocol(
        contact_locations_u=(0.2, 0.5, 0.8),
        indenter_radii_mm=(3.5, 4.5, 5.0),
        checkpoint_depths_mm=(0.5, 1.0, 1.5, 2.0),
    )
    study = create_lumo3d_trajectory_study(tmp_path, protocol=protocol)
    evaluator = study.create_evaluator()
    assert evaluator.protocol is protocol
    assert evaluator.protocol.optical_state_count == 36
    assert evaluator.mechanics_contract.max_load_increment_mm == 0.05


def test_radius_six_is_rejected_as_domain_incompatible_before_mesh_or_newton(tmp_path) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (6.0,), (1.5,))
    result = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol).evaluate(FingertipParameters())
    assert result.status == "domain_incompatible"
    assert result.diagnostics["failure_stage"] == "domain_validation"
    assert result.trajectory_diagnostics == ()


def test_shared_mesh_dependency_is_not_recorded_as_candidate_failure(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_volume_mesh(solid, settings):
        raise VolumeMeshDependencyError("gmsh unavailable")

    monkeypatch.setattr(evaluator_module, "generate_volume_mesh", fail_volume_mesh)
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path)

    with pytest.raises(VolumeMeshDependencyError, match="gmsh unavailable"):
        evaluator.evaluate(FingertipParameters())


def test_unexpected_evaluator_runtime_error_is_not_reclassified(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_volume_mesh(solid, settings):
        raise RuntimeError("unexpected evaluator bug")

    monkeypatch.setattr(evaluator_module, "generate_volume_mesh", fail_volume_mesh)
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path)

    with pytest.raises(RuntimeError, match="unexpected evaluator bug"):
        evaluator.evaluate(FingertipParameters())


def test_objective_pathology_flags_extinct_transport_without_changing_formula() -> None:
    observations = [
        TrajectoryObservation(0.25, 5.0, 1.0, np.array([1.0, 0.0]), {"total_transport": 0.0}),
        TrajectoryObservation(0.50, 5.0, 1.0, np.array([0.0, 1.0]), {"total_transport": 1.0}),
    ]
    result = compute_trajectory_objective(observations)
    assert result.objective_pathology is True
    assert result.objective_value == result.d_inter


def test_zero_signal_objective_is_explicit_candidate_failure_before_float_conversion() -> None:
    observations = [
        TrajectoryObservation(0.25, 5.0, 1.0, np.zeros(2)),
        TrajectoryObservation(0.50, 5.0, 1.0, np.zeros(2)),
    ]
    objective = compute_trajectory_objective(observations)
    failure = _objective_failure(objective)
    assert failure is not None
    assert failure.status == "optics_failure"
    assert failure.diagnostics["failure_scenario"] == "objective_pathology"
    assert failure.objective_value is None
    assert isinstance(failure.diagnostics["objective"], dict)
    encoded = json.dumps(objective.to_dict(), allow_nan=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    assert decoded["objective_value"] is None
    assert isinstance(decoded["observation_count"], int)


def test_candidate_contact_error_is_translated_to_candidate_failure(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (5.0,), (1.5,))
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)

    monkeypatch.setattr(
        evaluator_module,
        "generate_volume_mesh",
        lambda *_args: type(
            "VolumeMesh",
            (),
            {"solid": object(), "morphology_fingerprint": "mesh"},
        )(),
    )
    monkeypatch.setattr(evaluator_module, "prepare_fingertip_mesh", lambda _mesh: object())
    monkeypatch.setattr(evaluator_module, "make_outer_compliant_surface", lambda _solid: object())
    monkeypatch.setattr(evaluator_module, "make_distal_phalanx_mesh", lambda _solid: object())

    def fail_contact(*_args, **_kwargs):
        raise CandidateContactError("candidate contact is impossible")

    monkeypatch.setattr(evaluator, "_trajectory_mechanics", fail_contact)

    result = evaluator.evaluate(FingertipParameters())
    assert result.status == "mechanics_failure"
    assert result.diagnostics["failure_scenario"] == "candidate_contact"
    assert "candidate contact is impossible" in (result.failure_message or "")


def test_unexpected_objective_error_propagates(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = TrajectoryEvaluationProtocol((0.25, 0.50), (5.0,), (1.5,))
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)

    monkeypatch.setattr(
        evaluator_module,
        "generate_volume_mesh",
        lambda *_args: type(
            "VolumeMesh",
            (),
            {"solid": object(), "morphology_fingerprint": "mesh"},
        )(),
    )
    monkeypatch.setattr(
        evaluator_module,
        "prepare_fingertip_mesh",
        lambda _mesh: type("Prepared", (), {"source_node_ids": ()})(),
    )
    monkeypatch.setattr(evaluator_module, "make_outer_compliant_surface", lambda _solid: object())
    monkeypatch.setattr(evaluator_module, "make_distal_phalanx_mesh", lambda _solid: object())

    def mechanics(*_args, **kwargs):
        return ({
            "trajectory_id": f"u_{kwargs['location_u']:.3f}",
            "normalized_location": kwargs["location_u"],
            "radius_mm": kwargs["radius_mm"],
            "checkpoint_index": 0,
            "checkpoint_depth_mm": 1.5,
            "checkpoint_fraction": 1.0,
            "normalized_indentation_ratio": 0.3,
            "post_contact_travel_mm": 1.5,
            "unintended_boundary_clearance_mm": 1.0,
            "cumulative_step_index": 1,
            "first_contact_travel_mm": 0.1,
            "first_contact_fingerprint": "contact",
            "mechanics_artifact_path": str(tmp_path / "state.npz"),
            "mechanics_artifact_sha256": "artifact",
            "final_pose_error_mm": 0.0,
            "mechanics_diagnostics": {},
        },)

    monkeypatch.setattr(evaluator, "_trajectory_mechanics", mechanics)
    monkeypatch.setattr(
        evaluator_module,
        "restore_deformed_optical_state",
        lambda *_args, **_kwargs: type(
            "Restored", (), {"geometry": object(), "artifact_path": tmp_path / "state.npz"}
        )(),
    )
    monkeypatch.setattr(evaluator_module, "create_runtime", lambda: object())
    monkeypatch.setattr(evaluator_module, "save_case_artifact", lambda *_args, **_kwargs: None)

    class _Result:
        field = np.ones((2, 2), dtype=float)
        total_transport = 1.0
        launched_weight = 1.0
        escaped_weight = 1.0
        absorbed_weight = 0.0
        terminated_weight = 0.0
        object_interface_incident_weight = 0.0
        object_absorbed_weight = 0.0
        object_transmitted_weight = 0.0
        object_reflected_weight = 0.0
        carrier_absorbed_weight = 0.0
        carrier_transmitted_weight = 0.0
        carrier_interface_incident_weight = 0.0
        carrier_reflected_weight = 0.0
        carrier_contact_triangle_count = 0
        energy_balance_error = 0.0

    monkeypatch.setattr(
        evaluator_module,
        "trace_geometry",
        lambda *_args, **_kwargs: _Result(),
    )

    def fail_objective(*_args, **_kwargs):
        raise RuntimeError("unexpected objective implementation error")

    monkeypatch.setattr(evaluator_module, "compute_trajectory_objective", fail_objective)

    with pytest.raises(RuntimeError, match="unexpected objective implementation error"):
        evaluator.evaluate(FingertipParameters())
