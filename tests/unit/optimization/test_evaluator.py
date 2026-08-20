from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    def fail_volume_mesh(*_args, **_kwargs):
        raise VolumeMeshDependencyError("gmsh unavailable")

    class _SimulationFactory:
        from_fingertip = staticmethod(fail_volume_mesh)

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _SimulationFactory)
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path)

    with pytest.raises(VolumeMeshDependencyError, match="gmsh unavailable"):
        evaluator.evaluate(FingertipParameters())


def test_unexpected_evaluator_runtime_error_is_not_reclassified(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_volume_mesh(*_args, **_kwargs):
        raise RuntimeError("unexpected evaluator bug")

    class _SimulationFactory:
        from_fingertip = staticmethod(fail_volume_mesh)

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _SimulationFactory)
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

    class _Simulation:
        volume_mesh = SimpleNamespace(morphology_fingerprint="mesh")

        @staticmethod
        def run_sphere_contact(*_args, **_kwargs):
            raise CandidateContactError("candidate contact is impossible")

    class _SimulationFactory:
        @staticmethod
        def from_fingertip(*_args, **_kwargs):
            return _Simulation()

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _SimulationFactory)

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

    def contact_result(location_u: float) -> SimpleNamespace:
        checkpoint = SimpleNamespace(
            checkpoint_index=0,
            checkpoint_fraction=1.0,
            normalized_indentation_ratio=0.3,
            post_contact_travel_mm=1.5,
            cumulative_step_index=1,
            diagnostics={},
        )
        state = SimpleNamespace(
            normalized_location=location_u,
            indenter_radius_mm=5.0,
            trajectory_id=f"u_{location_u:.3f}",
            checkpoint=checkpoint,
            optics=_Result(),
            mechanics_artifact_path=Path(tmp_path / "state.npz"),
            mechanics_artifact_sha256="artifact",
            final_pose_error_mm=0.0,
            contact_state={
                "unintended_boundary_clearance_mm": 1.0,
                "contact_state_fingerprint": "contact",
                "carrier_contact_active": False,
                "carrier_contact_occurred": False,
            },
        )
        return SimpleNamespace(
            alignment=SimpleNamespace(
                target_point_mm=(0.0, 0.0, 0.0),
                outward_normal=(0.0, 1.0, 0.0),
                approach_direction=(0.0, -1.0, 0.0),
            ),
            first_contact=SimpleNamespace(
                travel_to_contact_mm=0.1,
                contact_pose=SimpleNamespace(translation_mm=(0.0, 0.0, 0.0)),
            ),
            checkpoints=(state,),
        )

    class _Simulation:
        volume_mesh = SimpleNamespace(morphology_fingerprint="mesh")

        @staticmethod
        def from_fingertip(*_args, **_kwargs):
            return _Simulation()

        @staticmethod
        def run_sphere_contact(*, location_u, **_kwargs):
            return contact_result(location_u)

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _Simulation)

    def fail_objective(*_args, **_kwargs):
        raise RuntimeError("unexpected objective implementation error")

    monkeypatch.setattr(evaluator_module, "compute_trajectory_objective", fail_objective)

    with pytest.raises(RuntimeError, match="unexpected objective implementation error"):
        evaluator.evaluate(FingertipParameters())
