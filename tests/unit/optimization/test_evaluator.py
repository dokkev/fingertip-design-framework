from __future__ import annotations

import numpy as np
import pytest

from mesh.volume3d import VolumeMeshDependencyError
from model import FingertipParameters
from optimization.objectives import TrajectoryObservation, compute_trajectory_objective
from optimization.protocol import DEFAULT_TRAJECTORY_PROTOCOL, TrajectoryEvaluationProtocol
from optimization.evaluator import (
    Lumo3DTrajectoryEvaluator,
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
