from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from contact import CandidateContactError
from mesh.volume.mesh import VolumeMeshDependencyError
from lumo import MechanicsContract
from finger import (
    FingertipParameters,
    LED,
    OpticalParameters,
    ViscoelasticParameters,
)
from physics import CandidateMechanicsError
from optimization.objectives import TrajectoryObservation, compute_trajectory_objective
from optimization.design_space import ParameterSpec
from optimization.protocol import DEFAULT_TRAJECTORY_PROTOCOL, TrajectoryEvaluationProtocol
from optimization.evaluator import (
    Lumo3DTrajectoryEvaluator,
    _objective_failure,
    create_lumo3d_trajectory_study,
)
from ray_tracing.optical_mechanics import Transport3DResultError
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


def test_study_accepts_explicit_search_bounds(tmp_path) -> None:
    bounds = (
        ParameterSpec("flat_pad_height", 4.0, 6.0),
        ParameterSpec("semielliptical_pad_height", 8.0, 10.0),
        ParameterSpec("stem_width", 7.0, 8.0),
        ParameterSpec("stem_height", 5.0, 7.0),
        ParameterSpec("void_width", 0.5, 2.0),
        ParameterSpec("void_height", 0.0, 1.0),
    )

    study = create_lumo3d_trajectory_study(tmp_path, search_bounds=bounds)

    assert tuple(
        (variable.name.value, variable.lower, variable.upper)
        for variable in study.design_space.active_variables
    ) == tuple((bound.name.value, bound.lower, bound.upper) for bound in bounds)


def test_evaluation_contract_id_changes_with_fixed_scientific_inputs(tmp_path) -> None:
    base = create_lumo3d_trajectory_study(tmp_path / "base")
    changed_protocol = create_lumo3d_trajectory_study(
        tmp_path / "protocol",
        protocol=TrajectoryEvaluationProtocol(
            contact_locations_u=(0.2, 0.5, 0.8),
            indenter_radii_mm=(5.0,),
            checkpoint_depths_mm=(1.0,),
        ),
    )
    changed_led = create_lumo3d_trajectory_study(
        tmp_path / "led",
        led=LED(emission_half_angle_deg=60.0),
    )
    changed_led_size = create_lumo3d_trajectory_study(
        tmp_path / "led-size",
        led=LED(width_mm=5.0),
    )
    changed_material = create_lumo3d_trajectory_study(
        tmp_path / "material",
        nominal_parameters=FingertipParameters(
            optical=OpticalParameters(absorption_per_mm=0.03),
        ),
    )
    changed_fixed_geometry = create_lumo3d_trajectory_study(
        tmp_path / "fixed-geometry",
        nominal_parameters=FingertipParameters(
            link_thickness=4.0,
            void_height=0.25,
        ),
    )
    changed_mechanics = create_lumo3d_trajectory_study(
        tmp_path / "mechanics",
        mechanics_contract=MechanicsContract(vbd_iterations=11),
    )
    changed_first_contact = create_lumo3d_trajectory_study(
        tmp_path / "first-contact",
        mechanics_contract=replace(
            base.mechanics_contract,
            first_contact=replace(
                base.mechanics_contract.first_contact,
                coarse_step_mm=0.2,
            ),
        ),
    )
    changed_path_sampling = create_lumo3d_trajectory_study(
        tmp_path / "path-sampling",
        optical_settings=replace(
            base.optical_settings,
            internal_max_samples_per_segment=8,
        ),
    )
    changed_viscoelastic = create_lumo3d_trajectory_study(
        tmp_path / "viscoelastic",
        nominal_parameters=FingertipParameters(
            viscoelastic=ViscoelasticParameters(k_mu_pa=2.0e5),
        ),
    )

    assert len(base.evaluation_contract_id.split(":", 1)[1]) == 64
    assert base.evaluation_contract_id != changed_protocol.evaluation_contract_id
    assert base.evaluation_contract_id != changed_led.evaluation_contract_id
    assert base.evaluation_contract_id != changed_led_size.evaluation_contract_id
    assert base.evaluation_contract_id != changed_material.evaluation_contract_id
    assert base.evaluation_contract_id != changed_fixed_geometry.evaluation_contract_id
    assert base.evaluation_contract_id != changed_mechanics.evaluation_contract_id
    assert base.evaluation_contract_id != changed_first_contact.evaluation_contract_id
    assert base.evaluation_contract_id != changed_path_sampling.evaluation_contract_id
    assert base.evaluation_contract_id != changed_viscoelastic.evaluation_contract_id


def test_radius_six_is_rejected_as_domain_incompatible_before_mesh_or_newton(tmp_path) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (6.0,), (1.5,))
    result = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol).evaluate(FingertipParameters())
    assert result.status == "domain_incompatible"
    assert result.diagnostics["failure_stage"] == "domain_validation"
    assert result.checkpoint_diagnostics == ()


def test_candidate_fixed_inputs_must_match_the_evaluation_contract(tmp_path) -> None:
    evaluator = Lumo3DTrajectoryEvaluator(
        tmp_path,
        fixed_parameters=FingertipParameters(link_thickness=4.0),
    )

    with pytest.raises(ValueError, match="fixed fingertip inputs"):
        evaluator.evaluate(FingertipParameters())


def test_evaluator_requires_a_named_execution_device(tmp_path) -> None:
    with pytest.raises(ValueError, match="device"):
        Lumo3DTrajectoryEvaluator(tmp_path, device="")


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


def test_candidate_mechanics_error_is_translated_to_candidate_failure(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (5.0,), (1.5,))
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)

    class _Simulation:
        volume_mesh = SimpleNamespace(morphology_fingerprint="mesh")

        @staticmethod
        def run_sphere_contact(*_args, **_kwargs):
            raise CandidateMechanicsError("candidate state is inverted")

    class _SimulationFactory:
        @staticmethod
        def from_fingertip(*_args, **_kwargs):
            return _Simulation()

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _SimulationFactory)

    result = evaluator.evaluate(FingertipParameters())
    assert result.status == "mechanics_failure"
    assert result.diagnostics["failure_scenario"] == "candidate_mechanics_state"
    assert "candidate state is inverted" in (result.failure_message or "")


def test_optical_contract_error_is_not_reclassified_as_candidate_failure(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (5.0,), (1.5,))
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)

    class _Simulation:
        volume_mesh = SimpleNamespace(morphology_fingerprint="mesh")

        @staticmethod
        def run_sphere_contact(*_args, **_kwargs):
            raise Transport3DResultError("inconsistent optical result")

    class _SimulationFactory:
        @staticmethod
        def from_fingertip(*_args, **_kwargs):
            return _Simulation()

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _SimulationFactory)

    with pytest.raises(Transport3DResultError, match="inconsistent optical result"):
        evaluator.evaluate(FingertipParameters())


def test_unexpected_objective_error_propagates(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = TrajectoryEvaluationProtocol((0.25, 0.50), (5.0,), (1.5,))
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)

    monkeypatch.setattr(evaluator_module, "save_case_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluator_module,
        "write_mechanics_artifact",
        lambda *_args, **_kwargs: "artifact",
    )
    monkeypatch.setattr(
        evaluator_module,
        "build_contact_state_record",
        lambda **_kwargs: {
            "contact_state_fingerprint": "contact",
            "carrier_contact_active": False,
            "carrier_contact_occurred": False,
        },
    )
    monkeypatch.setattr(evaluator_module, "_final_pose_error_mm", lambda *_args: 0.0)

    class _Result:
        field = np.ones((2, 2), dtype=float)
        total_transport = 1.0
        launched_weight = 1.0
        escaped_weight = 1.0
        outgoing_surface_weight = 1.0
        absorbed_weight = 0.0
        terminated_weight = 0.0
        processed_segment_count = 1
        periodic_wrap_termination_count = 0
        periodic_wrap_termination_weight = 0.0
        no_event_termination_count = 0
        no_event_termination_weight = 0.0
        branch_cutoff_termination_count = 0
        branch_cutoff_termination_weight = 0.0
        max_interaction_termination_count = 0
        max_interaction_termination_weight = 0.0
        segment_budget_termination_count = 0
        segment_budget_termination_weight = 0.0
        rigid_surface_termination_count = 0
        rigid_surface_termination_weight = 0.0
        interface_normal_fallback_count = 0
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
            checkpoint=checkpoint,
            optics=_Result(),
        )
        return SimpleNamespace(
            normalized_location=location_u,
            indenter_radius_mm=5.0,
            unintended_boundary_clearance_mm=1.0,
            mechanics_seconds=0.0,
            optics_seconds=0.0,
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
        prepared = SimpleNamespace(source_node_ids=np.array([1], dtype=np.int64))

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
