from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lumo.contact import CandidateContactError
from lumo.mesh.volume.mesh import VolumeMeshDependencyError
from lumo import MechanicsContract
from lumo.finger import (
    FingertipParameters,
    LED,
    OpticalParameters,
    ViscoelasticParameters,
)
from lumo.physics import CandidateMechanicsError
from lumo.mesh import volume_mesh_settings_for_tier
from lumo.simulation import CandidateOpticsError
from lumo.optimization.objectives import TrajectoryObservation, compute_trajectory_objective
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    ParameterSpec,
    PRODUCTION_SEARCH_BOUNDS,
)
from lumo.optimization.protocol import DEFAULT_TRAJECTORY_PROTOCOL, TrajectoryEvaluationProtocol
from lumo.optimization.deformed_state_artifact import ContactState, ContactStateIdentity
from lumo.optimization.evaluator import (
    Lumo3DTrajectoryEvaluator,
    _objective_failure,
)
from lumo.ray_tracing.optical_mechanics import (
    Transport3DCandidateGeometryError,
    Transport3DGeometryError,
    Transport3DResultError,
    Transport3DSettings,
)
import lumo.optimization.evaluator as evaluator_module


def _design_space(
    nominal_parameters: FingertipParameters | None = None,
    bounds: tuple[ParameterSpec, ...] = PRODUCTION_SEARCH_BOUNDS,
) -> DesignSpace:
    return DesignSpace(
        nominal_parameters or FingertipParameters(void_height=0.25),
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in bounds
        ),
    )


def test_default_checkpoint_state_identities_are_unique() -> None:
    states = DEFAULT_TRAJECTORY_PROTOCOL.checkpoint_states()
    assert len(states) == 18
    assert len(set(states)) == 18


def test_contact_state_reads_checkpoint_identity_from_one_source_of_truth() -> None:
    identity = ContactStateIdentity(
        morphology_fingerprint="morphology",
        protocol_fingerprint="protocol",
        contact_location_u=0.5,
        indenter_radius_mm=5.0,
        checkpoint_depth_mm=1.0,
        checkpoint_fraction=0.5,
        normalized_indentation_ratio=0.2,
        post_contact_travel_mm=1.0,
        unintended_boundary_clearance_mm=2.0,
        mechanics_artifact_sha256="artifact",
    )
    state = ContactState(
        identity=identity,
        contact_state_fingerprint="contact-state",
        initial_gap_mm=0.25,
        first_contact_travel_mm=0.1,
        spawn_clearance_mm=0.01,
        carrier_contact_active=False,
        carrier_contact_occurred=False,
        carrier_mechanical_contact_count=0,
        carrier_mechanical_contact_vertex_count=0,
        first_carrier_contact_step=None,
        carrier_contact_source_node_ids=(),
        carrier_mapping_tolerance_mm=0.0625,
    )

    assert state.normalized_location == identity.contact_location_u
    assert state.indenter_radius_mm == identity.indenter_radius_mm
    assert state.checkpoint_depth_mm == identity.checkpoint_depth_mm
    assert state.post_contact_travel_mm == identity.post_contact_travel_mm
    assert state.mechanics_artifact_sha256 == identity.mechanics_artifact_sha256
    assert "normalized_location" not in state.__dict__


def test_custom_protocol_is_consumed_by_evaluator_without_wrapper_study(tmp_path) -> None:
    protocol = TrajectoryEvaluationProtocol(
        contact_locations_u=(0.2, 0.5, 0.8),
        indenter_radii_mm=(3.5, 4.5, 5.0),
        checkpoint_depths_mm=(0.5, 1.0, 1.5, 2.0),
    )
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)
    assert evaluator.protocol is protocol
    assert evaluator.protocol.optical_state_count == 36
    assert evaluator.mechanics_contract.max_load_increment_mm == 0.05


def test_evaluator_requires_the_native_field_for_its_objective(tmp_path) -> None:
    with pytest.raises(ValueError, match="retain_internal_path_field=True"):
        Lumo3DTrajectoryEvaluator(
            tmp_path,
            optical_settings=Transport3DSettings(retain_internal_path_field=False),
        )


def test_design_space_accepts_explicit_search_bounds() -> None:
    bounds = (
        ParameterSpec("flat_pad_height", 4.0, 6.0),
        ParameterSpec("semielliptical_pad_height", 8.0, 10.0),
        ParameterSpec("stem_width", 7.0, 8.0),
        ParameterSpec("stem_height", 5.0, 7.0),
        ParameterSpec("void_width", 0.5, 2.0),
        ParameterSpec("void_height", 0.0, 1.0),
    )

    design_space = _design_space(bounds=bounds)

    assert tuple(
        (variable.name.value, variable.lower, variable.upper)
        for variable in design_space.active_variables
    ) == tuple((bound.name.value, bound.lower, bound.upper) for bound in bounds)


def test_evaluation_contract_id_changes_with_fixed_scientific_inputs(tmp_path) -> None:
    base = Lumo3DTrajectoryEvaluator(tmp_path / "base")
    changed_protocol = Lumo3DTrajectoryEvaluator(
        tmp_path / "protocol",
        protocol=TrajectoryEvaluationProtocol(
            contact_locations_u=(0.2, 0.5, 0.8),
            indenter_radii_mm=(5.0,),
            checkpoint_depths_mm=(1.0,),
        ),
    )
    changed_led = Lumo3DTrajectoryEvaluator(
        tmp_path / "led",
        led=LED(emission_half_angle_deg=60.0),
    )
    changed_led_size = Lumo3DTrajectoryEvaluator(
        tmp_path / "led-size",
        led=LED(width_mm=5.0),
    )
    changed_material = Lumo3DTrajectoryEvaluator(
        tmp_path / "material",
        fixed_parameters=FingertipParameters(
            optical=OpticalParameters(absorption_per_mm=0.03),
        ),
    )
    changed_fixed_geometry = Lumo3DTrajectoryEvaluator(
        tmp_path / "fixed-geometry",
        fixed_parameters=FingertipParameters(
            link_thickness=4.0,
            void_height=0.25,
        ),
    )
    changed_mechanics = Lumo3DTrajectoryEvaluator(
        tmp_path / "mechanics",
        mechanics_contract=MechanicsContract(vbd_iterations=11),
    )
    changed_first_contact = Lumo3DTrajectoryEvaluator(
        tmp_path / "first-contact",
        mechanics_contract=replace(
            base.mechanics_contract,
            first_contact=replace(
                base.mechanics_contract.first_contact,
                coarse_step_mm=0.2,
            ),
        ),
    )
    changed_path_sampling = Lumo3DTrajectoryEvaluator(
        tmp_path / "path-sampling",
        optical_settings=replace(
            base.settings,
            internal_max_samples_per_segment=8,
        ),
    )
    changed_viscoelastic = Lumo3DTrajectoryEvaluator(
        tmp_path / "viscoelastic",
        fixed_parameters=FingertipParameters(
            viscoelastic=ViscoelasticParameters(k_mu_pa=2.0e5),
        ),
    )
    changed_mesh = Lumo3DTrajectoryEvaluator(
        tmp_path / "mesh",
        volume_mesh_settings=volume_mesh_settings_for_tier("reference"),
    )
    changed_evidence_collection = Lumo3DTrajectoryEvaluator(
        tmp_path / "complete-evidence",
        complete_trajectory_after_optical_failure=True,
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
    assert base.evaluation_contract_id != changed_mesh.evaluation_contract_id
    assert (
        base.evaluation_contract_id
        != changed_evidence_collection.evaluation_contract_id
    )


def test_validation_evidence_mode_completes_18_states_and_preserves_raw_failure_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class _Quality:
        minimum_scaled_jacobian: float = 0.25

    volume_mesh = SimpleNamespace(
        morphology_fingerprint="mesh",
        settings=volume_mesh_settings_for_tier("search"),
        gmsh_version="fake-gmsh",
        quality=_Quality(),
        validation=SimpleNamespace(passed=True, checks={"quality": True}, errors=()),
    )

    class _ContactState:
        def __init__(self, *, location_u, radius_mm, checkpoint, **_kwargs):
            self.normalized_location = float(location_u)
            self.indenter_radius_mm = float(radius_mm)
            self.checkpoint_depth_mm = float(checkpoint.post_contact_travel_mm)
            self.checkpoint_fraction = float(checkpoint.checkpoint_fraction)
            self.normalized_indentation_ratio = float(
                checkpoint.normalized_indentation_ratio
            )
            self.post_contact_travel_mm = float(checkpoint.post_contact_travel_mm)
            self.unintended_boundary_clearance_mm = 1.0
            self.mechanics_artifact_sha256 = "mechanics-sha"

        def to_dict(self):
            return {
                "normalized_location": self.normalized_location,
                "indenter_radius_mm": self.indenter_radius_mm,
                "checkpoint_depth_mm": self.checkpoint_depth_mm,
                "mechanics_artifact_sha256": self.mechanics_artifact_sha256,
            }

    monkeypatch.setattr(
        evaluator_module,
        "build_contact_state_record",
        lambda **kwargs: _ContactState(**kwargs),
    )
    monkeypatch.setattr(
        evaluator_module,
        "write_mechanics_artifact",
        lambda *_args, **_kwargs: "mechanics-sha",
    )
    monkeypatch.setattr(
        evaluator_module,
        "save_case_artifact",
        lambda *_args, **_kwargs: None,
    )

    def optical_result(location_u, radius_mm, depth_mm, *, hard_failure, pathology):
        return SimpleNamespace(
            field=np.asarray(
                [1.0 + float(location_u), 1.0 + 0.01 * radius_mm + depth_mm]
            ),
            total_transport=0.0 if pathology else 0.9,
            launched_weight=1.0,
            escaped_weight=0.0 if pathology else 0.9,
            outgoing_surface_weight=0.9,
            absorbed_weight=0.0,
            terminated_weight=0.1 if hard_failure else 0.0,
            processed_segment_count=1,
            processed_sample_count=1,
            clipped_sample_count=0,
            represented_weighted_path_length_mm=1.0,
            clipped_weighted_path_length_mm=0.0,
            processed_weighted_path_length_mm=1.0,
            periodic_wrap_termination_count=0,
            periodic_wrap_termination_weight=0.0,
            no_event_termination_count=0,
            no_event_termination_weight=0.0,
            branch_cutoff_termination_count=0,
            branch_cutoff_termination_weight=0.0,
            max_interaction_termination_count=0,
            max_interaction_termination_weight=0.0,
            segment_budget_termination_count=1 if hard_failure else 0,
            segment_budget_termination_weight=0.1 if hard_failure else 0.0,
            rigid_surface_termination_count=0,
            rigid_surface_termination_weight=0.0,
            interface_normal_fallback_count=0,
            object_interface_incident_weight=0.0,
            object_absorbed_weight=0.0,
            object_transmitted_weight=0.0,
            object_reflected_weight=0.0,
            carrier_absorbed_weight=0.0,
            carrier_transmitted_weight=0.0,
            carrier_interface_incident_weight=0.0,
            carrier_reflected_weight=0.0,
            carrier_contact_triangle_count=0,
            energy_balance_error=0.0,
        )

    class _Simulation:
        failure_mode = "numerical"
        prepared = SimpleNamespace(source_node_ids=np.asarray([1], dtype=np.int64))

        @classmethod
        def from_fingertip(cls, *_args, **_kwargs):
            return cls()

        @classmethod
        def run_sphere_contact(cls, *, location_u, radius_mm, checkpoint_depths_mm):
            states = []
            for checkpoint_index, depth_mm in enumerate(checkpoint_depths_mm):
                first_state = (
                    location_u == DEFAULT_TRAJECTORY_PROTOCOL.contact_locations_u[0]
                    and radius_mm == DEFAULT_TRAJECTORY_PROTOCOL.indenter_radii_mm[0]
                    and checkpoint_index == 0
                )
                mechanics_state = SimpleNamespace(
                    active_carrier_contact_vertex_indices=(),
                    rigid_sdf_target_voxel_mm=0.125,
                    final_pose_error_mm=0.0,
                    carrier_contact_active=False,
                    carrier_contact_occurred=False,
                    first_carrier_contact_step=None,
                    carrier_collision_enabled=False,
                    inverted_tetrahedra=0,
                    max_soft_contact_overflow=0,
                    max_rigid_contact_overflow=0,
                    max_support_displacement_mm=0.0,
                    max_carrier_penetration_mm=0.0,
                    carrier_interface_contact_count=0,
                )
                checkpoint = SimpleNamespace(
                    checkpoint_index=checkpoint_index,
                    checkpoint_fraction=(checkpoint_index + 1)
                    / len(checkpoint_depths_mm),
                    normalized_indentation_ratio=float(depth_mm) / radius_mm,
                    post_contact_travel_mm=float(depth_mm),
                    cumulative_step_index=checkpoint_index + 1,
                    diagnostics={},
                    state=mechanics_state,
                )
                states.append(
                    SimpleNamespace(
                        checkpoint=checkpoint,
                        optics=optical_result(
                            location_u,
                            radius_mm,
                            depth_mm,
                            hard_failure=(
                                first_state and cls.failure_mode == "numerical"
                            ),
                            pathology=(
                                first_state and cls.failure_mode == "pathology"
                            ),
                        ),
                    )
                )
            return SimpleNamespace(
                normalized_location=float(location_u),
                indenter_radius_mm=float(radius_mm),
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
                checkpoints=tuple(states),
            )

    _Simulation.volume_mesh = volume_mesh
    monkeypatch.setattr(evaluator_module, "LumoSimulation", _Simulation)
    evaluator = Lumo3DTrajectoryEvaluator(
        tmp_path / "numerical",
        complete_trajectory_after_optical_failure=True,
    )

    numerical_failure = evaluator.evaluate(FingertipParameters())

    assert numerical_failure.status == "optics_failure"
    assert numerical_failure.objective_value is None
    assert numerical_failure.objective is None
    assert len(numerical_failure.checkpoint_records) == 18
    assert numerical_failure.report["completed_checkpoint_count"] == 18
    assert numerical_failure.report["objective"]["objective_value"] is not None
    assert numerical_failure.report["evidence_collection_mode"] == (
        "complete_trajectory_after_optical_failure"
    )
    assert numerical_failure.report["volume_mesh"]["gmsh_version"] == "fake-gmsh"

    _Simulation.failure_mode = "pathology"
    pathology_failure = Lumo3DTrajectoryEvaluator(
        tmp_path / "pathology",
    ).evaluate(FingertipParameters())

    assert pathology_failure.failure_scenario == "objective_pathology"
    assert len(pathology_failure.checkpoint_records) == 18
    assert pathology_failure.report["volume_mesh"]["gmsh_version"] == "fake-gmsh"
    assert pathology_failure.report["failure_diagnostics"]["volume_mesh"] == (
        pathology_failure.report["volume_mesh"]
    )


def test_radius_six_is_rejected_as_domain_incompatible_before_mesh_or_newton(tmp_path) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (6.0,), (1.5,))
    result = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol).evaluate(FingertipParameters())
    assert result.status == "domain_incompatible"
    assert result.report["failure_stage"] == "domain_validation"
    assert result.report["failure_diagnostics"]["failure_stage"] == "domain_validation"
    assert Path(result.result_artifact_path).is_file()
    assert result.checkpoint_records == ()


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
        TrajectoryObservation(
            0.25, 5.0, 1.0, np.array([1.0, 0.0]), 0.0, 0.0,
            {"total_transport": 0.0},
        ),
        TrajectoryObservation(
            0.50, 5.0, 1.0, np.array([0.0, 1.0]), 1.0, 1.0,
            {"total_transport": 1.0},
        ),
    ]
    result = compute_trajectory_objective(observations)
    assert result.objective_pathology is True
    assert result.objective_value == result.d_inter


def test_zero_signal_objective_is_explicit_candidate_failure_before_float_conversion() -> None:
    observations = [
        TrajectoryObservation(0.25, 5.0, 1.0, np.zeros(2), 0.0, 0.0),
        TrajectoryObservation(0.50, 5.0, 1.0, np.zeros(2), 0.0, 0.0),
    ]
    objective = compute_trajectory_objective(observations)
    failure = _objective_failure(objective)
    assert failure is not None
    assert failure.status == "optics_failure"
    assert failure.report["failure_scenario"] == "objective_pathology"
    assert failure.objective_value is None
    assert isinstance(failure.report["objective"], dict)
    encoded = json.dumps(objective.to_dict(), allow_nan=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    assert decoded["objective_value"] is None
    assert isinstance(decoded["observation_count"], int)


def test_finite_objective_with_pathology_is_not_accepted() -> None:
    observations = [
        TrajectoryObservation(
            0.25, 5.0, 1.0, np.asarray([1.0, 0.0]), 0.0, 0.0
        ),
        TrajectoryObservation(
            0.50, 5.0, 1.0, np.asarray([0.0, 1.0]), 1.0, 1.0
        ),
    ]
    objective = compute_trajectory_objective(observations)

    failure = _objective_failure(objective)

    assert objective.objective_value is not None
    assert objective.objective_pathology is True
    assert failure is not None
    assert failure.status == "optics_failure"
    assert failure.failure_scenario == "objective_pathology"


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
    assert result.report["failure_scenario"] == "candidate_contact"
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
    assert result.report["failure_scenario"] == "candidate_mechanics_state"
    assert "candidate state is inverted" in (result.failure_message or "")


def test_candidate_optics_error_is_translated_but_base_geometry_error_propagates(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = TrajectoryEvaluationProtocol((0.5,), (5.0,), (1.5,))
    evaluator = Lumo3DTrajectoryEvaluator(tmp_path, protocol=protocol)

    class _Simulation:
        volume_mesh = SimpleNamespace(morphology_fingerprint="mesh")

        @staticmethod
        def run_sphere_contact(*_args, **_kwargs):
            raise CandidateOpticsError(
                "candidate surface is degenerate",
                cause_type="Transport3DCandidateGeometryError",
            )

    class _SimulationFactory:
        @staticmethod
        def from_fingertip(*_args, **_kwargs):
            return _Simulation()

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _SimulationFactory)
    result = evaluator.evaluate(FingertipParameters())

    assert result.status == "optics_failure"
    assert result.failure_scenario == "candidate_optics_geometry"
    assert result.report["cause_type"] == "Transport3DCandidateGeometryError"
    assert result.report["failure_diagnostics"]["cause_type"] == (
        "Transport3DCandidateGeometryError"
    )
    assert Path(result.result_artifact_path).is_file()

    class _FatalSimulation:
        volume_mesh = SimpleNamespace(morphology_fingerprint="mesh")

        @staticmethod
        def run_sphere_contact(*_args, **_kwargs):
            raise Transport3DGeometryError("fixed geometry invariant")

    class _FatalSimulationFactory:
        @staticmethod
        def from_fingertip(*_args, **_kwargs):
            return _FatalSimulation()

    monkeypatch.setattr(evaluator_module, "LumoSimulation", _FatalSimulationFactory)
    with pytest.raises(Transport3DGeometryError, match="fixed geometry invariant"):
        evaluator.evaluate(FingertipParameters())

    assert issubclass(
        Transport3DCandidateGeometryError,
        Transport3DGeometryError,
    )


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
    class _ContactState:
        contact_state_fingerprint = "contact"
        carrier_contact_active = False
        carrier_contact_occurred = False

        def to_dict(self):
            return {
                "contact_state_fingerprint": self.contact_state_fingerprint,
                "carrier_contact_active": self.carrier_contact_active,
                "carrier_contact_occurred": self.carrier_contact_occurred,
            }

    monkeypatch.setattr(
        evaluator_module,
        "build_contact_state_record",
        lambda **_kwargs: _ContactState(),
    )
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
            state=SimpleNamespace(
                active_carrier_contact_vertex_indices=(),
                rigid_sdf_target_voxel_mm=0.125,
                final_pose_error_mm=0.0,
                carrier_contact_active=False,
                carrier_contact_occurred=False,
                first_carrier_contact_step=None,
                carrier_collision_enabled=False,
                inverted_tetrahedra=0,
                max_soft_contact_overflow=0,
                max_rigid_contact_overflow=0,
                max_support_displacement_mm=0.0,
                max_carrier_penetration_mm=0.0,
                carrier_interface_contact_count=0,
            ),
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
