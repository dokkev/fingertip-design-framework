"""Configurable continuous-trajectory FULL_3D LUMO evaluator.

This module owns orchestration and provenance only; mechanics and optical
physics remain in their neutral backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Literal, Mapping

from contact import CandidateContactError
from physics import (
    CandidateMechanicsError,
    PhysicsDependencyError,
    InvalidFingertipMesh,
    MechanicsCheckpointState,
)
from mesh import volume_mesh_settings_for_tier
from mesh.volume.mesh import VolumeMeshDependencyError, VolumeMeshingError
from finger import (
    Fingertip,
    FingertipParameters,
    LED,
    InvalidFingertip,
    InvalidFingertipParameters,
    fingertip_parameters_fingerprint,
    validate_minimum_silicone_thickness,
)
from ray_tracing.optical_mechanics import (
    Transport3DSettings,
    Transport3DDependencyError,
)
from ray_tracing.optical_mechanics.result import Transport3DResult
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    ParameterSpec,
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
)
from lumo.mechanics_contract import (
    DEFAULT_MECHANICS_CONTRACT,
    MechanicsContract,
)
from optimization.objectives import (
    TRAJECTORY_SEPARATION_OBJECTIVE,
    TrajectoryObjectiveConfig,
    TrajectoryObservation,
    TrajectoryObjectiveResult,
    compute_trajectory_objective,
    normalized_field_distance,
)
from optimization.deformed_state_artifact import (
    ContactState,
    build_contact_state_record,
    write_mechanics_artifact,
)
from optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)
from optimization.optical_artifact import (
    energy_record,
    save_case_artifact,
)
from optimization.optical_contract import (
    fingerprint_mapping,
    optical_physics_parameters,
    transport_configuration,
)
from lumo.simulation import (
    ContactOpticalState,
    ContactSimulationResult,
    LumoSimulation,
    LUMO3D_OBSERVATION_LEVEL,
    lumo_optical_settings,
)


TRAJECTORY_EVALUATION_SCHEMA = "lumo3d-trajectory-evaluation-v2"
LUMO_EXECUTION_CONTRACT = "newton-1.4-vbd+full3d-optix-v4"
CURRENT_CELL_HALF_LENGTH_MM = 5.5


@dataclass(frozen=True)
class CheckpointEvaluationRecord:
    """Typed morphology-level record for one mechanics/optics checkpoint."""

    trajectory_id: str
    normalized_location: float
    radius_mm: float
    checkpoint_index: int
    checkpoint_depth_mm: float
    checkpoint_fraction: float
    normalized_indentation_ratio: float
    post_contact_travel_mm: float
    unintended_boundary_clearance_mm: float
    cumulative_step_index: int
    first_contact_travel_mm: float
    first_contact_fingerprint: str
    mechanics_artifact_path: Path
    mechanics_artifact_sha256: str
    final_pose_error_mm: float
    mechanics_state: MechanicsCheckpointState
    contact_state: ContactState
    optical_artifact_path: Path
    optics: Transport3DResult
    debug_diagnostics: Mapping[str, Any] | None = None

    @property
    def total_transport(self) -> float:
        return self.optics.total_transport

    @property
    def carrier_absorbed_weight(self) -> float:
        return float(self.optics.carrier_absorbed_weight)

    def to_dict(self) -> dict[str, Any]:
        """Return the persistence/report representation at the boundary."""

        energy = energy_record(self.optics)
        return {
            "trajectory_id": self.trajectory_id,
            "normalized_location": self.normalized_location,
            "radius_mm": self.radius_mm,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_depth_mm": self.checkpoint_depth_mm,
            "checkpoint_fraction": self.checkpoint_fraction,
            "normalized_indentation_ratio": self.normalized_indentation_ratio,
            "post_contact_travel_mm": self.post_contact_travel_mm,
            "unintended_boundary_clearance_mm": self.unintended_boundary_clearance_mm,
            "cumulative_step_index": self.cumulative_step_index,
            "first_contact_travel_mm": self.first_contact_travel_mm,
            "first_contact_fingerprint": self.first_contact_fingerprint,
            "mechanics_artifact_path": str(self.mechanics_artifact_path),
            "mechanics_artifact_sha256": self.mechanics_artifact_sha256,
            "final_pose_error_mm": self.final_pose_error_mm,
            "mechanics_state": {
                "active_carrier_contact_vertex_indices": list(
                    self.mechanics_state.active_carrier_contact_vertex_indices
                ),
                "rigid_sdf_target_voxel_mm": self.mechanics_state.rigid_sdf_target_voxel_mm,
                "final_pose_error_mm": self.mechanics_state.final_pose_error_mm,
                "carrier_contact_active": self.mechanics_state.carrier_contact_active,
                "carrier_contact_occurred": self.mechanics_state.carrier_contact_occurred,
                "first_carrier_contact_step": self.mechanics_state.first_carrier_contact_step,
                "carrier_collision_enabled": self.mechanics_state.carrier_collision_enabled,
                "inverted_tetrahedra": self.mechanics_state.inverted_tetrahedra,
                "max_soft_contact_overflow": self.mechanics_state.max_soft_contact_overflow,
                "max_rigid_contact_overflow": self.mechanics_state.max_rigid_contact_overflow,
                "max_support_displacement_mm": self.mechanics_state.max_support_displacement_mm,
                "max_carrier_penetration_mm": self.mechanics_state.max_carrier_penetration_mm,
                "carrier_interface_contact_count": self.mechanics_state.carrier_interface_contact_count,
            },
            "contact_state": self.contact_state.to_dict(),
            "optical_artifact_path": str(self.optical_artifact_path),
            "mechanics_diagnostics": (
                {} if self.debug_diagnostics is None else dict(self.debug_diagnostics)
            ),
            "artifact": str(self.optical_artifact_path),
            "artifact_field": str(self.optical_artifact_path.with_suffix(".npz")),
            **energy,
        }


@dataclass(frozen=True)
class Lumo3DTrajectoryEvaluation:
    status: Literal[
        "success",
        "invalid_design",
        "mesh_failure",
        "domain_incompatible",
        "mechanics_failure",
        "optics_failure",
    ]
    objective_value: float | None
    objective: TrajectoryObjectiveResult | None
    checkpoint_records: tuple[CheckpointEvaluationRecord, ...]
    report: Mapping[str, Any]
    failure_message: str | None = None
    failure_scenario: str | None = None
    result_artifact_path: str | None = None

    @property
    def score(self) -> float | None:
        return self.objective_value


def _failure(
    status: str,
    message: str,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    failure_scenario: str | None = None,
    checkpoint_records: tuple[CheckpointEvaluationRecord, ...] = (),
) -> Lumo3DTrajectoryEvaluation:
    return Lumo3DTrajectoryEvaluation(
        status=status,  # type: ignore[arg-type]
        objective_value=None,
        objective=None,
        checkpoint_records=checkpoint_records,
        report={} if diagnostics is None else dict(diagnostics),
        failure_message=message,
        failure_scenario=failure_scenario,
    )


def _objective_failure(
    objective: TrajectoryObjectiveResult,
    checkpoint_records: tuple[CheckpointEvaluationRecord, ...] = (),
) -> Lumo3DTrajectoryEvaluation | None:
    """Translate the expected zero-signal objective pathology explicitly."""
    if objective.objective_value is not None:
        return None
    return _failure(
        "optics_failure",
        "objective pathology produced no finite objective value: "
        f"{objective.objective_pathology_reason or 'unspecified reason'}",
        diagnostics={
            "failure_stage": "objective",
            "failure_scenario": "objective_pathology",
            "objective": objective.to_dict(),
        },
        failure_scenario="objective_pathology",
        checkpoint_records=checkpoint_records,
    )


class Lumo3DTrajectoryEvaluator:
    """Evaluate each protocol trajectory once and trace each checkpoint."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL,
        objective_config: TrajectoryObjectiveConfig | None = None,
        mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
        device: str = "cuda:0",
        optical_settings: Transport3DSettings | None = None,
        led: LED = LED(),
        fixed_parameters: FingertipParameters | None = None,
    ) -> None:
        if not isinstance(protocol, TrajectoryEvaluationProtocol):
            raise TypeError("protocol must be TrajectoryEvaluationProtocol")
        if not isinstance(mechanics_contract, MechanicsContract):
            raise TypeError("mechanics_contract must be MechanicsContract")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self.artifact_root = Path(artifact_root)
        self.protocol = protocol
        self.objective_config = objective_config or TrajectoryObjectiveConfig()
        self.mechanics_contract = mechanics_contract
        self.device = device
        self.fixed_parameters = fixed_parameters or FingertipParameters()
        if not isinstance(self.fixed_parameters, FingertipParameters):
            raise TypeError("fixed_parameters must be FingertipParameters")
        if not isinstance(led, LED):
            raise TypeError("led must be an LED")
        self.led = led
        self.settings = optical_settings or lumo_optical_settings()
        if not isinstance(self.settings, Transport3DSettings):
            raise TypeError("optical_settings must be a Transport3DSettings")
        if not self.settings.retain_internal_path_field:
            raise ValueError(
                "optimization requires optical_settings.retain_internal_path_field=True"
            )
        self.evaluation_contract_id = trajectory_evaluation_contract_id(
            protocol=self.protocol,
            objective_config=self.objective_config,
            mechanics_contract=self.mechanics_contract,
            optical_settings=self.settings,
            led=self.led,
            fixed_parameters=self.fixed_parameters,
            device=self.device,
        )

    def _domain_failure(self, radius_mm: float) -> str | None:
        clearance = CURRENT_CELL_HALF_LENGTH_MM - float(radius_mm)
        if clearance <= 0.0:
            return (
                f"radius {radius_mm:g} mm is incompatible with the current "
                f"11 mm representative cell (end clearance={clearance:g} mm)"
            )
        return None

    def evaluate(self, parameters: FingertipParameters) -> Lumo3DTrajectoryEvaluation:
        """Run the visible candidate-to-objective scientific flow."""
        if not isinstance(parameters, FingertipParameters):
            raise TypeError("parameters must be FingertipParameters")
        if _fixed_fingertip_inputs(parameters) != _fixed_fingertip_inputs(
            self.fixed_parameters
        ):
            raise ValueError(
                "candidate fixed fingertip inputs do not match the evaluation contract"
            )
        started = time.perf_counter()
        stage = "candidate_validation"
        checkpoint_records: list[CheckpointEvaluationRecord] = []
        try:
            validate_minimum_silicone_thickness(parameters)
            tip = Fingertip(
                parameters,
                led=self.led,
            )
            morphology_id = _candidate_id(parameters)
            candidate_root = (
                self.artifact_root
                / f"contract_{self.evaluation_contract_id.replace(':', '_')}"
                / f"protocol_{self.protocol.fingerprint}"
                / "candidates"
                / morphology_id
            )
            for radius in self.protocol.indenter_radii_mm:
                reason = self._domain_failure(radius)
                if reason is not None:
                    return _failure(
                        "domain_incompatible",
                        reason,
                        diagnostics={"radius_mm": radius, "failure_stage": "domain_validation"},
                        failure_scenario="domain_validation",
                    )

            stage = "mesh_preparation"
            simulation = LumoSimulation.from_fingertip(
                tip,
                initial_gap_mm=self.protocol.initial_gap_mm,
                mechanics_contract=self.mechanics_contract,
                device=self.device,
                optical_settings=self.settings,
            )
            volume_mesh = simulation.volume_mesh
            configuration = transport_configuration(
                self.settings,
                material=optical_physics_parameters(tip),
            )
            configuration_fingerprint = fingerprint_mapping(configuration)
            objective_contract = {
                "schema": "trajectory-objective-contract-fixed-depth-v1",
                "name": TRAJECTORY_SEPARATION_OBJECTIVE.serialized_name,
                "radius_penalty_weight": self.objective_config.radius_penalty_weight,
            }
            evaluation_identity = {
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "protocol_fingerprint": self.protocol.fingerprint,
                "mechanics_contract_fingerprint": self.mechanics_contract.fingerprint,
                "optical_configuration_fingerprint": configuration_fingerprint,
                "objective_contract_fingerprint": fingerprint_mapping(objective_contract),
                "evaluation_contract_id": self.evaluation_contract_id,
            }
            observations: list[TrajectoryObservation] = []
            mechanics_runtime_s = 0.0
            optics_runtime_s = 0.0
            stage = "trajectory_evaluation"
            for radius in self.protocol.indenter_radii_mm:
                for location in self.protocol.contact_locations_u:
                    contact_result = simulation.run_sphere_contact(
                        location_u=location,
                        radius_mm=radius,
                        checkpoint_depths_mm=self.protocol.checkpoint_depths_mm,
                    )
                    mechanics_runtime_s += contact_result.mechanics_seconds
                    optics_runtime_s += contact_result.optics_seconds
                    for state in contact_result.checkpoints:
                        trajectory_id = (
                            f"u_{contact_result.normalized_location:.3f}"
                            f"__radius_{contact_result.indenter_radius_mm:.3f}"
                        )
                        mechanics_artifact = (
                            candidate_root
                            / "mechanics"
                            / trajectory_id
                            / f"checkpoint_{state.checkpoint.checkpoint_index:02d}.npz"
                        )
                        mechanics_artifact_sha256 = write_mechanics_artifact(
                            mechanics_artifact,
                            state.checkpoint,
                            simulation.prepared,
                        )
                        contact_state = build_contact_state_record(
                            morphology_fingerprint=volume_mesh.morphology_fingerprint,
                            protocol=self.protocol,
                            location_u=contact_result.normalized_location,
                            radius_mm=contact_result.indenter_radius_mm,
                            checkpoint=state.checkpoint,
                            post_contact_travel_mm=state.checkpoint.post_contact_travel_mm,
                            unintended_boundary_clearance_mm=(
                                contact_result.unintended_boundary_clearance_mm
                            ),
                            source_node_ids=tuple(
                                int(value) for value in simulation.prepared.source_node_ids
                            ),
                            mechanics_artifact_sha256=mechanics_artifact_sha256,
                        )
                        pose_error_mm = _final_pose_error_mm(state)
                        optical_artifact = (
                            candidate_root
                            / f"location_u_{contact_result.normalized_location:.3f}"
                            f"__radius_{contact_result.indenter_radius_mm:.3f}"
                            f"__depth_{state.checkpoint.post_contact_travel_mm:.3f}mm"
                            f"__checkpoint_{state.checkpoint.checkpoint_index:02d}.json"
                        )
                        record = _checkpoint_record(
                            contact_result,
                            state,
                            mechanics_artifact_path=mechanics_artifact,
                            mechanics_artifact_sha256=mechanics_artifact_sha256,
                            contact_state=contact_state,
                            final_pose_error_mm=pose_error_mm,
                            optical_artifact_path=optical_artifact,
                        )
                        checkpoint_records.append(record)
                        contract = {
                            "schema": TRAJECTORY_EVALUATION_SCHEMA,
                            "objective": TRAJECTORY_SEPARATION_OBJECTIVE.serialized_name,
                            "observation_level": LUMO3D_OBSERVATION_LEVEL,
                            "morphology_id": morphology_id,
                            "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                            "protocol": self.protocol.to_dict(),
                            "protocol_fingerprint": self.protocol.fingerprint,
                            "mechanics_contract": self.mechanics_contract.to_dict(),
                            "evaluation_identity": evaluation_identity,
                            "mechanics_artifact_sha256": mechanics_artifact_sha256,
                            "mechanics_dimension": "3D",
                            "geometry_mode": "full3d_surface",
                            "full3d_surface_provenance": "actual_deformed_3d_volume_state",
                            "contact_state": contact_state.to_dict(),
                            "transport_configuration": configuration,
                            "transport_configuration_fingerprint": configuration_fingerprint,
                        }
                        save_case_artifact(optical_artifact, state.optics, contract)
                        observations.append(
                            TrajectoryObservation(
                                location_u=contact_result.normalized_location,
                                radius_mm=contact_result.indenter_radius_mm,
                                checkpoint_depth_mm=state.checkpoint.post_contact_travel_mm,
                                field=state.optics.field,
                                total_transport=state.optics.total_transport,
                                escaped_weight=state.optics.escaped_weight,
                                debug_diagnostics=None,
                            )
                        )

            objective = compute_trajectory_objective(observations, self.objective_config)
            objective_failure = _objective_failure(
                objective,
                tuple(checkpoint_records),
            )
            if objective_failure is not None:
                return objective_failure
            trajectory_metrics = _trajectory_metrics(observations, checkpoint_records)
            summary = {
                "schema": TRAJECTORY_EVALUATION_SCHEMA,
                "objective_name": TRAJECTORY_SEPARATION_OBJECTIVE.serialized_name,
                "objective_value": objective.objective_value,
                "protocol": self.protocol.to_dict(),
                "protocol_fingerprint": self.protocol.fingerprint,
                "mechanics_contract": self.mechanics_contract.to_dict(),
                "objective_contract": objective_contract,
                "evaluation_identity": evaluation_identity,
                "evaluation_identity_fingerprint": fingerprint_mapping(evaluation_identity),
                "requested_trajectory_count": self.protocol.trajectory_count,
                "actual_newton_trajectory_count": self.protocol.trajectory_count,
                "checkpoint_count": self.protocol.checkpoint_count,
                "optical_state_count": len(checkpoint_records),
                "mechanics_runtime_s": mechanics_runtime_s,
                "optics_runtime_s": optics_runtime_s,
                "total_runtime_s": time.perf_counter() - started,
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "transport_configuration_fingerprint": configuration_fingerprint,
                "checkpoint_records": [record.to_dict() for record in checkpoint_records],
                "objective": objective.to_dict(),
                "trajectory_metrics": trajectory_metrics,
            }
            candidate_root.mkdir(parents=True, exist_ok=True)
            result_artifact_path = (
                candidate_root / "trajectory_evaluation.json"
            ).resolve()
            summary["result_artifact_path"] = str(result_artifact_path)
            result_artifact_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return Lumo3DTrajectoryEvaluation(
                status="success",
                objective_value=float(objective.objective_value),
                objective=objective,
                checkpoint_records=tuple(checkpoint_records),
                report=summary,
                result_artifact_path=str(result_artifact_path),
            )
        except (
            Transport3DDependencyError,
            VolumeMeshDependencyError,
            PhysicsDependencyError,
        ):
            raise
        except (InvalidFingertip, InvalidFingertipParameters) as exc:
            return _failure(
                "invalid_design",
                f"{type(exc).__name__}: {exc}",
                diagnostics={"failure_stage": stage},
                checkpoint_records=tuple(checkpoint_records),
            )
        except (VolumeMeshingError, InvalidFingertipMesh) as exc:
            return _failure(
                "mesh_failure",
                f"{type(exc).__name__}: {exc}",
                diagnostics={"failure_stage": stage},
                checkpoint_records=tuple(checkpoint_records),
            )
        except CandidateContactError as exc:
            return _failure(
                "mechanics_failure",
                f"{type(exc).__name__}: {exc}",
                diagnostics={
                    "failure_stage": "mechanics",
                    "failure_scenario": "candidate_contact",
                },
                failure_scenario="candidate_contact",
                checkpoint_records=tuple(checkpoint_records),
            )
        except CandidateMechanicsError as exc:
            return _failure(
                "mechanics_failure",
                f"{type(exc).__name__}: {exc}",
                diagnostics={
                    "failure_stage": "mechanics",
                    "failure_scenario": "candidate_mechanics_state",
                },
                failure_scenario="candidate_mechanics_state",
                checkpoint_records=tuple(checkpoint_records),
            )
@dataclass(frozen=True)
class Lumo3DTrajectoryStudy:
    design_space: DesignSpace
    artifact_root: Path
    protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL
    objective_config: TrajectoryObjectiveConfig = TrajectoryObjectiveConfig()
    mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT
    device: str = "cuda:0"
    optical_settings: Transport3DSettings = lumo_optical_settings()
    led: LED = LED()

    @property
    def evaluation_contract_id(self) -> str:
        return trajectory_evaluation_contract_id(
            protocol=self.protocol,
            objective_config=self.objective_config,
            mechanics_contract=self.mechanics_contract,
            optical_settings=self.optical_settings,
            led=self.led,
            fixed_parameters=self.design_space.nominal_parameters,
            device=self.device,
        )

    def create_evaluator(self) -> Lumo3DTrajectoryEvaluator:
        return Lumo3DTrajectoryEvaluator(
            self.artifact_root,
            protocol=self.protocol,
            objective_config=self.objective_config,
            mechanics_contract=self.mechanics_contract,
            device=self.device,
            optical_settings=self.optical_settings,
            led=self.led,
            fixed_parameters=self.design_space.nominal_parameters,
        )


def create_lumo3d_trajectory_study(
    artifact_root: str | Path,
    *,
    protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL,
    objective_config: TrajectoryObjectiveConfig | None = None,
    mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
    device: str = "cuda:0",
    optical_settings: Transport3DSettings | None = None,
    led: LED = LED(),
    nominal_parameters: FingertipParameters | None = None,
    search_bounds: tuple[ParameterSpec, ...] = PRODUCTION_SEARCH_BOUNDS,
) -> Lumo3DTrajectoryStudy:
    """Build the lightweight 3D study/configuration boundary."""

    bounds = tuple(search_bounds)
    if len(bounds) != len(OPTIMIZABLE_PARAMETER_NAMES):
        raise ValueError(
            "search_bounds must contain exactly one bound for each active "
            "morphology variable"
        )
    if any(not isinstance(bound, ParameterSpec) for bound in bounds):
        raise TypeError("search_bounds must contain ParameterSpec values")

    design_space = DesignSpace(
        nominal_parameters
        or FingertipParameters(void_height=PRODUCTION_NOMINAL_VOID_HEIGHT_MM),
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in bounds
        ),
    )
    return Lumo3DTrajectoryStudy(
        design_space=design_space,
        artifact_root=Path(artifact_root),
        protocol=protocol,
        objective_config=objective_config or TrajectoryObjectiveConfig(),
        mechanics_contract=mechanics_contract,
        device=device,
        optical_settings=optical_settings or lumo_optical_settings(),
        led=led,
    )


def _fixed_fingertip_inputs(
    parameters: FingertipParameters,
) -> dict[str, object]:
    values = asdict(parameters)
    for name in OPTIMIZABLE_PARAMETER_NAMES:
        values.pop(name.value)
    return values


def trajectory_evaluation_contract_id(
    *,
    protocol: TrajectoryEvaluationProtocol,
    objective_config: TrajectoryObjectiveConfig,
    mechanics_contract: MechanicsContract,
    optical_settings: Transport3DSettings,
    led: LED,
    fixed_parameters: FingertipParameters,
    device: str,
) -> str:
    """Fingerprint every fixed input that can change a production result."""
    payload = {
        "schema": TRAJECTORY_EVALUATION_SCHEMA,
        "protocol": protocol.to_dict(),
        "objective": asdict(objective_config),
        "mechanics": mechanics_contract.to_dict(),
        "execution": {
            "schema": LUMO_EXECUTION_CONTRACT,
            "device": device,
            "representative_cell_half_length_mm": CURRENT_CELL_HALF_LENGTH_MM,
            "volume_mesh": asdict(volume_mesh_settings_for_tier("search")),
            "fixed_fingertip_inputs": _fixed_fingertip_inputs(fixed_parameters),
        },
        "transport": asdict(optical_settings),
        "led": {
            "width_mm": float(led.width_mm),
            "height_mm": float(led.height_mm),
            "relative_radiant_power": float(led.relative_radiant_power),
            "emission_half_angle_deg": float(led.emission_half_angle_deg),
        },
        "optical_parameters": asdict(fixed_parameters.optical),
    }
    return f"{TRAJECTORY_EVALUATION_SCHEMA}:{fingerprint_mapping(payload)}"


def _candidate_id(parameters: FingertipParameters) -> str:
    """Derive the artifact label from the canonical morphology identity."""
    return fingertip_parameters_fingerprint(parameters)


def _checkpoint_record(
    contact_result: ContactSimulationResult,
    state: ContactOpticalState,
    *,
    mechanics_artifact_path: Path,
    mechanics_artifact_sha256: str,
    contact_state: ContactState,
    final_pose_error_mm: float,
    optical_artifact_path: Path,
) -> CheckpointEvaluationRecord:
    """Create the authoritative in-memory record for one checkpoint."""

    checkpoint = state.checkpoint
    alignment = contact_result.alignment
    first_contact = contact_result.first_contact
    mechanics_state = checkpoint.state
    trajectory_id = (
        f"u_{contact_result.normalized_location:.3f}"
        f"__radius_{contact_result.indenter_radius_mm:.3f}"
    )
    return CheckpointEvaluationRecord(
        trajectory_id=trajectory_id,
        normalized_location=contact_result.normalized_location,
        radius_mm=contact_result.indenter_radius_mm,
        checkpoint_index=checkpoint.checkpoint_index,
        checkpoint_depth_mm=checkpoint.post_contact_travel_mm,
        checkpoint_fraction=checkpoint.checkpoint_fraction,
        normalized_indentation_ratio=checkpoint.normalized_indentation_ratio,
        post_contact_travel_mm=checkpoint.post_contact_travel_mm,
        unintended_boundary_clearance_mm=float(
            contact_result.unintended_boundary_clearance_mm
        ),
        cumulative_step_index=checkpoint.cumulative_step_index,
        first_contact_travel_mm=first_contact.travel_to_contact_mm,
        first_contact_fingerprint=fingerprint_mapping(
            {
                "target_point_mm": alignment.target_point_mm,
                "outward_normal": alignment.outward_normal,
                "approach_direction": alignment.approach_direction,
                "radius_mm": contact_result.indenter_radius_mm,
                "contact_pose_mm": first_contact.contact_pose.translation_mm,
            }
        ),
        mechanics_artifact_path=mechanics_artifact_path,
        mechanics_artifact_sha256=mechanics_artifact_sha256,
        final_pose_error_mm=float(final_pose_error_mm),
        mechanics_state=mechanics_state,
        contact_state=contact_state,
        optical_artifact_path=optical_artifact_path,
        optics=state.optics,
        debug_diagnostics=checkpoint.diagnostics,
    )


def _final_pose_error_mm(state: ContactOpticalState) -> float:
    return state.checkpoint.state.final_pose_error_mm


def _trajectory_metrics(
    observations: tuple[TrajectoryObservation, ...],
    checkpoint_records: list[CheckpointEvaluationRecord],
) -> dict[str, Mapping[str, Any]]:
    """Compute supporting progression metrics without changing the objective."""
    grouped: dict[str, list[tuple[TrajectoryObservation, CheckpointEvaluationRecord]]] = {}
    for observation, record in zip(observations, checkpoint_records, strict=True):
        trajectory_id = record.trajectory_id
        grouped.setdefault(trajectory_id, []).append((observation, record))
    metrics: dict[str, Mapping[str, Any]] = {}
    for trajectory_id, values in grouped.items():
        ordered = sorted(values, key=lambda item: item[0].checkpoint_depth_mm)
        adjacent = [
            normalized_field_distance(first[0].field, second[0].field)
            for first, second in zip(ordered, ordered[1:])
        ]
        onset = next(
            (
                record.checkpoint_index
                for _, record in ordered
                if record.contact_state.carrier_contact_active
            ),
            None,
        )
        metrics[trajectory_id] = {
            "optical_path_length": float(sum(adjacent)),
            "adjacent_checkpoint_optical_distances": [float(value) for value in adjacent],
            "carrier_contact_onset_checkpoint": onset,
            "transport_progression": [
                {
                    "checkpoint_index": record.checkpoint_index,
                    "post_contact_travel_mm": record.post_contact_travel_mm,
                    "total_transport": record.total_transport,
                    "carrier_absorbed_weight": record.carrier_absorbed_weight,
                }
                for _, record in ordered
            ],
        }
    same_depth_pairs: list[tuple[float, dict[str, Any]]] = []
    for index, first in enumerate(observations):
        for second in observations[index + 1:]:
            if first.location_u == second.location_u or first.checkpoint_depth_mm != second.checkpoint_depth_mm:
                continue
            distance = normalized_field_distance(first.field, second.field)
            same_depth_pairs.append(
                (
                    distance,
                    {
                        "distance": float(distance),
                        "first_location_u": first.location_u,
                        "second_location_u": second.location_u,
                        "checkpoint_depth_mm": first.checkpoint_depth_mm,
                    },
                )
            )
    return {
        "per_trajectory": metrics,
        "minimum_same_depth_inter_location_separation": (
            min(same_depth_pairs, key=lambda item: item[0])[1]
            if same_depth_pairs
            else None
        ),
    }


__all__ = [
    "CURRENT_CELL_HALF_LENGTH_MM",
    "TRAJECTORY_EVALUATION_SCHEMA",
    "trajectory_evaluation_contract_id",
    "Lumo3DTrajectoryEvaluation",
    "CheckpointEvaluationRecord",
    "Lumo3DTrajectoryEvaluator",
    "Lumo3DTrajectoryStudy",
    "create_lumo3d_trajectory_study",
]
