"""Configurable continuous-trajectory FULL_3D LUMO evaluator.

This module owns orchestration and provenance only; mechanics and optical
physics remain in their neutral backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal, Mapping

import numpy as np

from contact import CandidateContactError
from physics import (
    PhysicsDependencyError,
    InvalidFingertipMesh,
)
from mesh.volume.mesh import VolumeMeshDependencyError, VolumeMeshingError
from mesh.fingertip.geometry import GmshDependencyError
from model import (
    Fingertip,
    FingertipParameters,
    InvalidFingertip,
    InvalidFingertipParameters,
    validate_minimum_silicone_thickness,
)
from optics.transport3d import (
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DResultError,
    Transport3DTraceError,
)
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
)
from optimization.mechanics_contract import (
    DEFAULT_MECHANICS_CONTRACT,
    MechanicsContract,
)
from optimization.objectives import (
    OBJECTIVE_NAME,
    TrajectoryObjectiveConfig,
    TrajectoryObservation,
    TrajectoryObjectiveResult,
    compute_trajectory_objective,
    normalized_field_distance,
)
from optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)
from optimization.optical_artifact import (
    energy_record,
    fingerprint_mapping,
    optical_physics_parameters,
    save_case_artifact,
    transport_configuration,
)
from lumo.simulation import (
    ContactOpticalState,
    ContactSimulationResult,
    LumoSimulation,
    LUMO3D_OBSERVATION_LEVEL,
    lumo_optical_settings,
)


TRAJECTORY_EVALUATION_SCHEMA = "lumo3d-trajectory-evaluation-v1"
TRAJECTORY_EVALUATION_CONTRACT_ID = TRAJECTORY_EVALUATION_SCHEMA
CURRENT_CELL_HALF_LENGTH_MM = 5.5


def _candidate_id(parameters: FingertipParameters) -> str:
    """Preserve the artifact identity contract for the six search fields."""
    payload = {
        "flat_pad_height": float(parameters.flat_pad_height),
        "semielliptical_pad_height": float(parameters.semielliptical_pad_height),
        "stem_width": float(parameters.stem_width),
        "stem_height": float(parameters.stem_height),
        "void_width": float(parameters.void_width),
        "void_height": float(parameters.void_height),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


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
    objective: TrajectoryObjectiveResult | Mapping[str, Any]
    trajectory_diagnostics: tuple[Mapping[str, Any], ...]
    checkpoint_diagnostics: tuple[Mapping[str, Any], ...]
    optical_diagnostics: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    failure_message: str | None = None

    @property
    def score(self) -> float | None:
        return self.objective_value


def _failure(status: str, message: str, *, diagnostics: Mapping[str, Any] | None = None) -> Lumo3DTrajectoryEvaluation:
    return Lumo3DTrajectoryEvaluation(
        status=status,  # type: ignore[arg-type]
        objective_value=None,
        objective={},
        trajectory_diagnostics=(),
        checkpoint_diagnostics=(),
        optical_diagnostics=(),
        diagnostics={} if diagnostics is None else dict(diagnostics),
        failure_message=message,
    )


def _objective_failure(
    objective: TrajectoryObjectiveResult,
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
    )


def _checkpoint_record(
    contact_result: ContactSimulationResult,
    state: ContactOpticalState,
) -> dict[str, Any]:
    """Convert one simulation state to the persisted evaluator boundary."""

    checkpoint = state.checkpoint
    alignment = contact_result.alignment
    first_contact = contact_result.first_contact
    return {
        "trajectory_id": state.trajectory_id,
        "normalized_location": state.normalized_location,
        "radius_mm": state.indenter_radius_mm,
        "checkpoint_index": checkpoint.checkpoint_index,
        "checkpoint_depth_mm": checkpoint.post_contact_travel_mm,
        "checkpoint_fraction": checkpoint.checkpoint_fraction,
        "normalized_indentation_ratio": checkpoint.normalized_indentation_ratio,
        "post_contact_travel_mm": checkpoint.post_contact_travel_mm,
        "unintended_boundary_clearance_mm": float(
            state.contact_state["unintended_boundary_clearance_mm"]
        ),
        "cumulative_step_index": checkpoint.cumulative_step_index,
        "first_contact_travel_mm": first_contact.travel_to_contact_mm,
        "first_contact_fingerprint": fingerprint_mapping(
            {
                "target_point_mm": alignment.target_point_mm,
                "outward_normal": alignment.outward_normal,
                "approach_direction": alignment.approach_direction,
                "radius_mm": state.indenter_radius_mm,
                "contact_pose_mm": first_contact.contact_pose.translation_mm,
            }
        ),
        "mechanics_artifact_path": str(state.mechanics_artifact_path),
        "mechanics_artifact_sha256": state.mechanics_artifact_sha256,
        "final_pose_error_mm": state.final_pose_error_mm,
        "mechanics_diagnostics": dict(checkpoint.diagnostics),
    }


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
        mechanics_mode: str = "search",
    ) -> None:
        if not isinstance(protocol, TrajectoryEvaluationProtocol):
            raise TypeError("protocol must be TrajectoryEvaluationProtocol")
        if not isinstance(mechanics_contract, MechanicsContract):
            raise TypeError("mechanics_contract must be MechanicsContract")
        if mechanics_mode != "search":
            raise ValueError("Lumo3DTrajectoryEvaluator uses the frozen search mechanics contract")
        self.artifact_root = Path(artifact_root)
        self.protocol = protocol
        self.objective_config = objective_config or TrajectoryObjectiveConfig()
        self.mechanics_contract = mechanics_contract
        self.device = device
        self.mechanics_mode = mechanics_mode
        self.settings = lumo_optical_settings()

    def evaluate(self, parameters: FingertipParameters) -> Lumo3DTrajectoryEvaluation:
        """Run validation, mesh/contact mechanics, optical states, and objective.

        The implementation below follows the scientific order directly:
        candidate validation → fingertip/mesh → contact trajectories → mechanics
        checkpoints → optical observations → objective → persisted result.
        """
        return self._evaluate(parameters)

    def _domain_failure(self, radius_mm: float) -> str | None:
        clearance = CURRENT_CELL_HALF_LENGTH_MM - float(radius_mm)
        if clearance <= 0.0:
            return (
                f"radius {radius_mm:g} mm is incompatible with the current "
                f"11 mm representative cell (end clearance={clearance:g} mm)"
            )
        return None

    def _evaluate(self, parameters: FingertipParameters) -> Lumo3DTrajectoryEvaluation:
        started = time.perf_counter()
        stage = "mechanics"
        try:
            validate_minimum_silicone_thickness(parameters)
            tip = Fingertip(parameters)
            morphology_id = _candidate_id(parameters)
            candidate_root = (
                self.artifact_root
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
                    )

            simulation = LumoSimulation.from_fingertip(
                tip,
                artifact_root=candidate_root,
                protocol=self.protocol,
                mechanics_contract=self.mechanics_contract,
                device=self.device,
                settings=self.settings,
            )
            volume_mesh = simulation.volume_mesh
            mechanics_finished = time.perf_counter()
            stage = "optics"
            configuration = transport_configuration(
                self.settings,
                material=optical_physics_parameters(tip),
                source={"model": "existing Fingertip optical source", "evaluator": TRAJECTORY_EVALUATION_SCHEMA},
            )
            objective_contract = {
                "schema": "trajectory-objective-contract-fixed-depth-v1",
                "name": OBJECTIVE_NAME,
                "radius_penalty_weight": self.objective_config.radius_penalty_weight,
            }
            evaluation_identity = {
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "protocol_fingerprint": self.protocol.fingerprint,
                "mechanics_contract_fingerprint": self.mechanics_contract.fingerprint,
                "optical_configuration_fingerprint": fingerprint_mapping(configuration),
                "objective_contract_fingerprint": fingerprint_mapping(objective_contract),
            }
            trajectory_records: list[Mapping[str, Any]] = []
            optical_records: list[Mapping[str, Any]] = []
            observations: list[TrajectoryObservation] = []
            optics_started = time.perf_counter()
            for radius in self.protocol.indenter_radii_mm:
                for location in self.protocol.contact_locations_u:
                    contact_result = simulation.run_sphere_contact(
                        location_u=location,
                        radius_mm=radius,
                        checkpoint_depths_mm=self.protocol.checkpoint_depths_mm,
                    )
                    for state in contact_result.checkpoints:
                        record = _checkpoint_record(contact_result, state)
                        trajectory_records.append(record)
                        configuration_fingerprint = fingerprint_mapping(configuration)
                        artifact = (
                            candidate_root
                            / f"location_u_{state.normalized_location:.3f}"
                            f"__radius_{state.indenter_radius_mm:.3f}"
                            f"__depth_{state.checkpoint.post_contact_travel_mm:.3f}mm"
                            f"__checkpoint_{state.checkpoint.checkpoint_index:02d}.json"
                        )
                        contract = {
                            "schema": TRAJECTORY_EVALUATION_SCHEMA,
                            "objective": OBJECTIVE_NAME,
                            "observation_level": LUMO3D_OBSERVATION_LEVEL,
                            "morphology_id": morphology_id,
                            "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                            "protocol": self.protocol.to_dict(),
                            "protocol_fingerprint": self.protocol.fingerprint,
                            "mechanics_contract": self.mechanics_contract.to_dict(),
                            "evaluation_identity": evaluation_identity,
                            "mechanics_artifact_sha256": state.mechanics_artifact_sha256,
                            "mechanics_dimension": "3D",
                            "geometry_mode": "full3d_surface",
                            "full3d_surface_provenance": "actual_deformed_3d_volume_state",
                            "contact_state": dict(state.contact_state),
                            "transport_configuration": configuration,
                            "transport_configuration_fingerprint": configuration_fingerprint,
                        }
                        save_case_artifact(artifact, state.optics, contract)
                        energy = energy_record(state.optics)
                        optical_record = dict(record)
                        optical_record.update(energy)
                        optical_record.update(
                            {
                                "artifact": str(artifact),
                                "artifact_field": str(artifact.with_suffix(".npz")),
                                "contact_state": dict(state.contact_state),
                                "contact_state_fingerprint": state.contact_state[
                                    "contact_state_fingerprint"
                                ],
                                "carrier_contact_active": state.contact_state[
                                    "carrier_contact_active"
                                ],
                                "carrier_contact_occurred": state.contact_state[
                                    "carrier_contact_occurred"
                                ],
                                "transport_configuration_fingerprint": configuration_fingerprint,
                                "evaluation_identity": evaluation_identity,
                            }
                        )
                        optical_records.append(optical_record)
                        observations.append(
                            TrajectoryObservation(
                                location_u=state.normalized_location,
                                radius_mm=state.indenter_radius_mm,
                                checkpoint_depth_mm=state.checkpoint.post_contact_travel_mm,
                                field=state.optics.field,
                                diagnostics=energy,
                            )
                        )

            objective = compute_trajectory_objective(observations, self.objective_config)
            objective_failure = _objective_failure(objective)
            if objective_failure is not None:
                return objective_failure
            trajectory_metrics = _trajectory_metrics(tuple(observations), optical_records)
            summary = {
                "schema": TRAJECTORY_EVALUATION_SCHEMA,
                "objective_name": OBJECTIVE_NAME,
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
                "optical_state_count": len(optical_records),
                "mechanics_runtime_s": mechanics_finished - started,
                "optics_runtime_s": time.perf_counter() - optics_started,
                "total_runtime_s": time.perf_counter() - started,
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "transport_configuration_fingerprint": fingerprint_mapping(configuration),
                "trajectory_records": trajectory_records,
                "optical_records": optical_records,
                "objective": objective.to_dict(),
                "trajectory_metrics": trajectory_metrics,
            }
            candidate_root.mkdir(parents=True, exist_ok=True)
            (candidate_root / "trajectory_evaluation.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return Lumo3DTrajectoryEvaluation(
                status="success",
                objective_value=float(objective.objective_value),
                objective=objective,
                trajectory_diagnostics=tuple(trajectory_records),
                checkpoint_diagnostics=tuple(trajectory_records),
                optical_diagnostics=tuple(optical_records),
                diagnostics=summary,
            )
        except (
            Transport3DDependencyError,
            VolumeMeshDependencyError,
            GmshDependencyError,
            PhysicsDependencyError,
        ):
            raise
        except (InvalidFingertip, InvalidFingertipParameters) as exc:
            return _failure("invalid_design", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})
        except (VolumeMeshingError, InvalidFingertipMesh) as exc:
            return _failure("mesh_failure", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})
        except CandidateContactError as exc:
            return _failure(
                "mechanics_failure",
                f"{type(exc).__name__}: {exc}",
                diagnostics={
                    "failure_stage": "mechanics",
                    "failure_scenario": "candidate_contact",
                },
            )
        except (Transport3DGeometryError, Transport3DPhysicsError, Transport3DResultError, Transport3DTraceError) as exc:
            return _failure(
                "optics_failure",
                f"{type(exc).__name__}: {exc}",
                diagnostics={"failure_stage": "optics"},
            )


@dataclass(frozen=True)
class Lumo3DTrajectoryStudy:
    design_space: DesignSpace
    artifact_root: Path
    protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL
    objective_config: TrajectoryObjectiveConfig = TrajectoryObjectiveConfig()
    mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT
    device: str = "cuda:0"
    mechanics_mode: str = "search"

    def create_evaluator(self) -> Lumo3DTrajectoryEvaluator:
        return Lumo3DTrajectoryEvaluator(
            self.artifact_root,
            protocol=self.protocol,
            objective_config=self.objective_config,
            mechanics_contract=self.mechanics_contract,
            device=self.device,
            mechanics_mode=self.mechanics_mode,
        )


def create_lumo3d_trajectory_study(
    artifact_root: str | Path,
    *,
    protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL,
    objective_config: TrajectoryObjectiveConfig | None = None,
    mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
    device: str = "cuda:0",
    mechanics_mode: str = "search",
) -> Lumo3DTrajectoryStudy:
    """Build the lightweight 3D study/configuration boundary."""

    design_space = DesignSpace(
        FingertipParameters(void_height=PRODUCTION_NOMINAL_VOID_HEIGHT_MM),
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in PRODUCTION_SEARCH_BOUNDS
        ),
    )
    return Lumo3DTrajectoryStudy(
        design_space=design_space,
        artifact_root=Path(artifact_root),
        protocol=protocol,
        objective_config=objective_config or TrajectoryObjectiveConfig(),
        mechanics_contract=mechanics_contract,
        device=device,
        mechanics_mode=mechanics_mode,
    )


def _trajectory_metrics(
    observations: tuple[TrajectoryObservation, ...],
    optical_records: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Compute supporting progression metrics without changing the objective."""
    grouped: dict[str, list[tuple[TrajectoryObservation, Mapping[str, Any]]]] = {}
    for observation, record in zip(observations, optical_records, strict=True):
        trajectory_id = str(record["trajectory_id"])
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
                int(record["checkpoint_index"])
                for _, record in ordered
                if bool(record.get("carrier_contact_active", False))
            ),
            None,
        )
        metrics[trajectory_id] = {
            "optical_path_length": float(sum(adjacent)),
            "adjacent_checkpoint_optical_distances": [float(value) for value in adjacent],
            "carrier_contact_onset_checkpoint": onset,
            "transport_progression": [
                {
                    "checkpoint_index": int(record["checkpoint_index"]),
                    "post_contact_travel_mm": float(record["post_contact_travel_mm"]),
                    "total_transport": float(record.get("total_transport", 0.0)),
                    "carrier_absorbed_weight": float(record.get("carrier_absorbed_weight", 0.0)),
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
    "TRAJECTORY_EVALUATION_CONTRACT_ID",
    "TRAJECTORY_EVALUATION_SCHEMA",
    "Lumo3DTrajectoryEvaluation",
    "Lumo3DTrajectoryEvaluator",
    "Lumo3DTrajectoryStudy",
    "create_lumo3d_trajectory_study",
]
