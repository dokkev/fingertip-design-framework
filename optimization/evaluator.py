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

from contact import (
    CandidateContactError,
    FirstContactSettings,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
    unintended_boundary_clearance_mm,
)
from physics import (
    IndentationSettings,
    NewtonSettings,
    PhysicsDependencyError,
    RigidIndenter3D,
    solve_fingertip_indentation_trajectory,
    prepare_fingertip_mesh,
    InvalidFingertipMesh,
)
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.rigid.object import make_sphere_mesh, RigidObjectMesh
from mesh import volume_mesh_settings_for_tier
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.mesh import VolumeMeshDependencyError, VolumeMeshingError
from mesh.fingertip.geometry import GmshDependencyError
from model import (
    Fingertip,
    FingertipParameters,
    InvalidFingertip,
    InvalidFingertipParameters,
    validate_minimum_silicone_thickness,
)
from optics.contracts.objects import CarrierOptics
from optics.transport3d import (
    OptiXTransport,
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DResultError,
    Transport3DTraceError,
    Transport3DSettings,
    fingerprint_mapping,
    save_case_artifact,
    transport_configuration,
)
from optics.transport3d.optix_backend import create_runtime
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
from optimization.deformed_state_artifact import (
    build_contact_state_record,
    restore_deformed_optical_state,
    write_mechanics_artifact,
)


TRAJECTORY_EVALUATION_SCHEMA = "lumo3d-trajectory-evaluation-v1"
TRAJECTORY_EVALUATION_CONTRACT_ID = TRAJECTORY_EVALUATION_SCHEMA
CURRENT_CELL_HALF_LENGTH_MM = 5.5
LUMO3D_OBSERVATION_LEVEL = "FULL_3D native internal transport redistribution proxy"
LUMO3D_OPTICAL_X_BOUNDS_MM = (-16.0, 16.0)
LUMO3D_OPTICAL_Y_BOUNDS_MM = (-31.0, 4.5)


def _optical_settings() -> Transport3DSettings:
    """Return the frozen optical contract for the production evaluator."""
    return Transport3DSettings(
        ray_count=256,
        max_interactions=6,
        maximum_segment_count=4096,
        maximum_periodic_wraps=8,
        surface_u_bins=32,
        surface_z_bins=16,
        internal_grid_width=32,
        internal_grid_height=32,
        internal_z_bins=8,
        x_bounds_mm=LUMO3D_OPTICAL_X_BOUNDS_MM,
        y_bounds_mm=LUMO3D_OPTICAL_Y_BOUNDS_MM,
        terminate_on_periodic_wrap_limit=True,
        terminate_on_no_event=True,
        retain_internal_path_field=True,
    )


def _load_steps(travel_mm: float, increment_mm: float) -> int:
    return max(1, int(np.ceil(float(travel_mm) / float(increment_mm))))


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
        self.settings = _optical_settings()

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

    def _trajectory_mechanics(
        self,
        tip: Fingertip,
        prepared: Any,
        contact_surface: Any,
        carrier_mesh: RigidObjectMesh,
        *,
        location_u: float,
        radius_mm: float,
        candidate_root: Path,
    ) -> tuple[Mapping[str, Any], ...]:
        sphere_mesh = make_sphere_mesh(
            radius_mm,
            subdivisions=self.mechanics_contract.sphere_subdivisions,
        )
        alignment = sphere_alignment_at_normalized_location(
            tip.geometry,
            sphere_mesh,
            location_u,
            initial_gap_mm=self.protocol.initial_gap_mm,
        )
        if intersects(contact_surface, sphere_mesh, alignment.nominal_pose):
            raise CandidateContactError(
                f"u={location_u:g} nominal pose is not collision-free"
            )
        first_contact = find_first_contact(
            contact_surface,
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
            FirstContactSettings(
                coarse_step_mm=0.25,
                tolerance_mm=1.0e-3,
                spawn_clearance_mm=0.05,
                max_travel_mm=20.0,
            ),
        )
        if intersects(contact_surface, sphere_mesh, first_contact.spawn_pose):
            raise CandidateContactError(
                f"u={location_u:g} spawn pose is not collision-free"
            )
        boundary_clearance = unintended_boundary_clearance_mm(
            tip.geometry,
            alignment,
            first_contact,
        )
        if boundary_clearance <= 0.0:
            raise CandidateContactError(
                f"u={location_u:g} reaches an unintended external boundary "
                f"before arc contact: clearance={boundary_clearance:g} mm"
            )
        travels = self.protocol.checkpoint_depths_mm
        # The mechanics API retains these generic checkpoint annotations for
        # scheduling/provenance. They are derived from the fixed absolute
        # depths; neither controls the physical loading path.
        checkpoint_fractions = self.protocol.checkpoint_fractions
        normalized_ratios = self.protocol.normalized_indentation_ratios(radius_mm)
        mechanics_settings = NewtonSettings(
            device=self.device,
            gravity=0.0,
            dt=self.mechanics_contract.dt_s,
            steps=1,
            iterations=self.mechanics_contract.vbd_iterations,
            fixed_vertex_indices=prepared.support_vertex_indices,
        )
        indentation_settings = IndentationSettings(
            travel_mm=travels[-1],
            load_steps=_load_steps(travels[-1], self.mechanics_contract.max_load_increment_mm),
            soft_contact_margin_mm=self.mechanics_contract.soft_contact_margin_mm,
            soft_contact_ke=self.mechanics_contract.soft_contact_ke,
            soft_contact_kd=self.mechanics_contract.soft_contact_kd,
        )
        indenter = RigidIndenter3D(
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
        )
        trajectory = solve_fingertip_indentation_trajectory(
            prepared,
            indenter,
            mechanics_settings,
            indentation_settings,
            travels,
            checkpoint_fractions=checkpoint_fractions,
            normalized_indentation_ratios=normalized_ratios,
            max_load_increment_mm=self.mechanics_contract.max_load_increment_mm,
            first_contact=first_contact,
            rigid_carrier_mesh=carrier_mesh,
        )
        records: list[Mapping[str, Any]] = []
        trajectory_id = f"u_{location_u:.3f}__radius_{radius_mm:.3f}"
        for checkpoint in trajectory.checkpoints:
            checkpoint_path = (
                candidate_root
                / "mechanics"
                / trajectory_id
                / (
                    f"checkpoint_{checkpoint.checkpoint_index:02d}"
                    f"_depth_{checkpoint.post_contact_travel_mm:.3f}mm.npz"
                )
            )
            artifact_sha = write_mechanics_artifact(checkpoint_path, checkpoint, prepared)
            expected_pose = first_contact.pose_at_post_contact_travel(checkpoint.post_contact_travel_mm)
            pose_error = float(
                np.linalg.norm(
                    np.asarray(checkpoint.indenter_pose.translation_mm)
                    - np.asarray(expected_pose.translation_mm)
                )
            )
            record = {
                "trajectory_id": trajectory_id,
                "normalized_location": location_u,
                "radius_mm": radius_mm,
                "checkpoint_index": checkpoint.checkpoint_index,
                "checkpoint_depth_mm": checkpoint.post_contact_travel_mm,
                "checkpoint_fraction": checkpoint.checkpoint_fraction,
                "normalized_indentation_ratio": checkpoint.normalized_indentation_ratio,
                "post_contact_travel_mm": checkpoint.post_contact_travel_mm,
                "unintended_boundary_clearance_mm": boundary_clearance,
                "cumulative_step_index": checkpoint.cumulative_step_index,
                "first_contact_travel_mm": first_contact.travel_to_contact_mm,
                "first_contact_fingerprint": fingerprint_mapping({
                    "target_point_mm": alignment.target_point_mm,
                    "outward_normal": alignment.outward_normal,
                    "approach_direction": alignment.approach_direction,
                    "radius_mm": radius_mm,
                    "contact_pose_mm": first_contact.contact_pose.translation_mm,
                }),
                "mechanics_artifact_path": str(checkpoint_path),
                "mechanics_artifact_sha256": artifact_sha,
                "final_pose_error_mm": pose_error,
                "mechanics_diagnostics": dict(checkpoint.diagnostics),
            }
            records.append(record)
        return tuple(records)

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

            volume_mesh = generate_volume_mesh(
                tip.solid(),
                volume_mesh_settings_for_tier("search"),
            )
            prepared = prepare_fingertip_mesh(volume_mesh)
            contact_surface = make_outer_compliant_surface(volume_mesh.solid)
            carrier_mesh = make_distal_phalanx_mesh(volume_mesh.solid)

            trajectory_records: list[Mapping[str, Any]] = []
            for radius in self.protocol.indenter_radii_mm:
                for location in self.protocol.contact_locations_u:
                    trajectory_records.extend(
                        self._trajectory_mechanics(
                            tip,
                            prepared,
                            contact_surface,
                            carrier_mesh,
                            location_u=location,
                            radius_mm=radius,
                            candidate_root=candidate_root,
                        )
                    )
            mechanics_finished = time.perf_counter()
            stage = "optics"
            configuration = transport_configuration(
                self.settings,
                material=tip.optical.to_dict(),
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
            runtime = create_runtime()
            transport = OptiXTransport()
            optical_records: list[Mapping[str, Any]] = []
            observations: list[TrajectoryObservation] = []
            optics_started = time.perf_counter()
            for record in trajectory_records:
                contact_state = build_contact_state_record(
                    morphology_fingerprint=volume_mesh.morphology_fingerprint,
                    protocol=self.protocol,
                    location_u=float(record["normalized_location"]),
                    radius_mm=float(record["radius_mm"]),
                    checkpoint_depth_mm=float(record["checkpoint_depth_mm"]),
                    checkpoint_fraction=float(record["checkpoint_fraction"]),
                    normalized_indentation_ratio=float(record["normalized_indentation_ratio"]),
                    post_contact_travel_mm=float(record["post_contact_travel_mm"]),
                    unintended_boundary_clearance_mm=float(record["unintended_boundary_clearance_mm"]),
                    checkpoint_diagnostics=record["mechanics_diagnostics"],
                    source_node_ids=prepared.source_node_ids,
                    mechanics_artifact_sha256=str(record["mechanics_artifact_sha256"]),
                )
                restored = restore_deformed_optical_state(
                    tip,
                    volume_mesh,
                    prepared,
                    record["mechanics_artifact_path"],
                    record["mechanics_artifact_sha256"],
                    carrier_optics=CarrierOptics("absorber"),
                    carrier_contact_source_node_ids=contact_state["carrier_contact_source_node_ids"],
                    metadata={
                        "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
                        "contact_location_u": record["normalized_location"],
                        "checkpoint_depth_mm": record["checkpoint_depth_mm"],
                        "checkpoint_fraction": record["checkpoint_fraction"],
                        "normalized_indentation_ratio": record["normalized_indentation_ratio"],
                        "post_contact_travel_mm": record["post_contact_travel_mm"],
                        "unintended_boundary_clearance_mm": record["unintended_boundary_clearance_mm"],
                        "observation_level": LUMO3D_OBSERVATION_LEVEL,
                        "carrier_optical_boundary_model": "absorber",
                        "carrier_mapping_tolerance_mm": contact_state["carrier_mapping_tolerance_mm"],
                    },
                )
                result = transport.trace(
                    tip,
                    restored.geometry,
                    settings=self.settings,
                    morphology_id=morphology_id,
                    morphology_fingerprint=volume_mesh.morphology_fingerprint,
                    mechanics_source=str(restored.artifact_path),
                    mechanics_dimension="3D",
                    contact_state=contact_state,
                    transport_configuration=configuration,
                    runtime=runtime,
                )
                artifact = (
                    candidate_root
                    / f"location_u_{float(record['normalized_location']):.3f}"
                    f"__radius_{float(record['radius_mm']):.3f}"
                    f"__depth_{float(record['checkpoint_depth_mm']):.3f}mm"
                    f"__checkpoint_{int(record['checkpoint_index']):02d}.json"
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
                    "mechanics_artifact_sha256": record["mechanics_artifact_sha256"],
                    "mechanics_dimension": "3D",
                    "geometry_mode": "full3d_surface",
                    "full3d_surface_provenance": "actual_deformed_3d_volume_state",
                    "contact_state": contact_state,
                    "transport_configuration": configuration,
                    "transport_configuration_fingerprint": result.transport_configuration_fingerprint,
                }
                save_case_artifact(artifact, result, contract)
                energy = result.energy_record()
                optical_record = dict(record)
                optical_record.update(energy)
                optical_record.update(
                    {
                        "artifact": str(artifact),
                        "artifact_field": str(artifact.with_suffix(".npz")),
                        "contact_state": contact_state,
                        "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
                        "carrier_contact_active": contact_state["carrier_contact_active"],
                        "carrier_contact_occurred": contact_state["carrier_contact_occurred"],
                        "transport_configuration_fingerprint": result.transport_configuration_fingerprint,
                        "evaluation_identity": evaluation_identity,
                    }
                )
                optical_records.append(optical_record)
                observations.append(
                    TrajectoryObservation(
                        location_u=float(record["normalized_location"]),
                        radius_mm=float(record["radius_mm"]),
                        checkpoint_depth_mm=float(record["checkpoint_depth_mm"]),
                        field=result.field,
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
                    "failure_stage": stage,
                    "failure_scenario": "candidate_contact",
                },
            )
        except (Transport3DGeometryError, Transport3DPhysicsError, Transport3DResultError, Transport3DTraceError) as exc:
            return _failure("optics_failure", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})


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
