"""Configurable continuous-trajectory FULL_3D LUMO evaluator.

This module is deliberately beside the fixed three-state compatibility
evaluator.  It owns orchestration and provenance only; mechanics and optical
physics remain in their existing neutral backends.
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
    FirstContactSettings,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
)
from mechanics3d import (
    IndentationSettings,
    Mechanics3DSettings,
    RigidIndenter3D,
    solve_fingertip_indentation_trajectory,
    prepare_fingertip_mechanics_mesh,
    InvalidFingertipMechanicsMesh,
)
from mesh.rigid_carrier import make_distal_phalanx_mesh
from mesh.rigid_object import make_sphere_mesh, RigidObjectMesh
from mesh import volume_mesh_settings_for_tier
from mesh.volume3d import VolumeMeshDependencyError, VolumeMeshingError
from model import (
    Fingertip,
    FingertipParameters,
    InvalidFingertip,
    InvalidFingertipParameters,
    validate_minimum_silicone_thickness,
)
from optics.contact_object import CarrierOptics
from optics.transport3d import (
    OptiXTransport,
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DResultError,
    Transport3DTraceError,
    fingerprint_mapping,
    save_case_artifact,
    transport_configuration,
)
from optics.transport3d.optix_backend import create_runtime
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
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
    compute_trajectory_objective,
    normalized_field_distance,
)
from optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)
from validation.mechanics3d.deformed_state_artifact import restore_deformed_optical_state
from validation.optimization.lumo3d_evaluator import (
    LUMO3D_OBSERVATION_LEVEL,
    LUMO3D_OPTICAL_X_BOUNDS_MM,
    LUMO3D_OPTICAL_Y_BOUNDS_MM,
    _candidate_id,
    _energy_record,
    _material,
    _optical_settings,
)


TRAJECTORY_EVALUATION_SCHEMA = "lumo3d-trajectory-evaluation-v1"
CURRENT_CELL_HALF_LENGTH_MM = 5.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices)[np.asarray(tetrahedra)]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


def _load_steps(travel_mm: float, increment_mm: float) -> int:
    return max(1, int(np.ceil(float(travel_mm) / float(increment_mm))))


def _state_identity(
    morphology_fingerprint: str,
    protocol: TrajectoryEvaluationProtocol,
    *,
    location_u: float,
    radius_mm: float,
    checkpoint_depth_mm: float,
    checkpoint_fraction: float,
    normalized_indentation_ratio: float,
    post_contact_travel_mm: float,
    mechanics_artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "morphology_fingerprint": morphology_fingerprint,
        "protocol_fingerprint": protocol.fingerprint,
        "contact_location_u": float(location_u),
        "indenter_radius_mm": float(radius_mm),
        "checkpoint_depth_mm": float(checkpoint_depth_mm),
        "checkpoint_fraction": float(checkpoint_fraction),
        "normalized_indentation_ratio": float(normalized_indentation_ratio),
        "post_contact_travel_mm": float(post_contact_travel_mm),
        "mechanics_artifact_sha256": mechanics_artifact_sha256,
        "mechanics_artifact_fingerprint": mechanics_artifact_sha256,
    }


def _contact_state(
    *,
    morphology_fingerprint: str,
    protocol: TrajectoryEvaluationProtocol,
    location_u: float,
    radius_mm: float,
    checkpoint_depth_mm: float,
    checkpoint_fraction: float,
    normalized_indentation_ratio: float,
    post_contact_travel_mm: float,
    checkpoint_diagnostics: Mapping[str, Any],
    source_node_ids: tuple[int, ...],
    mechanics_artifact_sha256: str,
) -> dict[str, Any]:
    local_indices = tuple(
        int(index)
        for index in checkpoint_diagnostics.get("active_carrier_contact_vertex_indices", ())
    )
    source_ids = tuple(
        int(source_node_ids[index])
        for index in local_indices
        if 0 <= index < len(source_node_ids)
    )
    identity = _state_identity(
        morphology_fingerprint,
        protocol,
        location_u=location_u,
        radius_mm=radius_mm,
        checkpoint_depth_mm=checkpoint_depth_mm,
        checkpoint_fraction=checkpoint_fraction,
        normalized_indentation_ratio=normalized_indentation_ratio,
        post_contact_travel_mm=post_contact_travel_mm,
        mechanics_artifact_sha256=mechanics_artifact_sha256,
    )
    return {
        "state_identity": identity,
        "contact_state_fingerprint": fingerprint_mapping(identity | {"carrier_contact_source_node_ids": source_ids}),
        "normalized_location": float(location_u),
        "indenter_radius_mm": float(radius_mm),
        "initial_gap_mm": protocol.initial_gap_mm,
        "checkpoint_depth_mm": float(checkpoint_depth_mm),
        "checkpoint_fraction": float(checkpoint_fraction),
        "normalized_indentation_ratio": float(normalized_indentation_ratio),
        "post_contact_travel_mm": float(post_contact_travel_mm),
        "first_contact_travel_mm": float(checkpoint_diagnostics.get("first_contact_travel_mm", 0.0)),
        "spawn_clearance_mm": float(checkpoint_diagnostics.get("spawn_clearance_mm", 0.0)),
        "carrier_contact_active": bool(checkpoint_diagnostics.get("carrier_contact_active", False)),
        "carrier_mechanical_contact_count": int(checkpoint_diagnostics.get("carrier_interface_contact_count", 0)),
        "carrier_mechanical_contact_vertex_count": len(source_ids),
        "first_carrier_contact_step": checkpoint_diagnostics.get("first_carrier_contact_step"),
        "carrier_contact_source_node_ids": list(source_ids),
        "carrier_mapping_tolerance_mm": 0.5 * float(checkpoint_diagnostics.get("rigid_sdf_target_voxel_mm", 0.125)),
        "mechanics_artifact_sha256": mechanics_artifact_sha256,
    }


def _write_mechanics_artifact(
    path: Path,
    checkpoint: Any,
    prepared: Any,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rest_vertices_mm": np.asarray(checkpoint.mechanics_result.rest_vertices, dtype=np.float32),
        "deformed_vertices_mm": np.asarray(checkpoint.mechanics_result.deformed_vertices, dtype=np.float32),
        "tetrahedra": np.asarray(checkpoint.mechanics_result.tetrahedra, dtype=np.int32),
        "source_node_ids": np.asarray(prepared.source_node_ids, dtype=np.int64),
        "carrier_contact_vertex_indices": np.asarray(
            checkpoint.diagnostics.get("active_carrier_contact_vertex_indices", ()),
            dtype=np.int64,
        ),
    }
    arrays["carrier_contact_source_node_ids"] = np.asarray(
        [prepared.source_node_ids[index] for index in arrays["carrier_contact_vertex_indices"]],
        dtype=np.int64,
    )
    arrays.update(
        {
            f"surface_{tag}": np.asarray(triangles, dtype=np.int32)
            for tag, triangles in prepared.surface_triangles.items()
        }
    )
    np.savez_compressed(path, **arrays)
    return _sha256(path)


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
    objective: Mapping[str, Any]
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
        volume_mesh: Any,
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
            raise RuntimeError(f"u={location_u:g} nominal pose is not collision-free")
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
            raise RuntimeError(f"u={location_u:g} spawn pose is not collision-free")
        travels = self.protocol.checkpoint_depths_mm
        # The mechanics API retains these generic checkpoint annotations for
        # scheduling/provenance. They are derived from the fixed absolute
        # depths; neither controls the physical loading path.
        checkpoint_fractions = self.protocol.checkpoint_fractions
        normalized_ratios = self.protocol.normalized_indentation_ratios(radius_mm)
        mechanics_settings = Mechanics3DSettings(
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
            artifact_sha = _write_mechanics_artifact(checkpoint_path, checkpoint, prepared)
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

    def evaluate(self, parameters: FingertipParameters) -> Lumo3DTrajectoryEvaluation:
        started = time.perf_counter()
        stage = "mechanics"
        try:
            validate_minimum_silicone_thickness(parameters)
            tip = Fingertip(parameters)
            candidate_id = _candidate_id(parameters)
            candidate_root = (
                self.artifact_root
                / f"protocol_{self.protocol.fingerprint}"
                / "candidates"
                / candidate_id
            )
            for radius in self.protocol.indenter_radii_mm:
                reason = self._domain_failure(radius)
                if reason is not None:
                    return _failure(
                        "domain_incompatible",
                        reason,
                        diagnostics={"radius_mm": radius, "failure_stage": "domain_validation"},
                    )

            volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
            prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
            contact_surface = make_outer_compliant_surface(volume_mesh.solid)
            carrier_mesh = make_distal_phalanx_mesh(volume_mesh.solid)

            trajectory_records: list[Mapping[str, Any]] = []
            for radius in self.protocol.indenter_radii_mm:
                for location in self.protocol.contact_locations_u:
                    trajectory_records.extend(
                        self._trajectory_mechanics(
                            tip,
                            volume_mesh,
                            prepared,
                            contact_surface,
                            carrier_mesh,
                            location_u=location,
                            radius_mm=radius,
                            candidate_root=candidate_root,
                        )
                    )
            stage = "optics"
            configuration = transport_configuration(
                self.settings,
                material=_material(tip),
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
                contact_state = _contact_state(
                    morphology_fingerprint=volume_mesh.morphology_fingerprint,
                    protocol=self.protocol,
                    location_u=float(record["normalized_location"]),
                    radius_mm=float(record["radius_mm"]),
                    checkpoint_depth_mm=float(record["checkpoint_depth_mm"]),
                    checkpoint_fraction=float(record["checkpoint_fraction"]),
                    normalized_indentation_ratio=float(record["normalized_indentation_ratio"]),
                    post_contact_travel_mm=float(record["post_contact_travel_mm"]),
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
                        "observation_level": LUMO3D_OBSERVATION_LEVEL,
                        "carrier_optical_boundary_model": "absorber",
                        "carrier_mapping_tolerance_mm": contact_state["carrier_mapping_tolerance_mm"],
                    },
                )
                result = transport.trace(
                    tip,
                    restored.geometry,
                    settings=self.settings,
                    morphology_id=candidate_id,
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
                    "morphology_id": candidate_id,
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
                energy = _energy_record(result)
                optical_record = dict(record)
                optical_record.update(energy)
                optical_record.update(
                    {
                        "artifact": str(artifact),
                        "artifact_field": str(artifact.with_suffix(".npz")),
                        "contact_state": contact_state,
                        "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
                        "carrier_contact_active": contact_state["carrier_contact_active"],
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
            trajectory_metrics = _trajectory_metrics(tuple(observations), optical_records)
            summary = {
                "schema": TRAJECTORY_EVALUATION_SCHEMA,
                "objective_name": OBJECTIVE_NAME,
                "objective_value": objective["objective_value"],
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
                "mechanics_runtime_s": time.perf_counter() - started,
                "optics_runtime_s": time.perf_counter() - optics_started,
                "total_runtime_s": time.perf_counter() - started,
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "transport_configuration_fingerprint": fingerprint_mapping(configuration),
                "trajectory_records": trajectory_records,
                "optical_records": optical_records,
                "objective": objective,
                "trajectory_metrics": trajectory_metrics,
            }
            candidate_root.mkdir(parents=True, exist_ok=True)
            (candidate_root / "trajectory_evaluation.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            return Lumo3DTrajectoryEvaluation(
                status="success",
                objective_value=float(objective["objective_value"]),
                objective=objective,
                trajectory_diagnostics=tuple(trajectory_records),
                checkpoint_diagnostics=tuple(trajectory_records),
                optical_diagnostics=tuple(optical_records),
                diagnostics=summary,
            )
        except Transport3DDependencyError:
            raise
        except (InvalidFingertip, InvalidFingertipParameters) as exc:
            return _failure("invalid_design", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})
        except (VolumeMeshDependencyError, VolumeMeshingError, InvalidFingertipMechanicsMesh) as exc:
            return _failure("mesh_failure", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})
        except (Transport3DGeometryError, Transport3DPhysicsError, Transport3DResultError, Transport3DTraceError) as exc:
            return _failure("optics_failure", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})
        except Exception as exc:
            return _failure(
                "mechanics_failure" if stage == "mechanics" else "optics_failure",
                f"{type(exc).__name__}: {exc}",
                diagnostics={"failure_stage": stage},
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
            DesignVariable(name, True, lower, upper)
            for name, lower, upper in PRODUCTION_SEARCH_BOUNDS
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


__all__ = [
    "CURRENT_CELL_HALF_LENGTH_MM",
    "Lumo3DTrajectoryEvaluation",
    "Lumo3DTrajectoryEvaluator",
    "Lumo3DTrajectoryStudy",
    "create_lumo3d_trajectory_study",
]
