"""Representative-morphology Newton, mesh, and optical convergence harness.

This module implements the expensive validation workflow but never runs it at
import time. Mesh and optical comparisons intentionally remain INCONCLUSIVE in
the absence of an approved scientific delta threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from lumo.config import LumoExecutionConfig, load_lumo_execution_config
from lumo.finger import Fingertip
from lumo.mesh import generate_volume_mesh, volume_mesh_settings_for_tier
from lumo.mesh.rigid.carrier import make_distal_phalanx_mesh
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    PRODUCTION_LINEAR_CONSTRAINTS,
)
from lumo.optimization.evaluator import Lumo3DTrajectoryEvaluation, Lumo3DTrajectoryEvaluator
from lumo.optimization.objectives import (
    TrajectoryObservation,
    compute_trajectory_objective,
)
from lumo.optimization.optical_artifact import energy_record, save_case_artifact
from lumo.optimization.optical_contract import (
    DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE,
    fingerprint_mapping,
    optical_physics_parameters,
    transport_configuration,
)
from lumo.physics import prepare_fingertip_mesh
from lumo.ray_tracing.contracts.objects import CarrierOptics
from lumo.ray_tracing.optical_mechanics import (
    Transport3DCandidateGeometryError,
    trace_geometry,
)
from lumo.ray_tracing.optical_mechanics.optix_backend import create_runtime
from validation.common.io import atomic_write_json
from validation.optimization.representative_morphologies import (
    RepresentativeMorphology,
    representative_morphologies,
)
from validation.physics.sweep_newton_sphere_parameters import (
    RELATIVE_MAX_DISPLACEMENT_THRESHOLD as NEWTON_RELATIVE_MAX_THRESHOLD,
    RMS_VERTEX_THRESHOLD_MM as NEWTON_RMS_THRESHOLD_MM,
    comparison_metrics,
)
from validation.ray_tracing.deformed_state_restore import restore_deformed_optical_state
from scripts.optimization.run_bo import (
    USER_LED,
    USER_OBJECTIVE,
    USER_PARAMETERS,
    USER_PROTOCOL,
    USER_SEARCH_BOUNDS,
    _enforce_source_policy,
    _source_provenance,
)


SCHEMA = "lumo3d-scientific-convergence-v1"
DEFAULT_EXECUTION_CONFIG = Path(__file__).resolve().parents[2] / "config" / "lumo_execution.yaml"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "optimization"
    / "lumo3d_scientific_convergence"
)


def _validate_production_baseline(execution: LumoExecutionConfig) -> None:
    expected_mesh = volume_mesh_settings_for_tier("search")
    if execution.volume_mesh != expected_mesh:
        raise ValueError(
            "Phase-E baseline requires the fixed 1.5 mm search mesh settings"
        )
    mechanics = execution.mechanics
    if mechanics.vbd_iterations != 10 or mechanics.max_load_increment_mm != 0.05:
        raise ValueError(
            "Phase-E baseline requires 10 Newton iterations and 0.05 mm increment"
        )
    transport = execution.transport
    if (
        transport.ray_count != 256
        or transport.max_interactions != 6
        or transport.maximum_segment_count != 4096
    ):
        raise ValueError(
            "Phase-E optical baseline requires 256 rays, 6 interactions, and "
            "4096 segments"
        )


def _production_design_space() -> DesignSpace:
    return DesignSpace(
        USER_PARAMETERS,
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in USER_SEARCH_BOUNDS
        ),
        linear_constraints=PRODUCTION_LINEAR_CONSTRAINTS,
        fixed_led=USER_LED,
    )


def optical_sweep_settings(execution: LumoExecutionConfig) -> tuple[dict[str, Any], ...]:
    """Return baseline plus one-family-at-a-time optical refinement settings."""

    _validate_production_baseline(execution)
    base = execution.transport
    return (
        {
            "setting_id": "production",
            "family": "baseline",
            "role": "production",
            "settings": base,
        },
        {
            "setting_id": "rays_512",
            "family": "ray_count",
            "role": "intermediate",
            "settings": replace(base, ray_count=512),
        },
        {
            "setting_id": "rays_1024",
            "family": "ray_count",
            "role": "reference",
            "settings": replace(base, ray_count=1024),
        },
        {
            "setting_id": "interactions_8",
            "family": "max_interactions",
            "role": "intermediate",
            "settings": replace(base, max_interactions=8),
        },
        {
            "setting_id": "interactions_10",
            "family": "max_interactions",
            "role": "reference",
            "settings": replace(base, max_interactions=10),
        },
        {
            "setting_id": "segments_8192",
            "family": "maximum_segment_count",
            "role": "intermediate",
            "settings": replace(base, maximum_segment_count=8192),
        },
        {
            "setting_id": "segments_16384",
            "family": "maximum_segment_count",
            "role": "reference",
            "settings": replace(base, maximum_segment_count=16384),
        },
        {
            "setting_id": "grid_64x64x16",
            "family": "path_field_grid",
            "role": "intermediate",
            "settings": replace(
                base,
                internal_grid_width=64,
                internal_grid_height=64,
                internal_z_bins=16,
            ),
        },
        {
            "setting_id": "grid_128x128x32",
            "family": "path_field_grid",
            "role": "reference",
            "settings": replace(
                base,
                internal_grid_width=128,
                internal_grid_height=128,
                internal_z_bins=32,
            ),
        },
    )


def convergence_plan(execution: LumoExecutionConfig) -> dict[str, Any]:
    """Return the frozen one-factor validation matrix without running science."""

    _validate_production_baseline(execution)
    reference_mechanics = replace(
        execution.mechanics,
        vbd_iterations=20,
        max_load_increment_mm=min(0.025, execution.mechanics.max_load_increment_mm),
    )
    optical_specs = optical_sweep_settings(execution)
    return {
        "schema": SCHEMA,
        "newton": {
            "production": execution.mechanics.to_dict(),
            "reference": reference_mechanics.to_dict(),
            "evidence_collection": {
                "complete_trajectory_after_optical_failure": True,
                "production_acceptance_unchanged": True,
            },
            "acceptance": {
                "rms_vertex_difference_mm_max": NEWTON_RMS_THRESHOLD_MM,
                "relative_max_displacement_difference_max": NEWTON_RELATIVE_MAX_THRESHOLD,
            },
        },
        "mesh": {
            "search": asdict(volume_mesh_settings_for_tier("search")),
            "reference": asdict(volume_mesh_settings_for_tier("reference")),
            "force_metric": {
                "status": "unsupported",
                "value_n": None,
                "reason": "prescribed-indentation production artifacts expose no reviewed reaction-force contract",
            },
            "scientific_threshold": None,
        },
        "optics": {
            "settings": [
                {
                    "setting_id": item["setting_id"],
                    "family": item["family"],
                    "role": item["role"],
                    "settings": asdict(item["settings"]),
                }
                for item in optical_specs
            ],
            "family_reference_pairs": {
                family: {
                    "production_setting_id": "production",
                    "reference_setting_id": next(
                        item["setting_id"]
                        for item in optical_specs
                        if item["family"] == family and item["role"] == "reference"
                    ),
                }
                for family in (
                    "ray_count",
                    "max_interactions",
                    "maximum_segment_count",
                    "path_field_grid",
                )
            },
        },
    }


def _objective_metrics(evaluation: Lumo3DTrajectoryEvaluation) -> dict[str, Any]:
    objective = evaluation.objective
    diagnostic = evaluation.report.get("objective")
    diagnostic_raw = (
        {
            "accepted": False,
            "source": "failure_diagnostics",
            "objective": diagnostic.get("objective_value"),
            "D_inter": diagnostic.get("D_inter"),
            "D_radius": diagnostic.get("D_radius"),
            "objective_pathology": diagnostic.get("objective_pathology"),
        }
        if evaluation.status == "optics_failure" and isinstance(diagnostic, Mapping)
        else None
    )
    return {
        "objective": evaluation.objective_value,
        "D_inter": None if objective is None else objective.d_inter,
        "D_radius": None if objective is None else objective.d_radius,
        "diagnostic_raw_objective": diagnostic_raw,
    }


def _baseline_numerical_acceptance(
    evaluation: Lumo3DTrajectoryEvaluation,
) -> str:
    """Separate transport/objective acceptance from upstream execution status."""

    if evaluation.status == "success":
        return "PASS"
    if evaluation.status == "optics_failure" and evaluation.failure_scenario in {
        "numerical_acceptance",
        "objective_pathology",
    }:
        return "FAIL"
    return "NOT_RUN"


def _optical_sensitivity(
    production: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Report optical deltas without inventing a scientific threshold."""

    metrics: dict[str, Any] = {}
    for name in ("objective", "D_inter", "D_radius"):
        left = production.get(name)
        if left is None:
            raw = production.get("diagnostic_raw_objective")
            if isinstance(raw, Mapping):
                left = raw.get(name)
        right = comparison.get(name)
        if right is None:
            raw = comparison.get("diagnostic_raw_objective")
            if isinstance(raw, Mapping):
                right = raw.get(name)
        metrics[name] = (
            None
            if left is None or right is None
            else {
                "production": float(left),
                "comparison": float(right),
                "signed_delta": float(right) - float(left),
                "absolute_delta": abs(float(right) - float(left)),
                "relative_delta": abs(float(right) - float(left))
                / max(abs(float(left)), 1.0e-30),
            }
        )
    return {
        "production_setting_id": "production",
        "comparison_setting_id": comparison.get("setting_id"),
        "family": comparison.get("family"),
        "comparison_role": comparison.get("role"),
        "metrics": metrics,
        "scientific_threshold": None,
        "scientific_convergence": "INCONCLUSIVE",
    }


def _mechanics_checkpoint_evidence(
    evaluation: Lumo3DTrajectoryEvaluation,
) -> dict[str, Any]:
    """Classify mechanics completeness independently of downstream optics."""

    expected = {
        (
            f"u_{location:.3f}__radius_{radius:.3f}",
            checkpoint_index,
        )
        for location, radius in USER_PROTOCOL.trajectories
        for checkpoint_index in range(USER_PROTOCOL.checkpoint_count)
    }
    actual_rows = [
        (str(record.trajectory_id), int(record.checkpoint_index))
        for record in evaluation.checkpoint_records
    ]
    actual = set(actual_rows)
    duplicates = len(actual_rows) - len(actual)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    complete = not missing and not unexpected and duplicates == 0
    return {
        "status": "completed" if complete else "incomplete",
        "complete": complete,
        "expected_checkpoint_count": len(expected),
        "actual_checkpoint_count": len(actual_rows),
        "duplicate_checkpoint_count": duplicates,
        "missing_checkpoint_identities": [list(item) for item in missing],
        "unexpected_checkpoint_identities": [list(item) for item in unexpected],
        "downstream_evaluation_status": evaluation.status,
    }


def _evaluation_record(evaluation: Lumo3DTrajectoryEvaluation) -> dict[str, Any]:
    return {
        "execution_status": (
            "completed" if evaluation.status == "success" else "candidate_failure"
        ),
        "evaluation_status": evaluation.status,
        **_objective_metrics(evaluation),
        "failure_message": evaluation.failure_message,
        "failure_scenario": evaluation.failure_scenario,
        "result_artifact_path": evaluation.result_artifact_path,
        "optical_numerical_summary": evaluation.report.get(
            "optical_numerical_summary"
        ),
        "checkpoint_count": len(evaluation.checkpoint_records),
        "mechanics_checkpoint_evidence": _mechanics_checkpoint_evidence(evaluation),
        "states": [
            {
                "trajectory_id": record.trajectory_id,
                "checkpoint_index": record.checkpoint_index,
                "location_u": record.normalized_location,
                "radius_mm": record.radius_mm,
                "checkpoint_depth_mm": record.checkpoint_depth_mm,
                **energy_record(record.optics),
            }
            for record in evaluation.checkpoint_records
        ],
    }


def _load_mechanics_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "rest_vertices_mm",
            "deformed_vertices_mm",
            "tetrahedra",
            "source_node_ids",
        )
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"mechanics artifact is missing arrays: {missing}")
        return {name: np.array(archive[name], copy=True) for name in required}


def compare_newton_evaluations(
    production: Lumo3DTrajectoryEvaluation,
    reference: Lumo3DTrajectoryEvaluation,
) -> dict[str, Any]:
    """Compare exact-topology Newton states and objective sensitivity separately."""

    result: dict[str, Any] = {
        "production": _evaluation_record(production),
        "reference": _evaluation_record(reference),
        "objective_sensitivity": {
            name: (
                None
                if left is None or right is None
                else {
                    "signed_delta": float(left - right),
                    "absolute_delta": abs(float(left - right)),
                    "relative_delta": abs(float(left - right))
                    / max(abs(float(right)), 1.0e-30),
                }
            )
            for name, left, right in (
                ("objective", production.objective_value, reference.objective_value),
                (
                    "D_inter",
                    None
                    if production.objective is None
                    else production.objective.d_inter,
                    None
                    if reference.objective is None
                    else reference.objective.d_inter,
                ),
                (
                    "D_radius",
                    None
                    if production.objective is None
                    else production.objective.d_radius,
                    None
                    if reference.objective is None
                    else reference.objective.d_radius,
                ),
            )
        },
        "state_comparisons": [],
    }
    production_mechanics = _mechanics_checkpoint_evidence(production)
    reference_mechanics = _mechanics_checkpoint_evidence(reference)
    result["mechanics_execution"] = {
        "production": production_mechanics,
        "reference": reference_mechanics,
    }
    if not production_mechanics["complete"] or not reference_mechanics["complete"]:
        mechanics_failed = "mechanics_failure" in {
            production.status,
            reference.status,
        }
        result["execution_status"] = (
            "mechanics_failure" if mechanics_failed else "mechanics_checkpoint_incomplete"
        )
        result["scientific_convergence"] = (
            "FAIL" if mechanics_failed else "INCONCLUSIVE"
        )
        return result
    result["execution_status"] = "completed"
    production_records = {
        (item.trajectory_id, item.checkpoint_index): item
        for item in production.checkpoint_records
    }
    reference_records = {
        (item.trajectory_id, item.checkpoint_index): item
        for item in reference.checkpoint_records
    }
    if set(production_records) != set(reference_records):
        raise ValueError("Newton comparison checkpoint identities do not match")
    passed = True
    comparisons: list[dict[str, Any]] = []
    for key in sorted(production_records):
        left_record = production_records[key]
        right_record = reference_records[key]
        left = _load_mechanics_arrays(left_record.mechanics_artifact_path)
        right = _load_mechanics_arrays(right_record.mechanics_artifact_path)
        for name in ("rest_vertices_mm", "tetrahedra", "source_node_ids"):
            if not np.array_equal(left[name], right[name]):
                raise ValueError(
                    f"Newton comparison requires exact topology/source identity: {name}"
                )
        metrics = comparison_metrics(
            left["deformed_vertices_mm"],
            right["deformed_vertices_mm"],
            left["deformed_vertices_mm"] - left["rest_vertices_mm"],
            right["deformed_vertices_mm"] - right["rest_vertices_mm"],
        )
        accepted = bool(
            metrics["rms_vertex_difference_mm"] <= NEWTON_RMS_THRESHOLD_MM
            and metrics["relative_max_displacement_difference"]
            <= NEWTON_RELATIVE_MAX_THRESHOLD
        )
        passed = passed and accepted
        comparisons.append(
            {
                "trajectory_id": key[0],
                "checkpoint_index": key[1],
                **metrics,
                "accepted": accepted,
            }
        )
    result["state_comparisons"] = comparisons
    result["scientific_convergence"] = "PASS" if passed else "FAIL"
    return result


def _displacement_scalars(
    evaluation: Lumo3DTrajectoryEvaluation,
) -> dict[tuple[str, int], dict[str, float]]:
    result: dict[tuple[str, int], dict[str, float]] = {}
    for record in evaluation.checkpoint_records:
        arrays = _load_mechanics_arrays(record.mechanics_artifact_path)
        displacement = arrays["deformed_vertices_mm"] - arrays["rest_vertices_mm"]
        magnitudes = np.linalg.norm(displacement, axis=1)
        result[(record.trajectory_id, record.checkpoint_index)] = {
            "maximum_displacement_mm": float(np.max(magnitudes)),
            "rms_displacement_mm": float(np.sqrt(np.mean(np.square(magnitudes)))),
        }
    return result


def compare_mesh_evaluations(
    search: Lumo3DTrajectoryEvaluation,
    reference: Lumo3DTrajectoryEvaluation,
) -> dict[str, Any]:
    """Report cross-mesh scalar sensitivity without inventing node correspondence."""

    result: dict[str, Any] = {
        "search": _evaluation_record(search),
        "reference": _evaluation_record(reference),
        "mesh_statistics": {
            "search": search.report.get("volume_mesh"),
            "reference": reference.report.get("volume_mesh"),
        },
        "force_metric": {
            "status": "unsupported",
            "value_n": None,
            "reason": (
                "prescribed-indentation mechanics artifacts expose no reviewed "
                "reaction-force contract"
            ),
        },
        "objective_sensitivity": {},
        "mechanics_scalar_sensitivity": [],
    }
    search_mechanics = _mechanics_checkpoint_evidence(search)
    reference_mechanics = _mechanics_checkpoint_evidence(reference)
    result["mechanics_execution"] = {
        "search": search_mechanics,
        "reference": reference_mechanics,
    }
    failure_statuses = {search.status, reference.status}
    if "mesh_failure" in failure_statuses:
        result["execution_status"] = "mesh_failure"
        result["scientific_convergence"] = "FAIL"
        return result
    if not search_mechanics["complete"] or not reference_mechanics["complete"]:
        mechanics_failed = "mechanics_failure" in failure_statuses
        result["execution_status"] = (
            "mechanics_failure" if mechanics_failed else "mechanics_checkpoint_incomplete"
        )
        result["scientific_convergence"] = (
            "FAIL" if mechanics_failed else "INCONCLUSIVE"
        )
        return result
    result["execution_status"] = "completed"
    for name, left, right in (
        ("objective", search.objective_value, reference.objective_value),
        (
            "D_inter",
            None if search.objective is None else search.objective.d_inter,
            None if reference.objective is None else reference.objective.d_inter,
        ),
        (
            "D_radius",
            None if search.objective is None else search.objective.d_radius,
            None if reference.objective is None else reference.objective.d_radius,
        ),
    ):
        result["objective_sensitivity"][name] = (
            None
            if left is None or right is None
            else {
                "signed_delta": float(left - right),
                "absolute_delta": abs(float(left - right)),
                "relative_delta": abs(float(left - right))
                / max(abs(float(right)), 1.0e-30),
            }
        )
    left_states = _displacement_scalars(search)
    right_states = _displacement_scalars(reference)
    if set(left_states) != set(right_states):
        raise ValueError("mesh comparison checkpoint identities do not match")
    for key in sorted(left_states):
        row: dict[str, Any] = {"trajectory_id": key[0], "checkpoint_index": key[1]}
        for metric in ("maximum_displacement_mm", "rms_displacement_mm"):
            left = left_states[key][metric]
            right = right_states[key][metric]
            row[metric] = {
                "search": left,
                "reference": right,
                "signed_delta": left - right,
                "absolute_delta": abs(left - right),
                "relative_delta": abs(left - right) / max(abs(right), 1.0e-30),
            }
        result["mechanics_scalar_sensitivity"].append(row)
    result["scientific_convergence"] = "INCONCLUSIVE"
    result["scientific_threshold"] = None
    return result


def _replay_optics(
    root: Path,
    case: RepresentativeMorphology,
    production: Lumo3DTrajectoryEvaluation,
    execution: LumoExecutionConfig,
    setting_id: str,
    settings: Any,
) -> dict[str, Any]:
    mechanics_evidence = _mechanics_checkpoint_evidence(production)
    if not mechanics_evidence["complete"]:
        return {
            "execution_status": "mechanics_baseline_incomplete",
            "baseline_evaluation_status": production.status,
            "mechanics_checkpoint_evidence": mechanics_evidence,
            "numerical_acceptance": "NOT_RUN",
            "scientific_convergence": "INCONCLUSIVE",
            "failure_message": "production mechanics checkpoint set is incomplete",
        }
    tip = Fingertip(case.parameters, led=USER_LED)
    volume_mesh = generate_volume_mesh(
        tip.solid(extrusion_depth_mm=execution.transport.extrusion_depth_mm),
        execution.volume_mesh,
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    carrier_mesh = make_distal_phalanx_mesh(volume_mesh.solid)
    runtime = create_runtime(execution.device)
    observations: list[TrajectoryObservation] = []
    states: list[dict[str, Any]] = []
    accepted = True
    configuration = transport_configuration(
        settings,
        material=optical_physics_parameters(tip),
    )
    configuration_fingerprint = fingerprint_mapping(configuration)
    for record in production.checkpoint_records:
        restored = restore_deformed_optical_state(
            tip,
            volume_mesh,
            prepared,
            record.mechanics_artifact_path,
            record.mechanics_artifact_sha256,
            carrier_mesh=carrier_mesh,
            carrier_optics=CarrierOptics("absorber"),
            carrier_mapping_tolerance_mm=(
                0.5 * execution.mechanics.rigid_sdf_target_voxel_mm
            ),
            source_epsilon_mm=execution.transport.source_epsilon_mm,
        )
        try:
            traced = trace_geometry(
                tip,
                restored.geometry,
                settings=settings,
                runtime=runtime,
            )
        except Transport3DCandidateGeometryError as exc:
            return {
                "execution_status": "candidate_failure",
                "baseline_evaluation_status": production.status,
                "mechanics_checkpoint_evidence": mechanics_evidence,
                "numerical_acceptance": "NOT_RUN",
                "scientific_convergence": "FAIL",
                "failure_scenario": "candidate_optics_geometry",
                "cause_type": type(exc).__name__,
                "failure_message": str(exc),
                "failed_trajectory_id": record.trajectory_id,
                "failed_checkpoint_index": record.checkpoint_index,
                "states": states,
            }
        decision = DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE.assess(traced)
        accepted = accepted and decision.accepted
        artifact = root / f"{record.trajectory_id}__checkpoint_{record.checkpoint_index:02d}.json"
        save_case_artifact(
            artifact,
            traced,
            {
                "schema": SCHEMA,
                "morphology_id": case.case_id,
                "morphology_fingerprint": case.morphology_fingerprint,
                "mechanics_source": str(record.mechanics_artifact_path),
                "mechanics_artifact_sha256": record.mechanics_artifact_sha256,
                "mechanics_dimension": "3D",
                "contact_state": record.contact_state.to_dict(),
                "transport_configuration": configuration,
                "transport_configuration_fingerprint": configuration_fingerprint,
                "setting_id": setting_id,
            },
        )
        diagnostics = energy_record(traced) | {
            "optical_numerical_acceptance": decision.to_dict()
        }
        states.append(
            {
                "trajectory_id": record.trajectory_id,
                "checkpoint_index": record.checkpoint_index,
                "location_u": record.normalized_location,
                "radius_mm": record.radius_mm,
                "checkpoint_depth_mm": record.checkpoint_depth_mm,
                "artifact": str(artifact.resolve()),
                **diagnostics,
            }
        )
        observations.append(
            TrajectoryObservation(
                location_u=record.normalized_location,
                radius_mm=record.radius_mm,
                checkpoint_depth_mm=record.checkpoint_depth_mm,
                field=traced.field,
                total_transport=traced.total_transport,
                escaped_weight=traced.escaped_weight,
                debug_diagnostics=diagnostics,
            )
        )
    objective = compute_trajectory_objective(observations, USER_OBJECTIVE)
    if objective.objective_pathology:
        accepted = False
    return {
        "execution_status": "completed",
        "baseline_evaluation_status": production.status,
        "mechanics_checkpoint_evidence": mechanics_evidence,
        "numerical_acceptance": "PASS" if accepted else "FAIL",
        "scientific_convergence": "INCONCLUSIVE" if accepted else "FAIL",
        "scientific_threshold": None,
        "objective": objective.objective_value,
        "D_inter": objective.d_inter,
        "D_radius": objective.d_radius,
        "objective_pathology": objective.objective_pathology,
        "states": states,
    }


def _execute_scientific_convergence(
    output: str | Path,
    *,
    execution_config: str | Path = DEFAULT_EXECUTION_CONFIG,
) -> dict[str, Any]:
    """Execute and persist the full representative convergence workflow."""

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    execution = load_lumo_execution_config(execution_config)
    design_space = _production_design_space()
    cases = representative_morphologies(design_space)
    plan = convergence_plan(execution)
    source = _source_provenance()
    _enforce_source_policy(source, smoke=True, allow_dirty=True)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": SCHEMA,
        "execution_config": execution.to_dict(),
        "source": source,
        "plan": plan,
        "representative_morphologies": [case.to_dict() for case in cases],
    }
    atomic_write_json(root / "config.json", config)
    started = time.perf_counter()
    newton_results: list[dict[str, Any]] = []
    mesh_results: list[dict[str, Any]] = []
    optical_results: list[dict[str, Any]] = []
    optical_reference_sensitivities: list[dict[str, Any]] = []
    reference_mechanics = replace(
        execution.mechanics,
        vbd_iterations=20,
        max_load_increment_mm=min(0.025, execution.mechanics.max_load_increment_mm),
    )
    for case in cases:
        production = Lumo3DTrajectoryEvaluator(
            root / "newton" / case.case_id / "production",
            protocol=USER_PROTOCOL,
            objective_config=USER_OBJECTIVE,
            mechanics_contract=execution.mechanics,
            device=execution.device,
            optical_settings=execution.transport,
            led=USER_LED,
            fixed_parameters=USER_PARAMETERS,
            volume_mesh_settings=execution.volume_mesh,
            complete_trajectory_after_optical_failure=True,
        ).evaluate(case.parameters)
        reference = Lumo3DTrajectoryEvaluator(
            root / "newton" / case.case_id / "reference",
            protocol=USER_PROTOCOL,
            objective_config=USER_OBJECTIVE,
            mechanics_contract=reference_mechanics,
            device=execution.device,
            optical_settings=execution.transport,
            led=USER_LED,
            fixed_parameters=USER_PARAMETERS,
            volume_mesh_settings=execution.volume_mesh,
            complete_trajectory_after_optical_failure=True,
        ).evaluate(case.parameters)
        newton_case = {"case": case.to_dict(), **compare_newton_evaluations(production, reference)}
        newton_results.append(newton_case)
        atomic_write_json(root / "newton" / "summary.json", {"schema": SCHEMA, "cases": newton_results})

        mesh_reference = Lumo3DTrajectoryEvaluator(
            root / "mesh" / case.case_id / "reference",
            protocol=USER_PROTOCOL,
            objective_config=USER_OBJECTIVE,
            mechanics_contract=execution.mechanics,
            device=execution.device,
            optical_settings=execution.transport,
            led=USER_LED,
            fixed_parameters=USER_PARAMETERS,
            volume_mesh_settings=volume_mesh_settings_for_tier("reference"),
            complete_trajectory_after_optical_failure=True,
        ).evaluate(case.parameters)
        mesh_case = {"case": case.to_dict(), **compare_mesh_evaluations(production, mesh_reference)}
        mesh_results.append(mesh_case)
        atomic_write_json(root / "mesh" / "summary.json", {"schema": SCHEMA, "cases": mesh_results})

        baseline = _evaluation_record(production)
        baseline_acceptance = _baseline_numerical_acceptance(production)
        case_optics: list[dict[str, Any]] = [
            {
                "case": case.to_dict(),
                "setting_id": "production",
                "family": "baseline",
                "role": "production",
                **baseline,
                "numerical_acceptance": baseline_acceptance,
                "scientific_convergence": (
                    "FAIL" if baseline_acceptance == "FAIL" else "INCONCLUSIVE"
                ),
                "scientific_threshold": None,
                "sensitivity_to_production": None,
            }
        ]
        for spec in optical_sweep_settings(execution)[1:]:
            replay = _replay_optics(
                root / "optics" / case.case_id / spec["family"] / spec["setting_id"],
                case,
                production,
                execution,
                spec["setting_id"],
                spec["settings"],
            )
            row = {
                "case": case.to_dict(),
                "setting_id": spec["setting_id"],
                "family": spec["family"],
                "role": spec["role"],
                "settings": asdict(spec["settings"]),
                **replay,
            }
            row["sensitivity_to_production"] = _optical_sensitivity(
                case_optics[0],
                row,
            )
            case_optics.append(row)
            if spec["role"] == "reference":
                optical_reference_sensitivities.append(
                    {
                        "case_id": case.case_id,
                        **row["sensitivity_to_production"],
                    }
                )
        optical_results.extend(case_optics)
        atomic_write_json(
            root / "optics" / "summary.json",
            {
                "schema": SCHEMA,
                "runs": optical_results,
                "reference_sensitivities": optical_reference_sensitivities,
            },
        )

    overall = {
        "schema": SCHEMA,
        "status": (
            "FAIL"
            if any(item.get("scientific_convergence") == "FAIL" for item in newton_results)
            or any(item.get("scientific_convergence") == "FAIL" for item in mesh_results)
            or any(item.get("scientific_convergence") == "FAIL" for item in optical_results)
            else "INCONCLUSIVE"
        ),
        "newton": newton_results,
        "mesh": mesh_results,
        "optics": optical_results,
        "optical_reference_sensitivities": optical_reference_sensitivities,
        "elapsed_seconds": time.perf_counter() - started,
        "source": source,
    }
    atomic_write_json(root / "summary.json", overall)
    return overall


def run_scientific_convergence(
    output: str | Path,
    *,
    execution_config: str | Path = DEFAULT_EXECUTION_CONFIG,
) -> dict[str, Any]:
    """Execute the workflow and preserve an abort summary after config creation."""

    root = Path(output)
    try:
        return _execute_scientific_convergence(
            root,
            execution_config=execution_config,
        )
    except Exception as exc:
        if (root / "config.json").is_file():
            atomic_write_json(
                root / "summary.json",
                {
                    "schema": SCHEMA,
                    "status": "ERROR",
                    "execution_status": "infrastructure_or_invariant_failure",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
            )
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the expensive convergence workflow only after explicit invocation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=DEFAULT_EXECUTION_CONFIG,
    )
    args = parser.parse_args(argv)
    try:
        summary = run_scientific_convergence(
            args.output,
            execution_config=args.execution_config,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"SCIENTIFIC_CONVERGENCE_ABORTED: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary.get("status") == "PASS" else 3


__all__ = [
    "DEFAULT_EXECUTION_CONFIG",
    "DEFAULT_OUTPUT",
    "NEWTON_RELATIVE_MAX_THRESHOLD",
    "NEWTON_RMS_THRESHOLD_MM",
    "SCHEMA",
    "compare_mesh_evaluations",
    "compare_newton_evaluations",
    "convergence_plan",
    "optical_sweep_settings",
    "run_scientific_convergence",
]


if __name__ == "__main__":
    raise SystemExit(main())
