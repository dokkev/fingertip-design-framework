"""Bounded six-dimensional production BO integration run.

This runner is intentionally a validation workflow, not a second optimizer.
It reuses the production Ax adapter and LUMO FULL_3D evaluator, while keeping
the trial schema and plots specific to the bounded Test BO contract.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mesh.volume.mesh import VolumeMeshDependencyError
from model import FingertipParameters, silicone_thickness_measures
from optics.transport3d import Transport3DDependencyError, fingerprint_mapping
from optics.optix.smoke import run_production_optix_smoke
from optimization.adapters.ax import (
    AxSettings,
    CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
    CampaignInfrastructureError,
    GMSH_RUNTIME_FAILURE_SIGNATURE,
    OPTIX_RUNTIME_FAILURE_SIGNATURE,
    PHYSICS_RUNTIME_FAILURE_SIGNATURE,
    PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS,
    create_ax_client,
    run_ax_optimization,
)
from optimization.design_space import (
    OPTIMIZABLE_PARAMETER_NAMES,
    PRODUCTION_SEARCH_BOUNDS,
)
from physics import PhysicsDependencyError
from optimization.evaluation_registry import EvaluationRegistry, REGISTRY_SCHEMA_VERSION
from optimization.evaluator import (
    LUMO3D_OBSERVATION_LEVEL,
    LUMO3D_OPTICAL_X_BOUNDS_MM,
    LUMO3D_OPTICAL_Y_BOUNDS_MM,
)
from optimization.evaluator import (
    Lumo3DTrajectoryStudy,
    TRAJECTORY_EVALUATION_CONTRACT_ID,
    TRAJECTORY_EVALUATION_SCHEMA,
    create_lumo3d_trajectory_study,
)


SEED = 20260819
SOBOL_TRIALS = 6
BO_TRIALS = 4
MAX_ATTEMPTED_TRIALS = SOBOL_TRIALS + BO_TRIALS
OUTPUT_DIRECTORY = Path("output/validation/optimization/lumo6d_test_bo")
TRAJECTORY_EVALUATION_CONTRACT = {
    "schema": TRAJECTORY_EVALUATION_SCHEMA,
    "objective": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
    "observation_level": LUMO3D_OBSERVATION_LEVEL,
    "optical_mode": "FULL_3D",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parameter_values(parameters: Mapping[str, object]) -> dict[str, float]:
    return {
        name: float(parameters[name])
        for name in OPTIMIZABLE_PARAMETER_NAMES
    }


def _optical_grid() -> dict[str, Any]:
    grid = {
        "x_bounds_mm": list(LUMO3D_OPTICAL_X_BOUNDS_MM),
        "y_bounds_mm": list(LUMO3D_OPTICAL_Y_BOUNDS_MM),
        "mode": "FULL_3D",
    }
    return {
        **grid,
        "fingerprint": fingerprint_mapping(grid),
    }


def _search_mechanics(study: Lumo3DTrajectoryStudy) -> dict[str, object]:
    """Return the serializable mechanics contract for campaign metadata."""
    return study.create_evaluator().mechanics_contract.to_dict()


def _rms_displacement_from_artifact(path: object) -> float | None:
    if not path:
        return None
    try:
        with np.load(Path(str(path)), allow_pickle=False) as archive:
            rest = np.asarray(archive["rest_vertices_mm"], dtype=float)
            deformed = np.asarray(archive["deformed_vertices_mm"], dtype=float)
        displacement = deformed - rest
        return float(np.sqrt(np.mean(np.square(displacement))))
    except (KeyError, OSError, ValueError):
        return None


def _aggregate_mechanics(evaluation: object) -> dict[str, Any]:
    records = tuple(getattr(evaluation, "checkpoint_diagnostics", ()))
    displacements = [
        float(item["max_displacement_mm"])
        for item in records
        if item.get("max_displacement_mm") is not None
    ]
    rms_values = [
        _rms_displacement_from_artifact(item.get("mechanics_artifact_path"))
        for item in records
    ]
    rms_values = [value for value in rms_values if value is not None]
    return {
        "max_displacement_mm": max(displacements) if displacements else None,
        "rms_displacement_mm": max(rms_values) if rms_values else None,
        "inverted_tetrahedra": max(
            (int(item.get("inverted_tetrahedra", 0)) for item in records),
            default=0,
        ),
        "contact_overflow": max(
            (
                int(item.get("max_soft_contact_overflow", 0))
                + int(item.get("max_rigid_contact_overflow", 0))
                for item in records
            ),
            default=0,
        ),
        "first_contact": [
            {
                "normalized_location": item.get("normalized_location"),
                "first_contact_travel_mm": item.get("first_contact_travel_mm"),
                "bracket_width_mm": item.get("first_contact_bracket_width_mm"),
                "spawn_clearance_mm": item.get("spawn_clearance_mm"),
                "contact_pose_mm": item.get("contact_pose_mm"),
                "spawn_pose_mm": item.get("spawn_pose_mm"),
            }
            for item in records
        ],
    }


def _aggregate_optics(evaluation: object) -> dict[str, Any]:
    records = tuple(getattr(evaluation, "optical_diagnostics", ()))
    escaped = [float(item["escaped_weight"]) for item in records]
    carrier_absorbed = [
        item.get("carrier_absorbed_weight") for item in records
    ]
    carrier_absorbed_values = [
        float(value) for value in carrier_absorbed if value is not None
    ]
    total_transport = [
        float(item["total_transport"])
        for item in records
        if item.get("total_transport") is not None
    ]
    silicone_absorbed: list[float] = []
    for item in records:
        if item.get("absorbed_weight") is None or item.get("carrier_absorbed_weight") is None:
            continue
        silicone_absorbed.append(
            float(item["absorbed_weight"]) - float(item["carrier_absorbed_weight"])
        )
    return {
        "carrier_contact_active": any(
            bool(item.get("carrier_optical_contact_triangle_count", 0))
            for item in records
        ),
        "carrier_contact_triangle_count": max(
            (
                int(item.get("carrier_optical_contact_triangle_count", 0))
                for item in records
            ),
            default=0,
        ),
        "escaped_transport": sum(escaped) if escaped else None,
        "silicone_absorbed_weight": sum(silicone_absorbed) if silicone_absorbed else None,
        "carrier_absorbed_weight": sum(carrier_absorbed_values)
        if carrier_absorbed_values
        else None,
        "energy_balance_error": max(
            (abs(float(item.get("energy_balance_error", 0.0))) for item in records),
            default=0.0,
        ),
        "total_surviving_transport": sum(total_transport) if total_transport else None,
        "states": [dict(item) for item in records],
    }


def _status_contract(status: str) -> str:
    return {
        "success": "valid_success",
        "invalid_design": "geometry_rejected",
        "domain_incompatible": "domain_incompatible",
        "mechanics_failure": "mechanics_failed",
        "optics_failure": "optics_failed",
        "mesh_failure": "geometry_rejected",
        "duplicate_skipped": "duplicate_skipped",
    }.get(status, "infrastructure_failed")


def _trial_payload(
    record: Any,
    study: Lumo3DTrajectoryStudy,
    attempt_index: int | None,
) -> dict[str, Any]:
    parameters = _parameter_values(record.parameters)
    candidate = None
    measures = None
    try:
        candidate = study.design_space.decode(parameters)
        measures = silicone_thickness_measures(candidate)
    except Exception:
        pass
    evaluation = record.evaluation
    status = _status_contract(record.status)
    mechanics = (
        _aggregate_mechanics(evaluation)
        if evaluation is not None
        else {
            "max_displacement_mm": None,
            "rms_displacement_mm": None,
            "inverted_tetrahedra": None,
            "contact_overflow": None,
            "first_contact": [],
        }
    )
    optics = (
        _aggregate_optics(evaluation)
        if evaluation is not None
        else {
            "carrier_contact_active": None,
            "carrier_contact_triangle_count": None,
            "escaped_transport": None,
            "silicone_absorbed_weight": None,
            "carrier_absorbed_weight": None,
            "energy_balance_error": None,
            "total_surviving_transport": None,
            "states": [],
        }
    )
    evaluation_diagnostics = (
        {} if evaluation is None else dict(evaluation.diagnostics)
    )
    return {
        "trial_index": attempt_index,
        "ax_trial_index": (
            None if record.trial_index is None else int(record.trial_index)
        ),
        "generation_method": (
            "sobol" if record.phase == "initialization"
            else "bo" if record.phase == "search"
            else "nominal"
        ),
        "ax_generation_node": (
            "Sobol" if record.phase == "initialization"
            else "MBM" if record.phase == "search"
            else None
        ),
        "phase": record.phase,
        "status": status,
        "raw_status": record.status,
        **parameters,
        "total_pad_depth_mm": (
            None
            if candidate is None
            else float(candidate.flat_pad_height + candidate.semielliptical_pad_height)
        ),
        "minimum_silicone_thickness_mm": (
            None if measures is None else float(measures.minimum_silicone_thickness_mm)
        ),
        "shortest_boundary_pair": (
            None if measures is None else measures.shortest_boundary_pair
        ),
        "geometry_valid": status == "valid_success" or (
            status not in {
                "geometry_rejected",
                "domain_incompatible",
                "infrastructure_failed",
            }
            and candidate is not None
        ),
        "geometry_failure_reason": (
            None
            if status not in {
                "geometry_rejected",
                "domain_incompatible",
                "invalid_design",
            }
            else record.failure_message
        ),
        "objective": (
            None if evaluation is None else getattr(evaluation, "objective_value", None)
        ),
        "pairwise_contact_state_distances": (
            None
            if evaluation is None
            else getattr(
                getattr(evaluation, "objective", None),
                "all_pairwise_distances",
                None,
            )
        ),
        **mechanics,
        **optics,
        "first_contact_fingerprint": [
            item.get("contact_state_fingerprint")
            for item in (
                () if evaluation is None else getattr(evaluation, "optical_diagnostics", ())
            )
            if isinstance(item, Mapping)
        ],
        "optical_grid_fingerprint": _optical_grid()["fingerprint"],
        "transport_configuration_fingerprint": evaluation_diagnostics.get(
            "transport_configuration_fingerprint"
        ),
        "mechanics_runtime_s": evaluation_diagnostics.get("mechanics_runtime_s"),
        "optics_runtime_s": evaluation_diagnostics.get("optics_runtime_s"),
        "total_runtime_s": record.wall_time_seconds,
        "failure_message": record.failure_message,
        "registry_key": record.registry_key,
        "artifact_paths": [
            item.get("artifact")
            for item in (
                () if evaluation is None else getattr(evaluation, "optical_diagnostics", ())
            )
            if isinstance(item, Mapping)
        ],
        "evaluation_diagnostics": evaluation_diagnostics,
    }


def _pre_run_sanity(study: Lumo3DTrajectoryStudy) -> dict[str, Any]:
    if len(study.design_space.active_variables) != 6:
        raise RuntimeError("SIX_D_PARAMETERIZATION_BLOCKER")
    if tuple(variable.name for variable in study.design_space.active_variables) != tuple(
        OPTIMIZABLE_PARAMETER_NAMES
    ):
        raise RuntimeError("active variable names do not match the six-variable contract")

    client = create_ax_client(
        study,
        AxSettings(
            initialization_trials=SOBOL_TRIALS,
            search_trials=BO_TRIALS,
            seed=SEED,
            objective_name=CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        ),
    )
    suggestions = client.get_next_trials(max_trials=SOBOL_TRIALS)
    values = [dict(parameters) for parameters in suggestions.values()]
    for trial_index in suggestions:
        client.mark_trial_abandoned(trial_index=trial_index)
    total_depths = [
        float(item["flat_pad_height"] + item["semielliptical_pad_height"])
        for item in values
    ]
    flat_values = {item["flat_pad_height"] for item in values}
    ellipse_values = {item["semielliptical_pad_height"] for item in values}
    if len(flat_values) < 2 or len(ellipse_values) < 2:
        raise RuntimeError("h_fp/h_ep did not vary in the pre-run Ax suggestions")
    if len(set(total_depths)) < 2:
        raise RuntimeError("total pad depth remained coupled in the pre-run suggestions")
    return {
        "active_variable_count": len(study.design_space.active_variables),
        "active_variables": list(OPTIMIZABLE_PARAMETER_NAMES),
        "numerical_envelopes": [spec.to_dict() for spec in PRODUCTION_SEARCH_BOUNDS],
        "linear_constraints": list(PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS),
        "suggestions": values,
        "flat_pad_height_unique_count": len(flat_values),
        "semielliptical_pad_height_unique_count": len(ellipse_values),
        "total_pad_depth_unique_count": len(set(total_depths)),
        "independent_heights_confirmed": True,
    }


def _persist_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    records = list(records)
    fields = [
        "trial_index", "ax_trial_index", "generation_method", "ax_generation_node", "status",
        *OPTIMIZABLE_PARAMETER_NAMES, "total_pad_depth_mm", "minimum_silicone_thickness_mm",
        "geometry_valid", "geometry_failure_reason", "objective",
        "max_displacement_mm", "rms_displacement_mm", "inverted_tetrahedra", "contact_overflow",
        "carrier_contact_active", "carrier_contact_triangle_count", "escaped_transport",
        "silicone_absorbed_weight", "carrier_absorbed_weight", "energy_balance_error",
        "first_contact_fingerprint", "optical_grid_fingerprint", "transport_configuration_fingerprint",
        "mechanics_runtime_s", "optics_runtime_s", "total_runtime_s", "failure_message", "registry_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["first_contact_fingerprint"] = json.dumps(row.get("first_contact_fingerprint"))
            writer.writerow(row)


def _successful(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [record for record in records if record.get("status") == "valid_success"]


def _finite_objective(record: Mapping[str, Any]) -> float | None:
    value = record.get("objective")
    return None if value is None else float(value)


def _scatter(
    records: list[Mapping[str, Any]],
    nominal: Mapping[str, Any],
    x_field: str,
    x_label: str,
    path: Path,
    *,
    y_field: str = "objective",
    y_label: str = "objective",
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    sobol = [record for record in records if record.get("generation_method") == "sobol"]
    bo = [record for record in records if record.get("generation_method") == "bo"]
    rejected = [record for record in records if record.get("objective") is None]
    for values, color, label in ((sobol, "#4C78A8", "Sobol"), (bo, "#E15759", "BO")):
        valid = [record for record in values if record.get(y_field) is not None]
        if valid:
            ax.scatter(
                [float(record[x_field]) for record in valid],
                [float(record[y_field]) for record in valid],
                color=color,
                label=label,
                s=34,
            )
    if rejected:
        valid_rejected = [record for record in rejected if record.get(x_field) is not None]
        if valid_rejected:
            ax.scatter(
                [float(record[x_field]) for record in valid_rejected],
                [0.0 for _ in valid_rejected],
                marker="x",
                color="#777777",
                label="geometry rejected",
            )
    if nominal.get(x_field) is not None and nominal.get(y_field) is not None:
        ax.scatter(
            [float(nominal[x_field])], [float(nominal[y_field])],
            color="#222222", marker="*", s=110, label="nominal",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(records: list[Mapping[str, Any]], plots: Path) -> None:
    values = [float(record["objective"]) if record.get("objective") is not None else np.nan for record in records]
    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, values, marker="o", color="#4C78A8")
    ax.set_xlabel("attempted Ax trial")
    ax.set_ylabel("minimum pairwise normalized FULL_3D separation")
    ax.set_title("6D Test BO objective history")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(plots / "objective_history.png", dpi=160); plt.close(fig)

    running: list[float] = []
    best = -float("inf")
    for value in values:
        if np.isfinite(value):
            best = max(best, float(value))
        running.append(best if np.isfinite(best) else np.nan)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, running, marker="o", color="#222222")
    ax.set_xlabel("attempted Ax trial")
    ax.set_ylabel("running best objective")
    ax.set_title("Running best (bounded Test BO)")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(plots / "running_best.png", dpi=160); plt.close(fig)


def _plot_parameter_history(records: list[Mapping[str, Any]], plots: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.0), sharex=True)
    x = np.arange(1, len(records) + 1)
    for axis, name in zip(axes.flat, OPTIMIZABLE_PARAMETER_NAMES, strict=True):
        for phase, color, label in (("sobol", "#4C78A8", "Sobol"), ("bo", "#E15759", "BO")):
            selected = [i for i, record in enumerate(records) if record.get("generation_method") == phase]
            if selected:
                axis.scatter(x[selected], [records[i][name] for i in selected], color=color, s=24, label=label)
        axis.set_ylabel(name.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("attempted Ax trial")
    axes[-1, 1].set_xlabel("attempted Ax trial")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("6D morphology parameter history")
    fig.tight_layout()
    fig.savefig(plots / "parameter_history.png", dpi=160)
    plt.close(fig)


def _diagnostics(
    records: list[Mapping[str, Any]],
    nominal: Mapping[str, Any],
) -> dict[str, Any]:
    successful = _successful(records)
    sobol = [record for record in successful if record.get("generation_method") == "sobol"]
    bo = [record for record in successful if record.get("generation_method") == "bo"]
    objective_values = [float(record["objective"]) for record in successful]
    def correlation(left: str, right: str) -> float | None:
        pairs = [(float(item[left]), float(item[right])) for item in successful if item.get(left) is not None and item.get(right) is not None]
        if len(pairs) < 2:
            return None
        x, y = np.asarray(pairs, dtype=float).T
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    best = max(successful, key=lambda item: float(item["objective"])) if successful else None
    best_sobol = max(sobol, key=lambda item: float(item["objective"])) if sobol else None
    best_bo = max(bo, key=lambda item: float(item["objective"])) if bo else None
    same_depth_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(successful):
        for right in successful[left_index + 1:]:
            if abs(float(left["total_pad_depth_mm"]) - float(right["total_pad_depth_mm"])) <= 0.5:
                if abs(float(left["flat_pad_height"]) - float(right["flat_pad_height"])) > 0.5:
                    same_depth_pairs.append({
                        "left_objective": left["objective"], "right_objective": right["objective"],
                        "objective_difference": abs(float(left["objective"]) - float(right["objective"])),
                        "total_depth_difference_mm": abs(float(left["total_pad_depth_mm"]) - float(right["total_pad_depth_mm"])),
                    })
    carrier_values = [item["carrier_absorbed_weight"] for item in successful if item.get("carrier_absorbed_weight") is not None]
    transport_values = [item["total_surviving_transport"] for item in successful if item.get("total_surviving_transport") is not None]
    pathology = False
    if best is not None and carrier_values and transport_values:
        median_transport = float(np.median(transport_values))
        best_transport = best.get("total_surviving_transport")
        best_carrier = best.get("carrier_absorbed_weight")
        pathology = bool(
            best_transport is not None
            and median_transport > 0.0
            and float(best_transport) < 0.1 * median_transport
            and best_carrier is not None
            and float(best_carrier) > float(np.median(carrier_values))
        )
    return {
        "successful_count": len(successful),
        "geometry_valid_count": sum(
            item.get("status") not in {
                "geometry_rejected",
                "domain_incompatible",
                "infrastructure_failed",
            }
            for item in records
        ),
        "geometry_rejected_count": sum(item.get("status") == "geometry_rejected" for item in records),
        "domain_incompatible_count": sum(
            item.get("status") == "domain_incompatible" for item in records
        ),
        "mechanics_failure_count": sum(item.get("status") == "mechanics_failed" for item in records),
        "optics_failure_count": sum(item.get("status") == "optics_failed" for item in records),
        "sobol_successful_count": len(sobol),
        "bo_successful_count": len(bo),
        "best_sobol_objective": None if best_sobol is None else best_sobol["objective"],
        "best_bo_objective": None if best_bo is None else best_bo["objective"],
        "best_overall_objective": None if best is None else best["objective"],
        "best_parameters": None if best is None else {name: best[name] for name in OPTIMIZABLE_PARAMETER_NAMES},
        "nominal_objective": nominal.get("objective"),
        "improvement_over_nominal": (
            None if best is None or nominal.get("objective") is None
            else float(best["objective"]) - float(nominal["objective"])
        ),
        "best_d_min_mm": None if best is None else best.get("minimum_silicone_thickness_mm"),
        "best_total_pad_depth_mm": None if best is None else best.get("total_pad_depth_mm"),
        "best_near_d_min_5mm": bool(best is not None and best.get("minimum_silicone_thickness_mm") is not None and float(best["minimum_silicone_thickness_mm"]) <= 5.5),
        "best_on_numerical_envelope": bool(
            best is not None
            and any(
                abs(float(best[spec.name.value]) - bound) <= 1.0e-8
                for spec in PRODUCTION_SEARCH_BOUNDS
                for bound in (spec.lower, spec.upper)
            )
        ),
        "objective_range": None if not objective_values else [min(objective_values), max(objective_values)],
        "objective_vs_d_min_correlation": correlation("minimum_silicone_thickness_mm", "objective"),
        "objective_vs_void_width_correlation": correlation("void_width", "objective"),
        "objective_vs_total_depth_correlation": correlation("total_pad_depth_mm", "objective"),
        "h_fp_unique_count": len({item["flat_pad_height"] for item in records}),
        "h_ep_unique_count": len({item["semielliptical_pad_height"] for item in records}),
        "same_total_depth_split_pairs": same_depth_pairs,
        "bo_beat_sobol": bool(best_bo is not None and (best_sobol is None or float(best_bo["objective"]) > float(best_sobol["objective"]))),
        "carrier_absorption_vs_objective_correlation": correlation("carrier_absorbed_weight", "objective"),
        "best_candidate_carrier_contact_state": None if best is None else {
            "active": best.get("carrier_contact_active"),
            "triangle_count": best.get("carrier_contact_triangle_count"),
        },
        "best_candidate_carrier_absorption": None if best is None else best.get("carrier_absorbed_weight"),
        "objective_extinction_pathology": pathology,
        "diagnostic_limits": "n=10 bounded Test BO diagnostics only; no strong scientific trend is inferred",
    }


def run_lumo6d_test_bo(output_dir: str | Path = OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Run the nominal + six-Sobol + four-MBM bounded integration test."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Test BO directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    study = create_lumo3d_trajectory_study(output / "artifacts", mechanics_mode="search")
    search_mechanics = _search_mechanics(study)
    sanity = _pre_run_sanity(study)
    grid = _optical_grid()
    configuration = {
        "schema": "lumo6d-test-bo-v1",
        "status": "INITIALIZING",
        "created_at": _now(),
        "seed": SEED,
        "ax": {
            "initialization_trials": SOBOL_TRIALS,
            "search_trials": BO_TRIALS,
            "max_attempted_proposals": MAX_ATTEMPTED_TRIALS,
        },
        "active_variables": list(OPTIMIZABLE_PARAMETER_NAMES),
        "numerical_envelopes": [spec.to_dict() for spec in PRODUCTION_SEARCH_BOUNDS],
        "linear_constraints": list(PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS),
        "contract_id": TRAJECTORY_EVALUATION_CONTRACT_ID,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "evaluation_contract": TRAJECTORY_EVALUATION_CONTRACT,
        "observation_level": LUMO3D_OBSERVATION_LEVEL,
        "objective_name": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        "objective_direction": "maximize",
        "common_optical_grid": grid,
        "search_mechanics": search_mechanics,
        "pre_run_sanity": sanity,
    }
    _write_json(output / "config.json", configuration)
    state: dict[str, Any] = {"schema": configuration["schema"], "status": "INITIALIZING", "records": [], "created_at": _now()}
    _write_json(output / "checkpoint.json", state)

    try:
        preflight = run_production_optix_smoke()
        state["optix_preflight"] = {"status": "PASS", "evidence": preflight.to_dict()}
        _write_json(output / "preflight.json", state["optix_preflight"])
        _write_json(output / "checkpoint.json", state)

        # Evaluate and persist nominal before the first get_next_trials call.
        nominal_evaluator = study.create_evaluator()
        nominal_parameters = _parameter_values({
            name: getattr(study.design_space.nominal_parameters, name)
            for name in OPTIMIZABLE_PARAMETER_NAMES
        })
        nominal_started = time.perf_counter()
        try:
            nominal_evaluation = nominal_evaluator.evaluate(
                study.design_space.nominal_parameters
            )
        except Transport3DDependencyError as exc:
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                signature=OPTIX_RUNTIME_FAILURE_SIGNATURE,
            ) from exc
        except VolumeMeshDependencyError as exc:
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                signature=GMSH_RUNTIME_FAILURE_SIGNATURE,
            ) from exc
        except PhysicsDependencyError as exc:
            raise CampaignInfrastructureError(
                f"{type(exc).__name__}: {exc}",
                signature=PHYSICS_RUNTIME_FAILURE_SIGNATURE,
            ) from exc
        nominal_record = type("NominalRecord", (), {
            "parameters": nominal_parameters,
            "phase": "nominal",
            "trial_index": None,
            "evaluation": nominal_evaluation,
            "status": nominal_evaluation.status,
            "failure_message": nominal_evaluation.failure_message,
            "wall_time_seconds": time.perf_counter() - nominal_started,
            "registry_key": None,
        })()
        nominal_payload = _trial_payload(nominal_record, study, None)
        if nominal_evaluation.status != "success":
            raise RuntimeError(
                "NOMINAL_FEASIBILITY_BLOCKER: "
                f"{nominal_evaluation.status}: {nominal_evaluation.failure_message}"
            )
        _write_json(output / "nominal.json", nominal_payload)

        # Seed the fresh registry with the separately evaluated nominal so the
        # Ax adapter can bootstrap its observation without reevaluating it.
        registry = EvaluationRegistry(output / "registry.json")
        nominal_registry = registry.register(
            TRAJECTORY_EVALUATION_CONTRACT_ID,
            nominal_parameters,
            status="success",
            first_trial_index=0,
            first_campaign_id=output.name,
            result_artifact_path=str((output / "nominal.json").resolve()),
            minimum_auc=None,
            objective_value=float(nominal_evaluation.objective_value),
            failure_category=None,
            failure_message=None,
            failure_scenario=None,
            evaluation_wall_time_seconds=nominal_record.wall_time_seconds,
        )
        nominal_payload["registry_key"] = nominal_registry.key
        _write_json(output / "nominal.json", nominal_payload)

        records_by_trial: dict[int, dict[str, Any]] = {}

        def persist(client: Any, records: tuple[Any, ...]) -> None:
            attempt = 0
            for record in records:
                if record.phase == "nominal":
                    continue
                payload = _trial_payload(record, study, attempt)
                records_by_trial[int(record.trial_index)] = payload
                attempt += 1
            ordered = [records_by_trial[index] for index in sorted(records_by_trial)]
            state["records"] = ordered
            state["updated_at"] = _now()
            _write_json(output / "trials.json", ordered)
            _persist_csv(output / "trials.csv", ordered)
            _write_json(output / "ax_client.json", client._to_json_snapshot())
            _write_json(output / "checkpoint.json", state)

        settings = AxSettings(
            initialization_trials=SOBOL_TRIALS,
            search_trials=BO_TRIALS,
            seed=SEED,
            objective_name=CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        )
        result = run_ax_optimization(
            study,
            settings,
            on_record=persist,
            evaluation_registry=registry,
            evaluation_contract_id=TRAJECTORY_EVALUATION_CONTRACT_ID,
            campaign_id=output.name,
            result_artifact_path=str((output / "checkpoint.json").resolve()),
            max_consecutive_known_proposals=20,
            max_proposals=MAX_ATTEMPTED_TRIALS,
        )
        ordered = [records_by_trial[index] for index in sorted(records_by_trial)]
        _write_json(output / "trials.json", ordered)
        _persist_csv(output / "trials.csv", ordered)
        diagnostics = _diagnostics(ordered, nominal_payload)
        status = "PASS"
        if (
            diagnostics["objective_extinction_pathology"]
            or result.status != "COMPLETE"
            or diagnostics["bo_successful_count"] < 1
        ):
            status = "FAIL"
        elif (
            diagnostics["mechanics_failure_count"] > 0
            or diagnostics["optics_failure_count"] > 0
            or diagnostics["successful_count"] < 8
        ):
            status = "PASS_WITH_LIMITATION"
        state.update({
            "status": status,
            "ax_status": result.status,
            "ax_proposal_count": result.ax_proposal_count,
            "successful_count": result.unique_success_count,
            "completed_at": _now(),
            "total_wall_time_seconds": time.perf_counter() - started,
        })
        configuration["status"] = status
        configuration["completed_at"] = state["completed_at"]
        _write_json(output / "config.json", configuration)
        _write_json(output / "checkpoint.json", state)
        _plot_history(ordered, plots)
        _plot_parameter_history(ordered, plots)
        nominal_for_plot = nominal_payload
        _scatter(ordered, nominal_for_plot, "minimum_silicone_thickness_mm", "minimum silicone thickness d_min [mm]", plots / "objective_vs_dmin.png")
        _scatter(ordered, nominal_for_plot, "void_width", "void width [mm]", plots / "objective_vs_void_width.png")
        _scatter(ordered, nominal_for_plot, "d_min_placeholder" if False else "void_width", "void width [mm]", plots / "void_width_vs_dmin.png", y_field="minimum_silicone_thickness_mm", y_label="minimum silicone thickness d_min [mm]")
        _scatter(ordered, nominal_for_plot, "flat_pad_height", "flat pad height h_fp [mm]", plots / "objective_vs_flat_pad_height.png")
        _scatter(ordered, nominal_for_plot, "semielliptical_pad_height", "semi-elliptical height h_ep [mm]", plots / "objective_vs_semielliptical_pad_height.png")
        _scatter(ordered, nominal_for_plot, "total_pad_depth_mm", "total pad depth [mm]", plots / "objective_vs_total_pad_depth.png")
        _scatter(ordered, nominal_for_plot, "carrier_absorbed_weight", "carrier absorbed weight", plots / "carrier_absorption_vs_objective.png", y_label="objective")
        summary = {
            "status": status,
            "ax_status": result.status,
            "nominal": nominal_payload,
            "attempted_trials": len(ordered),
            "diagnostics": diagnostics,
            "objective_name": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
            "objective_direction": "maximize",
            "contract_id": TRAJECTORY_EVALUATION_CONTRACT_ID,
            "artifact_directory": str(output),
            "total_wall_clock_runtime_s": time.perf_counter() - started,
            "plots": sorted(str(path.relative_to(output)) for path in plots.glob("*.png")),
        }
        _write_json(output / "summary.json", summary)
        return summary
    except CampaignInfrastructureError as exc:
        state.update({
            "status": "FAIL",
            "failure_category": "infrastructure_failed",
            "infrastructure_signature": exc.signature,
            "error": f"{type(exc).__name__}: {exc}",
            "total_wall_time_seconds": time.perf_counter() - started,
        })
        _write_json(output / "checkpoint.json", state)
        raise
    except Exception as exc:
        state.update({
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "total_wall_time_seconds": time.perf_counter() - started,
        })
        _write_json(output / "checkpoint.json", state)
        raise


__all__ = ["run_lumo6d_test_bo", "OUTPUT_DIRECTORY"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    result = run_lumo6d_test_bo(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
