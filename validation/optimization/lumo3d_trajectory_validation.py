"""Deterministic no-BO validation for the LUMO trajectory evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from finger import FingertipParameters
from optimization.objectives import normalized_field_distance
from optimization.protocol import DEFAULT_TRAJECTORY_PROTOCOL, TrajectoryEvaluationProtocol
from validation.physics.multi_location_sphere_contact import run_multi_location_sphere_contact
from optimization.evaluator import Lumo3DTrajectoryEvaluator
from validation.reference.lumo3d_fixed_state_oracle import FixedStateLumo3DOracle


OUTPUT_NAME = "lumo3d_fixed_depth_trajectory_evaluator_v1"


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return signed six-volumes for the local mechanics regression check."""

    points = np.asarray(vertices)[np.asarray(tetrahedra)]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


def _nominal() -> FingertipParameters:
    return FingertipParameters(
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        stem_width=7.6,
        stem_height=6.0,
        void_width=1.0,
        void_height=0.25,
    )


def _probe() -> FingertipParameters:
    """Return a nearby, deterministic feasible morphology for M6 coverage."""

    return FingertipParameters(
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        stem_width=7.6,
        stem_height=6.0,
        void_width=1.2,
        void_height=0.25,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _objective_payload(evaluation: Any) -> dict[str, Any]:
    """Convert the domain objective at the validation persistence boundary."""
    objective = evaluation.objective
    if hasattr(objective, "to_dict"):
        payload = objective.to_dict()
    elif isinstance(objective, Mapping):
        payload = dict(objective)
    else:
        raise TypeError("evaluation objective must be a structured domain result")
    if not isinstance(payload, dict):
        raise TypeError("objective to_dict() must return an object")
    return payload


def _fields(evaluation: Any) -> tuple[np.ndarray, ...]:
    fields = []
    for record in evaluation.optical_diagnostics:
        fields.append(np.asarray(np.load(record["artifact_field"], allow_pickle=False)["field"], dtype=float))
    return tuple(fields)


def _distance_matrix(fields: tuple[np.ndarray, ...]) -> np.ndarray:
    matrix = np.zeros((len(fields), len(fields)), dtype=float)
    for left in range(len(fields)):
        for right in range(left + 1, len(fields)):
            matrix[left, right] = matrix[right, left] = normalized_field_distance(fields[left], fields[right])
    return matrix


def _compare_legacy_reduction(
    reduced_new: Any,
    legacy: Any,
    *,
    objective_tolerance: float = 1.0e-3,
    field_tolerance: float = 5.0e-2,
) -> dict[str, Any]:
    """Compare independent legacy and reduced-trajectory evaluator outputs."""

    if legacy.status != "success":
        return {
            "legacy_status": legacy.status,
            "legacy_failure_message": legacy.failure_message,
            "pass": False,
        }
    new_records = tuple(
        sorted(reduced_new.optical_diagnostics, key=lambda item: float(item["normalized_location"]))
    )
    legacy_records = tuple(
        sorted(legacy.optical_diagnostics, key=lambda item: float(item["normalized_location"]))
    )
    if len(new_records) != len(legacy_records):
        return {
            "legacy_status": legacy.status,
            "new_state_count": len(new_records),
            "legacy_state_count": len(legacy_records),
            "pass": False,
        }
    if not new_records:
        return {
            "legacy_status": legacy.status,
            "new_state_count": 0,
            "legacy_state_count": 0,
            "pass": False,
        }
    field_comparisons: list[dict[str, Any]] = []
    for new_record, legacy_record in zip(new_records, legacy_records, strict=True):
        if float(new_record["normalized_location"]) != float(legacy_record["normalized_location"]):
            return {
                "legacy_status": legacy.status,
                "location_mismatch": {
                    "new": float(new_record["normalized_location"]),
                    "legacy": float(legacy_record["normalized_location"]),
                },
                "pass": False,
            }
        new_field = np.asarray(
            np.load(new_record["artifact_field"], allow_pickle=False)["field"],
            dtype=float,
        )
        legacy_field = np.asarray(
            np.load(legacy_record["artifact_field"], allow_pickle=False)["field"],
            dtype=float,
        )
        if new_field.shape != legacy_field.shape:
            return {
                "legacy_status": legacy.status,
                "field_shape_mismatch": {
                    "new": list(new_field.shape),
                    "legacy": list(legacy_field.shape),
                },
                "pass": False,
            }
        field_comparisons.append(
            {
                "normalized_location": float(new_record["normalized_location"]),
                "max_abs_field_error": float(np.max(np.abs(new_field - legacy_field))),
                "normalized_field_distance": normalized_field_distance(new_field, legacy_field),
            }
        )
    objective_error = abs(float(reduced_new.objective_value) - float(legacy.objective_value))
    maximum_field_distance = max(
        comparison["normalized_field_distance"] for comparison in field_comparisons
    )
    return {
        "legacy_status": legacy.status,
        "new_objective": float(reduced_new.objective_value),
        "legacy_objective": float(legacy.objective_value),
        "objective_abs_error": float(objective_error),
        "objective_tolerance": objective_tolerance,
        "field_tolerance": field_tolerance,
        "field_comparisons": field_comparisons,
        "maximum_normalized_field_distance": float(maximum_field_distance),
        "comparison_basis": (
            "independent fixed-state oracle versus reduced Lumo3DTrajectoryEvaluator; "
            "same nominal morphology, R=5 mm, travel=1.5 mm, u=0.25/0.50/0.75"
        ),
        "pass": bool(
            objective_error <= objective_tolerance
            and maximum_field_distance <= field_tolerance
        ),
    }


def _plot_outputs(root: Path, evaluations: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for label, evaluation in evaluations.items():
        records = evaluation.checkpoint_diagnostics
        figure, axis = plt.subplots(figsize=(6, 4))
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record["trajectory_id"]), []).append(record)
        for trajectory_id, items in grouped.items():
            items = sorted(items, key=lambda item: item["checkpoint_index"])
            axis.plot(
                [item["post_contact_travel_mm"] for item in items],
                [item["mechanics_diagnostics"].get("max_displacement_mm", np.nan) for item in items],
                marker="o",
                label=trajectory_id,
            )
        axis.set_xlabel("post-contact travel [mm]")
        axis.set_ylabel("maximum displacement [mm]")
        axis.set_title(f"{label} continuous contact trajectories")
        axis.legend(fontsize=6, ncol=2)
        figure.tight_layout()
        figure.savefig(plots / f"contact_trajectories_{label}.png", dpi=160)
        plt.close(figure)

        fields = _fields(evaluation)
        matrix = _distance_matrix(fields)
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="viridis")
        figure.colorbar(image, ax=axis, label="normalized field distance")
        axis.set_title(f"{label} trajectory state distances")
        figure.tight_layout()
        figure.savefig(plots / f"distance_matrix_{label}.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(
            [record["normalized_location"] for record in evaluation.optical_diagnostics],
            [record["post_contact_travel_mm"] for record in evaluation.optical_diagnostics],
            c=[record.get("carrier_absorbed_weight", 0.0) for record in evaluation.optical_diagnostics],
            cmap="plasma",
        )
        axis.set_xlabel("contact location u")
        axis.set_ylabel("post-contact travel [mm]")
        axis.set_title(f"{label} trajectory checkpoint embedding")
        figure.tight_layout()
        figure.savefig(plots / f"trajectory_embedding_{label}.png", dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    for label, evaluation in evaluations.items():
        records = evaluation.optical_diagnostics
        axis.plot(
            [record["post_contact_travel_mm"] for record in records],
            [record.get("carrier_absorbed_weight", 0.0) for record in records],
            "o-",
            label=label,
        )
    axis.set_xlabel("post-contact travel [mm]")
    axis.set_ylabel("carrier absorbed weight")
    axis.set_title("carrier-contact progression")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "carrier_contact_progression.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    for label, evaluation in evaluations.items():
        records = evaluation.optical_diagnostics
        axis.plot(
            [record["post_contact_travel_mm"] for record in records],
            [record.get("total_transport", 0.0) for record in records],
            "o-",
            label=label,
        )
    axis.set_xlabel("post-contact travel [mm]")
    axis.set_ylabel("surviving transport")
    axis.set_title("transport progression")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "transport_progression.png", dpi=160)
    plt.close(figure)


def run_validation(output: str | Path, *, device: str = "cuda:0") -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    protocol = DEFAULT_TRAJECTORY_PROTOCOL
    _write_json(root / "protocol.json", protocol.to_dict() | {"fingerprint": protocol.fingerprint})
    _write_json(
        root / "config.json",
        {
            "schema": "lumo3d-fixed-depth-trajectory-validation-v1",
            "device": device,
            "protocol": protocol.to_dict(),
            "morphologies": {"nominal": _nominal().__dict__, "probe": _probe().__dict__},
            "bo_run": False,
        },
    )
    (root / "architecture_audit.md").write_text(
        "# Architecture audit\n\n"
        "The trajectory evaluator consumes one immutable fixed-depth factorial "
        "optimization protocol (radius and absolute depth are independent). "
        "The fixed-state Lumo3D oracle is used only as an explicit regression "
        "oracle and is not part of the production Ax path. "
        "Newton stepping is shared by final-state and checkpoint APIs; OptiX is "
        "called from the in-memory checkpoint state, with artifacts retained for "
        "provenance and regression comparison.\n",
        encoding="utf-8",
    )

    evaluations: dict[str, Any] = {}
    for label, parameters in (("nominal", _nominal()), ("probe", _probe())):
        evaluator = Lumo3DTrajectoryEvaluator(root / label, device=device)
        result = evaluator.evaluate(parameters)
        if result.status != "success":
            raise RuntimeError(f"{label} trajectory evaluation failed: {result.status}: {result.failure_message}")
        evaluations[label] = result

    all_trajectory_records = {
        label: list(result.checkpoint_diagnostics) for label, result in evaluations.items()
    }
    _write_json(root / "trajectories.json", all_trajectory_records)
    with (root / "checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [record for records in all_trajectory_records.values() for record in records]
        fieldnames = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list, tuple))})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    _write_json(
        root / "trajectory_metrics.json",
        {label: result.diagnostics.get("trajectory_metrics", {}) for label, result in evaluations.items()},
    )
    _write_json(
        root / "trajectory_objective.json",
        {label: _objective_payload(result) for label, result in evaluations.items()},
    )
    _write_json(root / "optics_diagnostics.json", {label: result.optical_diagnostics for label, result in evaluations.items()})
    _plot_outputs(root, evaluations)

    reduced_protocol = TrajectoryEvaluationProtocol(
        contact_locations_u=(0.25, 0.50, 0.75),
        indenter_radii_mm=(5.0,),
        checkpoint_depths_mm=(1.5,),
    )
    reduced_new = Lumo3DTrajectoryEvaluator(root / "legacy_reduction_new", protocol=reduced_protocol, device=device).evaluate(_nominal())
    if reduced_new.status != "success":
        raise RuntimeError("legacy reduction comparison could not complete")
    legacy = FixedStateLumo3DOracle(
        root / "legacy_reduction_legacy",
        device=device,
        normalized_locations=(0.25, 0.50, 0.75),
    ).evaluate(_nominal())
    legacy_comparison = _compare_legacy_reduction(reduced_new, legacy)
    _write_json(root / "legacy_reduction_check.json", legacy_comparison)

    new_final = next(
        record for record in evaluations["nominal"].checkpoint_diagnostics
        if record["normalized_location"] == 0.5 and record["radius_mm"] == 5.0 and record["checkpoint_depth_mm"] == 1.5
    )
    old_root = root / "mechanics_final_state_old"
    old = run_multi_location_sphere_contact(
        parameters=_nominal(),
        device=device,
        radius_mm=5.0,
        travel_mm=1.5,
        normalized_locations=(0.5,),
        artifact_dir=old_root,
        sphere_subdivisions=3,
        max_load_increment_mm=0.05,
        vbd_iterations=10,
        carrier_contact=True,
    )
    old_case = old.locations[0]
    old_vertices = old_case.indentation.mechanics_result.deformed_vertices
    new_archive = np.load(new_final["mechanics_artifact_path"], allow_pickle=False)
    new_vertices = np.asarray(new_archive["deformed_vertices_mm"], dtype=float)
    mechanics_error = float(np.max(np.abs(old_vertices - new_vertices)))
    new_mechanics = dict(new_final["mechanics_diagnostics"])
    old_mechanics = dict(old_case.indentation.diagnostics)
    old_inversions = int(np.count_nonzero(
        _six_volumes(old_vertices, old_case.indentation.mechanics_result.tetrahedra) <= 0.0
    ))
    _write_json(root / "mechanics_final_state_regression.json", {
        "max_abs_deformed_vertex_error_mm": mechanics_error,
        "old_steps": old_case.indentation.mechanics_result.steps,
        "new_steps": new_final["cumulative_step_index"],
        "old_final_pose_mm": list(old_case.indentation.final_indenter_pose.translation_mm),
        "new_final_body_translation_mm": [
            float(new_mechanics.get("final_body_x_mm", 0.0)),
            float(new_mechanics.get("final_body_y_mm", 0.0)),
            float(new_mechanics.get("final_body_z_mm", 0.0)),
        ],
        "old_carrier_contact_active": bool(old_mechanics.get("carrier_contact_active", False)),
        "new_carrier_contact_active": bool(new_mechanics.get("carrier_contact_active", False)),
        "old_carrier_contact_vertex_indices": list(old_mechanics.get("carrier_contact_vertex_indices", ())),
        "new_carrier_contact_vertex_indices": list(new_mechanics.get("carrier_contact_vertex_indices", ())),
        "old_inverted_tetrahedra": old_inversions,
        "new_inverted_tetrahedra": int(new_mechanics.get("inverted_tetrahedra", 0)),
        "old_max_soft_contact_overflow": int(old_mechanics.get("max_soft_contact_overflow", 0)),
        "new_max_soft_contact_overflow": int(new_mechanics.get("max_soft_contact_overflow", 0)),
        "old_max_rigid_contact_overflow": int(old_mechanics.get("max_rigid_contact_overflow", 0)),
        "new_max_rigid_contact_overflow": int(new_mechanics.get("max_rigid_contact_overflow", 0)),
        "pass": mechanics_error <= 2.0e-2,
    })

    incompatible = TrajectoryEvaluationProtocol(
        contact_locations_u=(0.5,),
        indenter_radii_mm=(6.0,),
        checkpoint_depths_mm=(1.5,),
    )
    domain_result = Lumo3DTrajectoryEvaluator(root / "domain_check", protocol=incompatible, device=device).evaluate(_nominal())
    _write_json(root / "radius_domain_check.json", {
        "radius_mm": 6.0,
        "status": domain_result.status,
        "message": domain_result.failure_message,
        "pass": domain_result.status == "domain_incompatible",
    })

    objective_pathology = {
        label: bool(getattr(result.objective, "objective_pathology", False))
        for label, result in evaluations.items()
    }
    legacy_reduction_pass = bool(legacy_comparison.get("pass", False))
    mechanics_regression_pass = mechanics_error <= 2.0e-2
    domain_check_pass = domain_result.status == "domain_incompatible"
    no_objective_pathology = not any(objective_pathology.values())
    hard_checks_pass = (
        legacy_reduction_pass
        and mechanics_regression_pass
        and domain_check_pass
        and no_objective_pathology
    )
    validation_status = (
        "PASS" if hard_checks_pass else "FAIL"
    )
    summary = {
        "schema": "lumo3d-fixed-depth-trajectory-validation-summary-v1",
        "status": validation_status,
        "protocol": protocol.to_dict(),
        "protocol_fingerprint": protocol.fingerprint,
        "morphologies": {
            label: {
                "status": result.status,
                "objective": _objective_payload(result),
                "trajectory_count": len({record["trajectory_id"] for record in result.checkpoint_diagnostics}),
                "checkpoint_count": len(result.checkpoint_diagnostics),
                "optical_state_count": len(result.optical_diagnostics),
                "minimum_transport": min(record["total_transport"] for record in result.optical_diagnostics),
                "maximum_carrier_absorption": max(record["carrier_absorbed_weight"] for record in result.optical_diagnostics),
            }
            for label, result in evaluations.items()
        },
        "total_newton_trajectory_count": sum(len({record["trajectory_id"] for record in result.checkpoint_diagnostics}) for result in evaluations.values()),
        "total_full3d_optical_state_count": sum(len(result.optical_diagnostics) for result in evaluations.values()),
        "legacy_reduction": legacy_comparison,
        "legacy_reduction_absolute_error": legacy_comparison.get("objective_abs_error"),
        "mechanics_final_state_max_abs_error_mm": mechanics_error,
        "domain_check_status": domain_result.status,
        "legacy_reduction_pass": legacy_reduction_pass,
        "mechanics_regression_pass": mechanics_regression_pass,
        "domain_check_pass": domain_check_pass,
        "no_objective_pathology": no_objective_pathology,
        "objective_pathology": objective_pathology,
        "code_cleanup": {
            "files_simplified": [
                "physics/trajectory/indentation.py",
                "physics/newton/vbd.py",
                "optimization/__init__.py",
            ],
            "modules_added": [
                "optimization/protocol.py",
                "lumo/mechanics_contract.py",
                "optimization/objectives.py",
                "optimization/evaluator.py",
                "optimization/deformed_state_artifact.py",
            ],
            "duplicate_constants_removed_from_active_path": True,
            "legacy_modules_retained": {
                "validation.reference.lumo3d_fixed_state_oracle": "independent fixed-state regression oracle only",
            },
            "legacy_modules_removed": [
                "validation/optimization/lumo3d_trajectory_evaluator.py",
                "validation/optimization/lumo3d_common.py",
                "validation/physics/deformed_state_artifact.py",
                "optimization.scenarios",
                "optimization.study",
                "case",
                "fem",
                "examples",
                "visualization",
            ],
        },
        "bo_run": False,
    }
    _write_json(root / "summary.json", summary)
    (root / "reviewer_audit.md").write_text(
        "# Reviewer audit placeholder\n\n"
        "The implementation-agent validation bundle is complete. A fresh read-only "
        "reviewer audit is attached after this deterministic run.\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"output/validation/optimization/{OUTPUT_NAME}")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run_validation(args.output, device=args.device)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
