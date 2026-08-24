"""Artifact-only plots and interpretation inputs for the LUMO 3D pilot.

This module deliberately does not call Newton, OptiX, Ax, or the evaluator.
It reads the persisted pilot/validation artifacts and creates presentation
figures from copies of the raw arrays.  In particular, the optical panels are
labelled as a native FULL_3D internal transport redistribution proxy, not as a
camera observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm


_PARAMETERS = (
    "flat_pad_height",
    "stem_width",
    "stem_height",
    "void_width",
    "void_height",
)
_LOCATIONS = (0.25, 0.50, 0.75)
_FIELD_PERCENTILE = 99.5
_FIELD_GAMMA = 0.5
_OBSERVATION_LEVEL = "FULL_3D native internal transport redistribution proxy"
_OBJECTIVE_LABEL = (
    "Minimum pairwise normalized native FULL_3D internal transport "
    "redistribution proxy [dimensionless]"
)
_OPTICAL_FIELD_LABEL = "z-integrated native FULL_3D internal weighted path length [mm]"


def _wrapped_objective_label() -> str:
    return textwrap.fill(_OBJECTIVE_LABEL, width=32)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _resolve_artifact_path(raw_path: str | Path, pilot_dir: Path) -> Path:
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(pilot_dir / path)
        candidates.extend(ancestor / path for ancestor in [Path.cwd(), *pilot_dir.parents])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"artifact path does not exist: {raw_path}")


def _record_evaluation_path(record: dict[str, Any], pilot_dir: Path) -> Path:
    artifact_paths = record.get("artifact_paths")
    if not isinstance(artifact_paths, list) or not artifact_paths:
        raise ValueError("successful record has no optical artifact paths")
    first = _resolve_artifact_path(str(artifact_paths[0]), pilot_dir)
    evaluation = first.parent / "evaluation.json"
    if not evaluation.exists():
        raise FileNotFoundError(f"missing candidate evaluation artifact: {evaluation}")
    return evaluation


def _successful_records(pilot_dir: Path) -> list[dict[str, Any]]:
    records = _read_json(pilot_dir / "bo_trials.json")
    if not isinstance(records, list):
        raise ValueError("bo_trials.json must contain a list")
    successful = [
        record
        for record in records
        if isinstance(record, dict) and record.get("status") == "success"
    ]
    if not successful:
        raise ValueError("pilot contains no successful evaluations")
    return successful


def _positive_values(fields: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(field, dtype=float).ravel() for field in fields]
    if not values:
        raise ValueError("at least one field is required")
    positive = np.concatenate(values)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if not len(positive):
        raise ValueError("fields contain no finite positive values")
    return positive


def shared_field_norm(fields: Iterable[np.ndarray]) -> PowerNorm:
    """Return one display-only robust normalization for all supplied fields."""

    positive = _positive_values(fields)
    vmax = float(np.percentile(positive, _FIELD_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(positive))
    return PowerNorm(gamma=_FIELD_GAMMA, vmin=0.0, vmax=vmax, clip=True)


def _load_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as artifact:
        required = {"field", "axis_0", "axis_1", "axis_2"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"{path} is missing field keys: {sorted(missing)}")
        field = np.array(artifact["field"], dtype=float, copy=True)
        axis_0 = np.array(artifact["axis_0"], dtype=float, copy=True)
        axis_1 = np.array(artifact["axis_1"], dtype=float, copy=True)
        axis_2 = np.array(artifact["axis_2"], dtype=float, copy=True)
    if field.ndim != 3 or not np.all(np.isfinite(field)):
        raise ValueError(f"field must be finite and 3D: {path}")
    if field.shape != (len(axis_0) - 1, len(axis_1) - 1, len(axis_2) - 1):
        raise ValueError(f"field/axis shape mismatch: {path}")
    return field, axis_0, axis_1, axis_2


def _field_paths(candidate_evaluation: Path) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for raw in candidate_evaluation.parent.glob("location_u_*.npz"):
        token = raw.stem.removeprefix("location_u_")
        location = float(token)
        result[location] = raw
    missing = set(_LOCATIONS).difference(result)
    if missing:
        raise ValueError(f"candidate is missing contact-state fields: {sorted(missing)}")
    return result


def _evaluation_summary(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    summary = value.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"evaluation summary missing in {path}")
    return value


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bo_history(records: list[dict[str, Any]], plots: Path) -> None:
    x = np.asarray([int(record["trial_index"]) for record in records], dtype=float)
    y = np.asarray([float(record["objective_value"]) for record in records], dtype=float)
    running = np.maximum.accumulate(y)
    phases = [str(record.get("phase", "")) for record in records]
    colors = {"nominal": "#4C78A8", "initialization": "#59A14F", "search": "#F28E2B"}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for phase in dict.fromkeys(phases):
        mask = np.asarray([item == phase for item in phases])
        ax.scatter(x[mask], y[mask], s=34, label=phase, color=colors.get(phase, "#666666"))
    ax.plot(x, running, color="#222222", linewidth=1.4, label="running best")
    ax.set_xlabel("Ax trial index")
    ax.set_ylabel(_wrapped_objective_label())
    ax.set_title("LUMO 3D BO objective history")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, plots / "bo_history.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(x, running, marker="o", color="#222222")
    ax.set_xlabel("Ax trial index")
    ax.set_ylabel(_wrapped_objective_label())
    ax.set_title("Running best objective")
    ax.grid(alpha=0.25)
    _save(fig, plots / "running_best.png")


def _plot_parameter_history(records: list[dict[str, Any]], config: dict[str, Any], plots: Path) -> None:
    bounds = {
        name: (float(pair[1]), float(pair[2]))
        for pair in config["contract"]["bounds_mm"]
        for name in [pair[0]]
    }
    x = np.asarray([int(record["trial_index"]) for record in records], dtype=float)
    for name in _PARAMETERS:
        y = np.asarray([float(record["parameters"][name]) for record in records], dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 3.7))
        ax.plot(x, y, marker="o", linewidth=1.2)
        lower, upper = bounds[name]
        ax.axhspan(lower, upper, color="#A0CBE8", alpha=0.18, label="active bounds")
        ax.set_xlabel("Ax trial index")
        ax.set_ylabel(f"{name} [mm]")
        ax.set_title(f"{name} history")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, plots / f"parameter_history_{name}.png")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    for ax, name in zip(axes.flat, _PARAMETERS):
        x_values = np.asarray([float(record["parameters"][name]) for record in records])
        y_values = np.asarray([float(record["objective_value"]) for record in records])
        ax.scatter(x_values, y_values, c=np.arange(len(records)), cmap="viridis", s=26)
        ax.set_xlabel(f"{name} [mm]")
        ax.set_ylabel(_wrapped_objective_label())
        ax.grid(alpha=0.2)
    fig.suptitle("Morphology parameters versus objective")
    _save(fig, plots / "parameter_objective_scatter.png")


def _plot_distance_matrix(matrix: Any, path: Path, title: str, *, vmax: float) -> None:
    values = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(values, cmap="magma", vmin=0.0, vmax=vmax)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", color="white")
    ax.set_xticks(range(3), ["u=.25", "u=.50", "u=.75"])
    ax.set_yticks(range(3), ["u=.25", "u=.50", "u=.75"])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="normalized native FULL_3D field L1")
    _save(fig, path)


def _plot_metrics(nominal: dict[str, Any], best: dict[str, Any], plots: Path) -> None:
    labels = ["nominal", "best"]
    summaries = [nominal["summary"], best["summary"]]
    objective = [float(item["objective_value"]) for item in summaries]
    escaped = [float(np.mean([item["escaped_weight"] for item in record["optics"]])) for record in (nominal, best)]
    absorbed = [float(np.mean([item["absorbed_weight"] for item in record["optics"]])) for record in (nominal, best)]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(x, objective, color=["#4C78A8", "#E15759"])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(textwrap.fill(_OBJECTIVE_LABEL, width=28))
    axes[0].set_title("Optimization objective")
    axes[1].bar(x - 0.18, escaped, width=0.36, label="mean escaped weight", color="#59A14F")
    axes[1].bar(x + 0.18, absorbed, width=0.36, label="mean absorbed weight", color="#F28E2B")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("weight fraction of launched transport")
    axes[1].set_title("Energy-related diagnostics (not objective)")
    axes[1].legend(frameon=False, fontsize=8)
    _save(figure, plots / "nominal_vs_best_metrics.png")


def _mechanics_displacement(path: Path) -> tuple[float, float]:
    with np.load(path) as artifact:
        rest = np.asarray(artifact["rest_vertices_mm"], dtype=float)
        deformed = np.asarray(artifact["deformed_vertices_mm"], dtype=float)
    displacement = np.linalg.norm(deformed - rest, axis=1)
    return float(np.max(displacement)), float(np.sqrt(np.mean(displacement**2)))


def _plot_mechanics(
    nominal_eval_path: Path,
    best_eval_path: Path,
    plots: Path,
) -> None:
    rows: list[tuple[str, float, float]] = []
    for label, evaluation_path in (("nominal", nominal_eval_path), ("best", best_eval_path)):
        evaluation = _evaluation_summary(evaluation_path)
        for mechanics in evaluation["mechanics"]:
            artifact = _resolve_artifact_path(mechanics["mechanics_artifact_path"], evaluation_path.parent)
            maximum, rms = _mechanics_displacement(artifact)
            rows.append((label, float(mechanics["normalized_location"]), maximum, rms))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    for label, color in (("nominal", "#4C78A8"), ("best", "#E15759")):
        subset = [row for row in rows if row[0] == label]
        axes[0].plot([row[1] for row in subset], [row[2] for row in subset], marker="o", label=label, color=color)
        axes[1].plot([row[1] for row in subset], [row[3] for row in subset], marker="o", label=label, color=color)
    for ax, title, ylabel in (
        (axes[0], "maximum displacement", "max |u| [mm]"),
        (axes[1], "RMS displacement", "RMS |u| [mm]"),
    ):
        ax.set_xlabel("normalized contact location u")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    _save(fig, plots / "deformation_summary.png")


def _plot_optical_fields(
    nominal_eval_path: Path,
    best_eval_path: Path,
    plots: Path,
) -> dict[str, Any]:
    evaluations = {"nominal": nominal_eval_path, "best": best_eval_path}
    fields: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    field_metadata: dict[tuple[str, float], dict[str, Any]] = {}
    field_digests_before: dict[str, str] = {}
    for label, evaluation_path in evaluations.items():
        for location, path in _field_paths(evaluation_path).items():
            fields[(label, location)] = _load_field(path)
            metadata_path = path.with_suffix(".json")
            metadata = _read_object(metadata_path)
            if metadata.get("field_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"optical field checksum mismatch: {path}")
            if metadata.get("field_axis_order") != "x,y,z":
                raise ValueError(f"optical field axis order is not FULL_3D: {path}")
            field_metadata[(label, location)] = metadata
            field_digests_before[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference_axes = fields[ next(iter(fields)) ][1:]
    for key, (_, axis_x, axis_y, axis_z) in fields.items():
        if not all(np.array_equal(actual, expected) for actual, expected in zip((axis_x, axis_y, axis_z), reference_axes)):
            raise ValueError(f"optical grids are not shared: {key}")
    norm = shared_field_norm(field[0] for field in fields.values())
    vmax = float(norm.vmax)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), constrained_layout=True)
    image = None
    for row, label in enumerate(("nominal", "best")):
        for column, location in enumerate(_LOCATIONS):
            field, axis_x, axis_y, _ = fields[(label, location)]
            projected = np.sum(field, axis=2).T
            masked = np.ma.masked_less_equal(projected, 0.0)
            image = axes[row, column].pcolormesh(
                axis_x,
                axis_y,
                masked,
                cmap="viridis",
                norm=norm,
                shading="flat",
            )
            axes[row, column].set_title(f"{label}, contact u={location:.2f}")
            axes[row, column].set_xlabel("x [mm]")
            axes[row, column].set_ylabel("y [mm]")
            axes[row, column].set_aspect("equal")
    if image is None:
        raise RuntimeError("no optical fields were plotted")
    fig.colorbar(image, ax=axes, label=_OPTICAL_FIELD_LABEL)
    fig.suptitle(
        "z-integrated FULL_3D internal weighted path length\n"
        "shared display-only PowerNorm (gamma=0.5, positive 99.5th percentile); "
        "not a camera image"
    )
    _save(fig, plots / "nominal_vs_best_optical_outputs.png")
    field_digests_after = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in field_digests_before
    }
    fingerprints = {
        metadata.get("result", {}).get("transport_configuration_fingerprint")
        for metadata in field_metadata.values()
    }
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("optical transport configuration fingerprints are not shared")
    first_metadata = next(iter(field_metadata.values()))
    settings = first_metadata["contract"]["transport_configuration"]["settings"]
    grid = {
        "x_bounds_mm": settings["x_bounds_mm"],
        "y_bounds_mm": settings["y_bounds_mm"],
        "z_bounds_mm": [
            float(fields[next(iter(fields))][3][0]),
            float(fields[next(iter(fields))][3][-1]),
        ],
        "x_bins": int(settings["internal_grid_width"]),
        "y_bins": int(settings["internal_grid_height"]),
        "z_bins": int(settings["internal_z_bins"]),
    }
    return {
        "field_count": len(fields),
        "field_percentile": _FIELD_PERCENTILE,
        "power_gamma": _FIELD_GAMMA,
        "shared_vmax": vmax,
        "raw_field_unchanged": field_digests_before == field_digests_after,
        "quantity": _OPTICAL_FIELD_LABEL,
        "no_smoothing": True,
        "shared_grid": True,
        "field_checksums_verified": len(field_metadata) == len(fields),
        "field_digests_before": field_digests_before,
        "field_digests_after": field_digests_after,
        "transport_configuration_fingerprint": next(iter(fingerprints)),
        "grid": grid,
    }


def _write_report_bundle(
    *,
    pilot: Path,
    destination: Path,
    nominal_record: dict[str, Any],
    best_record: dict[str, Any],
    nominal_eval_path: Path,
    best_eval_path: Path,
    nominal_eval: dict[str, Any],
    best_eval: dict[str, Any],
) -> None:
    """Preserve the source campaign and selected observations in the report."""

    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "preflight.json",
        "bo_trials.csv",
        "bo_trials.json",
        "checkpoint.json",
        "registry.json",
        "ax_client.json",
        "validation.json",
    ):
        source = pilot / name
        if source.exists():
            shutil.copy2(source, destination / name)
    for label, evaluation_path in (("nominal", nominal_eval_path), ("best", best_eval_path)):
        shutil.copytree(evaluation_path.parent, destination / "observations" / label, dirs_exist_ok=True)
    for label, record, evaluation in (
        ("nominal", nominal_record, nominal_eval),
        ("best", best_record, best_eval),
    ):
        summary = {
            "record": record,
            "evaluation_summary": evaluation["summary"],
            "mechanics_locations": [
                {
                    "normalized_location": item.get("normalized_location"),
                    "max_displacement_mm": item.get("max_displacement_mm"),
                    "inverted_tetrahedra": item.get("inverted_tetrahedra"),
                    "max_soft_contact_overflow": item.get("max_soft_contact_overflow"),
                }
                for item in evaluation.get("mechanics", [])
            ],
            "optical_locations": [
                {
                    "escaped_weight": item.get("escaped_weight"),
                    "absorbed_weight": item.get("absorbed_weight"),
                    "energy_balance_error": item.get("energy_balance_error"),
                }
                for item in evaluation.get("optics", [])
            ],
        }
        (destination / f"{label}_candidate.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (destination / "logs").mkdir(parents=True, exist_ok=True)
    (destination / "logs" / "stage9_report.log").write_text(
        "Generated from persisted LUMO 3D pilot artifacts; no Newton, OptiX, Ax, or evaluator call.\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        "# LUMO 3D overnight artifact bundle\n\n"
        "This directory preserves the bounded Stage 8 pilot, its Ax/checkpoint/registry "
        "state, selected nominal/best observations, Stage 9 plots, and the final audit.\n\n"
        "The optical panels use raw persisted FULL_3D internal fields with a shared "
        "display-only PowerNorm and z integration for visualization. They are not camera images.\n\n"
        f"Source pilot: `{pilot}`\n",
        encoding="utf-8",
    )


def run_lumo3d_report(pilot_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate Stage 9 plots from an existing pilot without recomputation."""

    pilot = Path(pilot_dir).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else pilot.parent.parent / "overnight_lumo_3d_20260819"
    plots = destination / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    all_records = _read_json(pilot / "bo_trials.json")
    if not isinstance(all_records, list):
        raise ValueError("bo_trials.json must contain a list")
    records = [record for record in all_records if isinstance(record, dict) and record.get("status") == "success"]
    config = _read_object(pilot / "config.json")
    nominal_record = next((record for record in records if record.get("phase") == "nominal"), records[0])
    best_record = max(records, key=lambda record: float(record["objective_value"]))
    nominal_eval_path = _record_evaluation_path(nominal_record, pilot)
    best_eval_path = _record_evaluation_path(best_record, pilot)
    nominal_eval = _evaluation_summary(nominal_eval_path)
    best_eval = _evaluation_summary(best_eval_path)

    _plot_bo_history(records, plots)
    _plot_parameter_history(records, config, plots)
    nominal_matrix = np.asarray(nominal_eval["summary"]["pairwise_distance_matrix"], dtype=float)
    best_matrix = np.asarray(best_eval["summary"]["pairwise_distance_matrix"], dtype=float)
    matrix_vmax = float(max(np.max(nominal_matrix), np.max(best_matrix)))
    _plot_distance_matrix(nominal_matrix, plots / "nominal_distance_matrix.png", "Nominal contact-state separation", vmax=matrix_vmax)
    _plot_distance_matrix(best_matrix, plots / "best_distance_matrix.png", "Best contact-state separation", vmax=matrix_vmax)
    _plot_metrics(nominal_eval, best_eval, plots)
    _plot_mechanics(nominal_eval_path, best_eval_path, plots)
    optical = _plot_optical_fields(nominal_eval_path, best_eval_path, plots)
    _write_report_bundle(
        pilot=pilot,
        destination=destination,
        nominal_record=nominal_record,
        best_record=best_record,
        nominal_eval_path=nominal_eval_path,
        best_eval_path=best_eval_path,
        nominal_eval=nominal_eval,
        best_eval=best_eval,
    )
    validation = _read_object(pilot / "validation.json")
    search = validation["search"]
    validation_tier = validation["validation"]
    search_validation = {
        "search": {
            "nominal": search["nominal"]["diagnostics"]["objective_value"],
            "best": search["best"]["diagnostics"]["objective_value"],
        },
        "validation": {
            "nominal": validation_tier["nominal"]["diagnostics"]["objective_value"],
            "best": validation_tier["best"]["diagnostics"]["objective_value"],
        },
    }

    nominal_objective = float(nominal_record["objective_value"])
    best_objective = float(best_record["objective_value"])
    report = {
        "schema": "lumo3d-stage9-report-v1",
        "pilot_dir": str(pilot),
        "output_dir": str(destination),
        "observation_level": _OBSERVATION_LEVEL,
        "objective_name": "contact_state_separation",
        "objective_label": _OBJECTIVE_LABEL,
        "objective_direction": "maximize",
        "successful_evaluations": len(records),
        "failed_evaluations": len(all_records) - len(records),
        "nominal_parameters": nominal_record["parameters"],
        "best_parameters": best_record["parameters"],
        "nominal_objective": nominal_objective,
        "best_objective": best_objective,
        "improvement_percent": 100.0 * (best_objective / nominal_objective - 1.0),
        "best_trial_index": int(best_record["trial_index"]),
        "search_validation_ordering": _read_object(pilot / "summary.json").get("validation_ordering"),
        "search_validation_objectives": search_validation,
        "optical_display": optical,
        "plots": sorted(str(path.relative_to(destination)) for path in plots.glob("*.png")),
        "interpretation_limits": [
            "Objective is the minimum pairwise normalized native FULL_3D internal transport redistribution proxy.",
            "Total escaped/absorbed transport is reported separately and is not the BO objective.",
            "No object-interface optics or camera observation operator is present in this deformation-only scene.",
            "Optical fields are raw persisted outputs; only display normalization and z-integration are applied in plotting.",
            "The optical panels are not camera images or camera observations.",
        ],
    }
    (destination / "stage9_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_lumo3d_report(args.pilot_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_lumo3d_report", "shared_field_norm"]
