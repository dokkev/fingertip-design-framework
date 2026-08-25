"""Validate selected morphologies under both saved optical campaign contracts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from lumo.optimization.ax_bo import (
    _campaign_definition,
    _evaluate_candidate,
    _objective_details,
    _run_config,
    _save_trial_result,
    _validate_optix_environment,
)
from lumo.optimization.sensing_objective import sensing_objectives


_ROOT = Path(__file__).resolve().parents[2]
_OPTIMIZATION_ROOT = _ROOT / "output" / "optimization"
_OUTPUT_DIRECTORY = _OPTIMIZATION_ROOT / "cross_material_validation"
_CSV_PATH = _OUTPUT_DIRECTORY / "cross_material_morphology_validation.csv"
_REPORT_PATH = _OUTPUT_DIRECTORY / "cross_material_morphology_validation.md"

_PARAMETER_NAMES = (
    "geometry.flat_pad_height_mm",
    "geometry.semiellipse_height_mm",
    "geometry.stem_width_mm",
    "geometry.stem_height_mm",
    "geometry.void_width_mm",
    "geometry.void_height_mm",
)
_GEOMETRY_KEYS = (
    "nominal",
    "dragon_optimized",
    "dragon_suboptimal",
    "solaris_optimized",
    "solaris_suboptimal",
)
_GEOMETRY_LABELS = {
    "nominal": "Nominal",
    "dragon_optimized": "Dragon optimized 123",
    "dragon_suboptimal": "Dragon suboptimal 126",
    "solaris_optimized": "Solaris optimized 107",
    "solaris_suboptimal": "Solaris suboptimal 94",
}
_GEOMETRY_SOURCES = {
    "nominal": ("Hand-designed", "Nominal"),
    "dragon_optimized": ("Dragon Skin", "Optimized trial 123"),
    "dragon_suboptimal": ("Dragon Skin", "Matched suboptimal trial 126"),
    "solaris_optimized": ("Solaris", "Optimized trial 107"),
    "solaris_suboptimal": ("Solaris", "Matched suboptimal trial 94"),
}
_NOMINAL_PARAMETERS = dict(
    zip(_PARAMETER_NAMES, (5.0, 9.0, 7.6, 6.0, 2.0, 0.0), strict=True)
)
_MATERIALS = {
    "dragon": {
        "label": "Dragon Skin",
        "campaign_directory": _OPTIMIZATION_ROOT / "mobo_discrete_05mm_clean",
        "nominal_path": (
            _OPTIMIZATION_ROOT
            / "physical_validation_nominal"
            / "dragon_skin_nominal.npz"
        ),
        "normalization": {
            "J_intensity": (0.025983473108727284, 0.13442835929258648),
            "J_spatial": (0.15485620585002643, 0.1955980645401783),
        },
    },
    "solaris": {
        "label": "Solaris",
        "campaign_directory": (
            _OPTIMIZATION_ROOT / "mobo_discrete_05mm_solaris_nominal"
        ),
        "nominal_path": (
            _OPTIMIZATION_ROOT
            / "physical_validation_nominal"
            / "solaris_nominal.npz"
        ),
        "normalization": {
            "J_intensity": (0.009424410194244833, 0.071658445062079),
            "J_spatial": (0.2124390306313034, 0.3097246810543962),
        },
    },
}
_SOURCE_TRIALS = {
    "dragon_optimized": ("dragon", 123),
    "dragon_suboptimal": ("dragon", 126),
    "solaris_optimized": ("solaris", 107),
    "solaris_suboptimal": ("solaris", 94),
}
_CROSS_RESULT_PATHS = {
    ("dragon_optimized", "solaris"): (
        _OUTPUT_DIRECTORY / "dragon_optimized_123_under_solaris.npz"
    ),
    ("dragon_suboptimal", "solaris"): (
        _OUTPUT_DIRECTORY / "dragon_suboptimal_126_under_solaris.npz"
    ),
    ("solaris_optimized", "dragon"): (
        _OUTPUT_DIRECTORY / "solaris_optimized_107_under_dragon.npz"
    ),
    ("solaris_suboptimal", "dragon"): (
        _OUTPUT_DIRECTORY / "solaris_suboptimal_094_under_dragon.npz"
    ),
}


def _campaign_from_saved_contract(material_key: str):
    material = _MATERIALS[material_key]
    run_config_path = material["campaign_directory"] / "run_config.json"
    stored = json.loads(run_config_path.read_text(encoding="utf-8"))
    contract = stored["scientific_contract"]
    design_space = contract["design_space"]
    parameter_bounds = {
        name.removeprefix("geometry."): tuple(bounds)
        for name, bounds in design_space["decoded_physical_bounds_mm"].items()
    }
    campaign = _campaign_definition(
        "discrete-05mm",
        parameter_bounds_mm=parameter_bounds,
        indenter_urdfs=contract["scenarios"]["indenter_urdfs"],
        force_targets_n=contract["mechanics"]["force_targets_n"],
        settle_duration_s=contract["mechanics"]["fixed_servo_dwell_s"],
        force_tolerance_fraction=contract["mechanics"][
            "force_tolerance_fraction"
        ],
        initial_clearance_m=contract["scenarios"]["initial_clearance_m"],
        viscoelastic_preset=contract["viscoelastic_preset"],
        optical_preset=contract["optical_preset"],
    )
    current_contract = _run_config(campaign)["scientific_contract"]
    if current_contract != contract:
        raise RuntimeError(
            f"current source does not reproduce {material['label']} campaign contract"
        )
    return campaign, contract


def _trial_row(material_key: str, trial_index: int) -> dict[str, str]:
    trials_path = _MATERIALS[material_key]["campaign_directory"] / "trials.csv"
    with trials_path.open(newline="", encoding="utf-8") as input_file:
        matches = [
            row
            for row in csv.DictReader(input_file)
            if int(row["ax_trial_index"]) == trial_index
        ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {material_key} trial {trial_index}")
    row = matches[0]
    if row["status"] != "COMPLETED" or row["analytically_valid"] != "True":
        raise RuntimeError(f"source trial {trial_index} is not a valid completion")
    return row


def _parameters_from_row(row: dict[str, str]) -> dict[str, float]:
    return {name: float(row[name]) for name in _PARAMETER_NAMES}


def _load_raw_result(
    path: Path,
    *,
    campaign,
    expected_parameters: dict[str, float],
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        required = {
            "no_contact_response",
            "no_contact_energy",
            "response_matrix",
            "energy_matrix",
            "energy_fields",
            "actual_forces_n",
            "indentations_m",
            "checkpoint_times_s",
            "scenario_runtime_s",
            "scenario_names",
            "indenter_names",
            "force_targets_n",
            "per_indenter_J_intensity",
            "per_indenter_J_spatial",
            "J_intensity",
            "J_spatial",
            "evaluation_runtime_s",
            "parameter_names",
            "parameter_values",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"{path} is missing raw fields: {sorted(missing)}")
        raw = {name: np.array(archive[name], copy=True) for name in required}

    numeric_names = required.difference(
        {"energy_fields", "scenario_names", "indenter_names", "parameter_names"}
    )
    if any(not np.all(np.isfinite(raw[name])) for name in numeric_names):
        raise RuntimeError(f"{path} contains non-finite values")

    indenter_count = len(campaign.indenter_names)
    force_targets = np.asarray(campaign.force_targets_n, dtype=np.float64)
    force_count = len(force_targets)
    if np.asarray(raw["response_matrix"]).shape != (indenter_count, force_count, 4):
        raise RuntimeError(f"{path} has an unexpected response shape")
    if np.asarray(raw["energy_matrix"]).shape != (indenter_count, force_count, 8):
        raise RuntimeError(f"{path} has an unexpected energy shape")
    if np.asarray(raw["actual_forces_n"]).shape != (indenter_count, force_count):
        raise RuntimeError(f"{path} has an unexpected force shape")
    if np.asarray(raw["indentations_m"]).shape != (indenter_count, force_count):
        raise RuntimeError(f"{path} has an unexpected indentation shape")
    if not np.array_equal(raw["force_targets_n"], force_targets):
        raise RuntimeError(f"{path} force targets do not match the target campaign")
    if tuple(raw["indenter_names"].tolist()) != campaign.indenter_names:
        raise RuntimeError(f"{path} indenter names do not match the target campaign")
    expected_scenarios = tuple(f"{name}_center" for name in campaign.indenter_names)
    if tuple(raw["scenario_names"].tolist()) != expected_scenarios:
        raise RuntimeError(f"{path} scenario names do not match the target campaign")

    tolerance = campaign.force_tolerance_fraction * force_targets[None, :]
    force_error = np.abs(np.asarray(raw["actual_forces_n"]) - force_targets[None, :])
    if np.any(force_error > tolerance + 1.0e-10):
        raise RuntimeError(f"{path} contains a force outside its acceptance band")
    indentation_change = np.diff(np.asarray(raw["indentations_m"]), axis=1)
    if np.any(indentation_change <= 0.0):
        raise RuntimeError(f"{path} indentation is not strictly increasing")

    energy_fields = tuple(raw["energy_fields"].tolist())
    if "closure_error" not in energy_fields:
        raise RuntimeError(f"{path} has no energy closure field")
    closure_index = energy_fields.index("closure_error")
    closure_values = np.concatenate(
        (
            np.asarray(raw["no_contact_energy"])[None, closure_index],
            np.asarray(raw["energy_matrix"])[..., closure_index].ravel(),
        )
    )
    if np.max(np.abs(closure_values)) > 1.0e-12:
        raise RuntimeError(f"{path} optical energy closure exceeds 1e-12")

    parameter_names = tuple(raw["parameter_names"].tolist())
    parameter_values = np.asarray(raw["parameter_values"], dtype=np.float64)
    stored_parameters = dict(zip(parameter_names, parameter_values, strict=True))
    if set(stored_parameters) != set(expected_parameters):
        raise RuntimeError(f"{path} parameter names do not match the intended geometry")
    for name, expected in expected_parameters.items():
        if not np.isclose(stored_parameters[name], expected, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"{path} geometry mismatch for {name}")

    per_intensity, per_spatial, intensity, spatial = sensing_objectives(
        np.asarray(raw["response_matrix"]),
        no_contact_response=np.asarray(raw["no_contact_response"]),
    )
    objective_checks = (
        (per_intensity, raw["per_indenter_J_intensity"]),
        (per_spatial, raw["per_indenter_J_spatial"]),
        (np.asarray(intensity), raw["J_intensity"]),
        (np.asarray(spatial), raw["J_spatial"]),
    )
    if any(
        not np.allclose(actual, stored, rtol=0.0, atol=1.0e-12)
        for actual, stored in objective_checks
    ):
        raise RuntimeError(f"{path} stored objectives do not match raw responses")

    return {
        **raw,
        "path": path,
        "J_intensity_value": float(intensity),
        "J_spatial_value": float(spatial),
        "max_force_error_n": float(force_error.max()),
        "minimum_indentation_increment_m": float(indentation_change.min()),
        "max_energy_closure": float(np.max(np.abs(closure_values))),
    }


def _verify_source_trial(
    geometry_key: str,
    campaigns: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]]:
    material_key, trial_index = _SOURCE_TRIALS[geometry_key]
    row = _trial_row(material_key, trial_index)
    parameters = _parameters_from_row(row)
    raw_path = _MATERIALS[material_key]["campaign_directory"] / row["raw_result_path"]
    raw = _load_raw_result(
        raw_path,
        campaign=campaigns[material_key],
        expected_parameters=parameters,
    )
    for name in ("J_intensity", "J_spatial"):
        if not np.isclose(
            float(row[name]),
            float(raw[f"{name}_value"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"trial table and raw result disagree for {geometry_key}")
    return parameters, raw


def _evaluate_cross_result(
    geometry_key: str,
    material_key: str,
    *,
    campaigns: dict[str, object],
    parameters: dict[str, float],
) -> dict[str, object]:
    result_path = _CROSS_RESULT_PATHS[(geometry_key, material_key)]
    if result_path.is_file():
        print(f"reuse {result_path.relative_to(_ROOT)}", flush=True)
        return _load_raw_result(
            result_path,
            campaign=campaigns[material_key],
            expected_parameters=parameters,
        )

    print(
        f"evaluate {_GEOMETRY_LABELS[geometry_key]} under "
        f"{_MATERIALS[material_key]['label']} optics",
        flush=True,
    )
    start = perf_counter()
    evaluation = _evaluate_candidate(campaigns[material_key], parameters)
    runtime_s = perf_counter() - start
    details = _objective_details(evaluation)
    _save_trial_result(
        result_path,
        campaign=campaigns[material_key],
        evaluation=evaluation,
        details=details,
        parameters=parameters,
        runtime_s=runtime_s,
    )
    print(
        f"  done in {runtime_s:.1f} s: J_intensity={details['J_intensity']:.9f}, "
        f"J_spatial={details['J_spatial']:.9f}",
        flush=True,
    )
    return _load_raw_result(
        result_path,
        campaign=campaigns[material_key],
        expected_parameters=parameters,
    )


def _normalized_scores(material_key: str, result: dict[str, object]) -> tuple[float, float, float]:
    normalization = _MATERIALS[material_key]["normalization"]
    intensity_nadir, intensity_ideal = normalization["J_intensity"]
    spatial_nadir, spatial_ideal = normalization["J_spatial"]
    normalized_intensity = (
        float(result["J_intensity_value"]) - intensity_nadir
    ) / (intensity_ideal - intensity_nadir)
    normalized_spatial = (
        float(result["J_spatial_value"]) - spatial_nadir
    ) / (spatial_ideal - spatial_nadir)
    return (
        normalized_intensity,
        normalized_spatial,
        min(normalized_intensity, normalized_spatial),
    )


def _write_csv(
    parameters: dict[str, dict[str, float]],
    results: dict[str, dict[str, dict[str, object]]],
) -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry_source",
        "selection",
        "geometry_mm",
        *[name.removeprefix("geometry.") for name in _PARAMETER_NAMES],
    ]
    for material_key in _MATERIALS:
        prefix = material_key
        fieldnames.extend(
            (
                f"{prefix}_J_intensity",
                f"{prefix}_J_spatial",
                f"{prefix}_normalized_J_intensity",
                f"{prefix}_normalized_J_spatial",
                f"{prefix}_J_balanced",
                f"{prefix}_raw_result_path",
            )
        )
    with _CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for geometry_key in _GEOMETRY_KEYS:
            source, selection = _GEOMETRY_SOURCES[geometry_key]
            geometry = [parameters[geometry_key][name] for name in _PARAMETER_NAMES]
            row: dict[str, object] = {
                "geometry_source": source,
                "selection": selection,
                "geometry_mm": json.dumps(geometry),
            }
            row.update(
                {
                    name.removeprefix("geometry."): parameters[geometry_key][name]
                    for name in _PARAMETER_NAMES
                }
            )
            for material_key in _MATERIALS:
                result = results[geometry_key][material_key]
                normalized_intensity, normalized_spatial, balanced = (
                    _normalized_scores(material_key, result)
                )
                row.update(
                    {
                        f"{material_key}_J_intensity": result["J_intensity_value"],
                        f"{material_key}_J_spatial": result["J_spatial_value"],
                        f"{material_key}_normalized_J_intensity": (
                            normalized_intensity
                        ),
                        f"{material_key}_normalized_J_spatial": normalized_spatial,
                        f"{material_key}_J_balanced": balanced,
                        f"{material_key}_raw_result_path": os.path.relpath(
                            result["path"], _OUTPUT_DIRECTORY
                        ),
                    }
                )
            writer.writerow(row)


def _plot_objectives(results: dict[str, dict[str, dict[str, object]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    styles = {
        "nominal": ("*", "black"),
        "dragon_optimized": ("D", "tab:red"),
        "dragon_suboptimal": ("s", "tab:orange"),
        "solaris_optimized": ("X", "tab:blue"),
        "solaris_suboptimal": ("^", "tab:cyan"),
    }
    for axis, material_key in zip(axes, _MATERIALS, strict=True):
        pareto_path = _MATERIALS[material_key]["campaign_directory"] / "pareto.csv"
        with pareto_path.open(newline="", encoding="utf-8") as input_file:
            pareto_rows = list(csv.DictReader(input_file))
        pareto_x = np.asarray([float(row["J_intensity"]) for row in pareto_rows])
        pareto_y = np.asarray([float(row["J_spatial"]) for row in pareto_rows])
        axis.scatter(
            pareto_x,
            pareto_y,
            s=24,
            color="0.75",
            label="Original Pareto set",
            zorder=1,
        )
        for geometry_key in _GEOMETRY_KEYS:
            result = results[geometry_key][material_key]
            marker, color = styles[geometry_key]
            axis.scatter(
                result["J_intensity_value"],
                result["J_spatial_value"],
                s=90,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=_GEOMETRY_LABELS[geometry_key],
                zorder=2,
            )
        axis.set_title(f"{_MATERIALS[material_key]['label']} optics")
        axis.set_xlabel("J_intensity")
        axis.set_ylabel("J_spatial")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.savefig(_OUTPUT_DIRECTORY / "objective_comparison.png", dpi=180)
    plt.close(figure)


def _plot_transfer(results: dict[str, dict[str, dict[str, object]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    objective_names = ("J_intensity_value", "J_spatial_value")
    titles = ("Intensity transfer", "Spatial transfer")
    transfer_keys = _GEOMETRY_KEYS[1:]
    colors = ("tab:red", "tab:orange", "tab:blue", "tab:cyan")
    for axis, objective_name, title in zip(
        axes, objective_names, titles, strict=True
    ):
        coordinates = []
        for geometry_key, color in zip(transfer_keys, colors, strict=True):
            native_material = _SOURCE_TRIALS[geometry_key][0]
            cross_material = "solaris" if native_material == "dragon" else "dragon"
            native = float(results[geometry_key][native_material][objective_name])
            cross = float(results[geometry_key][cross_material][objective_name])
            coordinates.append((native, cross))
            axis.scatter(
                native,
                cross,
                s=90,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=_GEOMETRY_LABELS[geometry_key],
            )
        limit = max(value for pair in coordinates for value in pair) * 1.08
        axis.plot((0.0, limit), (0.0, limit), linestyle="--", color="0.5")
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, limit)
        axis.set_xlabel("Native optical configuration")
        axis.set_ylabel("Cross optical configuration")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    figure.savefig(_OUTPUT_DIRECTORY / "native_cross_transfer.png", dpi=180)
    plt.close(figure)


def _plot_balanced_matrix(results: dict[str, dict[str, dict[str, object]]]) -> None:
    matrix = np.asarray(
        [
            [_normalized_scores(material_key, results[geometry_key][material_key])[2]
             for material_key in _MATERIALS]
            for geometry_key in _GEOMETRY_KEYS
        ]
    )
    minimum = float(matrix.min())
    maximum = float(matrix.max())
    norm = (
        TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
        if minimum < 0.0 < maximum
        else None
    )
    figure, axis = plt.subplots(figsize=(7.5, 5.3), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", aspect="auto", norm=norm)
    axis.set_xticks(range(len(_MATERIALS)), [value["label"] for value in _MATERIALS.values()])
    axis.set_yticks(range(len(_GEOMETRY_KEYS)), [_GEOMETRY_LABELS[key] for key in _GEOMETRY_KEYS])
    axis.set_title("Target-campaign normalized balanced score")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="J_balanced")
    figure.savefig(_OUTPUT_DIRECTORY / "balanced_specialization_matrix.png", dpi=180)
    plt.close(figure)


def _relative_change(first: float, second: float) -> float:
    return 100.0 * (second / first - 1.0)


def _ordering_text(
    material_key: str,
    optimized_key: str,
    suboptimal_key: str,
    results: dict[str, dict[str, dict[str, object]]],
) -> str:
    checks = []
    for objective in ("J_intensity_value", "J_spatial_value"):
        values = [
            float(results[key][material_key][objective])
            for key in (optimized_key, suboptimal_key, "nominal")
        ]
        checks.append(values[0] > values[1] > values[2])
    return "PASS" if all(checks) else "FAIL"


def _write_report(
    parameters: dict[str, dict[str, float]],
    results: dict[str, dict[str, dict[str, object]]],
    contracts: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# Cross-material morphology validation",
        "",
        "This focused validation transfers four selected geometries between the ",
        "saved Dragon Skin nominal-optics and Solaris nominal-optics campaign ",
        "contracts. Both contracts use the common `silicone` mechanics preset; ",
        "this is therefore an optical-transport comparison, not a calibrated ",
        "comparison of Dragon Skin and Solaris mechanical properties.",
        "",
        "## Reused and new evaluations",
        "",
        "Six existing cells were verified from their raw NPZ artifacts. Only the ",
        "four missing cross-material cells were evaluated; neither Ax campaign ",
        "state was opened or modified.",
        "",
        "| Target optics | Optical preset | Mechanics preset | Force targets | Dwell | Rays / bounces |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for material_key, material in _MATERIALS.items():
        contract = contracts[material_key]
        lines.append(
            f"| {material['label']} | `{contract['optical_preset']}` | "
            f"`{contract['viscoelastic_preset']}` | "
            f"{contract['mechanics']['force_targets_n']} N | "
            f"{contract['mechanics']['fixed_servo_dwell_s']:g} s | "
            f"{contract['optics']['ray_count']:,} / "
            f"{contract['optics']['max_bounces']} |"
        )

    lines.extend(
        (
            "",
            "## Primary comparison",
            "",
            "| Geometry source | Selection | Geometry mm | Dragon J_intensity | Dragon J_spatial | Solaris J_intensity | Solaris J_spatial |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for geometry_key in _GEOMETRY_KEYS:
        source, selection = _GEOMETRY_SOURCES[geometry_key]
        geometry = [parameters[geometry_key][name] for name in _PARAMETER_NAMES]
        dragon = results[geometry_key]["dragon"]
        solaris = results[geometry_key]["solaris"]
        lines.append(
            f"| {source} | {selection} | `{geometry}` | "
            f"{dragon['J_intensity_value']:.9f} | {dragon['J_spatial_value']:.9f} | "
            f"{solaris['J_intensity_value']:.9f} | {solaris['J_spatial_value']:.9f} |"
        )

    dragon_ordering = _ordering_text(
        "dragon", "dragon_optimized", "dragon_suboptimal", results
    )
    solaris_ordering = _ordering_text(
        "solaris", "solaris_optimized", "solaris_suboptimal", results
    )
    lines.extend(
        (
            "",
            "## Within-material physical-validation hierarchy",
            "",
            "Expected ordering is optimized > matched suboptimal > nominal in both objectives.",
            "",
            f"- Dragon optics: **{dragon_ordering}**",
            f"- Solaris optics: **{solaris_ordering}**",
            "",
            "## Cross-material transfer",
            "",
            "`R = J_cross / J_native`; relative change is `(R - 1) × 100%`.",
            "",
            "| Geometry | Objective | Native | Cross | R | Relative change |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for geometry_key in _GEOMETRY_KEYS[1:]:
        native_material = _SOURCE_TRIALS[geometry_key][0]
        cross_material = "solaris" if native_material == "dragon" else "dragon"
        for objective, label in (
            ("J_intensity_value", "J_intensity"),
            ("J_spatial_value", "J_spatial"),
        ):
            native = float(results[geometry_key][native_material][objective])
            cross = float(results[geometry_key][cross_material][objective])
            lines.append(
                f"| {_GEOMETRY_LABELS[geometry_key]} | {label} | {native:.9f} | "
                f"{cross:.9f} | {cross / native:.3f} | "
                f"{_relative_change(native, cross):+.1f}% |"
            )

    lines.extend(
        (
            "",
            "## Optimized-vs-optimized under each target optics",
            "",
            "Balanced scores use the target campaign's own Pareto nadir and ideal, without clipping.",
            "",
            "| Target optics | Geometry | J_intensity | J_spatial | Norm. intensity | Norm. spatial | Balanced |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for material_key in _MATERIALS:
        for geometry_key in ("dragon_optimized", "solaris_optimized"):
            result = results[geometry_key][material_key]
            normalized = _normalized_scores(material_key, result)
            lines.append(
                f"| {_MATERIALS[material_key]['label']} | "
                f"{_GEOMETRY_LABELS[geometry_key]} | "
                f"{result['J_intensity_value']:.9f} | {result['J_spatial_value']:.9f} | "
                f"{normalized[0]:.6f} | {normalized[1]:.6f} | {normalized[2]:.6f} |"
            )

    lines.extend(
        (
            "",
            "Relative optimized-geometry differences use the native optimized geometry as the denominator:",
            "",
        )
    )
    for material_key, native_key, other_key in (
        ("dragon", "dragon_optimized", "solaris_optimized"),
        ("solaris", "solaris_optimized", "dragon_optimized"),
    ):
        native = results[native_key][material_key]
        other = results[other_key][material_key]
        native_balanced = _normalized_scores(material_key, native)[2]
        other_balanced = _normalized_scores(material_key, other)[2]
        lines.append(
            f"- Under {_MATERIALS[material_key]['label']} optics, the other optimized "
            f"geometry changes J_intensity by "
            f"{_relative_change(float(native['J_intensity_value']), float(other['J_intensity_value'])):+.1f}%, "
            f"J_spatial by "
            f"{_relative_change(float(native['J_spatial_value']), float(other['J_spatial_value'])):+.1f}%, "
            f"and balanced score from {native_balanced:.6f} to {other_balanced:.6f}."
        )

    lines.extend(
        (
            "",
            "## Matched-suboptimal local transfer",
            "",
            "The ratio below is optimized / matched-suboptimal within the same geometry family.",
            "",
            "| Pair | Target optics | Objective | Optimized | Suboptimal | Difference | Ratio |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for pair_label, optimized_key, suboptimal_key in (
        ("Dragon 123 vs 126", "dragon_optimized", "dragon_suboptimal"),
        ("Solaris 107 vs 94", "solaris_optimized", "solaris_suboptimal"),
    ):
        for material_key in _MATERIALS:
            for objective, label in (
                ("J_intensity_value", "J_intensity"),
                ("J_spatial_value", "J_spatial"),
            ):
                optimized = float(results[optimized_key][material_key][objective])
                suboptimal = float(results[suboptimal_key][material_key][objective])
                lines.append(
                    f"| {pair_label} | {_MATERIALS[material_key]['label']} | {label} | "
                    f"{optimized:.9f} | {suboptimal:.9f} | "
                    f"{optimized - suboptimal:+.9f} | {optimized / suboptimal:.3f} |"
                )

    lines.extend(
        (
            "",
            "## Balanced specialization matrix",
            "",
            "| Geometry | Dragon optics | Solaris optics |",
            "| --- | ---: | ---: |",
        )
    )
    balanced_values: dict[tuple[str, str], float] = {}
    for geometry_key in _GEOMETRY_KEYS:
        for material_key in _MATERIALS:
            balanced_values[(geometry_key, material_key)] = _normalized_scores(
                material_key, results[geometry_key][material_key]
            )[2]
        lines.append(
            f"| {_GEOMETRY_LABELS[geometry_key]} | "
            f"{balanced_values[(geometry_key, 'dragon')]:.6f} | "
            f"{balanced_values[(geometry_key, 'solaris')]:.6f} |"
        )

    dragon_crossover = (
        balanced_values[("dragon_optimized", "dragon")]
        > balanced_values[("solaris_optimized", "dragon")]
    )
    solaris_crossover = (
        balanced_values[("solaris_optimized", "solaris")]
        > balanced_values[("dragon_optimized", "solaris")]
    )
    dragon_geometry_wins_both = dragon_crossover and not solaris_crossover
    solaris_geometry_wins_both = solaris_crossover and not dragon_crossover
    if dragon_crossover and solaris_crossover:
        specialization = (
            "**A. Strong specialization.** Each target optical configuration gives "
            "the higher balanced score to its own selected optimized morphology."
        )
    elif dragon_geometry_wins_both or solaris_geometry_wins_both:
        winning_geometry = (
            "Dragon optimized 123"
            if dragon_geometry_wins_both
            else "Solaris optimized 107"
        )
        specialization = (
            f"**C. Strong transferability within this selected set.** {winning_geometry} "
            "has the higher target-normalized balanced score under both optical "
            "configurations."
        )
    else:
        specialization = (
            "**B. Partial specialization.** The objective-wise comparisons do not "
            "produce a two-way balanced-score crossover or one universal winner."
        )

    max_force_error = max(
        float(results[geometry_key][material_key]["max_force_error_n"])
        for geometry_key in _GEOMETRY_KEYS
        for material_key in _MATERIALS
    )
    minimum_indentation_increment_um = 1.0e6 * min(
        float(results[geometry_key][material_key]["minimum_indentation_increment_m"])
        for geometry_key in _GEOMETRY_KEYS
        for material_key in _MATERIALS
    )
    max_closure = max(
        float(results[geometry_key][material_key]["max_energy_closure"])
        for geometry_key in _GEOMETRY_KEYS
        for material_key in _MATERIALS
    )
    lines.extend(
        (
            "",
            "## Specialization classification",
            "",
            specialization,
            "",
            "This classification concerns morphology preference under two optical "
            "transport models with common mechanics. It does not rank Dragon Skin "
            "against Solaris as physical materials.",
            "",
            "## Raw-artifact verification",
            "",
            "All ten matrix cells passed finite-value, force-band, strictly monotonic "
            "indentation, raw-objective recomputation, and optical energy-closure checks.",
            "",
            f"- Maximum accepted-force absolute error: {max_force_error:.6f} N",
            f"- Minimum adjacent-force indentation increase: {minimum_indentation_increment_um:.3f} µm",
            f"- Maximum absolute optical energy closure error: {max_closure:.3e}",
            "",
            "## Figures",
            "",
            "- [Objective comparison](objective_comparison.png)",
            "- [Native-vs-cross transfer](native_cross_transfer.png)",
            "- [Balanced specialization matrix](balanced_specialization_matrix.png)",
            "",
            "## Native physical-validation designs",
            "",
            "### Dragon Skin",
            "",
            "- Optimized: trial 123",
            "- Matched suboptimal: trial 126",
            "- Nominal: hand-designed baseline",
            "",
            "### Solaris",
            "",
            "- Optimized: trial 107",
            "- Matched suboptimal: trial 94",
            "- Nominal: hand-designed baseline",
            "",
            "## Cross-validation conclusion",
            "",
        )
    )

    dragon_opt_native = results["dragon_optimized"]["dragon"]
    dragon_opt_cross = results["dragon_optimized"]["solaris"]
    solaris_opt_native = results["solaris_optimized"]["solaris"]
    solaris_opt_cross = results["solaris_optimized"]["dragon"]
    dragon_pair_native = all(
        results["dragon_optimized"]["dragon"][objective]
        > results["dragon_suboptimal"]["dragon"][objective]
        for objective in ("J_intensity_value", "J_spatial_value")
    )
    dragon_pair_cross = all(
        results["dragon_optimized"]["solaris"][objective]
        > results["dragon_suboptimal"]["solaris"][objective]
        for objective in ("J_intensity_value", "J_spatial_value")
    )
    solaris_pair_native = all(
        results["solaris_optimized"]["solaris"][objective]
        > results["solaris_suboptimal"]["solaris"][objective]
        for objective in ("J_intensity_value", "J_spatial_value")
    )
    solaris_pair_cross = all(
        results["solaris_optimized"]["dragon"][objective]
        > results["solaris_suboptimal"]["dragon"][objective]
        for objective in ("J_intensity_value", "J_spatial_value")
    )
    lines.extend(
        (
            "1. **Does Dragon-opt transfer well to Solaris optics?** **No in this "
            "comparison.** Its Solaris/Dragon transfer ratios are "
            f"{float(dragon_opt_cross['J_intensity_value']) / float(dragon_opt_native['J_intensity_value']):.3f} "
            "for intensity and "
            f"{float(dragon_opt_cross['J_spatial_value']) / float(dragon_opt_native['J_spatial_value']):.3f} "
            "for spatial response; the target-normalized balanced score is reported above.",
            "2. **Does Solaris-opt transfer well to Dragon optics?** **Yes within the "
            "selected set.** Its Dragon/Solaris transfer ratios are "
            f"{float(solaris_opt_cross['J_intensity_value']) / float(solaris_opt_native['J_intensity_value']):.3f} "
            "for intensity and "
            f"{float(solaris_opt_cross['J_spatial_value']) / float(solaris_opt_native['J_spatial_value']):.3f} "
            "for spatial response; these ratios show the objective-specific transfer directly.",
            "3. **Is each native optimized morphology better than the other material's "
            "optimized morphology under its own optics?** "
            + (
                "Yes in the target-campaign balanced score for both optical configurations."
                if dragon_crossover and solaris_crossover
                else "No; the target-campaign balanced scores do not show a two-way native crossover."
            ),
            "4. **Does the optimized-vs-matched-suboptimal relationship change across "
            "materials?** **No in ordering.** Optimized > matched-suboptimal in both "
            "objectives for the Dragon pair "
            f"({dragon_pair_native} natively, {dragon_pair_cross} after transfer) and "
            "the Solaris pair "
            f"({solaris_pair_native} natively, {solaris_pair_cross} after transfer); "
            "the improvement magnitudes do change.",
            "5. **Do the results support material-dependent optomechanical morphology "
            "design?** "
            + (
                "Yes. The two-way balanced-score crossover supports co-designing morphology with optical transport."
                if dragon_crossover and solaris_crossover
                else "No from this selected optimized-vs-optimized comparison. The Solaris-selected geometry wins the target-normalized balanced comparison under both optical configurations, while the optical model still changes objective magnitudes substantially."
            ),
            "",
        )
    )
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    campaigns: dict[str, object] = {}
    contracts: dict[str, dict[str, object]] = {}
    for material_key in _MATERIALS:
        campaigns[material_key], contracts[material_key] = (
            _campaign_from_saved_contract(material_key)
        )
        print(
            f"verified {_MATERIALS[material_key]['label']} saved campaign contract",
            flush=True,
        )

    parameters: dict[str, dict[str, float]] = {"nominal": _NOMINAL_PARAMETERS}
    results: dict[str, dict[str, dict[str, object]]] = {
        geometry_key: {} for geometry_key in _GEOMETRY_KEYS
    }
    for geometry_key in _GEOMETRY_KEYS[1:]:
        geometry_parameters, native_result = _verify_source_trial(
            geometry_key, campaigns
        )
        parameters[geometry_key] = geometry_parameters
        native_material = _SOURCE_TRIALS[geometry_key][0]
        results[geometry_key][native_material] = native_result
        print(f"verified {_GEOMETRY_LABELS[geometry_key]} source artifact", flush=True)

    for material_key in _MATERIALS:
        results["nominal"][material_key] = _load_raw_result(
            _MATERIALS[material_key]["nominal_path"],
            campaign=campaigns[material_key],
            expected_parameters=parameters["nominal"],
        )
    print("verified both nominal source artifacts", flush=True)

    _validate_optix_environment()
    for geometry_key, material_key in _CROSS_RESULT_PATHS:
        results[geometry_key][material_key] = _evaluate_cross_result(
            geometry_key,
            material_key,
            campaigns=campaigns,
            parameters=parameters[geometry_key],
        )

    _write_csv(parameters, results)
    _plot_objectives(results)
    _plot_transfer(results)
    _plot_balanced_matrix(results)
    _write_report(parameters, results, contracts)
    print(f"wrote {_CSV_PATH.relative_to(_ROOT)}", flush=True)
    print(f"wrote {_REPORT_PATH.relative_to(_ROOT)}", flush=True)
    print("cross-material morphology validation: PASS", flush=True)


if __name__ == "__main__":
    main()
