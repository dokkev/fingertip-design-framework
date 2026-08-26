"""Persistence and provenance for the current Ax campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ax.api.client import Client

from .design_space import (
    MAX_FINGERTIP_HEIGHT_MM,
    MINIMUM_SILICONE_THICKNESS_MM,
)

if TYPE_CHECKING:
    from .ax_bo import CampaignDefinition


RUN_CONFIG_FILENAME = "run_config.json"
AX_STATE_FILENAME = "ax_state.json"
TRIALS_FILENAME = "trials.csv"
PARETO_FILENAME = "pareto.csv"
SUMMARY_FILENAME = "run_summary.json"
TRIAL_RESULT_DIRECTORY = "trials"

_OBJECTIVE_NAMES = ("J_contact", "J_obs")
_RUN_CONFIG_SCHEMA = 11
_OBJECTIVE_DEFINITION = "fingertip-contact-and-threshold-conditioned-observation-v2"


def _fieldnames(campaign: CampaignDefinition) -> list[str]:
    return [
        "ax_trial_index",
        "design",
        "generation_node",
        "status",
        "analytically_valid",
        *campaign.parameter_columns,
        *_OBJECTIVE_NAMES,
        "limiting_contact_scenario",
        "limiting_obs_sphere_diameter_mm",
        "limiting_obs_force_n",
        "limiting_obs_contact_y_pair_mm",
        "d_onset_diagnostic",
        "max_outside_roi_power_fraction",
        "runtime_s",
        "raw_result_path",
        "failure",
        "is_pareto",
    ]


def _empty_result_fields() -> dict[str, object]:
    return {
        **{name: "" for name in _OBJECTIVE_NAMES},
        "limiting_contact_scenario": "",
        "limiting_obs_sphere_diameter_mm": "",
        "limiting_obs_force_n": "",
        "limiting_obs_contact_y_pair_mm": "",
        "d_onset_diagnostic": "",
        "max_outside_roi_power_fraction": "",
        "runtime_s": "",
        "raw_result_path": "",
        "failure": "",
        "is_pareto": False,
    }


def running_row(
    trial_index: int,
    ax_parameters: Mapping[str, object],
    parameters: Mapping[str, object],
    generation_node: str,
) -> dict[str, object]:
    """Return the persisted row created before an expensive evaluation."""
    return {
        "ax_trial_index": trial_index,
        "design": f"bo_{trial_index:04d}",
        "generation_node": generation_node,
        "status": "RUNNING",
        "analytically_valid": True,
        **ax_parameters,
        **parameters,
        **_empty_result_fields(),
    }


def _read_trials(
    path: Path,
    campaign: CampaignDefinition,
) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = []
        for raw_row in csv.DictReader(input_file):
            row: dict[str, object] = dict(raw_row)
            row["ax_trial_index"] = int(raw_row["ax_trial_index"])
            row["analytically_valid"] = raw_row["analytically_valid"] == "True"
            row["is_pareto"] = raw_row["is_pareto"] == "True"
            for name in campaign.ax_parameter_names:
                value = raw_row.get(name, "")
                row[name] = int(float(value)) if value else ""
            for name in (
                *campaign.physical_parameter_names,
                *_OBJECTIVE_NAMES,
                "limiting_obs_sphere_diameter_mm",
                "limiting_obs_force_n",
                "d_onset_diagnostic",
                "max_outside_roi_power_fraction",
                "runtime_s",
            ):
                value = raw_row.get(name, "")
                row[name] = float(value) if value else ""
            rows.append(row)
    return rows


def _update_pareto_status(rows: list[dict[str, object]]) -> None:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    for row in rows:
        row["is_pareto"] = False
    for candidate in completed:
        objectives = np.asarray(
            [candidate[name] for name in _OBJECTIVE_NAMES], dtype=np.float64
        )
        candidate["is_pareto"] = not any(
            np.all(np.asarray([other[name] for name in _OBJECTIVE_NAMES]) >= objectives)
            and np.any(
                np.asarray([other[name] for name in _OBJECTIVE_NAMES]) > objectives
            )
            for other in completed
            if other is not candidate
        )


def _flush_file(output_file: object) -> None:
    output_file.flush()
    os.fsync(output_file.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def save_ax(client: Client, path: Path) -> None:
    """Atomically persist Ax state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    client.save_to_json_file(filepath=str(temporary))
    with temporary.open("rb") as input_file:
        os.fsync(input_file.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def write_tables(
    output_directory: Path,
    rows: list[dict[str, object]],
    campaign: CampaignDefinition,
) -> None:
    """Atomically write the complete and Pareto trial tables."""
    _update_pareto_status(rows)
    fields = _fieldnames(campaign)
    _write_csv(output_directory / TRIALS_FILENAME, rows, fields)
    _write_csv(
        output_directory / PARETO_FILENAME,
        [row for row in rows if row["status"] == "COMPLETED" and row["is_pareto"]],
        fields,
    )


def persist_campaign(
    client: Client,
    rows: list[dict[str, object]],
    output_directory: Path,
    campaign: CampaignDefinition,
) -> None:
    """Persist Ax state followed by human-readable tables."""
    save_ax(client, output_directory / AX_STATE_FILENAME)
    write_tables(output_directory, rows, campaign)


def _sha256_files(paths: tuple[Path, ...], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _scientific_source_sha256(repository_root: Path) -> str:
    excluded = {
        "lumo/optimization/ax_bo.py",
        "lumo/optimization/campaign_io.py",
    }
    paths = tuple(
        path
        for path in sorted((repository_root / "lumo").rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".cu", ".urdf"}
        and path.relative_to(repository_root).as_posix() not in excluded
    )
    return _sha256_files(paths, repository_root)


def _optimizer_source_sha256(repository_root: Path) -> str:
    return _sha256_files(
        (
            repository_root / "lumo" / "optimization" / "ax_bo.py",
            Path(__file__).resolve(),
        ),
        repository_root,
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _git_output(repository_root: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_run_config(campaign: CampaignDefinition) -> dict[str, object]:
    """Serialize the scientific and optimizer contract for strict resume."""
    from lumo.fingertip import (
        ACTIVE_Y_BOUNDS_MM,
        LED_CENTERS_Y_MM,
        LED_RECESS_DEPTH_MM,
        LED_RECESS_WIDTH_MM,
    )
    from lumo.ray_tracing import LONGITUDINAL_SIDE_BIN_COUNT

    from . import evaluator

    repository_root = Path(__file__).resolve().parents[2]
    parameter_bounds = {
        f"geometry.{name}": [bound.lower, bound.upper]
        for name, bound in campaign.space.geometry_bounds.items()
    }
    return {
        "schema_version": _RUN_CONFIG_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_revision": _git_output(repository_root, ["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_output(repository_root, ["status", "--porcelain"])),
            "scientific_source_sha256": _scientific_source_sha256(repository_root),
            "optimizer_source_sha256": _optimizer_source_sha256(repository_root),
            "versions": {
                "ax-platform": _package_version("ax-platform"),
                "botorch": _package_version("botorch"),
                "newton": _package_version("newton"),
                "warp-lang": _package_version("warp-lang"),
                "numpy": _package_version("numpy"),
            },
        },
        "scientific_contract": {
            "mechanics_preset": campaign.mechanics_preset,
            "optical_preset": campaign.optical_preset,
            "fingertip_parameters": asdict(campaign.space.base_parameters),
            "mechanics": {
                "loading_protocol": "constant_speed_force_thresholds",
                "capture_rule": "first reaction-force sample >= threshold",
                "sim_frequency_hz": evaluator._SIM_FREQUENCY_HZ,
                "vbd_iterations": evaluator._VBD_ITERATIONS,
                "approach_speed_m_s": campaign.approach_speed_m_s,
                "displacement_m_tick": (
                    campaign.approach_speed_m_s / evaluator._SIM_FREQUENCY_HZ
                ),
                "force_targets_n": list(campaign.force_targets_n),
                "max_sim_time_s": campaign.max_sim_time_s,
                "element_size_mm": evaluator._ELEMENT_SIZE_MM,
                "soft_contact_margin_m": evaluator._SOFT_CONTACT_MARGIN_M,
                "carrier_contact_stiffness_n_m": (
                    evaluator._CARRIER_CONTACT_STIFFNESS_N_M
                ),
                "indenter_contact_stiffness_n_m": (evaluator._CONTACT_STIFFNESS_N_M),
                "indenter_contact_damping_n_s_m": (evaluator._CONTACT_DAMPING_N_S_M),
            },
            "scenarios": {
                "indenter_urdfs": [filename for filename, _ in campaign.indenters],
                "indenter_names": list(campaign.indenter_names),
                "sphere_diameters_mm": list(campaign.sphere_diameters_mm),
                "initial_clearance_m": campaign.initial_clearance_m,
                "contact_x_mm": 0.0,
                "contact_y_mm": list(campaign.contact_y_mm),
            },
            "optics": {
                "sample_side_count": evaluator._SAMPLE_SIDE_COUNT,
                "ray_count": evaluator._SAMPLE_SIDE_COUNT**2,
                "max_bounces": evaluator._MAX_BOUNCES,
                "deterministic_seed": evaluator._RNG_SEED,
                "source_seed": evaluator._SOURCE_RNG_SEED,
                "source_model": "uniform_finite_package_window",
                "source_window_mm": [
                    campaign.space.base_parameters.led.emitting_window_x_mm,
                    campaign.space.base_parameters.led.emitting_window_y_mm,
                ],
                "carrier_albedo": evaluator._CARRIER_ALBEDO,
                "source_medium": ("resolved per geometry from LED air-gap boundary"),
                "led_centers_y_mm": list(LED_CENTERS_Y_MM),
                "led_count": len(LED_CENTERS_Y_MM),
                "led_recess_width_mm": LED_RECESS_WIDTH_MM,
                "led_recess_depth_mm": LED_RECESS_DEPTH_MM,
                "observation_view_direction": "+X",
                "longitudinal_coordinate": "Y",
                "spatial_roi_y_mm": list(ACTIVE_Y_BOUNDS_MM),
                "spatial_bin_count": LONGITUDINAL_SIDE_BIN_COUNT,
                "spatial_bin_width_mm": (
                    (ACTIVE_Y_BOUNDS_MM[1] - ACTIVE_Y_BOUNDS_MM[0])
                    / LONGITUDINAL_SIDE_BIN_COUNT
                ),
                "simultaneous_emitted_power": float(len(LED_CENTERS_Y_MM)),
            },
            "design_space": {
                "representation": "integer_half_millimeter_steps",
                "resolution_mm": campaign.resolution_mm,
                "fixed": {
                    "geometry.flat_pad_width_mm": (
                        campaign.space.base_parameters.geometry.flat_pad_width_mm
                    )
                },
                "integer_step_bounds": {
                    name: [int(parameter.bounds[0]), int(parameter.bounds[1])]
                    for name, parameter in zip(
                        campaign.ax_parameter_names,
                        campaign.ax_parameters,
                        strict=True,
                    )
                },
                "decoded_physical_bounds_mm": parameter_bounds,
                "ax_linear_constraints": list(campaign.ax_parameter_constraints),
                "candidate_generation": "exact_feasible_sobol_candidate_pool",
                "acquisition_pool_size": campaign.acquisition_pool_size,
                "minimum_silicone_thickness_mm": (MINIMUM_SILICONE_THICKNESS_MM),
                "full_fingertip_height_max_mm": MAX_FINGERTIP_HEIGHT_MM,
                "full_height_relation_mm": (
                    "link_thickness + flat_pad_height + semiellipse_height"
                ),
                "fixed_link_thickness_mm": (
                    campaign.space.base_parameters.geometry.link_thickness_mm
                ),
            },
            "objectives": {
                "names": list(_OBJECTIVE_NAMES),
                "directions": ["maximize", "maximize"],
                "definition": _OBJECTIVE_DEFINITION,
                "J_contact": (
                    "min over diameter/location scenarios of "
                    "cuberoot(q_form*q_stable*q_stiff)"
                ),
                "J_obs": (
                    "min over diameter, force threshold, and distinct contact-Y "
                    "pairs of L2((y-y0)/P_emit); d_onset is diagnostic only"
                ),
            },
            "ax_random_seed": campaign.random_seed,
        },
    }


def _validate_run_config(
    stored: dict[str, object],
    current: dict[str, object],
) -> None:
    if stored.get("schema_version") != _RUN_CONFIG_SCHEMA:
        raise RuntimeError("stored run_config.json has an unsupported schema")
    if stored.get("scientific_contract") != current.get("scientific_contract"):
        raise RuntimeError(
            "current scientific settings differ from run_config.json; "
            "refusing to mix incompatible evaluations"
        )
    stored_provenance = stored.get("provenance")
    current_provenance = current.get("provenance")
    if not isinstance(stored_provenance, dict) or not isinstance(
        current_provenance, dict
    ):
        raise RuntimeError("run_config.json has invalid provenance")
    for field, message in (
        (
            "scientific_source_sha256",
            "LUMO scientific source differs from the saved run contract",
        ),
        (
            "optimizer_source_sha256",
            "Ax campaign code differs from the saved run contract",
        ),
        ("versions", "dependency versions differ from the saved run contract"),
    ):
        if stored_provenance.get(field) != current_provenance.get(field):
            raise RuntimeError(message + "; resume with the original source snapshot")


def trial_result_path(output_directory: Path, trial_index: int) -> Path:
    return output_directory / TRIAL_RESULT_DIRECTORY / f"trial_{trial_index:04d}.npz"


def save_trial_result(
    path: Path,
    *,
    campaign: CampaignDefinition,
    evaluation: object,
    details: dict[str, object],
    parameters: dict[str, float],
    runtime_s: float,
) -> None:
    """Atomically save raw scientific state and derived objective diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    contact = details["contact"]
    observation = details["observation"]
    with temporary.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            reference_vertices_m=np.asarray(evaluation.reference_vertices_m),
            tet_indices=np.asarray(evaluation.tet_indices),
            surface_triangles=np.asarray(evaluation.surface_triangles),
            bonded_vertex_indices=np.asarray(evaluation.bonded_vertex_indices),
            led_centers_m=np.asarray(evaluation.led_centers_m),
            no_contact_response=np.asarray(evaluation.no_contact_response),
            no_contact_energy=np.asarray(evaluation.no_contact_energy),
            no_contact_inside_roi_power=np.asarray(
                evaluation.no_contact_inside_roi_power
            ),
            no_contact_outside_roi_power=np.asarray(
                evaluation.no_contact_outside_roi_power
            ),
            no_contact_visible_side_power=np.asarray(
                evaluation.no_contact_visible_side_power
            ),
            no_contact_outside_roi_power_fraction=np.asarray(
                evaluation.no_contact_outside_roi_power_fraction
            ),
            response_matrix=np.asarray(evaluation.response_matrix),
            energy_matrix=np.asarray(evaluation.energy_matrix),
            energy_fields=np.asarray(evaluation.energy_fields),
            inside_roi_power=np.asarray(evaluation.inside_roi_power),
            outside_roi_power=np.asarray(evaluation.outside_roi_power),
            visible_side_power=np.asarray(evaluation.visible_side_power),
            outside_roi_power_fraction=np.asarray(
                evaluation.outside_roi_power_fraction
            ),
            actual_forces_n=np.asarray(evaluation.actual_forces_n),
            indentations_m=np.asarray(evaluation.indentations_m),
            checkpoint_steps=np.asarray(evaluation.checkpoint_steps),
            checkpoint_times_s=np.asarray(evaluation.checkpoint_times_s),
            maximum_particle_speeds_m_s=np.asarray(
                evaluation.maximum_particle_speeds_m_s
            ),
            mean_particle_speeds_m_s=np.asarray(evaluation.mean_particle_speeds_m_s),
            rms_particle_speeds_m_s=np.asarray(evaluation.rms_particle_speeds_m_s),
            particle_speed_p95_m_s=np.asarray(evaluation.particle_speed_p95_m_s),
            kinetic_energy_j=np.asarray(evaluation.kinetic_energy_j),
            force_overshoots_n=np.asarray(evaluation.force_overshoots_n),
            reaction_force_rates_n_s=np.asarray(evaluation.reaction_force_rates_n_s),
            indentation_rates_m_s=np.asarray(evaluation.indentation_rates_m_s),
            indenter_contact_counts=np.asarray(evaluation.indenter_contact_counts),
            total_contact_counts=np.asarray(evaluation.total_contact_counts),
            contact_buffer_overflow=np.asarray(evaluation.contact_buffer_overflow),
            minimum_det_f=np.asarray(evaluation.minimum_det_f),
            inverted_tet_counts=np.asarray(evaluation.inverted_tet_counts),
            contact_centroids_W_m=np.asarray(evaluation.contact_centroids_W_m),
            contact_record_offsets=np.asarray(evaluation.contact_record_offsets),
            contact_particle_indices=np.asarray(evaluation.contact_particle_indices),
            contact_barycentric=np.asarray(evaluation.contact_barycentric),
            contact_positions_W_m=np.asarray(evaluation.contact_positions_W_m),
            contact_normals_W=np.asarray(evaluation.contact_normals_W),
            contact_body_positions=np.asarray(evaluation.contact_body_positions),
            silicone_vertices_m=np.asarray(evaluation.silicone_vertices_m),
            scenario_runtime_s=np.asarray(evaluation.scenario_runtime_s),
            checkpoint_optics_runtime_s=np.asarray(
                evaluation.checkpoint_optics_runtime_s
            ),
            no_contact_optics_runtime_s=np.asarray(
                evaluation.no_contact_optics_runtime_s
            ),
            scenario_names=np.asarray(evaluation.scenario_names),
            sphere_diameters_mm=np.asarray(evaluation.sphere_diameters_mm),
            contact_y_mm=np.asarray(evaluation.contact_y_mm),
            force_targets_n=np.asarray(evaluation.force_targets_n),
            J_contact=np.asarray(details["J_contact"]),
            limiting_contact_scenario=np.asarray(contact.limiting_scenario),
            q_form=np.asarray(contact.q_form),
            q_stable=np.asarray(contact.q_stable),
            q_stiff=np.asarray(contact.q_stiff),
            q_contact=np.asarray(contact.q_contact),
            q_normal_diagnostic=np.asarray(contact.q_normal),
            patch_area_5_m2=np.asarray(contact.patch_area_5_m2),
            k_early_n_m=np.asarray(contact.k_early_n_m),
            k_late_n_m=np.asarray(contact.k_late_n_m),
            J_obs=np.asarray(details["J_obs"]),
            limiting_obs_sphere_diameter_mm=np.asarray(
                observation.limiting_sphere_diameter_mm
            ),
            limiting_obs_force_n=np.asarray(observation.limiting_force_n),
            limiting_obs_contact_y_pair_mm=np.asarray(
                observation.limiting_contact_y_pair_mm
            ),
            d_onset_diagnostic=np.asarray(observation.d_onset),
            normalized_observation=np.asarray(observation.normalized_response),
            observation_sphere_diameters_mm=np.asarray(observation.sphere_diameters_mm),
            observation_contact_y_mm=np.asarray(observation.contact_y_mm),
            same_force_location_separations=np.asarray(
                observation.location_separations
            ),
            evaluation_runtime_s=np.asarray(runtime_s),
            parameter_names=np.asarray(campaign.physical_parameter_names),
            parameter_values=np.asarray(
                [parameters[name] for name in campaign.physical_parameter_names]
            ),
        )
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def apply_result_to_row(
    row: dict[str, object],
    details: dict[str, object],
    *,
    runtime_s: float,
    raw_result_path: str,
) -> None:
    """Copy compact objective diagnostics into one CSV row."""
    contact = details["contact"]
    observation = details["observation"]
    row.update(
        J_contact=float(details["J_contact"]),
        J_obs=float(details["J_obs"]),
        limiting_contact_scenario=contact.limiting_scenario,
        limiting_obs_sphere_diameter_mm=(observation.limiting_sphere_diameter_mm),
        limiting_obs_force_n=observation.limiting_force_n,
        limiting_obs_contact_y_pair_mm=(
            f"{observation.limiting_contact_y_pair_mm[0]:g},"
            f"{observation.limiting_contact_y_pair_mm[1]:g}"
        ),
        d_onset_diagnostic=observation.d_onset,
        max_outside_roi_power_fraction=details["max_outside_roi_power_fraction"],
        runtime_s=runtime_s,
        raw_result_path=raw_result_path,
        failure="",
    )


def ax_statuses(client: Client) -> dict[int, str]:
    summary = client.summarize()
    return {
        int(row["trial_index"]): str(row["trial_status"]).upper().split(".")[-1]
        for _, row in summary.iterrows()
    }


def _reconcile_resume(
    client: Client,
    rows: list[dict[str, object]],
    output_directory: Path,
    campaign: CampaignDefinition,
) -> None:
    """Reconcile crash windows between Ax, CSV, and raw NPZ state."""
    summary = client.summarize()
    statuses = ax_statuses(client)
    row_indices = {int(row["ax_trial_index"]) for row in rows}
    ax_changed = False

    for _, summary_row in summary.iterrows():
        trial_index = int(summary_row["trial_index"])
        if trial_index in row_indices:
            continue
        if statuses[trial_index] != "RUNNING":
            raise RuntimeError(
                f"Ax trial {trial_index} is absent from trials.csv with status "
                f"{statuses[trial_index]}"
            )
        raw_parameters = {
            name: summary_row[name] for name in campaign.ax_parameter_names
        }
        parameters = campaign.decode(raw_parameters)
        campaign.validate(raw_parameters, parameters)
        row = running_row(
            trial_index,
            raw_parameters,
            parameters,
            str(summary_row.get("generation_node", "")),
        )
        row["runtime_s"] = 0.0
        row["failure"] = "interrupted before proposal CSV persistence"
        client.mark_trial_abandoned(trial_index)
        row["status"] = "ABANDONED"
        rows.append(row)
        ax_changed = True

    unexpected = {int(row["ax_trial_index"]) for row in rows} - set(statuses)
    if unexpected:
        raise RuntimeError(
            f"trials.csv contains trials absent from Ax: {sorted(unexpected)}"
        )

    statuses = ax_statuses(client)
    for row in rows:
        trial_index = int(row["ax_trial_index"])
        csv_status = str(row["status"])
        ax_status = statuses[trial_index]
        if csv_status == "EVALUATED":
            raw_path = output_directory / str(row["raw_result_path"])
            if not raw_path.is_file():
                raise RuntimeError(
                    f"evaluated trial {trial_index} has no raw-result NPZ"
                )
            if ax_status == "RUNNING":
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={name: float(row[name]) for name in _OBJECTIVE_NAMES},
                )
                ax_changed = True
            elif ax_status != "COMPLETED":
                raise RuntimeError(
                    f"evaluated trial {trial_index} has Ax status {ax_status}"
                )
            row["status"] = "COMPLETED"
        elif csv_status == "RUNNING":
            if ax_status != "RUNNING":
                raise RuntimeError(
                    f"running CSV trial {trial_index} has Ax status {ax_status}"
                )
            client.mark_trial_abandoned(trial_index)
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = "interrupted during morphology evaluation"
            ax_changed = True
        elif csv_status == "FAILED" and ax_status == "ABANDONED":
            continue
        elif csv_status != ax_status:
            raise RuntimeError(
                f"trial {trial_index} status mismatch: CSV={csv_status}, Ax={ax_status}"
            )

    if ax_changed:
        save_ax(client, output_directory / AX_STATE_FILENAME)
    write_tables(output_directory, rows, campaign)


def completed_trial_count(rows: list[dict[str, object]]) -> int:
    return sum(row["status"] == "COMPLETED" for row in rows)


def initialize_campaign(
    output_directory: Path,
    campaign: CampaignDefinition,
    client: Client,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create an empty campaign without reusing historical observations."""
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise FileExistsError(f"fresh campaign output is not empty: {output_directory}")
    config = build_run_config(campaign)
    _write_json(output_directory / RUN_CONFIG_FILENAME, config)
    rows: list[dict[str, object]] = []
    persist_campaign(client, rows, output_directory, campaign)
    print("created campaign with no reused objective observations", flush=True)
    return rows, config


def resume_campaign(
    output_directory: Path,
    campaign: CampaignDefinition,
) -> tuple[Client, list[dict[str, object]], dict[str, object]]:
    """Load and reconcile one interrupted campaign."""
    required = (
        output_directory / RUN_CONFIG_FILENAME,
        output_directory / AX_STATE_FILENAME,
        output_directory / TRIALS_FILENAME,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(
            "campaign directory is incomplete; missing " + ", ".join(missing)
        )
    with required[0].open(encoding="utf-8") as input_file:
        stored_config = json.load(input_file)
    _validate_run_config(stored_config, build_run_config(campaign))
    client = Client.load_from_json_file(filepath=str(required[1]))
    rows = _read_trials(required[2], campaign)
    _reconcile_resume(client, rows, output_directory, campaign)
    print(
        f"resumed campaign with {completed_trial_count(rows)} completed BO trials",
        flush=True,
    )
    return client, rows, stored_config


def _write_plots(output_directory: Path, rows: list[dict[str, object]]) -> None:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    if not completed:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contact = np.asarray([row["J_contact"] for row in completed], dtype=np.float64)
    observation = np.asarray([row["J_obs"] for row in completed], dtype=np.float64)
    pareto = np.asarray([bool(row["is_pareto"]) for row in completed])
    figure, axes = plt.subplots(figsize=(7.0, 5.5))
    axes.scatter(contact[~pareto], observation[~pareto], label="completed")
    if np.any(pareto):
        axes.scatter(contact[pareto], observation[pareto], label="Pareto")
    axes.set_xlabel("J_contact")
    axes.set_ylabel("J_obs")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "objectives.png", dpi=180)
    plt.close(figure)


def finalize_outputs(
    output_directory: Path,
    rows: list[dict[str, object]],
    config: dict[str, object],
    campaign: CampaignDefinition,
    *,
    command_wall_runtime_s: float,
) -> dict[str, object]:
    """Write final tables, plot, and compact campaign summary."""
    write_tables(output_directory, rows, campaign)
    _write_plots(output_directory, rows)
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    best_contact = (
        max(completed, key=lambda row: float(row["J_contact"])) if completed else None
    )
    best_observation = (
        max(completed, key=lambda row: float(row["J_obs"])) if completed else None
    )
    previous_active_runtime_s = 0.0
    summary_path = output_directory / SUMMARY_FILENAME
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as input_file:
            previous_active_runtime_s = float(
                json.load(input_file).get("active_wall_runtime_s", 0.0)
            )
    now = datetime.now(UTC)
    summary = {
        "updated_utc": now.isoformat(),
        "campaign_elapsed_wall_runtime_s": (
            now - datetime.fromisoformat(str(config["created_utc"]))
        ).total_seconds(),
        "active_wall_runtime_s": previous_active_runtime_s + command_wall_runtime_s,
        "total_evaluation_runtime_s": sum(
            float(row["runtime_s"])
            for row in rows
            if row["status"] in {"COMPLETED", "FAILED"} and row["runtime_s"] != ""
        ),
        "counts": {
            "bo_completed": completed_trial_count(rows),
            "bo_failed": sum(row["status"] == "FAILED" for row in rows),
            "bo_abandoned": sum(row["status"] == "ABANDONED" for row in rows),
            "pareto": sum(
                row["status"] == "COMPLETED" and row["is_pareto"] for row in rows
            ),
        },
        "best_J_contact": (
            {
                "design": best_contact["design"],
                "ax_trial_index": best_contact["ax_trial_index"],
                "value": best_contact["J_contact"],
            }
            if best_contact is not None
            else None
        ),
        "best_J_obs": (
            {
                "design": best_observation["design"],
                "ax_trial_index": best_observation["ax_trial_index"],
                "value": best_observation["J_obs"],
            }
            if best_observation is not None
            else None
        ),
    }
    _write_json(summary_path, summary)
    return summary


__all__ = [
    "AX_STATE_FILENAME",
    "RUN_CONFIG_FILENAME",
    "TRIALS_FILENAME",
    "apply_result_to_row",
    "ax_statuses",
    "build_run_config",
    "completed_trial_count",
    "finalize_outputs",
    "initialize_campaign",
    "persist_campaign",
    "resume_campaign",
    "running_row",
    "save_ax",
    "save_trial_result",
    "trial_result_path",
    "write_tables",
]
