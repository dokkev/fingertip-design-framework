"""Sequential Ax optimization of the current LUMO sensing objectives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from lumo.fingertip import FingertipParameters

from .design_param_bound import DesignParameterBounds, ParameterBound
from .design_space import DesignSpace, LinearConstraint


_PARAMETER_NAMES = (
    "geometry.flat_pad_width_mm",
    "geometry.flat_pad_height_mm",
    "geometry.semiellipse_height_mm",
    "geometry.stem_width_mm",
    "geometry.void_width_mm",
    "geometry.void_height_mm",
)
_OBJECTIVE_NAMES = ("J_intensity", "J_spatial")
_SPHERE_DIAMETERS_MM = (5.0, 10.0, 20.0)
_SPHERES = tuple(
    (f"sphere_{diameter_mm:g}mm.urdf", diameter_mm)
    for diameter_mm in _SPHERE_DIAMETERS_MM
)
_PER_DIAMETER_FIELDS = tuple(
    field
    for diameter_mm in _SPHERE_DIAMETERS_MM
    for field in (
        f"J_intensity_{diameter_mm:g}mm",
        f"J_spatial_{diameter_mm:g}mm",
    )
)
_WARM_START_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "sensing_objective_tradeoff.csv"
)
_DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "optimization"
    / "mobo"
)
_RUN_CONFIG_FILENAME = "run_config.json"
_AX_STATE_FILENAME = "ax_state.json"
_TRIALS_FILENAME = "trials.csv"
_PARETO_FILENAME = "pareto.csv"
_SUMMARY_FILENAME = "run_summary.json"
_TRIAL_RESULT_DIRECTORY = "trials"
_RANDOM_SEED = 20260823
_WARM_START_COUNT = 13
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_MAX_PROPOSALS_PER_COMPLETED_TRIAL = 50
_OBJECTIVE_DEFINITION = "diameter-wise-force-pair-min-v1"
_RUN_CONFIG_SCHEMA = 1


def _design_space() -> DesignSpace:
    bounds = DesignParameterBounds(
        parameters=FingertipParameters(),
        geometry={
            "flat_pad_width_mm": ParameterBound(25.0, 35.0),
            "flat_pad_height_mm": ParameterBound(3.0, 8.0),
            "semiellipse_height_mm": ParameterBound(6.0, 20.0),
            "stem_width_mm": ParameterBound(7.0, 10.0),
            "void_width_mm": ParameterBound(0.0, 3.0),
            "void_height_mm": ParameterBound(0.0, 3.0),
        },
    )
    return DesignSpace(
        parameter_bounds=bounds,
        linear_constraints=(
            LinearConstraint(
                coefficients={
                    "geometry.flat_pad_height_mm": 1.0,
                    "geometry.semiellipse_height_mm": 1.0,
                },
                upper=30.0,
            ),
        ),
        minimum_silicone_thickness_mm=5.0,
    )


def _read_warm_start(space: DesignSpace) -> list[dict[str, object]]:
    if not _WARM_START_PATH.is_file():
        raise FileNotFoundError(_WARM_START_PATH)

    rows: list[dict[str, object]] = []
    with _WARM_START_PATH.open(newline="", encoding="utf-8") as input_file:
        for raw_row in csv.DictReader(input_file):
            if raw_row["status"] != "PASS":
                continue
            parameters = {
                name: float(raw_row[name]) for name in space.variable_names
            }
            objectives = {
                name: float(raw_row[name]) for name in _OBJECTIVE_NAMES
            }
            if not space.is_feasible(parameters):
                raise ValueError(
                    f"warm-start design {raw_row['design']!r} is analytically invalid"
                )
            if not all(isfinite(value) for value in objectives.values()):
                raise ValueError(
                    f"warm-start design {raw_row['design']!r} has invalid objectives"
                )
            rows.append(
                {
                    "design": raw_row["design"],
                    "parameters": parameters,
                    "objectives": objectives,
                    "per_diameter": {
                        field: float(raw_row[field])
                        for field in _PER_DIAMETER_FIELDS
                    },
                    "worst_intensity_diameter_mm": raw_row[
                        "worst_intensity_diameter_mm"
                    ],
                    "worst_intensity_force_pair_n": raw_row[
                        "worst_intensity_force_pair_n"
                    ],
                    "worst_spatial_diameter_mm": raw_row[
                        "worst_spatial_diameter_mm"
                    ],
                    "worst_spatial_force_pair_n": raw_row[
                        "worst_spatial_force_pair_n"
                    ],
                    "runtime_s": float(raw_row["runtime_s"]),
                }
            )

    if len(rows) != _WARM_START_COUNT:
        raise ValueError(
            f"expected {_WARM_START_COUNT} completed warm-start designs, "
            f"found {len(rows)}"
        )
    if len({str(row["design"]) for row in rows}) != len(rows):
        raise ValueError("warm-start design names must be unique")
    return rows


def _new_client(space: DesignSpace) -> Client:
    client = Client(random_seed=_RANDOM_SEED)
    parameters = []
    for name in space.variable_names:
        bound = space.parameter_bounds.geometry[
            name.removeprefix("geometry.")
        ]
        parameters.append(
            RangeParameterConfig(
                name=name,
                bounds=(bound.lower, bound.upper),
                parameter_type="float",
            )
        )
    client.configure_experiment(
        name="lumo_center_contact_sensing",
        description=(
            "Center-contact LUMO sensing optimization over 5/10/20 mm spheres "
            "and 5/10/15/20 N force states"
        ),
        parameters=parameters,
        parameter_constraints=[
            "geometry.flat_pad_height_mm + "
            "geometry.semiellipse_height_mm <= 30"
        ],
    )
    client.configure_optimization(objective="J_intensity, J_spatial")
    client.configure_generation_strategy(
        method="fast",
        initialization_budget=_WARM_START_COUNT,
        initialization_random_seed=_RANDOM_SEED,
        initialize_with_center=False,
        use_existing_trials_for_initialization=True,
    )
    return client


def _fieldnames() -> list[str]:
    return [
        "ax_trial_index",
        "source",
        "design",
        "generation_node",
        "status",
        "analytically_valid",
        *_PARAMETER_NAMES,
        *_PER_DIAMETER_FIELDS,
        *_OBJECTIVE_NAMES,
        "worst_intensity_diameter_mm",
        "worst_intensity_force_pair_n",
        "worst_spatial_diameter_mm",
        "worst_spatial_force_pair_n",
        "runtime_s",
        "raw_result_path",
        "failure",
        "is_pareto",
    ]


def _empty_result_fields() -> dict[str, object]:
    return {
        **{name: "" for name in _PER_DIAMETER_FIELDS},
        **{name: "" for name in _OBJECTIVE_NAMES},
        "worst_intensity_diameter_mm": "",
        "worst_intensity_force_pair_n": "",
        "worst_spatial_diameter_mm": "",
        "worst_spatial_force_pair_n": "",
        "runtime_s": "",
        "raw_result_path": "",
        "failure": "",
        "is_pareto": False,
    }


def _read_trials(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows: list[dict[str, object]] = []
        for raw_row in csv.DictReader(input_file):
            row: dict[str, object] = dict(raw_row)
            row["ax_trial_index"] = int(raw_row["ax_trial_index"])
            row["analytically_valid"] = raw_row["analytically_valid"] == "True"
            row["is_pareto"] = raw_row["is_pareto"] == "True"
            numeric_fields = (
                *_PARAMETER_NAMES,
                *_PER_DIAMETER_FIELDS,
                *_OBJECTIVE_NAMES,
                "runtime_s",
            )
            for name in numeric_fields:
                value = raw_row.get(name, "")
                row[name] = float(value) if value else ""
            rows.append(row)
    return rows


def _update_pareto_status(rows: list[dict[str, object]]) -> None:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    for row in rows:
        row["is_pareto"] = False
    for candidate in completed:
        candidate_objectives = np.array(
            [candidate[name] for name in _OBJECTIVE_NAMES],
            dtype=np.float64,
        )
        candidate["is_pareto"] = not any(
            np.all(
                np.array([other[name] for name in _OBJECTIVE_NAMES])
                >= candidate_objectives
            )
            and np.any(
                np.array([other[name] for name in _OBJECTIVE_NAMES])
                > candidate_objectives
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


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_save_ax(client: Client, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    client.save_to_json_file(filepath=str(temporary))
    with temporary.open("rb") as input_file:
        os.fsync(input_file.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_write_csv(
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


def _write_tables(output_directory: Path, rows: list[dict[str, object]]) -> None:
    _update_pareto_status(rows)
    _atomic_write_csv(
        output_directory / _TRIALS_FILENAME,
        rows,
        _fieldnames(),
    )
    pareto_rows = [
        row
        for row in rows
        if row["status"] == "COMPLETED" and row["is_pareto"]
    ]
    _atomic_write_csv(
        output_directory / _PARETO_FILENAME,
        pareto_rows,
        _fieldnames(),
    )


def _persist_ax_and_tables(
    client: Client,
    rows: list[dict[str, object]],
    output_directory: Path,
) -> None:
    _atomic_save_ax(client, output_directory / _AX_STATE_FILENAME)
    _write_tables(output_directory, rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    source_root = repository_root / "lumo"
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".cu", ".urdf"}:
            continue
        relative_path = path.relative_to(repository_root).as_posix()
        if relative_path == "lumo/optimization/ax_bo.py":
            continue
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def _run_config(space: DesignSpace) -> dict[str, object]:
    from . import evaluator

    repository_root = Path(__file__).resolve().parents[2]
    parameter_bounds = {
        f"geometry.{name}": [bound.lower, bound.upper]
        for name, bound in space.parameter_bounds.geometry.items()
    }
    linear_constraints = [
        {
            "coefficients": dict(constraint.coefficients),
            "lower": constraint.lower,
            "upper": constraint.upper,
        }
        for constraint in space.linear_constraints
    ]
    warm_start_sha256 = (
        _sha256_file(_WARM_START_PATH) if _WARM_START_PATH.is_file() else None
    )
    return {
        "schema_version": _RUN_CONFIG_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_revision": _git_output(repository_root, ["rev-parse", "HEAD"]),
            "git_dirty": bool(
                _git_output(repository_root, ["status", "--porcelain"])
            ),
            "scientific_source_sha256": _scientific_source_sha256(
                repository_root
            ),
            "optimizer_source_sha256": _sha256_file(Path(__file__).resolve()),
            "warm_start_sha256": warm_start_sha256,
            "versions": {
                "ax-platform": _package_version("ax-platform"),
                "botorch": _package_version("botorch"),
                "newton": _package_version("newton"),
                "warp-lang": _package_version("warp-lang"),
                "numpy": _package_version("numpy"),
            },
        },
        "scientific_contract": {
            "fingertip_parameters": asdict(FingertipParameters()),
            "mechanics": {
                "sim_frequency_hz": evaluator._SIM_FREQUENCY_HZ,
                "vbd_iterations": evaluator._VBD_ITERATIONS,
                "force_gain_m_s_n": evaluator._FORCE_GAIN_M_S_N,
                "position_gain_m_n_tick": (
                    evaluator._FORCE_GAIN_M_S_N
                    / evaluator._SIM_FREQUENCY_HZ
                ),
                "max_indenter_speed_m_s": _APPROACH_SPEED_M_S,
                "max_displacement_m_tick": (
                    _APPROACH_SPEED_M_S / evaluator._SIM_FREQUENCY_HZ
                ),
                "force_targets_n": list(evaluator._FORCE_TARGETS_N),
                "force_tolerance_fraction": (
                    evaluator._FORCE_TOLERANCE_FRACTION
                ),
                "fixed_servo_dwell_s": evaluator._SERVO_SETTLE_DURATION_S,
                "max_sim_time_s": _MAX_SIM_TIME_S,
                "element_size_mm": evaluator._ELEMENT_SIZE_MM,
                "soft_contact_margin_m": evaluator._SOFT_CONTACT_MARGIN_M,
                "carrier_contact_stiffness_n_m": (
                    evaluator._CARRIER_CONTACT_STIFFNESS_N_M
                ),
                "indenter_contact_stiffness_n_m": (
                    evaluator._CONTACT_STIFFNESS_N_M
                ),
                "indenter_contact_damping_n_s_m": (
                    evaluator._CONTACT_DAMPING_N_S_M
                ),
            },
            "scenarios": {
                "sphere_diameters_mm": list(_SPHERE_DIAMETERS_MM),
                "contact_x_mm": 0.0,
            },
            "optics": {
                "sample_side_count": evaluator._SAMPLE_SIDE_COUNT,
                "ray_count": evaluator._SAMPLE_SIDE_COUNT**2,
                "max_bounces": evaluator._MAX_BOUNCES,
                "deterministic_seed": evaluator._RNG_SEED,
                "carrier_albedo": evaluator._CARRIER_ALBEDO,
                "source_medium": "resolved per geometry from LED air-gap boundary",
            },
            "design_space": {
                "bounds": parameter_bounds,
                "linear_constraints": linear_constraints,
                "minimum_silicone_thickness_mm": (
                    space.minimum_silicone_thickness_mm
                ),
            },
            "objectives": {
                "names": list(_OBJECTIVE_NAMES),
                "directions": ["maximize", "maximize"],
                "definition": _OBJECTIVE_DEFINITION,
                "grouping": "force pairs within diameter, then minimum over diameter",
            },
            "ax_random_seed": _RANDOM_SEED,
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
    if stored_provenance.get(
        "scientific_source_sha256"
    ) != current_provenance.get(
        "scientific_source_sha256"
    ):
        raise RuntimeError(
            "LUMO scientific source differs from the saved run contract; "
            "resume with the original source snapshot"
        )


def _verify_warm_start_in_ax(
    client: Client,
    warm_start: list[dict[str, object]],
) -> None:
    summary = client.summarize()
    warm_rows = summary.iloc[:_WARM_START_COUNT]
    if len(summary) < _WARM_START_COUNT or len(warm_rows) != _WARM_START_COUNT:
        raise RuntimeError("Ax does not contain all 13 warm-start observations")
    for (_, ax_row), expected in zip(warm_rows.iterrows(), warm_start, strict=True):
        if str(ax_row["trial_status"]).upper().split(".")[-1] != "COMPLETED":
            raise RuntimeError("an Ax warm-start trial is not completed")
        expected_parameters = expected["parameters"]
        expected_objectives = expected["objectives"]
        for name in _PARAMETER_NAMES:
            if not np.isclose(
                float(ax_row[name]),
                float(expected_parameters[name]),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(f"Ax warm-start parameter mismatch for {name}")
        for name in _OBJECTIVE_NAMES:
            if not np.isclose(
                float(ax_row[name]),
                float(expected_objectives[name]),
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError(f"Ax warm-start objective mismatch for {name}")


def _evaluate_candidate(space: DesignSpace, parameters: dict[str, float]):
    import warp as wp

    from lumo.fingertip import Fingertip
    from lumo.simulation import DesignTrial

    from .evaluator import evaluate_contact_sensing

    fingertip = Fingertip(space.to_parameters(parameters))
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    with ExitStack() as resources:
        sphere_resources = tuple(
            (
                resources.enter_context(
                    as_file(resource_root.joinpath(filename))
                ),
                diameter_mm,
            )
            for filename, diameter_mm in _SPHERES
        )
        trials = tuple(
            DesignTrial(
                name=f"sphere_{diameter_mm:g}mm_center",
                urdf_path=urdf_path,
                initial_tf=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        fingertip.tip_z_m
                        - _INITIAL_CLEARANCE_M
                        - 0.5e-3 * diameter_mm,
                    ),
                    wp.quat_identity(),
                ),
                motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                approach_speed_m_s=_APPROACH_SPEED_M_S,
                target_force_n=20.0,
                max_sim_time_s=_MAX_SIM_TIME_S,
                initial_clearance_m=_INITIAL_CLEARANCE_M,
            )
            for urdf_path, diameter_mm in sphere_resources
        )
        return evaluate_contact_sensing(fingertip, trials)


def _minimum_pair(
    values: np.ndarray,
    force_targets_n: np.ndarray,
    *,
    spatial: bool,
) -> tuple[float, tuple[float, float]]:
    minimum = float("inf")
    pair = (float("nan"), float("nan"))
    for first in range(len(values) - 1):
        for second in range(first + 1, len(values)):
            if spatial:
                separation = float(np.linalg.norm(values[first] - values[second]))
            else:
                separation = abs(float(values[first] - values[second]))
            if separation < minimum:
                minimum = separation
                pair = (
                    float(force_targets_n[first]),
                    float(force_targets_n[second]),
                )
    return minimum, pair


def _objective_details(evaluation: object) -> dict[str, object]:
    from .sensing_objective import sensing_descriptors, sensing_objectives

    per_intensity, per_spatial, intensity, spatial = sensing_objectives(
        evaluation.response_matrix,
        no_contact_response=evaluation.no_contact_response,
    )
    intensity_pairs = []
    spatial_pairs = []
    for responses in evaluation.response_matrix:
        descriptors = sensing_descriptors(
            np.vstack((evaluation.no_contact_response, responses))
        )
        _, intensity_pair = _minimum_pair(
            descriptors[0][1:],
            evaluation.force_targets_n,
            spatial=False,
        )
        _, spatial_pair = _minimum_pair(
            descriptors[1][1:],
            evaluation.force_targets_n,
            spatial=True,
        )
        intensity_pairs.append(intensity_pair)
        spatial_pairs.append(spatial_pair)

    worst_intensity_index = int(np.argmin(per_intensity))
    worst_spatial_index = int(np.argmin(per_spatial))
    return {
        "per_intensity": per_intensity,
        "per_spatial": per_spatial,
        "J_intensity": float(intensity),
        "J_spatial": float(spatial),
        "worst_intensity_index": worst_intensity_index,
        "worst_spatial_index": worst_spatial_index,
        "intensity_pairs": tuple(intensity_pairs),
        "spatial_pairs": tuple(spatial_pairs),
    }


def _pair_text(pair: tuple[float, float]) -> str:
    return f"{pair[0]:g}-{pair[1]:g}"


def _trial_result_path(output_directory: Path, trial_index: int) -> Path:
    return (
        output_directory
        / _TRIAL_RESULT_DIRECTORY
        / f"trial_{trial_index:04d}.npz"
    )


def _save_trial_result(
    path: Path,
    *,
    evaluation: object,
    details: dict[str, object],
    parameters: dict[str, float],
    runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            no_contact_response=np.asarray(evaluation.no_contact_response),
            no_contact_energy=np.asarray(evaluation.no_contact_energy),
            response_matrix=np.asarray(evaluation.response_matrix),
            energy_matrix=np.asarray(evaluation.energy_matrix),
            energy_fields=np.asarray(evaluation.energy_fields),
            actual_forces_n=np.asarray(evaluation.actual_forces_n),
            indentations_m=np.asarray(evaluation.indentations_m),
            checkpoint_times_s=np.asarray(evaluation.checkpoint_times_s),
            scenario_runtime_s=np.asarray(evaluation.scenario_runtime_s),
            scenario_names=np.asarray(evaluation.scenario_names),
            sphere_diameters_mm=np.asarray(_SPHERE_DIAMETERS_MM),
            force_targets_n=np.asarray(evaluation.force_targets_n),
            per_diameter_J_intensity=np.asarray(details["per_intensity"]),
            per_diameter_J_spatial=np.asarray(details["per_spatial"]),
            J_intensity=np.asarray(details["J_intensity"]),
            J_spatial=np.asarray(details["J_spatial"]),
            evaluation_runtime_s=np.asarray(runtime_s),
            parameter_names=np.asarray(_PARAMETER_NAMES),
            parameter_values=np.asarray(
                [parameters[name] for name in _PARAMETER_NAMES]
            ),
        )
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _apply_result_to_row(
    row: dict[str, object],
    details: dict[str, object],
    *,
    runtime_s: float,
    raw_result_path: str,
) -> None:
    per_intensity = np.asarray(details["per_intensity"])
    per_spatial = np.asarray(details["per_spatial"])
    for index, diameter_mm in enumerate(_SPHERE_DIAMETERS_MM):
        row[f"J_intensity_{diameter_mm:g}mm"] = float(per_intensity[index])
        row[f"J_spatial_{diameter_mm:g}mm"] = float(per_spatial[index])
    worst_intensity_index = int(details["worst_intensity_index"])
    worst_spatial_index = int(details["worst_spatial_index"])
    row.update(
        J_intensity=float(details["J_intensity"]),
        J_spatial=float(details["J_spatial"]),
        worst_intensity_diameter_mm=(
            _SPHERE_DIAMETERS_MM[worst_intensity_index]
        ),
        worst_intensity_force_pair_n=_pair_text(
            details["intensity_pairs"][worst_intensity_index]
        ),
        worst_spatial_diameter_mm=_SPHERE_DIAMETERS_MM[worst_spatial_index],
        worst_spatial_force_pair_n=_pair_text(
            details["spatial_pairs"][worst_spatial_index]
        ),
        runtime_s=runtime_s,
        raw_result_path=raw_result_path,
        failure="",
    )


def _ax_statuses(client: Client) -> dict[int, str]:
    summary = client.summarize()
    return {
        int(row["trial_index"]): str(row["trial_status"]).upper().split(".")[-1]
        for _, row in summary.iterrows()
    }


def _parameters_from_summary_row(row: object) -> dict[str, float]:
    return {name: float(row[name]) for name in _PARAMETER_NAMES}


def _running_row(
    trial_index: int,
    parameters: dict[str, float],
    generation_node: str,
) -> dict[str, object]:
    return {
        "ax_trial_index": trial_index,
        "source": "bo",
        "design": f"bo_{trial_index:04d}",
        "generation_node": generation_node,
        "status": "RUNNING",
        "analytically_valid": False,
        **parameters,
        **_empty_result_fields(),
    }


def _reconcile_resume(
    client: Client,
    rows: list[dict[str, object]],
    output_directory: Path,
) -> None:
    summary = client.summarize()
    ax_statuses = _ax_statuses(client)
    row_indices = {int(row["ax_trial_index"]) for row in rows}
    ax_changed = False

    for _, summary_row in summary.iterrows():
        trial_index = int(summary_row["trial_index"])
        if trial_index in row_indices:
            continue
        status = ax_statuses[trial_index]
        if status != "RUNNING":
            raise RuntimeError(
                f"Ax trial {trial_index} is absent from trials.csv with status {status}"
            )
        row = _running_row(
            trial_index,
            _parameters_from_summary_row(summary_row),
            str(summary_row.get("generation_node", "")),
        )
        row["runtime_s"] = 0.0
        row["failure"] = "interrupted before proposal CSV persistence"
        client.mark_trial_abandoned(trial_index)
        ax_changed = True
        row["status"] = "ABANDONED"
        rows.append(row)

    ax_indices = set(ax_statuses)
    unexpected_rows = {
        int(row["ax_trial_index"]) for row in rows
    } - ax_indices
    if unexpected_rows:
        raise RuntimeError(
            f"trials.csv contains trials absent from Ax: {sorted(unexpected_rows)}"
        )

    ax_statuses = _ax_statuses(client)
    for row in rows:
        trial_index = int(row["ax_trial_index"])
        csv_status = str(row["status"])
        ax_status = ax_statuses[trial_index]
        if csv_status == "EVALUATED":
            raw_path = output_directory / str(row["raw_result_path"])
            if not raw_path.is_file():
                raise RuntimeError(
                    f"evaluated trial {trial_index} has no raw-result NPZ"
                )
            if ax_status == "RUNNING":
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        name: float(row[name]) for name in _OBJECTIVE_NAMES
                    },
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
            ax_changed = True
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = "interrupted during morphology evaluation"
        elif csv_status == "FAILED" and ax_status == "ABANDONED":
            continue
        elif csv_status != ax_status:
            raise RuntimeError(
                f"trial {trial_index} status mismatch: CSV={csv_status}, "
                f"Ax={ax_status}"
            )

    if ax_changed:
        _atomic_save_ax(client, output_directory / _AX_STATE_FILENAME)
    _write_tables(output_directory, rows)


def _completed_bo_count(rows: list[dict[str, object]]) -> int:
    return sum(
        row["source"] == "bo" and row["status"] == "COMPLETED"
        for row in rows
    )


def _duplicate_terminal(
    rows: list[dict[str, object]],
    parameters: dict[str, float],
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in rows
            if row["status"] in {"COMPLETED", "FAILED"}
            and all(
                np.isclose(
                    parameters[name],
                    float(row[name]),
                    rtol=0.0,
                    atol=1.0e-8,
                )
                for name in _PARAMETER_NAMES
            )
        ),
        None,
    )


def _validate_optix_environment() -> None:
    required = {
        "OPTIX_INCLUDE_DIR": Path("optix.h"),
        "OTK_INCLUDE_DIR": Path("OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h"),
    }
    failures = []
    for variable, relative_path in required.items():
        raw_directory = os.environ.get(variable)
        if not raw_directory:
            failures.append(f"{variable} is not set")
            continue
        expected = Path(raw_directory) / relative_path
        if not expected.is_file():
            failures.append(f"{variable} does not contain {relative_path}")
    if failures:
        raise RuntimeError("OptiX environment preflight failed: " + "; ".join(failures))


def _is_infrastructure_failure(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    markers = (
        "optix_include_dir",
        "otk_include_dir",
        "cuda driver",
        "cuda error",
        "nvrtc",
        "failed to load optix",
        "no cuda",
        "out of memory",
    )
    return any(marker in message for marker in markers)


def _make_warm_row(
    trial_index: int,
    warm: dict[str, object],
) -> dict[str, object]:
    return {
        "ax_trial_index": trial_index,
        "source": "warm_start",
        "design": warm["design"],
        "generation_node": "",
        "status": "COMPLETED",
        "analytically_valid": True,
        **warm["parameters"],
        **warm["per_diameter"],
        **warm["objectives"],
        "worst_intensity_diameter_mm": warm[
            "worst_intensity_diameter_mm"
        ],
        "worst_intensity_force_pair_n": warm[
            "worst_intensity_force_pair_n"
        ],
        "worst_spatial_diameter_mm": warm["worst_spatial_diameter_mm"],
        "worst_spatial_force_pair_n": warm["worst_spatial_force_pair_n"],
        "runtime_s": warm["runtime_s"],
        "raw_result_path": "",
        "failure": "",
        "is_pareto": False,
    }


def _initialize_campaign(
    output_directory: Path,
    space: DesignSpace,
) -> tuple[Client, list[dict[str, object]], dict[str, object]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    contents = list(output_directory.iterdir())
    if contents:
        raise FileExistsError(
            f"fresh campaign output is not empty: {output_directory}"
        )
    config = _run_config(space)
    _atomic_write_json(output_directory / _RUN_CONFIG_FILENAME, config)
    warm_start = _read_warm_start(space)
    client = _new_client(space)
    rows = []
    for warm in warm_start:
        trial_index = client.attach_trial(
            parameters=dict(warm["parameters"]),
            arm_name=str(warm["design"]),
        )
        client.complete_trial(
            trial_index=trial_index,
            raw_data=dict(warm["objectives"]),
        )
        rows.append(_make_warm_row(trial_index, warm))
        _persist_ax_and_tables(client, rows, output_directory)
    _verify_warm_start_in_ax(client, warm_start)
    print("loaded and verified 13 completed warm-start trials", flush=True)
    return client, rows, config


def _resume_campaign(
    output_directory: Path,
    space: DesignSpace,
) -> tuple[Client, list[dict[str, object]], dict[str, object]]:
    required = (
        output_directory / _RUN_CONFIG_FILENAME,
        output_directory / _AX_STATE_FILENAME,
        output_directory / _TRIALS_FILENAME,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "campaign directory is incomplete; missing " + ", ".join(missing)
        )
    with required[0].open(encoding="utf-8") as input_file:
        stored_config = json.load(input_file)
    _validate_run_config(stored_config, _run_config(space))
    client = Client.load_from_json_file(filepath=str(required[1]))
    rows = _read_trials(required[2])
    if sum(row["source"] == "warm_start" for row in rows) != _WARM_START_COUNT:
        raise RuntimeError("persisted campaign does not contain 13 warm starts")
    _reconcile_resume(client, rows, output_directory)
    if _WARM_START_PATH.is_file():
        _verify_warm_start_in_ax(client, _read_warm_start(space))
    print(
        f"resumed campaign with {_completed_bo_count(rows)} completed BO trials",
        flush=True,
    )
    return client, rows, stored_config


def _write_plots(output_directory: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    intensity = np.asarray(
        [float(row["J_intensity"]) for row in completed],
        dtype=np.float64,
    )
    spatial = np.asarray(
        [float(row["J_spatial"]) for row in completed],
        dtype=np.float64,
    )
    is_pareto = np.asarray(
        [bool(row["is_pareto"]) for row in completed],
        dtype=bool,
    )
    is_warm = np.asarray(
        [row["source"] == "warm_start" for row in completed],
        dtype=bool,
    )

    figure, axes = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axes.scatter(
        intensity[is_warm],
        spatial[is_warm],
        color="tab:gray",
        label="warm start",
        alpha=0.8,
    )
    axes.scatter(
        intensity[~is_warm],
        spatial[~is_warm],
        color="tab:blue",
        label="BO",
        alpha=0.8,
    )
    axes.set_xlabel("J_intensity")
    axes.set_ylabel("J_spatial")
    axes.set_title("Observed LUMO sensing objectives")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(output_directory / "objective_scatter.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axes.scatter(intensity, spatial, color="0.7", label="dominated")
    axes.scatter(
        intensity[is_pareto],
        spatial[is_pareto],
        color="tab:red",
        label="observed Pareto",
        zorder=3,
    )
    if np.count_nonzero(is_pareto) > 1:
        order = np.argsort(intensity[is_pareto])
        axes.plot(
            intensity[is_pareto][order],
            spatial[is_pareto][order],
            color="tab:red",
            alpha=0.6,
        )
    axes.set_xlabel("J_intensity")
    axes.set_ylabel("J_spatial")
    axes.set_title("Observed nondominated front")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(output_directory / "pareto_front.png", dpi=180)
    plt.close(figure)


def _finalize_outputs(
    output_directory: Path,
    rows: list[dict[str, object]],
    config: dict[str, object],
    *,
    command_wall_runtime_s: float,
) -> dict[str, object]:
    _write_tables(output_directory, rows)
    _write_plots(output_directory, rows)
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    best_intensity = max(completed, key=lambda row: float(row["J_intensity"]))
    best_spatial = max(completed, key=lambda row: float(row["J_spatial"]))
    previous_active_runtime_s = 0.0
    summary_path = output_directory / _SUMMARY_FILENAME
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as input_file:
            previous_summary = json.load(input_file)
        previous_active_runtime_s = float(
            previous_summary.get("active_wall_runtime_s", 0.0)
        )
    created_utc = datetime.fromisoformat(str(config["created_utc"]))
    now_utc = datetime.now(UTC)
    summary = {
        "updated_utc": now_utc.isoformat(),
        "campaign_elapsed_wall_runtime_s": (now_utc - created_utc).total_seconds(),
        "active_wall_runtime_s": (
            previous_active_runtime_s + command_wall_runtime_s
        ),
        "total_evaluation_runtime_s": sum(
            float(row["runtime_s"])
            for row in rows
            if row["source"] == "bo"
            and row["status"] in {"COMPLETED", "FAILED"}
            and row["runtime_s"] != ""
        ),
        "counts": {
            "warm_start_completed": sum(
                row["source"] == "warm_start" and row["status"] == "COMPLETED"
                for row in rows
            ),
            "bo_completed": _completed_bo_count(rows),
            "bo_failed": sum(
                row["source"] == "bo" and row["status"] == "FAILED"
                for row in rows
            ),
            "bo_abandoned": sum(
                row["source"] == "bo" and row["status"] == "ABANDONED"
                for row in rows
            ),
            "pareto": sum(
                row["status"] == "COMPLETED" and row["is_pareto"]
                for row in rows
            ),
        },
        "best_J_intensity": {
            "design": best_intensity["design"],
            "ax_trial_index": best_intensity["ax_trial_index"],
            "value": best_intensity["J_intensity"],
        },
        "best_J_spatial": {
            "design": best_spatial["design"],
            "ax_trial_index": best_spatial["ax_trial_index"],
            "value": best_spatial["J_spatial"],
        },
    }
    _atomic_write_json(summary_path, summary)
    return summary


def run(
    *,
    output_directory: Path,
    target_bo_trials: int,
) -> list[dict[str, object]]:
    """Create or resume the concrete cumulative center-contact campaign."""
    if target_bo_trials < 0:
        raise ValueError("target_bo_trials must be nonnegative")
    command_start_s = perf_counter()
    output_directory = output_directory.resolve()
    space = _design_space()
    campaign_files_exist = any(
        (output_directory / filename).exists()
        for filename in (
            _RUN_CONFIG_FILENAME,
            _AX_STATE_FILENAME,
            _TRIALS_FILENAME,
        )
    )
    if campaign_files_exist:
        client, rows, config = _resume_campaign(output_directory, space)
    else:
        client, rows, config = _initialize_campaign(output_directory, space)

    initial_completed_bo = _completed_bo_count(rows)
    if initial_completed_bo < target_bo_trials:
        _validate_optix_environment()
    remaining = target_bo_trials - initial_completed_bo
    proposal_limit = max(1, remaining) * _MAX_PROPOSALS_PER_COMPLETED_TRIAL
    proposal_count = 0
    smoke_pending = initial_completed_bo == 0 and target_bo_trials > 0

    while _completed_bo_count(rows) < target_bo_trials:
        if proposal_count >= proposal_limit:
            raise RuntimeError(
                "Ax did not produce enough successful feasible candidates "
                f"within {proposal_limit} proposals"
            )
        generated = client.get_next_trials(max_trials=1)
        if len(generated) != 1:
            raise RuntimeError("Ax did not return exactly one sequential trial")
        trial_index, raw_parameters = generated.popitem()
        parameters = {
            name: float(raw_parameters[name]) for name in space.variable_names
        }
        summary_row = client.summarize(trial_indices=[trial_index]).iloc[0]
        row = _running_row(
            trial_index,
            parameters,
            str(summary_row.get("generation_node", "")),
        )
        rows.append(row)
        proposal_count += 1
        _persist_ax_and_tables(client, rows, output_directory)
        print(
            f"proposed trial {trial_index} ({row['generation_node']}): "
            f"{parameters}",
            flush=True,
        )

        duplicate = _duplicate_terminal(rows[:-1], parameters)
        if duplicate is not None:
            client.mark_trial_abandoned(trial_index)
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = f"duplicates terminal design {duplicate['design']}"
            _persist_ax_and_tables(client, rows, output_directory)
            print(f"abandoned duplicate trial {trial_index}", flush=True)
            continue

        if not space.is_feasible(parameters):
            client.mark_trial_abandoned(trial_index)
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = "analytically invalid morphology"
            _persist_ax_and_tables(client, rows, output_directory)
            print(
                f"abandoned analytically invalid trial {trial_index}",
                flush=True,
            )
            continue

        row["analytically_valid"] = True
        evaluation_start_s = perf_counter()
        try:
            evaluation = _evaluate_candidate(space, parameters)
            runtime_s = perf_counter() - evaluation_start_s
            details = _objective_details(evaluation)
            raw_result_path = _trial_result_path(output_directory, trial_index)
            _save_trial_result(
                raw_result_path,
                evaluation=evaluation,
                details=details,
                parameters=parameters,
                runtime_s=runtime_s,
            )
            _apply_result_to_row(
                row,
                details,
                runtime_s=runtime_s,
                raw_result_path=raw_result_path.relative_to(
                    output_directory
                ).as_posix(),
            )
            row["status"] = "EVALUATED"
            _write_tables(output_directory, rows)
            objectives = {
                name: float(row[name]) for name in _OBJECTIVE_NAMES
            }
            client.complete_trial(trial_index=trial_index, raw_data=objectives)
            _atomic_save_ax(client, output_directory / _AX_STATE_FILENAME)
            row["status"] = "COMPLETED"
            _write_tables(output_directory, rows)
            del evaluation
        except Exception as error:
            runtime_s = perf_counter() - evaluation_start_s
            row["runtime_s"] = runtime_s
            row["status"] = "FAILED"
            row["failure"] = f"{type(error).__name__}: {error}"
            if _ax_statuses(client)[trial_index] == "RUNNING":
                client.mark_trial_abandoned(trial_index)
            _persist_ax_and_tables(client, rows, output_directory)
            print(f"trial {trial_index} failed: {row['failure']}", flush=True)
            if smoke_pending or _is_infrastructure_failure(error):
                raise RuntimeError(
                    f"trial {trial_index} evaluation failed; campaign was saved"
                ) from error
            continue

        print(
            f"completed trial {trial_index}: "
            f"J_intensity={float(row['J_intensity']):.9e}, "
            f"J_spatial={float(row['J_spatial']):.9e}, "
            f"runtime={float(row['runtime_s']):.3f} s",
            flush=True,
        )
        if smoke_pending:
            client = Client.load_from_json_file(
                filepath=str(output_directory / _AX_STATE_FILENAME)
            )
            rows = _read_trials(output_directory / _TRIALS_FILENAME)
            _reconcile_resume(client, rows, output_directory)
            if _completed_bo_count(rows) != 1:
                raise RuntimeError("smoke resume check lost or repeated its BO trial")
            smoke_pending = False
            print(
                "smoke PASS: raw NPZ, CSV, atomic Ax state, and resume verified",
                flush=True,
            )

    summary = _finalize_outputs(
        output_directory,
        rows,
        config,
        command_wall_runtime_s=perf_counter() - command_start_s,
    )
    print(
        f"target reached: {_completed_bo_count(rows)} completed BO trials; "
        f"Pareto={summary['counts']['pareto']}",
        flush=True,
    )
    print(f"outputs: {output_directory}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cumulative sequential Ax multi-objective LUMO BO."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_DIRECTORY,
        help="campaign output directory; an existing campaign resumes automatically",
    )
    parser.add_argument(
        "--target-bo-trials",
        type=int,
        required=True,
        help="cumulative successful BO target, excluding 13 warm-start trials",
    )
    arguments = parser.parse_args()
    run(
        output_directory=arguments.output,
        target_bo_trials=arguments.target_bo_trials,
    )


if __name__ == "__main__":
    main()
