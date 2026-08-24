"""Sequential Ax optimization of the current LUMO sensing objectives."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
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
_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
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
    / "sensing_bo"
)
_AX_STATE_FILENAME = "ax_client.json"
_TRIALS_FILENAME = "trials.csv"
_RANDOM_SEED = 20260823
_WARM_START_COUNT = 13
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_MAX_PROPOSALS_PER_COMPLETED_TRIAL = 50


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
                name: float(raw_row[name])
                for name in space.variable_names
            }
            objectives = {
                name: float(raw_row[name])
                for name in _OBJECTIVE_NAMES
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
        *_OBJECTIVE_NAMES,
        "runtime_s",
        "failure",
        "is_pareto",
    ]


def _read_trials(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows: list[dict[str, object]] = []
        for raw_row in csv.DictReader(input_file):
            row: dict[str, object] = dict(raw_row)
            row["ax_trial_index"] = int(raw_row["ax_trial_index"])
            row["analytically_valid"] = raw_row["analytically_valid"] == "True"
            row["is_pareto"] = raw_row["is_pareto"] == "True"
            for name in (*_PARAMETER_NAMES, *_OBJECTIVE_NAMES, "runtime_s"):
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


def _persist(
    client: Client,
    rows: list[dict[str, object]],
    *,
    state_path: Path,
    trials_path: Path,
) -> None:
    _update_pareto_status(rows)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_state = state_path.with_suffix(".tmp.json")
    client.save_to_json_file(filepath=str(temporary_state))
    temporary_state.replace(state_path)

    temporary_trials = trials_path.with_suffix(".tmp.csv")
    with temporary_trials.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    temporary_trials.replace(trials_path)


def _evaluate_candidate(
    space: DesignSpace,
    parameters: dict[str, float],
) -> tuple[float, float]:
    import warp as wp

    from lumo.fingertip import Fingertip
    from lumo.simulation import DesignTrial

    from .evaluator import evaluate_contact_sensing
    from .sensing_objective import sensing_objectives

    fingertip = Fingertip(space.to_parameters(parameters))
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    with ExitStack() as resources:
        sphere_resources = tuple(
            (
                resources.enter_context(as_file(resource_root.joinpath(filename))),
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
        evaluation = evaluate_contact_sensing(fingertip, trials)

    _, _, intensity, spatial = sensing_objectives(
        evaluation.response_matrix,
        no_contact_response=evaluation.no_contact_response,
    )
    return intensity, spatial


def run(
    *,
    output_directory: Path,
    new_trials: int,
    resume: bool,
) -> list[dict[str, object]]:
    """Run or resume the concrete sequential center-contact Ax campaign."""
    if new_trials < 0:
        raise ValueError("new_trials must be nonnegative")

    output_directory = output_directory.resolve()
    state_path = output_directory / _AX_STATE_FILENAME
    trials_path = output_directory / _TRIALS_FILENAME
    space = _design_space()
    warm_start = _read_warm_start(space)

    if resume:
        if not state_path.is_file() or not trials_path.is_file():
            raise FileNotFoundError(
                "resume requires both ax_client.json and trials.csv"
            )
        client = Client.load_from_json_file(filepath=str(state_path))
        rows = _read_trials(trials_path)
        if sum(row["source"] == "warm_start" for row in rows) != len(warm_start):
            raise RuntimeError("persisted campaign does not contain all warm starts")
        persisted_indices = {int(row["ax_trial_index"]) for row in rows}
        ax_indices = set(client.summarize()["trial_index"].astype(int))
        if persisted_indices != ax_indices:
            raise RuntimeError("Ax state and trial CSV contain different trials")
        for row in rows:
            if row["status"] == "RUNNING":
                client.mark_trial_abandoned(int(row["ax_trial_index"]))
                row["status"] = "ABANDONED"
                row["failure"] = "interrupted before a result was persisted"
        _persist(
            client,
            rows,
            state_path=state_path,
            trials_path=trials_path,
        )
        print(f"resumed {len(rows)} persisted trials", flush=True)
    else:
        if state_path.exists() or trials_path.exists():
            raise FileExistsError(
                f"campaign already exists in {output_directory}; use --resume"
            )
        client = _new_client(space)
        rows = []
        for warm in warm_start:
            parameters = dict(warm["parameters"])
            objectives = dict(warm["objectives"])
            trial_index = client.attach_trial(
                parameters=parameters,
                arm_name=str(warm["design"]),
            )
            client.complete_trial(trial_index=trial_index, raw_data=objectives)
            rows.append(
                {
                    "ax_trial_index": trial_index,
                    "source": "warm_start",
                    "design": warm["design"],
                    "generation_node": "",
                    "status": "COMPLETED",
                    "analytically_valid": True,
                    **parameters,
                    **objectives,
                    "runtime_s": warm["runtime_s"],
                    "failure": "",
                    "is_pareto": False,
                }
            )
            _persist(
                client,
                rows,
                state_path=state_path,
                trials_path=trials_path,
            )
        print(f"loaded {len(warm_start)} completed warm-start trials", flush=True)

    completed_new_trials = 0
    proposal_count = 0
    proposal_limit = max(
        new_trials,
        new_trials * _MAX_PROPOSALS_PER_COMPLETED_TRIAL,
    )
    while completed_new_trials < new_trials:
        if proposal_count >= proposal_limit:
            raise RuntimeError(
                "Ax did not produce enough feasible successful proposals "
                f"within {proposal_limit} proposals"
            )
        generated = client.get_next_trials(max_trials=1)
        if len(generated) != 1:
            raise RuntimeError("Ax did not return exactly one sequential trial")
        trial_index, raw_parameters = generated.popitem()
        parameters = {
            name: float(raw_parameters[name])
            for name in space.variable_names
        }
        proposal_count += 1
        summary = client.summarize(trial_indices=[trial_index])
        generation_node = str(summary.iloc[0].get("generation_node", ""))
        row: dict[str, object] = {
            "ax_trial_index": trial_index,
            "source": "bo",
            "design": f"bo_{trial_index:03d}",
            "generation_node": generation_node,
            "status": "RUNNING",
            "analytically_valid": False,
            **parameters,
            "J_intensity": "",
            "J_spatial": "",
            "runtime_s": "",
            "failure": "",
            "is_pareto": False,
        }
        rows.append(row)
        _persist(
            client,
            rows,
            state_path=state_path,
            trials_path=trials_path,
        )
        print(
            f"proposed trial {trial_index} ({generation_node}): {parameters}",
            flush=True,
        )

        if not space.is_feasible(parameters):
            client.mark_trial_abandoned(trial_index)
            row["status"] = "ABANDONED"
            row["failure"] = "analytically invalid morphology"
            _persist(
                client,
                rows,
                state_path=state_path,
                trials_path=trials_path,
            )
            print(f"abandoned analytically invalid trial {trial_index}", flush=True)
            continue

        row["analytically_valid"] = True
        evaluation_start_s = perf_counter()
        try:
            intensity, spatial = _evaluate_candidate(space, parameters)
        except Exception as error:
            row["runtime_s"] = perf_counter() - evaluation_start_s
            row["status"] = "FAILED"
            row["failure"] = f"{type(error).__name__}: {error}"
            client.mark_trial_failed(trial_index, failed_reason=str(row["failure"]))
            _persist(
                client,
                rows,
                state_path=state_path,
                trials_path=trials_path,
            )
            print(f"trial {trial_index} failed: {row['failure']}", flush=True)
            continue

        runtime_s = perf_counter() - evaluation_start_s
        objectives = {"J_intensity": intensity, "J_spatial": spatial}
        client.complete_trial(trial_index=trial_index, raw_data=objectives)
        row.update(
            status="COMPLETED",
            runtime_s=runtime_s,
            failure="",
            **objectives,
        )
        completed_new_trials += 1
        _persist(
            client,
            rows,
            state_path=state_path,
            trials_path=trials_path,
        )
        print(
            f"completed trial {trial_index}: "
            f"J_intensity={intensity:.9e}, J_spatial={spatial:.9e}, "
            f"runtime={runtime_s:.3f} s",
            flush=True,
        )

    pareto = [
        row
        for row in rows
        if row["status"] == "COMPLETED" and row["is_pareto"]
    ]
    print("observed Pareto set:", flush=True)
    for row in pareto:
        print(
            f"  {row['design']}: J_intensity={float(row['J_intensity']):.9e}, "
            f"J_spatial={float(row['J_spatial']):.9e}",
            flush=True,
        )
    print(f"Ax state: {state_path}", flush=True)
    print(f"trial CSV: {trials_path}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential Ax multi-objective LUMO sensing BO."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_DIRECTORY,
        help="campaign output directory",
    )
    parser.add_argument(
        "--new-trials",
        type=int,
        default=1,
        help="number of new successful BO evaluations",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the Ax state and CSV already stored under --output",
    )
    arguments = parser.parse_args()
    run(
        output_directory=arguments.output,
        new_trials=arguments.new_trials,
        resume=arguments.resume,
    )


if __name__ == "__main__":
    main()
