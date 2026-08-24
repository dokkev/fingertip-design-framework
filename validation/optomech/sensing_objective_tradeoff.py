"""Sample fingertip morphologies and inspect sensing-objective trade-offs."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from scipy.stats import pearsonr, qmc, spearmanr

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization import (
    DesignParameterBounds,
    DesignSpace,
    LinearConstraint,
    ParameterBound,
    sensing_descriptors,
    sensing_objectives,
)
from lumo.optimization.evaluator import evaluate_contact_sensing
from lumo.simulation import DesignTrial


plt.switch_backend("Agg")


_ADDITIONAL_MORPHOLOGIES = 12
_SOBOL_SEED = 20260823
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
)
_OUTPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "output" / "validation"
_CSV_PATH = _OUTPUT_DIRECTORY / "sensing_objective_tradeoff.csv"
_PLOT_PATH = _OUTPUT_DIRECTORY / "sensing_objective_tradeoff.png"


def _make_design_space() -> DesignSpace:
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


def _baseline_candidate(space: DesignSpace) -> dict[str, float]:
    return {
        name: space.parameter_values(space.parameter_bounds.parameters)[name]
        for name in space.variable_names
    }


def _sobol_candidates(
    space: DesignSpace,
    count: int,
) -> list[dict[str, float]]:
    unit_samples = qmc.Sobol(
        d=len(space.variable_names),
        scramble=True,
        seed=_SOBOL_SEED,
    ).random_base2(m=8)
    bounds = tuple(
        space.parameter_bounds.geometry[name.removeprefix("geometry.")]
        for name in space.variable_names
    )
    candidates = []
    for unit_sample in unit_samples:
        candidate = {
            name: bound.lower + float(sample) * (bound.upper - bound.lower)
            for name, bound, sample in zip(
                space.variable_names,
                bounds,
                unit_sample,
                strict=True,
            )
        }
        if space.is_feasible(candidate):
            candidates.append(candidate)
            if len(candidates) == count:
                return candidates
    raise RuntimeError(f"Sobol sampling found fewer than {count} feasible designs")


def _make_trials(
    fingertip: Fingertip,
    sphere_resources: tuple[tuple[Path, float], ...],
) -> list[DesignTrial]:
    trials = []
    for urdf_path, diameter_mm in sphere_resources:
        radius_m = 0.5e-3 * diameter_mm
        trials.append(
            DesignTrial(
                name=f"sphere_{diameter_mm:g}mm_center",
                urdf_path=urdf_path,
                initial_tf=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        fingertip.tip_z_m - _INITIAL_CLEARANCE_M - radius_m,
                    ),
                    wp.quat_identity(),
                ),
                motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
                approach_speed_m_s=_APPROACH_SPEED_M_S,
                target_force_n=20.0,
                max_sim_time_s=_MAX_SIM_TIME_S,
                initial_clearance_m=_INITIAL_CLEARANCE_M,
            )
        )
    return trials


def _worst_force_pairs(
    no_contact_response: np.ndarray,
    response_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    intensity_pairs = np.empty((len(response_matrix), 2), dtype=np.int64)
    spatial_pairs = np.empty((len(response_matrix), 2), dtype=np.int64)
    for sphere_index, sphere_responses in enumerate(response_matrix):
        intensity, spatial = sensing_descriptors(
            np.vstack((no_contact_response, sphere_responses))
        )
        intensity = intensity[1:]
        spatial = spatial[1:]
        minimum_intensity = float("inf")
        minimum_spatial = float("inf")
        for first in range(len(intensity) - 1):
            for second in range(first + 1, len(intensity)):
                intensity_distance = abs(float(intensity[first] - intensity[second]))
                if intensity_distance < minimum_intensity:
                    minimum_intensity = intensity_distance
                    intensity_pairs[sphere_index] = (first, second)
                spatial_distance = float(
                    np.linalg.norm(spatial[first] - spatial[second])
                )
                if spatial_distance < minimum_spatial:
                    minimum_spatial = spatial_distance
                    spatial_pairs[sphere_index] = (first, second)
    return intensity_pairs, spatial_pairs


def _fieldnames(variable_names: tuple[str, ...]) -> list[str]:
    fields = ["design", "is_baseline", *variable_names]
    for _, diameter_mm in _SPHERES:
        fields.extend(
            (
                f"J_intensity_{diameter_mm:g}mm",
                f"J_spatial_{diameter_mm:g}mm",
            )
        )
    return fields + [
        "J_intensity",
        "J_spatial",
        "worst_intensity_diameter_mm",
        "worst_intensity_force_pair_n",
        "worst_spatial_diameter_mm",
        "worst_spatial_force_pair_n",
        "runtime_s",
        "status",
        "failure",
        "is_nondominated",
    ]


def _write_csv(rows: list[dict[str, object]], variable_names: tuple[str, ...]) -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with _CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_fieldnames(variable_names))
        writer.writeheader()
        writer.writerows(rows)


def _read_previous_rows(
    variable_names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    if not _CSV_PATH.is_file():
        return {}
    objective_fields = []
    for _, diameter_mm in _SPHERES:
        objective_fields.extend(
            (
                f"J_intensity_{diameter_mm:g}mm",
                f"J_spatial_{diameter_mm:g}mm",
            )
        )
    numeric_fields = (
        *variable_names,
        *objective_fields,
        "J_intensity",
        "J_spatial",
        "worst_intensity_diameter_mm",
        "worst_spatial_diameter_mm",
        "runtime_s",
    )
    previous = {}
    with _CSV_PATH.open(newline="", encoding="utf-8") as input_file:
        for raw_row in csv.DictReader(input_file):
            row: dict[str, object] = dict(raw_row)
            row["is_baseline"] = raw_row["is_baseline"] == "True"
            row["is_nondominated"] = raw_row.get("is_nondominated") == "True"
            for field in numeric_fields:
                value = raw_row.get(field, "")
                if value:
                    row[field] = float(value)
            previous[str(row["design"])] = row
    return previous


def _nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    mask = np.ones(len(objectives), dtype=bool)
    for index, value in enumerate(objectives):
        dominates = np.all(objectives >= value, axis=1) & np.any(
            objectives > value,
            axis=1,
        )
        mask[index] = not np.any(dominates)
    return mask


def _write_plot(rows: list[dict[str, object]]) -> tuple[float, float, list[str]]:
    passed = [row for row in rows if row["status"] == "PASS"]
    if len(passed) < 2:
        raise RuntimeError("at least two valid morphology evaluations are required")
    objectives = np.array(
        [[row["J_intensity"], row["J_spatial"]] for row in passed],
        dtype=np.float64,
    )
    nondominated = _nondominated_mask(objectives)

    figure, axes = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    axes.scatter(
        objectives[~nondominated, 0],
        objectives[~nondominated, 1],
        label="sampled morphology",
        alpha=0.75,
    )
    axes.scatter(
        objectives[nondominated, 0],
        objectives[nondominated, 1],
        marker="D",
        label="nondominated",
    )
    baseline_index = next(
        index for index, row in enumerate(passed) if row["is_baseline"]
    )
    axes.scatter(
        objectives[baseline_index, 0],
        objectives[baseline_index, 1],
        marker="*",
        s=180,
        color="black",
        label="baseline",
        zorder=4,
    )
    axes.annotate(
        "baseline",
        objectives[baseline_index],
        xytext=(6, 6),
        textcoords="offset points",
    )
    axes.set_xlabel("worst-diameter J_intensity")
    axes.set_ylabel("worst-diameter J_spatial")
    axes.set_title("LUMO sensing-objective morphology sample")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(_PLOT_PATH, dpi=180)
    plt.close(figure)

    pearson = float(pearsonr(objectives[:, 0], objectives[:, 1]).statistic)
    spearman = float(spearmanr(objectives[:, 0], objectives[:, 1]).statistic)
    pareto_names = [
        str(row["design"])
        for row, is_nondominated in zip(passed, nondominated, strict=True)
        if is_nondominated
    ]
    return pearson, spearman, pareto_names


def main() -> None:
    wall_start_s = perf_counter()
    space = _make_design_space()
    named_candidates = [("baseline", _baseline_candidate(space))]
    named_candidates.extend(
        (f"sobol_{index:02d}", candidate)
        for index, candidate in enumerate(
            _sobol_candidates(space, _ADDITIONAL_MORPHOLOGIES),
            start=1,
        )
    )
    previous_rows = _read_previous_rows(space.variable_names)
    stored_rows = dict(previous_rows)

    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    rows: list[dict[str, object]] = []
    with ExitStack() as resources:
        sphere_resources = tuple(
            (
                resources.enter_context(as_file(resource_root.joinpath(filename))),
                diameter_mm,
            )
            for filename, diameter_mm in _SPHERES
        )
        for design_index, (name, candidate) in enumerate(named_candidates, start=1):
            print(
                f"[{design_index:02d}/{len(named_candidates):02d}] {name}",
                flush=True,
            )
            previous = previous_rows.get(name)
            if previous is not None and previous.get("status") == "PASS":
                if any(
                    not np.isclose(
                        float(previous[parameter_name]),
                        parameter_value,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    for parameter_name, parameter_value in candidate.items()
                ):
                    raise RuntimeError(
                        f"stored parameters for {name} do not match the Sobol sample"
                    )
                intensity_by_diameter = np.array(
                    [
                        previous[f"J_intensity_{diameter_mm:g}mm"]
                        for _, diameter_mm in _SPHERES
                    ],
                    dtype=np.float64,
                )
                spatial_by_diameter = np.array(
                    [
                        previous[f"J_spatial_{diameter_mm:g}mm"]
                        for _, diameter_mm in _SPHERES
                    ],
                    dtype=np.float64,
                )
                previous["worst_intensity_diameter_mm"] = _SPHERES[
                    int(np.argmin(intensity_by_diameter))
                ][1]
                previous["worst_spatial_diameter_mm"] = _SPHERES[
                    int(np.argmin(spatial_by_diameter))
                ][1]
                previous.setdefault("worst_intensity_force_pair_n", "")
                previous.setdefault("worst_spatial_force_pair_n", "")
                rows.append(previous)
                print(
                    "  reused completed evaluation | "
                    f"runtime={float(previous['runtime_s']):.3f} s",
                    flush=True,
                )
                continue
            row: dict[str, object] = {
                "design": name,
                "is_baseline": name == "baseline",
                **candidate,
            }
            evaluation_start_s = perf_counter()
            try:
                fingertip = Fingertip(space.to_parameters(candidate))
                trials = _make_trials(fingertip, sphere_resources)
                evaluation = evaluate_contact_sensing(fingertip, trials)
                (
                    diameter_intensity,
                    diameter_spatial,
                    worst_intensity,
                    worst_spatial,
                ) = sensing_objectives(
                    evaluation.response_matrix,
                    no_contact_response=evaluation.no_contact_response,
                )
                intensity_pairs, spatial_pairs = _worst_force_pairs(
                    evaluation.no_contact_response,
                    evaluation.response_matrix,
                )
                for sphere_index, (_, diameter_mm) in enumerate(_SPHERES):
                    row[f"J_intensity_{diameter_mm:g}mm"] = float(
                        diameter_intensity[sphere_index]
                    )
                    row[f"J_spatial_{diameter_mm:g}mm"] = float(
                        diameter_spatial[sphere_index]
                    )
                row["J_intensity"] = worst_intensity
                row["J_spatial"] = worst_spatial
                intensity_sphere_index = int(np.argmin(diameter_intensity))
                spatial_sphere_index = int(np.argmin(diameter_spatial))
                row["worst_intensity_diameter_mm"] = _SPHERES[intensity_sphere_index][1]
                row["worst_spatial_diameter_mm"] = _SPHERES[spatial_sphere_index][1]
                intensity_pair = intensity_pairs[intensity_sphere_index]
                spatial_pair = spatial_pairs[spatial_sphere_index]
                row["worst_intensity_force_pair_n"] = "-".join(
                    f"{evaluation.force_targets_n[index]:g}" for index in intensity_pair
                )
                row["worst_spatial_force_pair_n"] = "-".join(
                    f"{evaluation.force_targets_n[index]:g}" for index in spatial_pair
                )
                row["status"] = "PASS"
                row["failure"] = ""
                print(
                    f"  J_intensity={worst_intensity:.9e} | "
                    f"J_spatial={worst_spatial:.9e}",
                    flush=True,
                )
                del evaluation, trials, fingertip
            except Exception as error:  # report invalid numerical designs
                row["status"] = "FAIL"
                row["failure"] = f"{type(error).__name__}: {error}"
                print(f"  FAIL: {row['failure']}", flush=True)
            row["runtime_s"] = perf_counter() - evaluation_start_s
            rows.append(row)
            stored_rows[name] = row
            _write_csv(
                [
                    stored_rows[candidate_name]
                    for candidate_name, _ in named_candidates
                    if candidate_name in stored_rows
                ],
                space.variable_names,
            )
            print(f"  runtime={row['runtime_s']:.3f} s", flush=True)

    pearson, spearman, pareto_names = _write_plot(rows)
    for row in rows:
        row["is_nondominated"] = row["design"] in pareto_names
    _write_csv(rows, space.variable_names)
    passed_count = sum(row["status"] == "PASS" for row in rows)
    completed_runtimes = np.array(
        [row["runtime_s"] for row in rows if row["status"] == "PASS"],
        dtype=np.float64,
    )
    print()
    print(f"completed morphologies: {passed_count}/{len(rows)}")
    print(f"Pearson correlation:  {pearson:+.6f}")
    print(f"Spearman correlation: {spearman:+.6f}")
    print(f"nondominated samples: {', '.join(pareto_names)}")
    print(
        "completed runtime [s]: "
        f"min={completed_runtimes.min():.3f} | "
        f"Q1={np.quantile(completed_runtimes, 0.25):.3f} | "
        f"median={np.median(completed_runtimes):.3f} | "
        f"Q3={np.quantile(completed_runtimes, 0.75):.3f} | "
        f"max={completed_runtimes.max():.3f}"
    )
    print("worst-case determinants:")
    for row in rows:
        if row["status"] != "PASS":
            continue
        intensity_pair = row.get("worst_intensity_force_pair_n") or "not retained"
        spatial_pair = row.get("worst_spatial_force_pair_n") or "not retained"
        print(
            f"  {row['design']}: intensity="
            f"{float(row['worst_intensity_diameter_mm']):g} mm / "
            f"{intensity_pair} N; spatial="
            f"{float(row['worst_spatial_diameter_mm']):g} mm / "
            f"{spatial_pair} N"
        )
    if passed_count < 10:
        conclusion = "fewer than 10 morphologies completed; no trade-off conclusion"
    elif pearson <= -0.4 and len(pareto_names) >= 2:
        conclusion = "objectives conflict and show a visible sampled Pareto trade-off"
    elif abs(spearman) < 0.4 and len(pareto_names) >= 2:
        conclusion = (
            "objectives have weak rank correlation and a visible sampled "
            "Pareto trade-off"
        )
    elif pearson >= 0.4:
        conclusion = "objectives are positively correlated in this sample"
    else:
        conclusion = "objective dependence is weak or inconclusive in this sample"
    print(f"conclusion: {conclusion}")
    print(f"CSV:  {_CSV_PATH}")
    print(f"plot: {_PLOT_PATH}")
    print(f"total runtime: {perf_counter() - wall_start_s:.3f} s")


if __name__ == "__main__":
    main()
