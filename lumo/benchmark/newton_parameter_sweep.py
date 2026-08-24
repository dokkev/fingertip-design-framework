"""Benchmark numerical sensitivity of settled 20 N indentation."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.resources import as_file, files
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.newton import Indenter
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_TARGET_FORCE_N = 20.0
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_MAX_SIM_TIME_S = 30.0

_FORCE_TOLERANCE_N = 1.0
_SETTLE_DURATION_S = 5.0e-3
_MAX_BONDED_DRIFT_M = 1.0e-8
_MAX_CARRIER_PENETRATION_M = 1.0e-5

_BASELINE_ELEMENT_SIZE_MM = 1.0
_BASELINE_SIM_FREQUENCY_HZ = 1.0e3
_BASELINE_ITERATIONS = 10
_BASELINE_SOFT_CONTACT_MARGIN_M = 1.0e-4
_BASELINE_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6

# This is only a reporting convention for this script, not a production gate.
_TRAVEL_CLOSE_TO_BASELINE_PERCENT = 0.5

_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
)
_CONTACT_X_MM = (-7.5, 0.0, 7.5)
_DEFAULT_OUTPUT_PATH = Path(
    "output/benchmark/newton_parameter_sweep.json"
)


@dataclass
class _SweepResult:
    """Lightweight measurements from one released indentation runtime."""

    passed: bool = False
    failure: str = ""
    reaction_force_n: float = float("nan")
    force_error_n: float = float("nan")
    maximum_particle_speed_m_s: float = float("nan")
    force_change_n: float = float("nan")
    maximum_bonded_drift_m: float = float("nan")
    maximum_carrier_penetration_m: float = float("nan")
    finite_state: bool = False
    indenter_contact_count: int = 0
    travel_from_zero_contact_m: float = float("nan")
    simulation_ticks: int = 0
    wall_time_s: float = float("nan")
    particle_count: int = 0
    tetrahedron_count: int = 0


def _result_record(result: _SweepResult) -> dict[str, object]:
    """Return one strict-JSON-compatible result record."""
    record = asdict(result)
    for name, value in record.items():
        if isinstance(value, float) and not isfinite(value):
            record[name] = None
    return record


def _family_record(
    *,
    unit: str,
    baseline_value: float,
    baseline: _SweepResult,
    results: list[tuple[float, _SweepResult]],
) -> dict[str, object]:
    runs = []
    for value, result in results:
        travel_change_m: float | None = None
        travel_change_percent: float | None = None
        if (
            baseline.passed
            and result.passed
            and baseline.travel_from_zero_contact_m != 0.0
        ):
            travel_change_m = (
                result.travel_from_zero_contact_m
                - baseline.travel_from_zero_contact_m
            )
            travel_change_percent = (
                100.0
                * travel_change_m
                / baseline.travel_from_zero_contact_m
            )

        runtime_ratio: float | None = None
        if (
            isfinite(result.wall_time_s)
            and isfinite(baseline.wall_time_s)
            and baseline.wall_time_s > 0.0
        ):
            runtime_ratio = result.wall_time_s / baseline.wall_time_s

        runs.append(
            {
                "value": value,
                "result": _result_record(result),
                "comparison_to_baseline": {
                    "travel_change_m": travel_change_m,
                    "travel_change_percent": travel_change_percent,
                    "runtime_ratio": runtime_ratio,
                },
            }
        )

    return {
        "unit": unit,
        "baseline_value": baseline_value,
        "runs": runs,
    }


def _write_results(
    output_path: Path,
    *,
    include_fine_mesh: bool,
    include_matrix: bool,
    warmup: _SweepResult,
    baseline: _SweepResult,
    mesh_results: list[tuple[float, _SweepResult]],
    frequency_results: list[tuple[float, _SweepResult]],
    iteration_results: list[tuple[float, _SweepResult]],
    margin_results: list[tuple[float, _SweepResult]],
    stiffness_results: list[tuple[float, _SweepResult]],
    matrix_results: list[tuple[float, float, _SweepResult]],
) -> None:
    payload = {
        "schema_version": 3,
        "benchmark": "newton_parameter_sweep",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "options": {
            "fine_mesh": include_fine_mesh,
            "robustness_matrix": include_matrix,
        },
        "representative_case": {
            "sphere_diameter_mm": 10.0,
            "contact_x_mm": 0.0,
            "target_force_n": _TARGET_FORCE_N,
            "initial_clearance_m": _INITIAL_CLEARANCE_M,
            "approach_speed_m_s": _APPROACH_SPEED_M_S,
            "maximum_simulation_time_s": _MAX_SIM_TIME_S,
        },
        "baseline_numerical_parameters": {
            "element_size_mm": _BASELINE_ELEMENT_SIZE_MM,
            "sim_frequency_hz": _BASELINE_SIM_FREQUENCY_HZ,
            "solver_vbd_iterations": _BASELINE_ITERATIONS,
            "soft_contact_margin_m": _BASELINE_SOFT_CONTACT_MARGIN_M,
            "carrier_contact_stiffness_n_m": (
                _BASELINE_CARRIER_CONTACT_STIFFNESS_N_M
            ),
        },
        "acceptance": {
            "force_tolerance_n": _FORCE_TOLERANCE_N,
            "settle_duration_s": _SETTLE_DURATION_S,
            "maximum_bonded_drift_m": _MAX_BONDED_DRIFT_M,
            "maximum_carrier_penetration_m": (
                _MAX_CARRIER_PENETRATION_M
            ),
        },
        "warmup": _result_record(warmup),
        "baseline": _result_record(baseline),
        "families": {
            "mesh_element_size": _family_record(
                unit="mm",
                baseline_value=_BASELINE_ELEMENT_SIZE_MM,
                baseline=baseline,
                results=mesh_results,
            ),
            "simulation_frequency": _family_record(
                unit="Hz",
                baseline_value=_BASELINE_SIM_FREQUENCY_HZ,
                baseline=baseline,
                results=frequency_results,
            ),
            "solver_vbd_iterations": _family_record(
                unit="count",
                baseline_value=float(_BASELINE_ITERATIONS),
                baseline=baseline,
                results=iteration_results,
            ),
            "soft_contact_margin": _family_record(
                unit="m",
                baseline_value=_BASELINE_SOFT_CONTACT_MARGIN_M,
                baseline=baseline,
                results=margin_results,
            ),
            "carrier_contact_stiffness": _family_record(
                unit="N/m",
                baseline_value=(
                    _BASELINE_CARRIER_CONTACT_STIFFNESS_N_M
                ),
                baseline=baseline,
                results=stiffness_results,
            ),
        },
        "robustness_matrix": [
            {
                "sphere_diameter_mm": diameter_mm,
                "contact_x_mm": contact_x_mm,
                "result": _result_record(result),
            }
            for diameter_mm, contact_x_mm, result in matrix_results
        ],
        "production_defaults_changed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _carrier_interior_depths_m(
    positions_m: np.ndarray,
    *,
    carrier_cross_section: Polygon,
    carrier_y_limits_m: tuple[float, float],
) -> np.ndarray:
    """Return analytic cross-section depth for points inside the carrier."""
    y_min_m, y_max_m = carrier_y_limits_m
    inside = (
        (positions_m[:, 1] > y_min_m)
        & (positions_m[:, 1] < y_max_m)
        & contains_xy(
            carrier_cross_section,
            1.0e3 * positions_m[:, 0],
            1.0e3 * positions_m[:, 2],
        )
    )
    depths_m = np.zeros(positions_m.shape[0], dtype=np.float64)
    if np.any(inside):
        depths_m[inside] = 1.0e-3 * distance(
            carrier_cross_section.boundary,
            points(
                1.0e3 * positions_m[inside, 0],
                1.0e3 * positions_m[inside, 2],
            ),
        )
    return depths_m


def _run_case(
    fingertip: Fingertip,
    urdf_path: Path,
    *,
    sphere_diameter_mm: float = 10.0,
    contact_x_mm: float = 0.0,
    element_size_mm: float = _BASELINE_ELEMENT_SIZE_MM,
    sim_frequency_hz: float = _BASELINE_SIM_FREQUENCY_HZ,
    iterations: int = _BASELINE_ITERATIONS,
    soft_contact_margin_m: float = _BASELINE_SOFT_CONTACT_MARGIN_M,
    carrier_contact_stiffness_n_m: float = (
        _BASELINE_CARRIER_CONTACT_STIFFNESS_N_M
    ),
) -> _SweepResult:
    local_surface_z_mm = (
        fingertip.silicone.ellipse_center_z_mm
        - semiellipse_depth_at_x_mm(
            half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
            height_mm=fingertip.silicone.ellipse_radius_z_mm,
            x_mm=contact_x_mm,
        )
    )
    sphere_radius_m = 0.5e-3 * sphere_diameter_mm
    initial_tf = wp.transform(
        wp.vec3(
            1.0e-3 * contact_x_mm,
            0.0,
            1.0e-3 * local_surface_z_mm
            - _INITIAL_CLEARANCE_M
            - sphere_radius_m,
        ),
        wp.quat_identity(),
    )
    trial = DesignTrial(
        name=(
            f"sphere_{sphere_diameter_mm:g}mm_x{contact_x_mm:+g}mm"
        ),
        urdf_path=urdf_path,
        initial_tf=initial_tf,
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=_APPROACH_SPEED_M_S,
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )
    study = DesignStudy(
        fingertip,
        (trial,),
        sim_frequency=sim_frequency_hz,
        force_tolerance_n=_FORCE_TOLERANCE_N,
        settle_duration_s=_SETTLE_DURATION_S,
        element_size_mm=element_size_mm,
        iterations=iterations,
        soft_contact_margin_m=soft_contact_margin_m,
        carrier_contact_stiffness_n_m=carrier_contact_stiffness_n_m,
    )

    result: _SweepResult | None = None
    simulation_wall_time_s: float | None = None

    def inspect_trial(
        completed_trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal result, simulation_wall_time_s
        wp.synchronize()
        simulation_wall_time_s = perf_counter() - start_time
        if (
            completed_trial.travel_m is None
            or completed_trial.reaction_force_n is None
            or completed_trial.maximum_particle_speed_m_s is None
            or completed_trial.force_change_n is None
        ):
            raise RuntimeError("indentation completed without scalar results")

        particle_q = simulation.state.particle_q.numpy()
        particle_qd = simulation.state.particle_qd.numpy()
        finite_state = bool(
            np.all(np.isfinite(particle_q))
            and np.all(np.isfinite(particle_qd))
        )
        bonded_indices = (
            simulation.fingertip_model.bonded_particle_indices.numpy()
        )
        contact_count = simulation.soft_contact_count(
            indenter.body_index
        )
        particle_count = int(
            simulation.fingertip_mesh.silicone.vertex_count
        )
        tet_indices = np.asarray(
            simulation.fingertip_mesh.silicone.tet_indices,
            dtype=np.int32,
        ).reshape(-1, 4)

        maximum_bonded_drift_m = float("inf")
        maximum_carrier_penetration_m = float("inf")
        if finite_state:
            bonded_drift_m = np.linalg.norm(
                particle_q[bonded_indices]
                - simulation.fingertip_model.bonded_local_positions.numpy(),
                axis=1,
            )
            maximum_bonded_drift_m = float(bonded_drift_m.max())

            nonbonded = np.ones(particle_q.shape[0], dtype=bool)
            nonbonded[bonded_indices] = False
            carrier_cross_section = Polygon(
                simulation.fingertip.carrier.cross_section
            )
            carrier_vertices = np.asarray(
                simulation.fingertip_mesh.carrier.vertices,
                dtype=np.float64,
            )
            carrier_y_limits_m = (
                float(carrier_vertices[:, 1].min()),
                float(carrier_vertices[:, 1].max()),
            )
            particle_depths_m = _carrier_interior_depths_m(
                particle_q,
                carrier_cross_section=carrier_cross_section,
                carrier_y_limits_m=carrier_y_limits_m,
            )
            particle_depths_m[~nonbonded] = 0.0
            tet_depths_m = _carrier_interior_depths_m(
                particle_q[tet_indices].mean(axis=1),
                carrier_cross_section=carrier_cross_section,
                carrier_y_limits_m=carrier_y_limits_m,
            )
            maximum_carrier_penetration_m = max(
                float(particle_depths_m.max()),
                float(tet_depths_m.max()),
            )

        force_error_n = abs(
            completed_trial.reaction_force_n - _TARGET_FORCE_N
        )
        failures = []
        if force_error_n > _FORCE_TOLERANCE_N:
            failures.append("force tolerance")
        if not finite_state:
            failures.append("non-finite state")
        if contact_count == 0:
            failures.append("no indenter contact")
        if maximum_bonded_drift_m > _MAX_BONDED_DRIFT_M:
            failures.append("bonded drift")
        if maximum_carrier_penetration_m > _MAX_CARRIER_PENETRATION_M:
            failures.append("carrier penetration")

        result = _SweepResult(
            passed=not failures,
            failure=", ".join(failures),
            reaction_force_n=completed_trial.reaction_force_n,
            force_error_n=force_error_n,
            maximum_particle_speed_m_s=(
                completed_trial.maximum_particle_speed_m_s
            ),
            force_change_n=completed_trial.force_change_n,
            maximum_bonded_drift_m=maximum_bonded_drift_m,
            maximum_carrier_penetration_m=(
                maximum_carrier_penetration_m
            ),
            finite_state=finite_state,
            indenter_contact_count=contact_count,
            travel_from_zero_contact_m=(
                completed_trial.travel_m - _INITIAL_CLEARANCE_M
            ),
            simulation_ticks=completed_trial.step_count,
            particle_count=particle_count,
            tetrahedron_count=int(tet_indices.shape[0]),
        )

    wp.synchronize()
    start_time = perf_counter()
    try:
        study.run(inspect_trial=inspect_trial)
        wp.synchronize()
    except Exception as error:
        wp.synchronize()
        return _SweepResult(
            failure=f"{type(error).__name__}: {error}",
            wall_time_s=perf_counter() - start_time,
        )

    if result is None:
        return _SweepResult(
            failure="indentation finished without live-state inspection",
            wall_time_s=perf_counter() - start_time,
        )
    result.wall_time_s = (
        simulation_wall_time_s
        if simulation_wall_time_s is not None
        else perf_counter() - start_time
    )
    return result


def _print_family(
    name: str,
    unit: str,
    baseline_value: float,
    baseline: _SweepResult,
    results: list[tuple[float, _SweepResult]],
) -> None:
    print()
    print(name)
    print(
        f"{'value':>15} {'F[N]':>9} {'Ferr[N]':>10} "
        f"{'travel[mm]':>11} "
        f"{'dtravel[mm]':>12} {'dtravel[%]':>11} {'pen[um]':>9} "
        f"{'ticks':>7} {'wall[s]':>9} {'ratio':>7} "
        f"{'particles':>10} {'tets':>9} {'status':>7}"
    )
    for value, result in results:
        travel_change_mm = float("nan")
        travel_change_percent = float("nan")
        runtime_ratio = float("nan")
        if baseline.wall_time_s > 0.0:
            runtime_ratio = result.wall_time_s / baseline.wall_time_s
        if (
            baseline.passed
            and result.passed
            and baseline.travel_from_zero_contact_m != 0.0
        ):
            travel_change_m = (
                result.travel_from_zero_contact_m
                - baseline.travel_from_zero_contact_m
            )
            travel_change_mm = 1.0e3 * travel_change_m
            travel_change_percent = (
                100.0
                * travel_change_m
                / baseline.travel_from_zero_contact_m
            )

        value_text = f"{value:g} {unit}".strip()
        print(
            f"{value_text:>15} {result.reaction_force_n:9.4f} "
            f"{result.force_error_n:10.3e} "
            f"{1.0e3 * result.travel_from_zero_contact_m:11.4f} "
            f"{travel_change_mm:+12.4f} "
            f"{travel_change_percent:+11.3f} "
            f"{1.0e6 * result.maximum_carrier_penetration_m:9.3f} "
            f"{result.simulation_ticks:7d} "
            f"{result.wall_time_s:9.3f} {runtime_ratio:7.2f} "
            f"{result.particle_count:10d} "
            f"{result.tetrahedron_count:9d} "
            f"{'PASS' if result.passed else 'FAIL':>7}"
        )
        print(
            f"  diagnostics: vmax="
            f"{result.maximum_particle_speed_m_s:.3e} m/s, "
            f"dF={result.force_change_n:.3e} N, "
            f"bond={result.maximum_bonded_drift_m:.3e} m, "
            f"contacts={result.indenter_contact_count}, "
            f"finite={'PASS' if result.finite_state else 'FAIL'}"
        )
        if result.failure:
            print(f"  failure: {result.failure}")

    if not any(value == baseline_value for value, _ in results):
        raise RuntimeError(f"{name} does not include its baseline value")


def _print_summary(
    name: str,
    baseline: _SweepResult,
    results: list[tuple[float, _SweepResult]],
    *,
    cheaper_value: float | None,
    expensive_value: float | None,
) -> None:
    print(f"{name}:")
    if not baseline.passed or not isfinite(
        baseline.travel_from_zero_contact_m
    ) or baseline.travel_from_zero_contact_m == 0.0:
        print("  baseline failed; sensitivity and cost comparison are inconclusive")
        return

    successful_changes = []
    for _, result in results:
        if result.passed:
            successful_changes.append(
                100.0
                * (
                    result.travel_from_zero_contact_m
                    - baseline.travel_from_zero_contact_m
                )
                / baseline.travel_from_zero_contact_m
            )
    maximum_change_percent = max(
        (abs(change) for change in successful_changes),
        default=float("nan"),
    )
    sensitivity = (
        "sensitive"
        if maximum_change_percent > _TRAVEL_CLOSE_TO_BASELINE_PERCENT
        else "close"
    )
    print(
        f"  successful travel results are {sensitivity} under the script-local "
        f"{_TRAVEL_CLOSE_TO_BASELINE_PERCENT:g}% convention; maximum "
        f"absolute change={maximum_change_percent:.3f}%"
    )
    failed_values = [value for value, result in results if not result.passed]
    if failed_values:
        print(f"  failed values: {', '.join(f'{value:g}' for value in failed_values)}")

    if cheaper_value is None and expensive_value is None:
        smallest_value, smallest = min(results, key=lambda item: item[0])
        largest_value, largest = max(results, key=lambda item: item[0])
        print(
            "  no cost ordering is assumed for this parameter; "
            f"smallest={smallest_value:g} ({'PASS' if smallest.passed else 'FAIL'}), "
            f"largest={largest_value:g} ({'PASS' if largest.passed else 'FAIL'})"
        )

    for description, candidate_value in (
        ("nominally cheaper", cheaper_value),
        ("nominally more expensive", expensive_value),
    ):
        if candidate_value is None:
            continue
        candidate = next(
            result
            for value, result in results
            if value == candidate_value
        )
        if not candidate.passed:
            print(
                f"  {description} value {candidate_value:g}: FAIL "
                f"({candidate.failure})"
            )
            continue
        travel_change_percent = (
            100.0
            * (
                candidate.travel_from_zero_contact_m
                - baseline.travel_from_zero_contact_m
            )
            / baseline.travel_from_zero_contact_m
        )
        runtime_ratio = candidate.wall_time_s / baseline.wall_time_s
        penetration_change_um = 1.0e6 * (
            candidate.maximum_carrier_penetration_m
            - baseline.maximum_carrier_penetration_m
        )
        close = (
            abs(travel_change_percent)
            <= _TRAVEL_CLOSE_TO_BASELINE_PERCENT
        )
        conclusion = (
            f"appears acceptable={'yes' if close else 'no'}"
            if description == "nominally cheaper"
            else f"measurable travel change={'no' if close else 'yes'}"
        )
        print(
            f"  {description} value {candidate_value:g}: "
            f"travel change={travel_change_percent:+.3f}%, "
            f"penetration change={penetration_change_um:+.3f} um, "
            f"runtime={runtime_ratio:.2f}x, "
            f"{conclusion} under the script-local convention"
        )


def _print_matrix(
    results: list[tuple[float, float, _SweepResult]],
) -> None:
    print()
    print("Baseline 3 x 3 robustness matrix")
    print(
        f"{'sphere[mm]':>11} {'x[mm]':>8} {'Ferr[N]':>10} "
        f"{'travel[mm]':>11} {'pen[um]':>9} {'ticks':>7} "
        f"{'wall[s]':>9} {'status':>7}"
    )
    for diameter_mm, contact_x_mm, result in results:
        print(
            f"{diameter_mm:11.1f} {contact_x_mm:+8.1f} "
            f"{result.force_error_n:10.3e} "
            f"{1.0e3 * result.travel_from_zero_contact_m:11.4f} "
            f"{1.0e6 * result.maximum_carrier_penetration_m:9.3f} "
            f"{result.simulation_ticks:7d} {result.wall_time_s:9.3f} "
            f"{'PASS' if result.passed else 'FAIL':>7}"
        )
        if result.failure:
            print(f"  failure: {result.failure}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep numerical Newton parameters for one settled 20 N "
            "spherical indentation."
        )
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="include the expensive 0.5 mm mesh case",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="also run the baseline 3-sphere by 3-location robustness matrix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_PATH,
        help=(
            "JSON result path written after all requested runs finish "
            f"(default: {_DEFAULT_OUTPUT_PATH})"
        ),
    )
    arguments = parser.parse_args()

    fingertip = Fingertip(FingertipParameters())
    resource_root = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
    )

    with ExitStack() as resources:
        sphere_paths = {
            diameter_mm: resources.enter_context(
                as_file(resource_root.joinpath(filename))
            )
            for filename, diameter_mm in _SPHERES
        }
        representative_path = sphere_paths[10.0]

        print("Warming the current baseline before wall-time measurements...")
        warmup = _run_case(fingertip, representative_path)
        if not warmup.passed:
            print(f"warm-up warning: {warmup.failure}")

        print("Running measured current baseline...")
        baseline = _run_case(fingertip, representative_path)

        mesh_values = [1.5, 1.0, 0.75]
        if arguments.fine:
            mesh_values.append(0.5)
        mesh_results = []
        print("Running mesh element-size family...")
        for value in mesh_values:
            print(f"  element_size_mm={value:g}")
            result = (
                baseline
                if value == _BASELINE_ELEMENT_SIZE_MM
                else _run_case(
                    fingertip,
                    representative_path,
                    element_size_mm=value,
                )
            )
            mesh_results.append((value, result))

        frequency_values = (500.0, 1000.0, 2000.0)
        frequency_results = []
        print("Running simulation-frequency family...")
        for value in frequency_values:
            print(f"  sim_frequency_hz={value:g}")
            result = (
                baseline
                if value == _BASELINE_SIM_FREQUENCY_HZ
                else _run_case(
                    fingertip,
                    representative_path,
                    sim_frequency_hz=value,
                )
            )
            frequency_results.append((value, result))

        iteration_values = (5, 10, 20)
        iteration_results = []
        print("Running SolverVBD-iteration family...")
        for value in iteration_values:
            print(f"  iterations={value}")
            result = (
                baseline
                if value == _BASELINE_ITERATIONS
                else _run_case(
                    fingertip,
                    representative_path,
                    iterations=value,
                )
            )
            iteration_results.append((float(value), result))

        margin_values = (0.0, 5.0e-5, 1.0e-4, 2.0e-4)
        margin_results = []
        print("Running soft-contact-margin family...")
        for value in margin_values:
            print(f"  soft_contact_margin_m={value:.1e}")
            result = (
                baseline
                if value == _BASELINE_SOFT_CONTACT_MARGIN_M
                else _run_case(
                    fingertip,
                    representative_path,
                    soft_contact_margin_m=value,
                )
            )
            margin_results.append((value, result))

        stiffness_values = (5.0e5, 1.0e6, 2.0e6)
        stiffness_results = []
        print("Running carrier-contact-stiffness family...")
        for value in stiffness_values:
            print(f"  carrier_contact_stiffness_n_m={value:.1e}")
            result = (
                baseline
                if value
                == _BASELINE_CARRIER_CONTACT_STIFFNESS_N_M
                else _run_case(
                    fingertip,
                    representative_path,
                    carrier_contact_stiffness_n_m=value,
                )
            )
            stiffness_results.append((value, result))

        _print_family(
            "Mesh element size",
            "mm",
            _BASELINE_ELEMENT_SIZE_MM,
            baseline,
            mesh_results,
        )
        _print_family(
            "Simulation frequency",
            "Hz",
            _BASELINE_SIM_FREQUENCY_HZ,
            baseline,
            frequency_results,
        )
        _print_family(
            "SolverVBD iterations",
            "",
            float(_BASELINE_ITERATIONS),
            baseline,
            iteration_results,
        )
        _print_family(
            "Soft contact margin",
            "m",
            _BASELINE_SOFT_CONTACT_MARGIN_M,
            baseline,
            margin_results,
        )
        _print_family(
            "Carrier contact stiffness",
            "N/m",
            _BASELINE_CARRIER_CONTACT_STIFFNESS_N_M,
            baseline,
            stiffness_results,
        )

        matrix_results = []
        if arguments.matrix:
            print("Running baseline-only 3 x 3 robustness matrix...")
            for _, diameter_mm in _SPHERES:
                for contact_x_mm in _CONTACT_X_MM:
                    print(
                        f"  sphere={diameter_mm:g} mm, "
                        f"x={contact_x_mm:+g} mm"
                    )
                    if diameter_mm == 10.0 and contact_x_mm == 0.0:
                        result = baseline
                    else:
                        result = _run_case(
                            fingertip,
                            sphere_paths[diameter_mm],
                            sphere_diameter_mm=diameter_mm,
                            contact_x_mm=contact_x_mm,
                        )
                    matrix_results.append(
                        (diameter_mm, contact_x_mm, result)
                    )
            _print_matrix(matrix_results)

    print()
    print("Evidence summary")
    _print_summary(
        "Mesh element size",
        baseline,
        mesh_results,
        cheaper_value=1.5,
        expensive_value=min(mesh_values),
    )
    _print_summary(
        "Simulation frequency",
        baseline,
        frequency_results,
        cheaper_value=500.0,
        expensive_value=2000.0,
    )
    _print_summary(
        "SolverVBD iterations",
        baseline,
        iteration_results,
        cheaper_value=5.0,
        expensive_value=20.0,
    )
    _print_summary(
        "Soft contact margin",
        baseline,
        margin_results,
        cheaper_value=None,
        expensive_value=None,
    )
    _print_summary(
        "Carrier contact stiffness",
        baseline,
        stiffness_results,
        cheaper_value=5.0e5,
        expensive_value=2.0e6,
    )
    _write_results(
        arguments.output,
        include_fine_mesh=arguments.fine,
        include_matrix=arguments.matrix,
        warmup=warmup,
        baseline=baseline,
        mesh_results=mesh_results,
        frequency_results=frequency_results,
        iteration_results=iteration_results,
        margin_results=margin_results,
        stiffness_results=stiffness_results,
        matrix_results=matrix_results,
    )
    print("Production defaults were not changed.")
    print(f"JSON results: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
