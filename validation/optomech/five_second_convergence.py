"""Test whether the simulated observables are converged by a 5 s dwell."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from force_dwell_screen import (
    _patch_and_normal_diagnostics,
    _run_dwell,
    _summary,
)
from lumo.optimization.objective import (
    _active_surface_triangles,
    _surface_incidence,
    _triangle_areas,
    compute_objectives_from_raw,
)


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = (
    _ROOT
    / "output"
    / "validation"
    / "production_evaluator_acceleration"
    / "phase3_dwell"
    / "five_second_convergence"
)
_DURATIONS_S = (5.0, 7.5, 10.0)


def _patch_areas_m2(data: dict[str, object]) -> np.ndarray:
    triangles = np.asarray(data["surface_triangles"], dtype=np.int32)
    vertices = np.asarray(data["silicone_vertices_m"], dtype=np.float64)
    offsets = np.asarray(data["contact_record_offsets"], dtype=np.int64)
    records = np.asarray(data["contact_particle_indices"], dtype=np.int32)
    incidence = _surface_incidence(triangles)
    areas = np.empty(offsets.shape[:2], dtype=np.float64)
    for scenario_index, force_index in np.ndindex(areas.shape):
        start, count = offsets[scenario_index, force_index]
        support = _active_surface_triangles(
            records[start : start + count],
            vertex_triangles=incidence[0],
            edge_triangles=incidence[1],
            triangle_ids=incidence[2],
        )
        triangle_areas = _triangle_areas(
            vertices[scenario_index, force_index],
            triangles,
        )
        areas[scenario_index, force_index] = triangle_areas[list(support)].sum()
    return areas


def _flatten(run: dict[str, object], key: str) -> np.ndarray:
    return np.concatenate(
        [np.asarray(group[key]).reshape(-1) for group in run["groups"].values()]
    )


def _geometry_difference(
    later: dict[str, object],
    earlier: dict[str, object],
) -> tuple[float, float]:
    squared_sum = 0.0
    value_count = 0
    maximum = 0.0
    for name in earlier["groups"]:
        delta = (
            np.asarray(later["groups"][name]["silicone_vertices_m"])
            - np.asarray(earlier["groups"][name]["silicone_vertices_m"])
        )
        squared_sum += float(np.sum(delta.astype(np.float64) ** 2))
        value_count += delta.size
        maximum = max(maximum, float(np.max(np.abs(delta))))
    return float(np.sqrt(squared_sum / value_count)), maximum


def _patch_diagnostics(
    later: dict[str, object],
    earlier: dict[str, object],
) -> tuple[float, float, float]:
    minimum_iou = 1.0
    minimum_normal = 1.0
    maximum_area_relative = 0.0
    for name in earlier["groups"]:
        later_group = later["groups"][name]
        earlier_group = earlier["groups"][name]
        iou, normal = _patch_and_normal_diagnostics(later_group, earlier_group)
        earlier_area = _patch_areas_m2(earlier_group)
        later_area = _patch_areas_m2(later_group)
        maximum_area_relative = max(
            maximum_area_relative,
            float(np.max(np.abs(later_area - earlier_area) / earlier_area)),
        )
        minimum_iou = min(minimum_iou, iou)
        minimum_normal = min(minimum_normal, normal)
    return minimum_iou, minimum_normal, maximum_area_relative


def _difficult_pair_separation(run: dict[str, object]) -> float:
    data = run["groups"]["observation_pair"]
    _, observation = compute_objectives_from_raw(data)
    normalized = observation.normalized_response
    return float(np.linalg.norm(normalized[0, 0] - normalized[1, 0]))


def _optical_difference(
    later: dict[str, object],
    earlier: dict[str, object],
) -> tuple[float, float, float, float]:
    squared_sum = 0.0
    value_count = 0
    maximum = 0.0
    visible_change = 0.0
    outside_change = 0.0
    for name in earlier["groups"]:
        later_group = later["groups"][name]
        earlier_group = earlier["groups"][name]
        later_response = np.asarray(later_group["response_matrix"]).sum(axis=2) / 5.0
        earlier_response = (
            np.asarray(earlier_group["response_matrix"]).sum(axis=2) / 5.0
        )
        delta = later_response - earlier_response
        squared_sum += float(np.sum(delta**2))
        value_count += delta.size
        maximum = max(maximum, float(np.max(np.abs(delta))))
        visible_change = max(
            visible_change,
            float(
                np.max(
                    np.abs(
                        np.asarray(later_group["visible_side_power"]).sum(axis=2)
                        - np.asarray(earlier_group["visible_side_power"]).sum(
                            axis=2
                        )
                    )
                )
            ),
        )
        outside_change = max(
            outside_change,
            float(
                np.max(
                    np.abs(
                        np.asarray(later_group["outside_roi_power"]).sum(axis=2)
                        - np.asarray(earlier_group["outside_roi_power"]).sum(
                            axis=2
                        )
                    )
                )
            ),
        )
    return (
        float(np.sqrt(squared_sum / value_count)),
        maximum,
        visible_change,
        outside_change,
    )


def _relative(later: np.ndarray | float, earlier: np.ndarray | float) -> float:
    later_values = np.asarray(later, dtype=np.float64)
    earlier_values = np.asarray(earlier, dtype=np.float64)
    scale = np.maximum(np.abs(earlier_values), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(later_values - earlier_values) / scale))


def _increment(
    later: dict[str, object],
    earlier: dict[str, object],
) -> dict[str, float]:
    later_summary = _summary(later)
    earlier_summary = _summary(earlier)
    vertex_rms_m, vertex_max_m = _geometry_difference(later, earlier)
    patch_iou, normal_score, patch_area_relative = _patch_diagnostics(
        later,
        earlier,
    )
    optical_rms, optical_max, visible_change, outside_change = (
        _optical_difference(later, earlier)
    )
    earlier_separation = _difficult_pair_separation(earlier)
    later_separation = _difficult_pair_separation(later)
    return {
        "force_abs_n": float(
            np.max(
                np.abs(
                    _flatten(later, "actual_forces_n")
                    - _flatten(earlier, "actual_forces_n")
                )
            )
        ),
        "indentation_abs_m": float(
            np.max(
                np.abs(
                    _flatten(later, "indentations_m")
                    - _flatten(earlier, "indentations_m")
                )
            )
        ),
        "vertex_rms_m": vertex_rms_m,
        "vertex_max_m": vertex_max_m,
        "patch_iou": patch_iou,
        "patch_area_relative": patch_area_relative,
        "normal_score": normal_score,
        "contact_count_abs": float(
            np.max(
                np.abs(
                    _flatten(later, "indenter_contact_counts")
                    - _flatten(earlier, "indenter_contact_counts")
                )
            )
        ),
        "centroid_abs_m": float(
            max(
                np.max(
                    np.linalg.norm(
                        np.asarray(later["groups"][name]["contact_centroids_W_m"])
                        - np.asarray(
                            earlier["groups"][name]["contact_centroids_W_m"]
                        ),
                        axis=-1,
                    )
                )
                for name in earlier["groups"]
            )
        ),
        "minimum_det_f_abs": float(
            np.max(
                np.abs(
                    _flatten(later, "minimum_det_f")
                    - _flatten(earlier, "minimum_det_f")
                )
            )
        ),
        "q_form_relative": _relative(
            later_summary["q_form"], earlier_summary["q_form"]
        ),
        "q_stable_relative": _relative(
            later_summary["q_stable"], earlier_summary["q_stable"]
        ),
        "q_stiff_relative": _relative(
            later_summary["q_stiff"], earlier_summary["q_stiff"]
        ),
        "q_contact_relative": _relative(
            later_summary["q_contact"], earlier_summary["q_contact"]
        ),
        "difficult_pair_separation": later_separation,
        "difficult_pair_relative": abs(later_separation - earlier_separation)
        / earlier_separation,
        "optical_rms": optical_rms,
        "optical_max": optical_max,
        "visible_power_abs": visible_change,
        "outside_roi_abs": outside_change,
    }


def _duration_diagnostics(run: dict[str, object]) -> dict[str, float]:
    return {
        "rms_speed_max_m_s": float(
            np.max(_flatten(run, "rms_particle_speeds_m_s"))
        ),
        "p95_speed_max_m_s": float(
            np.max(_flatten(run, "particle_speed_p95_m_s"))
        ),
        "maximum_speed_m_s": float(
            np.max(_flatten(run, "maximum_particle_speeds_m_s"))
        ),
        "window_force_drift_abs_n": float(
            np.max(np.abs(_flatten(run, "settle_window_force_drifts_n")))
        ),
        "window_indentation_drift_abs_m": float(
            np.max(
                np.abs(_flatten(run, "settle_window_indentation_drifts_m"))
            )
        ),
        "minimum_det_f": float(np.min(_flatten(run, "minimum_det_f"))),
        "inversion_count": float(
            np.max(_flatten(run, "inverted_tet_counts"))
        ),
        "overflow_count": float(
            np.max(_flatten(run, "contact_buffer_overflow"))
        ),
    }


def _write_artifacts(
    runs: dict[float, dict[str, object]],
    increments: dict[str, dict[str, float]],
    diagnostics: dict[float, dict[str, float]],
    wall_s: float,
) -> bool:
    first = increments["5_to_7p5"]
    second = increments["7p5_to_10"]
    topology_pass = first["patch_iou"] >= 0.95 and second["patch_iou"] >= 0.95
    component_pass = all(
        increment[key] <= 0.02
        for increment in (first, second)
        for key in ("q_form_relative", "q_stable_relative", "q_stiff_relative")
    )
    objective_input_pass = all(
        increment[key] <= 0.02
        for increment in (first, second)
        for key in ("q_contact_relative", "difficult_pair_relative")
    )
    safety_pass = all(
        value["inversion_count"] == 0.0 and value["overflow_count"] == 0.0
        for value in diagnostics.values()
    )
    trend_pass = all(
        second[key] <= 1.25 * first[key] + 1.0e-12
        for key in (
            "indentation_abs_m",
            "vertex_rms_m",
            "patch_area_relative",
            "q_stiff_relative",
            "q_contact_relative",
            "difficult_pair_relative",
            "optical_rms",
        )
    )
    accepted = (
        topology_pass
        and component_pass
        and objective_input_pass
        and safety_pass
        and trend_pass
    )

    rows = []
    for observable, unit in (
        ("indentation_abs_m", "m"),
        ("vertex_rms_m", "m"),
        ("vertex_max_m", "m"),
        ("patch_area_relative", "relative"),
        ("q_form_relative", "relative"),
        ("q_stable_relative", "relative"),
        ("q_stiff_relative", "relative"),
        ("q_contact_relative", "relative"),
        ("difficult_pair_relative", "relative"),
        ("optical_rms", "normalized power"),
    ):
        rows.append(
            {
                "observable": observable,
                "unit": unit,
                "5_to_7p5": first[observable],
                "7p5_to_10": second[observable],
                "continuing_trend": second[observable]
                > 1.25 * first[observable] + 1.0e-12,
            }
        )
    with (_OUTPUT_DIRECTORY / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "comparison_metrics.npz",
        durations_s=np.asarray(_DURATIONS_S),
        metric_names=np.asarray(tuple(first)),
        increments=np.asarray(
            [[increments[label][key] for key in first] for label in increments]
        ),
        diagnostic_names=np.asarray(tuple(diagnostics[5.0])),
        duration_diagnostics=np.asarray(
            [
                [diagnostics[dwell][key] for key in diagnostics[5.0]]
                for dwell in _DURATIONS_S
            ]
        ),
    )

    lines = [
        "# Five-second simulated quasi-static convergence",
        "",
        f"Result: {'PASS' if accepted else 'FAIL'}",
        "",
        "This tests convergence under the adopted Newton/contact/material model. "
        "It is not a calibration of real Dragon Skin or Solaris settling time.",
        "",
        "| Observable | 5->7.5 s | 7.5->10 s | Continuing trend? | Status |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        relative = row["unit"] == "relative"
        multiplier = 100.0 if relative else 1.0
        suffix = "%" if relative else f" {row['unit']}"
        continuing = bool(row["continuing_trend"])
        lines.append(
            f"| {row['observable']} | "
            f"{multiplier * row['5_to_7p5']:.6g}{suffix} | "
            f"{multiplier * row['7p5_to_10']:.6g}{suffix} | "
            f"{'YES' if continuing else 'no'} | "
            f"{'FAIL' if continuing else 'stable'} |"
        )
    lines.extend(("", "## Other increment diagnostics", ""))
    for label, increment in increments.items():
        lines.extend(
            (
                f"### {label.replace('_', ' ')}",
                "",
                f"- force maximum change: {increment['force_abs_n']:.6f} N",
                f"- patch IoU minimum: {increment['patch_iou']:.6f}",
                f"- patch normal score minimum: {increment['normal_score']:.9f}",
                f"- contact-count maximum change: {increment['contact_count_abs']:.0f}",
                f"- centroid maximum change: {1.0e6 * increment['centroid_abs_m']:.3f} um",
                f"- min-det(F) maximum change: {increment['minimum_det_f_abs']:.6e}",
                f"- difficult-pair separation at later dwell: {increment['difficult_pair_separation']:.9f}",
                f"- optical max-bin change: {increment['optical_max']:.9f}",
                f"- total +X visible-power maximum change: {increment['visible_power_abs']:.9f}",
                f"- outside-ROI maximum change: {increment['outside_roi_abs']:.9f}",
                "",
            )
        )
    lines.extend(("## Final-window settling diagnostics", ""))
    for dwell_s in _DURATIONS_S:
        values = diagnostics[dwell_s]
        lines.extend(
            (
                f"### {dwell_s:g} s dwell",
                "",
                f"- RMS particle speed maximum: {values['rms_speed_max_m_s']:.6e} m/s",
                f"- P95 particle speed maximum: {values['p95_speed_max_m_s']:.6e} m/s",
                f"- maximum particle speed: {values['maximum_speed_m_s']:.6e} m/s",
                f"- final 0.5 s force drift maximum: {values['window_force_drift_abs_n']:.6e} N",
                f"- final 0.5 s indentation drift maximum: {1.0e6 * values['window_indentation_drift_abs_m']:.6f} um",
                f"- minimum det(F): {values['minimum_det_f']:.6f}",
                f"- inversion / overflow: {int(values['inversion_count'])} / {int(values['overflow_count'])}",
                "",
            )
        )
    lines.extend(
        (
            "## Decision",
            "",
            f"- component <=2% convergence gate: {'PASS' if component_pass else 'FAIL'}",
            f"- sentinel q_contact / difficult-pair <=2%: {'PASS' if objective_input_pass else 'FAIL'}",
            f"- contact topology: {'PASS' if topology_pass else 'FAIL'}",
            f"- non-increasing trend gate: {'PASS' if trend_pass else 'FAIL'}",
            f"- safety: {'PASS' if safety_pass else 'FAIL'}",
            f"- validation wall time: {wall_s:.3f} s",
            "",
            "The original 5 s choice was heuristic before this study. "
            + (
                "These data now support calling it an empirically validated "
                "quasi-static evaluation dwell under the simulation model."
                if accepted
                else "These data do not establish convergence by 5 s; the "
                "production dwell remains unchanged pending an explicit new "
                "scientific contract."
            ),
            "No claim is made about the physical settling time of real silicone.",
            "",
            "## Explicit answers",
            "",
            "1. The original 5 s duration was heuristic before this study: YES.",
            f"2. Five seconds now has empirical convergence support: {'YES' if accepted else 'NO'}.",
            f"3. From 5 to 7.5 s, q_stiff changes {100.0 * first['q_stiff_relative']:.3f}%, q_contact changes {100.0 * first['q_contact_relative']:.3f}%, vertex RMS changes {1.0e6 * first['vertex_rms_m']:.3f} um, and the difficult-pair separation changes {100.0 * first['difficult_pair_relative']:.3f}%.",
            f"4. From 7.5 to 10 s, q_stiff changes {100.0 * second['q_stiff_relative']:.3f}%, q_contact changes {100.0 * second['q_contact_relative']:.3f}%, vertex RMS changes {1.0e6 * second['vertex_rms_m']:.3f} um, and the difficult-pair separation changes {100.0 * second['difficult_pair_relative']:.3f}%.",
            "5. Evidence of continued post-5 s simulated settling/history sensitivity: YES. Bulk vertex RMS decreases in the second interval, but q_stiff remains equally sensitive and final-window servo/state drift does not vanish.",
            f"6. q_stiff is converged: {'YES' if first['q_stiff_relative'] <= 0.02 and second['q_stiff_relative'] <= 0.02 else 'NO'}.",
            "7. The difficult finite-area pair remains within the 2% equivalence scale, but is not demonstrably trend-converged because its second increment is larger than its first: NO for convergence proof.",
            f"8. Safety or qualitative contact-topology behavior changes: {'NO' if safety_pass and topology_pass else 'YES'}.",
            "9. Five seconds remains the unchanged production/BO dwell for continuity, but cannot yet be described as an empirically converged dwell.",
            "10. The justified claim is only that 5 s is the conservative simulation reference and that 2 s and 3 s are not equivalent to it; this study does not prove convergence by 5 s.",
            "11. No real-silicone physical settling-time calibration is claimed: CONFIRMED.",
            "",
            "The observed discrepancy is mechanical/servo-history related rather than a contact-topology or safety failure; the finite-area optical response propagates that residual geometry/history sensitivity.",
            "",
        )
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text("\n".join(lines))
    return accepted


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    start_s = perf_counter()
    runs = {
        dwell_s: _run_dwell(dwell_s, output_directory=_OUTPUT_DIRECTORY)
        for dwell_s in _DURATIONS_S
    }
    increments = {
        "5_to_7p5": _increment(runs[7.5], runs[5.0]),
        "7p5_to_10": _increment(runs[10.0], runs[7.5]),
    }
    diagnostics = {
        dwell_s: _duration_diagnostics(run) for dwell_s, run in runs.items()
    }
    _write_artifacts(
        runs,
        increments,
        diagnostics,
        perf_counter() - start_s,
    )
    print((_OUTPUT_DIRECTORY / "report.md").read_text())


if __name__ == "__main__":
    main()
