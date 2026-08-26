"""Compare point/area LED sources and hard/linear longitudinal bins.

Newton is not run. The script reuses the frozen reference and force-ramp
silicone states, rebuilding only the OptiX scene and optical paths.
"""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.optimization.objective import compute_observation_objective
from lumo.ray_tracing import (
    LED,
    OptixScene,
    longitudinal_side_view_power,
    safe_secondary_origins,
    source_inside_silicone,
    trace_bounded_paths,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = (
    _REPOSITORY_ROOT
    / "output"
    / "validation"
    / "full_finger_production_objective_freeze"
    / "nominal_full_finger_objectives.npz"
)
_RAMP_DIRECTORY = (
    _REPOSITORY_ROOT / "output" / "validation" / "quasistatic_ramp_protocol"
)
_INPUTS = (
    ("reference_dwell", _REFERENCE_PATH),
    ("ramp_20s", _RAMP_DIRECTORY / "ramp_20s.npz"),
    ("ramp_10s", _RAMP_DIRECTORY / "ramp_10s.npz"),
    ("ramp_5s", _RAMP_DIRECTORY / "ramp_5s.npz"),
    ("ramp_2p5s", _RAMP_DIRECTORY / "ramp_2p5s.npz"),
)
_OUTPUT_DIRECTORY = (
    _REPOSITORY_ROOT
    / "output"
    / "validation"
    / "optical_observation_model_sensitivity"
)
_OUTPUT_PATH = _OUTPUT_DIRECTORY / "observation_models.npz"
_SUMMARY_PATH = _OUTPUT_DIRECTORY / "model_summary.csv"
_RAMP_PATH = _OUTPUT_DIRECTORY / "ramp_comparison.csv"
_PLOT_PATH = _OUTPUT_DIRECTORY / "model_comparison.png"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"

_MODEL_NAMES = (
    "A_point_hard",
    "B_point_linear",
    "C_area_hard",
    "D_area_linear",
)
_SOURCE_NAMES = ("point", "finite_area")
_BINNING_NAMES = ("hard", "linear")
_MODEL_SOURCE_INDEX = np.array((0, 0, 1, 1), dtype=np.int64)
_MODEL_LINEAR_SPLAT = (False, True, False, True)

_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK
_SAMPLE_SIDE_COUNT = 256
_MAX_BOUNCES = 24
_PATH_RNG_SEED = 20260823
_AREA_RNG_SEED = 20260826
_CARRIER_ALBEDO = 0.7
_EMITTING_WINDOW_X_M = 1.8e-3
_EMITTING_WINDOW_Y_M = 1.6e-3
_EMITTED_POWER = 5.0
_ENERGY_FIELDS = (
    "emitted_power",
    "escaped_power",
    "carrier_absorbed_power",
    "bulk_loss_power",
    "unresolved_internal_miss_power",
    "remaining_power",
    "accounted_power",
    "closure_error",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        return {name: saved[name] for name in saved.files}


def _validate_saved_contract(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> None:
    for name in (
        "reference_vertices_m",
        "led_centers_m",
        "scenario_names",
        "sphere_diameters_mm",
        "contact_y_mm",
        "force_targets_n",
    ):
        if not np.array_equal(candidate[name], reference[name]):
            raise RuntimeError(f"saved deformation contract differs in {name}")


def _make_leds(fingertip: Fingertip, led_centers_m: np.ndarray) -> tuple[LED, ...]:
    normal_W = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    return tuple(
        LED(
            position_W_m=center_m,
            normal_W=normal_W,
            parameters=fingertip.parameters.led,
        )
        for center_m in led_centers_m
    )


def _point_emission(
    scene: OptixScene,
    led: LED,
    angular_u1: np.ndarray,
    angular_u2: np.ndarray,
) -> np.ndarray:
    probe_distance_m = 0.5e-3 * led.parameters.height_mm
    probe_origin = (led.position_W_m - probe_distance_m * led.normal_W)[None, :]
    direction = led.normal_W[None, :]
    hit = scene.trace_closest(probe_origin, direction, mask=_CARRIER_MASK)
    if not hit["hit"][0]:
        raise RuntimeError("point-source carrier probe missed the recess floor")
    hit_position = probe_origin[0] + hit["t"][0] * led.normal_W
    if not np.allclose(hit_position, led.position_W_m, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("point-source carrier probe found the wrong surface")
    emission = led.emit(angular_u1, angular_u2)
    emission["origin_W_m"] = safe_secondary_origins(hit, direction)[0]
    return emission


def _area_emission(
    scene: OptixScene,
    led: LED,
    angular_u1: np.ndarray,
    angular_u2: np.ndarray,
    area_u_x: np.ndarray,
    area_u_y: np.ndarray,
) -> np.ndarray:
    emission = led.emit(angular_u1, angular_u2)
    source_positions = np.repeat(led.position_W_m[None, :], len(emission), axis=0)
    source_positions[:, 0] += _EMITTING_WINDOW_X_M * (area_u_x - 0.5)
    source_positions[:, 1] += _EMITTING_WINDOW_Y_M * (area_u_y - 0.5)

    directions = np.repeat(led.normal_W[None, :], len(emission), axis=0)
    probe_distance_m = 0.5e-3 * led.parameters.height_mm
    probe_origins = source_positions - probe_distance_m * directions
    hits = scene.trace_closest(probe_origins, directions, mask=_CARRIER_MASK)
    if not np.all(hits["hit"]):
        raise RuntimeError("finite-area carrier probe missed the recess floor")
    hit_positions = probe_origins + hits["t"][:, None] * directions
    maximum_error_m = float(np.max(np.linalg.norm(hit_positions - source_positions, axis=1)))
    if maximum_error_m > 1.0e-7:
        raise RuntimeError(
            "finite-area carrier probe found the wrong surface; "
            f"maximum error={maximum_error_m:.3e} m"
        )
    emission["origin_W_m"] = safe_secondary_origins(hits, directions)
    return emission


def _source_media(
    scene: OptixScene,
    led: LED,
    emission: np.ndarray,
    *,
    finite_area: bool,
) -> np.ndarray:
    if not finite_area:
        inside = source_inside_silicone(
            scene,
            led,
            emission,
            silicone_mask=_SILICONE_MASK,
        )
        return np.full(len(emission), inside, dtype=np.bool_)

    directions = np.repeat(led.normal_W[None, :], len(emission), axis=0)
    hits = scene.trace_closest(
        emission["origin_W_m"],
        directions,
        mask=_SILICONE_MASK,
    )
    if not np.all(hits["hit"]):
        raise RuntimeError("finite-area source normal does not reach silicone")
    projections = np.einsum("ij,ij->i", hits["normal_W"], directions)
    if np.any(np.abs(projections) <= 1.0e-6):
        raise RuntimeError("finite-area source interface is geometrically ambiguous")
    return projections > 0.0


def _path_energy(paths: object) -> np.ndarray:
    return np.array(
        (
            paths.emitted_power,
            paths.escaped_power,
            paths.absorbed_power,
            paths.bulk_loss_power,
            paths.unresolved_internal_miss_power,
            paths.remaining_power,
            paths.accounted_power,
            paths.closure_error,
        ),
        dtype=np.float64,
    )


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    leds: tuple[LED, ...],
    emissions: tuple[np.ndarray, ...],
    *,
    finite_area: bool,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hard = np.empty((len(leds), 11), dtype=np.float64)
    linear = np.empty_like(hard)
    energy = np.empty((len(leds), len(_ENERGY_FIELDS)), dtype=np.float64)
    outside = np.empty(len(leds), dtype=np.float64)
    inside_fraction = np.empty(len(leds), dtype=np.float64)
    optics = fingertip.parameters.optical
    for led_index, (led, emission) in enumerate(zip(leds, emissions, strict=True)):
        inside = _source_media(
            scene,
            led,
            emission,
            finite_area=finite_area,
        )
        paths = trace_bounded_paths(
            scene,
            emission["origin_W_m"],
            emission["direction_W"],
            emission["power"],
            inside_silicone=inside,
            n_air=1.0,
            n_silicone=optics.refractive_index,
            extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
            carrier_albedo=_CARRIER_ALBEDO,
            max_bounces=_MAX_BOUNCES,
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
            silicone_instance_id=_SILICONE_INSTANCE_ID,
            carrier_instance_id=_CARRIER_INSTANCE_ID,
            mask=_ALL_MASK,
        )
        hard[led_index], hard_outside, hard_visible = longitudinal_side_view_power(
            paths.escaped_rays,
        )
        (
            linear[led_index],
            linear_outside,
            linear_visible,
        ) = longitudinal_side_view_power(
            paths.escaped_rays,
            linear_splat=True,
        )
        if not np.isclose(hard_outside, linear_outside, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("hard and linear binning changed outside-ROI power")
        if not np.isclose(hard_visible, linear_visible, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("hard and linear binning changed visible power")
        energy[led_index] = _path_energy(paths)
        outside[led_index] = hard_outside
        inside_fraction[led_index] = float(np.mean(inside))
        del paths
    return hard, linear, energy, outside, inside_fraction


def _objective(
    response_matrix: np.ndarray,
    no_contact_response: np.ndarray,
    data: dict[str, np.ndarray],
) -> object:
    return compute_observation_objective(
        response_matrix=response_matrix,
        no_contact_response=no_contact_response,
        scenario_names=tuple(str(name) for name in data["scenario_names"]),
        sphere_diameters_mm=data["sphere_diameters_mm"],
        contact_y_mm=data["contact_y_mm"],
        force_targets_n=data["force_targets_n"],
        emitted_power=_EMITTED_POWER,
    )


def _center_half_pitch_separations(
    normalized: np.ndarray,
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diameters = np.asarray(data["sphere_diameters_mm"], dtype=np.float64)
    locations = np.asarray(data["contact_y_mm"], dtype=np.float64)
    unique_diameters = np.asarray(tuple(dict.fromkeys(diameters)), dtype=np.float64)
    separations = np.empty((len(unique_diameters), len(data["force_targets_n"])))
    for diameter_index, diameter in enumerate(unique_diameters):
        center = np.flatnonzero((diameters == diameter) & (locations == 0.0))
        half_pitch = np.flatnonzero((diameters == diameter) & (locations == 5.5))
        if len(center) != 1 or len(half_pitch) != 1:
            raise RuntimeError("center and +5.5 mm scenarios must be unique")
        separations[diameter_index] = np.linalg.norm(
            normalized[center[0]] - normalized[half_pitch[0]],
            axis=1,
        )
    return unique_diameters, np.asarray(data["force_targets_n"]), separations


def _model_responses(
    source_responses: np.ndarray,
    model_index: int,
) -> np.ndarray:
    source_index = int(_MODEL_SOURCE_INDEX[model_index])
    binning_index = int(_MODEL_LINEAR_SPLAT[model_index])
    return source_responses[source_index, binning_index]


def _write_results(
    data_sets: tuple[dict[str, np.ndarray], ...],
    source_responses: np.ndarray,
    no_contact_source_responses: np.ndarray,
    source_energy: np.ndarray,
    no_contact_source_energy: np.ndarray,
    source_outside_power: np.ndarray,
    source_inside_fraction: np.ndarray,
    no_contact_inside_fraction: np.ndarray,
    runtime_s: float,
) -> None:
    artifact_names = tuple(name for name, _ in _INPUTS)
    reference = data_sets[0]
    objectives: list[list[object]] = []
    center_half_pitch = np.empty((len(data_sets), len(_MODEL_NAMES), 3, 4))
    for artifact_index, data in enumerate(data_sets):
        artifact_objectives = []
        for model_index in range(len(_MODEL_NAMES)):
            response = _model_responses(source_responses[artifact_index], model_index)
            baseline = _model_responses(no_contact_source_responses, model_index)
            objective = _objective(response, baseline, data)
            artifact_objectives.append(objective)
            _, _, center_half_pitch[artifact_index, model_index] = (
                _center_half_pitch_separations(objective.normalized_response, data)
            )
        objectives.append(artifact_objectives)

    reference_vertices = np.asarray(reference["silicone_vertices_m"], dtype=np.float64)
    geometry_rms_mm = np.zeros((len(data_sets), 21, 4), dtype=np.float64)
    response_error = np.zeros((len(data_sets), len(_MODEL_NAMES), 21, 4))
    for artifact_index in range(1, len(data_sets)):
        vertices = np.asarray(
            data_sets[artifact_index]["silicone_vertices_m"],
            dtype=np.float64,
        )
        geometry_rms_mm[artifact_index] = 1.0e3 * np.sqrt(
            np.mean(np.sum((vertices - reference_vertices) ** 2, axis=3), axis=2)
        )
        for model_index in range(len(_MODEL_NAMES)):
            reference_normalized = objectives[0][model_index].normalized_response
            normalized = objectives[artifact_index][model_index].normalized_response
            response_error[artifact_index, model_index] = np.linalg.norm(
                normalized - reference_normalized,
                axis=2,
            )

    ramp_20_index = artifact_names.index("ramp_20s")
    summary_rows = []
    for model_index, model_name in enumerate(_MODEL_NAMES):
        objective = objectives[0][model_index]
        minimum_index = np.unravel_index(
            int(np.argmin(center_half_pitch[0, model_index])),
            center_half_pitch[0, model_index].shape,
        )
        paired_ratios = []
        diameters = np.asarray(reference["sphere_diameters_mm"])
        locations = np.asarray(reference["contact_y_mm"])
        unique_diameters = np.asarray(tuple(dict.fromkeys(diameters)))
        for diameter_index, diameter in enumerate(unique_diameters):
            center_index = int(
                np.flatnonzero((diameters == diameter) & (locations == 0.0))[0]
            )
            half_pitch_index = int(
                np.flatnonzero((diameters == diameter) & (locations == 5.5))[0]
            )
            for force_index in range(len(reference["force_targets_n"])):
                perturbation = max(
                    response_error[ramp_20_index, model_index, center_index, force_index],
                    response_error[
                        ramp_20_index,
                        model_index,
                        half_pitch_index,
                        force_index,
                    ],
                )
                ratio = (
                    center_half_pitch[0, model_index, diameter_index, force_index]
                    / perturbation
                    if perturbation > 0.0
                    else float("inf")
                )
                paired_ratios.append(ratio)
        summary_rows.append(
            {
                "model": model_name,
                "J_obs": objective.J_obs,
                "limiting_sphere_mm": objective.limiting_sphere_diameter_mm,
                "limiting_force_n": objective.limiting_force_n,
                "limiting_y_pair_mm": str(objective.limiting_contact_y_pair_mm),
                "minimum_center_half_pitch_separation": center_half_pitch[
                    0, model_index
                ][minimum_index],
                "center_half_pitch_limiting_sphere_mm": unique_diameters[
                    minimum_index[0]
                ],
                "center_half_pitch_limiting_force_n": reference["force_targets_n"][
                    minimum_index[1]
                ],
                "ramp_20s_response_error_rms": float(
                    np.sqrt(np.mean(response_error[ramp_20_index, model_index] ** 2))
                ),
                "ramp_20s_response_error_max": float(
                    np.max(response_error[ramp_20_index, model_index])
                ),
                "minimum_location_to_perturbation_ratio": float(min(paired_ratios)),
            }
        )

    ramp_rows = []
    for artifact_index in range(1, len(data_sets)):
        for model_index, model_name in enumerate(_MODEL_NAMES):
            reference_objective = objectives[0][model_index]
            objective = objectives[artifact_index][model_index]
            ramp_rows.append(
                {
                    "artifact": artifact_names[artifact_index],
                    "model": model_name,
                    "J_obs": objective.J_obs,
                    "J_obs_relative_error": abs(
                        objective.J_obs - reference_objective.J_obs
                    )
                    / reference_objective.J_obs,
                    "response_error_rms": float(
                        np.sqrt(np.mean(response_error[artifact_index, model_index] ** 2))
                    ),
                    "response_error_max": float(
                        np.max(response_error[artifact_index, model_index])
                    ),
                    "geometry_rms_mm_mean": float(
                        np.mean(geometry_rms_mm[artifact_index])
                    ),
                    "geometry_rms_mm_max": float(
                        np.max(geometry_rms_mm[artifact_index])
                    ),
                    "limiting_sphere_mm": objective.limiting_sphere_diameter_mm,
                    "limiting_force_n": objective.limiting_force_n,
                    "limiting_y_pair_mm": str(objective.limiting_contact_y_pair_mm),
                }
            )

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with _SUMMARY_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with _RAMP_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=tuple(ramp_rows[0]))
        writer.writeheader()
        writer.writerows(ramp_rows)

    np.savez_compressed(
        _OUTPUT_PATH,
        artifact_names=np.asarray(artifact_names),
        model_names=np.asarray(_MODEL_NAMES),
        source_names=np.asarray(_SOURCE_NAMES),
        binning_names=np.asarray(_BINNING_NAMES),
        scenario_names=reference["scenario_names"],
        sphere_diameters_mm=reference["sphere_diameters_mm"],
        contact_y_mm=reference["contact_y_mm"],
        force_targets_n=reference["force_targets_n"],
        no_contact_source_responses=no_contact_source_responses,
        source_responses=source_responses,
        no_contact_source_energy=no_contact_source_energy,
        source_energy=source_energy,
        source_outside_power=source_outside_power,
        source_inside_fraction=source_inside_fraction,
        no_contact_inside_fraction=no_contact_inside_fraction,
        center_half_pitch_separations=center_half_pitch,
        geometry_rms_mm=geometry_rms_mm,
        response_error=response_error,
        trace_runtime_s=np.array(runtime_s),
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(_MODEL_NAMES))
    for artifact_index, artifact_name in enumerate(artifact_names):
        axes[0, 0].plot(
            x,
            [objectives[artifact_index][index].J_obs for index in x],
            marker="o",
            label=artifact_name,
        )
    axes[0, 0].set_ylabel("J_obs")
    axes[0, 0].set_title("Force-conditioned worst location separation")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(
        x,
        [row["minimum_center_half_pitch_separation"] for row in summary_rows],
    )
    axes[0, 1].set_ylabel("normalized L2 separation")
    axes[0, 1].set_title("Reference center vs +5.5 mm")

    axes[1, 0].bar(
        x,
        [row["ramp_20s_response_error_rms"] for row in summary_rows],
    )
    axes[1, 0].set_ylabel("normalized response RMS")
    axes[1, 0].set_title("Reference vs 20 s ramp perturbation")

    axes[1, 1].bar(
        x,
        [row["minimum_location_to_perturbation_ratio"] for row in summary_rows],
    )
    axes[1, 1].set_ylabel("separation / perturbation")
    axes[1, 1].set_title("Worst matched center/half-pitch ratio")
    for axis in axes.ravel():
        axis.set_xticks(x, _MODEL_NAMES, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(_PLOT_PATH, dpi=180)
    plt.close(figure)

    mixed_fraction = source_inside_fraction[:, 1]
    mixed_count = int(np.count_nonzero((mixed_fraction > 0.0) & (mixed_fraction < 1.0)))
    maximum_closure = float(
        max(
            np.max(np.abs(no_contact_source_energy[..., -1])),
            np.max(np.abs(source_energy[..., -1])),
        )
    )
    summary_by_model = {row["model"]: row for row in summary_rows}
    point_hard = summary_by_model["A_point_hard"]
    point_linear = summary_by_model["B_point_linear"]
    area_hard = summary_by_model["C_area_hard"]
    area_linear = summary_by_model["D_area_linear"]
    linear_only_ratio_change = (
        point_linear["minimum_location_to_perturbation_ratio"]
        / point_hard["minimum_location_to_perturbation_ratio"]
        - 1.0
    )
    area_only_ratio_change = (
        area_hard["minimum_location_to_perturbation_ratio"]
        / point_hard["minimum_location_to_perturbation_ratio"]
        - 1.0
    )
    proposed_ratio_change = (
        area_linear["minimum_location_to_perturbation_ratio"]
        / point_hard["minimum_location_to_perturbation_ratio"]
        - 1.0
    )
    lines = [
        "# Optical observation-model sensitivity",
        "",
        "Newton was not rerun. Frozen reference and ramp silicone vertices were",
        "replayed through one rebuilt OptiX scene with common deterministic angular",
        "and path-branch samples.",
        "",
        "## Models",
        "",
        "- A: point source + hard 11-bin histogram",
        "- B: point source + linear cloud-in-cell splatting",
        "- C: finite-area source + hard histogram",
        "- D: finite-area source + linear splatting",
        "- finite source: uniform 1.8 mm (X) x 1.6 mm (Y) LuckyLight resin window",
        "- ballistic Fresnel/refraction/reflection and material parameters unchanged",
        "- J_obs definition unchanged",
        "",
        "## Reference results",
        "",
        "| model | J_obs | limiting sphere/force/Y pair | min center-half pitch | ramp20 RMS error | min separation/perturbation |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['J_obs']:.9f} | "
            f"{row['limiting_sphere_mm']:g} mm / {row['limiting_force_n']:g} N / "
            f"{row['limiting_y_pair_mm']} | "
            f"{row['minimum_center_half_pitch_separation']:.9f} | "
            f"{row['ramp_20s_response_error_rms']:.9f} | "
            f"{row['minimum_location_to_perturbation_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Dwell-reference versus ramp states",
            "",
            "| ramp | model | J_obs | relative error | response RMS | response max | geometry RMS mean/max [mm] | limiting pair |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ramp_rows:
        lines.append(
            f"| {row['artifact']} | {row['model']} | {row['J_obs']:.9f} | "
            f"{row['J_obs_relative_error']:.3%} | {row['response_error_rms']:.9f} | "
            f"{row['response_error_max']:.9f} | "
            f"{row['geometry_rms_mm_mean']:.6f}/{row['geometry_rms_mm_max']:.6f} | "
            f"{row['limiting_sphere_mm']:g} mm, {row['limiting_force_n']:g} N, "
            f"{row['limiting_y_pair_mm']} |"
        )
    lines.extend(
        [
            "",
            "## Checks and diagnostics",
            "",
            f"- finite-area source-state/LED entries with partial air/silicone coverage: {mixed_count}",
            f"- maximum optical energy closure error: {maximum_closure:.3e}",
            f"- optical replay runtime: {runtime_s:.3f} s",
            "- A point/hard responses reproduce every saved production response: PASS",
            "- hard and linear binning conserve identical +X visible and outside-ROI power: PASS",
            "",
            "The 20 s ramp comparison is the measured small numerical/history geometry",
            "perturbation; no synthetic displacement or optical smoothing parameter was",
            "introduced.",
            "",
            "## Interpretation",
            "",
            f"- Linear splatting alone changed the worst separation/perturbation ratio by {linear_only_ratio_change:+.1%} relative to A.",
            f"- Finite-area emission with hard bins changed that ratio by {area_only_ratio_change:+.1%} relative to A.",
            f"- Finite-area emission plus linear splatting changed it by {proposed_ratio_change:+.1%} relative to A.",
            "- The measured robustness improvement comes primarily from replacing the point-wide 0/1 initial-medium switch with spatial source samples.",
            "- Linear splatting is continuous and power-conserving, but in this data it removes more location separation than history sensitivity; it is therefore not selected as a production default by this validation alone.",
            "- No production source or binning default was changed.",
        ]
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    loaded = tuple(_load(path) for _, path in _INPUTS)
    reference = loaded[0]
    for candidate in loaded[1:]:
        _validate_saved_contract(candidate, reference)

    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_5led_mesh(fingertip)
    if not np.allclose(
        mesh.silicone.vertices,
        reference["reference_vertices_m"],
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("saved deformation states do not match the current mesh")
    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    leds = _make_leds(fingertip, reference["led_centers_m"])

    coordinate = (
        np.arange(_SAMPLE_SIDE_COUNT, dtype=np.float64) + 0.5
    ) / _SAMPLE_SIDE_COUNT
    angular_u1, angular_u2 = np.meshgrid(coordinate, coordinate, indexing="ij")
    angular_u1 = angular_u1.ravel()
    angular_u2 = angular_u2.ravel()
    ray_count = len(angular_u1)
    area_coordinate = (np.arange(ray_count, dtype=np.float64) + 0.5) / ray_count
    area_rng = np.random.default_rng(_AREA_RNG_SEED)
    area_u_x = area_coordinate[area_rng.permutation(ray_count)]
    area_u_y = area_coordinate[area_rng.permutation(ray_count)]

    point_emissions = tuple(
        _point_emission(scene, led, angular_u1, angular_u2) for led in leds
    )
    area_emissions = tuple(
        _area_emission(
            scene,
            led,
            angular_u1,
            angular_u2,
            area_u_x,
            area_u_y,
        )
        for led in leds
    )
    emissions_by_source = (point_emissions, area_emissions)

    path_rng = np.random.default_rng(_PATH_RNG_SEED)
    sample_shape = (_MAX_BOUNCES, ray_count)
    dielectric_branch_u = path_rng.random(sample_shape)
    carrier_u1 = path_rng.random(sample_shape)
    carrier_u2 = path_rng.random(sample_shape)

    artifact_count = len(loaded)
    scenario_count, force_count = reference["actual_forces_n"].shape
    led_count = len(leds)
    source_responses = np.empty(
        (artifact_count, 2, 2, scenario_count, force_count, led_count, 11),
        dtype=np.float64,
    )
    no_contact_source_responses = np.empty((2, 2, led_count, 11), dtype=np.float64)
    source_energy = np.empty(
        (artifact_count, 2, scenario_count, force_count, led_count, len(_ENERGY_FIELDS)),
        dtype=np.float64,
    )
    no_contact_source_energy = np.empty((2, led_count, len(_ENERGY_FIELDS)))
    source_outside_power = np.empty(
        (artifact_count, 2, scenario_count, force_count, led_count),
        dtype=np.float64,
    )
    source_inside_fraction = np.empty_like(source_outside_power)
    no_contact_inside_fraction = np.empty((2, led_count), dtype=np.float64)

    trace_start_s = perf_counter()
    for source_index, emissions in enumerate(emissions_by_source):
        finite_area = source_index == 1
        scene.update_silicone(reference["reference_vertices_m"])
        hard, linear, energy, _, inside_fraction = _trace_state(
            scene,
            fingertip,
            leds,
            emissions,
            finite_area=finite_area,
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
        )
        no_contact_source_responses[source_index] = np.stack((hard, linear))
        no_contact_source_energy[source_index] = energy
        no_contact_inside_fraction[source_index] = inside_fraction
        if np.any(inside_fraction != 0.0):
            raise RuntimeError("unloaded finite source is not entirely inside air")

        for artifact_index, data in enumerate(loaded):
            for scenario_index in range(scenario_count):
                for force_index in range(force_count):
                    scene.update_silicone(
                        data["silicone_vertices_m"][scenario_index, force_index]
                    )
                    hard, linear, energy, outside, inside_fraction = _trace_state(
                        scene,
                        fingertip,
                        leds,
                        emissions,
                        finite_area=finite_area,
                        dielectric_branch_u=dielectric_branch_u,
                        carrier_u1=carrier_u1,
                        carrier_u2=carrier_u2,
                    )
                    source_responses[
                        artifact_index, source_index, :, scenario_index, force_index
                    ] = np.stack((hard, linear))
                    source_energy[
                        artifact_index, source_index, scenario_index, force_index
                    ] = energy
                    source_outside_power[
                        artifact_index, source_index, scenario_index, force_index
                    ] = outside
                    source_inside_fraction[
                        artifact_index, source_index, scenario_index, force_index
                    ] = inside_fraction
            print(
                f"{_SOURCE_NAMES[source_index]} | {_INPUTS[artifact_index][0]} | "
                f"{scenario_count * force_count} states traced",
                flush=True,
            )
    trace_runtime_s = perf_counter() - trace_start_s

    if not np.allclose(
        no_contact_source_responses[0, 0],
        reference["no_contact_response"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("point/hard no-contact response did not reproduce")
    for artifact_index, data in enumerate(loaded):
        if not np.allclose(
            source_responses[artifact_index, 0, 0],
            data["response_matrix"],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"point/hard response did not reproduce {_INPUTS[artifact_index][0]}"
            )

    _write_results(
        loaded,
        source_responses,
        no_contact_source_responses,
        source_energy,
        no_contact_source_energy,
        source_outside_power,
        source_inside_fraction,
        no_contact_inside_fraction,
        trace_runtime_s,
    )
    print(f"Optical observation-model sensitivity PASS: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
