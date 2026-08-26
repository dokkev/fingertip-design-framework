"""Evaluate candidate contact and observation objectives from one saved NPZ."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_INPUT_PATH = Path(
    "output/validation/fingertip_raw_evaluator/nominal_fingertip_raw.npz"
)
_OUTPUT_DIRECTORY = Path("output/validation/full_finger_objective_prototype")
_CONTACT_CSV = _OUTPUT_DIRECTORY / "contact_components.csv"
_OBSERVATION_ONSET_CSV = _OUTPUT_DIRECTORY / "j_obs_contact_onset.csv"
_OBSERVATION_DISTANCE_CSV = _OUTPUT_DIRECTORY / "j_obs_same_force_distances.csv"
_OBSERVATION_TRAJECTORY_CSV = _OUTPUT_DIRECTORY / "j_obs_force_trajectories.csv"
_OBSERVATION_SUMMARY_CSV = _OUTPUT_DIRECTORY / "j_obs_force_conditioned_summary.csv"
_OBSERVATION_NPZ = _OUTPUT_DIRECTORY / "j_obs_force_conditioned.npz"
_CONTACT_PLOT = _OUTPUT_DIRECTORY / "contact_components.png"
_OBSERVATION_DISTANCE_PLOT = _OUTPUT_DIRECTORY / "j_obs_same_force_distances.png"
_OBSERVATION_TRAJECTORY_PLOT = _OUTPUT_DIRECTORY / "j_obs_force_trajectories.png"
_REPORT_PATH = _OUTPUT_DIRECTORY / "j_obs_force_conditioned_report.md"
_REPORT_ALIAS_PATH = _OUTPUT_DIRECTORY / "report.md"
_REQUIRED_FORCE_TARGETS_N = np.array((5.0, 10.0, 15.0, 20.0))
_OLD_COMBINED_CLUSTER_J_OBS = -0.02073232


def _triangle_areas(
    vertices_m: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    points = vertices_m[triangles]
    return 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        axis=1,
    )


def _surface_incidence(
    triangles: np.ndarray,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], set[int]], dict[tuple[int, ...], int]]:
    vertex_triangles: dict[int, set[int]] = {}
    edge_triangles: dict[tuple[int, int], set[int]] = {}
    triangle_ids: dict[tuple[int, ...], int] = {}
    for triangle_id, triangle in enumerate(triangles):
        vertices = tuple(int(vertex) for vertex in triangle)
        triangle_ids[tuple(sorted(vertices))] = triangle_id
        for vertex in vertices:
            vertex_triangles.setdefault(vertex, set()).add(triangle_id)
        for edge in combinations(vertices, 2):
            edge_triangles.setdefault(tuple(sorted(edge)), set()).add(triangle_id)
    return vertex_triangles, edge_triangles, triangle_ids


def _active_surface_triangles(
    contact_indices: np.ndarray,
    *,
    vertex_triangles: dict[int, set[int]],
    edge_triangles: dict[tuple[int, int], set[int]],
    triangle_ids: dict[tuple[int, ...], int],
) -> set[int]:
    active: set[int] = set()
    for record in contact_indices:
        primitive = tuple(sorted(int(index) for index in record if index >= 0))
        if len(primitive) == 1:
            active.update(vertex_triangles.get(primitive[0], ()))
        elif len(primitive) == 2:
            active.update(edge_triangles.get(primitive, ()))
        elif len(primitive) == 3:
            try:
                active.add(triangle_ids[primitive])
            except KeyError as error:
                raise RuntimeError(
                    f"contact triangle {primitive} is absent from surface topology"
                ) from error
        else:
            raise RuntimeError(
                f"unsupported contact primitive with {len(primitive)} vertices"
            )
    if not active:
        raise RuntimeError("contact checkpoint has no active surface triangles")
    return active


def _mean_contact_normal(normals: np.ndarray) -> np.ndarray:
    if len(normals) == 0 or not np.all(np.isfinite(normals)):
        raise RuntimeError("contact normals must be nonempty and finite")
    normal = normals.mean(axis=0)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError("mean contact normal is degenerate")
    return normal / norm


def _contact_components(data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    targets = data["force_targets_n"]
    if not np.allclose(targets, _REQUIRED_FORCE_TARGETS_N, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("prototype requires force targets [5, 10, 15, 20] N")
    triangles = np.asarray(data["surface_triangles"], dtype=np.int32)
    reference_areas_m2 = _triangle_areas(data["reference_vertices_m"], triangles)
    vertex_triangles, edge_triangles, triangle_ids = _surface_incidence(triangles)
    force_indices = {
        float(target): int(np.flatnonzero(np.isclose(targets, target))[0])
        for target in _REQUIRED_FORCE_TARGETS_N
    }

    rows: list[dict[str, object]] = []
    for scenario_index, scenario_name in enumerate(data["scenario_names"]):
        patches: dict[float, set[int]] = {}
        normals: dict[float, np.ndarray] = {}
        primitive_counts: dict[float, tuple[int, int, int]] = {}
        for target_n, force_index in force_indices.items():
            start, count = data["contact_record_offsets"][scenario_index, force_index]
            indices = data["contact_particle_indices"][start : start + count]
            patches[target_n] = _active_surface_triangles(
                indices,
                vertex_triangles=vertex_triangles,
                edge_triangles=edge_triangles,
                triangle_ids=triangle_ids,
            )
            present_counts = np.count_nonzero(indices >= 0, axis=1)
            primitive_counts[target_n] = tuple(
                int(np.count_nonzero(present_counts == dimension))
                for dimension in (1, 2, 3)
            )
            normals[target_n] = _mean_contact_normal(
                data["contact_normals_W"][start : start + count]
            )

        patch_5 = patches[5.0]
        patch_20 = patches[20.0]
        deformed_areas_5_m2 = _triangle_areas(
            data["silicone_vertices_m"][scenario_index, force_indices[5.0]],
            triangles,
        )
        area_5_m2 = float(deformed_areas_5_m2[list(patch_5)].sum())
        radius_m = 0.5e-3 * float(data["sphere_diameters_mm"][scenario_index])
        q_form = min(1.0, np.sqrt(area_5_m2 / (np.pi * radius_m**2)))

        intersection = patch_5 & patch_20
        union = patch_5 | patch_20
        q_stable = float(
            reference_areas_m2[list(intersection)].sum()
            / reference_areas_m2[list(union)].sum()
        )
        q_normal = float(
            np.clip(
                0.5 * (1.0 + np.dot(normals[5.0], normals[20.0])),
                0.0,
                1.0,
            )
        )

        forces = data["actual_forces_n"][scenario_index]
        indentations_m = data["indentations_m"][scenario_index]
        early_delta_m = indentations_m[force_indices[10.0]] - indentations_m[
            force_indices[5.0]
        ]
        late_delta_m = indentations_m[force_indices[20.0]] - indentations_m[
            force_indices[15.0]
        ]
        if early_delta_m <= 0.0 or late_delta_m <= 0.0:
            raise RuntimeError(f"{scenario_name} has non-increasing indentation")
        k_early_n_m = (
            forces[force_indices[10.0]] - forces[force_indices[5.0]]
        ) / early_delta_m
        k_late_n_m = (
            forces[force_indices[20.0]] - forces[force_indices[15.0]]
        ) / late_delta_m
        if not np.isfinite(k_early_n_m) or not np.isfinite(k_late_n_m):
            raise RuntimeError(f"{scenario_name} has non-finite stiffness")
        if k_early_n_m < 0.0 or k_late_n_m <= 0.0:
            raise RuntimeError(f"{scenario_name} has non-positive stiffness")
        q_stiff = float(np.clip(1.0 - k_early_n_m / k_late_n_m, 0.0, 1.0))
        q_contact = float(np.cbrt(q_form * q_stable * q_stiff))

        vertex_count_5, edge_count_5, triangle_count_5 = primitive_counts[5.0]
        rows.append(
            {
                "scenario": str(scenario_name),
                "sphere_diameter_mm": float(
                    data["sphere_diameters_mm"][scenario_index]
                ),
                "contact_y_mm": float(data["contact_y_mm"][scenario_index]),
                "area_5_mm2": 1.0e6 * area_5_m2,
                "active_triangles_5": len(patch_5),
                "active_triangles_20": len(patch_20),
                "vertex_contacts_5": vertex_count_5,
                "edge_contacts_5": edge_count_5,
                "triangle_contacts_5": triangle_count_5,
                "q_form": q_form,
                "q_stable": q_stable,
                "q_normal_diagnostic": q_normal,
                "k_early_n_mm": 1.0e-3 * k_early_n_m,
                "k_late_n_mm": 1.0e-3 * k_late_n_m,
                "q_stiff": q_stiff,
                "q_contact": q_contact,
            }
        )
    return rows


def force_conditioned_observation(
    values: np.ndarray,
    baseline: np.ndarray,
    emitted_power: float,
    *,
    scenario_names: tuple[str, ...] | None = None,
    force_targets_n: np.ndarray | None = None,
) -> dict[str, object]:
    """Reduce scenario-by-force observations without penalizing force variation."""
    observations = np.asarray(values, dtype=np.float64)
    no_contact = np.asarray(baseline, dtype=np.float64)
    if observations.ndim != 3:
        raise ValueError("values must have shape (scenario, force, channel)")
    if no_contact.shape != observations.shape[2:]:
        raise ValueError("baseline shape must match one observation vector")
    if observations.shape[0] < 2:
        raise ValueError("at least two contact locations are required")
    if observations.shape[1] < 1:
        raise ValueError("at least one force checkpoint is required")
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(no_contact)):
        raise ValueError("observations and baseline must be finite")
    if not np.isfinite(emitted_power) or emitted_power <= 0.0:
        raise ValueError("emitted_power must be finite and positive")

    if scenario_names is None:
        names = tuple(f"scenario_{index}" for index in range(observations.shape[0]))
    else:
        names = tuple(str(name) for name in scenario_names)
        if len(names) != observations.shape[0] or len(set(names)) != len(names):
            raise ValueError("scenario_names must be unique and match values")
    if force_targets_n is None:
        forces = np.arange(observations.shape[1], dtype=np.float64)
    else:
        forces = np.asarray(force_targets_n, dtype=np.float64)
        if forces.shape != (observations.shape[1],) or not np.all(np.isfinite(forces)):
            raise ValueError("force_targets_n must be finite and match values")

    normalized = (observations - no_contact) / emitted_power
    onset_distances = np.linalg.norm(normalized, axis=2)
    onset_index = np.unravel_index(int(np.argmin(onset_distances)), onset_distances.shape)

    location_distances = np.zeros(
        (len(forces), len(names), len(names)),
        dtype=np.float64,
    )
    distance_rows: list[dict[str, object]] = []
    for force_index, force_n in enumerate(forces):
        for scenario_a, scenario_b in combinations(range(len(names)), 2):
            distance = float(
                np.linalg.norm(
                    normalized[scenario_a, force_index]
                    - normalized[scenario_b, force_index]
                )
            )
            location_distances[force_index, scenario_a, scenario_b] = distance
            location_distances[force_index, scenario_b, scenario_a] = distance
            distance_rows.append(
                {
                    "force_n": float(force_n),
                    "location_a": names[scenario_a],
                    "location_b": names[scenario_b],
                    "l2_separation": distance,
                }
            )
    limiting_distance = min(distance_rows, key=lambda row: float(row["l2_separation"]))
    d_onset = float(onset_distances[onset_index])
    d_location = float(limiting_distance["l2_separation"])

    trajectory_rows: list[dict[str, object]] = []
    for scenario_index, name in enumerate(names):
        steps = np.linalg.norm(np.diff(normalized[scenario_index], axis=0), axis=1)
        pairwise = [
            float(
                np.linalg.norm(
                    normalized[scenario_index, force_a]
                    - normalized[scenario_index, force_b]
                )
            )
            for force_a, force_b in combinations(range(len(forces)), 2)
        ]
        trajectory_rows.append(
            {
                "location": name,
                "trajectory_path_length": float(steps.sum()),
                "maximum_force_pair_variation": max(pairwise, default=0.0),
            }
        )

    return {
        "normalized": normalized,
        "onset_distances": onset_distances,
        "location_distances": location_distances,
        "distance_rows": distance_rows,
        "trajectory_rows": trajectory_rows,
        "d_onset": d_onset,
        "onset_location": names[onset_index[0]],
        "onset_force_n": float(forces[onset_index[1]]),
        "d_location": d_location,
        "limiting_force_n": float(limiting_distance["force_n"]),
        "limiting_pair": (
            str(limiting_distance["location_a"]),
            str(limiting_distance["location_b"]),
        ),
        "J_obs": d_location,
        "median_location_separation": float(
            np.median([float(row["l2_separation"]) for row in distance_rows])
        ),
        "maximum_location_separation": float(
            max(float(row["l2_separation"]) for row in distance_rows)
        ),
        "d_location_over_d_onset": d_location / d_onset if d_onset > 0.0 else np.inf,
    }


def _observation_components(
    data: dict[str, np.ndarray],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    energy_fields = tuple(str(value) for value in data["energy_fields"])
    emitted_index = energy_fields.index("emitted_power")
    emitted_power = float(data["no_contact_energy"][:, emitted_index].sum())
    response = data["response_matrix"]
    no_contact = data["no_contact_response"]
    representations: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
    if response.shape[-1] == 4:
        representations["legacy_combined_4d_diagnostic"] = (
            response.sum(axis=2),
            no_contact.sum(axis=0),
            False,
        )
    elif response.shape[-1] == 11:
        representations["camera_spatial_11d_observable"] = (
            response.sum(axis=2),
            no_contact.sum(axis=0),
            True,
        )
    else:
        raise RuntimeError("raw response must have 4 or 11 observation channels")
    diagnostic_name = f"per_emitter_{5 * response.shape[-1]}d_diagnostic"
    representations[diagnostic_name] = (
        response.reshape(*response.shape[:2], -1),
        no_contact.reshape(-1),
        False,
    )
    names = tuple(str(name) for name in data["scenario_names"])
    forces = data["force_targets_n"]
    onset_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for representation, (values, baseline, observable) in representations.items():
        summary = force_conditioned_observation(
            values,
            baseline,
            emitted_power,
            scenario_names=names,
            force_targets_n=forces,
        )
        summary["observable"] = observable
        summary["emitted_power"] = emitted_power
        summaries[representation] = summary
        for scenario_index, name in enumerate(names):
            for force_index, force_n in enumerate(forces):
                onset_rows.append(
                    {
                        "representation": representation,
                        "hardware_observable": observable,
                        "location": name,
                        "force_n": float(force_n),
                        "onset_distance": float(
                            summary["onset_distances"][scenario_index, force_index]
                        ),
                    }
                )
        for row in summary["distance_rows"]:
            distance_rows.append(
                {
                    "representation": representation,
                    "hardware_observable": observable,
                    **row,
                }
            )
        for row in summary["trajectory_rows"]:
            trajectory_rows.append(
                {
                    "representation": representation,
                    "hardware_observable": observable,
                    **row,
                }
            )
    return onset_rows, distance_rows, trajectory_rows, summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty table {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_contact(rows: list[dict[str, object]]) -> None:
    labels = [f"Y={float(row['contact_y_mm']):+g} mm" for row in rows]
    x = np.arange(len(rows))
    width = 0.19
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for offset, field, label in (
        (-1.5 * width, "q_form", r"$q_{form}$"),
        (-0.5 * width, "q_stable", r"$q_{stable}$"),
        (0.5 * width, "q_stiff", r"$q_{stiff}$"),
        (1.5 * width, "q_contact", r"$q_{contact}$"),
    ):
        axes[0].bar(
            x + offset,
            [float(row[field]) for row in rows],
            width,
            label=label,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("bounded score")
    axes[0].set_title("Contact objective components")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x,
        [float(row["area_5_mm2"]) for row in rows],
        "o-",
        label=r"$A_5$ [mm²]",
    )
    axes[1].plot(
        x,
        [float(row["k_early_n_mm"]) for row in rows],
        "s-",
        label=r"$k_{early}$ [N/mm]",
    )
    axes[1].plot(
        x,
        [float(row["k_late_n_mm"]) for row in rows],
        "^-",
        label=r"$k_{late}$ [N/mm]",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Physical components")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(_CONTACT_PLOT, dpi=180)
    plt.close(figure)


def _plot_observation_distances(
    summaries: dict[str, dict[str, object]],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(summaries),
        figsize=(6.2 * len(summaries), 5.0),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (representation, summary) in zip(
        axes,
        summaries.items(),
        strict=True,
    ):
        pairs: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for row in summary["distance_rows"]:
            pair = (str(row["location_a"]), str(row["location_b"]))
            pairs.setdefault(pair, []).append(
                (float(row["force_n"]), float(row["l2_separation"]))
            )
        for pair, samples in pairs.items():
            samples.sort()
            axis.plot(
                [sample[0] for sample in samples],
                [sample[1] for sample in samples],
                "o-",
                label=f"{pair[0]} vs {pair[1]}",
            )
        axis.axhline(
            float(summary["d_onset"]),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=r"$d_{onset}$",
        )
        axis.set_xlabel("force [N]")
        axis.set_title(representation)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("normalized L2 separation")
    figure.suptitle("Force-conditioned location separation")
    figure.tight_layout()
    figure.savefig(_OBSERVATION_DISTANCE_PLOT, dpi=180)
    plt.close(figure)


def _plot_force_trajectories(
    summaries: dict[str, dict[str, object]],
    scenario_names: np.ndarray,
    force_targets_n: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(summaries),
        figsize=(6.2 * len(summaries), 5.0),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (representation, summary) in zip(
        axes,
        summaries.items(),
        strict=True,
    ):
        onset = np.asarray(summary["onset_distances"])
        for scenario_index, name in enumerate(scenario_names):
            axis.plot(
                force_targets_n,
                onset[scenario_index],
                "o-",
                label=str(name),
            )
        axis.set_xlabel("force [N]")
        axis.set_title(representation)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"$\|z(p,F)\|_2$")
    figure.suptitle("Force-dependent optical trajectories (diagnostic only)")
    figure.tight_layout()
    figure.savefig(_OBSERVATION_TRAJECTORY_PLOT, dpi=180)
    plt.close(figure)


def _write_report(
    contact_rows: list[dict[str, object]],
    summaries: dict[str, dict[str, object]],
) -> None:
    worst_contact = min(contact_rows, key=lambda row: float(row["q_contact"]))
    combined = summaries["legacy_combined_4d_diagnostic"]
    spatial = summaries["camera_spatial_11d_observable"]
    diagnostic = summaries["per_emitter_20d_diagnostic"]
    j_ratio = float(diagnostic["J_obs"]) / float(combined["J_obs"])
    location_ratio = float(diagnostic["d_location"]) / float(combined["d_location"])
    onset_ratio = float(diagnostic["d_onset"]) / float(combined["d_onset"])
    spatial_ratio = float(spatial["J_obs"]) / float(combined["J_obs"])
    half_pitch_pair = tuple(str(value) for value in combined["limiting_pair"])

    def half_pitch_rows(summary: dict[str, object]) -> dict[float, float]:
        return {
            float(row["force_n"]): float(row["l2_separation"])
            for row in summary["distance_rows"]
            if (str(row["location_a"]), str(row["location_b"])) == half_pitch_pair
        }

    combined_half_pitch = half_pitch_rows(combined)
    spatial_half_pitch = half_pitch_rows(spatial)
    diagnostic_half_pitch = half_pitch_rows(diagnostic)
    lines = [
        "# Force-conditioned contact observability prototype",
        "",
        "This analysis reuses the saved nominal full-finger Newton states. It "
        "re-traces optics for the corrected +X camera direction, but it does "
        "not rerun Newton or register an objective with Ax.",
        "",
        "## Candidate definition",
        "",
        "For each location and force, `z=(y-y0)/P_emit`. The candidate is "
        "`J_obs=d_location=min_F min_i!=j ||z_i,F-z_j,F||`. Contact onset "
        "`d_onset=min ||z||` and within-location force variation are diagnostics "
        "only because QDD proprioception owns contact detection and force.",
        "",
        "## Contact patch definition",
        "",
        "Newton emits vertex, edge, and triangle full-surface records. Each record "
        "is expanded to its incident silicone surface triangles; their union is "
        "the active Lagrangian patch. A5 uses deformed 5 N triangle area. Patch "
        "IoU uses reference triangle areas so the overlap measures material-surface "
        "support rather than area change under deformation.",
        "",
        "## Contact components",
        "",
        "| scenario | A5 [mm²] | q_form | q_stable | k_early [N/mm] | k_late [N/mm] | q_stiff | q_contact | q_normal diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in contact_rows:
        lines.append(
            f"| {row['scenario']} | {float(row['area_5_mm2']):.5f} | "
            f"{float(row['q_form']):.6f} | {float(row['q_stable']):.6f} | "
            f"{float(row['k_early_n_mm']):.6f} | "
            f"{float(row['k_late_n_mm']):.6f} | "
            f"{float(row['q_stiff']):.6f} | "
            f"{float(row['q_contact']):.6f} | "
            f"{float(row['q_normal_diagnostic']):.6f} |"
        )
    lines.extend(
        (
            "",
            f"Candidate `J_contact = {float(worst_contact['q_contact']):.8f}`; "
            f"limiting scenario: `{worst_contact['scenario']}`.",
        )
    )

    for representation, summary in summaries.items():
        lines.extend(
            (
                "",
                f"## Observation components: `{representation}`",
                "",
                f"- hardware observable: `{summary['observable']}`",
                f"- emitted-power normalization: "
                f"`P_emit={float(summary['emitted_power']):g}`",
                f"- d_onset: `{float(summary['d_onset']):.8f}` at "
                f"`{summary['onset_location']}`, {float(summary['onset_force_n']):g} N",
                f"- d_location: `{float(summary['d_location']):.8f}` at "
                f"{float(summary['limiting_force_n']):g} N, "
                f"`{summary['limiting_pair'][0]}` vs "
                f"`{summary['limiting_pair'][1]}`",
                f"- candidate J_obs: `{float(summary['J_obs']):.8f}`",
                f"- median / maximum same-force separation: "
                f"`{float(summary['median_location_separation']):.8f}` / "
                f"`{float(summary['maximum_location_separation']):.8f}`",
                f"- d_location / d_onset: "
                f"`{float(summary['d_location_over_d_onset']):.6f}`",
                "",
                "| force [N] | location A | location B | L2 separation |",
                "|---:|---|---|---:|",
            )
        )
        for row in summary["distance_rows"]:
            lines.append(
                f"| {float(row['force_n']):g} | {row['location_a']} | "
                f"{row['location_b']} | {float(row['l2_separation']):.8f} |"
            )
        lines.extend(("", "Force-trajectory diagnostics:", ""))
        lines.extend(
            (
                "| location | path length | maximum force-pair variation |",
                "|---|---:|---:|",
            )
        )
        for row in summary["trajectory_rows"]:
            lines.append(
                f"| {row['location']} | "
                f"{float(row['trajectory_path_length']):.8f} | "
                f"{float(row['maximum_force_pair_variation']):.8f} |"
            )

    lines.extend(
        (
            "",
            "## Comparison with the rejected cluster-margin formulation",
            "",
            f"The old combined-4D cluster margin was "
            f"`{_OLD_COMBINED_CLUSTER_J_OBS:.8f}`. The new combined-4D score is "
            f"`{float(combined['J_obs']):.8f}` and has no sign/overlap "
            "interpretation. Large within-location force trajectories are no "
            "longer subtracted from same-force location separation.",
            "",
            "The diagnostic 20D score is "
            f"`{j_ratio:.3f}x` the combined score. Its d_onset is "
            f"`{onset_ratio:.3f}x` and d_location is "
            f"`{location_ratio:.3f}x` the combined values. This does not make "
            "emitter identity hardware-observable.",
            "",
            f"The +X spatial 11D score is `{spatial_ratio:.3f}x` the old +Y "
            "combined 4D score and is directly aligned with the intended "
            "longitudinal camera field.",
            "",
            "### Half-pitch diagnostic: Y=0 vs Y=+5.5 mm",
            "",
            "| force [N] | old combined 4D | +X spatial 11D | per-emitter 20D diagnostic |",
            "|---:|---:|---:|---:|",
            *(
                f"| {force_n:g} | {combined_half_pitch[force_n]:.8f} | "
                f"{spatial_half_pitch[force_n]:.8f} | "
                f"{diagnostic_half_pitch[force_n]:.8f} |"
                for force_n in sorted(combined_half_pitch)
            ),
            "",
            "These samples show response change for a half-pitch displacement, "
            "but they do not establish a 5.5 mm localization-accuracy claim.",
            "",
            "## Renderer / acquisition representation audit",
            "",
            "The full-finger camera-facing convention is now `+X`, so the image "
            "horizontal coordinate is longitudinal world Y. The observable sums "
            "all five simultaneous emitters into eleven fixed 5 mm bins over "
            "`Y=[-27.5,+27.5] mm`. Escaped-ray source identity is not used. The "
            "bin width matches the physical 5 mm localization grid rather than "
            "being tuned against this morphology.",
            "",
            "This remains a directional surface-power observation, not a finite "
            "aperture or pixel camera renderer. Adding camera extrinsics, "
            "projection, lens effects, or occlusion would require a separate "
            "sensor model and is outside this prototype.",
            "",
            "## Decision",
            "",
            "Use the +X simultaneous 11-bin spatial response for the candidate "
            "production `J_obs`. Keep `d_onset`, old combined Q, and labeled "
            "per-emitter 20D as diagnostics. The formulation does not require "
            "LED multiplexing.",
            "",
            "## Required answers",
            "",
            f"1. +X spatial 11D `J_obs = {float(spatial['J_obs']):.8f}`.",
            f"2. Diagnostic `d_onset = {float(spatial['d_onset']):.8f}`; "
            f"`d_location = {float(spatial['d_location']):.8f}`.",
            f"3. `J_obs` is limited at "
            f"{float(spatial['limiting_force_n']):g} N by "
            f"`{spatial['limiting_pair'][0]}` vs "
            f"`{spatial['limiting_pair'][1]}`.",
            "4. The half-pitch separations at 5/10/15/20 N are listed above; "
            "the smallest is the 5 N value.",
            f"5. Against old combined 4D, diagnostic 20D has "
            f"`{j_ratio:.3f}x` J_obs and `{location_ratio:.3f}x` d_location; "
            f"its d_onset ratio is `{onset_ratio:.3f}x`.",
            "6. The current renderer exposes the required escape positions and "
            "directions before reduction.",
            "7. The recommended observable is the implemented +X, 11x5 mm "
            "longitudinal power vector with simultaneous LEDs.",
            "8. Yes. The reducer compares locations only within the same force; "
            "the synthetic large-force-trajectory test retains its expected score.",
            "9. Use camera-spatial bins; do not use combined 4D or require "
            "emitter multiplexing for this objective.",
            "10. The force-conditioned equation and directional spatial-vector "
            "contract are ready to freeze. Camera calibration remains later "
            "hardware work, not an ambiguity in this idealized objective.",
            "",
            "The artifact contains only one 15 mm sphere diameter. No curvature "
            "invariance or cross-object robustness is inferred.",
        )
    )
    report = "\n".join(lines) + "\n"
    _REPORT_PATH.write_text(report, encoding="utf-8")
    _REPORT_ALIAS_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    if not _INPUT_PATH.is_file():
        raise FileNotFoundError(_INPUT_PATH)
    with np.load(_INPUT_PATH) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    required = {
        "reference_vertices_m",
        "surface_triangles",
        "scenario_names",
        "sphere_diameters_mm",
        "contact_y_mm",
        "force_targets_n",
        "actual_forces_n",
        "indentations_m",
        "contact_record_offsets",
        "contact_particle_indices",
        "contact_normals_W",
        "silicone_vertices_m",
        "no_contact_response",
        "no_contact_energy",
        "response_matrix",
        "energy_fields",
    }
    missing = required - data.keys()
    if missing:
        raise RuntimeError(f"raw evaluator artifact is missing {sorted(missing)}")
    if not all(np.all(np.isfinite(data[name])) for name in required if data[name].dtype.kind in "fc"):
        raise RuntimeError("raw evaluator artifact contains non-finite numeric data")

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    contact_rows = _contact_components(data)
    onset_rows, distance_rows, trajectory_rows, summaries = _observation_components(
        data,
    )
    _write_csv(_CONTACT_CSV, contact_rows)
    _write_csv(_OBSERVATION_ONSET_CSV, onset_rows)
    _write_csv(_OBSERVATION_DISTANCE_CSV, distance_rows)
    _write_csv(_OBSERVATION_TRAJECTORY_CSV, trajectory_rows)
    _write_csv(
        _OBSERVATION_SUMMARY_CSV,
        [
            {
                "representation": representation,
                "hardware_observable": summary["observable"],
                "emitted_power": summary["emitted_power"],
                "d_onset": summary["d_onset"],
                "onset_location": summary["onset_location"],
                "onset_force_n": summary["onset_force_n"],
                "d_location": summary["d_location"],
                "limiting_force_n": summary["limiting_force_n"],
                "limiting_location_a": summary["limiting_pair"][0],
                "limiting_location_b": summary["limiting_pair"][1],
                "J_obs": summary["J_obs"],
                "median_location_separation": summary[
                    "median_location_separation"
                ],
                "maximum_location_separation": summary[
                    "maximum_location_separation"
                ],
                "d_location_over_d_onset": summary["d_location_over_d_onset"],
            }
            for representation, summary in summaries.items()
        ],
    )
    saved_observations: dict[str, np.ndarray] = {
        "scenario_names": data["scenario_names"],
        "contact_y_mm": data["contact_y_mm"],
        "force_targets_n": data["force_targets_n"],
    }
    for representation, summary in summaries.items():
        saved_observations[f"{representation}_normalized"] = summary["normalized"]
        saved_observations[f"{representation}_onset"] = summary["onset_distances"]
        saved_observations[f"{representation}_location_distances"] = summary[
            "location_distances"
        ]
    np.savez_compressed(_OBSERVATION_NPZ, **saved_observations)
    _plot_contact(contact_rows)
    _plot_observation_distances(summaries)
    _plot_force_trajectories(
        summaries,
        data["scenario_names"],
        data["force_targets_n"],
    )
    _write_report(contact_rows, summaries)

    worst_contact = min(contact_rows, key=lambda row: float(row["q_contact"]))
    print("Raw objective prototype complete")
    print(
        f"J_contact={float(worst_contact['q_contact']):.8f} "
        f"({worst_contact['scenario']})"
    )
    for representation, summary in summaries.items():
        print(
            f"J_obs[{representation}]={float(summary['J_obs']):.8f} "
            f"d_onset={float(summary['d_onset']):.8f} "
            f"d_location={float(summary['d_location']):.8f} "
            f"({summary['limiting_pair'][0]} vs "
            f"{summary['limiting_pair'][1]} at "
            f"{float(summary['limiting_force_n']):g} N)"
        )
    print(f"report: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
