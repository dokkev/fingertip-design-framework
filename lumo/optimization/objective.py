"""Pure numerical objectives for one full-finger evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np


_REQUIRED_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class ContactObjective:
    """Worst-case contact quality and its per-scenario components."""

    J_contact: float
    limiting_scenario_index: int
    scenario_names: tuple[str, ...]
    q_form: np.ndarray
    q_stable: np.ndarray
    q_stiff: np.ndarray
    q_contact: np.ndarray
    q_normal: np.ndarray
    patch_area_5_m2: np.ndarray
    k_early_n_m: np.ndarray
    k_late_n_m: np.ndarray

    @property
    def limiting_scenario(self) -> str:
        return self.scenario_names[self.limiting_scenario_index]


@dataclass(frozen=True)
class ObservationObjective:
    """Worst same-force longitudinal separation and optical diagnostics."""

    J_obs: float
    limiting_sphere_diameter_mm: float
    limiting_force_n: float
    limiting_contact_y_pair_mm: tuple[float, float]
    d_onset: float
    onset_scenario: str
    onset_force_n: float
    normalized_response: np.ndarray
    sphere_diameters_mm: np.ndarray
    contact_y_mm: np.ndarray
    force_targets_n: np.ndarray
    location_separations: np.ndarray


def combine_led_responses(per_emitter_response: np.ndarray) -> np.ndarray:
    """Sum the emitter axis so LED identity cannot enter an observation."""
    values = np.asarray(per_emitter_response, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("per_emitter_response must end in (LED, channel)")
    if values.shape[-2] < 1 or values.shape[-1] < 1:
        raise ValueError("LED and channel dimensions must be nonempty")
    if not np.all(np.isfinite(values)):
        raise ValueError("per-emitter responses must be finite")
    return values.sum(axis=-2)


def _triangle_areas(vertices_m: np.ndarray, triangles: np.ndarray) -> np.ndarray:
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
                raise ValueError(
                    f"contact triangle {primitive} is absent from surface topology"
                ) from error
        else:
            raise ValueError(
                f"unsupported contact primitive with {len(primitive)} vertices"
            )
    if not active:
        raise ValueError("contact checkpoint has no active surface triangles")
    return active


def _mean_contact_normal(normals: np.ndarray) -> np.ndarray:
    if len(normals) == 0 or not np.all(np.isfinite(normals)):
        raise ValueError("contact normals must be nonempty and finite")
    normal = normals.mean(axis=0)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("mean contact normal is degenerate")
    return normal / norm


def _required_force_indices(force_targets_n: np.ndarray) -> dict[float, int]:
    targets = np.asarray(force_targets_n, dtype=np.float64)
    if targets.shape != (4,) or not np.allclose(
        targets,
        _REQUIRED_FORCE_TARGETS_N,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("contact objective requires [5, 10, 15, 20] N")
    return {target: index for index, target in enumerate(_REQUIRED_FORCE_TARGETS_N)}


def compute_contact_objective(
    *,
    reference_vertices_m: np.ndarray,
    surface_triangles: np.ndarray,
    scenario_names: tuple[str, ...],
    sphere_diameters_mm: np.ndarray,
    force_targets_n: np.ndarray,
    actual_forces_n: np.ndarray,
    indentations_m: np.ndarray,
    contact_record_offsets: np.ndarray,
    contact_particle_indices: np.ndarray,
    contact_normals_W: np.ndarray,
    silicone_vertices_m: np.ndarray,
) -> ContactObjective:
    """Compute finite patch, patch stability, and progressive stiffening."""
    names = tuple(str(name) for name in scenario_names)
    scenario_count = len(names)
    force_indices = _required_force_indices(force_targets_n)
    triangles = np.asarray(surface_triangles, dtype=np.int32)
    reference_vertices = np.asarray(reference_vertices_m, dtype=np.float64)
    diameters = np.asarray(sphere_diameters_mm, dtype=np.float64)
    forces = np.asarray(actual_forces_n, dtype=np.float64)
    indentations = np.asarray(indentations_m, dtype=np.float64)
    offsets = np.asarray(contact_record_offsets, dtype=np.int64)
    indices = np.asarray(contact_particle_indices, dtype=np.int32)
    normals = np.asarray(contact_normals_W, dtype=np.float64)
    vertices = np.asarray(silicone_vertices_m, dtype=np.float64)
    expected_state_shape = (scenario_count, len(force_indices))
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("surface_triangles must have shape (triangle, 3)")
    if diameters.shape != (scenario_count,) or np.any(diameters <= 0.0):
        raise ValueError("sphere diameters must be positive and match scenarios")
    if forces.shape != expected_state_shape or indentations.shape != expected_state_shape:
        raise ValueError("force and indentation arrays must match scenarios and forces")
    if offsets.shape != (*expected_state_shape, 2):
        raise ValueError("contact_record_offsets has the wrong shape")
    if vertices.shape[:2] != expected_state_shape or vertices.shape[-1] != 3:
        raise ValueError("silicone_vertices_m has the wrong state shape")
    numeric = (reference_vertices, diameters, forces, indentations, normals, vertices)
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("contact objective inputs must be finite")

    reference_areas_m2 = _triangle_areas(reference_vertices, triangles)
    vertex_triangles, edge_triangles, triangle_ids = _surface_incidence(triangles)
    q_form = np.empty(scenario_count, dtype=np.float64)
    q_stable = np.empty(scenario_count, dtype=np.float64)
    q_stiff = np.empty(scenario_count, dtype=np.float64)
    q_contact = np.empty(scenario_count, dtype=np.float64)
    q_normal = np.empty(scenario_count, dtype=np.float64)
    patch_area_5_m2 = np.empty(scenario_count, dtype=np.float64)
    k_early_n_m = np.empty(scenario_count, dtype=np.float64)
    k_late_n_m = np.empty(scenario_count, dtype=np.float64)

    for scenario_index, scenario_name in enumerate(names):
        patches: dict[float, set[int]] = {}
        mean_normals: dict[float, np.ndarray] = {}
        for target_n, force_index in force_indices.items():
            start, count = offsets[scenario_index, force_index]
            if start < 0 or count <= 0 or start + count > len(indices):
                raise ValueError(f"{scenario_name} has invalid contact offsets")
            contact_slice = indices[start : start + count]
            patches[target_n] = _active_surface_triangles(
                contact_slice,
                vertex_triangles=vertex_triangles,
                edge_triangles=edge_triangles,
                triangle_ids=triangle_ids,
            )
            mean_normals[target_n] = _mean_contact_normal(
                normals[start : start + count]
            )

        patch_5 = patches[5.0]
        patch_20 = patches[20.0]
        area_5_m2 = float(
            _triangle_areas(
                vertices[scenario_index, force_indices[5.0]],
                triangles,
            )[list(patch_5)].sum()
        )
        radius_m = 0.5e-3 * diameters[scenario_index]
        q_form[scenario_index] = min(
            1.0,
            np.sqrt(area_5_m2 / (np.pi * radius_m**2)),
        )
        intersection = patch_5 & patch_20
        union = patch_5 | patch_20
        union_area_m2 = float(reference_areas_m2[list(union)].sum())
        if union_area_m2 <= 0.0:
            raise ValueError(f"{scenario_name} has zero reference patch area")
        q_stable[scenario_index] = float(
            reference_areas_m2[list(intersection)].sum() / union_area_m2
        )
        q_normal[scenario_index] = float(
            np.clip(
                0.5 * np.dot(mean_normals[5.0], mean_normals[20.0]) + 0.5,
                0.0,
                1.0,
            )
        )

        early_delta_m = (
            indentations[scenario_index, force_indices[10.0]]
            - indentations[scenario_index, force_indices[5.0]]
        )
        late_delta_m = (
            indentations[scenario_index, force_indices[20.0]]
            - indentations[scenario_index, force_indices[15.0]]
        )
        if early_delta_m <= 0.0 or late_delta_m <= 0.0:
            raise ValueError(f"{scenario_name} has non-increasing indentation")
        k_early = (
            forces[scenario_index, force_indices[10.0]]
            - forces[scenario_index, force_indices[5.0]]
        ) / early_delta_m
        k_late = (
            forces[scenario_index, force_indices[20.0]]
            - forces[scenario_index, force_indices[15.0]]
        ) / late_delta_m
        if not np.isfinite(k_early) or not np.isfinite(k_late):
            raise ValueError(f"{scenario_name} has non-finite stiffness")
        if k_early < 0.0 or k_late <= 0.0:
            raise ValueError(f"{scenario_name} has non-positive stiffness")
        q_stiff[scenario_index] = float(np.clip(1.0 - k_early / k_late, 0.0, 1.0))
        q_contact[scenario_index] = float(
            np.cbrt(q_form[scenario_index] * q_stable[scenario_index] * q_stiff[scenario_index])
        )
        patch_area_5_m2[scenario_index] = area_5_m2
        k_early_n_m[scenario_index] = k_early
        k_late_n_m[scenario_index] = k_late

    limiting_index = int(np.argmin(q_contact))
    return ContactObjective(
        J_contact=float(q_contact[limiting_index]),
        limiting_scenario_index=limiting_index,
        scenario_names=names,
        q_form=q_form,
        q_stable=q_stable,
        q_stiff=q_stiff,
        q_contact=q_contact,
        q_normal=q_normal,
        patch_area_5_m2=patch_area_5_m2,
        k_early_n_m=k_early_n_m,
        k_late_n_m=k_late_n_m,
    )


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    return np.asarray(tuple(dict.fromkeys(float(value) for value in values)))


def compute_observation_objective(
    *,
    response_matrix: np.ndarray,
    no_contact_response: np.ndarray,
    scenario_names: tuple[str, ...],
    sphere_diameters_mm: np.ndarray,
    contact_y_mm: np.ndarray,
    force_targets_n: np.ndarray,
    emitted_power: float,
) -> ObservationObjective:
    """Compute worst same-force location separation within each sphere size."""
    combined = combine_led_responses(response_matrix)
    baseline = combine_led_responses(no_contact_response)
    if combined.ndim != 3:
        raise ValueError("response_matrix must have shape (scenario, force, LED, bin)")
    if baseline.shape != combined.shape[2:]:
        raise ValueError("no-contact response must match one combined observation")
    names = tuple(str(name) for name in scenario_names)
    scenario_count, force_count, _ = combined.shape
    diameters = np.asarray(sphere_diameters_mm, dtype=np.float64)
    locations = np.asarray(contact_y_mm, dtype=np.float64)
    forces = np.asarray(force_targets_n, dtype=np.float64)
    if len(names) != scenario_count or len(set(names)) != scenario_count:
        raise ValueError("scenario names must be unique and match responses")
    if diameters.shape != (scenario_count,) or locations.shape != (scenario_count,):
        raise ValueError("sphere diameters and contact locations must match scenarios")
    if forces.shape != (force_count,) or not np.all(np.isfinite(forces)):
        raise ValueError("force targets must be finite and match responses")
    if not np.isfinite(emitted_power) or emitted_power <= 0.0:
        raise ValueError("emitted_power must be finite and positive")

    normalized = (combined - baseline) / emitted_power
    unique_diameters = _ordered_unique(diameters)
    unique_locations = _ordered_unique(locations)
    if len(unique_locations) < 2:
        raise ValueError("at least two contact locations are required")
    lookup: dict[tuple[float, float], int] = {}
    for scenario_index, key in enumerate(zip(diameters, locations, strict=True)):
        numeric_key = (float(key[0]), float(key[1]))
        if numeric_key in lookup:
            raise ValueError("sphere/contact-location scenarios must be unique")
        lookup[numeric_key] = scenario_index
    expected = len(unique_diameters) * len(unique_locations)
    if scenario_count != expected or any(
        (float(diameter), float(location)) not in lookup
        for diameter in unique_diameters
        for location in unique_locations
    ):
        raise ValueError("responses must contain the full diameter/location product")

    separations = np.zeros(
        (len(unique_diameters), force_count, len(unique_locations), len(unique_locations)),
        dtype=np.float64,
    )
    minimum = float("inf")
    limiting = (0, 0, 0, 1)
    for diameter_index, diameter in enumerate(unique_diameters):
        for force_index in range(force_count):
            for first, second in combinations(range(len(unique_locations)), 2):
                first_index = lookup[(float(diameter), float(unique_locations[first]))]
                second_index = lookup[(float(diameter), float(unique_locations[second]))]
                distance = float(
                    np.linalg.norm(
                        normalized[first_index, force_index]
                        - normalized[second_index, force_index]
                    )
                )
                separations[diameter_index, force_index, first, second] = distance
                separations[diameter_index, force_index, second, first] = distance
                if distance < minimum:
                    minimum = distance
                    limiting = (diameter_index, force_index, first, second)

    onset_distances = np.linalg.norm(normalized, axis=2)
    onset_index = np.unravel_index(int(np.argmin(onset_distances)), onset_distances.shape)
    diameter_index, force_index, first, second = limiting
    return ObservationObjective(
        J_obs=minimum,
        limiting_sphere_diameter_mm=float(unique_diameters[diameter_index]),
        limiting_force_n=float(forces[force_index]),
        limiting_contact_y_pair_mm=(
            float(unique_locations[first]),
            float(unique_locations[second]),
        ),
        d_onset=float(onset_distances[onset_index]),
        onset_scenario=names[onset_index[0]],
        onset_force_n=float(forces[onset_index[1]]),
        normalized_response=normalized,
        sphere_diameters_mm=unique_diameters,
        contact_y_mm=unique_locations,
        force_targets_n=forces,
        location_separations=separations,
    )


def compute_objectives_from_raw(
    data: Mapping[str, object],
) -> tuple[ContactObjective, ObservationObjective]:
    """Recompute both objectives from a saved raw full-finger artifact."""
    energy_fields = tuple(str(field) for field in data["energy_fields"])
    emitted_index = energy_fields.index("emitted_power")
    no_contact_energy = np.asarray(data["no_contact_energy"], dtype=np.float64)
    energy_matrix = np.asarray(data["energy_matrix"], dtype=np.float64)
    emitted_power = float(no_contact_energy[:, emitted_index].sum())
    if not np.allclose(
        energy_matrix[..., emitted_index].sum(axis=2),
        emitted_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("emitted optical power changes between evaluation states")
    if not np.isclose(emitted_power, 5.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("production full-finger emitted power must equal 5")

    names = tuple(str(name) for name in data["scenario_names"])
    contact = compute_contact_objective(
        reference_vertices_m=np.asarray(data["reference_vertices_m"]),
        surface_triangles=np.asarray(data["surface_triangles"]),
        scenario_names=names,
        sphere_diameters_mm=np.asarray(data["sphere_diameters_mm"]),
        force_targets_n=np.asarray(data["force_targets_n"]),
        actual_forces_n=np.asarray(data["actual_forces_n"]),
        indentations_m=np.asarray(data["indentations_m"]),
        contact_record_offsets=np.asarray(data["contact_record_offsets"]),
        contact_particle_indices=np.asarray(data["contact_particle_indices"]),
        contact_normals_W=np.asarray(data["contact_normals_W"]),
        silicone_vertices_m=np.asarray(data["silicone_vertices_m"]),
    )
    observation = compute_observation_objective(
        response_matrix=np.asarray(data["response_matrix"]),
        no_contact_response=np.asarray(data["no_contact_response"]),
        scenario_names=names,
        sphere_diameters_mm=np.asarray(data["sphere_diameters_mm"]),
        contact_y_mm=np.asarray(data["contact_y_mm"]),
        force_targets_n=np.asarray(data["force_targets_n"]),
        emitted_power=emitted_power,
    )
    return contact, observation


__all__ = [
    "ContactObjective",
    "ObservationObjective",
    "combine_led_responses",
    "compute_contact_objective",
    "compute_objectives_from_raw",
    "compute_observation_objective",
]
