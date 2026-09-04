"""Run-level longitudinal templates, repeat variability, and separability."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def rms_signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.all(np.isfinite(a + b)):
        raise ValueError("signatures must be equal finite one-dimensional arrays")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def repeat_variability(signatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the median template and each independent run's RMS deviation."""

    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("signatures must be a nonempty finite runs x bins array")
    template = np.median(values, axis=0)
    variability = np.sqrt(np.mean((values - template) ** 2, axis=1))
    return template, variability


def minimum_pairwise_separation(
    distance_matrix: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    """Return the smallest off-diagonal value and its pair."""

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if (
        distances.ndim != 2
        or distances.shape[0] != distances.shape[1]
        or len(distances) < 2
        or not np.all(np.isfinite(distances))
        or not np.allclose(distances, distances.T)
        or not np.allclose(np.diag(distances), 0.0)
    ):
        raise ValueError("distance_matrix must be finite, symmetric, and zero-diagonal")
    rows, columns = np.triu_indices(len(distances), k=1)
    index = int(np.argmin(distances[rows, columns]))
    pair = int(rows[index]), int(columns[index])
    return float(distances[pair]), pair


def pairwise_rows(
    run_rows: list[dict[str, Any]],
    signatures: np.ndarray,
    *,
    hole_spacing_mm: float | None = None,
) -> list[dict[str, Any]]:
    """Compare robust hole templates within session, indenter, and target force."""

    groups: dict[tuple[str, str, float], dict[int, list[int]]] = {}
    for index, row in enumerate(run_rows):
        key = (
            str(row["specimen_id"]),
            str(row["indenter"]),
            float(row["target_force_n"]),
        )
        groups.setdefault(key, {}).setdefault(int(row["hole_index"]), []).append(index)
    output: list[dict[str, Any]] = []
    for (specimen_id, indenter, target), holes in sorted(groups.items()):
        templates: dict[int, np.ndarray] = {}
        variability: dict[int, float] = {}
        for hole, indices in holes.items():
            template, deviations = repeat_variability(signatures[indices])
            templates[hole] = template
            variability[hole] = float(np.median(deviations))
        first_row = run_rows[next(iter(holes.values()))[0]]
        for hole_i, hole_j in combinations(sorted(templates), 2):
            distance = rms_signature_distance(templates[hole_i], templates[hole_j])
            within = 0.5 * (variability[hole_i] + variability[hole_j])
            output.append(
                {
                    "specimen_id": specimen_id,
                    "material": first_row["material"],
                    "morphology": first_row["morphology"],
                    "indenter": indenter,
                    "target_force_n": target,
                    "hole_i": hole_i,
                    "hole_j": hole_j,
                    "hole_index_separation": hole_j - hole_i,
                    "physical_separation_mm": ""
                    if hole_spacing_mm is None
                    else (hole_j - hole_i) * hole_spacing_mm,
                    "optical_rms_distance_dn": distance,
                    "within_variability_i_dn": variability[hole_i],
                    "within_variability_j_dn": variability[hole_j],
                    "mean_within_variability_dn": within,
                    "distance_to_variability_ratio": ""
                    if within <= 0.0
                    else distance / within,
                }
            )
    return output


__all__ = [
    "minimum_pairwise_separation",
    "pairwise_rows",
    "repeat_variability",
    "rms_signature_distance",
]
