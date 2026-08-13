"""Camera-independent metrics for deterministic transport proxies."""

from __future__ import annotations

import math

import numpy as np

from optics.transport import TransportResult


class OpticalMetricError(ValueError):
    """Raised when a transport result cannot support metric evaluation."""


def _path_mass(result: TransportResult) -> np.ndarray:
    return np.where(result.optical_mask, result.density, 0.0)


def _centroid(result: TransportResult) -> tuple[float, float]:
    mass = _path_mass(result)
    total = float(np.sum(mass))
    if total <= 0.0:
        raise OpticalMetricError(
            "transport path density must have positive total weight"
        )
    x = 0.5 * (result.x_edges[:-1] + result.x_edges[1:])
    y = 0.5 * (result.y_edges[:-1] + result.y_edges[1:])
    return (
        float(np.sum(mass * x[None, :]) / total),
        float(np.sum(mass * y[:, None]) / total),
    )


def _overlap_fractions(
    target_edges: np.ndarray,
    source_edges: np.ndarray,
) -> np.ndarray:
    left = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    right = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    overlap = np.maximum(0.0, right - left)
    return overlap / np.diff(source_edges)[None, :]


def _mass_on_grid(
    result: TransportResult,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    x_fraction = _overlap_fractions(x_edges, result.x_edges)
    y_fraction = _overlap_fractions(y_edges, result.y_edges)
    return y_fraction @ _path_mass(result) @ x_fraction.T


def field_difference(
    first: TransportResult,
    second: TransportResult,
) -> float:
    """Return TV distance between two normalized transport path fields."""
    if not isinstance(first, TransportResult) or not isinstance(
        second,
        TransportResult,
    ):
        raise TypeError("first and second must be TransportResult values")
    if first.launched_weight <= 0.0 or second.launched_weight <= 0.0:
        raise OpticalMetricError("evaluation requires positive launched weight")

    x_edges = np.linspace(
        min(float(first.x_edges[0]), float(second.x_edges[0])),
        max(float(first.x_edges[-1]), float(second.x_edges[-1])),
        max(len(first.x_edges), len(second.x_edges)),
    )
    y_edges = np.linspace(
        min(float(first.y_edges[0]), float(second.y_edges[0])),
        max(float(first.y_edges[-1]), float(second.y_edges[-1])),
        max(len(first.y_edges), len(second.y_edges)),
    )
    first_mass = _mass_on_grid(first, x_edges, y_edges)
    second_mass = _mass_on_grid(second, x_edges, y_edges)
    first_total = float(np.sum(first_mass))
    second_total = float(np.sum(second_mass))
    if first_total <= 0.0 or second_total <= 0.0:
        raise OpticalMetricError("evaluation requires positive path-density mass")
    first_distribution = first_mass / first_total
    second_distribution = second_mass / second_total
    distance = 0.5 * float(
        np.sum(np.abs(second_distribution - first_distribution))
    )
    return min(1.0, max(0.0, distance))


def evaluate(
    reference: TransportResult,
    loaded: TransportResult,
) -> dict[str, float]:
    """Compare two light-transport proxies without camera-image assumptions."""
    if not isinstance(reference, TransportResult) or not isinstance(
        loaded,
        TransportResult,
    ):
        raise TypeError("reference and loaded must be TransportResult values")
    if reference.launched_weight <= 0.0 or loaded.launched_weight <= 0.0:
        raise OpticalMetricError("evaluation requires positive launched weight")

    reference_centroid = _centroid(reference)
    loaded_centroid = _centroid(loaded)
    centroid_shift = math.hypot(
        loaded_centroid[0] - reference_centroid[0],
        loaded_centroid[1] - reference_centroid[1],
    )
    return {
        "field_difference": field_difference(reference, loaded),
        "centroid_shift_mm": centroid_shift,
        "escaped_fraction_change": (
            loaded.escaped_weight / loaded.launched_weight
            - reference.escaped_weight / reference.launched_weight
        ),
        "absorbed_fraction_change": (
            loaded.absorbed_weight / loaded.launched_weight
            - reference.absorbed_weight / reference.launched_weight
        ),
    }


__all__ = ["OpticalMetricError", "evaluate", "field_difference"]
