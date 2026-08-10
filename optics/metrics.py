"""Camera-independent metrics for deterministic transport proxies."""

from __future__ import annotations

import math

import numpy as np

from optics.transport import TransportResult


def _path_mass(result: TransportResult) -> np.ndarray:
    return np.where(result.optical_mask, result.density, 0.0)


def _centroid(result: TransportResult) -> tuple[float, float]:
    mass = _path_mass(result)
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("transport path density must have positive total weight")
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


def evaluate(
    reference: TransportResult,
    loaded: TransportResult,
) -> dict[str, float]:
    """Compare two light-transport proxies without camera-image assumptions.

    ``field_difference`` is total-variation distance after conservative
    redistribution onto a common grid spanning both physical domains. It lies
    in ``[0, 1]``.
    """
    if not isinstance(reference, TransportResult) or not isinstance(
        loaded,
        TransportResult,
    ):
        raise TypeError("reference and loaded must be TransportResult values")
    if reference.launched_weight <= 0.0 or loaded.launched_weight <= 0.0:
        raise ValueError("evaluation requires positive launched weight")

    x_edges = np.linspace(
        min(float(reference.x_edges[0]), float(loaded.x_edges[0])),
        max(float(reference.x_edges[-1]), float(loaded.x_edges[-1])),
        max(len(reference.x_edges), len(loaded.x_edges)),
    )
    y_edges = np.linspace(
        min(float(reference.y_edges[0]), float(loaded.y_edges[0])),
        max(float(reference.y_edges[-1]), float(loaded.y_edges[-1])),
        max(len(reference.y_edges), len(loaded.y_edges)),
    )
    reference_mass = _mass_on_grid(reference, x_edges, y_edges)
    loaded_mass = _mass_on_grid(loaded, x_edges, y_edges)
    reference_total = float(np.sum(reference_mass))
    loaded_total = float(np.sum(loaded_mass))
    if reference_total <= 0.0 or loaded_total <= 0.0:
        raise ValueError("evaluation requires positive path-density mass")
    reference_distribution = reference_mass / reference_total
    loaded_distribution = loaded_mass / loaded_total

    reference_centroid = _centroid(reference)
    loaded_centroid = _centroid(loaded)
    centroid_shift = math.hypot(
        loaded_centroid[0] - reference_centroid[0],
        loaded_centroid[1] - reference_centroid[1],
    )
    return {
        "field_difference": 0.5
        * float(np.sum(np.abs(loaded_distribution - reference_distribution))),
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


__all__ = ["evaluate"]
