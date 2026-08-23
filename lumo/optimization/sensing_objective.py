"""Pure numerical objectives for side-view sensing responses."""

from __future__ import annotations

import numpy as np


def sensing_descriptors(
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity and spatial signatures using state zero as no contact."""
    values = np.asarray(responses, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(
            "responses must have shape (at least 2 states, LEDs, 4)"
        )
    if values.shape[2] != 4:
        raise ValueError("responses must contain exactly four quadrants")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("responses must be finite and nonnegative")

    response_scale = float(values.max())
    relative_zero = np.finfo(np.float64).eps * response_scale
    led_power = values.sum(axis=2)
    reference_total = float(led_power[0].sum())
    if not np.isfinite(reference_total) or reference_total <= relative_zero:
        raise ValueError("no-contact side-visible power is numerically zero")
    intensity = (led_power - led_power[0]) / reference_total

    quadrant_power = values.sum(axis=1)
    state_totals = quadrant_power.sum(axis=1)
    state_relative_zero = np.finfo(np.float64).eps * float(state_totals.max())
    invalid_states = np.flatnonzero(
        ~np.isfinite(state_totals) | (state_totals <= state_relative_zero)
    )
    if len(invalid_states):
        raise ValueError(
            "side-visible power is numerically zero for state indices "
            f"{invalid_states.tolist()}"
        )
    spatial = quadrant_power / state_totals[:, None]
    if not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(spatial)):
        raise ValueError("normalized sensing descriptors must be finite")
    return intensity, spatial


def _minimum_pairwise_distance(descriptors: np.ndarray) -> float:
    minimum = float("inf")
    for first in range(len(descriptors) - 1):
        distances = np.linalg.norm(
            descriptors[first + 1 :] - descriptors[first],
            axis=1,
        )
        minimum = min(minimum, float(distances.min()))
    if not np.isfinite(minimum):
        raise ValueError("descriptor distances must be finite")
    return minimum


def sensing_objectives(responses: np.ndarray) -> tuple[float, float]:
    """Return worst-case LED-intensity and spatial separability."""
    intensity, spatial = sensing_descriptors(responses)
    return (
        _minimum_pairwise_distance(intensity),
        _minimum_pairwise_distance(spatial),
    )


__all__ = ["sensing_descriptors", "sensing_objectives"]
