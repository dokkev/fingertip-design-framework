"""Pure numerical objectives for side-view sensing responses."""

from __future__ import annotations

import numpy as np


def sensing_descriptors(
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity and spatial signatures using state zero as no contact."""
    values = np.asarray(responses, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 4:
        raise ValueError("responses must have shape (at least 2 states, 4)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("responses must be finite and nonnegative")

    response_scale = float(values.max())
    relative_zero = np.finfo(np.float64).eps * response_scale
    state_power = values.sum(axis=1)
    reference_total = float(state_power[0])
    if not np.isfinite(reference_total) or reference_total <= relative_zero:
        raise ValueError("no-contact side-visible power is numerically zero")
    intensity = (state_power - reference_total) / reference_total

    state_relative_zero = np.finfo(np.float64).eps * float(state_power.max())
    invalid_states = np.flatnonzero(
        ~np.isfinite(state_power) | (state_power <= state_relative_zero)
    )
    if len(invalid_states):
        raise ValueError(
            "side-visible power is numerically zero for state indices "
            f"{invalid_states.tolist()}"
        )
    spatial = values / state_power[:, None]
    if not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(spatial)):
        raise ValueError("normalized sensing descriptors must be finite")
    return intensity, spatial


def sensing_objectives(responses: np.ndarray) -> tuple[float, float]:
    """Return worst-case total-intensity and spatial separability."""
    intensity, spatial = sensing_descriptors(responses)
    minimum_intensity = float("inf")
    minimum_spatial = float("inf")
    for first in range(len(intensity) - 1):
        for second in range(first + 1, len(intensity)):
            minimum_intensity = min(
                minimum_intensity,
                abs(float(intensity[first] - intensity[second])),
            )
            minimum_spatial = min(
                minimum_spatial,
                float(np.linalg.norm(spatial[first] - spatial[second])),
            )
    if not np.isfinite(minimum_intensity) or not np.isfinite(minimum_spatial):
        raise ValueError("descriptor distances must be finite")
    return minimum_intensity, minimum_spatial


__all__ = ["sensing_descriptors", "sensing_objectives"]
