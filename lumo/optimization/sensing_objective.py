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


def _minimum_separations(
    intensity: np.ndarray,
    spatial: np.ndarray,
) -> tuple[float, float]:
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


def sensing_objectives(
    responses: np.ndarray,
    *,
    no_contact_response: np.ndarray | None = None,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray, float, float]:
    """Return separate worst-case intensity and spatial separability.

    A two-dimensional input preserves the single state-set calculation. For
    grouped contact responses shaped ``(groups, states, 4)``, supply the shared
    no-contact response separately. Each group's contact states are compared
    only with one another, and the final two scalars are the minima across
    groups.
    """
    values = np.asarray(responses, dtype=np.float64)
    if no_contact_response is None:
        intensity, spatial = sensing_descriptors(values)
        return _minimum_separations(intensity, spatial)

    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] != 4:
        raise ValueError(
            "grouped responses must have shape (groups, at least 2 states, 4)"
        )
    if values.shape[0] < 1:
        raise ValueError("grouped responses must contain at least one group")

    reference = np.asarray(no_contact_response, dtype=np.float64)
    if reference.shape != (4,):
        raise ValueError("no_contact_response must have shape (4,)")

    group_intensity = np.empty(values.shape[0], dtype=np.float64)
    group_spatial = np.empty(values.shape[0], dtype=np.float64)
    for group_index, group_responses in enumerate(values):
        intensity, spatial = sensing_descriptors(
            np.vstack((reference, group_responses))
        )
        group_intensity[group_index], group_spatial[group_index] = _minimum_separations(
            intensity[1:], spatial[1:]
        )

    return (
        group_intensity,
        group_spatial,
        float(group_intensity.min()),
        float(group_spatial.min()),
    )


__all__ = ["sensing_descriptors", "sensing_objectives"]
