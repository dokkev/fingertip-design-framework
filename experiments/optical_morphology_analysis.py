"""Parameter-free optical response measurements in one fixed finger frame."""

from __future__ import annotations

import numpy as np

from .localization.fixed_finger_calibration import (
    FixedFingerCalibration,
    warp_with_fixed_finger_calibration,
)


def fixed_red_differences(
    unloaded_rgb: np.ndarray,
    loaded_rgbs: np.ndarray,
    calibration: FixedFingerCalibration,
) -> np.ndarray:
    """Return signed canonical red-channel differences in camera DN."""

    unloaded = np.asarray(unloaded_rgb)
    loaded = np.asarray(loaded_rgbs)
    if loaded.ndim != 4 or loaded.shape[1:] != unloaded.shape:
        raise ValueError("loaded_rgbs must have shape N x H x W x 3")
    reference_red = warp_with_fixed_finger_calibration(
        unloaded,
        calibration,
    )[:, :, 0].astype(np.float32)
    differences = []
    for frame in loaded:
        current_red = warp_with_fixed_finger_calibration(
            frame,
            calibration,
        )[:, :, 0].astype(np.float32)
        differences.append(current_red - reference_red)
    return np.asarray(differences, dtype=np.float32)


def mean_absolute_red_response(red_differences: np.ndarray) -> np.ndarray:
    """Return the mean absolute unloaded-relative response of each state in DN."""

    differences = _response_array(red_differences)
    return np.mean(np.abs(differences), axis=(1, 2))


def longitudinal_red_signatures(red_differences: np.ndarray) -> np.ndarray:
    """Average signed red response transversely, preserving longitudinal shape."""

    differences = _response_array(red_differences)
    return np.mean(differences, axis=2)


def pairwise_signature_distances(signatures: np.ndarray) -> np.ndarray:
    """Return resolution-independent pairwise RMS distances in camera DN."""

    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signatures must be a finite N x samples array with N >= 2")
    differences = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.mean(np.square(differences), axis=2))


def minimum_pairwise_separation(
    distance_matrix: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    """Return the smallest off-diagonal distance and its state pair."""

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
    upper_rows, upper_columns = np.triu_indices(len(distances), k=1)
    index = int(np.argmin(distances[upper_rows, upper_columns]))
    pair = (int(upper_rows[index]), int(upper_columns[index]))
    return float(distances[pair]), pair


def _response_array(red_differences: np.ndarray) -> np.ndarray:
    values = np.asarray(red_differences, dtype=np.float64)
    if values.ndim != 3 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("red_differences must be a finite N x height x width array")
    return values


__all__ = [
    "fixed_red_differences",
    "longitudinal_red_signatures",
    "mean_absolute_red_response",
    "minimum_pairwise_separation",
    "pairwise_signature_distances",
]
