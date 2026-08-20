"""Deterministic primary-ray sets for FULL_3D transport."""

from __future__ import annotations

from math import radians

import numpy as np

from model.led import LED
def _radical_inverse_base_two(indices: np.ndarray) -> np.ndarray:
    """Vectorized base-two radical inverse for uint32 index values."""
    values = np.asarray(indices, dtype=np.uint32)
    reversed_bits = np.zeros(values.shape, dtype=np.uint32)
    for shift in range(32):
        reversed_bits |= ((values >> shift) & np.uint32(1)) << np.uint32(31 - shift)
    return reversed_bits.astype(np.float64) / float(2**32)


def sample_directions(
    led: LED,
    emission_axis_2d: tuple[float, float],
    *,
    ray_count: int,
) -> np.ndarray:
    """Return a deterministic Hammersley-like 3D direction set.

    With ``s = sin(theta_max)*sqrt(u)``, the radial area measure in the
    truncated cone is sampled uniformly; azimuth is the base-two radical
    inverse.  This is an idealized deterministic extension, not a measured
    LED radiation pattern.
    """
    axis_xy = np.asarray(emission_axis_2d, dtype=float)
    axis_xy /= np.linalg.norm(axis_xy)
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=float)
    basis_a = np.asarray([-axis_xy[1], axis_xy[0], 0.0], dtype=float)
    basis_b = np.asarray([0.0, 0.0, 1.0], dtype=float)
    half_angle = radians(led.emission_half_angle_deg)
    sin_max = np.sin(half_angle)
    indices = np.arange(ray_count, dtype=np.float64)
    u = (indices + 0.5) / ray_count
    radial_sine = sin_max * np.sqrt(u)
    axial_cosine = np.sqrt(np.maximum(0.0, 1.0 - radial_sine * radial_sine))
    phi = 2.0 * np.pi * _radical_inverse_base_two(
        np.arange(1, ray_count + 1, dtype=np.uint32)
    )
    directions = (
        axial_cosine[:, None] * axis[None, :]
        + (radial_sine * np.cos(phi))[:, None] * basis_a[None, :]
        + (radial_sine * np.sin(phi))[:, None] * basis_b[None, :]
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions.astype(np.float32)

__all__ = ["sample_directions"]
