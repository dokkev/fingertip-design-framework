"""Deterministic primary-ray sets for the 2D-to-3D comparison."""

from __future__ import annotations

from math import radians

import numpy as np

from model.optical import LED
from optics.cross_section.settings import TraceSettings
from optics.cross_section.transport import _sample_primary_directions


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
    mode: str,
    ray_count: int,
) -> np.ndarray:
    """Return one deterministic direction set for one comparison family.

    Planar mode delegates to the existing reduced sampler, so its directions
    are numerically identical.  Full 3D uses a deterministic Hammersley-like
    sequence.  With ``s = sin(theta_max)*sqrt(u)``, the radial area measure in
    the truncated cone is sampled uniformly; azimuth is the base-two radical
    inverse.  This is an idealized deterministic extension, not a measured LED
    radiation pattern.
    """
    if mode == "planar":
        directions = _sample_primary_directions(
            led,
            emission_axis_2d,
            TraceSettings(ray_count=ray_count),
        )
        # Keep the reduced sampler's float64 values exact at this API
        # boundary; the OptiX backend performs the explicit float32 upload
        # required by traversal after this equality contract is checked.
        result = np.asarray(
            [[float(direction[0]), float(direction[1]), 0.0] for direction in directions],
            dtype=float,
        )
        return result
    if mode != "full3d":
        raise ValueError(f"unsupported 3D sampling mode: {mode!r}")

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


def sample_planar_directions(
    led: LED,
    emission_axis_2d: tuple[float, float],
    ray_count: int,
) -> np.ndarray:
    """Return the exact reduced-tracer planar direction set."""
    return sample_directions(
        led,
        emission_axis_2d,
        mode="planar",
        ray_count=ray_count,
    )


__all__ = ["sample_directions", "sample_planar_directions"]
