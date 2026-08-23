"""Side-view observations derived from escaped optical paths."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip


def side_view_observation(
    escaped_rays: np.ndarray,
    *,
    fingertip: Fingertip,
) -> np.ndarray:
    """Accumulate +Y-facing escaped power into four X-Z quadrants.

    Quadrants are ordered ``Q1, Q2, Q3, Q4``: upper-right, upper-left,
    lower-left, lower-right. The origin is the analytic silicone semiellipse
    center in the canonical LUMO X-Z cross section.
    """
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")

    rays = np.asarray(escaped_rays)
    if rays.ndim != 1 or rays.dtype.names is None:
        raise ValueError("escaped_rays must be a structured array")
    required_fields = {"origin_W_m", "direction_W", "power"}
    if not required_fields.issubset(rays.dtype.names):
        raise ValueError(
            "escaped_rays requires origin_W_m, direction_W, and power"
        )

    response = np.zeros(4, dtype=np.float64)
    if not len(rays):
        return response

    left, right = fingertip.silicone.semiellipse_endpoints
    center_x_m = 0.5e-3 * (left[0] + right[0])
    center_z_m = 1.0e-3 * fingertip.silicone.ellipse_center_z_mm

    origins = np.asarray(rays["origin_W_m"], dtype=np.float64)
    directions = np.asarray(rays["direction_W"], dtype=np.float64)
    power = np.asarray(rays["power"], dtype=np.float64)
    if (
        origins.shape != (len(rays), 3)
        or directions.shape != origins.shape
        or power.shape != (len(rays),)
        or not np.all(np.isfinite(origins))
        or not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(power))
    ):
        raise ValueError("escaped ray fields must be finite and well shaped")
    if np.any(power < 0.0):
        raise ValueError("escaped ray power must be nonnegative")

    visible = directions[:, 1] > 0.0
    x_m = origins[visible, 0]
    z_m = origins[visible, 2]
    right_side = x_m >= center_x_m
    upper = z_m >= center_z_m
    quadrant = np.where(
        upper,
        np.where(right_side, 0, 1),
        np.where(right_side, 3, 2),
    )
    np.add.at(response, quadrant, power[visible])

    return response


__all__ = ["side_view_observation"]
