"""Side-view observations derived from escaped optical paths."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import ACTIVE_Y_BOUNDS_MM


LONGITUDINAL_SIDE_BIN_COUNT = 11
_MM_TO_M = 1.0e-3


def _escaped_ray_fields(
    escaped_rays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = np.asarray(escaped_rays)
    if rays.ndim != 1 or rays.dtype.names is None:
        raise ValueError("escaped_rays must be a structured array")
    required_fields = {"origin_W_m", "direction_W", "power"}
    if not required_fields.issubset(rays.dtype.names):
        raise ValueError(
            "escaped_rays requires origin_W_m, direction_W, and power"
        )

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
    return origins, directions, power


def longitudinal_side_view_power(
    escaped_rays: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return hard 5 mm active-ROI bins, outside power, and total +X power."""
    response = np.zeros(LONGITUDINAL_SIDE_BIN_COUNT, dtype=np.float64)
    origins, directions, power = _escaped_ray_fields(escaped_rays)
    if not len(origins):
        return response, 0.0, 0.0

    y_min_m, y_max_m = (
        _MM_TO_M * value for value in ACTIVE_Y_BOUNDS_MM
    )
    edges_m = np.linspace(
        y_min_m,
        y_max_m,
        LONGITUDINAL_SIDE_BIN_COUNT + 1,
    )
    camera_visible = directions[:, 0] > 0.0
    active_section = (
        camera_visible
        & (origins[:, 1] >= y_min_m)
        & (origins[:, 1] <= y_max_m)
    )
    active_y_m = origins[active_section, 1]
    active_power = power[active_section]
    response[:] = np.histogram(
        active_y_m,
        bins=edges_m,
        weights=active_power,
    )[0]
    visible_power = float(power[camera_visible].sum())
    inside_roi_power = float(response.sum())
    outside_roi_power = visible_power - inside_roi_power
    tolerance = 1.0e-12 * max(1.0, visible_power)
    if outside_roi_power < -tolerance:
        raise RuntimeError("longitudinal side-view ROI power exceeds +X power")
    outside_roi_power = max(0.0, outside_roi_power)
    return response, outside_roi_power, visible_power


__all__ = [
    "LONGITUDINAL_SIDE_BIN_COUNT",
    "longitudinal_side_view_power",
]
