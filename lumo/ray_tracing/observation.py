"""Side-view observations derived from escaped optical paths."""

from __future__ import annotations

import numpy as np

from lumo.fingertip import Fingertip
from lumo.mesh import MAIN_Y_BOUNDS_MM


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

    response = np.zeros(4, dtype=np.float64)
    origins, directions, power = _escaped_ray_fields(escaped_rays)
    if not len(origins):
        return response

    left, right = fingertip.silicone.semiellipse_endpoints
    center_x_m = 0.5e-3 * (left[0] + right[0])
    center_z_m = 1.0e-3 * fingertip.silicone.ellipse_center_z_mm

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


def longitudinal_side_view_observation(
    escaped_rays: np.ndarray,
    *,
    linear_splat: bool = False,
) -> np.ndarray:
    """Accumulate +X-facing power into eleven 5 mm bins along world Y.

    The fixed receiver ROI is the 55 mm active section, ``Y=[-27.5,+27.5]``
    mm. The camera looks toward the canonical ``+X`` side, so its horizontal
    image coordinate is fingertip-longitudinal ``Y``. Contributions from every
    simultaneously active LED are added according to escape position; emitter
    identity is not part of the observation.
    """
    response, _, _ = longitudinal_side_view_power(
        escaped_rays,
        linear_splat=linear_splat,
    )
    return response


def longitudinal_side_view_power(
    escaped_rays: np.ndarray,
    *,
    linear_splat: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Return active-ROI bins, outside-ROI power, and total +X power.

    When ``linear_splat`` is true, each active-ROI ray distributes its power
    between the two nearest bin centers. Rays in the half-bin strips at the ROI
    ends accumulate entirely in the nearest end bin. This preserves power and
    removes internal hard-bin boundary jumps without introducing a point-spread
    parameter.
    """
    if not isinstance(linear_splat, bool):
        raise TypeError("linear_splat must be a bool")
    response = np.zeros(LONGITUDINAL_SIDE_BIN_COUNT, dtype=np.float64)
    origins, directions, power = _escaped_ray_fields(escaped_rays)
    if not len(origins):
        return response, 0.0, 0.0

    y_min_m, y_max_m = (
        _MM_TO_M * value for value in MAIN_Y_BOUNDS_MM
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
    if linear_splat and len(active_y_m):
        bin_width_m = (y_max_m - y_min_m) / LONGITUDINAL_SIDE_BIN_COUNT
        center_coordinate = (active_y_m - y_min_m) / bin_width_m - 0.5
        lower_bin = np.floor(center_coordinate).astype(np.int64)
        upper_weight = center_coordinate - lower_bin

        below_first_center = lower_bin < 0
        above_last_center = lower_bin >= LONGITUDINAL_SIDE_BIN_COUNT - 1
        interior = ~(below_first_center | above_last_center)
        response[0] += float(active_power[below_first_center].sum())
        response[-1] += float(active_power[above_last_center].sum())
        np.add.at(
            response,
            lower_bin[interior],
            active_power[interior] * (1.0 - upper_weight[interior]),
        )
        np.add.at(
            response,
            lower_bin[interior] + 1,
            active_power[interior] * upper_weight[interior],
        )
    else:
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
    "longitudinal_side_view_observation",
    "longitudinal_side_view_power",
    "side_view_observation",
]
