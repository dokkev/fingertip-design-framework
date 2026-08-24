"""Optional Newton viewer helpers for interactive examples.

This module owns viewer selection and window lifetime.  The mechanics backend
only receives an already-created Newton viewer and remains responsible for
logging accepted solver states and contacts.
"""

from __future__ import annotations

import time
from pathlib import Path

import newton
import numpy as np


def make_newton_viewer(
    *,
    no_viewer: bool,
    usd_path: Path | None,
    num_frames: int,
):
    """Create the requested Newton viewer, or ``None`` for headless runs."""

    if no_viewer:
        return None
    if usd_path is not None:
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        return newton.viewer.ViewerUSD(
            str(usd_path),
            fps=60,
            num_frames=num_frames,
        )

    viewer = newton.viewer.ViewerGL(
        width=1400,
        height=900,
        vsync=False,
        paused=False,
    )
    return viewer


def frame_newton_viewer(
    viewer: object | None,
    bounds_min_m: tuple[float, float, float] | np.ndarray,
    bounds_max_m: tuple[float, float, float] | np.ndarray,
    *,
    view_direction: tuple[float, float, float] = (0.0, -1.0, 0.4),
    padding: float = 1.4,
) -> None:
    """Frame a Newton GL viewer around scene bounds in metres.

    ``view_direction`` points from the scene target toward the camera.  The
    default therefore places the camera on the negative-y, positive-z side of
    the fingertip, matching the repository's physical viewing convention.
    USD viewers do not expose a live camera and are left unchanged.
    """

    if viewer is None or not hasattr(viewer, "camera"):
        return

    lower = np.asarray(bounds_min_m, dtype=float)
    upper = np.asarray(bounds_max_m, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("viewer bounds must each have shape (3,)")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("viewer bounds must be finite")
    if np.any(upper < lower):
        raise ValueError("viewer bounds must satisfy max >= min")

    extent = float(np.max(upper - lower))
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("viewer bounds must span a positive scene extent")

    direction = np.asarray(view_direction, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("view_direction must contain three finite values")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise ValueError("view_direction must be nonzero")
    direction /= direction_norm

    padding = float(padding)
    if not np.isfinite(padding) or padding <= 1.0:
        raise ValueError("padding must be finite and greater than one")

    camera = viewer.camera
    camera.fov = 30.0
    fov_half_rad = np.deg2rad(camera.fov * 0.5)
    center = 0.5 * (lower + upper)
    distance = 0.5 * extent * padding / np.tan(fov_half_rad)

    # Newton's default minimum pivot distance is 50 mm, which is larger than
    # the complete demo scene.  Scale it to the actual metre-sized bounds.
    camera.MIN_PIVOT_DISTANCE = max(1.0e-6, 0.1 * extent)
    distance = max(distance, 1.1 * camera.MIN_PIVOT_DISTANCE)
    camera.near = max(1.0e-6, 0.01 * extent)
    camera.far = max(1.0, distance + padding * extent)

    camera_position = center + direction * distance
    camera.pos = type(camera.pos)(
        float(camera_position[0]),
        float(camera_position[1]),
        float(camera_position[2]),
    )
    target = type(camera.pos)(float(center[0]), float(center[1]), float(center[2]))
    camera.look_at(target)


def keep_newton_viewer_open(viewer: object, *, poll_interval_s: float = 0.02) -> None:
    """Poll a live Newton GUI until the user closes it."""

    print("ViewerGL is active; close the window to finish.")
    while viewer.is_running():
        viewer.end_frame()
        time.sleep(poll_interval_s)


def close_newton_viewer(viewer: object | None) -> None:
    """Close a Newton viewer if one was created by an example."""

    if viewer is not None:
        viewer.close()


__all__ = [
    "close_newton_viewer",
    "frame_newton_viewer",
    "keep_newton_viewer_open",
    "make_newton_viewer",
]
