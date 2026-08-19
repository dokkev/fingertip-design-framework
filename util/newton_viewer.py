"""Optional Newton viewer helpers for interactive examples.

This module owns viewer selection and window lifetime.  The mechanics backend
only receives an already-created Newton viewer and remains responsible for
logging accepted solver states and contacts.
"""

from __future__ import annotations

import time
from pathlib import Path

import newton
import warp as wp


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
    # The scene is millimetre-scale at the repository boundary and metre-scale
    # inside Newton.  Keep the fingertip and indenter in the initial view.
    viewer.set_camera(wp.vec3(0.06, -0.08, 0.04), pitch=-10.0, yaw=90.0)
    return viewer


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
    "keep_newton_viewer_open",
    "make_newton_viewer",
]
