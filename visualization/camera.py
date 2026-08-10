"""Display helpers for optional camera-render results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from optics.mitsuba import RenderResult


def plot_camera(
    result: RenderResult,
    *,
    ax: Axes | None = None,
    gamma: float = 2.2,
    title: str = "Mitsuba camera validation",
) -> Axes:
    """Display a normalized copy of a raw linear-RGB render."""
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and greater than zero")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 7.0))
    scale = float(np.max(result.linear_rgb))
    normalized = (
        np.zeros_like(result.linear_rgb)
        if scale <= 0.0
        else np.clip(result.linear_rgb / scale, 0.0, 1.0)
    )
    ax.imshow(normalized ** (1.0 / gamma))
    ax.set_title(title)
    ax.axis("off")
    return ax


__all__ = ["plot_camera"]
