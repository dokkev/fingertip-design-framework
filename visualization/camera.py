"""Display helper for optional camera-render results."""

from __future__ import annotations

from math import isfinite
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _display_rgb(
    linear_rgb: np.ndarray,
    *,
    normalization_max: float | None,
    gamma: float,
) -> np.ndarray:
    if not isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and greater than zero")
    if linear_rgb.ndim not in (2, 3) or linear_rgb.ndim == 3 and linear_rgb.shape[2] not in (1, 3, 4):
        raise ValueError("linear_rgb must be a grayscale or RGB image")
    if not np.all(np.isfinite(linear_rgb)) or np.any(linear_rgb < 0.0):
        raise ValueError("linear_rgb must be finite and nonnegative")
    scale = (
        float(np.max(linear_rgb))
        if normalization_max is None
        else float(normalization_max)
    )
    if normalization_max is not None and (not isfinite(scale) or scale <= 0.0):
        raise ValueError("normalization_max must be finite and positive")
    normalized = (
        np.zeros_like(linear_rgb)
        if scale <= 0.0
        else np.clip(linear_rgb / scale, 0.0, 1.0)
    )
    return normalized ** (1.0 / gamma)


def plot_camera(
    result: Any,
    *,
    ax: Axes | None = None,
    normalization_max: float | None = None,
    gamma: float = 2.2,
    title: str = "Mitsuba camera validation",
) -> Axes:
    """Display a normalized copy of ``result.linear_rgb`` without mutation."""
    try:
        linear_rgb = np.asarray(result.linear_rgb, dtype=float)
    except AttributeError as exc:
        raise TypeError("result must expose a linear_rgb image") from exc
    display = _display_rgb(
        linear_rgb,
        normalization_max=normalization_max,
        gamma=gamma,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.imshow(display)
    ax.set_title(title)
    ax.axis("off")
    return ax


__all__ = ["plot_camera"]
