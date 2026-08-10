"""Display normalization and PNG export for raw camera-render results."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optics.mitsuba.result import RenderResult


def _display_rgb(
    linear_rgb: np.ndarray,
    *,
    normalization_max: float | None,
    gamma: float,
) -> np.ndarray:
    if not isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and greater than zero")
    scale = (
        float(np.max(linear_rgb))
        if normalization_max is None
        else float(normalization_max)
    )
    if normalization_max is not None and (
        not isfinite(scale) or scale <= 0.0
    ):
        raise ValueError("normalization_max must be finite and positive")
    normalized = (
        np.zeros_like(linear_rgb)
        if scale <= 0.0
        else np.clip(linear_rgb / scale, 0.0, 1.0)
    )
    return normalized ** (1.0 / gamma)


def save_camera_render(
    result: RenderResult,
    output_path: str | Path,
    *,
    normalization_max: float | None = None,
    gamma: float = 1.0,
) -> Path:
    """Normalize a copy for display and save it without changing raw RGB."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        output,
        _display_rgb(
            result.linear_rgb,
            normalization_max=normalization_max,
            gamma=gamma,
        ),
    )
    return output


def save_camera_comparison(
    reference: RenderResult,
    loaded: RenderResult,
    output_path: str | Path,
) -> Path:
    """Save reference, loaded, and absolute raw difference on one scale."""
    if reference.linear_rgb.shape != loaded.linear_rgb.shape:
        raise ValueError("camera images must have the same shape")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shared_scale = max(
        float(np.max(reference.linear_rgb)),
        float(np.max(loaded.linear_rgb)),
    )
    difference = np.abs(loaded.linear_rgb - reference.linear_rgb)
    images = (
        _display_rgb(
            reference.linear_rgb,
            normalization_max=shared_scale if shared_scale > 0.0 else None,
            gamma=1.0,
        ),
        _display_rgb(
            loaded.linear_rgb,
            normalization_max=shared_scale if shared_scale > 0.0 else None,
            gamma=1.0,
        ),
        _display_rgb(
            difference,
            normalization_max=shared_scale if shared_scale > 0.0 else None,
            gamma=1.0,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for axis, image, title in zip(
        axes,
        images,
        ("Reference", "Loaded", "Absolute difference"),
        strict=True,
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
