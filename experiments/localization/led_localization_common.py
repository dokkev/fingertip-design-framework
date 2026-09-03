"""Shared result contract for material-specific LED localization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LedLocalizationResult:
    """Five ordered LED positions and the image geometry that produced them.

    LEDs and ``longitudinal_positions_mm`` are ordered distal to proximal.
    Image lines use homogeneous coefficients ``[a, b, c]`` for
    ``a*x + b*y + c = 0``. Center and inter-LED responses are material-
    specific diagnostics of the optical score used to place the rigid array.
    """

    image_shape: tuple[int, int]
    led_centers_xy_px: np.ndarray
    longitudinal_positions_mm: np.ndarray
    led_line: np.ndarray
    dorsal_line: np.ndarray
    palmar_line: np.ndarray
    distal_limit: np.ndarray
    vanishing_point_h: np.ndarray
    led_center_responses: np.ndarray
    inter_led_responses: np.ndarray
    reference_mask: np.ndarray
    led_line_alpha: float
    longitudinal_scale_px_per_mm: float
    line_score: float

    def __post_init__(self) -> None:
        height, width = self.image_shape
        if (
            not isinstance(height, int)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or min(height, width) < 2
        ):
            raise ValueError("image_shape must contain two integer dimensions >= 2")

        arrays = {
            "led_centers_xy_px": ((5, 2), np.float64),
            "longitudinal_positions_mm": ((5,), np.float64),
            "led_line": ((3,), np.float64),
            "dorsal_line": ((3,), np.float64),
            "palmar_line": ((3,), np.float64),
            "distal_limit": ((3,), np.float64),
            "vanishing_point_h": ((3,), np.float64),
            "led_center_responses": ((5,), np.float64),
            "inter_led_responses": ((4,), np.float64),
        }
        for name, (shape, dtype) in arrays.items():
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        if not np.all(np.diff(self.longitudinal_positions_mm) > 0.0):
            raise ValueError(
                "longitudinal_positions_mm must increase distal to proximal"
            )
        mask = np.asarray(self.reference_mask, dtype=bool)
        if mask.shape != self.image_shape or not np.any(mask):
            raise ValueError("reference_mask must be nonempty and match image_shape")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "reference_mask", mask)

        if (
            not np.isfinite(self.led_line_alpha)
            or not 0.0 <= self.led_line_alpha <= 1.0
        ):
            raise ValueError("led_line_alpha must be finite and in [0, 1]")
        if (
            not np.isfinite(self.longitudinal_scale_px_per_mm)
            or self.longitudinal_scale_px_per_mm <= 0.0
        ):
            raise ValueError("longitudinal_scale_px_per_mm must be positive and finite")
        if not np.isfinite(self.line_score):
            raise ValueError("line_score must be finite")


__all__ = ["LedLocalizationResult"]
