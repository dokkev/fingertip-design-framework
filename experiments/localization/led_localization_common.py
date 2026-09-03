"""Shared result contract for material-specific LED localization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LedLocalizationResult:
    """Five ordered Solaris LED centers and their one-dimensional evidence."""

    image_shape: tuple[int, int]
    led_centers_xy_px: np.ndarray
    led_line: np.ndarray
    selected_side: str
    peak_rows_px: np.ndarray
    profile_rows_px: np.ndarray
    red_profile_dn: np.ndarray
    red_contrast_dn: np.ndarray
    peak_prominences_dn: np.ndarray
    sequence_score: float
    reference_mask: np.ndarray

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
        if self.selected_side not in {"left", "right"}:
            raise ValueError("selected_side must be left or right")
        if not np.isfinite(self.sequence_score):
            raise ValueError("sequence_score must be finite")

        arrays = {
            "led_centers_xy_px": ((5, 2), np.float64),
            "led_line": ((3,), np.float64),
            "peak_rows_px": ((5,), np.float64),
            "peak_prominences_dn": ((5,), np.float64),
        }
        for name, (shape, dtype) in arrays.items():
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        if not np.all(np.diff(self.peak_rows_px) > 0.0):
            raise ValueError("peak_rows_px must increase distal to proximal")
        if np.any(self.peak_prominences_dn < 0.0):
            raise ValueError("peak_prominences_dn must be nonnegative")

        profile_rows = np.asarray(self.profile_rows_px, dtype=np.float64)
        raw = np.asarray(self.red_profile_dn, dtype=np.float64)
        contrast = np.asarray(self.red_contrast_dn, dtype=np.float64)
        if (
            profile_rows.ndim != 1
            or len(profile_rows) < 2
            or raw.shape != profile_rows.shape
            or contrast.shape != profile_rows.shape
            or not np.all(np.isfinite(profile_rows))
            or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(contrast))
        ):
            raise ValueError("profile arrays must be matching finite vectors")
        for name, value in (
            ("profile_rows_px", profile_rows),
            ("red_profile_dn", raw),
            ("red_contrast_dn", contrast),
        ):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        mask = np.asarray(self.reference_mask, dtype=bool)
        if mask.shape != self.image_shape or not np.any(mask):
            raise ValueError("reference_mask must be nonempty and match image_shape")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "reference_mask", mask)


__all__ = ["LedLocalizationResult"]
