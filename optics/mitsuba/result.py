"""Neutral raw camera-render result from the optional Mitsuba backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RenderResult:
    """Unnormalized linear RGB data and render-state metadata."""

    linear_rgb: np.ndarray
    spp: int
    relative_led_power: float
    state_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = np.array(self.linear_rgb, dtype=float, copy=True)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("linear_rgb must have shape (height, width, 3)")
        if not np.all(np.isfinite(image)) or np.any(image < 0.0):
            raise ValueError("linear_rgb must be finite and nonnegative")
        if (
            not isinstance(self.spp, int)
            or isinstance(self.spp, bool)
            or self.spp < 1
        ):
            raise ValueError("spp must be a positive integer")
        if (
            not isfinite(self.relative_led_power)
            or self.relative_led_power < 0.0
        ):
            raise ValueError(
                "relative_led_power must be finite and nonnegative"
            )
        image.setflags(write=False)
        object.__setattr__(self, "linear_rgb", image)
        object.__setattr__(
            self,
            "state_metadata",
            MappingProxyType(dict(self.state_metadata)),
        )
