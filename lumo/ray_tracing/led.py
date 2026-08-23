"""Current Green Sequin LED point-source approximation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

import numpy as np

from .transport import lambertian_emission


_EMISSION_DTYPE = np.dtype(
    [
        ("origin_W_m", np.float64, (3,)),
        ("direction_W", np.float64, (3,)),
        ("power", np.float64),
    ]
)


@dataclass(frozen=True)
class LED:
    """One Adafruit Green LED Sequin modeled as a Lambertian point source.

    ``power`` is normalized modeled optical power. It is not a datasheet
    optical-watt value and remains uncalibrated.
    """

    ADAFRUIT_PRODUCT_ID: ClassVar[int] = 1756
    LED_PART_NUMBER: ClassVar[str] = "LuckyLight S150PGC-G5-1B"
    PACKAGE: ClassVar[str] = "1206 Pure Green InGaN"

    position_W_m: np.ndarray
    normal_W: np.ndarray
    power: float = 1.0
    dominant_wavelength_nm: float = 525.0
    peak_wavelength_nm: float = 520.0
    spectral_half_width_nm: float = 35.0
    viewing_half_angle_deg: float = 60.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_W_m, dtype=np.float64)
        normal = np.asarray(self.normal_W, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_W_m must be a finite vector of shape (3,)")
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("normal_W must be a finite vector of shape (3,)")
        with np.errstate(over="ignore"):
            normal_norm = float(np.linalg.norm(normal))
        if not isfinite(normal_norm) or normal_norm <= np.finfo(np.float64).tiny:
            raise ValueError("normal_W must have a finite nonzero norm")

        for name in (
            "dominant_wavelength_nm",
            "peak_wavelength_nm",
            "spectral_half_width_nm",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.power) or self.power < 0.0:
            raise ValueError("power must be finite and nonnegative")
        if not (
            isfinite(self.viewing_half_angle_deg)
            and 0.0 < self.viewing_half_angle_deg < 90.0
        ):
            raise ValueError(
                "viewing_half_angle_deg must be finite and between 0 and 90"
            )

        position = position.copy()
        normal = normal / normal_norm
        position.setflags(write=False)
        normal.setflags(write=False)
        object.__setattr__(self, "position_W_m", position)
        object.__setattr__(self, "normal_W", normal)

    def emit(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """Emit deterministic equal-power rays using caller-supplied samples."""
        u1 = np.asarray(u1, dtype=np.float64)
        u2 = np.asarray(u2, dtype=np.float64)
        if u1.ndim != 1 or u2.shape != u1.shape or not len(u1):
            raise ValueError("u1 and u2 must be nonempty arrays of equal shape")

        sampled = lambertian_emission(
            np.repeat(self.normal_W[None, :], len(u1), axis=0),
            total_power=self.power,
            u1=u1,
            u2=u2,
        )
        emission = np.empty(len(u1), dtype=_EMISSION_DTYPE)
        emission["origin_W_m"] = self.position_W_m
        emission["direction_W"] = sampled["direction"]
        emission["power"] = sampled["ray_power"]
        return emission


__all__ = ["LED"]
