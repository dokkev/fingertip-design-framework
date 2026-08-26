"""Current Green Sequin LED point-source approximation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from lumo.fingertip import LEDParameters

from .scene import OptixScene, safe_secondary_origins
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

    ``parameters.normalized_power`` is modeled optical power. It is not a
    datasheet optical-watt value and remains uncalibrated.
    """

    position_W_m: np.ndarray
    normal_W: np.ndarray
    parameters: LEDParameters

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, LEDParameters):
            raise TypeError("parameters must be LEDParameters")
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
            total_power=self.parameters.normalized_power,
            u1=u1,
            u2=u2,
        )
        emission = np.empty(len(u1), dtype=_EMISSION_DTYPE)
        emission["origin_W_m"] = self.position_W_m
        emission["direction_W"] = sampled["direction"]
        emission["power"] = sampled["ray_power"]
        return emission


def emit_from_stem_boundary(
    scene: OptixScene,
    led: LED,
    u1: np.ndarray,
    u2: np.ndarray,
    *,
    carrier_mask: int,
) -> np.ndarray:
    """Emit one LED from its resolved carrier stem boundary."""
    probe_distance_m = 0.5e-3 * led.parameters.height_mm
    probe_origin = (led.position_W_m - probe_distance_m * led.normal_W)[None, :]
    direction = led.normal_W[None, :]
    carrier_hit = scene.trace_closest(
        probe_origin,
        direction,
        mask=carrier_mask,
    )
    if not carrier_hit["hit"][0]:
        raise RuntimeError("carrier probe did not find the LED stem boundary")
    hit_position = probe_origin[0] + carrier_hit["t"][0] * led.normal_W
    if not np.allclose(
        hit_position,
        led.position_W_m,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("carrier probe found the wrong LED stem boundary")

    emission = led.emit(u1, u2)
    emission["origin_W_m"] = safe_secondary_origins(
        carrier_hit,
        direction,
    )[0]
    return emission


def source_inside_silicone(
    scene: OptixScene,
    led: LED,
    emission: np.ndarray,
    *,
    silicone_mask: int,
) -> bool:
    """Resolve the emitted ray's initial medium from the silicone boundary."""
    silicone_hit = scene.trace_closest(
        emission["origin_W_m"][:1],
        led.normal_W[None, :],
        mask=silicone_mask,
    )[0]
    if not silicone_hit["hit"]:
        raise RuntimeError("the LED normal does not reach silicone")
    normal_projection = float(np.dot(silicone_hit["normal_W"], led.normal_W))
    if abs(normal_projection) <= 1.0e-6:
        raise RuntimeError("the LED source interface is geometrically ambiguous")
    return normal_projection > 0.0


__all__ = ["LED", "emit_from_stem_boundary", "source_inside_silicone"]
