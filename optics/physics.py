"""Dependency-light scalar geometric-optics interface physics."""

from __future__ import annotations

import numpy as np


class OpticalPhysicsError(RuntimeError):
    """Raised when an interface calculation is geometrically invalid."""


def interface_directions_and_reflectance(
    incident_direction: np.ndarray,
    interface_normal: np.ndarray,
    refractive_index_1: float,
    refractive_index_2: float,
) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
    """Return reflected/transmitted directions, Fresnel R, and TIR status.

    The normal points from medium 1 into medium 2.  The scalar formula is the
    existing reduced 2D convention, generalized to explicit refractive indices.
    """
    incident = np.asarray(incident_direction, dtype=float)
    normal = np.asarray(interface_normal, dtype=float)
    incident /= np.linalg.norm(incident)
    normal /= np.linalg.norm(normal)
    if not np.all(np.isfinite(incident)) or not np.all(np.isfinite(normal)):
        raise OpticalPhysicsError("interface directions must be finite")
    if refractive_index_1 <= 0.0 or refractive_index_2 <= 0.0:
        raise OpticalPhysicsError("refractive indices must be positive")
    alignment = float(np.dot(incident, normal))
    if alignment <= 1.0e-12:
        raise OpticalPhysicsError(
            "the interface normal does not point into the next medium"
        )

    tangent_component = incident - alignment * normal
    sin_incident = float(np.linalg.norm(tangent_component))
    if sin_incident > 1.0e-15:
        tangent_direction = tangent_component / sin_incident
    else:
        tangent_direction = np.zeros_like(incident)
    sin_transmitted = (
        refractive_index_1 / refractive_index_2
    ) * sin_incident
    reflected = incident - 2.0 * alignment * normal
    reflected /= np.linalg.norm(reflected)
    if abs(sin_transmitted) > 1.0:
        return reflected, None, 1.0, True

    cos_incident = abs(alignment)
    cos_transmitted = max(0.0, 1.0 - sin_transmitted**2) ** 0.5
    transmitted = tangent_direction * sin_transmitted + normal * cos_transmitted
    transmitted /= np.linalg.norm(transmitted)
    s_denominator = (
        refractive_index_1 * cos_incident
        + refractive_index_2 * cos_transmitted
    )
    p_denominator = (
        refractive_index_1 * cos_transmitted
        + refractive_index_2 * cos_incident
    )
    if s_denominator <= 0.0 or p_denominator <= 0.0:
        raise OpticalPhysicsError("the Fresnel denominator is nonpositive")
    reflectance_s = (
        (
            refractive_index_1 * cos_incident
            - refractive_index_2 * cos_transmitted
        )
        / s_denominator
    ) ** 2
    reflectance_p = (
        (
            refractive_index_1 * cos_transmitted
            - refractive_index_2 * cos_incident
        )
        / p_denominator
    ) ** 2
    reflectance = float(
        np.clip(0.5 * (reflectance_s + reflectance_p), 0.0, 1.0)
    )
    return reflected, transmitted, reflectance, False


__all__ = ["OpticalPhysicsError", "interface_directions_and_reflectance"]
