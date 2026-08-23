"""Single-interface dielectric optical transport."""

from __future__ import annotations

from math import isfinite

import numpy as np


_RESULT_DTYPE = np.dtype(
    [
        ("reflected_direction", np.float64, (3,)),
        ("refracted_direction", np.float64, (3,)),
        ("reflectance", np.float64),
        ("transmittance", np.float64),
        ("reflected_power", np.float64),
        ("refracted_power", np.float64),
        ("total_internal_reflection", np.bool_),
    ]
)

_LAMBERTIAN_RESULT_DTYPE = np.dtype(
    [
        ("reflected_direction", np.float64, (3,)),
        ("reflected_power", np.float64),
        ("absorbed_power", np.float64),
    ]
)


def _normalized_vectors(value: np.ndarray, *, name: str) -> np.ndarray:
    vectors = np.asarray(value, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not len(vectors):
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must be finite")

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError(f"{name} must contain nonzero vectors")
    return vectors / norms[:, None]


def interface_transport(
    directions: np.ndarray,
    normals: np.ndarray,
    *,
    n_incident: float,
    n_transmitted: float,
    incident_power: float | np.ndarray,
) -> np.ndarray:
    """Apply one lossless dielectric interaction to a batch of rays.

    ``n_incident`` and ``n_transmitted`` are supplied explicitly; this function
    does not infer the current medium. Each geometric normal is locally flipped
    when necessary so that it opposes its incident direction. ``incident_power``
    is either one nonnegative scalar for the batch or one value per ray.
    """
    directions = _normalized_vectors(directions, name="directions")
    normals = _normalized_vectors(normals, name="normals")
    if directions.shape != normals.shape:
        raise ValueError("directions and normals must have the same shape")

    incident_power = np.asarray(incident_power, dtype=np.float64)
    if incident_power.ndim == 0:
        incident_power = np.full(len(directions), incident_power.item())
    elif incident_power.shape != (len(directions),):
        raise ValueError("incident_power must be a scalar or have shape (N,)")
    if not np.all(np.isfinite(incident_power)):
        raise ValueError("incident_power must be finite")
    if np.any(incident_power < 0.0):
        raise ValueError("incident_power must be nonnegative")

    n_incident = float(n_incident)
    n_transmitted = float(n_transmitted)
    if not isfinite(n_incident) or n_incident <= 0.0:
        raise ValueError("n_incident must be finite and positive")
    if not isfinite(n_transmitted) or n_transmitted <= 0.0:
        raise ValueError("n_transmitted must be finite and positive")

    dots = np.sum(directions * normals, axis=1)
    oriented_normals = np.where((dots > 0.0)[:, None], -normals, normals)
    cosine_incident = np.clip(
        -np.sum(directions * oriented_normals, axis=1),
        0.0,
        1.0,
    )

    reflected = directions + 2.0 * cosine_incident[:, None] * oriented_normals
    reflected /= np.linalg.norm(reflected, axis=1)[:, None]

    result = np.empty(len(directions), dtype=_RESULT_DTYPE)
    result["reflected_direction"] = reflected
    result["refracted_direction"] = np.nan

    if n_incident == n_transmitted:
        result["refracted_direction"] = directions
        result["reflectance"] = 0.0
        result["transmittance"] = 1.0
        result["reflected_power"] = 0.0
        result["refracted_power"] = incident_power
        result["total_internal_reflection"] = False
        return result

    index_ratio = n_incident / n_transmitted
    sine_transmitted_squared = index_ratio**2 * (1.0 - cosine_incident**2)
    total_internal_reflection = sine_transmitted_squared > 1.0
    transmitted = ~total_internal_reflection
    cosine_transmitted = np.sqrt(
        np.maximum(0.0, 1.0 - sine_transmitted_squared)
    )

    refracted = (
        index_ratio * directions[transmitted]
        + (
            index_ratio * cosine_incident[transmitted]
            - cosine_transmitted[transmitted]
        )[:, None]
        * oriented_normals[transmitted]
    )
    refracted /= np.linalg.norm(refracted, axis=1)[:, None]
    result["refracted_direction"][transmitted] = refracted

    reflectance = np.ones(len(directions), dtype=np.float64)
    cosine_i = cosine_incident[transmitted]
    cosine_t = cosine_transmitted[transmitted]
    reflection_s = (
        n_incident * cosine_i - n_transmitted * cosine_t
    ) / (
        n_incident * cosine_i + n_transmitted * cosine_t
    )
    reflection_p = (
        n_transmitted * cosine_i - n_incident * cosine_t
    ) / (
        n_transmitted * cosine_i + n_incident * cosine_t
    )
    reflectance[transmitted] = 0.5 * (
        reflection_s**2 + reflection_p**2
    )

    result["reflectance"] = reflectance
    result["transmittance"] = 1.0 - reflectance
    result["reflected_power"] = incident_power * result["reflectance"]
    result["refracted_power"] = incident_power * result["transmittance"]
    result["total_internal_reflection"] = total_internal_reflection
    return result


def lambertian_reflection(
    directions: np.ndarray,
    normals: np.ndarray,
    *,
    incident_power: float | np.ndarray,
    albedo: float | np.ndarray,
    u1: float | np.ndarray,
    u2: float | np.ndarray,
) -> np.ndarray:
    """Sample one opaque cosine-weighted Lambertian reflection per ray."""
    directions = _normalized_vectors(directions, name="directions")
    normals = _normalized_vectors(normals, name="normals")
    if directions.shape != normals.shape:
        raise ValueError("directions and normals must have the same shape")

    batch_values = []
    for name, value in (
        ("incident_power", incident_power),
        ("albedo", albedo),
        ("u1", u1),
        ("u2", u2),
    ):
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(len(directions), values.item())
        elif values.shape != (len(directions),):
            raise ValueError(f"{name} must be a scalar or have shape (N,)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        batch_values.append(values)
    incident_power, albedo, u1, u2 = batch_values

    if np.any(incident_power < 0.0):
        raise ValueError("incident_power must be nonnegative")
    if np.any((albedo < 0.0) | (albedo > 1.0)):
        raise ValueError("albedo must be in [0, 1]")
    if np.any((u1 < 0.0) | (u1 >= 1.0)):
        raise ValueError("u1 must be in [0, 1)")
    if np.any((u2 < 0.0) | (u2 >= 1.0)):
        raise ValueError("u2 must be in [0, 1)")

    dots = np.sum(directions * normals, axis=1)
    oriented_normals = np.where((dots > 0.0)[:, None], -normals, normals)

    reference_axes = np.zeros_like(oriented_normals)
    use_z_axis = np.abs(oriented_normals[:, 2]) < 0.999
    reference_axes[use_z_axis, 2] = 1.0
    reference_axes[~use_z_axis, 0] = 1.0
    tangents = np.cross(reference_axes, oriented_normals)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    bitangents = np.cross(oriented_normals, tangents)

    # Malley's method: project a uniform unit-disk sample onto the hemisphere.
    radius = np.sqrt(u1)
    azimuth = 2.0 * np.pi * u2
    local_x = radius * np.cos(azimuth)
    local_y = radius * np.sin(azimuth)
    local_z = np.sqrt(1.0 - u1)
    reflected = (
        local_x[:, None] * tangents
        + local_y[:, None] * bitangents
        + local_z[:, None] * oriented_normals
    )
    reflected /= np.linalg.norm(reflected, axis=1)[:, None]

    result = np.empty(len(directions), dtype=_LAMBERTIAN_RESULT_DTYPE)
    result["reflected_direction"] = reflected
    result["reflected_power"] = incident_power * albedo
    result["absorbed_power"] = incident_power * (1.0 - albedo)
    return result


__all__ = ["interface_transport", "lambertian_reflection"]
