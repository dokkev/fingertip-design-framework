"""Bounded host-side optical path orchestration."""

from __future__ import annotations

from math import isfinite

import numpy as np

from .scene import OptixScene, safe_secondary_origins
from .transport import interface_transport, lambertian_reflection


_ESCAPED_DTYPE = np.dtype(
    [
        ("ray_id", np.int64),
        ("bounce", np.int64),
        ("origin_W_m", np.float64, (3,)),
        ("direction_W", np.float64, (3,)),
        ("power", np.float64),
    ]
)


def _sample_dielectric_branches(
    optical: np.ndarray,
    branch_u: np.ndarray,
    inside_silicone: np.ndarray,
    power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one Fresnel branch without applying its probability twice."""
    branch_u = np.asarray(branch_u, dtype=np.float64)
    inside_silicone = np.asarray(inside_silicone, dtype=np.bool_)
    power = np.asarray(power, dtype=np.float64)
    expected_shape = optical.shape
    if (
        branch_u.shape != expected_shape
        or inside_silicone.shape != expected_shape
        or power.shape != expected_shape
    ):
        raise ValueError("dielectric branch inputs must have equal shapes")
    if not np.all(np.isfinite(branch_u)) or np.any(
        (branch_u < 0.0) | (branch_u >= 1.0)
    ):
        raise ValueError("branch_u must be finite and in [0, 1)")

    reflect = optical["total_internal_reflection"] | (
        branch_u < optical["reflectance"]
    )
    directions = np.where(
        reflect[:, None],
        optical["reflected_direction"],
        optical["refracted_direction"],
    )
    next_inside = inside_silicone.copy()
    next_inside[~reflect] = ~next_inside[~reflect]
    return directions, next_inside, power.copy()


def trace_bounded_paths(
    scene: OptixScene,
    origins_W_m: np.ndarray,
    directions_W: np.ndarray,
    power: np.ndarray,
    *,
    inside_silicone: bool | np.ndarray,
    n_air: float,
    n_silicone: float,
    extinction_coefficient_m_inv: float,
    carrier_albedo: float,
    max_bounces: int,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
    silicone_instance_id: int,
    carrier_instance_id: int,
    mask: int = 0xFF,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Trace one sampled optical path per input ray for a bounded depth.

    Dielectric branches are selected with their Fresnel probabilities, so a
    selected lossless branch retains its current path power. Silicone segments
    lose ballistic-path power through Beer-Lambert attenuation, and carrier
    events lose the fraction complementary to the carrier albedo.
    """
    origins = np.asarray(origins_W_m, dtype=np.float64)
    directions = np.asarray(directions_W, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1:] != (3,):
        raise ValueError("origins_W_m must have shape (N, 3)")
    if directions.shape != origins.shape:
        raise ValueError("directions_W must have shape (N, 3)")
    ray_count = len(origins)
    if not ray_count:
        raise ValueError("at least one path is required")
    if power.shape != (ray_count,):
        raise ValueError("power must have shape (N,)")
    if not (
        np.all(np.isfinite(origins))
        and np.all(np.isfinite(directions))
        and np.all(np.isfinite(power))
    ):
        raise ValueError("path inputs must be finite")
    if np.any(power < 0.0):
        raise ValueError("power must be nonnegative")
    direction_norms = np.linalg.norm(directions, axis=1)
    if np.any(direction_norms <= np.finfo(np.float64).tiny):
        raise ValueError("directions_W must contain nonzero vectors")
    directions = directions / direction_norms[:, None]

    if isinstance(max_bounces, bool) or not isinstance(max_bounces, int):
        raise ValueError("max_bounces must be a positive integer")
    if max_bounces <= 0:
        raise ValueError("max_bounces must be a positive integer")
    if not isfinite(n_air) or n_air <= 0.0:
        raise ValueError("n_air must be finite and positive")
    if not isfinite(n_silicone) or n_silicone <= 0.0:
        raise ValueError("n_silicone must be finite and positive")
    if (
        not isfinite(extinction_coefficient_m_inv)
        or extinction_coefficient_m_inv < 0.0
    ):
        raise ValueError(
            "extinction_coefficient_m_inv must be finite and nonnegative"
        )
    if not isfinite(carrier_albedo) or not 0.0 <= carrier_albedo <= 1.0:
        raise ValueError("carrier_albedo must be finite and in [0, 1]")
    if silicone_instance_id == carrier_instance_id:
        raise ValueError("silicone and carrier instance IDs must differ")

    inside = np.asarray(inside_silicone, dtype=np.bool_)
    if inside.ndim == 0:
        inside = np.full(ray_count, inside.item(), dtype=np.bool_)
    elif inside.shape != (ray_count,):
        raise ValueError("inside_silicone must be a bool or have shape (N,)")
    else:
        inside = inside.copy()

    sample_arrays = []
    for name, value in (
        ("dielectric_branch_u", dielectric_branch_u),
        ("carrier_u1", carrier_u1),
        ("carrier_u2", carrier_u2),
    ):
        samples = np.asarray(value, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != ray_count:
            raise ValueError(f"{name} must have shape (B, N)")
        if samples.shape[0] < max_bounces:
            raise ValueError(f"{name} has fewer than max_bounces rows")
        if not np.all(np.isfinite(samples)) or np.any(
            (samples < 0.0) | (samples >= 1.0)
        ):
            raise ValueError(f"{name} must be finite and in [0, 1)")
        sample_arrays.append(samples)
    dielectric_branch_u, carrier_u1, carrier_u2 = sample_arrays

    emitted_power = float(power.sum())
    power = power.copy()
    ray_id = np.arange(ray_count, dtype=np.int64)
    escaped_batches: list[np.ndarray] = []
    absorbed_power = 0.0
    bulk_loss_power = 0.0
    unresolved_internal_miss_power = 0.0

    for bounce in range(max_bounces):
        hits = scene.trace_closest(origins, directions, mask=mask)
        silicone_segment = inside & hits["hit"]
        if np.any(silicone_segment):
            segment_length_m = np.asarray(
                hits["t"][silicone_segment],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(segment_length_m)) or np.any(
                segment_length_m < 0.0
            ):
                raise RuntimeError(
                    "OptiX returned an invalid silicone segment length"
                )
            power_before = power[silicone_segment].copy()
            power[silicone_segment] *= np.exp(
                -extinction_coefficient_m_inv * segment_length_m
            )
            bulk_loss_power += float(
                (power_before - power[silicone_segment]).sum()
            )

        miss = ~hits["hit"]
        external_miss = miss & ~inside
        if np.any(external_miss):
            escaped = np.empty(np.count_nonzero(external_miss), dtype=_ESCAPED_DTYPE)
            escaped["ray_id"] = ray_id[external_miss]
            escaped["bounce"] = bounce
            escaped["origin_W_m"] = origins[external_miss]
            escaped["direction_W"] = directions[external_miss]
            escaped["power"] = power[external_miss]
            escaped_batches.append(escaped)
        unresolved_internal_miss_power += float(power[miss & inside].sum())

        silicone_hit = hits["hit"] & (
            hits["instance_id"] == silicone_instance_id
        )
        carrier_hit = hits["hit"] & (
            hits["instance_id"] == carrier_instance_id
        )
        unknown_hit = hits["hit"] & ~(silicone_hit | carrier_hit)
        if np.any(unknown_hit):
            unknown_ids = np.unique(hits["instance_id"][unknown_hit])
            raise RuntimeError(f"unhandled OptiX instance IDs: {unknown_ids.tolist()}")

        next_origins: list[np.ndarray] = []
        next_directions: list[np.ndarray] = []
        next_power: list[np.ndarray] = []
        next_inside: list[np.ndarray] = []
        next_ray_id: list[np.ndarray] = []

        if np.any(silicone_hit):
            hit_indices = np.flatnonzero(silicone_hit)
            silicone_hits = hits[silicone_hit]
            silicone_inside = inside[silicone_hit]
            silicone_directions = np.empty((len(hit_indices), 3), dtype=np.float64)
            silicone_next_inside = silicone_inside.copy()
            silicone_power = power[silicone_hit].copy()

            for currently_inside in (False, True):
                medium_group = silicone_inside == currently_inside
                if not np.any(medium_group):
                    continue
                optical = interface_transport(
                    directions[silicone_hit][medium_group],
                    silicone_hits["normal_W"][medium_group],
                    n_incident=n_silicone if currently_inside else n_air,
                    n_transmitted=n_air if currently_inside else n_silicone,
                    incident_power=silicone_power[medium_group],
                )
                selected = _sample_dielectric_branches(
                    optical,
                    dielectric_branch_u[bounce, ray_id[silicone_hit][medium_group]],
                    silicone_inside[medium_group],
                    silicone_power[medium_group],
                )
                silicone_directions[medium_group] = selected[0]
                silicone_next_inside[medium_group] = selected[1]
                silicone_power[medium_group] = selected[2]

            next_origins.append(
                safe_secondary_origins(silicone_hits, silicone_directions)
            )
            next_directions.append(silicone_directions)
            next_power.append(silicone_power)
            next_inside.append(silicone_next_inside)
            next_ray_id.append(ray_id[silicone_hit])

        if np.any(carrier_hit):
            carrier_hits = hits[carrier_hit]
            carrier_ray_ids = ray_id[carrier_hit]
            reflected = lambertian_reflection(
                directions[carrier_hit],
                carrier_hits["normal_W"],
                incident_power=power[carrier_hit],
                albedo=carrier_albedo,
                u1=carrier_u1[bounce, carrier_ray_ids],
                u2=carrier_u2[bounce, carrier_ray_ids],
            )
            absorbed_power += float(reflected["absorbed_power"].sum())
            next_origins.append(
                safe_secondary_origins(
                    carrier_hits,
                    reflected["reflected_direction"],
                )
            )
            next_directions.append(reflected["reflected_direction"])
            next_power.append(reflected["reflected_power"])
            next_inside.append(inside[carrier_hit])
            next_ray_id.append(carrier_ray_ids)

        if not next_origins:
            origins = np.empty((0, 3), dtype=np.float64)
            directions = np.empty((0, 3), dtype=np.float64)
            power = np.empty(0, dtype=np.float64)
            inside = np.empty(0, dtype=np.bool_)
            ray_id = np.empty(0, dtype=np.int64)
            break
        origins = np.concatenate(next_origins)
        directions = np.concatenate(next_directions)
        power = np.concatenate(next_power)
        inside = np.concatenate(next_inside)
        ray_id = np.concatenate(next_ray_id)

    escaped = (
        np.concatenate(escaped_batches)
        if escaped_batches
        else np.empty(0, dtype=_ESCAPED_DTYPE)
    )
    escaped_power = float(escaped["power"].sum())
    remaining_power = float(power.sum())
    accounted_power = (
        escaped_power
        + absorbed_power
        + bulk_loss_power
        + unresolved_internal_miss_power
        + remaining_power
    )
    closure_error = accounted_power - emitted_power
    closure_tolerance = 1.0e-12 * max(1.0, emitted_power)
    if not np.isfinite(accounted_power) or abs(closure_error) > closure_tolerance:
        raise RuntimeError("bounded optical paths do not conserve modeled power")

    statistics: dict[str, float | int] = {
        "emitted_power": emitted_power,
        "escaped_power": escaped_power,
        "absorbed_power": absorbed_power,
        "bulk_loss_power": bulk_loss_power,
        "unresolved_internal_miss_power": unresolved_internal_miss_power,
        "remaining_power": remaining_power,
        "accounted_power": accounted_power,
        "closure_error": closure_error,
        "escaped_ray_count": len(escaped),
        "remaining_ray_count": len(power),
    }
    return escaped, statistics


__all__ = ["trace_bounded_paths"]
