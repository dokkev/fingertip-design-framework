"""Deterministic camera-independent 3D optical transport."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from shapely import contains_xy

from model.fingertip import Fingertip
from optics.cross_section.result import _RawRaySegment
from optics.cross_section.settings import TraceSettings
from optics.cross_section.transport import _build_path_density_grid, _prepare_geometry
from optics.transport3d.geometry import (
    ExtrudedTransportGeometry,
    Transport3DGeometryError,
    build_transport_geometry,
)
from optics.transport3d.optix_backend import (
    OptixScene,
    Transport3DDependencyError,
    Transport3DTraceError,
    _Runtime,
)
from optics.transport3d.physics import (
    Transport3DPhysicsError,
    interface_split,
    periodic_plane_distance,
    wrapped_periodic_z,
)
from optics.transport3d.result import Transport3DResult
from optics.transport3d.sampling import sample_directions
from optics.transport3d.settings import Transport3DSettings


PLANAR_DIRECTION_TOLERANCE = 1.0e-6


def _enforce_planar_directions(
    cp: Any,
    directions: Any,
    *,
    valid: Any | None = None,
    context: str,
) -> Any:
    """Validate and reproject planar directions without permitting 3D drift."""
    if valid is None:
        valid = cp.ones(directions.shape[0], dtype=cp.bool_)
    if bool(cp.any(valid)):
        selected = directions[valid]
        max_abs_dz = float(cp.asnumpy(cp.max(cp.abs(selected[:, 2]))))
        if not np.isfinite(max_abs_dz) or max_abs_dz > PLANAR_DIRECTION_TOLERANCE:
            raise Transport3DPhysicsError(
                f"{context} introduced non-planar propagation: max |dz|={max_abs_dz:g}"
            )
        selected = selected.copy()
        selected[:, 2] = 0.0
        norms = cp.linalg.norm(selected[:, :2], axis=1)
        if bool(cp.any(~cp.isfinite(norms) | (norms <= 1.0e-12))):
            raise Transport3DPhysicsError(
                f"{context} produced a degenerate planar direction"
            )
        selected[:, :2] /= norms[:, None]
        result = directions.copy()
        result[valid] = selected
        return result
    return directions


def _xy_edges(
    geometry: ExtrudedTransportGeometry,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    if geometry.optical_domain is None:
        vertices = np.vstack(
            (
                geometry.silicone.vertices,
                geometry.rigid.vertices,
                geometry.envelope.vertices,
            )
        )
        min_x = float(np.min(vertices[:, 0]))
        min_y = float(np.min(vertices[:, 1]))
        max_x = float(np.max(vertices[:, 0]))
        max_y = float(np.max(vertices[:, 1]))
    else:
        domain = geometry.optical_domain
        min_x, min_y, max_x, max_y = domain.outer_envelope.bounds
    margin = 0.04 * max(max_x - min_x, max_y - min_y)
    x_edges = np.linspace(min_x - margin, max_x + margin, width + 1)
    y_edges = np.linspace(min_y - margin, max_y + margin, height + 1)
    return x_edges, y_edges


def _field_edges(
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
) -> tuple[np.ndarray, np.ndarray]:
    if settings.x_bounds_mm is not None and settings.y_bounds_mm is not None:
        return (
            np.linspace(*settings.x_bounds_mm, settings.projected_grid_width + 1),
            np.linspace(*settings.y_bounds_mm, settings.projected_grid_height + 1),
        )
    return _xy_edges(
        geometry,
        width=settings.projected_grid_width,
        height=settings.projected_grid_height,
    )


def _internal_field_edges(
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if settings.x_bounds_mm is not None and settings.y_bounds_mm is not None:
        x_edges = np.linspace(*settings.x_bounds_mm, settings.internal_grid_width + 1)
        y_edges = np.linspace(*settings.y_bounds_mm, settings.internal_grid_height + 1)
    else:
        x_edges, y_edges = _xy_edges(
            geometry,
            width=settings.internal_grid_width,
            height=settings.internal_grid_height,
        )
    z_edges = np.linspace(
        geometry.z_min_mm,
        geometry.z_max_mm,
        settings.internal_z_bins + 1,
    )
    return x_edges, y_edges, z_edges


def _concatenate(cp: Any, arrays: list[Any], *, dtype: Any, width: int | None = None) -> Any:
    if not arrays:
        shape = (0,) if width is None else (0, width)
        return cp.empty(shape, dtype=dtype)
    return cp.concatenate(arrays, axis=0).astype(dtype, copy=False)


def _surface_coordinates(
    cp: Any,
    surface: Any,
    primitive: Any,
    barycentrics: Any,
    vertices: Any,
    faces: Any,
) -> tuple[Any, Any, Any]:
    selected_faces = faces[primitive]
    first = vertices[selected_faces[:, 0]]
    second = vertices[selected_faces[:, 1]]
    third = vertices[selected_faces[:, 2]]
    bary_u = barycentrics[:, 0]
    bary_v = barycentrics[:, 1]
    points = (
        (1.0 - bary_u - bary_v)[:, None] * first
        + bary_u[:, None] * second
        + bary_v[:, None] * third
    )
    edge = second[:, :2] - first[:, :2]
    denominator = cp.sum(edge * edge, axis=1)
    along = cp.sum((points[:, :2] - first[:, :2]) * edge, axis=1) / cp.maximum(denominator, 1.0e-30)
    u = surface.u_start[primitive] + along * (surface.u_end[primitive] - surface.u_start[primitive])
    return points, cp.clip(u, 0.0, 1.0), points[:, 2]


def _projected_density(
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
    chunks: list[tuple[Any, Any, Any, Any, Any]],
    cp: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges, y_edges = _field_edges(geometry, settings)
    raw_segments: list[_RawRaySegment] = []
    for starts, ends, media, start_weights, end_weights in chunks:
        starts_np = cp.asnumpy(starts)
        ends_np = cp.asnumpy(ends)
        media_np = cp.asnumpy(media)
        start_np = cp.asnumpy(start_weights)
        end_np = cp.asnumpy(end_weights)
        for index in range(len(starts_np)):
            raw_segments.append(
                _RawRaySegment(
                    start_mm=(float(starts_np[index, 0]), float(starts_np[index, 1])),
                    end_mm=(float(ends_np[index, 0]), float(ends_np[index, 1])),
                    medium="silicone" if int(media_np[index]) else "air",
                    start_weight=float(start_np[index]),
                    end_weight=float(end_np[index]),
                    primary_ray_index=0,
                    interaction_index=0,
                )
            )
    prepared = _prepare_geometry(geometry.optical_domain)
    trace_settings = TraceSettings(
        ray_count=settings.ray_count,
        max_interactions=settings.max_interactions,
        minimum_ray_weight=settings.minimum_ray_weight,
        maximum_segment_count=settings.maximum_segment_count,
        grid_width=settings.projected_grid_width,
        grid_height=settings.projected_grid_height,
        source_epsilon_mm=settings.source_epsilon_mm,
        intersection_epsilon_mm=settings.intersection_epsilon_mm,
    )
    return _build_path_density_grid(
        geometry.optical_domain,
        prepared,
        trace_settings,
        tuple(raw_segments),
        x_bounds_mm=settings.x_bounds_mm,
        y_bounds_mm=settings.y_bounds_mm,
    )


def _accumulate_segment_path_3d(
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    optical_mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_weight: float,
    end_weight: float,
    *,
    maximum_spacing: float,
) -> None:
    """Accumulate one weighted straight path into a regular 3D grid.

    The stored value is weighted path length per voxel.  Midpoint sampling
    uses the same deterministic representative-weight convention as the
    reduced tracer; summing the z bins therefore performs the required
    discrete z integral without an extra width factor.
    """
    displacement = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(displacement))
    if length <= 0.0:
        return
    sample_count = max(1, int(np.ceil(length / maximum_spacing)))
    fractions = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    samples = np.asarray(start, dtype=float)[None, :] + fractions[:, None] * displacement[None, :]
    x_indices = np.searchsorted(x_edges, samples[:, 0], side="right") - 1
    y_indices = np.searchsorted(y_edges, samples[:, 1], side="right") - 1
    z_indices = np.searchsorted(z_edges, samples[:, 2], side="right") - 1
    valid = (
        (x_indices >= 0)
        & (x_indices < len(x_edges) - 1)
        & (y_indices >= 0)
        & (y_indices < len(y_edges) - 1)
        & (z_indices >= 0)
        & (z_indices < len(z_edges) - 1)
    )
    if not np.any(valid):
        return
    x_indices = x_indices[valid]
    y_indices = y_indices[valid]
    z_indices = z_indices[valid]
    inside = optical_mask[y_indices, x_indices]
    if not np.any(inside):
        return
    representative_weight = 0.5 * (float(start_weight) + float(end_weight))
    np.add.at(
        density,
        (z_indices[inside], y_indices[inside], x_indices[inside]),
        representative_weight * length / sample_count,
    )


def _new_internal_path_context(
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
 ) -> dict[str, Any]:
    x_edges, y_edges, z_edges = _internal_field_edges(geometry, settings)
    if geometry.optical_domain is None:
        # A true 3D surface artifact has no authoritative 2D projection to
        # use as an accessibility classifier.  The native P3 accumulator
        # therefore books every sampled transport segment; no collapsed 2D
        # geometry is invented for the FULL_3D path.
        optical_mask = np.ones(
            (settings.internal_grid_height, settings.internal_grid_width),
            dtype=bool,
        )
    else:
        prepared = _prepare_geometry(geometry.optical_domain)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        center_x, center_y = np.meshgrid(x_centers, y_centers)
        optical_mask = np.asarray(
            contains_xy(prepared.accessible_region, center_x, center_y),
            dtype=bool,
        )
    density = np.zeros(
        (settings.internal_z_bins, settings.internal_grid_height, settings.internal_grid_width),
        dtype=float,
    )
    maximum_spacing = 0.5 * min(
        float(x_edges[1] - x_edges[0]),
        float(y_edges[1] - y_edges[0]),
        float(z_edges[1] - z_edges[0]),
    )
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "optical_mask": optical_mask,
        "density": density,
        "maximum_spacing": maximum_spacing,
        "maximum_samples_per_segment": 32,
        "processed_segments": 0,
    }


def _accumulate_internal_chunk(
    context: dict[str, Any],
    chunk: tuple[Any, Any, Any, Any, Any],
    cp: Any,
) -> None:
    starts, ends, _media, start_weights, end_weights = chunk
    starts_np = cp.asnumpy(starts)
    ends_np = cp.asnumpy(ends)
    start_np = cp.asnumpy(start_weights)
    end_np = cp.asnumpy(end_weights)
    if not len(starts_np):
        return
    displacement = ends_np - starts_np
    lengths = np.linalg.norm(displacement, axis=1)
    counts = np.maximum(
        1,
        np.ceil(lengths / context["maximum_spacing"]).astype(np.int64),
    )
    counts = np.minimum(counts, context["maximum_samples_per_segment"])
    sample_count = int(np.max(counts))
    sample_indices = np.arange(sample_count, dtype=float)[None, :]
    fractions = (sample_indices + 0.5) / counts[:, None]
    samples = starts_np[:, None, :] + fractions[:, :, None] * displacement[:, None, :]
    x_indices = np.searchsorted(context["x_edges"], samples[:, :, 0], side="right") - 1
    y_indices = np.searchsorted(context["y_edges"], samples[:, :, 1], side="right") - 1
    z_indices = np.searchsorted(context["z_edges"], samples[:, :, 2], side="right") - 1
    valid = sample_indices < counts[:, None]
    valid &= (
        (x_indices >= 0)
        & (x_indices < len(context["x_edges"]) - 1)
        & (y_indices >= 0)
        & (y_indices < len(context["y_edges"]) - 1)
        & (z_indices >= 0)
        & (z_indices < len(context["z_edges"]) - 1)
    )
    if np.any(valid):
        valid_x = x_indices[valid]
        valid_y = y_indices[valid]
        valid_z = z_indices[valid]
        inside = context["optical_mask"][valid_y, valid_x]
        if np.any(inside):
            representative_weight = 0.5 * (start_np + end_np)
            represented = representative_weight * lengths / counts
            contributions = np.broadcast_to(
                represented[:, None],
                valid.shape,
            )
            np.add.at(
                context["density"],
                (valid_z[inside], valid_y[inside], valid_x[inside]),
                contributions[valid][inside],
            )
    context["processed_segments"] += len(starts_np)


def _internal_path_density(
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
    chunks: list[tuple[Any, Any, Any, Any, Any]],
    cp: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the raw P3 field and its z-integrated bridge field."""
    context = _new_internal_path_context(geometry, settings)
    for starts, ends, _media, start_weights, end_weights in chunks:
        _accumulate_internal_chunk(context, (starts, ends, _media, start_weights, end_weights), cp)
    return _finalize_internal_path_context(context, settings)


def _finalize_internal_path_context(
    context: dict[str, Any],
    settings: Transport3DSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_edges = context["x_edges"]
    y_edges = context["y_edges"]
    z_edges = context["z_edges"]
    density = context["density"]
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise Transport3DTraceError("3D internal path field is non-finite or negative")
    integrated = np.sum(density, axis=0)
    metadata = {
        "domain": {
            "x_min_mm": float(x_edges[0]),
            "x_max_mm": float(x_edges[-1]),
            "y_min_mm": float(y_edges[0]),
            "y_max_mm": float(y_edges[-1]),
            "z_min_mm": float(z_edges[0]),
            "z_max_mm": float(z_edges[-1]),
        },
        "resolution": {
            "x_bins": settings.internal_grid_width,
            "y_bins": settings.internal_grid_height,
            "z_bins": settings.internal_z_bins,
        },
        "normalization": "raw weighted path length per voxel; no TV normalization",
        "z_integration": "sum of raw z-bin path masses; no extra width factor",
        "segment_medium_scope": "air and silicone segments in the accessible optical domain, matching P2",
        "line_sampling": {
            "method": "deterministic segment midpoint sampling",
            "maximum_spacing_fraction_of_smallest_bin": 0.5,
            "maximum_samples_per_segment": context["maximum_samples_per_segment"],
        },
        "total_accumulated_weighted_path_length_mm": float(np.sum(density)),
        "processed_segment_count": context["processed_segments"],
    }
    return x_edges, y_edges, z_edges, density, integrated, metadata


def _trace_with_runtime(
    tip: Fingertip,
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
    runtime: _Runtime,
) -> Transport3DResult:
    cp = runtime.cp
    if settings.mode == "planar" and geometry.geometry_mode != "planar_extruded":
        raise Transport3DGeometryError(
            "PLANAR_2D requires the explicit 2D-to-z-wall geometry representation"
        )
    if settings.retain_projected_segments and geometry.optical_domain is None:
        raise Transport3DGeometryError(
            "projected path accumulation requires a validated 2D optical domain"
        )
    scene = OptixScene(runtime, geometry.silicone, geometry.rigid, geometry.envelope)
    # OptiX traverses float32 geometry.  Derive the self-hit offset from the
    # existing geometric epsilon and the float32 spacing at this cell scale.
    ray_offset = max(
        settings.intersection_epsilon_mm,
        8.0
        * float(np.finfo(np.float32).eps)
        * max(1.0, settings.extrusion_depth_mm),
    )
    device_arrays = {
        "silicone_vertices": cp.asarray(geometry.silicone.vertices),
        "silicone_faces": cp.asarray(geometry.silicone.faces),
        "silicone_normals": cp.asarray(geometry.silicone.normals),
        "silicone_external": cp.asarray(geometry.silicone.external_surface),
        "silicone_u_start": cp.asarray(geometry.silicone.u_start),
        "silicone_u_end": cp.asarray(geometry.silicone.u_end),
    }
    silicone_surface = type(
        "_DeviceSurface",
        (),
        {
            "normals": device_arrays["silicone_normals"],
            "external_surface": device_arrays["silicone_external"],
            "u_start": device_arrays["silicone_u_start"],
            "u_end": device_arrays["silicone_u_end"],
        },
    )()
    source = cp.asarray(geometry.source_position_mm, dtype=cp.float32)
    axis = cp.asarray([tip.emission_axis[0], tip.emission_axis[1], 0.0], dtype=cp.float32)
    directions_np = sample_directions(
        tip.led,
        tip.emission_axis,
        mode=settings.mode,
        ray_count=settings.ray_count,
    )
    directions = cp.asarray(directions_np, dtype=cp.float32)
    if settings.mode == "planar":
        directions = _enforce_planar_directions(
            cp,
            directions,
            context="deterministic planar source sampling",
        )
    origins = source[None, :] + settings.source_epsilon_mm * axis[None, :]
    origins = cp.repeat(origins, settings.ray_count, axis=0)
    launched_weight = float(tip.led.relative_radiant_power)
    primary_weight = launched_weight / settings.ray_count
    threshold = settings.minimum_ray_weight * launched_weight
    weights = cp.full(settings.ray_count, primary_weight, dtype=cp.float64)
    medium = cp.full(settings.ray_count, geometry.source_medium, dtype=cp.uint8)
    primary = cp.arange(settings.ray_count, dtype=cp.int64)
    interactions = cp.zeros(settings.ray_count, dtype=cp.int64)
    path_lengths = cp.zeros(settings.ray_count, dtype=cp.float64)
    wraps = cp.zeros(settings.ray_count, dtype=cp.int64)
    escaped_weight = 0.0
    absorbed_weight = 0.0
    terminated_weight = 0.0
    periodic_wrap_termination_count = 0
    periodic_wrap_termination_weight = 0.0
    no_event_termination_count = 0
    no_event_termination_weight = 0.0
    interface_normal_orientation_fallback_count = 0
    segment_count = 0
    internal_context = (
        _new_internal_path_context(geometry, settings)
        if settings.retain_internal_path_field
        else None
    )
    retain_segments = settings.retain_projected_segments
    segment_chunks: list[tuple[Any, Any, Any, Any, Any]] = []
    segment_metadata_chunks: list[tuple[Any, Any, Any]] = []
    escape_chunks: list[tuple[Any, ...]] = []
    surface_field = cp.zeros((settings.surface_z_bins, settings.surface_u_bins), dtype=cp.float64)
    surface_u_edges = cp.linspace(0.0, 1.0, settings.surface_u_bins + 1, dtype=cp.float64)
    surface_z_edges = cp.linspace(geometry.z_min_mm, geometry.z_max_mm, settings.surface_z_bins + 1, dtype=cp.float64)
    transport_started = time.perf_counter()
    while int(weights.size):
        # Primary rays are the configured launch set and must remain active
        # even when a high-resolution convergence count makes one primary
        # share smaller than the secondary-branch cutoff.  The cutoff itself
        # is unchanged for every reflected/transmitted branch and therefore
        # remains the reduced-tracer convention.
        # A configured cutoff is applied after the first interface event.
        # At high deterministic ray counts a primary share itself can be
        # below the cutoff; dropping its first transmitted/reflected child
        # would make the entire ray-count sweep artificially dark before a
        # physical escape or termination event can be booked.
        low = (weights < threshold) & (interactions > 1)
        if bool(cp.any(low)):
            terminated_weight += float(cp.asnumpy(cp.sum(weights[low])))
            keep = ~low
            origins, directions, weights, medium = origins[keep], directions[keep], weights[keep], medium[keep]
            primary, interactions, path_lengths, wraps = primary[keep], interactions[keep], path_lengths[keep], wraps[keep]
        if not int(weights.size):
            break
        maxed = interactions >= settings.max_interactions
        if bool(cp.any(maxed)):
            terminated_weight += float(cp.asnumpy(cp.sum(weights[maxed])))
            keep = ~maxed
            origins, directions, weights, medium = origins[keep], directions[keep], weights[keep], medium[keep]
            primary, interactions, path_lengths, wraps = primary[keep], interactions[keep], path_lengths[keep], wraps[keep]
        if not int(weights.size):
            break
        if segment_count + int(weights.size) > settings.maximum_segment_count:
            terminated_weight += float(cp.asnumpy(cp.sum(weights)))
            break
        segment_count += int(weights.size)

        silicone_hit = scene.trace("silicone", origins, directions, tmin=ray_offset)
        rigid_hit = scene.trace("rigid", origins, directions, tmin=ray_offset)
        envelope_hit = scene.trace("envelope", origins, directions, tmin=ray_offset)
        silicone_distance, silicone_primitive, silicone_bary, silicone_found = silicone_hit
        rigid_distance, rigid_primitive, _, rigid_found = rigid_hit
        envelope_distance, envelope_primitive, _, envelope_found = envelope_hit
        infinity = cp.asarray(cp.inf, dtype=cp.float32)
        silicone_distance = cp.where(silicone_found != 0, silicone_distance, infinity)
        rigid_distance = cp.where(rigid_found != 0, rigid_distance, infinity)
        envelope_distance = cp.where(envelope_found != 0, envelope_distance, infinity)

        dz = directions[:, 2]
        if settings.mode == "planar":
            directions = _enforce_planar_directions(
                cp,
                directions,
                context="planar propagation before intersection",
            )
            dz = directions[:, 2]
        periodic_distance = periodic_plane_distance(
            cp,
            origins[:, 2],
            dz,
            z_min_mm=geometry.z_min_mm,
            z_max_mm=geometry.z_max_mm,
            epsilon_mm=settings.intersection_epsilon_mm,
        )
        physical_min = cp.minimum(cp.minimum(silicone_distance, rigid_distance), envelope_distance)
        best_distance = physical_min
        event = cp.where(
            cp.isfinite(physical_min),
            cp.uint8(2),
            cp.uint8(3),
        )  # 0 silicone, 1 rigid, 2 envelope, 3 periodic
        rigid_selected = rigid_distance <= silicone_distance + settings.intersection_epsilon_mm
        rigid_selected &= rigid_distance <= envelope_distance + settings.intersection_epsilon_mm
        event = cp.where(rigid_selected & cp.isfinite(rigid_distance), 1, event)
        silicone_selected = silicone_distance < rigid_distance - settings.intersection_epsilon_mm
        silicone_selected &= silicone_distance <= envelope_distance + settings.intersection_epsilon_mm
        event = cp.where(silicone_selected & cp.isfinite(silicone_distance), 0, event)
        periodic_selected = periodic_distance < best_distance - settings.intersection_epsilon_mm
        event = cp.where(periodic_selected, 3, event)
        best_distance = cp.where(periodic_selected, periodic_distance, best_distance)
        no_event = ~cp.isfinite(best_distance)
        if bool(cp.any(no_event)):
            if not settings.terminate_on_no_event:
                raise Transport3DTraceError("an active branch has no physical hit or periodic event")
            no_event_weights = weights[no_event]
            no_event_termination_count += int(cp.asnumpy(cp.sum(no_event)))
            no_event_termination_weight += float(cp.asnumpy(cp.sum(no_event_weights)))
            terminated_weight += float(cp.asnumpy(cp.sum(no_event_weights)))
            keep = ~no_event
            origins, directions, weights, medium = origins[keep], directions[keep], weights[keep], medium[keep]
            primary, interactions, path_lengths, wraps = primary[keep], interactions[keep], path_lengths[keep], wraps[keep]
            best_distance = best_distance[keep]
            event = event[keep]
            silicone_primitive = silicone_primitive[keep]
            silicone_bary = silicone_bary[keep]
            if not int(weights.size):
                continue
        hit_positions = origins + best_distance[:, None] * directions
        end_weights = weights * cp.exp(
            cp.where(medium == 1, -tip.optical.absorption_per_mm * best_distance, 0.0)
        )
        removed = weights - end_weights
        if bool(cp.any(removed < -1.0e-12)):
            raise Transport3DPhysicsError("segment attenuation increased branch weight")
        absorbed_weight += float(cp.asnumpy(cp.sum(removed)))
        if retain_segments:
            segment_chunks.append((origins.copy(), hit_positions.copy(), medium.copy(), weights.copy(), end_weights.copy()))
        if settings.retain_projected_segments:
            segment_metadata_chunks.append(
                (
                    cp.linalg.norm(hit_positions - origins, axis=1).copy(),
                    primary.copy(),
                    interactions.copy(),
                )
            )
        if internal_context is not None:
            _accumulate_internal_chunk(
                internal_context,
                (origins, hit_positions, medium, weights, end_weights),
                cp,
            )
        new_states: list[tuple[Any, Any, Any, Any, Any, Any, Any, Any]] = []

        periodic = event == 3
        if bool(cp.any(periodic)):
            if settings.mode == "planar":
                raise Transport3DPhysicsError(
                    "PLANAR_2D reached a longitudinal periodic boundary"
                )
            periodic_weight = end_weights[periodic]
            periodic_wrapped = wraps[periodic] + 1
            pathological = periodic_wrapped > settings.maximum_periodic_wraps
            if bool(cp.any(pathological)):
                if not settings.terminate_on_periodic_wrap_limit:
                    raise Transport3DTraceError(
                        "periodic-wrap limit reached unexpectedly for an active branch"
                    )
                pathological_weight = periodic_weight[pathological]
                periodic_wrap_termination_count += int(cp.asnumpy(cp.sum(pathological)))
                periodic_wrap_termination_weight += float(cp.asnumpy(cp.sum(pathological_weight)))
                terminated_weight += float(cp.asnumpy(cp.sum(pathological_weight)))
            keep_periodic = ~pathological
            if bool(cp.any(keep_periodic)):
                selected_origins = hit_positions[periodic][keep_periodic]
                selected_directions = directions[periodic][keep_periodic]
                selected_origins = selected_origins.copy()
                selected_origins[:, 2] = wrapped_periodic_z(
                    cp,
                    selected_directions[:, 2],
                    z_min_mm=geometry.z_min_mm,
                    z_max_mm=geometry.z_max_mm,
                    offset_mm=ray_offset,
                )
                new_states.append((selected_origins, selected_directions, periodic_weight[keep_periodic], medium[periodic][keep_periodic], primary[periodic][keep_periodic], interactions[periodic][keep_periodic], path_lengths[periodic][keep_periodic] + best_distance[periodic][keep_periodic], periodic_wrapped[keep_periodic]))

        physical = ~periodic
        if bool(cp.any(physical)):
            physical_positions = hit_positions[physical]
            physical_end_weights = end_weights[physical]
            physical_medium = medium[physical]
            physical_primary = primary[physical]
            physical_interactions = interactions[physical]
            physical_paths = path_lengths[physical] + best_distance[physical]
            physical_directions = directions[physical]
            physical_event = event[physical]
            physical_silicone_primitive = silicone_primitive[physical]
            physical_silicone_bary = silicone_bary[physical]
            rigid = physical_event == 1
            if bool(cp.any(rigid)):
                terminated_weight += float(cp.asnumpy(cp.sum(physical_end_weights[rigid])))
            envelope = physical_event == 2
            if bool(cp.any(envelope)):
                envelope_weight = physical_end_weights[envelope]
                escaped_weight += float(cp.asnumpy(cp.sum(envelope_weight)))
            interface = physical_event == 0
            if bool(cp.any(interface)):
                indices = cp.where(interface)[0]
                interface_positions = physical_positions[indices]
                interface_directions = physical_directions[indices]
                interface_end_weights = physical_end_weights[indices]
                interface_medium = physical_medium[indices]
                interface_primary = physical_primary[indices]
                interface_interactions = physical_interactions[indices]
                interface_paths = physical_paths[indices]
                interface_primitive = physical_silicone_primitive[indices]
                interface_bary = physical_silicone_bary[indices]
                outward = device_arrays["silicone_normals"][interface_primitive]
                medium_normal = cp.where(
                    (interface_medium == 1)[:, None], outward, -outward
                )
                medium_alignment = cp.sum(
                    interface_directions * medium_normal, axis=1
                )
                orientation_fallback = medium_alignment <= 1.0e-7
                if bool(cp.any(orientation_fallback)):
                    interface_normal_orientation_fallback_count += int(
                        cp.asnumpy(cp.sum(orientation_fallback))
                    )
                # The triangle winding normally determines this sign from the
                # active medium.  A branch can nevertheless approach a
                # shared extruded boundary from the opposite side.  The
                # interface contract requires the normal to point into the
                # transmitted medium, so orient the fallback from the
                # incident direction instead of aborting a valid branch.
                interface_normal = cp.where(
                    orientation_fallback[:, None], -medium_normal, medium_normal
                )
                alignment = cp.sum(interface_directions * interface_normal, axis=1)
                if bool(cp.any(alignment <= 1.0e-7)):
                    bad = int(cp.asnumpy(cp.where(alignment <= 1.0e-7)[0][0]))
                    raise Transport3DPhysicsError(
                        "invalid interface orientation at primitive "
                        f"{int(cp.asnumpy(interface_primitive[bad]))}, "
                        f"position={cp.asnumpy(interface_positions[bad])}, "
                        f"direction={cp.asnumpy(interface_directions[bad])}, "
                        f"normal={cp.asnumpy(interface_normal[bad])}, "
                        f"medium={int(cp.asnumpy(interface_medium[bad]))}, "
                        f"primary={int(cp.asnumpy(interface_primary[bad]))}, "
                        f"interaction={int(cp.asnumpy(interface_interactions[bad]))}"
                    )
                reflected, transmitted, reflectance, tir = interface_split(
                    cp,
                    interface_directions,
                    interface_normal,
                    interface_medium,
                    tip.optical.refractive_index_air,
                    tip.optical.refractive_index_silicone,
                )
                if settings.mode == "planar":
                    reflected = _enforce_planar_directions(
                        cp,
                        reflected,
                        context="planar reflected branch",
                    )
                    transmitted = _enforce_planar_directions(
                        cp,
                        transmitted,
                        valid=~tir,
                        context="planar transmitted branch",
                    )
                reflected_weight = interface_end_weights * reflectance
                transmitted_weight = interface_end_weights * (1.0 - reflectance)
                next_interaction = interface_interactions + 1
                first_generation = next_interaction <= 1
                reflected_keep = (reflected_weight >= threshold) | first_generation
                reflected_terminate = ~reflected_keep
                if bool(cp.any(reflected_terminate)):
                    terminated_weight += float(cp.asnumpy(cp.sum(reflected_weight[reflected_terminate])))
                if bool(cp.any(reflected_keep)):
                    new_states.append((
                        interface_positions[reflected_keep] + ray_offset * reflected[reflected_keep],
                        reflected[reflected_keep],
                        reflected_weight[reflected_keep],
                        interface_medium[reflected_keep],
                        interface_primary[reflected_keep],
                        next_interaction[reflected_keep],
                        interface_paths[reflected_keep],
                        wraps[physical][indices][reflected_keep],
                    ))
                interface_points, interface_u, interface_z = _surface_coordinates(
                    cp,
                    silicone_surface,
                    interface_primitive,
                    interface_bary,
                    device_arrays["silicone_vertices"],
                    device_arrays["silicone_faces"],
                )
                external_escape = (interface_medium == 1) & silicone_surface.external_surface[interface_primitive]
                ordinary_transmission = ~external_escape
                if bool(cp.any(external_escape)):
                    outgoing_weight = transmitted_weight[external_escape] * (~tir[external_escape])
                    escaped_weight += float(cp.asnumpy(cp.sum(outgoing_weight)))
                    positive_outgoing = outgoing_weight > 0.0
                    if bool(cp.any(positive_outgoing)):
                        escape_chunks.append((
                            interface_points[external_escape][positive_outgoing],
                            transmitted[external_escape][positive_outgoing],
                            outgoing_weight[positive_outgoing],
                            interface_primary[external_escape][positive_outgoing],
                            interface_paths[external_escape][positive_outgoing],
                            next_interaction[external_escape][positive_outgoing],
                            outward[external_escape][positive_outgoing],
                            interface_u[external_escape][positive_outgoing],
                            interface_z[external_escape][positive_outgoing],
                            interface_primitive[external_escape][positive_outgoing],
                        ))
                    u_indices = cp.searchsorted(surface_u_edges, interface_u[external_escape], side="right") - 1
                    z_indices = cp.searchsorted(surface_z_edges, interface_z[external_escape], side="right") - 1
                    valid = (u_indices >= 0) & (u_indices < settings.surface_u_bins) & (z_indices >= 0) & (z_indices < settings.surface_z_bins)
                    if bool(cp.any(valid)):
                        cp.add.at(surface_field, (z_indices[valid], u_indices[valid]), outgoing_weight[valid])
                ordinary_transmission &= transmitted_weight > 0.0
                transmission_keep = ordinary_transmission & (
                    (transmitted_weight >= threshold) | first_generation
                ) & (~tir)
                transmission_terminate = ordinary_transmission & ~transmission_keep
                if bool(cp.any(transmission_terminate)):
                    terminated_weight += float(cp.asnumpy(cp.sum(transmitted_weight[transmission_terminate])))
                if bool(cp.any(transmission_keep)):
                    new_states.append((
                        interface_positions[transmission_keep] + ray_offset * transmitted[transmission_keep],
                        transmitted[transmission_keep],
                        transmitted_weight[transmission_keep],
                        cp.where(interface_medium[transmission_keep] == 0, 1, 0).astype(cp.uint8),
                        interface_primary[transmission_keep],
                        next_interaction[transmission_keep],
                        interface_paths[transmission_keep],
                        wraps[physical][indices][transmission_keep],
                    ))

        if new_states:
            origins = _concatenate(cp, [state[0] for state in new_states], dtype=cp.float32, width=3)
            directions = _concatenate(cp, [state[1] for state in new_states], dtype=cp.float32, width=3)
            if settings.mode == "planar":
                directions = _enforce_planar_directions(
                    cp,
                    directions,
                    context="planar branch state update",
                )
            weights = _concatenate(cp, [state[2] for state in new_states], dtype=cp.float64)
            medium = _concatenate(cp, [state[3] for state in new_states], dtype=cp.uint8)
            primary = _concatenate(cp, [state[4] for state in new_states], dtype=cp.int64)
            interactions = _concatenate(cp, [state[5] for state in new_states], dtype=cp.int64)
            path_lengths = _concatenate(cp, [state[6] for state in new_states], dtype=cp.float64)
            wraps = _concatenate(cp, [state[7] for state in new_states], dtype=cp.int64)
        else:
            weights = cp.empty(0, dtype=cp.float64)

    postprocessing_started = time.perf_counter()
    escaped_positions = _concatenate(cp, [chunk[0] for chunk in escape_chunks], dtype=cp.float32, width=3)
    escaped_directions = _concatenate(cp, [chunk[1] for chunk in escape_chunks], dtype=cp.float32, width=3)
    escaped_weights = _concatenate(cp, [chunk[2] for chunk in escape_chunks], dtype=cp.float64)
    escaped_primary = _concatenate(cp, [chunk[3] for chunk in escape_chunks], dtype=cp.int64)
    escaped_paths = _concatenate(cp, [chunk[4] for chunk in escape_chunks], dtype=cp.float64)
    escaped_interactions = _concatenate(cp, [chunk[5] for chunk in escape_chunks], dtype=cp.int64)
    escaped_normals = _concatenate(cp, [chunk[6] for chunk in escape_chunks], dtype=cp.float32, width=3)
    escaped_u = _concatenate(cp, [chunk[7] for chunk in escape_chunks], dtype=cp.float64)
    escaped_z = _concatenate(cp, [chunk[8] for chunk in escape_chunks], dtype=cp.float64)
    escaped_primitives = _concatenate(cp, [chunk[9] for chunk in escape_chunks], dtype=cp.int64)
    if geometry.silicone.semantic_tags is None:
        raise Transport3DGeometryError(
            "silicone surface is missing semantic boundary tags"
        )
    escaped_tags = tuple(
        geometry.silicone.semantic_tags[int(primitive)]
        for primitive in cp.asnumpy(escaped_primitives)
    )
    planar_direction_z_max = 0.0
    if settings.mode == "planar" and int(escaped_directions.size):
        planar_direction_z_max = float(
            cp.asnumpy(cp.max(cp.abs(escaped_directions[:, 2])))
        )
        if planar_direction_z_max > PLANAR_DIRECTION_TOLERANCE:
            raise Transport3DPhysicsError(
                "PLANAR_2D escape history contains a nonzero longitudinal direction"
            )
    projected = (None, None, None)
    if settings.retain_projected_segments:
        projected = _projected_density(geometry, settings, segment_chunks, cp)[0:3]
    internal = (None, None, None, None, None, None)
    if internal_context is not None:
        internal = _finalize_internal_path_context(internal_context, settings)
    retained_segments = (None, None, None)
    if settings.retain_projected_segments:
        retained_segments = (
            _concatenate(cp, [chunk[0] for chunk in segment_metadata_chunks], dtype=cp.float64),
            _concatenate(cp, [chunk[1] for chunk in segment_metadata_chunks], dtype=cp.int64),
            _concatenate(cp, [chunk[2] for chunk in segment_metadata_chunks], dtype=cp.int64),
        )
    geometry_metadata = dict(geometry.metadata)
    geometry_metadata["branch_cutoff"] = {
        "minimum_ray_weight_fraction": settings.minimum_ray_weight,
        "absolute_weight_threshold": threshold,
        "maximum_interactions": settings.max_interactions,
        "primary_and_first_generation_exempt": True,
        "convention": "cutoff applies to branches with interaction_count > 1",
    }
    geometry_metadata["processed_segment_count"] = segment_count
    geometry_metadata["periodic_wrap_termination"] = {
        "enabled": settings.terminate_on_periodic_wrap_limit,
        "count": periodic_wrap_termination_count,
        "weight": periodic_wrap_termination_weight,
        "maximum_periodic_wraps": settings.maximum_periodic_wraps,
    }
    geometry_metadata["no_event_termination"] = {
        "enabled": settings.terminate_on_no_event,
        "count": no_event_termination_count,
        "weight": no_event_termination_weight,
    }
    geometry_metadata["interface_normal_orientation_fallback_count"] = (
        interface_normal_orientation_fallback_count
    )
    geometry_metadata["planar_direction_invariant"] = {
        "required": settings.mode == "planar",
        "max_abs_direction_z": planar_direction_z_max,
        "tolerance": PLANAR_DIRECTION_TOLERANCE,
        "passed": settings.mode != "planar"
        or planar_direction_z_max <= PLANAR_DIRECTION_TOLERANCE,
    }
    geometry_metadata["retained_segment_count"] = len(segment_chunks) and int(
        sum(len(chunk[0]) for chunk in segment_chunks)
    ) or 0
    geometry_metadata["transport_material"] = {
        "refractive_index_air": float(tip.optical.refractive_index_air),
        "refractive_index_silicone": float(tip.optical.refractive_index_silicone),
        "absorption_per_mm": float(tip.optical.absorption_per_mm),
        "scattering_per_mm": float(tip.optical.scattering_per_mm),
    }
    if internal[5] is not None:
        internal_metadata = dict(internal[5])
        internal_metadata.update(
            {
                "source_position_mm": list(geometry.source_position_mm),
                "source_mode": settings.mode,
                "ray_count": settings.ray_count,
                "branch_cutoff": dict(geometry_metadata["branch_cutoff"]),
                "material": dict(geometry_metadata["transport_material"]),
            }
        )
        geometry_metadata["internal_path_field"] = internal_metadata
    escaped_weight += 0.0
    energy_balance_error = abs(launched_weight - escaped_weight - absorbed_weight - terminated_weight) / max(launched_weight, 1.0e-30)
    if not np.isfinite(energy_balance_error) or energy_balance_error > settings.energy_balance_tolerance:
        raise Transport3DPhysicsError(
            f"energy balance error {energy_balance_error:g} exceeds tolerance {settings.energy_balance_tolerance:g}"
        )
    return Transport3DResult(
        source_position_mm=geometry.source_position_mm,
        source_mode=settings.mode,
        extrusion_depth_mm=geometry.depth_mm,
        launched_ray_count=settings.ray_count,
        launched_weight=launched_weight,
        escaped_weight=escaped_weight,
        absorbed_weight=absorbed_weight,
        terminated_weight=terminated_weight,
        outgoing_surface_weight=float(cp.asnumpy(cp.sum(escaped_weights))),
        surface_u_edges=cp.asnumpy(surface_u_edges),
        surface_z_edges=cp.asnumpy(surface_z_edges),
        outgoing_surface_field=cp.asnumpy(surface_field),
        escape_positions_mm=cp.asnumpy(escaped_positions),
        escape_directions=cp.asnumpy(escaped_directions),
        escape_surface_normals=cp.asnumpy(escaped_normals),
        escape_surface_u=cp.asnumpy(escaped_u),
        escape_surface_z=cp.asnumpy(escaped_z),
        escape_surface_tags=escaped_tags,
        escape_surface_primitive_indices=cp.asnumpy(escaped_primitives),
        escape_weights=cp.asnumpy(escaped_weights),
        escape_primary_ray_indices=cp.asnumpy(escaped_primary),
        escape_path_lengths_mm=cp.asnumpy(escaped_paths),
        escape_interaction_counts=cp.asnumpy(escaped_interactions),
        energy_balance_error=float(energy_balance_error),
        energy_balance_tolerance=settings.energy_balance_tolerance,
        projected_x_edges_mm=None if projected[0] is None else projected[0],
        projected_y_edges_mm=None if projected[1] is None else projected[1],
        projected_weighted_path_density=None if projected[2] is None else projected[2],
        internal_path_x_edges_mm=None if internal[0] is None else internal[0],
        internal_path_y_edges_mm=None if internal[1] is None else internal[1],
        internal_path_z_edges_mm=None if internal[2] is None else internal[2],
        internal_weighted_path_density_3d=None if internal[3] is None else internal[3],
        internal_z_integrated_path_density=None if internal[4] is None else internal[4],
        retained_segment_lengths_mm=None if retained_segments[0] is None else cp.asnumpy(retained_segments[0]),
        retained_segment_primary_ray_indices=None if retained_segments[1] is None else cp.asnumpy(retained_segments[1]),
        retained_segment_interaction_counts=None if retained_segments[2] is None else cp.asnumpy(retained_segments[2]),
        geometry_metadata=geometry_metadata,
        timings_seconds={
            "gas_build": scene.gas_build_seconds,
            "transport": postprocessing_started - transport_started,
            "postprocessing": time.perf_counter() - postprocessing_started,
        },
    )


def trace_3d(
    tip: Fingertip,
    mesh: Any,
    *,
    reference_mesh: Any | None = None,
    settings: Transport3DSettings | None = None,
    runtime: Any | None = None,
) -> Transport3DResult:
    """Trace one neutral fingertip state through an OptiX periodic cell.

    ``mesh`` may be the reference ``FingertipMesh`` or a neutral deformed
    ``PadMesh``.  A deformed pad must be paired with the original full mesh so
    the fixed rigid carrier is taken from its existing neutral topology.
    """
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    trace_settings = settings or Transport3DSettings()
    if hasattr(mesh, "pad_elements") and hasattr(mesh, "carrier_elements"):
        full_mesh = mesh
        pad_mesh = mesh.pad
    else:
        pad_mesh = mesh
        full_mesh = reference_mesh
    if full_mesh is None:
        raise Transport3DGeometryError(
            "a deformed neutral pad requires reference_mesh for fixed carrier geometry"
        )
    geometry = build_transport_geometry(
        tip,
        pad_mesh,
        full_mesh,
        depth_mm=trace_settings.extrusion_depth_mm,
        source_epsilon_mm=trace_settings.source_epsilon_mm,
    )
    return trace_geometry(tip, geometry, settings=trace_settings, runtime=runtime)


def trace_geometry(
    tip: Fingertip,
    geometry: ExtrudedTransportGeometry,
    *,
    settings: Transport3DSettings | None = None,
    runtime: Any | None = None,
) -> Transport3DResult:
    """Trace a prevalidated neutral geometry through the shared OptiX core."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    if not isinstance(geometry, ExtrudedTransportGeometry):
        raise TypeError("geometry must be an ExtrudedTransportGeometry")
    trace_settings = settings or Transport3DSettings()
    actual_runtime = runtime or _Runtime.create()
    if not isinstance(actual_runtime, _Runtime):
        raise TypeError("runtime must be an internal OptiX runtime")
    return _trace_with_runtime(tip, geometry, trace_settings, actual_runtime)


__all__ = [
    "Transport3DDependencyError",
    "Transport3DGeometryError",
    "Transport3DPhysicsError",
    "Transport3DResult",
    "Transport3DSettings",
    "Transport3DTraceError",
    "trace_geometry",
    "trace_3d",
]
