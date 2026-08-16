"""Deterministic camera-independent 3D optical transport."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

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


def _field_edges(geometry: ExtrudedTransportGeometry, settings: Transport3DSettings) -> tuple[np.ndarray, np.ndarray]:
    domain = geometry.optical_domain
    min_x, min_y, max_x, max_y = domain.outer_envelope.bounds
    margin = 0.04 * max(max_x - min_x, max_y - min_y)
    x_edges = np.linspace(min_x - margin, max_x + margin, 241)
    y_edges = np.linspace(min_y - margin, max_y + margin, 241)
    return x_edges, y_edges


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
    )


def _trace_with_runtime(
    tip: Fingertip,
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
    runtime: _Runtime,
) -> Transport3DResult:
    cp = runtime.cp
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
    segment_count = 0
    segment_chunks: list[tuple[Any, Any, Any, Any, Any]] = []
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
        if bool(cp.any(~cp.isfinite(best_distance))):
            raise Transport3DTraceError("an active branch has no physical hit or periodic event")
        hit_positions = origins + best_distance[:, None] * directions
        end_weights = weights * cp.exp(
            cp.where(medium == 1, -tip.optical.absorption_per_mm * best_distance, 0.0)
        )
        removed = weights - end_weights
        if bool(cp.any(removed < -1.0e-12)):
            raise Transport3DPhysicsError("segment attenuation increased branch weight")
        absorbed_weight += float(cp.asnumpy(cp.sum(removed)))
        if settings.retain_projected_segments:
            segment_chunks.append((origins.copy(), hit_positions.copy(), medium.copy(), weights.copy(), end_weights.copy()))
        new_states: list[tuple[Any, Any, Any, Any, Any, Any, Any, Any]] = []

        periodic = event == 3
        if bool(cp.any(periodic)):
            periodic_weight = end_weights[periodic]
            periodic_wrapped = wraps[periodic] + 1
            pathological = periodic_wrapped > settings.maximum_periodic_wraps
            if bool(cp.any(pathological)):
                raise Transport3DTraceError(
                    "periodic-wrap limit reached unexpectedly for an active branch"
                )
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
                interface_normal = cp.where(
                    (interface_medium == 1)[:, None], outward, -outward
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
    projected = (None, None, None)
    if settings.retain_projected_segments:
        projected = _projected_density(geometry, settings, segment_chunks, cp)[0:3]
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
        escape_weights=cp.asnumpy(escaped_weights),
        escape_primary_ray_indices=cp.asnumpy(escaped_primary),
        escape_path_lengths_mm=cp.asnumpy(escaped_paths),
        escape_interaction_counts=cp.asnumpy(escaped_interactions),
        energy_balance_error=float(energy_balance_error),
        energy_balance_tolerance=settings.energy_balance_tolerance,
        projected_x_edges_mm=None if projected[0] is None else projected[0],
        projected_y_edges_mm=None if projected[1] is None else projected[1],
        projected_weighted_path_density=None if projected[2] is None else projected[2],
        geometry_metadata=geometry.metadata,
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
    "trace_3d",
]
