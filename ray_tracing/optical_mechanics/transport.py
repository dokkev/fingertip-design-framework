"""Deterministic camera-independent FULL_3D optical transport."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from finger.fingertip import Fingertip
from ray_tracing.optical_mechanics.geometry import (
    CARRIER_CONTACT_INTERFACE,
    OBJECT_CONTACT_INTERFACE,
    TransportGeometry,
    Transport3DGeometryError,
)
from ray_tracing.optical_mechanics.optix_backend import (
    OptixScene,
    Transport3DDependencyError,
    Transport3DTraceError,
    OptixRuntime,
    create_runtime,
)
from ray_tracing.optical_mechanics.path_field import PathFieldAccumulator
from ray_tracing.optical_mechanics.physics import (
    Transport3DPhysicsError,
    interface_split,
    object_interface_split,
    periodic_plane_distance,
    wrapped_periodic_z,
)
from ray_tracing.optical_mechanics.result import Transport3DResult
from ray_tracing.optical_mechanics.sampling import sample_directions
from ray_tracing.optical_mechanics.settings import Transport3DSettings


@dataclass(frozen=True)
class _RayBatch:
    """State carried by active branches between transport events."""

    origins: Any
    directions: Any
    weights: Any
    media: Any
    primary_indices: Any
    interaction_counts: Any
    path_lengths: Any
    periodic_wraps: Any


@dataclass(frozen=True)
class _SegmentBatch:
    """One batch of segments booked into the native path field."""

    starts: Any
    ends: Any
    start_weights: Any
    end_weights: Any


@dataclass(frozen=True)
class _EscapeBatch:
    """Escapes retained for surface diagnostics and downstream analysis."""

    positions: Any
    directions: Any
    weights: Any
    primary_indices: Any
    path_lengths: Any
    interaction_counts: Any
    normals: Any
    surface_u: Any
    surface_z: Any
    primitive_indices: Any


@dataclass(frozen=True)
class _InternalField:
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    density: np.ndarray


def _xy_edges(
    geometry: TransportGeometry,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    margin = 0.04 * max(max_x - min_x, max_y - min_y)
    x_edges = np.linspace(min_x - margin, max_x + margin, width + 1)
    y_edges = np.linspace(min_y - margin, max_y + margin, height + 1)
    return x_edges, y_edges


def _internal_field_edges(
    geometry: TransportGeometry,
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
    u_start: Any,
    u_end: Any,
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
    u = u_start[primitive] + along * (u_end[primitive] - u_start[primitive])
    return points, cp.clip(u, 0.0, 1.0), points[:, 2]


def _new_internal_path_context(
    geometry: TransportGeometry,
    settings: Transport3DSettings,
) -> PathFieldAccumulator:
    x_edges, y_edges, z_edges = _internal_field_edges(geometry, settings)
    # A direct 3D surface artifact has no authoritative 2D projection.  The
    # native accumulator therefore books every sampled transport segment.
    density = np.zeros(
        (settings.internal_z_bins, settings.internal_grid_height, settings.internal_grid_width),
        dtype=float,
    )
    maximum_spacing = 0.5 * min(
        float(x_edges[1] - x_edges[0]),
        float(y_edges[1] - y_edges[0]),
        float(z_edges[1] - z_edges[0]),
    )
    return PathFieldAccumulator(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        density_zyx=density,
        maximum_spacing_mm=maximum_spacing,
        maximum_samples_per_segment=settings.internal_max_samples_per_segment,
    )


def _accumulate_internal_chunk(
    context: PathFieldAccumulator,
    chunk: _SegmentBatch,
    cp: Any,
) -> None:
    context.accumulate(
        cp.asnumpy(chunk.starts),
        cp.asnumpy(chunk.ends),
        cp.asnumpy(chunk.start_weights),
        cp.asnumpy(chunk.end_weights),
    )


def _finalize_internal_path_context(
    context: PathFieldAccumulator,
) -> _InternalField:
    try:
        density_xyz = context.density_xyz()
    except ValueError as exc:
        raise Transport3DTraceError(
            "3D internal path field is non-finite or negative"
        ) from exc

    return _InternalField(
        x_edges=context.x_edges,
        y_edges=context.y_edges,
        z_edges=context.z_edges,
        density=density_xyz,
    )


def _trace_with_runtime(
    tip: Fingertip,
    geometry: TransportGeometry,
    settings: Transport3DSettings,
    runtime: OptixRuntime,
) -> Transport3DResult:
    cp = runtime.cp
    if geometry.indenter_optics is not None and geometry.silicone.interface_tags is None:
        raise Transport3DGeometryError(
            "indenter optics require semantic silicone interface tags"
        )
    scene = OptixScene(runtime, geometry.silicone, geometry.rigid, geometry.envelope)
    carrier_coincidence_tolerance_mm = float(
        geometry.carrier_mapping_tolerance_mm
        if geometry.carrier_mapping_tolerance_mm is not None
        else settings.intersection_epsilon_mm
    )
    if not np.isfinite(carrier_coincidence_tolerance_mm) or carrier_coincidence_tolerance_mm < 0.0:
        raise Transport3DGeometryError(
            "carrier_mapping_tolerance_mm must be finite and non-negative"
        )
    carrier_absorber_enabled = bool(
        geometry.carrier_optics is not None
        and geometry.carrier_optics.boundary_model == "absorber"
    )
    # OptiX traverses float32 geometry.  Derive the self-hit offset from the
    # existing geometric epsilon and the float32 spacing at this cell scale.
    ray_offset = max(
        settings.intersection_epsilon_mm,
        8.0
        * float(np.finfo(np.float32).eps)
        * max(1.0, settings.extrusion_depth_mm),
    )
    silicone_vertices = cp.asarray(geometry.silicone.vertices)
    silicone_faces = cp.asarray(geometry.silicone.faces)
    silicone_normals = cp.asarray(geometry.silicone.normals)
    silicone_external = cp.asarray(geometry.silicone.external_surface)
    silicone_u_start = cp.asarray(geometry.silicone.u_start)
    silicone_u_end = cp.asarray(geometry.silicone.u_end)
    interface_tags = geometry.silicone.interface_tags
    object_contact_mask = cp.asarray(
        np.asarray(
            [tag == OBJECT_CONTACT_INTERFACE for tag in interface_tags],
            dtype=bool,
        )
        if interface_tags is not None
        else np.zeros(len(geometry.silicone.faces), dtype=bool)
    )
    carrier_contact_mask = cp.asarray(
        np.asarray(
            [tag == CARRIER_CONTACT_INTERFACE for tag in interface_tags],
            dtype=bool,
        )
        if interface_tags is not None
        else np.zeros(len(geometry.silicone.faces), dtype=bool)
    )
    source = cp.asarray(geometry.source_position_mm, dtype=cp.float32)
    axis = cp.asarray([tip.emission_axis[0], tip.emission_axis[1], 0.0], dtype=cp.float32)
    directions_np = sample_directions(
        tip.led,
        tip.emission_axis,
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
    object_absorbed_weight = 0.0
    object_transmitted_weight = 0.0
    object_interface_incident_weight = 0.0
    object_reflected_weight = 0.0
    carrier_absorbed_weight = 0.0
    carrier_transmitted_weight = 0.0
    carrier_interface_incident_weight = 0.0
    carrier_reflected_weight = 0.0
    periodic_wrap_termination_count = 0
    periodic_wrap_termination_weight = 0.0
    no_event_termination_count = 0
    no_event_termination_weight = 0.0
    branch_cutoff_termination_count = 0
    branch_cutoff_termination_weight = 0.0
    max_interaction_termination_count = 0
    max_interaction_termination_weight = 0.0
    segment_budget_termination_count = 0
    segment_budget_termination_weight = 0.0
    rigid_surface_termination_count = 0
    rigid_surface_termination_weight = 0.0
    interface_normal_orientation_fallback_count = 0
    segment_count = 0
    internal_context = (
        _new_internal_path_context(geometry, settings)
        if settings.retain_internal_path_field
        else None
    )
    escape_chunks: list[_EscapeBatch] = []
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
            low_count = int(cp.asnumpy(cp.count_nonzero(low)))
            low_weight = float(cp.asnumpy(cp.sum(weights[low])))
            branch_cutoff_termination_count += low_count
            branch_cutoff_termination_weight += low_weight
            terminated_weight += low_weight
            keep = ~low
            origins, directions, weights, medium = origins[keep], directions[keep], weights[keep], medium[keep]
            primary, interactions, path_lengths, wraps = primary[keep], interactions[keep], path_lengths[keep], wraps[keep]
        if not int(weights.size):
            break
        maxed = interactions >= settings.max_interactions
        if bool(cp.any(maxed)):
            maxed_count = int(cp.asnumpy(cp.count_nonzero(maxed)))
            maxed_weight = float(cp.asnumpy(cp.sum(weights[maxed])))
            max_interaction_termination_count += maxed_count
            max_interaction_termination_weight += maxed_weight
            terminated_weight += maxed_weight
            keep = ~maxed
            origins, directions, weights, medium = origins[keep], directions[keep], weights[keep], medium[keep]
            primary, interactions, path_lengths, wraps = primary[keep], interactions[keep], path_lengths[keep], wraps[keep]
        if not int(weights.size):
            break
        if segment_count + int(weights.size) > settings.maximum_segment_count:
            segment_budget_termination_count += int(weights.size)
            segment_budget_weight = float(cp.asnumpy(cp.sum(weights)))
            segment_budget_termination_weight += segment_budget_weight
            terminated_weight += segment_budget_weight
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
        safe_silicone_primitive = cp.maximum(silicone_primitive, 0)
        carrier_patch = carrier_contact_mask[safe_silicone_primitive]
        carrier_coincident = (
            (silicone_found != 0)
            & (rigid_found != 0)
            & carrier_patch
            & (
                cp.abs(silicone_distance - rigid_distance)
                <= carrier_coincidence_tolerance_mm
            )
        )
        # OptiX can return the rigid carrier first when the mechanics-resolved
        # silicone patch is coincident with it.  The semantic patch is the
        # authoritative optical boundary; prefer that triangle only inside
        # the documented contact/SDF tolerance.
        event = cp.where(carrier_coincident, 0, event)
        best_distance = cp.where(
            carrier_coincident, silicone_distance, best_distance
        )
        carrier_air_absorber_event = (
            carrier_coincident
            & (medium == 0)
            & carrier_absorber_enabled
        )
        event = cp.where(carrier_air_absorber_event, 4, event)
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
        if internal_context is not None:
            _accumulate_internal_chunk(
                internal_context,
                _SegmentBatch(
                    starts=origins,
                    ends=hit_positions,
                    start_weights=weights,
                    end_weights=end_weights,
                ),
                cp,
            )
        new_states: list[_RayBatch] = []

        periodic = event == 3
        if bool(cp.any(periodic)):
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
                new_states.append(_RayBatch(
                    origins=selected_origins,
                    directions=selected_directions,
                    weights=periodic_weight[keep_periodic],
                    media=medium[periodic][keep_periodic],
                    primary_indices=primary[periodic][keep_periodic],
                    interaction_counts=interactions[periodic][keep_periodic],
                    path_lengths=path_lengths[periodic][keep_periodic] + best_distance[periodic][keep_periodic],
                    periodic_wraps=periodic_wrapped[keep_periodic],
                ))

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
                rigid_surface_termination_count += int(
                    cp.asnumpy(cp.count_nonzero(rigid))
                )
                rigid_weight = float(
                    cp.asnumpy(cp.sum(physical_end_weights[rigid]))
                )
                rigid_surface_termination_weight += rigid_weight
                terminated_weight += rigid_weight
            carrier_air_absorber = physical_event == 4
            if bool(cp.any(carrier_air_absorber)):
                carrier_weight = physical_end_weights[carrier_air_absorber]
                carrier_interface_incident_weight += float(
                    cp.asnumpy(cp.sum(carrier_weight))
                )
                carrier_absorbed_weight += float(
                    cp.asnumpy(cp.sum(carrier_weight))
                )
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
                outward = silicone_normals[interface_primitive]
                silicone_object_contact = (
                    object_contact_mask[interface_primitive]
                    & (interface_medium == 1)
                )
                silicone_carrier_contact = (
                    carrier_contact_mask[interface_primitive]
                    & (interface_medium == 1)
                )
                tagged_contact = silicone_object_contact | silicone_carrier_contact
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
                    tagged_contact[:, None], outward, medium_normal
                )
                interface_normal = cp.where(
                    orientation_fallback[:, None], -interface_normal, interface_normal
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
                if bool(cp.any(silicone_object_contact)):
                    object_indices = cp.where(silicone_object_contact)[0]
                    object_incident = interface_end_weights[object_indices]
                    object_interface_incident_weight += float(
                        cp.asnumpy(cp.sum(object_incident))
                    )
                    if geometry.indenter_optics is not None and geometry.indenter_optics.boundary_model == "absorber":
                        object_absorbed_weight += float(cp.asnumpy(cp.sum(object_incident)))
                        reflected[object_indices] = 0.0
                        transmitted[object_indices] = 0.0
                        reflectance[object_indices] = 0.0
                        tir[object_indices] = False
                    else:
                        if (
                            geometry.indenter_optics is None
                            or geometry.indenter_optics.refractive_index is None
                        ):
                            raise Transport3DPhysicsError(
                                "dielectric indenter optics requires a refractive index"
                            )
                        object_reflected, object_transmitted, object_reflectance, object_tir = (
                            object_interface_split(
                                cp,
                                interface_directions[object_indices],
                                interface_normal[object_indices],
                                tip.optical.refractive_index_silicone,
                                float(geometry.indenter_optics.refractive_index),
                            )
                        )
                        reflected[object_indices] = object_reflected
                        transmitted[object_indices] = object_transmitted
                        reflectance[object_indices] = object_reflectance
                        tir[object_indices] = object_tir
                        object_reflected_weight += float(
                            cp.asnumpy(
                                cp.sum(object_incident * object_reflectance)
                            )
                        )
                        object_transmitted_weight += float(
                            cp.asnumpy(
                                cp.sum(object_incident * (1.0 - object_reflectance))
                            )
                        )
                if bool(cp.any(silicone_carrier_contact)):
                    carrier_indices = cp.where(silicone_carrier_contact)[0]
                    carrier_incident = interface_end_weights[carrier_indices]
                    carrier_interface_incident_weight += float(
                        cp.asnumpy(cp.sum(carrier_incident))
                    )
                    if geometry.carrier_optics is None:
                        raise Transport3DPhysicsError(
                            "carrier contact interface requires carrier optics"
                        )
                    if geometry.carrier_optics.boundary_model == "absorber":
                        carrier_absorbed_weight += float(
                            cp.asnumpy(cp.sum(carrier_incident))
                        )
                        reflected[carrier_indices] = 0.0
                        transmitted[carrier_indices] = 0.0
                        reflectance[carrier_indices] = 0.0
                        tir[carrier_indices] = False
                    else:
                        if geometry.carrier_optics.refractive_index is None:
                            raise Transport3DPhysicsError(
                                "dielectric carrier optics requires a refractive index"
                            )
                        carrier_reflected, carrier_transmitted, carrier_reflectance, carrier_tir = (
                            object_interface_split(
                                cp,
                                interface_directions[carrier_indices],
                                interface_normal[carrier_indices],
                                tip.optical.refractive_index_silicone,
                                float(geometry.carrier_optics.refractive_index),
                            )
                        )
                        reflected[carrier_indices] = carrier_reflected
                        transmitted[carrier_indices] = carrier_transmitted
                        reflectance[carrier_indices] = carrier_reflectance
                        tir[carrier_indices] = carrier_tir
                        carrier_reflected_weight += float(
                            cp.asnumpy(cp.sum(carrier_incident * carrier_reflectance))
                        )
                        carrier_transmitted_weight += float(
                            cp.asnumpy(
                                cp.sum(carrier_incident * (1.0 - carrier_reflectance))
                            )
                        )
                reflected_weight = interface_end_weights * reflectance
                transmitted_weight = interface_end_weights * (1.0 - reflectance)
                next_interaction = interface_interactions + 1
                first_generation = next_interaction <= 1
                reflected_keep = (
                    ((reflected_weight >= threshold) | first_generation)
                    & (reflected_weight > 0.0)
                )
                reflected_terminate = ~reflected_keep
                if bool(cp.any(reflected_terminate)):
                    positive_reflected_terminate = reflected_terminate & (
                        reflected_weight > 0.0
                    )
                    branch_cutoff_termination_count += int(
                        cp.asnumpy(cp.count_nonzero(positive_reflected_terminate))
                    )
                    reflected_termination_weight = float(
                        cp.asnumpy(
                            cp.sum(reflected_weight[positive_reflected_terminate])
                        )
                    )
                    branch_cutoff_termination_weight += reflected_termination_weight
                    terminated_weight += reflected_termination_weight
                if bool(cp.any(reflected_keep)):
                    new_states.append(_RayBatch(
                        origins=interface_positions[reflected_keep] + ray_offset * reflected[reflected_keep],
                        directions=reflected[reflected_keep],
                        weights=reflected_weight[reflected_keep],
                        media=interface_medium[reflected_keep],
                        primary_indices=interface_primary[reflected_keep],
                        interaction_counts=next_interaction[reflected_keep],
                        path_lengths=interface_paths[reflected_keep],
                        periodic_wraps=wraps[physical][indices][reflected_keep],
                    ))
                interface_points, interface_u, interface_z = _surface_coordinates(
                    cp,
                    silicone_u_start,
                    silicone_u_end,
                    interface_primitive,
                    interface_bary,
                    silicone_vertices,
                    silicone_faces,
                )
                external_escape = (
                    (interface_medium == 1)
                    & silicone_external[interface_primitive]
                    & ~tagged_contact
                )
                # Contact-only optics is defined for silicone rays reaching
                # the mechanically contacted boundary.  A ray already in air
                # sees the ordinary air/silicone interface; the exposed
                # indenter body is deliberately absent from the scene.
                ordinary_transmission = (
                    ~external_escape
                    & ~silicone_object_contact
                    & ~silicone_carrier_contact
                )
                if bool(cp.any(external_escape)):
                    outgoing_weight = transmitted_weight[external_escape] * (~tir[external_escape])
                    escaped_weight += float(cp.asnumpy(cp.sum(outgoing_weight)))
                    positive_outgoing = outgoing_weight > 0.0
                    outgoing_u = interface_u[external_escape]
                    outgoing_z = interface_z[external_escape]
                    inside_observation_grid = (
                        (outgoing_u >= surface_u_edges[0])
                        & (outgoing_u <= surface_u_edges[-1])
                        & (outgoing_z >= surface_z_edges[0])
                        & (outgoing_z <= surface_z_edges[-1])
                    )
                    if bool(cp.any(positive_outgoing & ~inside_observation_grid)):
                        raise Transport3DGeometryError(
                            "positive silicone-surface escape lies outside the "
                            "declared (u, z) observation grid"
                        )
                    if bool(cp.any(positive_outgoing)):
                        escape_chunks.append(_EscapeBatch(
                            positions=interface_points[external_escape][positive_outgoing],
                            directions=transmitted[external_escape][positive_outgoing],
                            weights=outgoing_weight[positive_outgoing],
                            primary_indices=interface_primary[external_escape][positive_outgoing],
                            path_lengths=interface_paths[external_escape][positive_outgoing],
                            interaction_counts=next_interaction[external_escape][positive_outgoing],
                            normals=outward[external_escape][positive_outgoing],
                            surface_u=outgoing_u[positive_outgoing],
                            surface_z=outgoing_z[positive_outgoing],
                            primitive_indices=interface_primitive[external_escape][positive_outgoing],
                        ))
                    u_indices = cp.clip(
                        cp.searchsorted(
                            surface_u_edges,
                            outgoing_u,
                            side="right",
                        )
                        - 1,
                        0,
                        settings.surface_u_bins - 1,
                    )
                    z_indices = cp.clip(
                        cp.searchsorted(
                            surface_z_edges,
                            outgoing_z,
                            side="right",
                        )
                        - 1,
                        0,
                        settings.surface_z_bins - 1,
                    )
                    cp.add.at(surface_field, (z_indices, u_indices), outgoing_weight)
                ordinary_transmission &= transmitted_weight > 0.0
                transmission_keep = ordinary_transmission & (
                    (transmitted_weight >= threshold) | first_generation
                ) & (~tir)
                transmission_terminate = ordinary_transmission & ~transmission_keep
                if bool(cp.any(transmission_terminate)):
                    positive_transmission_terminate = transmission_terminate & (
                        transmitted_weight > 0.0
                    )
                    branch_cutoff_termination_count += int(
                        cp.asnumpy(cp.count_nonzero(positive_transmission_terminate))
                    )
                    transmission_termination_weight = float(
                        cp.asnumpy(
                            cp.sum(transmitted_weight[positive_transmission_terminate])
                        )
                    )
                    branch_cutoff_termination_weight += transmission_termination_weight
                    terminated_weight += transmission_termination_weight
                if bool(cp.any(transmission_keep)):
                    new_states.append(_RayBatch(
                        origins=interface_positions[transmission_keep] + ray_offset * transmitted[transmission_keep],
                        directions=transmitted[transmission_keep],
                        weights=transmitted_weight[transmission_keep],
                        media=cp.where(interface_medium[transmission_keep] == 0, 1, 0).astype(cp.uint8),
                        primary_indices=interface_primary[transmission_keep],
                        interaction_counts=next_interaction[transmission_keep],
                        path_lengths=interface_paths[transmission_keep],
                        periodic_wraps=wraps[physical][indices][transmission_keep],
                    ))

        if new_states:
            origins = _concatenate(cp, [state.origins for state in new_states], dtype=cp.float32, width=3)
            directions = _concatenate(cp, [state.directions for state in new_states], dtype=cp.float32, width=3)
            weights = _concatenate(cp, [state.weights for state in new_states], dtype=cp.float64)
            medium = _concatenate(cp, [state.media for state in new_states], dtype=cp.uint8)
            primary = _concatenate(cp, [state.primary_indices for state in new_states], dtype=cp.int64)
            interactions = _concatenate(cp, [state.interaction_counts for state in new_states], dtype=cp.int64)
            path_lengths = _concatenate(cp, [state.path_lengths for state in new_states], dtype=cp.float64)
            wraps = _concatenate(cp, [state.periodic_wraps for state in new_states], dtype=cp.int64)
        else:
            weights = cp.empty(0, dtype=cp.float64)

    postprocessing_started = time.perf_counter()
    escaped_positions = _concatenate(cp, [chunk.positions for chunk in escape_chunks], dtype=cp.float32, width=3)
    escaped_directions = _concatenate(cp, [chunk.directions for chunk in escape_chunks], dtype=cp.float32, width=3)
    escaped_weights = _concatenate(cp, [chunk.weights for chunk in escape_chunks], dtype=cp.float64)
    escaped_primary = _concatenate(cp, [chunk.primary_indices for chunk in escape_chunks], dtype=cp.int64)
    escaped_paths = _concatenate(cp, [chunk.path_lengths for chunk in escape_chunks], dtype=cp.float64)
    escaped_interactions = _concatenate(cp, [chunk.interaction_counts for chunk in escape_chunks], dtype=cp.int64)
    escaped_normals = _concatenate(cp, [chunk.normals for chunk in escape_chunks], dtype=cp.float32, width=3)
    escaped_u = _concatenate(cp, [chunk.surface_u for chunk in escape_chunks], dtype=cp.float64)
    escaped_z = _concatenate(cp, [chunk.surface_z for chunk in escape_chunks], dtype=cp.float64)
    escaped_primitives = _concatenate(cp, [chunk.primitive_indices for chunk in escape_chunks], dtype=cp.int64)
    if geometry.silicone.semantic_tags is None:
        raise Transport3DGeometryError(
            "silicone surface is missing semantic boundary tags"
        )
    escaped_tags = tuple(
        geometry.silicone.semantic_tags[int(primitive)]
        for primitive in cp.asnumpy(escaped_primitives)
    )
    internal: _InternalField | None = None
    if internal_context is not None:
        internal = _finalize_internal_path_context(internal_context)
    geometry_metadata = dict(geometry.metadata)
    carrier_contact_triangle_count = int(
        sum(
            tag == CARRIER_CONTACT_INTERFACE
            for tag in (geometry.silicone.interface_tags or ())
        )
    )
    energy_balance_error = abs(
        launched_weight
        - escaped_weight
        - absorbed_weight
        - terminated_weight
        - object_absorbed_weight
        - object_transmitted_weight
        - carrier_absorbed_weight
        - carrier_transmitted_weight
    ) / max(launched_weight, 1.0e-30)
    if not np.isfinite(energy_balance_error) or energy_balance_error > settings.energy_balance_tolerance:
        raise Transport3DPhysicsError(
            f"energy balance error {energy_balance_error:g} exceeds tolerance {settings.energy_balance_tolerance:g}"
        )
    return Transport3DResult(
        source_position_mm=geometry.source_position_mm,
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
        processed_segment_count=segment_count,
        periodic_wrap_termination_count=periodic_wrap_termination_count,
        periodic_wrap_termination_weight=periodic_wrap_termination_weight,
        no_event_termination_count=no_event_termination_count,
        no_event_termination_weight=no_event_termination_weight,
        branch_cutoff_termination_count=branch_cutoff_termination_count,
        branch_cutoff_termination_weight=branch_cutoff_termination_weight,
        max_interaction_termination_count=max_interaction_termination_count,
        max_interaction_termination_weight=max_interaction_termination_weight,
        segment_budget_termination_count=segment_budget_termination_count,
        segment_budget_termination_weight=segment_budget_termination_weight,
        rigid_surface_termination_count=rigid_surface_termination_count,
        rigid_surface_termination_weight=rigid_surface_termination_weight,
        interface_normal_fallback_count=interface_normal_orientation_fallback_count,
        carrier_contact_triangle_count=carrier_contact_triangle_count,
        object_absorbed_weight=object_absorbed_weight,
        object_transmitted_weight=object_transmitted_weight,
        object_interface_incident_weight=object_interface_incident_weight,
        object_reflected_weight=object_reflected_weight,
        carrier_absorbed_weight=carrier_absorbed_weight,
        carrier_transmitted_weight=carrier_transmitted_weight,
        carrier_interface_incident_weight=carrier_interface_incident_weight,
        carrier_reflected_weight=carrier_reflected_weight,
        field_x_edges_mm=None if internal is None else internal.x_edges,
        field_y_edges_mm=None if internal is None else internal.y_edges,
        field_z_edges_mm=None if internal is None else internal.z_edges,
        field_density_3d=None if internal is None else internal.density,
        geometry_metadata=geometry_metadata,
        timings_seconds={
            "gas_build": scene.gas_build_seconds,
            "transport": postprocessing_started - transport_started,
            "postprocessing": time.perf_counter() - postprocessing_started,
        },
    )


def trace_geometry(
    tip: Fingertip,
    geometry: TransportGeometry,
    *,
    settings: Transport3DSettings | None = None,
    runtime: Any | None = None,
) -> Transport3DResult:
    """Trace a prevalidated neutral geometry through the shared OptiX core."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    if not isinstance(geometry, TransportGeometry):
        raise TypeError("geometry must be a TransportGeometry")
    trace_settings = settings or Transport3DSettings()
    actual_runtime = runtime or create_runtime()
    if not isinstance(actual_runtime, OptixRuntime):
        raise TypeError("runtime must be an OptixRuntime")
    return _trace_with_runtime(tip, geometry, trace_settings, actual_runtime)


__all__ = [
    "Transport3DDependencyError",
    "Transport3DGeometryError",
    "Transport3DPhysicsError",
    "Transport3DResult",
    "Transport3DSettings",
    "Transport3DTraceError",
    "trace_geometry",
]
