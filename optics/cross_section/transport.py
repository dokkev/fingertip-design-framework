"""Deterministic geometric-optics transport in an x-y optical domain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import asin, ceil, cos, exp, hypot, radians, sin, sqrt
from typing import Iterable

import numpy as np
from shapely import contains_xy
from shapely.geometry import (
    GeometryCollection,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from model.fingertip_model import PolygonalGeometry
from model.optical import LED, OpticalMaterial
from optics.cross_section.domain import (
    CrossSectionOpticsError,
    _OpticalDomain,
)
from optics.cross_section.result import (
    OpticalMedium,
    _RawRaySegment,
    _RawTransportResult,
)
from optics.cross_section.settings import TraceSettings


@dataclass(frozen=True)
class _RayState:
    origin: np.ndarray
    direction: np.ndarray
    weight: float
    medium: OpticalMedium
    interaction_count: int
    primary_ray_index: int


@dataclass(frozen=True)
class _PreparedGeometry:
    accessible_region: PolygonalGeometry
    all_boundaries: BaseGeometry
    silicone_rings: tuple[LinearRing, ...]
    maximum_ray_length_mm: float


def _normalize(vector: np.ndarray, *, context: str) -> np.ndarray:
    if not np.all(np.isfinite(vector)):
        raise CrossSectionOpticsError(f"{context} contains non-finite values")
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 1.0e-14:
        raise CrossSectionOpticsError(f"{context} is degenerate")
    return vector / magnitude


def _iter_polygons(geometry: PolygonalGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    return geometry.geoms


def _prepare_geometry(domain: _OpticalDomain) -> _PreparedGeometry:
    accessible_region = domain.accessible_region
    if accessible_region.is_empty or not accessible_region.is_valid:
        raise CrossSectionOpticsError("the accessible optical region is invalid")

    rings = tuple(
        ring
        for polygon in _iter_polygons(domain.silicone_region)
        for ring in (polygon.exterior, *polygon.interiors)
    )
    if not rings:
        raise CrossSectionOpticsError("the silicone region has no boundary rings")

    all_boundaries = unary_union(
        [
            domain.silicone_region.boundary,
            domain.rigid_region.boundary,
            domain.outer_envelope.boundary,
        ]
    )
    min_x, min_y, max_x, max_y = domain.outer_envelope.bounds
    diagonal = hypot(max_x - min_x, max_y - min_y)
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise CrossSectionOpticsError("the outer-envelope diagonal is invalid")

    return _PreparedGeometry(
        accessible_region=accessible_region,
        all_boundaries=all_boundaries,
        silicone_rings=rings,
        maximum_ray_length_mm=4.0 * diagonal,
    )


def _sample_primary_directions(
    led: LED,
    emission_axis_2d: tuple[float, float],
    settings: TraceSettings,
) -> tuple[np.ndarray, ...]:
    half_angle = radians(led.emission_half_angle_deg)
    half_angle_sine = sin(half_angle)
    emission_axis = _normalize(
        np.asarray(emission_axis_2d, dtype=float),
        context="LED emission axis",
    )
    positive_angular_axis = np.asarray(
        [-emission_axis[1], emission_axis[0]],
        dtype=float,
    )
    directions: list[np.ndarray] = []
    for index in range(settings.ray_count):
        quantile = (index + 0.5) / settings.ray_count
        theta = asin((2.0 * quantile - 1.0) * half_angle_sine)
        directions.append(
            _normalize(
                emission_axis * cos(theta)
                + positive_angular_axis * sin(theta),
                context="sampled ray direction",
            )
        )
    return tuple(directions)


def _classify_medium(
    point_xy: np.ndarray,
    domain: _OpticalDomain,
    prepared: _PreparedGeometry,
) -> OpticalMedium:
    point = Point(float(point_xy[0]), float(point_xy[1]))
    if domain.silicone_region.covers(point):
        return "silicone"
    if prepared.accessible_region.covers(point):
        return "air"
    raise CrossSectionOpticsError(
        "a ray origin is outside the accessible optical region"
    )


def _intersection_points(geometry: BaseGeometry) -> Iterable[Point]:
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Point):
        return (geometry,)
    if isinstance(geometry, MultiPoint | GeometryCollection):
        return tuple(
            point
            for component in geometry.geoms
            for point in _intersection_points(component)
        )
    if isinstance(geometry, LinearRing | LineString):
        coordinates = list(geometry.coords)
        if not coordinates:
            return ()
        return (Point(coordinates[0]), Point(coordinates[-1]))
    if isinstance(geometry, MultiLineString):
        return tuple(
            point
            for component in geometry.geoms
            for point in _intersection_points(component)
        )
    return ()


def _find_next_hit(
    state: _RayState,
    prepared: _PreparedGeometry,
    settings: TraceSettings,
) -> np.ndarray:
    ray_end = (
        state.origin + prepared.maximum_ray_length_mm * state.direction
    )
    forward_line = LineString([state.origin, ray_end])
    candidates = _intersection_points(
        prepared.all_boundaries.intersection(forward_line)
    )

    nearest_distance = float("inf")
    nearest_point: np.ndarray | None = None
    for point in candidates:
        candidate = np.asarray([point.x, point.y], dtype=float)
        delta = candidate - state.origin
        forward_distance = float(np.dot(delta, state.direction))
        if forward_distance <= settings.intersection_epsilon_mm:
            continue
        lateral_distance = float(
            np.linalg.norm(delta - forward_distance * state.direction)
        )
        if lateral_distance > 10.0 * settings.intersection_epsilon_mm:
            continue
        if forward_distance < nearest_distance:
            nearest_distance = forward_distance
            nearest_point = candidate

    if nearest_point is None:
        raise CrossSectionOpticsError(
            "no forward geometry intersection was found for an active ray"
        )
    return nearest_point


def _silicone_outward_normal(
    hit_point: np.ndarray,
    domain: _OpticalDomain,
    prepared: _PreparedGeometry,
    settings: TraceSettings,
) -> np.ndarray:
    hit = Point(float(hit_point[0]), float(hit_point[1]))
    ring = min(prepared.silicone_rings, key=lambda candidate: candidate.distance(hit))
    projected_distance = float(ring.project(hit))
    arc_step = min(
        0.25 * ring.length,
        max(
            32.0 * settings.intersection_epsilon_mm,
            1.0e-6 * ring.length,
        ),
    )
    before = ring.interpolate((projected_distance - arc_step) % ring.length)
    after = ring.interpolate((projected_distance + arc_step) % ring.length)
    tangent = _normalize(
        np.asarray([after.x - before.x, after.y - before.y], dtype=float),
        context="silicone boundary tangent",
    )
    normal = _normalize(
        np.asarray([-tangent[1], tangent[0]], dtype=float),
        context="silicone boundary normal",
    )

    base_probe_distance = max(
        8.0 * settings.intersection_epsilon_mm,
        8.0 * domain.geometry_tolerance_mm,
    )
    for scale in (1.0, 4.0, 16.0, 64.0):
        probe_distance = scale * base_probe_distance
        positive_probe = Point(*(hit_point + probe_distance * normal))
        negative_probe = Point(*(hit_point - probe_distance * normal))
        positive_is_silicone = domain.silicone_region.covers(positive_probe)
        negative_is_silicone = domain.silicone_region.covers(negative_probe)
        if not positive_is_silicone and negative_is_silicone:
            return normal
        if positive_is_silicone and not negative_is_silicone:
            return -normal

    raise CrossSectionOpticsError(
        "a stable silicone-to-air normal could not be derived at an interface"
    )


def _interface_directions_and_reflectance(
    incident_direction: np.ndarray,
    interface_normal: np.ndarray,
    refractive_index_1: float,
    refractive_index_2: float,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    incident = _normalize(incident_direction, context="incident direction")
    normal = _normalize(interface_normal, context="interface normal")
    alignment = float(np.dot(incident, normal))
    if alignment <= 1.0e-12:
        raise CrossSectionOpticsError(
            "the interface normal does not point into the next medium"
        )

    tangent = np.asarray([-normal[1], normal[0]], dtype=float)
    sin_incident = float(np.dot(incident, tangent))
    sin_transmitted = (
        refractive_index_1 / refractive_index_2
    ) * sin_incident
    reflected = _normalize(
        incident - 2.0 * alignment * normal,
        context="reflected direction",
    )
    if abs(sin_transmitted) > 1.0:
        return reflected, None, 1.0

    cos_incident = abs(alignment)
    cos_transmitted = sqrt(max(0.0, 1.0 - sin_transmitted**2))
    transmitted = _normalize(
        tangent * sin_transmitted + normal * cos_transmitted,
        context="transmitted direction",
    )

    s_denominator = (
        refractive_index_1 * cos_incident
        + refractive_index_2 * cos_transmitted
    )
    p_denominator = (
        refractive_index_1 * cos_transmitted
        + refractive_index_2 * cos_incident
    )
    if s_denominator <= 0.0 or p_denominator <= 0.0:
        raise CrossSectionOpticsError("the Fresnel denominator is nonpositive")
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
    reflectance = min(1.0, max(0.0, 0.5 * (reflectance_s + reflectance_p)))
    return reflected, transmitted, reflectance


def _refractive_index(
    medium: OpticalMedium,
    material: OpticalMaterial,
) -> float:
    if medium == "air":
        return material.refractive_index_air
    return material.refractive_index_silicone


def _build_path_density_grid(
    domain: _OpticalDomain,
    prepared: _PreparedGeometry,
    settings: TraceSettings,
    segments: tuple[_RawRaySegment, ...],
    *,
    x_bounds_mm: tuple[float, float] | None = None,
    y_bounds_mm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = domain.outer_envelope.bounds
    margin = 0.04 * max(max_x - min_x, max_y - min_y)
    if x_bounds_mm is None:
        x_bounds_mm = (min_x - margin, max_x + margin)
    if y_bounds_mm is None:
        y_bounds_mm = (min_y - margin, max_y + margin)
    x_edges = np.linspace(
        float(x_bounds_mm[0]),
        float(x_bounds_mm[1]),
        settings.grid_width + 1,
    )
    y_edges = np.linspace(
        float(y_bounds_mm[0]),
        float(y_bounds_mm[1]),
        settings.grid_height + 1,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    center_x, center_y = np.meshgrid(x_centers, y_centers)
    optical_mask = np.asarray(
        contains_xy(prepared.accessible_region, center_x, center_y),
        dtype=bool,
    )
    density = np.zeros(
        (settings.grid_height, settings.grid_width),
        dtype=float,
    )
    cell_width = float(x_edges[1] - x_edges[0])
    cell_height = float(y_edges[1] - y_edges[0])
    maximum_spacing = 0.25 * min(cell_width, cell_height)

    for segment in segments:
        start = np.asarray(segment.start_mm, dtype=float)
        end = np.asarray(segment.end_mm, dtype=float)
        displacement = end - start
        length = float(np.linalg.norm(displacement))
        if length <= 0.0:
            continue
        sample_count = max(1, int(ceil(length / maximum_spacing)))
        fractions = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
        samples = start[None, :] + fractions[:, None] * displacement[None, :]
        x_indices = np.searchsorted(x_edges, samples[:, 0], side="right") - 1
        y_indices = np.searchsorted(y_edges, samples[:, 1], side="right") - 1
        valid = (
            (x_indices >= 0)
            & (x_indices < settings.grid_width)
            & (y_indices >= 0)
            & (y_indices < settings.grid_height)
        )
        if not np.any(valid):
            continue
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]
        inside_optical_region = optical_mask[y_indices, x_indices]
        x_indices = x_indices[inside_optical_region]
        y_indices = y_indices[inside_optical_region]
        if len(x_indices) == 0:
            continue
        representative_weight = 0.5 * (
            segment.start_weight + segment.end_weight
        )
        represented_length = length / sample_count
        np.add.at(
            density,
            (y_indices, x_indices),
            representative_weight * represented_length,
        )

    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise CrossSectionOpticsError(
            "weighted ray-path density is non-finite or negative"
        )
    return x_edges, y_edges, density, optical_mask


def _trace_transport(
    domain: _OpticalDomain,
    *,
    led: LED,
    material: OpticalMaterial,
    settings: TraceSettings | None = None,
) -> _RawTransportResult:
    """Trace deterministic Fresnel ray branches through a neutral 2D domain."""
    trace_settings = settings or TraceSettings()
    led_properties = led
    material_properties = material
    trace_settings.validate(geometry_tolerance_mm=domain.geometry_tolerance_mm)
    prepared = _prepare_geometry(domain)

    source = np.asarray(domain.source_position_mm, dtype=float)
    emission_axis = _normalize(
        np.asarray(domain.source_emission_axis_2d, dtype=float),
        context="domain source emission axis",
    )
    trace_origin = source + trace_settings.source_epsilon_mm * emission_axis
    initial_medium = _classify_medium(trace_origin, domain, prepared)
    primary_weight = led_properties.relative_radiant_power / trace_settings.ray_count
    branch_weight_threshold = (
        trace_settings.minimum_ray_weight
        * led_properties.relative_radiant_power
    )
    active = deque(
        _RayState(
            origin=trace_origin.copy(),
            direction=direction,
            weight=primary_weight,
            medium=initial_medium,
            interaction_count=0,
            primary_ray_index=index,
        )
        for index, direction in enumerate(
            _sample_primary_directions(
                led_properties,
                domain.source_emission_axis_2d,
                trace_settings,
            )
        )
        if led_properties.relative_radiant_power > 0.0
    )

    segments: list[_RawRaySegment] = []
    escaped_weight = 0.0
    absorbed_weight = 0.0
    terminated_weight = 0.0

    while active:
        state = active.popleft()
        if len(segments) >= trace_settings.maximum_segment_count:
            terminated_weight += state.weight + sum(ray.weight for ray in active)
            active.clear()
            break
        # Keep the primary launch and its first-generation interface children
        # active even when a high ray-count sweep makes one primary share
        # smaller than the secondary-branch cutoff.  This is the shared
        # convention used by the 3D transport; the default historical ray
        # count is numerically unaffected.
        if (
            state.weight < branch_weight_threshold
            and state.interaction_count > 1
        ):
            terminated_weight += state.weight
            continue
        if state.interaction_count >= trace_settings.max_interactions:
            terminated_weight += state.weight
            continue

        hit_point = _find_next_hit(state, prepared, trace_settings)
        segment_length = float(np.linalg.norm(hit_point - state.origin))
        if state.medium == "silicone":
            end_weight = state.weight * exp(
                -material_properties.absorption_per_mm * segment_length
            )
        else:
            end_weight = state.weight
        absorbed_weight += state.weight - end_weight
        segments.append(
            _RawRaySegment(
                start_mm=(float(state.origin[0]), float(state.origin[1])),
                end_mm=(float(hit_point[0]), float(hit_point[1])),
                medium=state.medium,
                start_weight=state.weight,
                end_weight=end_weight,
                primary_ray_index=state.primary_ray_index,
                interaction_index=state.interaction_count,
            )
        )

        forward_probe_xy = (
            hit_point
            + trace_settings.intersection_epsilon_mm * state.direction
        )
        forward_probe = Point(
            float(forward_probe_xy[0]),
            float(forward_probe_xy[1]),
        )
        if domain.rigid_region.covers(forward_probe):
            terminated_weight += end_weight
            continue
        if not domain.outer_envelope.covers(forward_probe):
            if state.medium == "air":
                escaped_weight += end_weight
                continue

            silicone_outward_normal = _silicone_outward_normal(
                hit_point,
                domain,
                prepared,
                trace_settings,
            )
            reflected_direction, transmitted_direction, reflectance = (
                _interface_directions_and_reflectance(
                    state.direction,
                    silicone_outward_normal,
                    material_properties.refractive_index_silicone,
                    material_properties.refractive_index_air,
                )
            )
            reflected_weight = end_weight * reflectance
            transmitted_weight = end_weight * (1.0 - reflectance)
            escaped_weight += transmitted_weight
            next_interaction = state.interaction_count + 1

            if reflected_weight >= branch_weight_threshold or next_interaction <= 1:
                active.append(
                    _RayState(
                        origin=(
                            hit_point
                            + trace_settings.intersection_epsilon_mm
                            * reflected_direction
                        ),
                        direction=reflected_direction,
                        weight=reflected_weight,
                        medium="silicone",
                        interaction_count=next_interaction,
                        primary_ray_index=state.primary_ray_index,
                    )
                )
            else:
                terminated_weight += reflected_weight
            continue

        next_medium: OpticalMedium = (
            "silicone"
            if domain.silicone_region.covers(forward_probe)
            else "air"
        )
        if next_medium == state.medium:
            active.append(
                _RayState(
                    origin=forward_probe_xy,
                    direction=state.direction,
                    weight=end_weight,
                    medium=state.medium,
                    interaction_count=state.interaction_count + 1,
                    primary_ray_index=state.primary_ray_index,
                )
            )
            continue

        silicone_outward_normal = _silicone_outward_normal(
            hit_point,
            domain,
            prepared,
            trace_settings,
        )
        interface_normal = (
            silicone_outward_normal
            if state.medium == "silicone"
            else -silicone_outward_normal
        )
        reflected_direction, transmitted_direction, reflectance = (
            _interface_directions_and_reflectance(
                state.direction,
                interface_normal,
                _refractive_index(state.medium, material_properties),
                _refractive_index(next_medium, material_properties),
            )
        )
        reflected_weight = end_weight * reflectance
        transmitted_weight = end_weight * (1.0 - reflectance)
        next_interaction = state.interaction_count + 1

        if reflected_weight >= branch_weight_threshold or next_interaction <= 1:
            active.append(
                _RayState(
                    origin=(
                        hit_point
                        + trace_settings.intersection_epsilon_mm
                        * reflected_direction
                    ),
                    direction=reflected_direction,
                    weight=reflected_weight,
                    medium=state.medium,
                    interaction_count=next_interaction,
                    primary_ray_index=state.primary_ray_index,
                )
            )
        else:
            terminated_weight += reflected_weight

        if (
            transmitted_direction is not None
            and (
                transmitted_weight >= branch_weight_threshold
                or next_interaction <= 1
            )
        ):
            active.append(
                _RayState(
                    origin=(
                        hit_point
                        + trace_settings.intersection_epsilon_mm
                        * transmitted_direction
                    ),
                    direction=transmitted_direction,
                    weight=transmitted_weight,
                    medium=next_medium,
                    interaction_count=next_interaction,
                    primary_ray_index=state.primary_ray_index,
                )
            )
        elif transmitted_weight > 0.0:
            terminated_weight += transmitted_weight

    retained_segments = tuple(segments)
    x_edges, y_edges, density, optical_mask = _build_path_density_grid(
        domain,
        prepared,
        trace_settings,
        retained_segments,
    )
    weights = np.asarray(
        [escaped_weight, absorbed_weight, terminated_weight],
        dtype=float,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise CrossSectionOpticsError(
            "transport accounting is non-finite or negative"
        )

    return _RawTransportResult(
        source_position_mm=(float(source[0]), float(source[1])),
        x_edges_mm=x_edges,
        y_edges_mm=y_edges,
        weighted_path_density=density,
        optical_mask=optical_mask,
        segments=retained_segments,
        launched_ray_count=trace_settings.ray_count,
        launched_weight=float(led_properties.relative_radiant_power),
        escaped_weight=float(escaped_weight),
        absorbed_weight=float(absorbed_weight),
        terminated_weight=float(terminated_weight),
    )
