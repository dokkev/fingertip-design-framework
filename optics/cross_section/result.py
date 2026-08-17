"""Neutral results produced by the cross-sectional optical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OpticalMedium = Literal["air", "silicone"]


@dataclass(frozen=True)
class _RawRaySegment:
    """One retained straight segment of a deterministic ray branch."""

    start_mm: tuple[float, float]
    end_mm: tuple[float, float]
    medium: OpticalMedium
    start_weight: float
    end_weight: float
    primary_ray_index: int
    interaction_index: int


@dataclass(frozen=True)
class _RawExitEvent:
    """One outgoing air escape event retained for later side-view analysis."""

    position_mm: tuple[float, float]
    direction: tuple[float, float]
    weight: float
    boundary_tag: str | None
    primary_ray_index: int
    interaction_index: int


@dataclass(frozen=True)
class _RawTransportResult:
    """Raw weighted paths and a regular-grid path-length accumulation."""

    source_position_mm: tuple[float, float]
    x_edges_mm: np.ndarray
    y_edges_mm: np.ndarray
    weighted_path_density: np.ndarray
    optical_mask: np.ndarray
    segments: tuple[_RawRaySegment, ...]
    exit_events: tuple[_RawExitEvent, ...]
    launched_ray_count: int
    launched_weight: float
    escaped_weight: float
    absorbed_weight: float
    terminated_weight: float
    object_absorbed_weight: float
    object_transmitted_weight: float
    object_interface_incident_weight: float
    object_reflected_weight: float
