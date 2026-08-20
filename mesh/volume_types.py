"""Neutral data contracts for independent 3D volume meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np

from model.fingertip_model import FingertipParameters
from model.solid import FingertipSolid


VolumeMeshTier = Literal["search", "reference"]
VolumeDomain = Literal["pad"]


@dataclass(frozen=True)
class VolumeMeshSettings:
    """Precommitted 3D Gmsh sizing tiers."""

    tier: VolumeMeshTier
    target_size_mm: float
    minimum_quality: float = 0.02

    def __post_init__(self) -> None:
        if self.tier not in ("search", "reference"):
            raise ValueError("tier must be 'search' or 'reference'")
        if not math.isfinite(self.target_size_mm) or self.target_size_mm <= 0.0:
            raise ValueError("target_size_mm must be finite and positive")
        if not math.isfinite(self.minimum_quality) or not 0.0 < self.minimum_quality < 1.0:
            raise ValueError("minimum_quality must lie in (0, 1)")


def volume_mesh_settings_for_tier(tier: VolumeMeshTier) -> VolumeMeshSettings:
    """Return the fixed search/reference sizing policy."""
    if tier == "search":
        return VolumeMeshSettings("search", 1.5, 0.02)
    if tier == "reference":
        return VolumeMeshSettings("reference", 1.0, 0.02)
    raise ValueError(f"unsupported volume mesh tier: {tier!r}")


@dataclass(frozen=True)
class VolumeNode:
    """One solver-independent 3D volume-mesh node."""

    id: int
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class Tetrahedron:
    """One positively oriented linear tetrahedral element."""

    id: int
    node_ids: tuple[int, int, int, int]
    domain: VolumeDomain


@dataclass(frozen=True)
class SurfaceTriangle:
    """One oriented surface triangle with stable semantic identity."""

    id: int
    node_ids: tuple[int, int, int]
    semantic_tag: str
    domain: VolumeDomain


@dataclass(frozen=True)
class VolumeMeshQuality:
    """Independent volume and tetrahedron quality statistics."""

    node_count: int
    tetrahedron_count: int
    surface_triangle_count: int
    minimum_scaled_jacobian: float
    maximum_edge_length_mm: float
    mesh_volume_mm3: float
    geometry_volume_mm3: float
    volume_relative_error: float
    inverted_tetrahedron_count: int
    semantic_surface_tags: tuple[str, ...]
    surface_triangle_degenerate_count: int
    surface_orientation_failure_count: int
    closed_surface_edge_failure_count: int
    bonded_surface_triangle_count: int
    bonded_surface_area_mm2: float
    bonded_surface_expected_area_mm2: float
    bonded_surface_area_relative_error: float


@dataclass(frozen=True)
class VolumeMeshValidation:
    """Named M2 acceptance checks."""

    passed: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class FingertipVolumeMesh:
    """Neutral 3D volume mesh and semantic surface topology."""

    solid: FingertipSolid
    nodes: dict[int, VolumeNode]
    tetrahedra: tuple[Tetrahedron, ...]
    surface_triangles: dict[str, tuple[SurfaceTriangle, ...]]
    volume_element_ids: dict[str, tuple[int, ...]]
    settings: VolumeMeshSettings
    quality: VolumeMeshQuality
    validation: VolumeMeshValidation
    gmsh_version: str

    @property
    def parameters(self) -> FingertipParameters:
        """Return morphology parameters carried by the solid."""
        return self.solid.parameters

    @property
    def morphology_fingerprint(self) -> str:
        """Return the source solid fingerprint."""
        return self.solid.morphology_fingerprint

    @property
    def volume_mm3(self) -> float:
        """Return the sum of positive tetrahedron volumes."""
        return self.quality.mesh_volume_mm3

    @property
    def semantic_surface_tags(self) -> tuple[str, ...]:
        """Return all surface labels present in the mesh."""
        return tuple(sorted(self.surface_triangles))

    def surface(self, tag: str) -> tuple[SurfaceTriangle, ...]:
        """Return one semantic surface family."""
        try:
            return self.surface_triangles[tag]
        except KeyError as exc:
            raise KeyError(f"unknown semantic surface tag: {tag!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology and acceptance diagnostics without solver objects."""
        return {
            "gmsh_version": self.gmsh_version,
            "morphology_fingerprint": self.morphology_fingerprint,
            "parameters": asdict(self.parameters),
            "settings": asdict(self.settings),
            "nodes": {str(k): asdict(v) for k, v in sorted(self.nodes.items())},
            "tetrahedra": [asdict(value) for value in self.tetrahedra],
            "surface_triangles": {
                tag: [asdict(value) for value in triangles]
                for tag, triangles in sorted(self.surface_triangles.items())
            },
            "volume_element_ids": self.volume_element_ids,
            "quality": asdict(self.quality),
            "validation": asdict(self.validation),
        }


__all__ = [
    "FingertipVolumeMesh",
    "SurfaceTriangle",
    "Tetrahedron",
    "VolumeMeshSettings",
    "VolumeMeshQuality",
    "VolumeMeshTier",
    "VolumeMeshValidation",
    "VolumeNode",
    "volume_mesh_settings_for_tier",
]
