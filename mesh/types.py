"""Kratos-independent data contracts for a meshed LIT fingertip."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

from model.fingertip_parameters import FingertipParameters

MeshLevel = Literal["medium", "fine"]
MeshDomain = Literal["pad", "rigid_carrier"]


class InvalidMeshSettings(ValueError):
    """Raised when a mesh setting is non-finite or geometrically unusable."""


@dataclass(frozen=True)
class MeshSettings:
    """Gmsh sizing and validation settings, with every length in millimeters."""

    level: MeshLevel
    bulk_target_size_mm: float
    contact_boundary_target_size_mm: float
    classification_tolerance_mm: float = 1.0e-7
    contact_refinement_distance_mm: float = 1.5
    minimum_angle_target_degrees: float = 15.0

    def __post_init__(self) -> None:
        values = {
            "bulk_target_size_mm": self.bulk_target_size_mm,
            "contact_boundary_target_size_mm": (
                self.contact_boundary_target_size_mm
            ),
            "classification_tolerance_mm": self.classification_tolerance_mm,
            "contact_refinement_distance_mm": self.contact_refinement_distance_mm,
            "minimum_angle_target_degrees": self.minimum_angle_target_degrees,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise InvalidMeshSettings(f"{name} must be finite and positive")
        if self.level not in ("medium", "fine"):
            raise InvalidMeshSettings("level must be 'medium' or 'fine'")
        if self.contact_boundary_target_size_mm > self.bulk_target_size_mm:
            raise InvalidMeshSettings(
                "contact_boundary_target_size_mm must not exceed "
                "bulk_target_size_mm"
            )
        if not 0.0 < self.minimum_angle_target_degrees < 60.0:
            raise InvalidMeshSettings(
                "minimum_angle_target_degrees must lie between 0 and 60"
            )


def mesh_settings_for_level(level: MeshLevel) -> MeshSettings:
    """Return the common Phase 4M sizing policy for a named mesh level."""
    if level == "medium":
        return MeshSettings(
            level="medium",
            bulk_target_size_mm=0.75,
            contact_boundary_target_size_mm=0.35,
        )
    if level == "fine":
        return MeshSettings(
            level="fine",
            bulk_target_size_mm=0.40,
            contact_boundary_target_size_mm=0.20,
        )
    raise InvalidMeshSettings(f"unsupported mesh level: {level!r}")


@dataclass(frozen=True)
class MeshNode:
    """One globally identified mesh node."""

    id: int
    x_mm: float
    y_mm: float
    domain: MeshDomain


@dataclass(frozen=True)
class T3Element:
    """A counter-clockwise linear triangular element."""

    id: int
    node_ids: tuple[int, int, int]
    domain: MeshDomain


@dataclass(frozen=True)
class BoundaryEdge:
    """An oriented two-node boundary edge with material on its left."""

    node_ids: tuple[int, int]
    domain: MeshDomain


@dataclass(frozen=True)
class MeshedContactPair:
    """Semantic contact pairing carried from ``FingertipModel``."""

    name: str
    pad_boundary_tag: str
    stem_boundary_tag: str
    initial_normal_gap_mm: float
    measured_mesh_gap_mm: float


@dataclass(frozen=True)
class MeshQualityStatistics:
    """Topology, area, and element-shape measurements."""

    node_count: int
    pad_node_count: int
    carrier_node_count: int
    t3_element_count: int
    pad_t3_element_count: int
    carrier_t3_element_count: int
    minimum_triangle_angle_degrees: float
    minimum_triangle_angle_element_id: int
    minimum_triangle_angle_centroid_mm: tuple[float, float]
    maximum_edge_length_mm: float
    pad_mesh_area_mm2: float
    pad_geometry_area_mm2: float
    pad_area_relative_error: float
    carrier_mesh_area_mm2: float
    carrier_geometry_area_mm2: float
    carrier_area_relative_error: float
    orphan_node_count: int
    duplicate_element_count: int
    nonpositive_area_element_count: int


@dataclass(frozen=True)
class MeshValidationReport:
    """Named acceptance checks and any actionable failures."""

    passed: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class FingertipMesh:
    """Complete solver-independent mesh and semantic topology."""

    nodes: dict[int, MeshNode]
    pad_elements: tuple[T3Element, ...]
    carrier_elements: tuple[T3Element, ...]
    domain_node_ids: dict[str, tuple[int, ...]]
    domain_element_ids: dict[str, tuple[int, ...]]
    boundary_edges: dict[str, tuple[BoundaryEdge, ...]]
    contact_pairs: tuple[MeshedContactPair, ...]
    parameters: FingertipParameters
    settings: MeshSettings
    quality: MeshQualityStatistics
    validation: MeshValidationReport
    gmsh_version: str

    @property
    def elements(self) -> tuple[T3Element, ...]:
        """Return pad and carrier elements in deterministic ID order."""
        return tuple(sorted((*self.pad_elements, *self.carrier_elements), key=lambda e: e.id))

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return a deterministic representation suitable for regression tests."""
        node_signature = tuple(
            (node.id, round(node.x_mm, 12), round(node.y_mm, 12), node.domain)
            for node in sorted(self.nodes.values(), key=lambda item: item.id)
        )
        element_signature = tuple(
            (element.id, element.node_ids, element.domain)
            for element in self.elements
        )
        boundary_signature = tuple(
            (tag, tuple(edge.node_ids for edge in edges))
            for tag, edges in sorted(self.boundary_edges.items())
        )
        return node_signature, element_signature, boundary_signature

    def to_dict(self) -> dict[str, Any]:
        """Serialize mesh metadata and validation without solver objects."""
        return {
            "gmsh_version": self.gmsh_version,
            "parameters": asdict(self.parameters),
            "settings": asdict(self.settings),
            "nodes": {
                str(node_id): asdict(node)
                for node_id, node in sorted(self.nodes.items())
            },
            "pad_elements": [asdict(element) for element in self.pad_elements],
            "carrier_elements": [
                asdict(element) for element in self.carrier_elements
            ],
            "domain_node_ids": self.domain_node_ids,
            "domain_element_ids": self.domain_element_ids,
            "boundary_edges": {
                tag: [asdict(edge) for edge in edges]
                for tag, edges in self.boundary_edges.items()
            },
            "contact_pairs": [asdict(pair) for pair in self.contact_pairs],
            "quality": asdict(self.quality),
            "validation": asdict(self.validation),
        }
