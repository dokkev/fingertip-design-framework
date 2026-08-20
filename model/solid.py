"""Semantic 3D solid construction from the authoritative fingertip section."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

from shapely.geometry import MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from model.fingertip_model import FingertipModel, PolygonalGeometry
from model.fingertip_parameters import (
    FingertipParameters,
    fingertip_parameters_fingerprint,
)


DEFAULT_EXTRUSION_DEPTH_MM = 11.0

SolidSurfaceKind = Literal[
    "support",
    "outer_compliant",
    "void",
    "contact",
    "longitudinal_end",
    "rigid_outer",
]


@dataclass(frozen=True)
class SolidSurfaceDefinition:
    """One named 3D surface family and its 2D semantic source."""

    name: str
    kind: SolidSurfaceKind
    material_region: Literal["pad", "rigid_carrier", "both"]
    source_geometry: BaseGeometry | None = None


@dataclass(frozen=True)
class FingertipSolid:
    """Watertight constant-depth compliant-pad solid with semantic surfaces.

    This object contains geometry and provenance only.  It deliberately does
    not contain FEA mesh nodes or solver state; volume meshing is a separate
    operation in :mod:`mesh.volume.mesh`.
    """

    parameters: FingertipParameters
    pad_geometry: PolygonalGeometry
    rigid_geometry: PolygonalGeometry
    material_geometry: PolygonalGeometry
    z_min_mm: float
    z_max_mm: float
    surfaces: tuple[SolidSurfaceDefinition, ...]
    morphology_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, FingertipParameters):
            raise TypeError("parameters must be FingertipParameters")
        if not math.isfinite(self.z_min_mm):
            raise ValueError("z_min_mm must be finite")
        if not math.isfinite(self.z_max_mm):
            raise ValueError("z_max_mm must be finite")
        if self.z_min_mm >= self.z_max_mm:
            raise ValueError("z_min_mm must be smaller than z_max_mm")
        if not math.isclose(
            self.z_max_mm - self.z_min_mm,
            DEFAULT_EXTRUSION_DEPTH_MM,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("the representative solid cell must be exactly 11 mm")
        if self.pad_geometry.is_empty or not self.pad_geometry.is_valid:
            raise ValueError(
                "pad_geometry must be a non-empty valid polygonal geometry"
            )
        if self.rigid_geometry.is_empty or not self.rigid_geometry.is_valid:
            raise ValueError(
                "rigid_geometry must be a non-empty valid polygonal geometry"
            )
        if self.material_geometry.is_empty or not self.material_geometry.is_valid:
            raise ValueError(
                "material_geometry must be a non-empty valid polygonal geometry"
            )
        if self.pad_geometry.intersection(self.rigid_geometry).area > self.parameters.geometry_tolerance:
            raise ValueError("pad and rigid solid regions overlap")
        if self.material_geometry.symmetric_difference(
            self.pad_geometry
        ).area > self.parameters.geometry_tolerance:
            raise ValueError("material geometry is not the compliant-pad solid")
        names = tuple(surface.name for surface in self.surfaces)
        if len(names) != len(set(names)):
            raise ValueError("solid surface names must be unique")
        if self.morphology_fingerprint != self._fingerprint(
            self.parameters,
            self.pad_geometry,
            self.rigid_geometry,
            self.material_geometry,
            self.z_min_mm,
            self.z_max_mm,
        ):
            raise ValueError("morphology_fingerprint does not match solid geometry")

    @property
    def extrusion_depth_mm(self) -> float:
        """Return the fixed representative longitudinal cell depth."""
        return self.z_max_mm - self.z_min_mm

    @property
    def volume_mm3(self) -> float:
        """Return the exact analytical material volume."""
        return float(self.material_geometry.area * self.extrusion_depth_mm)

    @property
    def pad_volume_mm3(self) -> float:
        """Return the analytical compliant-pad volume."""
        return float(self.pad_geometry.area * self.extrusion_depth_mm)

    @property
    def rigid_volume_mm3(self) -> float:
        """Return zero: the rigid link is support metadata, not a 3D solid."""
        return 0.0

    @property
    def surface_names(self) -> tuple[str, ...]:
        """Return stable semantic surface labels."""
        return tuple(surface.name for surface in self.surfaces)

    @property
    def watertight(self) -> bool:
        """Return the analytical closed-volume gate result."""
        return bool(self.closed_volume_gate["passed"])

    @property
    def closed_volume_gate(self) -> dict[str, bool]:
        """Return fail-closed evidence for the semantic extrusion shell.

        The semantic solid is a constant-depth extrusion of a valid polygonal
        material region.  Every exterior or hole ring therefore contributes a
        bottom cap, a top cap, and a two-sided lateral shell.  This gate keeps
        that closure claim explicit; the generated tetrahedral mesh performs
        the corresponding triangle-edge incidence check independently.
        """
        polygonal = True
        finite_boundary = True
        nonzero_area = True
        ring_count = 0
        geometries = (
            self.material_geometry.geoms
            if isinstance(self.material_geometry, MultiPolygon)
            else (self.material_geometry,)
        )
        for geometry in geometries:
            if geometry.is_empty or not geometry.is_valid or not isinstance(geometry, Polygon):
                polygonal = False
                continue
            ring_count += 1 + len(geometry.interiors)
            for ring in (geometry.exterior, *geometry.interiors):
                coordinates = tuple(ring.coords)
                finite_boundary &= all(
                    math.isfinite(float(value))
                    for point in coordinates
                    for value in point[:2]
                )
                nonzero_area &= len(coordinates) >= 4 and abs(float(ring.length)) > 0.0
        checks = {
            "polygonal_material": polygonal,
            "finite_boundary": finite_boundary,
            "nonzero_area": nonzero_area,
            "positive_extrusion_depth": self.extrusion_depth_mm > 0.0,
            "valid_material_geometry": bool(self.material_geometry.is_valid),
            "closed_extrusion_shell": polygonal and finite_boundary and nonzero_area,
        }
        checks["passed"] = all(checks.values()) and ring_count > 0 and self.volume_mm3 > 0.0
        return checks

    def cross_section_at(self, z_mm: float) -> PolygonalGeometry:
        """Return the authoritative section at one longitudinal coordinate."""
        z = float(z_mm)
        if not self.z_min_mm <= z <= self.z_max_mm:
            raise ValueError("z_mm lies outside the solid cell")
        return self.material_geometry

    @staticmethod
    def _fingerprint(
        parameters: FingertipParameters,
        pad_geometry: PolygonalGeometry,
        rigid_geometry: PolygonalGeometry,
        material_geometry: PolygonalGeometry,
        z_min_mm: float,
        z_max_mm: float,
    ) -> str:
        """Fingerprint the exact constructed solid and its physical source.

        The parameter portion uses the canonical physical morphology
        fingerprint.  WKB is retained here intentionally as exact solid
        provenance for a particular polygonal representation; callers that
        compare physical morphologies independently of sampling should use
        ``fingertip_parameters_fingerprint``.
        """
        payload = {
            "physical_morphology": fingertip_parameters_fingerprint(parameters),
            "pad_wkb_hex": pad_geometry.wkb_hex,
            "rigid_wkb_hex": rigid_geometry.wkb_hex,
            "material_wkb_hex": material_geometry.wkb_hex,
            "z_min_mm": float(z_min_mm),
            "z_max_mm": float(z_max_mm),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _surface_definitions(model: FingertipModel) -> tuple[SolidSurfaceDefinition, ...]:
    boundaries = model.boundaries.segments
    interface_segments = tuple(
        geometry
        for geometry in model.interface_definition.geometry.geoms
        if geometry.geom_type == "LineString"
    )
    if len(interface_segments) != 2:
        raise ValueError("authoritative bonded interface must contain two line segments")
    left_interface, right_interface = sorted(
        interface_segments, key=lambda geometry: geometry.centroid.x
    )
    return (
        SolidSurfaceDefinition(
            "support_bond_left", "support", "pad", left_interface
        ),
        SolidSurfaceDefinition(
            "support_bond_right", "support", "pad", right_interface
        ),
        SolidSurfaceDefinition(
            "outer_compliant_left", "outer_compliant", "pad", boundaries["pad_outer_left"].geometry
        ),
        SolidSurfaceDefinition(
            "outer_compliant_right", "outer_compliant", "pad", boundaries["pad_outer_right"].geometry
        ),
        SolidSurfaceDefinition(
            "outer_compliant_arc", "outer_compliant", "pad", boundaries["pad_outer_arc"].geometry
        ),
        SolidSurfaceDefinition(
            "outer_compliant_other", "outer_compliant", "pad", None
        ),
        SolidSurfaceDefinition(
            "void_left", "void", "pad", boundaries["pad_cutout_left"].geometry
        ),
        SolidSurfaceDefinition(
            "void_right", "void", "pad", boundaries["pad_cutout_right"].geometry
        ),
        SolidSurfaceDefinition(
            "void_bottom", "void", "pad", boundaries["pad_cutout_bottom"].geometry
        ),
        SolidSurfaceDefinition(
            "longitudinal_end_minus", "longitudinal_end", "both", None
        ),
        SolidSurfaceDefinition(
            "longitudinal_end_plus", "longitudinal_end", "both", None
        ),
    )


def build_fingertip_solid(
    model: FingertipModel,
    *,
    extrusion_depth_mm: float = DEFAULT_EXTRUSION_DEPTH_MM,
) -> FingertipSolid:
    """Build the 11 mm 3D solid directly from ``FingertipModel`` geometry."""
    if not isinstance(model, FingertipModel):
        raise TypeError("model must be FingertipModel")
    if not math.isclose(
        float(extrusion_depth_mm), DEFAULT_EXTRUSION_DEPTH_MM, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("extrusion_depth_mm must be the existing 11 mm cell depth")
    depth = float(extrusion_depth_mm)
    pad = model.pad_material_geometry
    rigid = model.link_geometry
    material = model.material_geometry
    z_min = -0.5 * depth
    z_max = 0.5 * depth
    return FingertipSolid(
        parameters=model.parameters,
        pad_geometry=pad,
        rigid_geometry=rigid,
        material_geometry=pad,
        z_min_mm=z_min,
        z_max_mm=z_max,
        surfaces=_surface_definitions(model),
        morphology_fingerprint=FingertipSolid._fingerprint(
            model.parameters, pad, rigid, pad, z_min, z_max
        ),
    )


__all__ = [
    "DEFAULT_EXTRUSION_DEPTH_MM",
    "FingertipSolid",
    "SolidSurfaceDefinition",
    "build_fingertip_solid",
]
