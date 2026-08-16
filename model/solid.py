"""Semantic 3D solid construction from the authoritative fingertip section."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

from shapely.geometry import MultiLineString
from shapely.geometry.base import BaseGeometry

from model.fingertip_model import FingertipModel, PolygonalGeometry
from model.fingertip_parameters import FingertipParameters


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
    """Watertight constant-depth solid with explicit semantic surfaces.

    This object contains geometry and provenance only.  It deliberately does
    not contain FEA mesh nodes or solver state; volume meshing is a separate
    operation in :mod:`mesh.volume3d`.
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
        for name, value in (
            ("z_min_mm", self.z_min_mm),
            ("z_max_mm", self.z_max_mm),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.z_min_mm >= self.z_max_mm:
            raise ValueError("z_min_mm must be smaller than z_max_mm")
        if not math.isclose(
            self.z_max_mm - self.z_min_mm,
            DEFAULT_EXTRUSION_DEPTH_MM,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("the representative solid cell must be exactly 11 mm")
        for name, geometry in (
            ("pad_geometry", self.pad_geometry),
            ("rigid_geometry", self.rigid_geometry),
            ("material_geometry", self.material_geometry),
        ):
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError(f"{name} must be a non-empty valid polygonal geometry")
        if self.pad_geometry.intersection(self.rigid_geometry).area > self.parameters.geometry_tolerance:
            raise ValueError("pad and rigid solid regions overlap")
        if self.material_geometry.symmetric_difference(
            self.pad_geometry.union(self.rigid_geometry)
        ).area > self.parameters.geometry_tolerance:
            raise ValueError("material geometry is not the pad/rigid union")
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
        """Return the analytical rigid-carrier volume."""
        return float(self.rigid_geometry.area * self.extrusion_depth_mm)

    @property
    def surface_names(self) -> tuple[str, ...]:
        """Return stable semantic surface labels."""
        return tuple(surface.name for surface in self.surfaces)

    @property
    def watertight(self) -> bool:
        """Return the analytical extrusion topology acceptance result."""
        return self.volume_mm3 > 0.0 and self.material_geometry.is_valid

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
        payload = {
            "parameters": asdict(parameters),
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
    return (
        SolidSurfaceDefinition(
            "support_bond_left", "support", "pad", boundaries["pad_bond_left"].geometry
        ),
        SolidSurfaceDefinition(
            "support_bond_right", "support", "pad", boundaries["pad_bond_right"].geometry
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
            "contact_left", "contact", "rigid_carrier", boundaries["stem_left"].geometry
        ),
        SolidSurfaceDefinition(
            "contact_right", "contact", "rigid_carrier", boundaries["stem_right"].geometry
        ),
        SolidSurfaceDefinition(
            "contact_bottom", "contact", "rigid_carrier", boundaries["stem_bottom"].geometry
        ),
        SolidSurfaceDefinition(
            "rigid_outer", "rigid_outer", "rigid_carrier", None
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
        material_geometry=material,
        z_min_mm=z_min,
        z_max_mm=z_max,
        surfaces=_surface_definitions(model),
        morphology_fingerprint=FingertipSolid._fingerprint(
            model.parameters, pad, rigid, material, z_min, z_max
        ),
    )


__all__ = [
    "DEFAULT_EXTRUSION_DEPTH_MM",
    "FingertipSolid",
    "SolidSurfaceDefinition",
    "build_fingertip_solid",
]
