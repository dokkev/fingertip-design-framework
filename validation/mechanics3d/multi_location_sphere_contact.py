"""Deterministic multi-location sphere contact for the LUMO 3D MVP.

This module is an orchestration-level validation path. It reuses the neutral
contact search and Newton indentation APIs; it does not define a second
contact model or a second mechanics configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import Point

from contact import (
    FirstContactResult,
    FirstContactSettings,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
)
from mechanics3d import (
    IndentationResult,
    IndentationSettings,
    Mechanics3DSettings,
    RigidIndenter3D,
    prepare_fingertip_mechanics_mesh,
    solve_fingertip_indentation,
)
from mesh.rigid_object import RigidObjectMesh, make_sphere_mesh
from mesh.rigid_carrier import make_distal_phalanx_mesh
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters


SEARCH_SPHERE_SUBDIVISIONS = 3
SEARCH_MAX_LOAD_INCREMENT_MM = 0.05
SEARCH_VBD_ITERATIONS = 10
SEARCH_DT_S = 1.0e-3
SEARCH_SOFT_CONTACT_MARGIN_MM = 0.02
SEARCH_SOFT_CONTACT_KE = 1.0e3
SEARCH_SOFT_CONTACT_KD = 10.0
DEFAULT_LOCATION_U = (0.25, 0.50, 0.75)
DEFAULT_RADIUS_MM = 5.0
DEFAULT_TRAVEL_MM = 1.5
VALIDATION_MAX_LOAD_INCREMENT_MM = 0.025
VALIDATION_VBD_ITERATIONS = 20


def load_steps_for_increment(
    travel_mm: float,
    *,
    max_increment_mm: float = SEARCH_MAX_LOAD_INCREMENT_MM,
) -> int:
    """Return the frozen SEARCH load-step count for one travel distance."""

    travel = float(travel_mm)
    if not np.isfinite(travel) or travel <= 0.0:
        raise ValueError("travel_mm must be finite and positive")
    increment = float(max_increment_mm)
    if not np.isfinite(increment) or increment <= 0.0:
        raise ValueError("max_increment_mm must be finite and positive")
    return max(1, int(math.ceil(travel / increment)))


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices)[np.asarray(tetrahedra)]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


def _unintended_boundary_clearance_mm(
    fingertip: Fingertip,
    sphere_mesh: RigidObjectMesh,
    alignment,
    first_contact: FirstContactResult,
    *,
    samples: int = 256,
) -> float:
    """Measure clearance to the two non-arc external side boundaries.

    The first-contact predicate is intentionally the authoritative outer arc.
    This diagnostic checks that the sphere does not reach any other semantic
    2D boundary earlier along the same approach path. Longitudinal end-cap
    clearance is checked separately by the fixed 11 mm cell depth.
    """

    radius = float(alignment.radius_mm)
    boundaries = fingertip.geometry.boundaries
    other_segments = tuple(
        segment.geometry
        for name, segment in boundaries.segments.items()
        if name != "pad_outer_arc"
    )
    reference = np.asarray(alignment.nominal_pose.translation_mm, dtype=float)
    direction = np.asarray(alignment.approach_direction, dtype=float)
    values: list[float] = []
    for travel in np.linspace(0.0, first_contact.travel_to_contact_mm, samples):
        center = reference + float(travel) * direction
        point = Point(float(center[0]), float(center[1]))
        values.extend(float(segment.distance(point) - radius) for segment in other_segments)
    return float(min(values))


@dataclass(frozen=True)
class MultiLocationContactCase:
    """One fully solved location with geometry and mechanics provenance."""

    normalized_location: float
    alignment: object
    first_contact: FirstContactResult
    indenter: RigidIndenter3D
    indentation: IndentationResult
    unintended_boundary_clearance_mm: float
    mechanics_artifact_path: str | None = None
    mechanics_artifact_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        mechanics = self.indentation.mechanics_result
        six_volumes = _six_volumes(mechanics.deformed_vertices, mechanics.tetrahedra)
        support = np.asarray(self.indentation.mechanics_result.deformed_vertices)[
            list(self.indentation.diagnostics.get("support_vertex_indices", ()))
        ] if self.indentation.diagnostics.get("support_vertex_indices") else np.empty((0, 3))
        return {
            "normalized_location": self.normalized_location,
            "target_point_mm": list(self.alignment.target_point_mm),
            "outward_normal": list(self.alignment.outward_normal),
            "approach_direction": list(self.alignment.approach_direction),
            "nominal_pose_mm": list(self.alignment.nominal_pose.translation_mm),
            "clear_pose_mm": list(self.first_contact.clear_pose.translation_mm),
            "hit_pose_mm": list(self.first_contact.hit_pose.translation_mm),
            "contact_pose_mm": list(self.first_contact.contact_pose.translation_mm),
            "spawn_pose_mm": list(self.first_contact.spawn_pose.translation_mm),
            "first_contact_travel_mm": self.first_contact.travel_to_contact_mm,
            "first_contact_bracket_width_mm": self.first_contact.bracket_width_mm,
            "spawn_clearance_mm": self.first_contact.spawn_clearance_mm,
            "post_contact_travel_mm": self.indentation.diagnostics["post_contact_travel_mm"],
            "final_pose_error_mm": self.indentation.diagnostics["final_pose_error_mm"],
            "max_soft_contact_count": self.indentation.diagnostics["max_soft_contact_count"],
            "max_soft_contact_overflow": self.indentation.diagnostics["max_soft_contact_overflow"],
            "max_rigid_contact_overflow": self.indentation.diagnostics["max_rigid_contact_overflow"],
            "finite_deformation": bool(np.all(np.isfinite(mechanics.deformed_vertices))),
            "inverted_tetrahedra": int(np.count_nonzero(six_volumes <= 0.0)),
            "min_six_volume": float(np.min(six_volumes)),
            "max_displacement_mm": float(
                np.max(np.linalg.norm(mechanics.displacement, axis=1))
            ),
            "max_support_displacement_mm": float(
                self.indentation.diagnostics["max_support_displacement_mm"]
            ),
            "unintended_boundary_clearance_mm": self.unintended_boundary_clearance_mm,
            "cell_end_clearance_mm": 5.5 - float(self.alignment.radius_mm),
            "mechanics_diagnostics": dict(self.indentation.diagnostics),
            "mechanics_artifact_path": self.mechanics_artifact_path,
            "mechanics_artifact_sha256": self.mechanics_artifact_sha256,
        }


@dataclass(frozen=True)
class MultiLocationContactResult:
    """Shared mesh plus all deterministic contact-location results."""

    parameters: FingertipParameters
    radius_mm: float
    travel_mm: float
    initial_gap_mm: float
    locations: tuple[MultiLocationContactCase, ...]
    sphere_subdivisions: int = SEARCH_SPHERE_SUBDIVISIONS
    max_load_increment_mm: float = SEARCH_MAX_LOAD_INCREMENT_MM
    vbd_iterations: int = SEARCH_VBD_ITERATIONS
    carrier_contact: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": self.parameters.__dict__.copy(),
            "radius_mm": self.radius_mm,
            "travel_mm": self.travel_mm,
            "initial_gap_mm": self.initial_gap_mm,
            "search_contract": {
                "sphere_subdivisions": self.sphere_subdivisions,
                "max_load_increment_mm": self.max_load_increment_mm,
                "load_steps": load_steps_for_increment(
                    self.travel_mm,
                    max_increment_mm=self.max_load_increment_mm,
                ),
                "iterations": self.vbd_iterations,
                "dt_s": SEARCH_DT_S,
                "soft_contact_margin_mm": SEARCH_SOFT_CONTACT_MARGIN_MM,
                "soft_contact_ke": SEARCH_SOFT_CONTACT_KE,
                "soft_contact_kd": SEARCH_SOFT_CONTACT_KD,
                "carrier_contact": self.carrier_contact,
            },
            "locations": [case.to_dict() for case in self.locations],
        }


def run_multi_location_sphere_contact(
    *,
    parameters: FingertipParameters | None = None,
    device: str = "cuda:0",
    radius_mm: float = DEFAULT_RADIUS_MM,
    travel_mm: float = DEFAULT_TRAVEL_MM,
    initial_gap_mm: float = 0.25,
    normalized_locations: Iterable[float] = DEFAULT_LOCATION_U,
    artifact_dir: str | Path | None = None,
    sphere_subdivisions: int = SEARCH_SPHERE_SUBDIVISIONS,
    max_load_increment_mm: float = SEARCH_MAX_LOAD_INCREMENT_MM,
    vbd_iterations: int = SEARCH_VBD_ITERATIONS,
    carrier_contact: bool = False,
) -> MultiLocationContactResult:
    """Run the frozen SEARCH mechanics contract at three arc locations."""

    selected_parameters = parameters or FingertipParameters()
    fingertip = Fingertip(selected_parameters)
    locations = tuple(float(value) for value in normalized_locations)
    if not locations:
        raise ValueError("normalized_locations must not be empty")
    if len(set(locations)) != len(locations):
        raise ValueError("normalized_locations must be unique")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in locations):
        raise ValueError("normalized_locations must lie in [0, 1]")
    radius = float(radius_mm)
    travel = float(travel_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    if not np.isfinite(travel) or travel <= 0.0:
        raise ValueError("travel_mm must be finite and positive")

    volume_mesh = fingertip.volume_mesh(volume_mesh_settings_for_tier("search"))
    prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
    if sphere_subdivisions < 1:
        raise ValueError("sphere_subdivisions must be positive")
    if not np.isfinite(max_load_increment_mm) or max_load_increment_mm <= 0.0:
        raise ValueError("max_load_increment_mm must be finite and positive")
    if vbd_iterations < 1:
        raise ValueError("vbd_iterations must be positive")
    sphere_mesh = make_sphere_mesh(radius, subdivisions=sphere_subdivisions)
    carrier_mesh = make_distal_phalanx_mesh(volume_mesh.solid) if carrier_contact else None
    contact_surface = make_outer_compliant_surface(volume_mesh.solid)
    contact_settings = FirstContactSettings(
        coarse_step_mm=0.25,
        tolerance_mm=1.0e-3,
        spawn_clearance_mm=0.05,
        max_travel_mm=20.0,
    )
    mechanics_settings = Mechanics3DSettings(
        device=device,
        gravity=0.0,
        dt=SEARCH_DT_S,
        steps=1,
        iterations=vbd_iterations,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )
    indentation_settings = IndentationSettings(
        travel_mm=travel,
        load_steps=load_steps_for_increment(
            travel,
            max_increment_mm=max_load_increment_mm,
        ),
        soft_contact_margin_mm=SEARCH_SOFT_CONTACT_MARGIN_MM,
        soft_contact_ke=SEARCH_SOFT_CONTACT_KE,
        soft_contact_kd=SEARCH_SOFT_CONTACT_KD,
    )
    selected_artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
    if selected_artifact_dir is not None:
        selected_artifact_dir.mkdir(parents=True, exist_ok=True)

    cases: list[MultiLocationContactCase] = []
    for location in locations:
        alignment = sphere_alignment_at_normalized_location(
            fingertip.geometry,
            sphere_mesh,
            location,
            initial_gap_mm=initial_gap_mm,
        )
        if intersects(contact_surface, sphere_mesh, alignment.nominal_pose):
            raise RuntimeError(f"location u={location:g} nominal pose is not collision-free")
        first_contact = find_first_contact(
            contact_surface,
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
            contact_settings,
        )
        if intersects(contact_surface, sphere_mesh, first_contact.spawn_pose):
            raise RuntimeError(f"location u={location:g} spawn pose is not collision-free")
        boundary_clearance = _unintended_boundary_clearance_mm(
            fingertip,
            sphere_mesh,
            alignment,
            first_contact,
        )
        if boundary_clearance <= 0.0:
            raise RuntimeError(
                f"location u={location:g} reaches an unintended external boundary "
                f"before arc contact: clearance={boundary_clearance:g} mm"
            )
        indenter = RigidIndenter3D(
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
        )
        indentation = solve_fingertip_indentation(
            prepared,
            indenter,
            mechanics_settings,
            indentation_settings,
            first_contact=first_contact,
            rigid_carrier_mesh=carrier_mesh,
        )
        expected_pose = first_contact.pose_at_post_contact_travel(travel)
        final_pose_error = float(
            np.linalg.norm(
                np.asarray(indentation.final_indenter_pose.translation_mm)
                - np.asarray(expected_pose.translation_mm)
            )
        )
        displacement = indentation.mechanics_result.displacement
        support_displacement = displacement[list(prepared.support_vertex_indices)]
        diagnostics = dict(indentation.diagnostics)
        diagnostics.update(
            {
                "final_pose_error_mm": final_pose_error,
                "max_support_displacement_mm": float(
                    np.max(np.linalg.norm(support_displacement, axis=1))
                ),
                "support_vertex_indices": prepared.support_vertex_indices,
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "carrier_contact_enabled": carrier_contact,
            }
        )
        indentation = IndentationResult(
            mechanics_result=indentation.mechanics_result,
            final_indenter_pose=indentation.final_indenter_pose,
            diagnostics=diagnostics,
        )
        artifact_path: str | None = None
        artifact_sha256: str | None = None
        if selected_artifact_dir is not None:
            artifact_file = selected_artifact_dir / f"location_u_{location:.3f}.npz"
            arrays = {
                "rest_vertices_mm": np.asarray(
                    indentation.mechanics_result.rest_vertices, dtype=np.float32
                ),
                "deformed_vertices_mm": np.asarray(
                    indentation.mechanics_result.deformed_vertices, dtype=np.float32
                ),
                "tetrahedra": np.asarray(
                    indentation.mechanics_result.tetrahedra, dtype=np.int32
                ),
                "source_node_ids": np.asarray(prepared.source_node_ids, dtype=np.int64),
            }
            arrays.update(
                {
                    f"surface_{tag}": np.asarray(triangles, dtype=np.int32)
                    for tag, triangles in prepared.surface_triangles.items()
                }
            )
            arrays["carrier_contact_vertex_indices"] = np.asarray(
                indentation.diagnostics.get("carrier_contact_vertex_indices", ()),
                dtype=np.int64,
            )
            arrays["carrier_contact_source_node_ids"] = np.asarray(
                [
                    prepared.source_node_ids[index]
                    for index in indentation.diagnostics.get(
                        "carrier_contact_vertex_indices", ()
                    )
                ],
                dtype=np.int64,
            )
            np.savez_compressed(artifact_file, **arrays)
            artifact_sha256 = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
            artifact_path = str(artifact_file)
        cases.append(
            MultiLocationContactCase(
                normalized_location=location,
                alignment=alignment,
                first_contact=first_contact,
                indenter=indenter,
                indentation=indentation,
                unintended_boundary_clearance_mm=boundary_clearance,
                mechanics_artifact_path=artifact_path,
                mechanics_artifact_sha256=artifact_sha256,
            )
        )
    return MultiLocationContactResult(
        parameters=selected_parameters,
        radius_mm=radius,
        travel_mm=travel,
        initial_gap_mm=float(initial_gap_mm),
        locations=tuple(cases),
        sphere_subdivisions=sphere_subdivisions,
        max_load_increment_mm=float(max_load_increment_mm),
        vbd_iterations=vbd_iterations,
        carrier_contact=carrier_contact,
    )


__all__ = [
    "MultiLocationContactCase",
    "MultiLocationContactResult",
    "SEARCH_MAX_LOAD_INCREMENT_MM",
    "SEARCH_SPHERE_SUBDIVISIONS",
    "SEARCH_VBD_ITERATIONS",
    "VALIDATION_MAX_LOAD_INCREMENT_MM",
    "VALIDATION_VBD_ITERATIONS",
    "run_multi_location_sphere_contact",
    "load_steps_for_increment",
]
