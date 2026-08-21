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
from lumo.contact import (
    FirstContactResult,
    FirstContactSettings,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
    unintended_boundary_clearance_mm,
)
from lumo.physics import (
    IndentationResult,
    IndentationSettings,
    NewtonSettings,
    RigidIndenter3D,
    prepare_fingertip_mesh,
    solve_fingertip_indentation,
)
from lumo.mesh.rigid.object import make_sphere_mesh
from lumo.mesh.rigid.carrier import make_distal_phalanx_mesh
from lumo.mesh.volume.mesh import generate_volume_mesh
from lumo.mesh.volume.contracts import volume_mesh_settings_for_tier
from lumo.finger import Fingertip, FingertipParameters
from lumo.mechanics_contract import MechanicsContract


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


def _direct_execution_settings(
    contract: MechanicsContract,
    parameters: FingertipParameters,
    *,
    device: str,
    travel_mm: float,
    support_vertex_indices: tuple[int, ...],
) -> tuple[FirstContactSettings, NewtonSettings, IndentationSettings]:
    """Translate the complete production mechanics contract for direct replay."""

    if not isinstance(contract, MechanicsContract):
        raise TypeError("contract must be a MechanicsContract")
    viscoelastic = parameters.viscoelastic
    newton_settings = NewtonSettings(
        device=device,
        gravity=0.0,
        dt=contract.dt_s,
        steps=1,
        iterations=contract.vbd_iterations,
        deterministic_mode=contract.deterministic_mode,
        density=viscoelastic.density_kg_m3,
        k_mu=viscoelastic.k_mu_pa,
        k_lambda=viscoelastic.k_lambda_pa,
        k_damp=viscoelastic.k_damp,
        fixed_vertex_indices=support_vertex_indices,
    )
    indentation_settings = IndentationSettings(
        travel_mm=travel_mm,
        load_steps=load_steps_for_increment(
            travel_mm,
            max_increment_mm=contract.max_load_increment_mm,
        ),
        soft_contact_margin_mm=contract.soft_contact_margin_mm,
        rigid_sdf_target_voxel_mm=contract.rigid_sdf_target_voxel_mm,
        soft_contact_ke=contract.soft_contact_ke,
        soft_contact_kd=contract.soft_contact_kd,
        soft_contact_mu=contract.soft_contact_mu,
    )
    return contract.first_contact, newton_settings, indentation_settings


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
    dt_s: float = SEARCH_DT_S
    carrier_contact: bool = False
    mechanics_contract: MechanicsContract | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": self.parameters.__dict__.copy(),
            "radius_mm": self.radius_mm,
            "travel_mm": self.travel_mm,
            "initial_gap_mm": self.initial_gap_mm,
            "search_contract": (
                self.mechanics_contract.to_dict()
                if self.mechanics_contract is not None
                else {
                "sphere_subdivisions": self.sphere_subdivisions,
                "max_load_increment_mm": self.max_load_increment_mm,
                "load_steps": load_steps_for_increment(
                    self.travel_mm,
                    max_increment_mm=self.max_load_increment_mm,
                ),
                "iterations": self.vbd_iterations,
                "dt_s": self.dt_s,
                "maximum_indentation_speed_mm_s": (
                    self.max_load_increment_mm / self.dt_s
                ),
                "soft_contact_margin_mm": SEARCH_SOFT_CONTACT_MARGIN_MM,
                "soft_contact_ke": SEARCH_SOFT_CONTACT_KE,
                "soft_contact_kd": SEARCH_SOFT_CONTACT_KD,
                "carrier_contact": self.carrier_contact,
                }
            ),
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
    dt_s: float = SEARCH_DT_S,
    carrier_contact: bool = False,
    mechanics_contract: MechanicsContract | None = None,
) -> MultiLocationContactResult:
    """Run one direct mechanics path at the requested arc locations.

    ``mechanics_contract`` is the authoritative route for same-contract
    production equivalence checks. The scalar arguments remain only for the
    frozen historical validation callers.
    """

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

    volume_mesh = generate_volume_mesh(
        fingertip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    if sphere_subdivisions < 1:
        raise ValueError("sphere_subdivisions must be positive")
    if not np.isfinite(max_load_increment_mm) or max_load_increment_mm <= 0.0:
        raise ValueError("max_load_increment_mm must be finite and positive")
    if vbd_iterations < 1:
        raise ValueError("vbd_iterations must be positive")
    selected_dt_s = float(dt_s)
    if not np.isfinite(selected_dt_s) or selected_dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if mechanics_contract is not None:
        if not isinstance(mechanics_contract, MechanicsContract):
            raise TypeError("mechanics_contract must be a MechanicsContract")
        sphere_subdivisions = mechanics_contract.sphere_subdivisions
        max_load_increment_mm = mechanics_contract.max_load_increment_mm
        vbd_iterations = mechanics_contract.vbd_iterations
        selected_dt_s = mechanics_contract.dt_s
    sphere_mesh = make_sphere_mesh(radius, subdivisions=sphere_subdivisions)
    carrier_mesh = make_distal_phalanx_mesh(volume_mesh.solid) if carrier_contact else None
    contact_surface = make_outer_compliant_surface(volume_mesh.solid)
    if mechanics_contract is not None:
        contact_settings, mechanics_settings, indentation_settings = (
            _direct_execution_settings(
                mechanics_contract,
                selected_parameters,
                device=device,
                travel_mm=travel,
                support_vertex_indices=prepared.support_vertex_indices,
            )
        )
    else:
        contact_settings = FirstContactSettings(
            coarse_step_mm=0.25,
            tolerance_mm=1.0e-3,
            spawn_clearance_mm=0.05,
            max_travel_mm=20.0,
        )
        viscoelastic = selected_parameters.viscoelastic
        mechanics_settings = NewtonSettings(
            device=device,
            gravity=0.0,
            dt=selected_dt_s,
            steps=1,
            iterations=vbd_iterations,
            density=viscoelastic.density_kg_m3,
            k_mu=viscoelastic.k_mu_pa,
            k_lambda=viscoelastic.k_lambda_pa,
            k_damp=viscoelastic.k_damp,
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
            location,
            radius_mm=radius,
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
        boundary_clearance = unintended_boundary_clearance_mm(
            fingertip.geometry,
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
        dt_s=selected_dt_s,
        carrier_contact=carrier_contact,
        mechanics_contract=mechanics_contract,
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
