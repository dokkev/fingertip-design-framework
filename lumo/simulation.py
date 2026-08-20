"""Concrete LUMO simulation orchestration for reusable morphology state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from contact import (
    CandidateContactError,
    FirstContactResult,
    FirstContactSettings,
    FingertipContactSurface,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
    unintended_boundary_clearance_mm,
)
from contact.sphere_alignment import SphereAlignment
from mesh import volume_mesh_settings_for_tier
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.rigid.object import RigidObjectMesh, make_sphere_mesh
from mesh.volume.contracts import FingertipVolumeMesh
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.state import FingertipVolumeState
from model import Fingertip
from optics.contracts.objects import CarrierOptics
from optics.transport3d import (
    Transport3DResult,
    Transport3DSettings,
    trace_geometry,
)
from optics.transport3d.optix_backend import create_runtime
from optics.optix.runtime import OptixRuntime
from optimization.deformed_state_artifact import (
    build_contact_state_record,
    restore_deformed_optical_state,
    write_mechanics_artifact,
)
from optimization.mechanics_contract import (
    DEFAULT_MECHANICS_CONTRACT,
    MechanicsContract,
)
from optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)
from physics import (
    IndentationCheckpoint,
    IndentationSettings,
    IndentationTrajectoryResult,
    NewtonSettings,
    PreparedFingertipMesh,
    RigidIndenter3D,
    solve_fingertip_indentation_trajectory,
    prepare_fingertip_mesh,
)


LUMO3D_OPTICAL_X_BOUNDS_MM = (-16.0, 16.0)
LUMO3D_OPTICAL_Y_BOUNDS_MM = (-31.0, 4.5)
LUMO3D_OBSERVATION_LEVEL = "FULL_3D native internal transport redistribution proxy"


def lumo_optical_settings() -> Transport3DSettings:
    """Return the frozen FULL_3D optical settings for LUMO simulation."""

    return Transport3DSettings(
        ray_count=256,
        max_interactions=6,
        maximum_segment_count=4096,
        maximum_periodic_wraps=8,
        surface_u_bins=32,
        surface_z_bins=16,
        internal_grid_width=32,
        internal_grid_height=32,
        internal_z_bins=8,
        x_bounds_mm=LUMO3D_OPTICAL_X_BOUNDS_MM,
        y_bounds_mm=LUMO3D_OPTICAL_Y_BOUNDS_MM,
        terminate_on_periodic_wrap_limit=True,
        terminate_on_no_event=True,
        retain_internal_path_field=True,
    )


@dataclass(frozen=True)
class ContactOpticalState:
    """One Newton checkpoint and its corresponding optical observation."""

    normalized_location: float
    indenter_radius_mm: float
    trajectory_id: str
    checkpoint: IndentationCheckpoint
    mechanics: FingertipVolumeState
    optics: Transport3DResult
    mechanics_artifact_path: Path
    mechanics_artifact_sha256: str
    contact_state: Mapping[str, Any] = field(default_factory=dict)
    final_pose_error_mm: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, IndentationCheckpoint):
            raise TypeError("checkpoint must be an IndentationCheckpoint")
        if not isinstance(self.mechanics, FingertipVolumeState):
            raise TypeError("mechanics must be a FingertipVolumeState")
        if not isinstance(self.optics, Transport3DResult):
            raise TypeError("optics must be a Transport3DResult")
        if not isinstance(self.contact_state, Mapping):
            raise TypeError("contact_state must be a mapping")
        object.__setattr__(
            self,
            "contact_state",
            MappingProxyType(dict(self.contact_state)),
        )


@dataclass(frozen=True)
class ContactSimulationResult:
    """All mechanics and optical states for one contact condition."""

    alignment: SphereAlignment
    first_contact: FirstContactResult
    trajectory: IndentationTrajectoryResult
    checkpoints: tuple[ContactOpticalState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, SphereAlignment):
            raise TypeError("alignment must be a SphereAlignment")
        if not isinstance(self.first_contact, FirstContactResult):
            raise TypeError("first_contact must be a FirstContactResult")
        if not isinstance(self.trajectory, IndentationTrajectoryResult):
            raise TypeError("trajectory must be an IndentationTrajectoryResult")
        checkpoints = tuple(self.checkpoints)
        if any(not isinstance(item, ContactOpticalState) for item in checkpoints):
            raise TypeError("checkpoints must contain ContactOpticalState values")
        object.__setattr__(self, "checkpoints", checkpoints)


class LumoSimulation:
    """Reusable Newton + OptiX orchestration for one prepared fingertip."""

    def __init__(
        self,
        *,
        tip: Fingertip,
        volume_mesh: FingertipVolumeMesh,
        prepared: PreparedFingertipMesh,
        contact_surface: FingertipContactSurface,
        carrier_mesh: RigidObjectMesh,
        artifact_root: str | Path,
        protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL,
        mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
        device: str = "cuda:0",
        settings: Transport3DSettings | None = None,
        optix_runtime: OptixRuntime | None = None,
    ) -> None:
        if not isinstance(tip, Fingertip):
            raise TypeError("tip must be a Fingertip")
        if not isinstance(volume_mesh, FingertipVolumeMesh):
            raise TypeError("volume_mesh must be a FingertipVolumeMesh")
        if not isinstance(prepared, PreparedFingertipMesh):
            raise TypeError("prepared must be a PreparedFingertipMesh")
        if not isinstance(contact_surface, FingertipContactSurface):
            raise TypeError("contact_surface must be a FingertipContactSurface")
        if not isinstance(carrier_mesh, RigidObjectMesh):
            raise TypeError("carrier_mesh must be a RigidObjectMesh")
        if not isinstance(protocol, TrajectoryEvaluationProtocol):
            raise TypeError("protocol must be TrajectoryEvaluationProtocol")
        if not isinstance(mechanics_contract, MechanicsContract):
            raise TypeError("mechanics_contract must be MechanicsContract")
        if optix_runtime is not None and not isinstance(optix_runtime, OptixRuntime):
            raise TypeError("optix_runtime must be an OptixRuntime or None")
        self.tip = tip
        self.volume_mesh = volume_mesh
        self.prepared = prepared
        self.contact_surface = contact_surface
        self.carrier_mesh = carrier_mesh
        self.artifact_root = Path(artifact_root)
        self.protocol = protocol
        self.mechanics_contract = mechanics_contract
        self.device = device
        self.settings = settings or lumo_optical_settings()
        self.optix_runtime = optix_runtime

    @classmethod
    def from_fingertip(
        cls,
        tip: Fingertip,
        *,
        artifact_root: str | Path,
        protocol: TrajectoryEvaluationProtocol = DEFAULT_TRAJECTORY_PROTOCOL,
        mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
        device: str = "cuda:0",
        settings: Transport3DSettings | None = None,
        optix_runtime: OptixRuntime | None = None,
    ) -> "LumoSimulation":
        """Prepare reusable mesh/contact/runtime state for one morphology."""

        if not isinstance(tip, Fingertip):
            raise TypeError("tip must be a Fingertip")
        volume_mesh = generate_volume_mesh(
            tip.solid(),
            volume_mesh_settings_for_tier("search"),
        )
        prepared = prepare_fingertip_mesh(volume_mesh)
        return cls(
            tip=tip,
            volume_mesh=volume_mesh,
            prepared=prepared,
            contact_surface=make_outer_compliant_surface(volume_mesh.solid),
            carrier_mesh=make_distal_phalanx_mesh(volume_mesh.solid),
            artifact_root=artifact_root,
            protocol=protocol,
            mechanics_contract=mechanics_contract,
            device=device,
            settings=settings,
            optix_runtime=optix_runtime,
        )

    def _runtime(self) -> OptixRuntime:
        if self.optix_runtime is None:
            self.optix_runtime = create_runtime()
        return self.optix_runtime

    @staticmethod
    def _checkpoint_values(
        checkpoint_depths_mm: tuple[float, ...],
        radius_mm: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        depths = tuple(float(value) for value in checkpoint_depths_mm)
        if not depths or any(not np.isfinite(value) or value <= 0.0 for value in depths):
            raise ValueError("checkpoint_depths_mm must contain positive finite values")
        if any(left >= right for left, right in zip(depths, depths[1:])):
            raise ValueError("checkpoint_depths_mm must be strictly increasing")
        radius = float(radius_mm)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_mm must be finite and positive")
        maximum = depths[-1]
        return (
            tuple(depth / maximum for depth in depths),
            tuple(depth / radius for depth in depths),
        )

    def run_sphere_contact(
        self,
        *,
        location_u: float,
        radius_mm: float,
        checkpoint_depths_mm: tuple[float, ...],
    ) -> ContactSimulationResult:
        """Run one spherical contact condition through Newton and OptiX."""

        sphere = make_sphere_mesh(
            radius_mm,
            subdivisions=self.mechanics_contract.sphere_subdivisions,
        )
        return self.run_contact(
            location_u=location_u,
            indenter=sphere,
            checkpoint_depths_mm=checkpoint_depths_mm,
        )

    def run_contact(
        self,
        *,
        location_u: float,
        indenter: RigidObjectMesh,
        checkpoint_depths_mm: tuple[float, ...],
    ) -> ContactSimulationResult:
        """Run first contact, one continuous trajectory, and all optical states."""

        alignment = sphere_alignment_at_normalized_location(
            self.tip.geometry,
            indenter,
            location_u,
            initial_gap_mm=self.protocol.initial_gap_mm,
        )
        if intersects(self.contact_surface, indenter, alignment.nominal_pose):
            raise CandidateContactError(
                f"u={location_u:g} nominal pose is not collision-free"
            )
        first_contact = find_first_contact(
            self.contact_surface,
            indenter,
            alignment.nominal_pose,
            alignment.approach_direction,
            FirstContactSettings(
                coarse_step_mm=0.25,
                tolerance_mm=1.0e-3,
                spawn_clearance_mm=0.05,
                max_travel_mm=20.0,
            ),
        )
        if intersects(self.contact_surface, indenter, first_contact.spawn_pose):
            raise CandidateContactError(
                f"u={location_u:g} spawn pose is not collision-free"
            )
        boundary_clearance = unintended_boundary_clearance_mm(
            self.tip.geometry,
            alignment,
            first_contact,
        )
        if boundary_clearance <= 0.0:
            raise CandidateContactError(
                f"u={location_u:g} reaches an unintended external boundary "
                f"before arc contact: clearance={boundary_clearance:g} mm"
            )

        checkpoint_fractions, normalized_ratios = self._checkpoint_values(
            checkpoint_depths_mm,
            alignment.radius_mm,
        )
        mechanics_settings = NewtonSettings(
            device=self.device,
            gravity=0.0,
            dt=self.mechanics_contract.dt_s,
            steps=1,
            iterations=self.mechanics_contract.vbd_iterations,
            fixed_vertex_indices=self.prepared.support_vertex_indices,
        )
        indentation_settings = IndentationSettings(
            travel_mm=checkpoint_depths_mm[-1],
            load_steps=max(
                1,
                int(
                    np.ceil(
                        float(checkpoint_depths_mm[-1])
                        / self.mechanics_contract.max_load_increment_mm
                    )
                ),
            ),
            soft_contact_margin_mm=self.mechanics_contract.soft_contact_margin_mm,
            soft_contact_ke=self.mechanics_contract.soft_contact_ke,
            soft_contact_kd=self.mechanics_contract.soft_contact_kd,
        )
        trajectory = solve_fingertip_indentation_trajectory(
            self.prepared,
            RigidIndenter3D(
                indenter,
                alignment.nominal_pose,
                alignment.approach_direction,
            ),
            mechanics_settings,
            indentation_settings,
            checkpoint_depths_mm,
            checkpoint_fractions=checkpoint_fractions,
            normalized_indentation_ratios=normalized_ratios,
            max_load_increment_mm=self.mechanics_contract.max_load_increment_mm,
            first_contact=first_contact,
            rigid_carrier_mesh=self.carrier_mesh,
        )

        trajectory_id = f"u_{location_u:.3f}__radius_{alignment.radius_mm:.3f}"
        checkpoints: list[ContactOpticalState] = []
        for checkpoint in trajectory.checkpoints:
            checkpoint_path = (
                self.artifact_root
                / "mechanics"
                / trajectory_id
                / (
                    f"checkpoint_{checkpoint.checkpoint_index:02d}"
                    f"_depth_{checkpoint.post_contact_travel_mm:.3f}mm.npz"
                )
            )
            artifact_sha = write_mechanics_artifact(
                checkpoint_path,
                checkpoint,
                self.prepared,
            )
            contact_state = build_contact_state_record(
                morphology_fingerprint=self.volume_mesh.morphology_fingerprint,
                protocol=self.protocol,
                location_u=float(location_u),
                radius_mm=alignment.radius_mm,
                checkpoint_depth_mm=checkpoint.post_contact_travel_mm,
                checkpoint_fraction=checkpoint.checkpoint_fraction,
                normalized_indentation_ratio=checkpoint.normalized_indentation_ratio,
                post_contact_travel_mm=checkpoint.post_contact_travel_mm,
                unintended_boundary_clearance_mm=boundary_clearance,
                checkpoint_diagnostics=checkpoint.diagnostics,
                source_node_ids=self.prepared.source_node_ids,
                mechanics_artifact_sha256=artifact_sha,
            )
            restored = restore_deformed_optical_state(
                self.tip,
                self.volume_mesh,
                self.prepared,
                checkpoint_path,
                artifact_sha,
                carrier_optics=CarrierOptics("absorber"),
                carrier_contact_source_node_ids=contact_state[
                    "carrier_contact_source_node_ids"
                ],
                carrier_mapping_tolerance_mm=contact_state[
                    "carrier_mapping_tolerance_mm"
                ],
                metadata={
                    "contact_state_fingerprint": contact_state[
                        "contact_state_fingerprint"
                    ],
                    "contact_location_u": location_u,
                    "checkpoint_depth_mm": checkpoint.post_contact_travel_mm,
                    "checkpoint_fraction": checkpoint.checkpoint_fraction,
                    "normalized_indentation_ratio": checkpoint.normalized_indentation_ratio,
                    "post_contact_travel_mm": checkpoint.post_contact_travel_mm,
                    "unintended_boundary_clearance_mm": boundary_clearance,
                    "observation_level": LUMO3D_OBSERVATION_LEVEL,
                    "carrier_optical_boundary_model": "absorber",
                },
            )
            expected_pose = first_contact.pose_at_post_contact_travel(
                checkpoint.post_contact_travel_mm
            )
            pose_error = float(
                np.linalg.norm(
                    np.asarray(checkpoint.indenter_pose.translation_mm)
                    - np.asarray(expected_pose.translation_mm)
                )
            )
            checkpoints.append(
                ContactOpticalState(
                    normalized_location=float(location_u),
                    indenter_radius_mm=alignment.radius_mm,
                    trajectory_id=trajectory_id,
                    checkpoint=checkpoint,
                    mechanics=restored.state,
                    optics=trace_geometry(
                        self.tip,
                        restored.geometry,
                        settings=self.settings,
                        runtime=self._runtime(),
                    ),
                    mechanics_artifact_path=checkpoint_path,
                    mechanics_artifact_sha256=artifact_sha,
                    contact_state=contact_state,
                    final_pose_error_mm=pose_error,
                )
            )
        return ContactSimulationResult(
            alignment=alignment,
            first_contact=first_contact,
            trajectory=trajectory,
            checkpoints=tuple(checkpoints),
        )


__all__ = [
    "ContactOpticalState",
    "ContactSimulationResult",
    "LumoSimulation",
    "LUMO3D_OBSERVATION_LEVEL",
    "lumo_optical_settings",
]
