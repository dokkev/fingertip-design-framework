"""Concrete LUMO simulation orchestration for reusable morphology state."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from contact import (
    CandidateContactError,
    FirstContactResult,
    FingertipContactSurface,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
    unintended_boundary_clearance_mm,
)
from contact.sphere_alignment import SphereAlignment
from lumo.mechanics_contract import DEFAULT_MECHANICS_CONTRACT, MechanicsContract
from mesh import volume_mesh_settings_for_tier
from mesh.rigid.carrier import RigidCarrierMesh, make_distal_phalanx_mesh
from mesh.rigid.object import make_sphere_mesh
from mesh.volume.contracts import FingertipVolumeMesh
from mesh.volume.mesh import generate_volume_mesh
from mesh.volume.state import FingertipVolumeState, InvalidDeformedFingertipState
from finger import Fingertip
from ray_tracing.contracts.objects import CarrierOptics
from ray_tracing.optical_mechanics import (
    Transport3DResult,
    Transport3DSettings,
    build_fingertip_volume_state_geometry,
    trace_geometry,
)
from ray_tracing.optical_mechanics.geometry import TransportGeometry
from ray_tracing.optical_mechanics.optix_backend import create_runtime
from ray_tracing.optix.runtime import OptixRuntime
from physics import (
    CandidateMechanicsError,
    IndentationCheckpoint,
    IndentationSettings,
    IndentationTrajectoryResult,
    NewtonSettings,
    PreparedFingertipMesh,
    RigidIndenter3D,
    make_fingertip_volume_state,
    prepare_fingertip_mesh,
    solve_fingertip_indentation_trajectory,
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


def _prepared_mesh_matches_volume_mesh(
    prepared: PreparedFingertipMesh,
    volume_mesh: FingertipVolumeMesh,
) -> bool:
    """Return whether a prepared mechanics view is the canonical mesh adapter."""

    expected = prepare_fingertip_mesh(volume_mesh)
    return bool(
        prepared.morphology_fingerprint == expected.morphology_fingerprint
        and np.array_equal(prepared.source_node_ids, expected.source_node_ids)
        and np.array_equal(prepared.tet_mesh.vertices, expected.tet_mesh.vertices)
        and np.array_equal(prepared.tet_mesh.tetrahedra, expected.tet_mesh.tetrahedra)
        and prepared.support_vertex_indices == expected.support_vertex_indices
        and set(prepared.surface_triangles) == set(expected.surface_triangles)
        and all(
            np.array_equal(
                prepared.surface_triangles[tag],
                expected.surface_triangles[tag],
            )
            for tag in expected.surface_triangles
        )
    )


@dataclass(frozen=True)
class ContactOpticalState:
    """One Newton checkpoint and its corresponding optical observation."""

    checkpoint: IndentationCheckpoint
    mechanics: FingertipVolumeState
    optics: Transport3DResult

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, IndentationCheckpoint):
            raise TypeError("checkpoint must be an IndentationCheckpoint")
        if not isinstance(self.mechanics, FingertipVolumeState):
            raise TypeError("mechanics must be a FingertipVolumeState")
        if not isinstance(self.optics, Transport3DResult):
            raise TypeError("optics must be a Transport3DResult")


@dataclass(frozen=True)
class ContactSimulationResult:
    """All mechanics and optical states for one spherical contact condition."""

    normalized_location: float
    indenter_radius_mm: float
    alignment: SphereAlignment
    first_contact: FirstContactResult
    trajectory: IndentationTrajectoryResult
    checkpoints: tuple[ContactOpticalState, ...]
    unintended_boundary_clearance_mm: float
    mechanics_seconds: float
    optics_seconds: float

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
        for name in (
            "normalized_location",
            "indenter_radius_mm",
            "unintended_boundary_clearance_mm",
            "mechanics_seconds",
            "optics_seconds",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.normalized_location <= 1.0:
            raise ValueError("normalized_location must lie in [0, 1]")
        if self.indenter_radius_mm <= 0.0:
            raise ValueError("indenter_radius_mm must be positive")
        if self.unintended_boundary_clearance_mm <= 0.0:
            raise ValueError("unintended_boundary_clearance_mm must be positive")


class LumoSimulation:
    """Reusable Newton + OptiX orchestration for one prepared fingertip."""

    def __init__(
        self,
        *,
        tip: Fingertip,
        volume_mesh: FingertipVolumeMesh,
        prepared: PreparedFingertipMesh,
        contact_surface: FingertipContactSurface,
        carrier_mesh: RigidCarrierMesh,
        initial_gap_mm: float,
        mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
        device: str = "cuda:0",
        optical_settings: Transport3DSettings | None = None,
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
        if not isinstance(carrier_mesh, RigidCarrierMesh):
            raise TypeError("carrier_mesh must be a RigidCarrierMesh")
        if not isinstance(mechanics_contract, MechanicsContract):
            raise TypeError("mechanics_contract must be a MechanicsContract")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        gap = float(initial_gap_mm)
        if not np.isfinite(gap) or gap <= 0.0:
            raise ValueError("initial_gap_mm must be finite and positive")
        if optix_runtime is not None and not isinstance(optix_runtime, OptixRuntime):
            raise TypeError("optix_runtime must be an OptixRuntime or None")
        selected_optical_settings = optical_settings or lumo_optical_settings()
        if not isinstance(selected_optical_settings, Transport3DSettings):
            raise TypeError("optical_settings must be a Transport3DSettings or None")
        solid = tip.solid()
        if volume_mesh.morphology_fingerprint != solid.morphology_fingerprint:
            raise ValueError("volume_mesh morphology does not match tip")
        if not _prepared_mesh_matches_volume_mesh(prepared, volume_mesh):
            raise ValueError(
                "prepared mechanics coordinates/topology do not match volume_mesh"
            )
        expected_contact_surface = make_outer_compliant_surface(solid)
        if (
            not contact_surface.outer_compliant_arc.equals(
                expected_contact_surface.outer_compliant_arc
            )
            or not np.isclose(
                contact_surface.z_min_mm,
                expected_contact_surface.z_min_mm,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not np.isclose(
                contact_surface.z_max_mm,
                expected_contact_surface.z_max_mm,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("contact_surface does not match volume_mesh")
        if carrier_mesh.morphology_fingerprint != volume_mesh.morphology_fingerprint:
            raise ValueError("carrier_mesh morphology does not match volume_mesh")
        self.tip = tip
        self.volume_mesh = volume_mesh
        self.prepared = prepared
        self.contact_surface = contact_surface
        self.carrier_mesh = carrier_mesh
        self.initial_gap_mm = gap
        self.mechanics_contract = mechanics_contract
        self.device = device
        self.optical_settings = selected_optical_settings
        self.optix_runtime = optix_runtime

    @classmethod
    def from_fingertip(
        cls,
        tip: Fingertip,
        *,
        initial_gap_mm: float = 0.25,
        mechanics_contract: MechanicsContract = DEFAULT_MECHANICS_CONTRACT,
        device: str = "cuda:0",
        optical_settings: Transport3DSettings | None = None,
        optix_runtime: OptixRuntime | None = None,
    ) -> "LumoSimulation":
        """Prepare reusable volume/contact/carrier state for one morphology."""

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
            initial_gap_mm=initial_gap_mm,
            mechanics_contract=mechanics_contract,
            device=device,
            optical_settings=optical_settings,
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

    def _carrier_contact_source_ids(self, checkpoint: IndentationCheckpoint) -> tuple[int, ...]:
        if "active_carrier_contact_vertex_indices" not in checkpoint.diagnostics:
            raise RuntimeError(
                "mechanics checkpoint is missing "
                "active_carrier_contact_vertex_indices"
            )
        raw_indices = np.asarray(
            checkpoint.diagnostics["active_carrier_contact_vertex_indices"]
        )
        if raw_indices.ndim != 1 or (
            raw_indices.size
            and not np.issubdtype(raw_indices.dtype, np.integer)
        ):
            raise RuntimeError(
                "active carrier-contact vertex indices must be a 1D integer sequence"
            )
        local_indices = tuple(int(index) for index in raw_indices)
        if len(set(local_indices)) != len(local_indices):
            raise RuntimeError(
                "mechanics checkpoint contains duplicate carrier-contact vertex indices"
            )
        invalid = tuple(
            index
            for index in local_indices
            if index < 0 or index >= len(self.prepared.source_node_ids)
        )
        if invalid:
            raise RuntimeError(
                "mechanics checkpoint contains out-of-range carrier-contact "
                f"vertex indices: {invalid!r}"
            )
        return tuple(
            int(self.prepared.source_node_ids[index])
            for index in local_indices
        )

    @staticmethod
    def _required_diagnostic(
        checkpoint: IndentationCheckpoint,
        name: str,
    ) -> object:
        if name not in checkpoint.diagnostics:
            raise RuntimeError(
                f"mechanics checkpoint is missing required diagnostic {name!r}"
            )
        return checkpoint.diagnostics[name]

    def _validate_checkpoint(
        self,
        checkpoint: IndentationCheckpoint,
    ) -> None:
        """Reject candidate-specific mechanics states before optical tracing."""
        if int(self._required_diagnostic(checkpoint, "inverted_tetrahedra")) != 0:
            raise CandidateMechanicsError("mechanics produced inverted tetrahedra")
        if (
            int(self._required_diagnostic(checkpoint, "max_soft_contact_overflow"))
            != 0
            or int(
                self._required_diagnostic(checkpoint, "max_rigid_contact_overflow")
            )
            != 0
        ):
            raise CandidateMechanicsError("mechanics contact buffer overflow")

        support_displacement = float(
            self._required_diagnostic(checkpoint, "max_support_displacement_mm")
        )
        if support_displacement < 0.0:
            raise RuntimeError(
                "mechanics checkpoint reports a negative support displacement"
            )
        if (
            not np.isfinite(support_displacement)
            or support_displacement
            > self.mechanics_contract.max_support_displacement_mm
        ):
            raise CandidateMechanicsError(
                "mechanics support displacement exceeds its contract: "
                f"{support_displacement:g} mm > "
                f"{self.mechanics_contract.max_support_displacement_mm:g} mm"
            )

        pose_error_mm = float(
            self._required_diagnostic(checkpoint, "final_pose_error_mm")
        )
        if pose_error_mm < 0.0:
            raise RuntimeError(
                "mechanics checkpoint reports a negative prescribed-pose error"
            )
        if (
            not np.isfinite(pose_error_mm)
            or pose_error_mm > self.mechanics_contract.max_final_pose_error_mm
        ):
            raise CandidateMechanicsError(
                "mechanics prescribed-pose error exceeds its contract: "
                f"{pose_error_mm:g} mm > "
                f"{self.mechanics_contract.max_final_pose_error_mm:g} mm"
            )

        voxel_mm = float(
            self._required_diagnostic(checkpoint, "rigid_sdf_target_voxel_mm")
        )
        if not np.isfinite(voxel_mm) or voxel_mm <= 0.0:
            raise RuntimeError(
                "mechanics checkpoint reports an invalid rigid-SDF voxel size: "
                f"{voxel_mm:g} mm"
            )
        if bool(self._required_diagnostic(checkpoint, "carrier_collision_enabled")):
            penetration_mm = float(
                self._required_diagnostic(checkpoint, "max_carrier_penetration_mm")
            )
            if penetration_mm < 0.0:
                raise RuntimeError(
                    "mechanics checkpoint reports negative carrier penetration"
                )
            limit_mm = (
                self.mechanics_contract.max_carrier_penetration_voxel_fraction
                * voxel_mm
            )
            if (
                not np.isfinite(penetration_mm)
                or penetration_mm > limit_mm
            ):
                raise CandidateMechanicsError(
                    "mechanics carrier penetration exceeds its contract: "
                    f"{penetration_mm:g} mm > {limit_mm:g} mm"
                )

    def _state_geometry(
        self,
        checkpoint: IndentationCheckpoint,
        *,
        location_u: float,
        carrier_contact_source_node_ids: tuple[int, ...],
    ) -> tuple[FingertipVolumeState, TransportGeometry]:
        try:
            mechanics = make_fingertip_volume_state(
                self.volume_mesh,
                self.prepared,
                checkpoint.mechanics_result,
            )
        except InvalidDeformedFingertipState as exc:
            raise CandidateMechanicsError(
                f"mechanics produced an invalid deformed state: {exc}"
            ) from exc
        mapping_tolerance_mm = 0.5 * float(
            self._required_diagnostic(checkpoint, "rigid_sdf_target_voxel_mm")
        )
        geometry = build_fingertip_volume_state_geometry(
            self.tip,
            mechanics,
            carrier_mesh=self.carrier_mesh,
            carrier_contact_source_node_ids=frozenset(
                carrier_contact_source_node_ids
            ),
            carrier_optics=CarrierOptics("absorber"),
            carrier_mapping_tolerance_mm=mapping_tolerance_mm,
            full3d_surface_provenance="actual_deformed_3d_volume_state",
            metadata={
                "contact_location_u": float(location_u),
                "checkpoint_depth_mm": checkpoint.post_contact_travel_mm,
            },
        )
        return mechanics, geometry

    def run_sphere_contact(
        self,
        *,
        location_u: float,
        radius_mm: float,
        checkpoint_depths_mm: tuple[float, ...],
    ) -> ContactSimulationResult:
        """Run one spherical contact condition through Newton and OptiX."""

        mechanics_started = time.perf_counter()
        sphere = make_sphere_mesh(
            radius_mm,
            subdivisions=self.mechanics_contract.sphere_subdivisions,
        )
        alignment = sphere_alignment_at_normalized_location(
            self.tip.geometry,
            location_u,
            radius_mm=radius_mm,
            initial_gap_mm=self.initial_gap_mm,
        )
        if intersects(self.contact_surface, sphere, alignment.nominal_pose):
            raise CandidateContactError(
                f"u={location_u:g} nominal pose is not collision-free"
            )
        first_contact = find_first_contact(
            self.contact_surface,
            sphere,
            alignment.nominal_pose,
            alignment.approach_direction,
            self.mechanics_contract.first_contact,
        )
        if intersects(self.contact_surface, sphere, first_contact.spawn_pose):
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
        depths = tuple(float(value) for value in checkpoint_depths_mm)
        viscoelastic = self.tip.parameters.viscoelastic
        mechanics_settings = NewtonSettings(
            device=self.device,
            gravity=0.0,
            dt=self.mechanics_contract.dt_s,
            steps=1,
            iterations=self.mechanics_contract.vbd_iterations,
            density=viscoelastic.density_kg_m3,
            k_mu=viscoelastic.k_mu_pa,
            k_lambda=viscoelastic.k_lambda_pa,
            k_damp=viscoelastic.k_damp,
            fixed_vertex_indices=self.prepared.support_vertex_indices,
        )
        indentation_settings = IndentationSettings(
            travel_mm=depths[-1],
            load_steps=max(
                1,
                int(
                    np.ceil(
                        depths[-1] / self.mechanics_contract.max_load_increment_mm
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
                sphere,
                alignment.nominal_pose,
                alignment.approach_direction,
            ),
            mechanics_settings,
            indentation_settings,
            depths,
            checkpoint_fractions=checkpoint_fractions,
            normalized_indentation_ratios=normalized_ratios,
            max_load_increment_mm=self.mechanics_contract.max_load_increment_mm,
            first_contact=first_contact,
            rigid_carrier_mesh=self.carrier_mesh,
        )
        mechanics_seconds = time.perf_counter() - mechanics_started

        optics_started = time.perf_counter()
        checkpoints: list[ContactOpticalState] = []
        for checkpoint in trajectory.checkpoints:
            self._validate_checkpoint(checkpoint)
            source_ids = self._carrier_contact_source_ids(checkpoint)
            mechanics, geometry = self._state_geometry(
                checkpoint,
                location_u=location_u,
                carrier_contact_source_node_ids=source_ids,
            )
            checkpoints.append(
                ContactOpticalState(
                    checkpoint=checkpoint,
                    mechanics=mechanics,
                    optics=trace_geometry(
                        self.tip,
                        geometry,
                        settings=self.optical_settings,
                        runtime=self._runtime(),
                    ),
                )
            )
        optics_seconds = time.perf_counter() - optics_started
        return ContactSimulationResult(
            normalized_location=float(location_u),
            indenter_radius_mm=alignment.radius_mm,
            alignment=alignment,
            first_contact=first_contact,
            trajectory=trajectory,
            checkpoints=tuple(checkpoints),
            unintended_boundary_clearance_mm=float(boundary_clearance),
            mechanics_seconds=mechanics_seconds,
            optics_seconds=optics_seconds,
        )


__all__ = [
    "ContactOpticalState",
    "ContactSimulationResult",
    "LumoSimulation",
    "LUMO3D_OBSERVATION_LEVEL",
    "lumo_optical_settings",
]
