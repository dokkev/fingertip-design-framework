"""Multi-contact FULL_3D evaluator for the LUMO optimization milestone."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from mechanics3d import prepare_fingertip_mechanics_mesh
from mesh import volume_mesh_settings_for_tier
from mesh.volume3d import VolumeMeshingError
from model import Fingertip, FingertipParameters, InvalidFingertip, InvalidFingertipParameters
from optics.contact_object import CarrierOptics
from optics.transport3d import (
    OptiXTransport,
    Transport3DDependencyError,
    Transport3DGeometryError,
    Transport3DPhysicsError,
    Transport3DResultError,
    Transport3DSettings,
    Transport3DTraceError,
    fingerprint_mapping,
    native_field_separability,
    save_case_artifact,
    transport_configuration,
)
from optics.transport3d.optix_backend import create_runtime
from optimization.design_space import (
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
)
from validation.mechanics3d.deformed_state_artifact import restore_deformed_optical_state
from validation.mechanics3d.multi_location_sphere_contact import (
    DEFAULT_LOCATION_U,
    DEFAULT_RADIUS_MM,
    DEFAULT_TRAVEL_MM,
    SEARCH_MAX_LOAD_INCREMENT_MM,
    SEARCH_SPHERE_SUBDIVISIONS,
    SEARCH_VBD_ITERATIONS,
    VALIDATION_MAX_LOAD_INCREMENT_MM,
    VALIDATION_VBD_ITERATIONS,
    run_multi_location_sphere_contact,
)


LUMO3D_OBSERVATION_LEVEL = "FULL_3D native internal transport redistribution proxy"
CONTACT_STATE_SEPARATION_OBJECTIVE_NAME = "contact_state_separation"
LUMO3D_OPTICAL_X_BOUNDS_MM = (-16.0, 16.0)
LUMO3D_OPTICAL_Y_BOUNDS_MM = (-15.0, 4.5)
LUMO3D_EVALUATION_CONTRACT: dict[str, Any] = {
    "schema": "lumo3d-multi-contact-evaluation-v1",
    "bounds_mm": PRODUCTION_SEARCH_BOUNDS,
    "contact": {
        "normalized_locations": DEFAULT_LOCATION_U,
        "sphere_radius_mm": DEFAULT_RADIUS_MM,
        "post_contact_travel_mm": DEFAULT_TRAVEL_MM,
    },
    "mechanics": {
        "backend": "newton_vbd",
        "tier": "search",
        "contract": "frozen-search-v1",
    },
    "optics": {
        "mode": "FULL_3D",
        "settings": "lumo3d-full3d-256r-v1",
        "x_bounds_mm": LUMO3D_OPTICAL_X_BOUNDS_MM,
        "y_bounds_mm": LUMO3D_OPTICAL_Y_BOUNDS_MM,
        "carrier_boundary_model": "absorber",
        "carrier_mapping": "exact_semantic_surface_triangle_any_contact_vertex",
    },
    "objective": {
        "name": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
        "direction": "maximize",
        "metric": "min-pairwise-native-field-normalized-l1-v1",
    },
    "observation_level": LUMO3D_OBSERVATION_LEVEL,
}
LUMO3D_EVALUATION_CONTRACT_ID = (
    "lumo3d-multi-contact-v1-"
    + hashlib.sha256(
        json.dumps(
            LUMO3D_EVALUATION_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()[:16]
)
def _optical_settings() -> Transport3DSettings:
    return Transport3DSettings(
        mode="full3d",
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
        retain_projected_segments=False,
    )


def _material(tip: Fingertip) -> dict[str, float]:
    return {
        "refractive_index_air": tip.optical.refractive_index_air,
        "refractive_index_silicone": tip.optical.refractive_index_silicone,
        "absorption_per_mm": tip.optical.absorption_per_mm,
        "scattering_per_mm": tip.optical.scattering_per_mm,
    }


def _candidate_id(parameters: FingertipParameters) -> str:
    payload = {
        name: float(getattr(parameters, name))
        for name in (
            "flat_pad_height",
            "stem_width",
            "stem_height",
            "void_width",
            "void_height",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _contact_state(
    case: Any,
    morphology_fingerprint: str,
    *,
    source_node_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    local_contact_indices = tuple(
        int(index)
        for index in case.indentation.diagnostics.get(
            "carrier_contact_vertex_indices", ()
        )
    )
    contact_source_ids = tuple(
        int(source_node_ids[index])
        for index in local_contact_indices
        if 0 <= index < len(source_node_ids)
    )
    return {
        "contact_state_fingerprint": fingerprint_mapping(
            {
                "morphology_fingerprint": morphology_fingerprint,
                "normalized_location": case.normalized_location,
                "radius_mm": DEFAULT_RADIUS_MM,
                "travel_mm": DEFAULT_TRAVEL_MM,
                "carrier_contact_source_node_ids": contact_source_ids,
            }
        ),
        "normalized_location": float(case.normalized_location),
        "target_point_mm": list(case.alignment.target_point_mm),
        "outward_normal": list(case.alignment.outward_normal),
        "approach_direction": list(case.alignment.approach_direction),
        "indenter_radius_mm": DEFAULT_RADIUS_MM,
        "initial_gap_mm": 0.25,
        "first_contact_travel_mm": float(case.first_contact.travel_to_contact_mm),
        "post_contact_travel_mm": float(
            case.indentation.diagnostics["post_contact_travel_mm"]
        ),
        "spawn_clearance_mm": float(case.first_contact.spawn_clearance_mm),
        "carrier_contact_active": bool(
            case.indentation.diagnostics.get("carrier_contact_active", False)
        ),
        "carrier_mechanical_contact_count": int(
            case.indentation.diagnostics.get(
                "carrier_interface_contact_count",
                case.indentation.diagnostics.get(
                    "max_void_bottom_carrier_contact_count", 0
                ),
            )
        ),
        "carrier_mechanical_contact_vertex_count": int(
            case.indentation.diagnostics.get("carrier_contact_vertex_count", 0)
        ),
        "first_carrier_contact_step": case.indentation.diagnostics.get(
            "first_carrier_contact_step"
        ),
        "carrier_contact_source_node_ids": list(contact_source_ids),
        "carrier_mapping_tolerance_mm": 0.5 * float(
            case.indentation.diagnostics.get("rigid_sdf_target_voxel_mm", 0.125)
        ),
    }


def _energy_record(result: Any) -> dict[str, Any]:
    launched = float(result.launched_weight)
    carrier_absorbed = float(result.carrier_absorbed_weight)
    escaped = float(result.escaped_weight)
    return {
        "launched_weight": launched,
        "escaped_weight": escaped,
        "escaped_transport_fraction": escaped / max(launched, 1.0e-30),
        "absorbed_weight": float(result.absorbed_weight),
        "terminated_weight": float(result.terminated_weight),
        "total_transport": float(result.total_transport),
        "object_interface_optics": "disabled_in_deformation_only_scene",
        "object_interface_incident_weight": float(result.object_interface_incident_weight),
        "object_absorbed_weight": float(result.object_absorbed_weight),
        "object_transmitted_weight": float(result.object_transmitted_weight),
        "object_reflected_weight": float(result.object_reflected_weight),
        "carrier_absorbed_weight": carrier_absorbed,
        "carrier_absorption_fraction": carrier_absorbed / max(launched, 1.0e-30),
        "carrier_transmitted_weight": float(result.carrier_transmitted_weight),
        "carrier_interface_incident_weight": float(
            result.carrier_interface_incident_weight
        ),
        "carrier_reflected_weight": float(result.carrier_reflected_weight),
        "carrier_optical_contact_triangle_count": int(
            result.path_diagnostics.get("carrier_interface", {}).get(
                "contact_triangle_count", 0
            )
        ),
        "energy_balance_error": float(result.energy_balance_error),
        "field_shape": list(result.field.shape),
        "field_finite_nonnegative": bool(
            np.all(np.isfinite(result.field)) and np.all(result.field >= 0.0)
        ),
    }


@dataclass(frozen=True)
class Lumo3DEvaluation:
    """Ax-compatible result without aliasing the 3D score to ``minimum_auc``."""

    status: Literal["success", "invalid_design", "mesh_failure", "fea_failure", "optics_failure"]
    objective_value: float | None
    pairwise_distance_matrix: tuple[tuple[float | None, ...], ...]
    contact_states: tuple[Mapping[str, Any], ...]
    mechanics_diagnostics: tuple[Mapping[str, Any], ...]
    optical_diagnostics: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    failure_message: str | None = None

    @property
    def score(self) -> float | None:
        return self.objective_value


def _failure(
    status: Literal["invalid_design", "mesh_failure", "fea_failure", "optics_failure"],
    message: str,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> Lumo3DEvaluation:
    return Lumo3DEvaluation(
        status=status,
        objective_value=None,
        pairwise_distance_matrix=(),
        contact_states=(),
        mechanics_diagnostics=(),
        optical_diagnostics=(),
        diagnostics={} if diagnostics is None else dict(diagnostics),
        failure_message=message,
    )


class Lumo3DEvaluator:
    """Evaluate one morphology at the three frozen sphere contact locations."""

    objective_name = CONTACT_STATE_SEPARATION_OBJECTIVE_NAME

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        device: str = "cuda:0",
        normalized_locations: tuple[float, ...] = DEFAULT_LOCATION_U,
        mechanics_mode: str = "search",
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.device = device
        self.normalized_locations = tuple(float(value) for value in normalized_locations)
        if self.normalized_locations != tuple(DEFAULT_LOCATION_U):
            raise ValueError("Lumo3DEvaluator requires the frozen three contact locations")
        if mechanics_mode not in {"search", "validation"}:
            raise ValueError("mechanics_mode must be 'search' or 'validation'")
        self.mechanics_mode = mechanics_mode
        if mechanics_mode == "search":
            self.mechanics_contract = {
                "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
                "max_load_increment_mm": SEARCH_MAX_LOAD_INCREMENT_MM,
                "vbd_iterations": SEARCH_VBD_ITERATIONS,
            }
        else:
            self.mechanics_contract = {
                "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
                "max_load_increment_mm": VALIDATION_MAX_LOAD_INCREMENT_MM,
                "vbd_iterations": VALIDATION_VBD_ITERATIONS,
            }
        self.settings = _optical_settings()

    def evaluate(self, parameters: FingertipParameters) -> Lumo3DEvaluation:
        stage = "mechanics"
        try:
            tip = Fingertip(parameters)
            candidate_id = _candidate_id(parameters)
            candidate_root = self.artifact_root / "candidates" / candidate_id
            mechanics_root = candidate_root / "mechanics"
            contact = run_multi_location_sphere_contact(
                parameters=parameters,
                device=self.device,
                radius_mm=DEFAULT_RADIUS_MM,
                travel_mm=DEFAULT_TRAVEL_MM,
                normalized_locations=self.normalized_locations,
                artifact_dir=mechanics_root,
                sphere_subdivisions=self.mechanics_contract["sphere_subdivisions"],
                max_load_increment_mm=self.mechanics_contract["max_load_increment_mm"],
                vbd_iterations=self.mechanics_contract["vbd_iterations"],
                carrier_contact=True,
            )
            mechanics_records = tuple(case.to_dict() for case in contact.locations)
            self._validate_mechanics(mechanics_records)

            volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
            prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
            material = _material(tip)
            configuration = transport_configuration(
                self.settings,
                material=material,
                source={"model": "existing Fingertip optical source"},
            )
            runtime = create_runtime()
            transport = OptiXTransport()
            optical_records: list[dict[str, Any]] = []
            results: list[Any] = []
            stage = "optics"
            for case in contact.locations:
                contact_state = _contact_state(
                    case,
                    volume_mesh.morphology_fingerprint,
                    source_node_ids=prepared.source_node_ids,
                )
                restored = restore_deformed_optical_state(
                    tip,
                    volume_mesh,
                    prepared,
                    case.mechanics_artifact_path,
                    case.mechanics_artifact_sha256,
                    carrier_optics=CarrierOptics("absorber"),
                    metadata={
                        "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
                        "contact_location_u": case.normalized_location,
                        "observation_level": LUMO3D_OBSERVATION_LEVEL,
                        "carrier_optical_boundary_model": "absorber",
                        "carrier_mapping_tolerance_mm": contact_state[
                            "carrier_mapping_tolerance_mm"
                        ],
                    },
                )
                result = transport.trace(
                    tip,
                    restored.geometry,
                    settings=self.settings,
                    morphology_id=candidate_id,
                    morphology_fingerprint=volume_mesh.morphology_fingerprint,
                    mechanics_source=str(restored.artifact_path),
                    mechanics_dimension="3D",
                    contact_state=contact_state,
                    transport_configuration=configuration,
                    runtime=runtime,
                )
                contract = {
                    "schema": "lumo3d-multi-contact-evaluation-v1",
                    "objective": self.objective_name,
                    "observation_level": LUMO3D_OBSERVATION_LEVEL,
                    "morphology_id": candidate_id,
                    "morphology_parameters": {
                        name: float(getattr(parameters, name))
                        for name in (
                            "flat_pad_height",
                            "stem_width",
                            "stem_height",
                            "void_width",
                            "void_height",
                        )
                    },
                    "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                    "mechanics_source": str(restored.artifact_path),
                    "mechanics_artifact_sha256": restored.artifact_sha256,
                    "mechanics_dimension": "3D",
                    "geometry_mode": "full3d_surface",
                    "full3d_surface_provenance": "actual_deformed_3d_volume_state",
                    "contact_state": contact_state,
                    "transport_configuration": configuration,
                    "transport_configuration_fingerprint": result.transport_configuration_fingerprint,
                }
                artifact = candidate_root / f"location_u_{case.normalized_location:.3f}.json"
                save_case_artifact(artifact, result, contract)
                record = _energy_record(result)
                record.update(
                    {
                        "normalized_location": case.normalized_location,
                        "artifact": str(artifact),
                        "artifact_field": str(artifact.with_suffix(".npz")),
                        "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
                        "mechanics_artifact_path": str(restored.artifact_path),
                        "mechanics_artifact_sha256": restored.artifact_sha256,
                    }
                )
                optical_records.append(record)
                results.append(result)

            pairwise: list[list[float | None]] = [
                [0.0 if row == column else None for column in range(3)]
                for row in range(3)
            ]
            for left in range(3):
                for right in range(left + 1, 3):
                    separation = native_field_separability(results[left], results[right])
                    value = separation["normalized_redistribution_l1"]
                    if value is None or not np.isfinite(value):
                        raise Transport3DResultError(
                            "pairwise FULL_3D field separation is not finite"
                        )
                    pairwise[left][right] = float(value)
                    pairwise[right][left] = float(value)
                    optical_records[left].setdefault("pairwise_to_other_states", {})[
                        str(right)
                    ] = separation
                    optical_records[right].setdefault("pairwise_to_other_states", {})[
                        str(left)
                    ] = separation
            values = [pairwise[row][column] for row in range(3) for column in range(row + 1, 3)]
            objective = float(min(value for value in values if value is not None))
            summary = {
                "objective_name": self.objective_name,
                "objective_direction": "maximize",
                "objective_value": objective,
                "observation_level": LUMO3D_OBSERVATION_LEVEL,
                "object_interface_optics": "disabled_in_deformation_only_scene",
                "carrier_boundary_model": "absorber",
                "carrier_contact_states": [
                    record.get("carrier_optical_contact_triangle_count", 0)
                    for record in optical_records
                ],
                "pairwise_distance_matrix": pairwise,
                "mechanics_contract": contact.to_dict()["search_contract"],
                "mechanics_mode": self.mechanics_mode,
                "morphology_fingerprint": volume_mesh.morphology_fingerprint,
                "transport_configuration_fingerprint": fingerprint_mapping(configuration),
            }
            summary_path = candidate_root / "evaluation.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {"summary": summary, "mechanics": mechanics_records, "optics": optical_records},
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            return Lumo3DEvaluation(
                status="success",
                objective_value=objective,
                pairwise_distance_matrix=tuple(tuple(row) for row in pairwise),
                contact_states=tuple(
                    {
                        "normalized_location": case.normalized_location,
                        **_contact_state(
                            case,
                            volume_mesh.morphology_fingerprint,
                            source_node_ids=prepared.source_node_ids,
                        ),
                    }
                    for case in contact.locations
                ),
                mechanics_diagnostics=mechanics_records,
                optical_diagnostics=tuple(optical_records),
                diagnostics=summary,
            )
        except Transport3DDependencyError:
            raise
        except (InvalidFingertip, InvalidFingertipParameters) as exc:
            return _failure("invalid_design", f"{type(exc).__name__}: {exc}")
        except VolumeMeshingError as exc:
            return _failure("mesh_failure", f"{type(exc).__name__}: {exc}")
        except (
            Transport3DGeometryError,
            Transport3DPhysicsError,
            Transport3DResultError,
            Transport3DTraceError,
        ) as exc:
            return _failure("optics_failure", f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})
        except Exception as exc:
            status = "optics_failure" if stage == "optics" else "fea_failure"
            return _failure(status, f"{type(exc).__name__}: {exc}", diagnostics={"failure_stage": stage})

    @staticmethod
    def _validate_mechanics(records: tuple[Mapping[str, Any], ...]) -> None:
        if len(records) != 3:
            raise RuntimeError("multi-contact evaluator requires exactly three states")
        for record in records:
            if not record["finite_deformation"]:
                raise RuntimeError("mechanics produced non-finite deformation")
            if record["inverted_tetrahedra"] != 0:
                raise RuntimeError("mechanics produced inverted tetrahedra")
            if record["max_soft_contact_overflow"] != 0 or record["max_rigid_contact_overflow"] != 0:
                raise RuntimeError("mechanics contact buffer overflow")
            if record["max_support_displacement_mm"] > 1.0e-9:
                raise RuntimeError("mechanics support vertices moved")
            if record["final_pose_error_mm"] > 1.0e-6:
                raise RuntimeError("mechanics final prescribed pose error is too large")
            if record["unintended_boundary_clearance_mm"] <= 0.0 or record["cell_end_clearance_mm"] <= 0.0:
                raise RuntimeError("mechanics contact violates geometric clearance")


@dataclass(frozen=True)
class Lumo3DStudy:
    """Small Ax study adapter that reuses the production five-variable space."""

    design_space: DesignSpace
    artifact_root: Path
    device: str = "cuda:0"
    mechanics_mode: str = "search"

    def create_evaluator(self) -> Lumo3DEvaluator:
        return Lumo3DEvaluator(
            self.artifact_root,
            device=self.device,
            mechanics_mode=self.mechanics_mode,
        )


def create_lumo3d_study(
    artifact_root: str | Path,
    *,
    device: str = "cuda:0",
    mechanics_mode: str = "search",
) -> Lumo3DStudy:
    """Create a 3D-active study without importing the legacy 2D evaluator."""
    return Lumo3DStudy(
        design_space=DesignSpace(
            FingertipParameters(void_height=PRODUCTION_NOMINAL_VOID_HEIGHT_MM),
            tuple(
                DesignVariable(name, True, lower, upper)
                for name, lower, upper in PRODUCTION_SEARCH_BOUNDS
            ),
        ),
        artifact_root=Path(artifact_root),
        device=device,
        mechanics_mode=mechanics_mode,
    )


__all__ = [
    "LUMO3D_OBSERVATION_LEVEL",
    "LUMO3D_EVALUATION_CONTRACT",
    "LUMO3D_EVALUATION_CONTRACT_ID",
    "LUMO3D_OPTICAL_X_BOUNDS_MM",
    "LUMO3D_OPTICAL_Y_BOUNDS_MM",
    "Lumo3DEvaluation",
    "Lumo3DEvaluator",
    "Lumo3DStudy",
    "create_lumo3d_study",
]
