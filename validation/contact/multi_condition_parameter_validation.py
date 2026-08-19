"""Deterministic validation of location, radius, and indentation semantics.

This module is deliberately an orchestration-level validation runner.  It
does not define a new mechanics or optics model and it does not call Ax.  All
Newton and FULL_3D transport work is routed through the existing production
helpers with explicit per-condition arguments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np

from contact import (
    FirstContactResult,
    FirstContactSettings,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
)
from mechanics3d import (
    InvalidFingertipMechanicsMesh,
    prepare_fingertip_mechanics_mesh,
)
from mechanics3d.indentation import RigidPose3D
from mesh.rigid_object import RigidObjectMesh, make_sphere_mesh
from mesh.volume3d import VolumeMeshDependencyError, VolumeMeshingError
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters
from optics.contact_object import CarrierOptics
from optics.transport3d import (
    OptiXTransport,
    fingerprint_mapping,
    save_case_artifact,
    transport_configuration,
)
from optics.transport3d.optix_backend import create_runtime
from validation.mechanics3d.multi_location_sphere_contact import (
    DEFAULT_LOCATION_U,
    SEARCH_DT_S,
    SEARCH_MAX_LOAD_INCREMENT_MM,
    SEARCH_SOFT_CONTACT_KD,
    SEARCH_SOFT_CONTACT_KE,
    SEARCH_SOFT_CONTACT_MARGIN_MM,
    SEARCH_SPHERE_SUBDIVISIONS,
    SEARCH_VBD_ITERATIONS,
    _unintended_boundary_clearance_mm,
    load_steps_for_increment,
    run_multi_location_sphere_contact,
)
from validation.mechanics3d.deformed_state_artifact import restore_deformed_optical_state
from validation.optimization.lumo3d_evaluator import (
    LUMO3D_OBSERVATION_LEVEL,
    _material,
    _optical_settings,
)


DEFAULT_OUTPUT = Path("output/validation/contact/multi_condition_parameter_validation")
NORMALIZED_LOCATIONS = (0.25, 0.50, 0.75)
PROPOSED_RADII_MM = (4.0, 6.0)
COMPATIBLE_RADII_MM = (4.0, 5.0)
ALL_RADII_MM = (4.0, 5.0, 6.0)
POST_CONTACT_DEPTHS_MM = (0.75, 1.50)
INITIAL_GAP_MM = 0.25
CELL_HALF_WIDTH_MM = 5.5
CELL_DEPTH_MM = 11.0
CONTACT_TOLERANCE_MM = 1.0e-3


@dataclass(frozen=True)
class ContactCondition:
    """One immutable contact condition used by the validation matrix."""

    normalized_location: float
    sphere_radius_mm: float
    post_contact_travel_mm: float

    def __post_init__(self) -> None:
        values = (
            ("normalized_location", self.normalized_location, 0.0, 1.0),
            ("sphere_radius_mm", self.sphere_radius_mm, 0.0, float("inf")),
            ("post_contact_travel_mm", self.post_contact_travel_mm, 0.0, float("inf")),
        )
        for name, value, lower, upper in values:
            resolved = float(value)
            if not np.isfinite(resolved) or not lower <= resolved <= upper:
                raise ValueError(f"{name} is outside its finite physical range")
            object.__setattr__(self, name, resolved)
        if self.sphere_radius_mm <= 0.0 or self.post_contact_travel_mm <= 0.0:
            raise ValueError("sphere radius and post-contact travel must be positive")


def condition_identity(
    morphology_fingerprint: str,
    condition: ContactCondition,
    *,
    initial_gap_mm: float = INITIAL_GAP_MM,
) -> str:
    """Return a collision-resistant identity for one condition artifact."""

    payload = {
        "schema": "contact-condition-validation-v1",
        "morphology_fingerprint": str(morphology_fingerprint),
        "normalized_location": float(condition.normalized_location),
        "sphere_radius_mm": float(condition.sphere_radius_mm),
        "post_contact_travel_mm": float(condition.post_contact_travel_mm),
        "initial_gap_mm": float(initial_gap_mm),
        "cell_depth_mm": CELL_DEPTH_MM,
        "mechanics": {
            "max_load_increment_mm": SEARCH_MAX_LOAD_INCREMENT_MM,
            "vbd_iterations": SEARCH_VBD_ITERATIONS,
            "dt_s": SEARCH_DT_S,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def cell_end_clearance_mm(radius_mm: float) -> float:
    """Return the remaining longitudinal clearance in the fixed 11 mm cell."""

    radius = float(radius_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    return CELL_HALF_WIDTH_MM - radius


def frozen_mechanics_contract(travel_mm: float) -> dict[str, float | int]:
    """Expose the unchanged SEARCH mechanics contract for reports/tests."""

    return {
        "max_load_increment_mm": SEARCH_MAX_LOAD_INCREMENT_MM,
        "vbd_iterations": SEARCH_VBD_ITERATIONS,
        "dt_s": SEARCH_DT_S,
        "load_steps": load_steps_for_increment(
            travel_mm, max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM
        ),
        "load_steps_by_depth_mm": {
            "0.75": load_steps_for_increment(
                0.75, max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM
            ),
            "1.50": load_steps_for_increment(
                1.50, max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM
            ),
        },
        "soft_contact_margin_mm": SEARCH_SOFT_CONTACT_MARGIN_MM,
        "soft_contact_ke": SEARCH_SOFT_CONTACT_KE,
        "soft_contact_kd": SEARCH_SOFT_CONTACT_KD,
        "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
    }


def validation_morphologies() -> dict[str, FingertipParameters]:
    """Return the nominal and deterministic, already validated probe."""

    return {
        "production_nominal": FingertipParameters(void_height=0.25),
        "shallow_wide_probe": FingertipParameters(
            flat_pad_height=2.0,
            semielliptical_pad_height=6.0,
            stem_width=6.0,
            stem_height=2.0,
            void_width=2.0,
            void_height=0.0,
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _parameters_payload(parameters: FingertipParameters) -> dict[str, Any]:
    return {
        name: _json_default(value)
        for name, value in parameters.__dict__.items()
        if name not in {"arc_resolution", "geometry_tolerance"}
    } | {
        "arc_resolution": parameters.arc_resolution,
        "geometry_tolerance": parameters.geometry_tolerance,
    }


def _morphology_fingerprint(tip: Fingertip) -> str:
    return str(tip.solid(extrusion_depth_mm=CELL_DEPTH_MM).morphology_fingerprint)


def _vector2(value: Iterable[float]) -> list[float]:
    return [float(item) for item in tuple(value)[:2]]


def _pose2(pose: Any) -> list[float]:
    return _vector2(pose.translation_mm)


def _contact_geometry(
    *,
    morphology_id: str,
    parameters: FingertipParameters,
    condition: ContactCondition,
    morphology_fingerprint: str,
) -> dict[str, Any]:
    """Register one geometry condition without invoking Newton."""

    tip = Fingertip(parameters)
    sphere_mesh = make_sphere_mesh(
        condition.sphere_radius_mm, subdivisions=SEARCH_SPHERE_SUBDIVISIONS
    )
    solid = tip.solid(extrusion_depth_mm=CELL_DEPTH_MM)
    contact_surface = make_outer_compliant_surface(solid)
    alignment = sphere_alignment_at_normalized_location(
        tip.geometry,
        sphere_mesh,
        condition.normalized_location,
        initial_gap_mm=INITIAL_GAP_MM,
    )
    record: dict[str, Any] = {
        "morphology_id": morphology_id,
        "morphology_fingerprint": morphology_fingerprint,
        "condition_identity": condition_identity(morphology_fingerprint, condition),
        **asdict(condition),
        "target_point_mm": list(alignment.target_point_mm),
        "outward_normal": list(alignment.outward_normal),
        "approach_direction": list(alignment.approach_direction),
        "nominal_pose_mm": list(alignment.nominal_pose.translation_mm),
        "sphere_mesh_radius_mm": float(
            np.max(np.linalg.norm(sphere_mesh.vertices_mm, axis=1))
        ),
        "cell_end_clearance_mm": cell_end_clearance_mm(
            condition.sphere_radius_mm
        ),
        "geometry_valid": False,
        "contact_valid": False,
        "mechanics_status": "not_run",
        "failure_class": None,
        "failure_message": None,
        "first_contact": None,
        "_alignment": alignment,
        "_sphere_mesh": sphere_mesh,
        "_tip": tip,
    }
    try:
        nominal_collision = intersects(
            contact_surface, sphere_mesh, alignment.nominal_pose
        )
        if nominal_collision:
            raise RuntimeError("nominal sphere pose is not collision-free")
        first_contact = find_first_contact(
            contact_surface,
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
            FirstContactSettings(
                coarse_step_mm=0.25,
                tolerance_mm=CONTACT_TOLERANCE_MM,
                spawn_clearance_mm=0.05,
                max_travel_mm=20.0,
            ),
        )
        if intersects(contact_surface, sphere_mesh, first_contact.spawn_pose):
            raise RuntimeError("spawn pose is not collision-free")
        boundary_clearance = _unintended_boundary_clearance_mm(
            tip, sphere_mesh, alignment, first_contact
        )
        if boundary_clearance <= 0.0:
            raise RuntimeError(
                "an unintended external boundary is reached before arc contact: "
                f"clearance={boundary_clearance:g} mm"
            )
        record.update(
            {
                "geometry_valid": True,
                "contact_valid": record["cell_end_clearance_mm"] > 0.0,
                "unintended_boundary_clearance_mm": float(boundary_clearance),
                "first_contact": first_contact,
                "clear_pose_mm": _pose2(first_contact.clear_pose),
                "hit_pose_mm": _pose2(first_contact.hit_pose),
                "first_contact_pose_mm": _pose2(first_contact.contact_pose),
                "spawn_pose_mm": _pose2(first_contact.spawn_pose),
                "first_contact_travel_mm": float(
                    first_contact.travel_to_contact_mm
                ),
                "first_contact_bracket_width_mm": float(
                    first_contact.bracket_width_mm
                ),
                "spawn_clearance_mm": float(first_contact.spawn_clearance_mm),
            }
        )
        if record["cell_end_clearance_mm"] <= 0.0:
            record.update(
                {
                    "failure_class": "CURRENT_DOMAIN_INCOMPATIBLE",
                    "failure_message": (
                        "sphere exceeds the available longitudinal half-width of "
                        "the fixed 11 mm representative cell"
                    ),
                }
            )
        else:
            record["failure_class"] = None
    except Exception as exc:
        record.update(
            {
                "failure_class": "contact_failure",
                "failure_message": f"{type(exc).__name__}: {exc}",
                "geometry_valid": False,
                "contact_valid": False,
            }
        )
    return record


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_") and key != "first_contact"
    }


def _apply_mechanics_record(
    record: dict[str, Any],
    case: Any,
) -> None:
    case_payload = case.to_dict()
    record.update(
        {
            "mechanics_status": "passed",
            "final_pose_mm": _pose2(case.indentation.final_indenter_pose),
            "final_pose_error_mm": float(case_payload["final_pose_error_mm"]),
            "max_displacement_mm": float(case_payload["max_displacement_mm"]),
            "max_support_displacement_mm": float(
                case_payload["max_support_displacement_mm"]
            ),
            "rms_displacement_mm": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            np.asarray(case.indentation.mechanics_result.displacement)
                            ** 2,
                            axis=1,
                        )
                    )
                )
            ),
            "inverted_tetrahedra": int(case_payload["inverted_tetrahedra"]),
            "max_soft_contact_count": int(case_payload["max_soft_contact_count"]),
            "max_soft_contact_overflow": int(
                case_payload["max_soft_contact_overflow"]
            ),
            "max_rigid_contact_overflow": int(
                case_payload["max_rigid_contact_overflow"]
            ),
            "carrier_contact_count": int(
                case.indentation.diagnostics.get("carrier_interface_contact_count", 0)
            ),
            "carrier_contact_vertex_count": int(
                case.indentation.diagnostics.get("carrier_contact_vertex_count", 0)
            ),
            "mechanics_diagnostics": dict(case.indentation.diagnostics),
            "mechanics_artifact_path": case.mechanics_artifact_path,
            "mechanics_artifact_sha256": case.mechanics_artifact_sha256,
            "failure_class": None,
            "failure_message": None,
        }
    )
    checks = (
        np.all(np.isfinite(case.indentation.mechanics_result.deformed_vertices)),
        record["inverted_tetrahedra"] == 0,
        record["max_soft_contact_overflow"] == 0,
        record["max_rigid_contact_overflow"] == 0,
        record["max_support_displacement_mm"] <= 1.0e-9,
        record["final_pose_error_mm"] <= 1.0e-6,
        record.get("unintended_boundary_clearance_mm", -1.0) > 0.0,
        record["cell_end_clearance_mm"] > 0.0,
    )
    if not all(checks):
        record["mechanics_status"] = "failed_contract"
        record["failure_class"] = "mechanics_contract_failure"
        record["failure_message"] = "one or more frozen mechanics checks failed"


def _condition_record_for_depth(
    base_record: Mapping[str, Any], depth_mm: float
) -> dict[str, Any]:
    """Copy geometry provenance before attaching depth-specific metadata."""

    depth = float(depth_mm)
    record = dict(base_record)
    record["post_contact_travel_mm"] = depth
    record["load_steps"] = load_steps_for_increment(
        depth, max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM
    )
    record["condition_identity"] = condition_identity(
        record["morphology_fingerprint"],
        ContactCondition(
            float(record["normalized_location"]),
            float(record["sphere_radius_mm"]),
            depth,
        ),
    )
    return record


def _classify_mechanics_exception(exc: Exception) -> tuple[str, str]:
    """Separate pre-Newton mesh rejection from an actual solver failure."""

    if isinstance(
        exc,
        (
            VolumeMeshDependencyError,
            VolumeMeshingError,
            InvalidFingertipMechanicsMesh,
        ),
    ):
        return "mesh_failure", "mesh_failure"
    return "solver_failed", "solver_failed"


def _run_mechanics(
    output: Path,
    morphologies: Mapping[str, FingertipParameters],
    geometry_records: Mapping[tuple[str, float, float], dict[str, Any]],
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float, float, float], Any]]:
    records: list[dict[str, Any]] = []
    cases: dict[tuple[str, float, float, float], Any] = {}
    for morphology_id, parameters in morphologies.items():
        for radius in COMPATIBLE_RADII_MM:
            for depth in POST_CONTACT_DEPTHS_MM:
                selected = [
                    _condition_record_for_depth(
                        geometry_records[(morphology_id, location, radius)], depth
                    )
                    for location in NORMALIZED_LOCATIONS
                ]
                if not all(item["contact_valid"] for item in selected):
                    for item in selected:
                        item["mechanics_status"] = "not_run_contact_invalid"
                        records.append(_public_record(item))
                    continue
                morphology_root = output / "artifacts" / morphology_id
                identity = condition_identity(
                    selected[0]["morphology_fingerprint"],
                    ContactCondition(0.5, radius, depth),
                )
                try:
                    result = run_multi_location_sphere_contact(
                        parameters=parameters,
                        device=device,
                        radius_mm=radius,
                        travel_mm=depth,
                        initial_gap_mm=INITIAL_GAP_MM,
                        normalized_locations=DEFAULT_LOCATION_U,
                        artifact_dir=morphology_root / f"r_{radius:g}" / f"d_{depth:g}" / identity,
                        sphere_subdivisions=SEARCH_SPHERE_SUBDIVISIONS,
                        max_load_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM,
                        vbd_iterations=SEARCH_VBD_ITERATIONS,
                        carrier_contact=True,
                    )
                    by_location = {
                        float(case.normalized_location): case
                        for case in result.locations
                    }
                    for item in selected:
                        case = by_location[float(item["normalized_location"])]
                        _apply_mechanics_record(item, case)
                        cases[
                            (
                                morphology_id,
                                float(item["normalized_location"]),
                                radius,
                                depth,
                            )
                        ] = case
                        records.append(_public_record(item))
                except Exception as exc:
                    status, failure_class = _classify_mechanics_exception(exc)
                    for item in selected:
                        item.update(
                            {
                                "mechanics_status": status,
                                "failure_class": failure_class,
                                "failure_message": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        records.append(_public_record(item))
    return records, cases


def _load_cached_mechanics(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float, float, float], Any]]:
    """Rehydrate report-level case handles without rerunning Newton."""

    payload = json.loads(path.read_text())
    records = list(payload.get("records", ()))
    cases: dict[tuple[str, float, float, float], Any] = {}
    for record in records:
        # Early validation runs wrote the group depth only into the artifact
        # path.  Repair that report-only field while reusing the valid arrays.
        artifact_path = str(record.get("mechanics_artifact_path", ""))
        for candidate in POST_CONTACT_DEPTHS_MM:
            if f"/d_{candidate:g}/" in artifact_path:
                record["post_contact_travel_mm"] = candidate
                break
        record["condition_identity"] = condition_identity(
            record["morphology_fingerprint"],
            ContactCondition(
                float(record["normalized_location"]),
                float(record["sphere_radius_mm"]),
                float(record["post_contact_travel_mm"]),
            ),
        )
        record["load_steps"] = load_steps_for_increment(
            float(record["post_contact_travel_mm"]),
            max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM,
        )
        if record.get("mechanics_status") != "passed":
            continue
        final = tuple(float(value) for value in record["final_pose_mm"][:2])
        diagnostics = dict(record.get("mechanics_diagnostics", {}))
        indentation = SimpleNamespace(
            diagnostics=diagnostics,
            final_indenter_pose=RigidPose3D(
                translation_mm=(final[0], final[1], 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        cases[
            (
                str(record["morphology_id"]),
                float(record["normalized_location"]),
                float(record["sphere_radius_mm"]),
                float(record["post_contact_travel_mm"]),
            )
        ] = SimpleNamespace(
            normalized_location=float(record["normalized_location"]),
            indentation=indentation,
            mechanics_artifact_path=record["mechanics_artifact_path"],
            mechanics_artifact_sha256=record["mechanics_artifact_sha256"],
        )
    return records, cases


def _contact_state(record: Mapping[str, Any], case: Any, source_node_ids: np.ndarray) -> dict[str, Any]:
    local_indices = tuple(
        int(index)
        for index in case.indentation.diagnostics.get("carrier_contact_vertex_indices", ())
    )
    source_ids = tuple(
        int(source_node_ids[index])
        for index in local_indices
        if 0 <= index < len(source_node_ids)
    )
    return {
        "contact_state_fingerprint": fingerprint_mapping(
            {
                "morphology_fingerprint": record["morphology_fingerprint"],
                "normalized_location": record["normalized_location"],
                "radius_mm": record["sphere_radius_mm"],
                "travel_mm": record["post_contact_travel_mm"],
                "carrier_contact_source_node_ids": source_ids,
            }
        ),
        "normalized_location": float(record["normalized_location"]),
        "target_point_mm": record["target_point_mm"],
        "outward_normal": record["outward_normal"],
        "approach_direction": record["approach_direction"],
        "indenter_radius_mm": float(record["sphere_radius_mm"]),
        "initial_gap_mm": INITIAL_GAP_MM,
        "first_contact_travel_mm": float(record["first_contact_travel_mm"]),
        "post_contact_travel_mm": float(record["post_contact_travel_mm"]),
        "spawn_clearance_mm": float(record["spawn_clearance_mm"]),
        "carrier_contact_active": bool(
            case.indentation.diagnostics.get("carrier_contact_active", False)
        ),
        "carrier_contact_source_node_ids": list(source_ids),
    }


def _optical_grid_fingerprint(settings: Any) -> str:
    return fingerprint_mapping(
        {
            "mode": settings.mode,
            "x_bounds_mm": settings.x_bounds_mm,
            "y_bounds_mm": settings.y_bounds_mm,
            "surface_u_bins": settings.surface_u_bins,
            "surface_z_bins": settings.surface_z_bins,
            "internal_grid_width": settings.internal_grid_width,
            "internal_grid_height": settings.internal_grid_height,
            "internal_z_bins": settings.internal_z_bins,
            "extrusion_depth_mm": CELL_DEPTH_MM,
        }
    )


def _run_optics(
    output: Path,
    parameters: FingertipParameters,
    geometry_records: Mapping[tuple[str, float, float], dict[str, Any]],
    cases: Mapping[tuple[str, float, float, float], Any],
) -> dict[str, Any]:
    selected_conditions = (
        (0.25, 4.0, 0.75),
        (0.25, 4.0, 1.50),
        (0.50, 5.0, 0.75),
        (0.50, 5.0, 1.50),
        (0.75, 4.0, 1.50),
    )
    settings = _optical_settings()
    grid_fingerprint = _optical_grid_fingerprint(settings)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        tip = Fingertip(parameters)
        volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
        prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
        configuration = transport_configuration(
            settings,
            material=_material(tip),
            source={"model": "existing Fingertip optical source"},
        )
        runtime = create_runtime()
        transport = OptiXTransport()
        morphology_id = "production_nominal"
        morphology_fingerprint = str(volume_mesh.morphology_fingerprint)
        for location, radius, depth in selected_conditions:
            key = (morphology_id, location, radius, depth)
            case = cases.get(key)
            geometry = _condition_record_for_depth(
                geometry_records[(morphology_id, location, radius)], depth
            )
            record: dict[str, Any] = {
                "condition_identity": condition_identity(
                    geometry["morphology_fingerprint"],
                    ContactCondition(location, radius, depth),
                ),
                "normalized_location": location,
                "sphere_radius_mm": radius,
                "post_contact_travel_mm": depth,
                "optical_grid_fingerprint": grid_fingerprint,
                "status": "failed",
            }
            if case is None:
                record.update(
                    {
                        "failure_class": "mechanics_not_available",
                        "failure_message": "selected mechanics artifact did not pass",
                    }
                )
                records.append(record)
                continue
            contact_state = _contact_state(geometry, case, prepared.source_node_ids)
            restored = restore_deformed_optical_state(
                tip,
                volume_mesh,
                prepared,
                case.mechanics_artifact_path,
                case.mechanics_artifact_sha256,
                carrier_optics=CarrierOptics("absorber"),
                metadata={
                    "contact_state_fingerprint": contact_state[
                        "contact_state_fingerprint"
                    ],
                    "contact_location_u": location,
                    "observation_level": LUMO3D_OBSERVATION_LEVEL,
                    "full3d_surface_provenance": "actual_deformed_3d_volume_state",
                    "carrier_optical_boundary_model": "absorber",
                    "carrier_mapping_tolerance_mm": 0.5
                    * float(
                        case.indentation.diagnostics.get(
                            "rigid_sdf_target_voxel_mm", 0.125
                        )
                    ),
                },
            )
            result = transport.trace(
                tip,
                restored.geometry,
                settings=settings,
                morphology_id=morphology_id,
                morphology_fingerprint=morphology_fingerprint,
                mechanics_source=str(restored.artifact_path),
                mechanics_dimension="3D",
                contact_state=contact_state,
                transport_configuration=configuration,
                runtime=runtime,
            )
            field = np.asarray(result.field)
            carrier_triangles = int(
                result.path_diagnostics.get("carrier_interface", {})
                .get("contact_triangle_count", 0)
            )
            energy_ok = bool(
                np.isfinite(result.energy_balance_error)
                and result.energy_balance_error <= 1.0e-6
            )
            field_ok = bool(np.all(np.isfinite(field)) and np.all(field >= 0.0))
            # A selected condition may legitimately miss the carrier while the
            # carrier mapping remains installed.  Contact triangle count is a
            # diagnostic, not a required positive physical outcome.
            mapping_ok = bool(restored.geometry.carrier_optics is not None)
            if not (energy_ok and field_ok and mapping_ok):
                raise RuntimeError(
                    "optical smoke contract failed: "
                    f"field={field_ok}, energy={energy_ok}, mapping={mapping_ok}"
                )
            artifact = output / "artifacts" / "optics" / f"{geometry['condition_identity']}.json"
            save_case_artifact(
                artifact,
                result,
                {
                    "schema": "contact-condition-optics-smoke-v1",
                    "condition_identity": geometry["condition_identity"],
                    "morphology_id": morphology_id,
                    "morphology_fingerprint": morphology_fingerprint,
                    "contact_state": contact_state,
                    "transport_configuration": configuration,
                    "optical_grid_fingerprint": grid_fingerprint,
                },
            )
            records.append(
                {
                    **record,
                    "status": "passed",
                    "field_shape": list(field.shape),
                    "field_finite_nonnegative": field_ok,
                    "energy_balance_error": float(result.energy_balance_error),
                    "carrier_contact_triangle_count": carrier_triangles,
                    "carrier_mapping_ok": mapping_ok,
                    "transport_configuration_fingerprint": result.transport_configuration_fingerprint,
                    "artifact": str(artifact),
                }
            )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        seen = {
            (
                record.get("normalized_location"),
                record.get("sphere_radius_mm"),
                record.get("post_contact_travel_mm"),
            )
            for record in records
        }
        for location, radius, depth in selected_conditions:
            if (location, radius, depth) not in seen:
                records.append(
                    {
                        "normalized_location": location,
                        "sphere_radius_mm": radius,
                        "post_contact_travel_mm": depth,
                        "optical_grid_fingerprint": grid_fingerprint,
                        "status": "failed",
                        "failure_class": "optics_smoke_failed",
                        "failure_message": message,
                    }
                )
    passed = sum(record.get("status") == "passed" for record in records)
    return {
        "status": "PASS" if passed == len(selected_conditions) else "FAIL",
        "selected_conditions": [
            {
                "normalized_location": location,
                "sphere_radius_mm": radius,
                "post_contact_travel_mm": depth,
            }
            for location, radius, depth in selected_conditions
        ],
        "grid_fingerprint": grid_fingerprint,
        "records": records,
        "passed": passed,
        "attempted": len(selected_conditions),
        "runtime_s": time.perf_counter() - started,
    }


def _draw_plots(
    output: Path,
    morphologies: Mapping[str, FingertipParameters],
    geometry_records: Mapping[tuple[str, float, float], dict[str, Any]],
    cases: Mapping[tuple[str, float, float, float], Any],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def draw_outer(ax: Any, parameters: FingertipParameters, **kwargs: Any) -> None:
        coords = np.asarray(Fingertip(parameters).geometry.outer_pad_geometry.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], **kwargs)

    colors = {"production_nominal": "tab:blue", "shallow_wide_probe": "tab:orange"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for morphology_id, parameters in morphologies.items():
        draw_outer(ax, parameters, color=colors[morphology_id], label=morphology_id)
        for location in NORMALIZED_LOCATIONS:
            item = geometry_records[(morphology_id, location, 4.0)]
            target = np.asarray(item["target_point_mm"])
            normal = np.asarray(item["outward_normal"])
            ax.scatter(target[0], target[1], color=colors[morphology_id], s=18)
            ax.arrow(
                target[0], target[1], normal[0], normal[1],
                color=colors[morphology_id], alpha=0.7, width=0.01,
                length_includes_head=True,
            )
            ax.add_patch(
                Circle(
                    item["nominal_pose_mm"], 4.0, fill=False,
                    color=colors[morphology_id], alpha=0.25,
                )
            )
            if item.get("first_contact_pose_mm"):
                ax.add_patch(
                    Circle(
                        item["first_contact_pose_mm"], 4.0, fill=False,
                        color=colors[morphology_id], linestyle="--", alpha=0.65,
                    )
                )
    ax.set_title("Contact frames by morphology and normalized location")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / "contact_frames.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    parameters = morphologies["production_nominal"]
    draw_outer(ax, parameters, color="black", label="outer compliant boundary")
    for radius, color in zip(ALL_RADII_MM, ("tab:green", "tab:blue", "tab:red")):
        item = geometry_records[("production_nominal", 0.5, radius)]
        ax.add_patch(
            Circle(
                item["nominal_pose_mm"], radius, fill=False,
                color=color, label=f"r={radius:g} mm; clearance={item['cell_end_clearance_mm']:+.1f} mm",
            )
        )
        if item.get("first_contact_pose_mm"):
            ax.add_patch(Circle(item["first_contact_pose_mm"], radius, fill=False, color=color, linestyle="--", alpha=0.6))
    ax.set_title("Sphere radius and finite 11 mm cell")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plots / "radius_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    draw_outer(ax, parameters, color="black", label="outer compliant boundary")
    contact_item = geometry_records[("production_nominal", 0.5, 5.0)]
    ax.add_patch(Circle(contact_item["first_contact_pose_mm"], 5.0, fill=False, color="black", linestyle="--", label="same first contact"))
    for depth, color in zip(POST_CONTACT_DEPTHS_MM, ("tab:purple", "tab:red")):
        case = cases.get(("production_nominal", 0.5, 5.0, depth))
        if case is not None:
            ax.add_patch(Circle(case.indentation.final_indenter_pose.translation_mm[:2], 5.0, fill=False, color=color, label=f"final d={depth:g} mm"))
    ax.set_title("Post-first-contact indentation depth")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plots / "indentation_comparison.png", dpi=160); plt.close(fig)


def _write_conditions_csv(output: Path, records: Iterable[Mapping[str, Any]]) -> None:
    fields = (
        "morphology_id", "morphology_fingerprint", "normalized_location",
        "sphere_radius_mm", "post_contact_travel_mm", "target_point_x_mm",
        "target_point_y_mm", "outward_normal_x", "outward_normal_y",
        "approach_direction_x", "approach_direction_y", "first_contact_travel_mm",
        "first_contact_pose_x_mm", "first_contact_pose_y_mm", "final_pose_x_mm",
        "final_pose_y_mm", "load_steps", "cell_end_clearance_mm", "geometry_valid",
        "contact_valid", "mechanics_status", "max_displacement_mm",
        "inverted_tetrahedra", "failure_class", "failure_message",
    )
    path = output / "conditions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in records:
            target = item.get("target_point_mm", (None, None))
            normal = item.get("outward_normal", (None, None))
            approach = item.get("approach_direction", (None, None))
            first = item.get("first_contact_pose_mm", (None, None)) or (None, None)
            final = item.get("final_pose_mm", (None, None)) or (None, None)
            writer.writerow({
                "morphology_id": item.get("morphology_id"),
                "morphology_fingerprint": item.get("morphology_fingerprint"),
                "normalized_location": item.get("normalized_location"),
                "sphere_radius_mm": item.get("sphere_radius_mm"),
                "post_contact_travel_mm": item.get("post_contact_travel_mm"),
                "target_point_x_mm": target[0], "target_point_y_mm": target[1],
                "outward_normal_x": normal[0], "outward_normal_y": normal[1],
                "approach_direction_x": approach[0], "approach_direction_y": approach[1],
                "first_contact_travel_mm": item.get("first_contact_travel_mm"),
                "first_contact_pose_x_mm": first[0], "first_contact_pose_y_mm": first[1],
                "final_pose_x_mm": final[0], "final_pose_y_mm": final[1],
                "load_steps": load_steps_for_increment(item["post_contact_travel_mm"], max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM),
                "cell_end_clearance_mm": item.get("cell_end_clearance_mm"),
                "geometry_valid": item.get("geometry_valid"),
                "contact_valid": item.get("contact_valid"),
                "mechanics_status": item.get("mechanics_status"),
                "max_displacement_mm": item.get("max_displacement_mm"),
                "inverted_tetrahedra": item.get("inverted_tetrahedra"),
                "failure_class": item.get("failure_class"),
                "failure_message": item.get("failure_message"),
            })


def run_validation(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    device: str = "cuda:0",
    reuse_existing_mechanics: bool = False,
) -> dict[str, Any]:
    """Run the complete deterministic contact-condition validation."""

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    morphologies = validation_morphologies()
    tips = {name: Fingertip(parameters) for name, parameters in morphologies.items()}
    fingerprints = {name: _morphology_fingerprint(tip) for name, tip in tips.items()}
    config = {
        "schema": "multi-condition-parameter-validation-v1",
        "morphologies": {
            name: {
                "parameters": _parameters_payload(parameters),
                "morphology_fingerprint": fingerprints[name],
            }
            for name, parameters in morphologies.items()
        },
        "proposed_radii_mm": PROPOSED_RADII_MM,
        "compatible_radii_mm": COMPATIBLE_RADII_MM,
        "normalized_locations": NORMALIZED_LOCATIONS,
        "post_contact_depths_mm": POST_CONTACT_DEPTHS_MM,
        "initial_gap_mm": INITIAL_GAP_MM,
        "cell_depth_mm": CELL_DEPTH_MM,
        "cell_end_clearance_rule": "5.5 - radius_mm",
        "mechanics_contract": frozen_mechanics_contract(1.5),
        "optics": {"mode": "FULL_3D", "selected_state_count": 5},
        "device": device,
        "bo_run": False,
    }
    _write_json(root / "config.json", config)

    geometry_records: dict[tuple[str, float, float], dict[str, Any]] = {}
    for morphology_id, parameters in morphologies.items():
        for radius in ALL_RADII_MM:
            for location in NORMALIZED_LOCATIONS:
                condition = ContactCondition(location, radius, 0.75)
                geometry_records[(morphology_id, location, radius)] = _contact_geometry(
                    morphology_id=morphology_id,
                    parameters=parameters,
                    condition=condition,
                    morphology_fingerprint=fingerprints[morphology_id],
                )

    condition_matrix = [
        {
            "morphology_id": morphology_id,
            "morphology_fingerprint": fingerprints[morphology_id],
            **asdict(ContactCondition(location, radius, depth)),
            "condition_identity": condition_identity(
                fingerprints[morphology_id], ContactCondition(location, radius, depth)
            ),
            "cell_end_clearance_mm": cell_end_clearance_mm(radius),
            "requested_set": "proposed" if radius in PROPOSED_RADII_MM else "compatible",
        }
        for morphology_id in morphologies
        for radius in ALL_RADII_MM
        for location in NORMALIZED_LOCATIONS
        for depth in POST_CONTACT_DEPTHS_MM
    ]
    _write_json(root / "condition_matrix.json", {"records": condition_matrix})
    cached_mechanics = root / "mechanics_validation.json"
    if reuse_existing_mechanics and cached_mechanics.is_file():
        mechanics_records, cases = _load_cached_mechanics(cached_mechanics)
    else:
        mechanics_records, cases = _run_mechanics(
            root, morphologies, geometry_records, device=device
        )
    mechanics_status = "PASS" if all(
        item["mechanics_status"] == "passed" for item in mechanics_records
    ) else "FAIL"
    _write_json(root / "mechanics_validation.json", {
        "status": mechanics_status,
        "records": mechanics_records,
        "frozen_contract": frozen_mechanics_contract(1.5),
        "compatible_state_count": len(mechanics_records),
    })

    optics = _run_optics(root, morphologies["production_nominal"], geometry_records, cases)
    _write_json(root / "optics_smoke.json", optics)
    _draw_plots(root, morphologies, geometry_records, cases)

    mechanics_by_identity = {
        item["condition_identity"]: item for item in mechanics_records
    }
    condition_records: list[dict[str, Any]] = []
    for matrix_item in condition_matrix:
        key = (
            matrix_item["morphology_id"],
            float(matrix_item["normalized_location"]),
            float(matrix_item["sphere_radius_mm"]),
        )
        row = _public_record(geometry_records[key])
        row.update(matrix_item)
        row.update(mechanics_by_identity.get(matrix_item["condition_identity"], {}))
        condition_records.append(row)
    _write_json(root / "contact_validation.json", {
        "status": "PASS_WITH_DOMAIN_LIMITATION" if all(
            item["geometry_valid"] for item in condition_records
        ) else "FAIL",
        "records": condition_records,
        "depth_independent_contact": True,
    })
    _write_conditions_csv(root, condition_records)
    compatible = [
        item for item in mechanics_records if item["cell_end_clearance_mm"] > 0.0
    ]
    proposed_r6 = [
        item for item in condition_records if item["sphere_radius_mm"] == 6.0
    ]
    semantics = {
        "radius_propagation": all(
            np.isclose(item["sphere_mesh_radius_mm"], item["sphere_radius_mm"], rtol=0.0, atol=1.0e-10)
            for item in geometry_records.values()
        ),
        "depth_propagation": all(
            load_steps_for_increment(depth, max_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM)
            == (15 if depth == 0.75 else 30)
            for depth in POST_CONTACT_DEPTHS_MM
        ),
        "u_propagation": len({
            tuple(geometry_records[("production_nominal", u, 4.0)]["target_point_mm"])
            for u in NORMALIZED_LOCATIONS
        }) == 3,
        "morphology_specific_contact": len({
            tuple(geometry_records[(m, 0.5, 4.0)]["target_point_mm"])
            for m in morphologies
        }) == 2,
        "cache_safe": len({
            item["condition_identity"] for item in condition_matrix
        }) == len(condition_matrix),
        "frozen_mechanics": frozen_mechanics_contract(0.75)["load_steps"] == 15 and frozen_mechanics_contract(1.5)["load_steps"] == 30,
    }
    r6_ok = bool(proposed_r6) and all(
        item["failure_class"] == "CURRENT_DOMAIN_INCOMPATIBLE"
        for item in proposed_r6
    )
    compatible_ok = all(semantics.values()) and all(
        item["mechanics_status"] == "passed" for item in compatible
    ) and optics["status"] == "PASS"
    r6_is_limited = r6_ok and all(
        item["failure_class"] == "CURRENT_DOMAIN_INCOMPATIBLE"
        for item in proposed_r6
    )
    if not compatible_ok or not r6_ok:
        status = "FAIL"
    elif r6_is_limited:
        status = "PASS_WITH_LIMITATION"
    else:
        status = "PASS"
    summary = {
        "status": status,
        "nominal_compatible_states": {
            "passed": sum(item["mechanics_status"] == "passed" and item["morphology_id"] == "production_nominal" for item in mechanics_records),
            "attempted": sum(item["morphology_id"] == "production_nominal" for item in mechanics_records),
        },
        "second_morphology_compatible_states": {
            "passed": sum(item["mechanics_status"] == "passed" and item["morphology_id"] == "shallow_wide_probe" for item in mechanics_records),
            "attempted": sum(item["morphology_id"] == "shallow_wide_probe" for item in mechanics_records),
        },
        "proposed_r6_states": {
            "valid": sum(item["failure_class"] is None for item in proposed_r6),
            "current_domain_incompatible": sum(item["failure_class"] == "CURRENT_DOMAIN_INCOMPATIBLE" for item in proposed_r6),
            "solver_failed": sum(item["failure_class"] == "solver_failed" for item in proposed_r6),
        },
        "semantics": semantics,
        "optics_downstream_smoke": optics["status"],
        "mechanics_status": mechanics_status,
        "compatible_radius_set_mm": COMPATIBLE_RADII_MM,
        "proposed_radius_set_mm": PROPOSED_RADII_MM,
        "message": (
            "The simulator correctly supports multi-condition evaluation over "
            "contact location, compatible sphere radius, and indentation depth."
            if status == "PASS" else
            "The multi-condition semantics are correct, but the current finite-width "
            "3D cell limits the maximum usable sphere radius."
        ),
    }
    _write_json(root / "summary.json", summary)
    (root / "reviewer_audit.md").write_text(
        "# Reviewer audit\n\nPending fresh independent reviewer after deterministic validation.\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reuse-existing-mechanics",
        action="store_true",
        help="reuse a completed mechanics_validation.json without rerunning Newton",
    )
    args = parser.parse_args(argv)
    summary = run_validation(
        args.output,
        device=args.device,
        reuse_existing_mechanics=args.reuse_existing_mechanics,
    )
    print(f"{summary['status']}: {summary['message']}")
    return 0 if summary["status"] in {"PASS", "PASS_WITH_LIMITATION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_RADII_MM",
    "COMPATIBLE_RADII_MM",
    "ContactCondition",
    "NORMALIZED_LOCATIONS",
    "POST_CONTACT_DEPTHS_MM",
    "PROPOSED_RADII_MM",
    "cell_end_clearance_mm",
    "condition_identity",
    "frozen_mechanics_contract",
    "run_validation",
    "validation_morphologies",
]
