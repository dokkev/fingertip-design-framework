"""Optimization-boundary persistence for native FULL_3D optical results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from model.fingertip import Fingertip
from optics.transport3d.result import Transport3DResult
from optics.transport3d.settings import Transport3DSettings


UNIFIED_ARTIFACT_SCHEMA = "unified-optix-transport-case-v5"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Return the stable fingerprint used by optimization contracts."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def optical_physics_parameters(tip: Fingertip) -> dict[str, float]:
    """Return exactly the optical values used by FULL_3D transport."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    return {
        "refractive_index_air": float(tip.optical.refractive_index_air),
        "refractive_index_silicone": float(tip.optical.refractive_index_silicone),
        "absorption_per_mm": float(tip.optical.absorption_per_mm),
        "relative_radiant_power": float(tip.led.relative_radiant_power),
        "emission_half_angle_deg": float(tip.led.emission_half_angle_deg),
    }


def transport_configuration(
    settings: Transport3DSettings,
    *,
    material: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the common settings/material contract for case fingerprints."""
    configuration: dict[str, Any] = {
        "settings": asdict(settings),
        "material": dict(material),
        "source_sampling": "optics.transport3d.sampling.sample_directions",
        "physics": "optics.transport3d.physics.interface_split+attenuation",
        "accumulation": "native P3(x,y,z)",
    }
    if source is not None:
        configuration["source"] = dict(source)
    return configuration


@dataclass(frozen=True)
class OpticalFieldArtifact:
    """Loaded persistence record, kept outside the optics execution core."""

    field: np.ndarray
    field_axes: tuple[np.ndarray, ...]
    total_transport: float
    launched_weight: float
    escaped_weight: float
    absorbed_weight: float
    terminated_weight: float
    energy_balance_error: float
    ray_count: int
    escape_event_count: int
    escaped_primary_count: int
    path_diagnostics: Mapping[str, Any]
    morphology_id: str = ""
    morphology_fingerprint: str = ""
    mechanics_source: str = ""
    mechanics_dimension: str = "3D"
    contact_state: Mapping[str, Any] = field(default_factory=dict)
    transport_configuration_fingerprint: str = ""
    object_absorbed_weight: float = 0.0
    object_transmitted_weight: float = 0.0
    object_interface_incident_weight: float = 0.0
    object_reflected_weight: float = 0.0
    carrier_absorbed_weight: float = 0.0
    carrier_transmitted_weight: float = 0.0
    carrier_interface_incident_weight: float = 0.0
    carrier_reflected_weight: float = 0.0
    carrier_contact_triangle_count: int = 0
    optical_mode: str = "FULL_3D"


def energy_record(result: Transport3DResult | OpticalFieldArtifact) -> dict[str, Any]:
    """Serialize scalar transport diagnostics at the optimization boundary."""
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
        "carrier_interface_incident_weight": float(result.carrier_interface_incident_weight),
        "carrier_reflected_weight": float(result.carrier_reflected_weight),
        "carrier_optical_contact_triangle_count": int(
            result.carrier_contact_triangle_count
        ),
        "energy_balance_error": float(result.energy_balance_error),
        "field_shape": list(result.field.shape),
        "field_finite_nonnegative": bool(
            np.all(np.isfinite(result.field)) and np.all(result.field >= 0.0)
        ),
    }


def _path_diagnostics(
    result: Transport3DResult, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Build report-oriented diagnostics without storing them in optics core."""
    transport = contract.get("transport_configuration", {})
    settings = transport.get("settings", {}) if isinstance(transport, Mapping) else {}
    return {
        "branch_cutoff": {
            "minimum_ray_weight_fraction": settings.get("minimum_ray_weight"),
            "maximum_interactions": settings.get("max_interactions"),
            "primary_and_first_generation_exempt": True,
            "convention": "cutoff applies to branches with interaction_count > 1",
        },
        "processed_segment_count": int(result.processed_segment_count),
        "periodic_wrap_termination": {
            "enabled": bool(settings.get("terminate_on_periodic_wrap_limit", False)),
            "count": int(result.periodic_wrap_termination_count),
            "weight": float(result.periodic_wrap_termination_weight),
            "maximum_periodic_wraps": settings.get("maximum_periodic_wraps"),
        },
        "no_event_termination": {
            "enabled": bool(settings.get("terminate_on_no_event", False)),
            "count": int(result.no_event_termination_count),
            "weight": float(result.no_event_termination_weight),
        },
        "interface_normal_orientation_fallback_count": int(
            result.interface_normal_fallback_count
        ),
        "object_interface": {
            "object_absorbed_weight": float(result.object_absorbed_weight),
            "object_transmitted_weight": float(result.object_transmitted_weight),
            "object_interface_incident_weight": float(
                result.object_interface_incident_weight
            ),
            "object_reflected_weight": float(result.object_reflected_weight),
        },
        "carrier_interface": {
            "carrier_absorbed_weight": float(result.carrier_absorbed_weight),
            "carrier_transmitted_weight": float(result.carrier_transmitted_weight),
            "carrier_interface_incident_weight": float(
                result.carrier_interface_incident_weight
            ),
            "carrier_reflected_weight": float(result.carrier_reflected_weight),
            "contact_triangle_count": int(result.carrier_contact_triangle_count),
        },
        "internal_path_field": {
            "field_axis_order": "x,y,z",
            "field_shape": list(result.field.shape),
            "normalization": "raw weighted path length per voxel; no TV normalization",
            "z_integration": "sum of raw z-bin path masses; no extra width factor",
            "segment_medium_scope": "air and silicone segments in the native FULL_3D field",
            "line_sampling": "deterministic segment midpoint sampling",
            "total_accumulated_weighted_path_length_mm": float(np.sum(result.field)),
        },
    }


def _result_record(result: Transport3DResult, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "morphology_id": contract.get("morphology_id", ""),
        "morphology_fingerprint": contract.get(
            "morphology_fingerprint",
            contract.get("morphology_parameters_fingerprint", ""),
        ),
        "mechanics_source": contract.get("mechanics_source", ""),
        "mechanics_dimension": contract.get("mechanics_dimension", "3D"),
        "contact_state": _plain(contract.get("contact_state", {})),
        "optical_mode": "FULL_3D",
        "ray_count": result.launched_ray_count,
        "transport_configuration_fingerprint": contract.get(
            "transport_configuration_fingerprint", ""
        ),
        "total_transport": result.escaped_weight,
        "launched_weight": result.launched_weight,
        "escaped_weight": result.escaped_weight,
        "absorbed_weight": result.absorbed_weight,
        "terminated_weight": result.terminated_weight,
        "escape_event_count": int(result.escape_event_count),
        "escaped_primary_count": int(result.escaped_primary_count),
        "energy_balance_error": result.energy_balance_error,
        "object_absorbed_weight": result.object_absorbed_weight,
        "object_transmitted_weight": result.object_transmitted_weight,
        "object_interface_incident_weight": result.object_interface_incident_weight,
        "object_reflected_weight": result.object_reflected_weight,
        "carrier_absorbed_weight": result.carrier_absorbed_weight,
        "carrier_transmitted_weight": result.carrier_transmitted_weight,
        "carrier_interface_incident_weight": result.carrier_interface_incident_weight,
        "carrier_reflected_weight": result.carrier_reflected_weight,
        "carrier_contact_triangle_count": int(result.carrier_contact_triangle_count),
        "path_diagnostics": _plain(_path_diagnostics(result, contract)),
    }


def native_field_separability(first: Any, second: Any) -> dict[str, float | str | None]:
    """Evaluate raw magnitude and normalized redistribution in native space."""
    if first.field.shape != second.field.shape or any(
        not np.array_equal(left, right)
        for left, right in zip(first.field_axes, second.field_axes)
    ):
        raise ValueError("native separability requires identical field grids")
    left = np.asarray(first.field, dtype=float)
    right = np.asarray(second.field, dtype=float)
    difference = left - right
    first_mass = float(np.sum(left))
    second_mass = float(np.sum(right))
    if first_mass > 0.0 and second_mass > 0.0:
        normalized: float | None = 0.5 * float(
            np.sum(np.abs(left / first_mass - right / second_mass))
        )
        normalized_status = "valid"
    else:
        normalized = None
        normalized_status = "singular_zero_field"
    return {
        "optical_mode": "FULL_3D",
        "raw_l1": float(np.sum(np.abs(difference))),
        "raw_l2": float(np.linalg.norm(difference)),
        "normalized_redistribution_l1": normalized,
        "normalized_status": normalized_status,
        "first_native_field_mass": first_mass,
        "second_native_field_mass": second_mass,
        "first_total_transport": float(first.total_transport),
        "second_total_transport": float(second.total_transport),
        "total_transport_difference": float(
            first.total_transport - second.total_transport
        ),
    }


def save_case_artifact(
    path: Path, result: Transport3DResult, contract: Mapping[str, Any]
) -> None:
    """Persist one direct transport result at the optimization boundary."""
    field = result.field
    axes = result.field_axes
    path.parent.mkdir(parents=True, exist_ok=True)
    field_path = path.with_suffix(".npz")
    field_tmp = field_path.with_name(field_path.name + ".tmp")
    metadata_tmp = path.with_name(path.name + ".tmp")
    with field_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            field=np.asarray(field),
            **{f"axis_{index}": axis for index, axis in enumerate(axes)},
        )
    record = _result_record(result, contract)
    metadata = {
        "schema": UNIFIED_ARTIFACT_SCHEMA,
        "field_axis_order": "x,y,z",
        "contract": _plain(contract),
        "contract_fingerprint": fingerprint_mapping(dict(contract)),
        "field_artifact": str(field_path),
        "field_sha256": hashlib.sha256(field_tmp.read_bytes()).hexdigest(),
        "result": record,
    }
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    field_tmp.replace(field_path)
    metadata_tmp.replace(path)


def _owned_field(value: Any) -> np.ndarray:
    field = np.array(value, dtype=float, copy=True)
    if (
        field.ndim != 3
        or not field.size
        or not np.all(np.isfinite(field))
        or np.any(field < 0.0)
    ):
        raise ValueError("native transport field must be a finite nonnegative 3D field")
    field.setflags(write=False)
    return field


def load_case_artifact(
    path: Path, *, expected_contract: Mapping[str, Any]
) -> OpticalFieldArtifact:
    """Load a current artifact only when its complete contract matches."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != UNIFIED_ARTIFACT_SCHEMA:
        raise ValueError("unsupported unified transport artifact schema")
    record = metadata.get("result")
    if not isinstance(record, Mapping) or record.get("optical_mode") != "FULL_3D":
        raise ValueError("unified transport result metadata is missing")
    if metadata.get("field_axis_order") != "x,y,z":
        raise ValueError("unified transport artifact must contain x,y,z axes")
    if metadata.get("contract") != _plain(expected_contract):
        raise ValueError("unified transport artifact contract mismatch")
    if metadata.get("contract_fingerprint") != fingerprint_mapping(dict(expected_contract)):
        raise ValueError("unified transport artifact contract fingerprint mismatch")
    field_path = Path(str(metadata.get("field_artifact", "")))
    if not field_path.is_absolute():
        field_path = path.parent / field_path.name
    if (
        not field_path.exists()
        or hashlib.sha256(field_path.read_bytes()).hexdigest()
        != metadata.get("field_sha256")
    ):
        raise ValueError("unified transport field artifact is missing or corrupt")
    with np.load(field_path, allow_pickle=False) as archive:
        field = _owned_field(archive["field"])
        axes = tuple(
            np.array(archive[f"axis_{index}"], dtype=float, copy=True)
            for index in range(3)
        )
    if field.shape != tuple(len(axis) - 1 for axis in axes):
        raise ValueError("unified transport field shape does not match its axes")
    for axis in axes:
        if len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
            raise ValueError("unified transport field axes are invalid")
        axis.setflags(write=False)
    checks = {
        "morphology_id": expected_contract.get("morphology_id"),
        "mechanics_dimension": expected_contract.get("mechanics_dimension"),
        "mechanics_source": expected_contract.get("mechanics_source"),
        "ray_count": expected_contract.get("ray_count"),
    }
    for key, expected in checks.items():
        if expected is not None and record.get(key) != expected:
            raise ValueError(f"unified transport result {key} mismatches its contract")
    expected_morphology = expected_contract.get("morphology_parameters_fingerprint")
    if expected_morphology is None:
        expected_morphology = expected_contract.get("morphology_fingerprint")
    if (
        expected_morphology is not None
        and record.get("morphology_fingerprint") != expected_morphology
    ):
        raise ValueError("unified transport result morphology fingerprint mismatch")
    expected_contact = expected_contract.get("contact_state_fingerprint")
    if expected_contact is not None:
        contact_state = record.get("contact_state")
        if not isinstance(contact_state, Mapping) or contact_state.get(
            "contact_state_fingerprint"
        ) != expected_contact:
            raise ValueError("unified transport result contact-state fingerprint mismatch")
    expected_configuration = expected_contract.get("transport_configuration_fingerprint")
    if (
        expected_configuration is not None
        and record.get("transport_configuration_fingerprint") != expected_configuration
    ):
        raise ValueError(
            "unified transport result transport-configuration fingerprint mismatch"
        )
    return OpticalFieldArtifact(
        field=field,
        field_axes=axes,
        total_transport=float(record["total_transport"]),
        launched_weight=float(record["launched_weight"]),
        escaped_weight=float(record["escaped_weight"]),
        absorbed_weight=float(record["absorbed_weight"]),
        terminated_weight=float(record["terminated_weight"]),
        energy_balance_error=float(record["energy_balance_error"]),
        ray_count=int(record["ray_count"]),
        escape_event_count=int(record["escape_event_count"]),
        escaped_primary_count=int(record["escaped_primary_count"]),
        path_diagnostics=record.get("path_diagnostics", {}),
        morphology_id=str(record.get("morphology_id", "")),
        morphology_fingerprint=str(record.get("morphology_fingerprint", "")),
        mechanics_source=str(record.get("mechanics_source", "")),
        mechanics_dimension=str(record.get("mechanics_dimension", "3D")),
        contact_state=record.get("contact_state", {}),
        transport_configuration_fingerprint=str(
            record.get("transport_configuration_fingerprint", "")
        ),
        object_absorbed_weight=float(record.get("object_absorbed_weight", 0.0)),
        object_transmitted_weight=float(record.get("object_transmitted_weight", 0.0)),
        object_interface_incident_weight=float(
            record.get("object_interface_incident_weight", 0.0)
        ),
        object_reflected_weight=float(record.get("object_reflected_weight", 0.0)),
        carrier_absorbed_weight=float(record.get("carrier_absorbed_weight", 0.0)),
        carrier_transmitted_weight=float(record.get("carrier_transmitted_weight", 0.0)),
        carrier_interface_incident_weight=float(
            record.get("carrier_interface_incident_weight", 0.0)
        ),
        carrier_reflected_weight=float(record.get("carrier_reflected_weight", 0.0)),
        carrier_contact_triangle_count=int(
            record.get("carrier_contact_triangle_count", 0)
        ),
    )


__all__ = [
    "OpticalFieldArtifact",
    "UNIFIED_ARTIFACT_SCHEMA",
    "fingerprint_mapping",
    "load_case_artifact",
    "native_field_separability",
    "energy_record",
    "optical_physics_parameters",
    "save_case_artifact",
    "transport_configuration",
]
