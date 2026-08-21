"""Optimization-boundary persistence for native FULL_3D optical results."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lumo.ray_tracing.optical_mechanics.result import Transport3DResult
from lumo.optimization.optical_contract import fingerprint_mapping as _fingerprint_mapping


UNIFIED_ARTIFACT_SCHEMA = "unified-optix-transport-case-v7"


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


@dataclass(frozen=True)
class OpticalFieldArtifact:
    """Loaded persistence record, kept outside the optics execution core."""

    field: np.ndarray
    field_axes: tuple[np.ndarray, ...]
    total_transport: float
    launched_weight: float
    escaped_weight: float
    outgoing_surface_weight: float
    absorbed_weight: float
    terminated_weight: float
    termination_count: int
    segment_budget_termination_count: int
    segment_budget_termination_weight: float
    energy_balance_error: float
    ray_count: int
    escape_event_count: int
    escaped_primary_count: int
    processed_sample_count: int
    clipped_sample_count: int
    represented_weighted_path_length_mm: float
    clipped_weighted_path_length_mm: float
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

    def __post_init__(self) -> None:
        for name in (
            "processed_sample_count",
            "clipped_sample_count",
            "termination_count",
            "segment_budget_termination_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.clipped_sample_count > self.processed_sample_count:
            raise ValueError("clipped_sample_count cannot exceed processed_sample_count")
        for name in (
            "represented_weighted_path_length_mm",
            "clipped_weighted_path_length_mm",
            "segment_budget_termination_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.path_diagnostics, Mapping):
            raise ValueError("path_diagnostics must be an object")

    @property
    def processed_weighted_path_length_mm(self) -> float:
        """Return represented plus clipped path length from the artifact."""

        return float(
            self.represented_weighted_path_length_mm
            + self.clipped_weighted_path_length_mm
        )


def energy_record(result: Transport3DResult) -> dict[str, Any]:
    """Serialize scalar transport diagnostics at the optimization boundary."""
    launched = float(result.launched_weight)
    carrier_absorbed = float(result.carrier_absorbed_weight)
    escaped = float(result.escaped_weight)
    return {
        "launched_weight": launched,
        "escaped_weight": escaped,
        "escaped_transport_fraction": escaped / max(launched, 1.0e-30),
        "outgoing_surface_weight": float(result.outgoing_surface_weight),
        "absorbed_weight": float(result.absorbed_weight),
        "terminated_weight": float(result.terminated_weight),
        "termination_count": int(
            getattr(result, "termination_count", 0)
        ),
        "terminated_weight_fraction": float(
            getattr(
                result,
                "terminated_weight_fraction",
                float(result.terminated_weight) / max(launched, 1.0e-30),
            )
        ),
        "processed_segment_count": int(result.processed_segment_count),
        "processed_sample_count": int(getattr(result, "processed_sample_count", 0)),
        "clipped_sample_count": int(getattr(result, "clipped_sample_count", 0)),
        "represented_weighted_path_length_mm": float(
            getattr(result, "represented_weighted_path_length_mm", 0.0)
        ),
        "clipped_weighted_path_length_mm": float(
            getattr(result, "clipped_weighted_path_length_mm", 0.0)
        ),
        "processed_weighted_path_length_mm": float(
            getattr(result, "processed_weighted_path_length_mm", 0.0)
        ),
        "periodic_wrap_termination_count": int(
            result.periodic_wrap_termination_count
        ),
        "periodic_wrap_termination_weight": float(
            result.periodic_wrap_termination_weight
        ),
        "periodic_wrap_termination_fraction": float(
            result.periodic_wrap_termination_weight
        ) / max(launched, 1.0e-30),
        "no_event_termination_count": int(result.no_event_termination_count),
        "no_event_termination_weight": float(
            result.no_event_termination_weight
        ),
        "no_event_termination_fraction": float(
            result.no_event_termination_weight
        ) / max(launched, 1.0e-30),
        "branch_cutoff_termination_count": int(
            result.branch_cutoff_termination_count
        ),
        "branch_cutoff_termination_weight": float(
            result.branch_cutoff_termination_weight
        ),
        "branch_cutoff_termination_fraction": float(
            result.branch_cutoff_termination_weight
        ) / max(launched, 1.0e-30),
        "max_interaction_termination_count": int(
            result.max_interaction_termination_count
        ),
        "max_interaction_termination_weight": float(
            result.max_interaction_termination_weight
        ),
        "max_interaction_termination_fraction": float(
            result.max_interaction_termination_weight
        ) / max(launched, 1.0e-30),
        "segment_budget_termination_count": int(
            result.segment_budget_termination_count
        ),
        "segment_budget_termination_weight": float(
            result.segment_budget_termination_weight
        ),
        "segment_budget_termination_fraction": float(
            getattr(
                result,
                "segment_budget_termination_fraction",
                float(result.segment_budget_termination_weight)
                / max(launched, 1.0e-30),
            )
        ),
        "rigid_surface_termination_count": int(
            result.rigid_surface_termination_count
        ),
        "rigid_surface_termination_weight": float(
            result.rigid_surface_termination_weight
        ),
        "rigid_surface_termination_fraction": float(
            result.rigid_surface_termination_weight
        ) / max(launched, 1.0e-30),
        "interface_normal_fallback_count": int(
            result.interface_normal_fallback_count
        ),
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
        "processed_sample_count": int(getattr(result, "processed_sample_count", 0)),
        "clipped_sample_count": int(getattr(result, "clipped_sample_count", 0)),
        "represented_weighted_path_length_mm": float(
            getattr(result, "represented_weighted_path_length_mm", 0.0)
        ),
        "clipped_weighted_path_length_mm": float(
            getattr(result, "clipped_weighted_path_length_mm", 0.0)
        ),
        "processed_weighted_path_length_mm": float(
            getattr(result, "processed_weighted_path_length_mm", 0.0)
        ),
        "termination_count": int(getattr(result, "termination_count", 0)),
        "termination_weight": float(result.terminated_weight),
        "termination_fraction": float(
            getattr(
                result,
                "terminated_weight_fraction",
                float(result.terminated_weight)
                / max(float(result.launched_weight), 1.0e-30),
            )
        ),
        "path_field": {
            "processed_sample_count": int(
                getattr(result, "processed_sample_count", 0)
            ),
            "clipped_sample_count": int(
                getattr(result, "clipped_sample_count", 0)
            ),
            "represented_weighted_path_length_mm": float(
                getattr(result, "represented_weighted_path_length_mm", 0.0)
            ),
            "clipped_weighted_path_length_mm": float(
                getattr(result, "clipped_weighted_path_length_mm", 0.0)
            ),
        },
        "branch_cutoff_termination": {
            "count": int(result.branch_cutoff_termination_count),
            "weight": float(result.branch_cutoff_termination_weight),
        },
        "max_interaction_termination": {
            "count": int(result.max_interaction_termination_count),
            "weight": float(result.max_interaction_termination_weight),
        },
        "segment_budget_termination": {
            "count": int(result.segment_budget_termination_count),
            "weight": float(result.segment_budget_termination_weight),
            "maximum_segment_count": settings.get("maximum_segment_count"),
        },
        "rigid_surface_termination": {
            "count": int(result.rigid_surface_termination_count),
            "weight": float(result.rigid_surface_termination_weight),
        },
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
        "outgoing_surface_weight": result.outgoing_surface_weight,
        "absorbed_weight": result.absorbed_weight,
        "terminated_weight": result.terminated_weight,
        "escape_event_count": int(result.escape_event_count),
        "escaped_primary_count": int(result.escaped_primary_count),
        "processed_sample_count": int(getattr(result, "processed_sample_count", 0)),
        "clipped_sample_count": int(getattr(result, "clipped_sample_count", 0)),
        "represented_weighted_path_length_mm": float(
            getattr(result, "represented_weighted_path_length_mm", 0.0)
        ),
        "clipped_weighted_path_length_mm": float(
            getattr(result, "clipped_weighted_path_length_mm", 0.0)
        ),
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
        "contract_fingerprint": _fingerprint_mapping(dict(contract)),
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


def _required_path_field_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the non-optional v7 path-field diagnostic contract."""

    names = (
        "processed_sample_count",
        "clipped_sample_count",
        "represented_weighted_path_length_mm",
        "clipped_weighted_path_length_mm",
    )
    missing = [name for name in names if name not in record]
    path_diagnostics = record.get("path_diagnostics")
    if not isinstance(path_diagnostics, Mapping):
        raise ValueError("v7 artifact path_diagnostics must be an object")
    path_field = path_diagnostics.get("path_field")
    if not isinstance(path_field, Mapping):
        raise ValueError("v7 artifact path_diagnostics.path_field must be an object")
    missing.extend(
        f"path_diagnostics.path_field.{name}"
        for name in names
        if name not in path_field
    )
    if missing:
        raise ValueError(
            "v7 artifact is missing required path-field diagnostics: "
            + ", ".join(missing)
        )

    counts: dict[str, int] = {}
    for name in names[:2]:
        top = record[name]
        nested = path_field[name]
        if (
            not isinstance(top, Integral)
            or isinstance(top, bool)
            or int(top) < 0
            or not isinstance(nested, Integral)
            or isinstance(nested, bool)
            or int(nested) < 0
        ):
            raise ValueError(f"v7 artifact {name} must be a non-negative integer")
        if int(top) != int(nested):
            raise ValueError(f"v7 artifact {name} disagrees with path_diagnostics")
        counts[name] = int(top)
    if counts["clipped_sample_count"] > counts["processed_sample_count"]:
        raise ValueError("v7 artifact clipped samples exceed processed samples")

    lengths: dict[str, float] = {}
    for name in names[2:]:
        try:
            top = float(record[name])
            nested = float(path_field[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"v7 artifact {name} must be numeric") from exc
        if not math.isfinite(top) or top < 0.0 or not math.isfinite(nested) or nested < 0.0:
            raise ValueError(f"v7 artifact {name} must be finite and non-negative")
        if top != nested:
            raise ValueError(f"v7 artifact {name} disagrees with path_diagnostics")
        lengths[name] = top

    processed_length = path_diagnostics.get("processed_weighted_path_length_mm")
    if processed_length is None:
        raise ValueError(
            "v7 artifact path_diagnostics is missing "
            "processed_weighted_path_length_mm"
        )
    try:
        processed_length_value = float(processed_length)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "v7 artifact processed_weighted_path_length_mm must be numeric"
        ) from exc
    expected_processed_length = (
        lengths["represented_weighted_path_length_mm"]
        + lengths["clipped_weighted_path_length_mm"]
    )
    if (
        not math.isfinite(processed_length_value)
        or processed_length_value < 0.0
        or not math.isclose(
            processed_length_value,
            expected_processed_length,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "v7 artifact processed path length does not equal represented plus clipped"
        )
    termination_count = path_diagnostics.get("termination_count")
    segment = path_diagnostics.get("segment_budget_termination")
    if (
        not isinstance(termination_count, Integral)
        or isinstance(termination_count, bool)
        or int(termination_count) < 0
    ):
        raise ValueError("v7 artifact termination_count must be a non-negative integer")
    if not isinstance(segment, Mapping):
        raise ValueError(
            "v7 artifact path_diagnostics.segment_budget_termination must be an object"
        )
    segment_count = segment.get("count")
    segment_weight = segment.get("weight")
    if (
        not isinstance(segment_count, Integral)
        or isinstance(segment_count, bool)
        or int(segment_count) < 0
    ):
        raise ValueError(
            "v7 artifact segment-budget termination count must be a non-negative integer"
        )
    try:
        segment_weight_value = float(segment_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "v7 artifact segment-budget termination weight must be numeric"
        ) from exc
    if not math.isfinite(segment_weight_value) or segment_weight_value < 0.0:
        raise ValueError(
            "v7 artifact segment-budget termination weight must be finite and non-negative"
        )
    return {
        **counts,
        **lengths,
        "termination_count": int(termination_count),
        "segment_budget_termination_count": int(segment_count),
        "segment_budget_termination_weight": segment_weight_value,
        "path_diagnostics": dict(path_diagnostics),
    }


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
    if metadata.get("contract_fingerprint") != _fingerprint_mapping(dict(expected_contract)):
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
    path_field_diagnostics = _required_path_field_diagnostics(record)
    return OpticalFieldArtifact(
        field=field,
        field_axes=axes,
        total_transport=float(record["total_transport"]),
        launched_weight=float(record["launched_weight"]),
        escaped_weight=float(record["escaped_weight"]),
        outgoing_surface_weight=float(record["outgoing_surface_weight"]),
        absorbed_weight=float(record["absorbed_weight"]),
        terminated_weight=float(record["terminated_weight"]),
        termination_count=path_field_diagnostics["termination_count"],
        segment_budget_termination_count=path_field_diagnostics[
            "segment_budget_termination_count"
        ],
        segment_budget_termination_weight=path_field_diagnostics[
            "segment_budget_termination_weight"
        ],
        energy_balance_error=float(record["energy_balance_error"]),
        ray_count=int(record["ray_count"]),
        escape_event_count=int(record["escape_event_count"]),
        escaped_primary_count=int(record["escaped_primary_count"]),
        processed_sample_count=path_field_diagnostics["processed_sample_count"],
        clipped_sample_count=path_field_diagnostics["clipped_sample_count"],
        represented_weighted_path_length_mm=path_field_diagnostics[
            "represented_weighted_path_length_mm"
        ],
        clipped_weighted_path_length_mm=path_field_diagnostics[
            "clipped_weighted_path_length_mm"
        ],
        path_diagnostics=path_field_diagnostics["path_diagnostics"],
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
    "load_case_artifact",
    "native_field_separability",
    "energy_record",
    "save_case_artifact",
]
