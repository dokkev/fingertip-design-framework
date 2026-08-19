"""Unified neutral contract around the shared OptiX transport core.

The CUDA/OptiX wavefront implementation is shared by both dimensional modes.
This module only assigns native field meaning, case provenance, and the common
separability calculation; it does not implement a second optical solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Literal

import numpy as np

from model.fingertip import Fingertip
from optics.transport3d.geometry import ExtrudedTransportGeometry
from optics.transport3d.result import Transport3DResult
from optics.transport3d.settings import Transport3DSettings
from optics.transport3d.transport import trace_geometry


UnifiedOpticalMode = Literal["PLANAR_2D", "FULL_3D"]
UNIFIED_ARTIFACT_SCHEMA = "unified-optix-transport-case-v3"
LEGACY_UNIFIED_ARTIFACT_SCHEMA_V2 = "unified-optix-transport-case-v2"
LEGACY_UNIFIED_ARTIFACT_SCHEMA = "unified-optix-transport-case-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Return the stable fingerprint used by transport configurations."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _owned_field(value: Any, *, name: str) -> np.ndarray:
    field = np.array(value, dtype=float, copy=True)
    if field.ndim not in (2, 3) or not field.size:
        raise ValueError(f"{name} must be a nonempty 2D or 3D field")
    if not np.all(np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    field.setflags(write=False)
    return field


def _owned_axes(value: Any, *, dimension: int) -> tuple[np.ndarray, ...]:
    axes = tuple(np.array(axis, dtype=float, copy=True) for axis in value)
    if len(axes) != dimension or any(len(axis) < 2 for axis in axes):
        raise ValueError("native field axes do not match field dimensionality")
    if any(not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0) for axis in axes):
        raise ValueError("native field axes must be finite and strictly increasing")
    for axis in axes:
        axis.setflags(write=False)
    return axes


@dataclass(frozen=True)
class UnifiedTransportResult:
    """Common neutral result while preserving P2 or P3 natively."""

    morphology_id: str
    morphology_fingerprint: str
    mechanics_source: str
    mechanics_dimension: str
    contact_state: Mapping[str, Any]
    optical_mode: UnifiedOpticalMode
    ray_count: int
    transport_configuration_fingerprint: str
    field: np.ndarray
    field_axes: tuple[np.ndarray, ...]
    total_transport: float
    launched_weight: float
    escaped_weight: float
    absorbed_weight: float
    terminated_weight: float
    valid_ray_count: int
    terminated_ray_count: int
    energy_balance_error: float
    path_diagnostics: Mapping[str, Any]
    object_absorbed_weight: float = 0.0
    object_transmitted_weight: float = 0.0
    object_interface_incident_weight: float = 0.0
    object_reflected_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.optical_mode not in ("PLANAR_2D", "FULL_3D"):
            raise ValueError("optical_mode must be PLANAR_2D or FULL_3D")
        expected_dimension = 2 if self.optical_mode == "PLANAR_2D" else 3
        if self.mechanics_dimension not in ("2D", "3D"):
            raise ValueError("mechanics_dimension must be '2D' or '3D'")
        if not isinstance(self.ray_count, int) or self.ray_count < 1:
            raise ValueError("ray_count must be a positive integer")
        field = _owned_field(self.field, name="native transport field")
        if field.ndim != expected_dimension:
            raise ValueError(
                f"{self.optical_mode} requires a native {expected_dimension}D field"
            )
        axes = _owned_axes(self.field_axes, dimension=expected_dimension)
        if field.shape != tuple(len(axis) - 1 for axis in axes):
            raise ValueError("native field shape does not match its axes")
        scalars = np.asarray(
            [
                self.total_transport,
                self.launched_weight,
                self.escaped_weight,
                self.absorbed_weight,
                self.terminated_weight,
                self.energy_balance_error,
                self.object_absorbed_weight,
                self.object_transmitted_weight,
                self.object_interface_incident_weight,
                self.object_reflected_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
            raise ValueError("transport totals must be finite and nonnegative")
        if not isinstance(self.valid_ray_count, int) or self.valid_ray_count < 0:
            raise ValueError("valid_ray_count must be a nonnegative integer")
        if not isinstance(self.terminated_ray_count, int) or self.terminated_ray_count < 0:
            raise ValueError("terminated_ray_count must be a nonnegative integer")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "field_axes", axes)
        object.__setattr__(self, "contact_state", dict(self.contact_state))
        object.__setattr__(self, "path_diagnostics", dict(self.path_diagnostics))

    @classmethod
    def from_transport_result(
        cls,
        result: Transport3DResult,
        *,
        morphology_id: str,
        morphology_fingerprint: str,
        mechanics_source: str,
        mechanics_dimension: str,
        contact_state: Mapping[str, Any],
        transport_configuration_fingerprint: str,
    ) -> "UnifiedTransportResult":
        if result.source_mode == "planar":
            mode: UnifiedOpticalMode = "PLANAR_2D"
            if (
                result.projected_x_edges_mm is None
                or result.projected_y_edges_mm is None
                or result.projected_weighted_path_density is None
            ):
                raise ValueError("PLANAR_2D result is missing its native P2 field")
            # The transport accumulator stores projected density as
            # density[y, x].  The neutral public field convention is
            # field[x, y], matching the (x_edges, y_edges) axes below.
            field = np.transpose(result.projected_weighted_path_density, (1, 0))
            axes = (result.projected_x_edges_mm, result.projected_y_edges_mm)
        elif result.source_mode == "full3d":
            mode = "FULL_3D"
            if (
                result.internal_path_x_edges_mm is None
                or result.internal_path_y_edges_mm is None
                or result.internal_path_z_edges_mm is None
                or result.internal_weighted_path_density_3d is None
            ):
                raise ValueError("FULL_3D result is missing its native P3 field")
            # The accumulator owns storage in (z, y, x) order for efficient
            # sample indexing; the neutral artifact contract owns axes in the
            # public (x, y, z) order.  Preserve all three native dimensions
            # while making that ordering explicit at the boundary.
            field = np.transpose(result.internal_weighted_path_density_3d, (2, 1, 0))
            axes = (
                result.internal_path_x_edges_mm,
                result.internal_path_y_edges_mm,
                result.internal_path_z_edges_mm,
            )
        else:
            raise ValueError(f"unsupported transport source mode: {result.source_mode!r}")
        return cls(
            morphology_id=morphology_id,
            morphology_fingerprint=morphology_fingerprint,
            mechanics_source=mechanics_source,
            mechanics_dimension=mechanics_dimension,
            contact_state=contact_state,
            optical_mode=mode,
            ray_count=result.launched_ray_count,
            transport_configuration_fingerprint=transport_configuration_fingerprint,
            field=field,
            field_axes=axes,
            total_transport=result.escaped_weight,
            launched_weight=result.launched_weight,
            escaped_weight=result.escaped_weight,
            absorbed_weight=result.absorbed_weight,
            terminated_weight=result.terminated_weight,
            valid_ray_count=int(len(result.escape_weights)),
            terminated_ray_count=max(
                0,
                result.launched_ray_count - len(result.escape_primary_ray_indices),
            ),
            energy_balance_error=result.energy_balance_error,
            path_diagnostics=result.geometry_metadata,
            object_absorbed_weight=result.object_absorbed_weight,
            object_transmitted_weight=result.object_transmitted_weight,
            object_interface_incident_weight=result.object_interface_incident_weight,
            object_reflected_weight=result.object_reflected_weight,
        )


class OptiXTransport:
    """One entry point for PLANAR_2D and FULL_3D shared transport."""

    def trace(
        self,
        tip: Fingertip,
        geometry: ExtrudedTransportGeometry,
        *,
        settings: Transport3DSettings,
        morphology_id: str,
        morphology_fingerprint: str,
        mechanics_source: str,
        mechanics_dimension: str,
        contact_state: Mapping[str, Any],
        transport_configuration: Mapping[str, Any],
        runtime: Any | None = None,
    ) -> UnifiedTransportResult:
        if settings.mode == "planar" and geometry.geometry_mode != "planar_extruded":
            raise ValueError("PLANAR_2D requires planar_extruded geometry")
        if settings.mode == "full3d" and geometry.geometry_mode != "full3d_surface":
            raise ValueError(
                "FULL_3D requires an actual deformed 3D surface artifact; "
                "2D extrusion is not accepted by the unified evaluator"
            )
        geometry_metadata = dict(geometry.metadata)
        if geometry.geometry_mode == "full3d_surface":
            if geometry_metadata.get("full3d_surface_provenance") not in {
                "actual_deformed_3d_fea_surface",
                "actual_deformed_3d_vbd_surface",
                "actual_deformed_3d_volume_state",
            }:
                raise ValueError(
                    "FULL_3D geometry lacks direct FEA/VBD-surface provenance"
                )
            if geometry_metadata.get("morphology_fingerprint") not in (
                None,
                morphology_fingerprint,
            ):
                raise ValueError("FULL_3D geometry morphology fingerprint mismatch")
            expected_contact_fingerprint = contact_state.get(
                "contact_state_fingerprint"
            )
            if expected_contact_fingerprint is not None and geometry_metadata.get(
                "contact_state_fingerprint"
            ) != expected_contact_fingerprint:
                raise ValueError("FULL_3D geometry contact-state fingerprint mismatch")
        result = trace_geometry(tip, geometry, settings=settings, runtime=runtime)
        return UnifiedTransportResult.from_transport_result(
            result,
            morphology_id=morphology_id,
            morphology_fingerprint=morphology_fingerprint,
            mechanics_source=mechanics_source,
            mechanics_dimension=mechanics_dimension,
            contact_state=contact_state,
            transport_configuration_fingerprint=fingerprint_mapping(
                dict(transport_configuration)
            ),
        )


def native_field_separability(
    first: UnifiedTransportResult,
    second: UnifiedTransportResult,
) -> dict[str, float | str | None]:
    """Evaluate raw magnitude and normalized redistribution in native space."""
    if first.optical_mode != second.optical_mode:
        raise ValueError("native separability requires one optical mode")
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
    normalized: float | None
    if first_mass > 0.0 and second_mass > 0.0:
        normalized = 0.5 * float(
            np.sum(np.abs(left / first_mass - right / second_mass))
        )
        normalized_status = "valid"
    else:
        normalized = None
        normalized_status = "singular_zero_field"
    return {
        "optical_mode": first.optical_mode,
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


def transport_configuration(
    settings: Transport3DSettings,
    *,
    material: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the common settings/material contract for case fingerprints."""
    configuration = {
        "settings": asdict(settings),
        "material": dict(material),
        "source_sampling": "optics.transport3d.sampling.sample_directions",
        "physics": "optics.transport3d.physics.interface_split+attenuation",
        "accumulation": "native P2 for PLANAR_2D; native P3(x,y,z) for FULL_3D",
    }
    if source is not None:
        configuration["source"] = dict(source)
    return configuration


def save_case_artifact(path: Path, result: UnifiedTransportResult, contract: Mapping[str, Any]) -> None:
    """Persist one independently verifiable transport case artifact."""
    field_axis_order = {
        "PLANAR_2D": "x,y",
        "FULL_3D": "x,y,z",
    }.get(result.optical_mode)
    if field_axis_order is None:
        raise ValueError(f"unsupported unified optical mode: {result.optical_mode!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    field_path = path.with_suffix(".npz")
    field_tmp = field_path.with_name(field_path.name + ".tmp")
    metadata_tmp = path.with_name(path.name + ".tmp")
    with field_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            field=np.asarray(result.field),
            **{f"axis_{index}": axis for index, axis in enumerate(result.field_axes)},
        )
    field_sha = hashlib.sha256(field_tmp.read_bytes()).hexdigest()
    metadata = {
        "schema": UNIFIED_ARTIFACT_SCHEMA,
        "field_axis_order": field_axis_order,
        "contract": dict(contract),
        "contract_fingerprint": fingerprint_mapping(dict(contract)),
        "field_artifact": str(field_path),
        "field_sha256": field_sha,
        "result": {
            "morphology_id": result.morphology_id,
            "morphology_fingerprint": result.morphology_fingerprint,
            "mechanics_source": result.mechanics_source,
            "mechanics_dimension": result.mechanics_dimension,
            "contact_state": dict(result.contact_state),
            "optical_mode": result.optical_mode,
            "ray_count": result.ray_count,
            "transport_configuration_fingerprint": result.transport_configuration_fingerprint,
            "total_transport": result.total_transport,
            "launched_weight": result.launched_weight,
            "escaped_weight": result.escaped_weight,
            "absorbed_weight": result.absorbed_weight,
            "terminated_weight": result.terminated_weight,
            "valid_ray_count": result.valid_ray_count,
            "terminated_ray_count": result.terminated_ray_count,
            "energy_balance_error": result.energy_balance_error,
            "object_absorbed_weight": result.object_absorbed_weight,
            "object_transmitted_weight": result.object_transmitted_weight,
            "object_interface_incident_weight": result.object_interface_incident_weight,
            "object_reflected_weight": result.object_reflected_weight,
            "path_diagnostics": dict(result.path_diagnostics),
        },
    }
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    field_tmp.replace(field_path)
    metadata_tmp.replace(path)


def load_case_artifact(path: Path, *, expected_contract: Mapping[str, Any]) -> UnifiedTransportResult:
    """Load an artifact only when the complete fingerprint contract matches."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    if schema not in (
        UNIFIED_ARTIFACT_SCHEMA,
        LEGACY_UNIFIED_ARTIFACT_SCHEMA_V2,
        LEGACY_UNIFIED_ARTIFACT_SCHEMA,
    ):
        raise ValueError("unsupported unified transport artifact schema")
    record = metadata.get("result")
    if not isinstance(record, Mapping):
        raise ValueError("unified transport result metadata is missing")
    if schema == UNIFIED_ARTIFACT_SCHEMA:
        expected_axis_order = {
            "PLANAR_2D": "x,y",
            "FULL_3D": "x,y,z",
        }.get(record.get("optical_mode"))
        if (
            expected_axis_order is None
            or metadata.get("field_axis_order") != expected_axis_order
        ):
            raise ValueError(
                "unified transport field axis order is missing or unsupported"
            )
    contract = metadata.get("contract")
    if contract != dict(expected_contract):
        raise ValueError("unified transport artifact contract mismatch")
    if metadata.get("contract_fingerprint") != fingerprint_mapping(dict(expected_contract)):
        raise ValueError("unified transport artifact contract fingerprint mismatch")
    field_path = Path(str(metadata.get("field_artifact", "")))
    if not field_path.is_absolute():
        field_path = path.parent / field_path.name
    if not field_path.exists():
        raise ValueError("unified transport field artifact is missing")
    if hashlib.sha256(field_path.read_bytes()).hexdigest() != metadata.get("field_sha256"):
        raise ValueError("unified transport field artifact checksum mismatch")
    with np.load(field_path, allow_pickle=False) as archive:
        if "field" not in archive:
            raise ValueError("unified transport field is missing")
        field = _owned_field(archive["field"], name="unified transport field")
        axes = tuple(
            np.asarray(archive[f"axis_{index}"], dtype=float)
            for index in range(field.ndim)
        )
    if schema == LEGACY_UNIFIED_ARTIFACT_SCHEMA and record.get("optical_mode") == "PLANAR_2D":
        # v1 persisted the raw accumulator convention density[y, x] while
        # labeling its axes as (x, y).  Convert only legacy planar artifacts;
        # v1 FULL_3D fields already used the public x,y,z ordering.
        field = np.transpose(field, (1, 0))
    contract_result_checks = {
        "morphology_id": contract.get("morphology_id"),
        "mechanics_dimension": contract.get("mechanics_dimension"),
        "mechanics_source": contract.get("mechanics_source"),
        "optical_mode": contract.get("optical_mode"),
        "ray_count": contract.get("ray_count"),
    }
    for key, expected in contract_result_checks.items():
        if expected is not None and record.get(key) != expected:
            raise ValueError(f"unified transport result {key} mismatches its contract")
    expected_morphology = contract.get("morphology_parameters_fingerprint")
    if expected_morphology is not None and record.get("morphology_fingerprint") != expected_morphology:
        raise ValueError("unified transport result morphology fingerprint mismatch")
    expected_contact = contract.get("contact_state_fingerprint")
    if expected_contact is not None:
        if not isinstance(record.get("contact_state"), Mapping):
            raise ValueError("unified transport result contact state is missing")
        if record["contact_state"].get("contact_state_fingerprint") != expected_contact:
            raise ValueError("unified transport result contact-state fingerprint mismatch")
    expected_configuration = contract.get("transport_configuration_fingerprint")
    if (
        expected_configuration is not None
        and record.get("transport_configuration_fingerprint") != expected_configuration
    ):
        raise ValueError("unified transport result transport-configuration fingerprint mismatch")
    return UnifiedTransportResult(
        morphology_id=str(record["morphology_id"]),
        morphology_fingerprint=str(record["morphology_fingerprint"]),
        mechanics_source=str(record["mechanics_source"]),
        mechanics_dimension=str(record["mechanics_dimension"]),
        contact_state=record["contact_state"],
        optical_mode=str(record["optical_mode"]),
        ray_count=int(record["ray_count"]),
        transport_configuration_fingerprint=str(
            record["transport_configuration_fingerprint"]
        ),
        field=field,
        field_axes=axes,
        total_transport=float(record["total_transport"]),
        launched_weight=float(record["launched_weight"]),
        escaped_weight=float(record["escaped_weight"]),
        absorbed_weight=float(record["absorbed_weight"]),
        terminated_weight=float(record["terminated_weight"]),
        valid_ray_count=int(record["valid_ray_count"]),
        terminated_ray_count=int(record["terminated_ray_count"]),
        energy_balance_error=float(record["energy_balance_error"]),
        path_diagnostics=record["path_diagnostics"],
        object_absorbed_weight=float(record.get("object_absorbed_weight", 0.0)),
        object_transmitted_weight=float(record.get("object_transmitted_weight", 0.0)),
        object_interface_incident_weight=float(
            record.get("object_interface_incident_weight", 0.0)
        ),
        object_reflected_weight=float(record.get("object_reflected_weight", 0.0)),
    )


__all__ = [
    "LEGACY_UNIFIED_ARTIFACT_SCHEMA",
    "LEGACY_UNIFIED_ARTIFACT_SCHEMA_V2",
    "OptiXTransport",
    "UNIFIED_ARTIFACT_SCHEMA",
    "UnifiedOpticalMode",
    "UnifiedTransportResult",
    "fingerprint_mapping",
    "load_case_artifact",
    "native_field_separability",
    "save_case_artifact",
    "transport_configuration",
]
