"""Typed, one-shot loader for production LUMO numerical execution settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError

from lumo.contact import FirstContactSettings
from lumo.finger import DEFAULT_EXTRUSION_DEPTH_MM
from lumo.mechanics_contract import MechanicsContract
from lumo.mesh import VolumeMeshSettings
from lumo.physics.contracts import VBDDeterminismMode
from lumo.ray_tracing.optical_mechanics import Transport3DSettings


EXECUTION_CONFIG_SCHEMA_VERSION = 2


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class LumoExecutionConfigError(ValueError):
    """Raised when an execution YAML cannot become a trusted typed config."""


@dataclass(frozen=True)
class ExecutionConfigSource:
    path: str
    sha256: str
    schema_version: int


@dataclass(frozen=True)
class LumoExecutionConfig:
    """Resolved numerical settings used by mesh, mechanics, and transport."""

    device: str
    volume_mesh: VolumeMeshSettings
    mechanics: MechanicsContract
    transport: Transport3DSettings
    source: ExecutionConfigSource

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "volume_mesh": asdict(self.volume_mesh),
            "mechanics": self.mechanics.to_dict(),
            "transport": asdict(self.transport),
            "source": asdict(self.source),
        }


def _mapping(value: Any, path: str, source: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LumoExecutionConfigError(f"{source}: {path} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
    source: Path,
) -> None:
    supplied = {str(key) for key in value}
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        raise LumoExecutionConfigError(
            f"{source}: {path} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _integer(value: Any, path: str, source: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LumoExecutionConfigError(f"{source}: {path} must be an integer")
    return value


def _number(value: Any, path: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LumoExecutionConfigError(f"{source}: {path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LumoExecutionConfigError(f"{source}: {path} must be finite")
    return result


def _boolean(value: Any, path: str, source: Path) -> bool:
    if not isinstance(value, bool):
        raise LumoExecutionConfigError(f"{source}: {path} must be a boolean")
    return value


def _bounds(value: Any, path: str, source: Path) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise LumoExecutionConfigError(
            f"{source}: {path} must be a two-element sequence in millimetres"
        )
    return (_number(value[0], f"{path}[0]", source), _number(value[1], f"{path}[1]", source))


def load_lumo_execution_config(path: str | Path) -> LumoExecutionConfig:
    """Load, strictly validate, and resolve one production execution YAML."""

    source = Path(path).expanduser().resolve()
    try:
        payload_bytes = source.read_bytes()
    except OSError as exc:
        raise LumoExecutionConfigError(f"cannot read execution config {source}: {exc}") from exc
    try:
        raw = yaml.load(payload_bytes, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise LumoExecutionConfigError(f"invalid YAML in {source}: {exc}") from exc
    root = _mapping(raw, "<root>", source)
    _exact_keys(root, {"schema_version", "runtime", "mesh", "newton", "indentation", "transport"}, "<root>", source)
    schema_version = _integer(root["schema_version"], "schema_version", source)
    if schema_version != EXECUTION_CONFIG_SCHEMA_VERSION:
        raise LumoExecutionConfigError(
            f"{source}: unsupported schema_version {schema_version}; "
            f"expected {EXECUTION_CONFIG_SCHEMA_VERSION}"
        )

    runtime = _mapping(root["runtime"], "runtime", source)
    _exact_keys(runtime, {"device"}, "runtime", source)
    device = runtime["device"]
    if not isinstance(device, str) or re.fullmatch(r"cuda:\d+", device.strip()) is None:
        raise LumoExecutionConfigError(
            f"{source}: runtime.device must use the form cuda:<non-negative index>"
        )
    device = device.strip()

    mesh = _mapping(root["mesh"], "mesh", source)
    _exact_keys(mesh, {"tier", "target_size_mm", "minimum_quality"}, "mesh", source)
    try:
        volume_mesh = VolumeMeshSettings(
            tier=mesh["tier"],
            target_size_mm=_number(mesh["target_size_mm"], "mesh.target_size_mm", source),
            minimum_quality=_number(mesh["minimum_quality"], "mesh.minimum_quality", source),
        )
    except (TypeError, ValueError) as exc:
        raise LumoExecutionConfigError(f"{source}: mesh: {exc}") from exc

    newton = _mapping(root["newton"], "newton", source)
    newton_keys = {
        "sphere_subdivisions", "max_load_increment_mm", "vbd_iterations", "deterministic_mode", "dt_s",
        "soft_contact_margin_mm", "soft_contact_ke", "soft_contact_kd", "soft_contact_mu",
        "rigid_sdf_target_voxel_mm", "max_support_displacement_mm", "max_final_pose_error_mm",
        "max_carrier_penetration_voxel_fraction",
    }
    _exact_keys(newton, newton_keys, "newton", source)
    indentation = _mapping(root["indentation"], "indentation", source)
    indentation_keys = {"coarse_step_mm", "tolerance_mm", "spawn_clearance_mm", "max_travel_mm"}
    _exact_keys(indentation, indentation_keys, "indentation", source)
    try:
        first_contact = FirstContactSettings(
            **{
                name: _number(indentation[name], f"indentation.{name}", source)
                for name in indentation_keys
            }
        )
        mechanics = MechanicsContract(
            sphere_subdivisions=_integer(newton["sphere_subdivisions"], "newton.sphere_subdivisions", source),
            max_load_increment_mm=_number(newton["max_load_increment_mm"], "newton.max_load_increment_mm", source),
            vbd_iterations=_integer(newton["vbd_iterations"], "newton.vbd_iterations", source),
            deterministic_mode=VBDDeterminismMode(
                newton["deterministic_mode"]
            ),
            dt_s=_number(newton["dt_s"], "newton.dt_s", source),
            soft_contact_margin_mm=_number(newton["soft_contact_margin_mm"], "newton.soft_contact_margin_mm", source),
            soft_contact_ke=_number(newton["soft_contact_ke"], "newton.soft_contact_ke", source),
            soft_contact_kd=_number(newton["soft_contact_kd"], "newton.soft_contact_kd", source),
            soft_contact_mu=_number(newton["soft_contact_mu"], "newton.soft_contact_mu", source),
            rigid_sdf_target_voxel_mm=_number(newton["rigid_sdf_target_voxel_mm"], "newton.rigid_sdf_target_voxel_mm", source),
            max_support_displacement_mm=_number(newton["max_support_displacement_mm"], "newton.max_support_displacement_mm", source),
            max_final_pose_error_mm=_number(newton["max_final_pose_error_mm"], "newton.max_final_pose_error_mm", source),
            max_carrier_penetration_voxel_fraction=_number(newton["max_carrier_penetration_voxel_fraction"], "newton.max_carrier_penetration_voxel_fraction", source),
            first_contact=first_contact,
        )
    except (TypeError, ValueError) as exc:
        raise LumoExecutionConfigError(f"{source}: mechanics: {exc}") from exc

    transport = _mapping(root["transport"], "transport", source)
    integer_transport = {
        "ray_count", "max_interactions", "maximum_segment_count", "maximum_periodic_wraps",
        "surface_u_bins", "surface_z_bins", "internal_grid_width", "internal_grid_height",
        "internal_z_bins", "internal_max_samples_per_segment",
    }
    number_transport = {
        "minimum_ray_weight", "extrusion_depth_mm", "source_epsilon_mm",
        "intersection_epsilon_mm", "energy_balance_tolerance",
    }
    boolean_transport = {
        "retain_internal_path_field", "terminate_on_periodic_wrap_limit", "terminate_on_no_event",
    }
    transport_keys = integer_transport | number_transport | boolean_transport | {"x_bounds_mm", "y_bounds_mm"}
    _exact_keys(transport, transport_keys, "transport", source)
    try:
        transport_settings = Transport3DSettings(
            **{name: _integer(transport[name], f"transport.{name}", source) for name in integer_transport},
            **{name: _number(transport[name], f"transport.{name}", source) for name in number_transport},
            **{name: _boolean(transport[name], f"transport.{name}", source) for name in boolean_transport},
            x_bounds_mm=_bounds(transport["x_bounds_mm"], "transport.x_bounds_mm", source),
            y_bounds_mm=_bounds(transport["y_bounds_mm"], "transport.y_bounds_mm", source),
        )
    except (TypeError, ValueError) as exc:
        raise LumoExecutionConfigError(f"{source}: transport: {exc}") from exc
    if not math.isclose(
        transport_settings.extrusion_depth_mm,
        DEFAULT_EXTRUSION_DEPTH_MM,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise LumoExecutionConfigError(
            f"{source}: transport.extrusion_depth_mm must equal the fixed "
            f"representative-cell depth {DEFAULT_EXTRUSION_DEPTH_MM:g} mm"
        )

    return LumoExecutionConfig(
        device=device,
        volume_mesh=volume_mesh,
        mechanics=mechanics,
        transport=transport_settings,
        source=ExecutionConfigSource(
            path=str(source),
            sha256=hashlib.sha256(payload_bytes).hexdigest(),
            schema_version=schema_version,
        ),
    )


__all__ = [
    "EXECUTION_CONFIG_SCHEMA_VERSION",
    "ExecutionConfigSource",
    "LumoExecutionConfig",
    "LumoExecutionConfigError",
    "load_lumo_execution_config",
]
