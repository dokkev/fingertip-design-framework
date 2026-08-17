"""Bounded sensor-facing verification from existing OptiX escape events.

The transport implementation remains camera-independent.  This validation
module applies one explicit idealized camera operator downstream of the
persisted external escape events, then computes pairwise response separation,
finite-difference Jacobians, Fisher summaries, and a global-gain nuisance
comparison for nominal and candidate49.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


OUTPUT = Path("output/optix_design_verification/baseline")
SOURCE_TRANSPORT_OUTPUT = Path(
    "output/validation/optics/transport3d/internal_bridge_convergence"
)
SOURCE_FIELDS = SOURCE_TRANSPORT_OUTPUT / "fields.npz"
SOURCE_SUMMARY = SOURCE_TRANSPORT_OUTPUT / "summary.json"
CANDIDATE_INPUT = Path(
    "output/validation/optimization/pre_bo_nominal_sweep/inputs/candidate_0049.json"
)

RAY_COUNT = 262_144
MAX_INTERACTIONS = 10
MINIMUM_RAY_WEIGHT = 1.0e-4
MAXIMUM_PERIODIC_WRAPS = 512
EXTRUSION_DEPTH_MM = 11.0
SURFACE_U_BINS = 128
SURFACE_Z_BINS = 64
EVENT_PROVENANCE_SCHEMA = 1

SENSOR_WIDTH_PX = 128
SENSOR_HEIGHT_PX = 64
DIAGNOSTIC_RESOLUTIONS = ((128, 64), (64, 32), (32, 16))
SENSOR_Y_BOUNDS_MM = (-16.0, 5.5)
SENSOR_Z_BOUNDS_MM = (-5.5, 5.5)
CAMERA_POSITION_MM = (17.0, -5.25, 0.0)
CAMERA_TARGET_MM = (0.0, -5.25, 0.0)
CAMERA_UP = (0.0, 0.0, 1.0)

READ_NOISE_SIGMA = 1.0e-4
STATE_SCALES = np.asarray([6.0, 1.0], dtype=float)
FISHER_RANK_TOLERANCE = 1.0e-10
ORDER_TOLERANCE = 1.0e-12
DERIVATIVE_MIN_ONE_SIDED_DIRECTION_COSINE = 0.0
DERIVATIVE_MIN_STEP_CHANGE_DIRECTION_COSINE = 0.5

STATE_DEFINITIONS: dict[str, dict[str, float | None]] = {
    "unloaded": {"x_c_mm": 0.0, "delta_mm": 0.0},
    "left_contact": {"x_c_mm": -3.0, "delta_mm": 0.5},
    "right_contact": {"x_c_mm": 3.0, "delta_mm": 0.5},
    "near_left_contact": {"x_c_mm": -1.5, "delta_mm": 0.5},
    "near_right_contact": {"x_c_mm": 1.5, "delta_mm": 0.5},
    "center_low": {"x_c_mm": 0.0, "delta_mm": 0.25},
    "center_shallow": {"x_c_mm": 0.0, "delta_mm": 0.5},
    "center_high": {"x_c_mm": 0.0, "delta_mm": 0.75},
    "center_deep": {"x_c_mm": 0.0, "delta_mm": 1.0},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_file_sha256(revision: str, relative_path: str) -> str | None:
    """Return a repository-file digest from a recorded source revision."""
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _event_provenance() -> dict[str, Any]:
    """Describe the exact transport contract used by sensor event archives."""
    return {
        "schema": EVENT_PROVENANCE_SCHEMA,
        "transport_module_sha256": hashlib.sha256(
            Path("optics/transport3d/transport.py").read_bytes()
        ).hexdigest(),
        "sampling_module_sha256": hashlib.sha256(
            Path("optics/transport3d/sampling.py").read_bytes()
        ).hexdigest(),
        "trace_settings": {
            "mode": "full3d",
            "ray_count": RAY_COUNT,
            "max_interactions": MAX_INTERACTIONS,
            "minimum_ray_weight": MINIMUM_RAY_WEIGHT,
            "maximum_periodic_wraps": MAXIMUM_PERIODIC_WRAPS,
            "extrusion_depth_mm": EXTRUSION_DEPTH_MM,
            "surface_u_bins": SURFACE_U_BINS,
            "surface_z_bins": SURFACE_Z_BINS,
            "terminate_on_periodic_wrap_limit": True,
            "terminate_on_no_event": True,
        },
    }


def _paired_sampling_contract(intrinsic_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the reusable event artifact's common-ray sampling contract."""
    source_revision = str(intrinsic_summary.get("git_revision", ""))
    sampling_path = "optics/transport3d/sampling.py"
    current_digest = hashlib.sha256(Path(sampling_path).read_bytes()).hexdigest()
    source_digest = _git_file_sha256(source_revision, sampling_path)
    checks: dict[str, Any] = {
        "source_sampling_module_unchanged": source_digest == current_digest,
        "source_sampling_module_sha256": source_digest,
        "current_sampling_module_sha256": current_digest,
    }
    primary_sets: dict[tuple[str, str], set[int]] = {}
    with np.load(SOURCE_FIELDS, allow_pickle=False) as archive:
        for design_name in ("nominal", "candidate49"):
            for state_name in ("left", "right"):
                key = f"{design_name}_current_{state_name}_escape_primary_ray_indices"
                indices = np.asarray(archive[key], dtype=np.int64)
                valid = (
                    indices.ndim == 1
                    and len(indices) > 0
                    and np.all(indices >= 0)
                    and np.all(indices < RAY_COUNT)
                    and len(np.unique(indices)) == len(indices)
                )
                checks[f"{design_name}_{state_name}_primary_indices_valid"] = bool(valid)
                primary_sets[(design_name, state_name)] = set(indices.tolist())
    for design_name in ("nominal", "candidate49"):
        left = primary_sets[(design_name, "left")]
        right = primary_sets[(design_name, "right")]
        checks[f"{design_name}_left_right_primary_index_overlap"] = len(left & right)
        checks[f"{design_name}_left_right_primary_index_union"] = len(left | right)
    checks.update(
        {
            "ray_sequence": "deterministic sample_directions(mode=full3d, ray_count=262144); no RNG seed",
            "pairing_key": "escape_primary_ray_indices identify the common base ray sequence",
            "state_dependent_event_sets": "expected; left/right acceptance and branching may differ by state",
            "derivative_use": "paired source-ray sequence is valid for matched camera-response finite differences",
            "status": (
                "PASS"
                if bool(checks["source_sampling_module_unchanged"])
                and all(
                    bool(value)
                    for key, value in checks.items()
                    if key.endswith("_primary_indices_valid")
                )
                else "FAIL"
            ),
        }
    )
    return checks


def _load_archived_event_state(
    path: Path,
    state_name: str,
    expected_provenance: Mapping[str, Any],
) -> dict[str, np.ndarray] | None:
    """Load one previously completed event state when its arrays are valid."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            provenance_key = f"{state_name}_event_provenance_json"
            if provenance_key not in archive:
                return None
            provenance = json.loads(str(archive[provenance_key].item()))
            if provenance != dict(expected_provenance):
                return None
            events = _event_from_archive(archive, state_name)
    except (KeyError, OSError, ValueError):
        return None
    positions = events["positions_mm"]
    directions = events["directions"]
    weights = events["weights"]
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or directions.shape != positions.shape
        or weights.ndim != 1
        or len(weights) != len(positions)
        or len(positions) == 0
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        return None
    return events


def _archived_event_metadata(path: Path, state_name: str) -> dict[str, Any]:
    """Read persisted transport metadata for one validated event state."""
    with np.load(path, allow_pickle=False) as archive:
        key = f"{state_name}_geometry_metadata_json"
        if key not in archive:
            return {}
        return json.loads(str(archive[key].item()))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def camera_configuration() -> dict[str, Any]:
    """Return the precommitted idealized finite-planar-sensor model."""
    return {
        "model": "ideal_finite_planar_sensor",
        "camera_position_mm": list(CAMERA_POSITION_MM),
        "camera_target_mm": list(CAMERA_TARGET_MM),
        "camera_up": list(CAMERA_UP),
        "sensor_plane_normal": [1.0, 0.0, 0.0],
        "sensor_y_bounds_mm": list(SENSOR_Y_BOUNDS_MM),
        "sensor_z_bounds_mm": list(SENSOR_Z_BOUNDS_MM),
        "resolution_px": [SENSOR_WIDTH_PX, SENSOR_HEIGHT_PX],
        "projection": "ray_to_finite_plane",
        "response_units": "relative transported power per sensor pixel",
        "pixel_gain": 1.0,
        "occlusion_model": "none beyond the already-computed external escape event",
        "source_geometry": str(
            Path("output/validation/optics/pre_bo_mitsuba_single_cell/summary.json")
        ),
    }


def project_escape_events(
    positions_mm: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    configuration: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project external escape rays onto a finite planar sensor.

    ``directions`` are required to be the post-interface external directions
    retained by ``Transport3DResult``.  No response normalization is applied.
    """
    config = configuration or camera_configuration()
    positions = np.asarray(positions_mm, dtype=float)
    outgoing = np.asarray(directions, dtype=float)
    ray_weights = np.asarray(weights, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions_mm must have shape (N, 3)")
    if outgoing.shape != positions.shape:
        raise ValueError("directions must match positions_mm")
    if ray_weights.ndim != 1 or len(ray_weights) != len(positions):
        raise ValueError("weights must match positions_mm")
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(outgoing))
        or not np.all(np.isfinite(ray_weights))
        or np.any(ray_weights < 0.0)
    ):
        raise ValueError("escape event arrays must be finite and nonnegative")

    camera_position = np.asarray(config["camera_position_mm"], dtype=float)
    camera_target = np.asarray(config["camera_target_mm"], dtype=float)
    camera_up = np.asarray(config["camera_up"], dtype=float)
    view_from_camera = camera_target - camera_position
    view_from_camera /= np.linalg.norm(view_from_camera)
    sensor_up = camera_up / np.linalg.norm(camera_up)
    sensor_right = np.cross(view_from_camera, sensor_up)
    sensor_right /= np.linalg.norm(sensor_right)
    toward_camera = -view_from_camera

    direction_norm = np.linalg.norm(outgoing, axis=1)
    valid_direction = direction_norm > 0.0
    unit_directions = np.zeros_like(outgoing)
    unit_directions[valid_direction] = (
        outgoing[valid_direction] / direction_norm[valid_direction, None]
    )
    facing = valid_direction & (unit_directions @ toward_camera > 1.0e-12)
    denominator = unit_directions @ toward_camera
    distances = np.full(len(positions), np.nan, dtype=float)
    distances[facing] = (
        (camera_position - positions[facing]) @ toward_camera
    ) / denominator[facing]
    reaches_plane = facing & np.isfinite(distances) & (distances > 0.0)
    intersections = positions + distances[:, None] * unit_directions
    horizontal = (intersections - camera_target) @ sensor_right
    vertical = (intersections - camera_target) @ sensor_up
    y_min, y_max = (float(value) for value in config["sensor_y_bounds_mm"])
    z_min, z_max = (float(value) for value in config["sensor_z_bounds_mm"])
    accepted = reaches_plane & (horizontal >= y_min) & (horizontal <= y_max)
    accepted &= vertical >= z_min
    accepted &= vertical <= z_max

    width = int(config["resolution_px"][0])
    height = int(config["resolution_px"][1])
    response = np.zeros((height, width), dtype=float)
    if np.any(accepted):
        columns = np.floor(
            (horizontal[accepted] - y_min) / (y_max - y_min) * width
        ).astype(int)
        rows = np.floor(
            (vertical[accepted] - z_min) / (z_max - z_min) * height
        ).astype(int)
        columns = np.clip(columns, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)
        np.add.at(response, (rows, columns), ray_weights[accepted])

    total_event_power = float(np.sum(ray_weights))
    accepted_power = float(np.sum(ray_weights[accepted]))
    return response, {
        "event_count": int(len(positions)),
        "camera_facing_event_count": int(np.count_nonzero(facing)),
        "reaches_sensor_plane_event_count": int(np.count_nonzero(reaches_plane)),
        "accepted_event_count": int(np.count_nonzero(accepted)),
        "total_external_escape_power": total_event_power,
        "accepted_camera_power": accepted_power,
        "accepted_fraction_of_external_escape": (
            accepted_power / total_event_power if total_event_power > 0.0 else None
        ),
    }


def pairwise_separation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    read_noise_sigma: float = READ_NOISE_SIGMA,
) -> dict[str, float | None | str]:
    """Return raw, fixed-covariance, and profiled-gain separations.

    The profiled quantity removes only the best global multiplicative scale
    from ``second``.  It is a pairwise shape diagnostic, not a Fisher metric
    or a complete treatment of photometric nuisance parameters.
    """
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("responses must be matching 2D arrays")
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        raise ValueError("responses must be finite and nonnegative")
    if not np.isfinite(read_noise_sigma) or read_noise_sigma <= 0.0:
        raise ValueError("read_noise_sigma must be finite and positive")
    difference = left - right
    squared_l2 = float(np.sum(difference * difference))
    variance = read_noise_sigma * read_noise_sigma
    second_weighted_norm = float(np.sum(right * right) / variance)
    if second_weighted_norm > 0.0:
        gain_star = float(np.sum(right * left) / np.sum(right * right))
        profiled_difference = left - gain_star * right
        profiled_d2 = float(np.sum(profiled_difference * profiled_difference) / variance)
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm > 0.0 and right_norm > 0.0:
            cosine = float(
                np.dot(left.reshape(-1), right.reshape(-1))
                / (left_norm * right_norm)
            )
            cosine = float(np.clip(cosine, -1.0, 1.0))
            angular_shape = float(2.0 * (1.0 - cosine))
        else:
            cosine = None
            angular_shape = None
        profile_status = "valid"
    else:
        gain_star = None
        profiled_d2 = None
        cosine = None
        angular_shape = None
        profile_status = "singular_reference_response"
    return {
        "noise_free_squared_l2": squared_l2,
        "noise_free_l2": float(np.sqrt(squared_l2)),
        "noise_normalized_d2": squared_l2 / variance,
        "gain_profile_status": profile_status,
        "best_global_gain_second_to_first": gain_star,
        "gain_profiled_shape_d2": profiled_d2,
        "gain_profiled_fraction_of_raw_d2": (
            profiled_d2 / (squared_l2 / variance)
            if profiled_d2 is not None and squared_l2 > 0.0
            else None
        ),
        "gain_profiled_interpretation": (
            "spatial_structural_dominant"
            if profiled_d2 is not None
            and squared_l2 > 0.0
            and profiled_d2 / (squared_l2 / variance) >= 0.8
            else "photometric_dominant"
            if profiled_d2 is not None
            and squared_l2 > 0.0
            and profiled_d2 / (squared_l2 / variance) <= 0.2
            else "mixed_or_unresolved"
            if profiled_d2 is not None
            else "unavailable"
        ),
        "symmetric_whitened_angular_squared": angular_shape,
        "whitened_response_cosine": cosine,
        "absolute_response_difference_sum": float(np.sum(np.abs(difference))),
        "first_total_power": float(np.sum(left)),
        "second_total_power": float(np.sum(right)),
        "total_power_difference": float(np.sum(left) - np.sum(right)),
        "relative_total_power_difference": (
            float((np.sum(left) - np.sum(right)) / np.sum(right))
            if np.sum(right) > 0.0
            else None
        ),
    }


def centered_jacobian(
    responses: Mapping[str, np.ndarray],
    *,
    position_step_mm: float = 1.5,
    indentation_step_mm: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the center-state Jacobian from physically valid FEM states."""
    required = (
        "near_left_contact",
        "near_right_contact",
        "center_low",
        "center_high",
        "center_shallow",
    )
    if any(name not in responses for name in required):
        raise ValueError("centered_jacobian requires the complete five-state stencil")
    center = np.asarray(responses["center_shallow"], dtype=float)
    left = np.asarray(responses["near_left_contact"], dtype=float)
    right = np.asarray(responses["near_right_contact"], dtype=float)
    unloaded = np.asarray(responses["center_low"], dtype=float)
    deep = np.asarray(responses["center_high"], dtype=float)
    if any(field.shape != center.shape for field in (left, right, unloaded, deep)):
        raise ValueError("all finite-difference responses must share one sensor grid")
    if position_step_mm <= 0.0 or indentation_step_mm <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    jacobian = np.column_stack(
        (
            ((right - left) / (2.0 * position_step_mm)).reshape(-1),
            ((deep - unloaded) / (2.0 * indentation_step_mm)).reshape(-1),
        )
    )
    return jacobian, {
        "state": "center_shallow",
        "coordinates": [0.0, 0.5],
        "method": "centered finite differences",
        "position_stencil": ["near_left_contact", "near_right_contact"],
        "indentation_stencil": ["center_low", "center_high"],
        "position_step_mm": position_step_mm,
        "indentation_step_mm": indentation_step_mm,
    }


def derivative_diagnostics(
    responses: Mapping[str, np.ndarray],
    *,
    position_step_mm: float = 3.0,
    indentation_step_mm: float = 0.5,
) -> dict[str, Any]:
    """Compare one-sided slopes available in the same five-state stencil."""
    center = np.asarray(responses["center_shallow"], dtype=float).reshape(-1)
    left = np.asarray(responses["left_contact"], dtype=float).reshape(-1)
    right = np.asarray(responses["right_contact"], dtype=float).reshape(-1)
    near_left = np.asarray(responses["near_left_contact"], dtype=float).reshape(-1)
    near_right = np.asarray(responses["near_right_contact"], dtype=float).reshape(-1)
    unloaded = np.asarray(responses["unloaded"], dtype=float).reshape(-1)
    low = np.asarray(responses["center_low"], dtype=float).reshape(-1)
    high = np.asarray(responses["center_high"], dtype=float).reshape(-1)
    deep = np.asarray(responses["center_deep"], dtype=float).reshape(-1)

    def compare(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        denominator = first_norm * second_norm
        cosine = (
            float(np.dot(first, second) / denominator)
            if denominator > 0.0
            else None
        )
        return {
            "first_norm": first_norm,
            "second_norm": second_norm,
            "direction_cosine": cosine,
            "relative_norm_difference": (
                abs(first_norm - second_norm) / max(first_norm, second_norm)
                if max(first_norm, second_norm) > 0.0
                else None
            ),
        }

    x_lower = (center - left) / position_step_mm
    x_upper = (right - center) / position_step_mm
    x_inner = (center - near_left) / (position_step_mm / 2.0)
    x_inner_upper = (near_right - center) / (position_step_mm / 2.0)
    delta_lower = (center - unloaded) / indentation_step_mm
    delta_upper = (deep - center) / indentation_step_mm
    delta_inner = (center - low) / (indentation_step_mm / 2.0)
    delta_inner_upper = (high - center) / (indentation_step_mm / 2.0)
    position_outer = compare(x_lower, x_upper)
    position_inner = compare(x_inner, x_inner_upper)
    position_step_change = compare(
        (x_lower + x_upper) / 2.0,
        (x_inner + x_inner_upper) / 2.0,
    )
    indentation_outer = compare(delta_lower, delta_upper)
    indentation_inner = compare(delta_inner, delta_inner_upper)
    indentation_step_change = compare(
        (delta_lower + delta_upper) / 2.0,
        (delta_inner + delta_inner_upper) / 2.0,
    )

    def one_sided_valid(record: Mapping[str, Any]) -> bool:
        cosine = record["direction_cosine"]
        return cosine is not None and cosine > DERIVATIVE_MIN_ONE_SIDED_DIRECTION_COSINE

    def step_change_valid(record: Mapping[str, Any]) -> bool:
        cosine = record["direction_cosine"]
        return cosine is not None and cosine >= DERIVATIVE_MIN_STEP_CHANGE_DIRECTION_COSINE

    validity_checks = {
        "position_outer_one_sided": one_sided_valid(position_outer),
        "position_inner_one_sided": one_sided_valid(position_inner),
        "indentation_outer_one_sided": one_sided_valid(indentation_outer),
        "indentation_inner_one_sided": one_sided_valid(indentation_inner),
        "position_step_change": step_change_valid(position_step_change),
        "indentation_step_change": step_change_valid(indentation_step_change),
    }
    return {
        "position_outer": position_outer,
        "position_inner": position_inner,
        "position_step_change": position_step_change,
        "indentation_outer": indentation_outer,
        "indentation_inner": indentation_inner,
        "indentation_step_change": indentation_step_change,
        "validity": {
            "checks": validity_checks,
            "pass": all(validity_checks.values()),
            "minimum_one_sided_direction_cosine": DERIVATIVE_MIN_ONE_SIDED_DIRECTION_COSINE,
            "minimum_step_change_direction_cosine": DERIVATIVE_MIN_STEP_CHANGE_DIRECTION_COSINE,
            "interpretation": "failed direction or step-size consistency blocks Fisher interpretation; no averaging is used to repair it",
        },
        "interpretation": "outer and inner one-sided slope agreement from the bounded nine-state stencil",
    }


def finite_step_diagnostic(responses: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Use the existing inner/outer states to expose finite-step behavior."""
    powers = {
        name: float(np.sum(np.asarray(field, dtype=float)))
        for name, field in responses.items()
    }
    scalar_slopes = {
        "position_outer_lower": (powers["center_shallow"] - powers["left_contact"]) / 3.0,
        "position_outer_upper": (powers["right_contact"] - powers["center_shallow"]) / 3.0,
        "position_inner_lower": (powers["center_shallow"] - powers["near_left_contact"]) / 1.5,
        "position_inner_upper": (powers["near_right_contact"] - powers["center_shallow"]) / 1.5,
        "indentation_outer_lower": (powers["center_shallow"] - powers["unloaded"]) / 0.5,
        "indentation_outer_upper": (powers["center_deep"] - powers["center_shallow"]) / 0.5,
        "indentation_inner_lower": (powers["center_shallow"] - powers["center_low"]) / 0.25,
        "indentation_inner_upper": (powers["center_high"] - powers["center_shallow"]) / 0.25,
    }
    sign_consistency = {
        "position_outer": scalar_slopes["position_outer_lower"]
        * scalar_slopes["position_outer_upper"]
        >= 0.0,
        "position_inner": scalar_slopes["position_inner_lower"]
        * scalar_slopes["position_inner_upper"]
        >= 0.0,
        "indentation_outer": scalar_slopes["indentation_outer_lower"]
        * scalar_slopes["indentation_outer_upper"]
        >= 0.0,
        "indentation_inner": scalar_slopes["indentation_inner_lower"]
        * scalar_slopes["indentation_inner_upper"]
        >= 0.0,
    }
    derivative = derivative_diagnostics(responses)
    return {
        "states_reused": list(responses),
        "new_states_generated": False,
        "camera_power_by_state": powers,
        "scalar_one_sided_slopes": scalar_slopes,
        "scalar_slope_sign_consistency": sign_consistency,
        "vector_derivative_diagnostics": derivative,
        "interpretation": (
            "existing inner/outer stencil shows finite-step or acceptance nonlinearity; "
            "this is diagnostic evidence only and does not restore Fisher validity"
            if not all(sign_consistency.values()) or not derivative["validity"]["pass"]
            else "existing bounded stencil is directionally consistent"
        ),
    }


def _predefined_pairwise(
    responses: Mapping[str, np.ndarray],
    *,
    read_noise_sigma: float = READ_NOISE_SIGMA,
) -> dict[str, dict[str, float | None | str]]:
    """Evaluate the three precommitted contact-state comparisons."""
    return {
        "contact_onset": pairwise_separation(
            responses["center_shallow"],
            responses["unloaded"],
            read_noise_sigma=read_noise_sigma,
        ),
        "indentation": pairwise_separation(
            responses["center_deep"],
            responses["center_shallow"],
            read_noise_sigma=read_noise_sigma,
        ),
        "position": pairwise_separation(
            responses["right_contact"],
            responses["left_contact"],
            read_noise_sigma=read_noise_sigma,
        ),
    }


def _ordering(
    first: float | None,
    second: float | None,
) -> str:
    if first is None or second is None:
        return "unavailable"
    if first > second + ORDER_TOLERANCE:
        return "candidate49_gt_nominal"
    if first < second - ORDER_TOLERANCE:
        return "candidate49_lt_nominal"
    return "tie"


def camera_discretization_diagnostic(
    event_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    resolutions: tuple[tuple[int, int], ...] = DIAGNOSTIC_RESOLUTIONS,
    read_noise_sigma: float = READ_NOISE_SIGMA,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Re-bin current-provenance escape events at the precommitted grids.

    No transport or FEM work is performed here.  The returned arrays preserve
    the response fields and centered derivative vectors needed to inspect the
    diagnostic without rerunning the expensive stage.
    """
    if not resolutions:
        raise ValueError("at least one response resolution is required")
    diagnostics: dict[str, Any] = {
        "settings": {
            "resolutions_px": [list(resolution) for resolution in resolutions],
            "physical_sensor_y_bounds_mm": list(SENSOR_Y_BOUNDS_MM),
            "physical_sensor_z_bounds_mm": list(SENSOR_Z_BOUNDS_MM),
            "camera_pose": camera_configuration(),
            "ray_events_reused": True,
            "retraced": False,
            "response_normalization": "none; raw accepted transported power",
            "noise_model": "fixed covariance",
            "read_noise_sigma": read_noise_sigma,
        },
        "resolutions": {},
    }
    arrays: dict[str, np.ndarray] = {}
    pairwise_by_resolution: dict[str, dict[str, dict[str, Any]]] = {}
    derivative_by_resolution: dict[str, dict[str, Any]] = {}
    accepted_power_by_resolution: dict[str, dict[str, dict[str, float]]] = {}

    for width, height in resolutions:
        if width <= 0 or height <= 0:
            raise ValueError("response resolutions must be positive")
        resolution_key = f"{width}x{height}"
        configuration = camera_configuration()
        configuration["resolution_px"] = [width, height]
        resolution_record: dict[str, Any] = {
            "camera": configuration,
            "morphologies": {},
        }
        pairwise_by_resolution[resolution_key] = {}
        derivative_by_resolution[resolution_key] = {}
        accepted_power_by_resolution[resolution_key] = {}
        for design_name in ("nominal", "candidate49"):
            design_responses: dict[str, np.ndarray] = {}
            state_records: dict[str, Any] = {}
            accepted_power_by_resolution[resolution_key][design_name] = {}
            for state_name in STATE_DEFINITIONS:
                events = event_data[design_name][state_name]
                response, record = project_escape_events(
                    events["positions_mm"],
                    events["directions"],
                    events["weights"],
                    configuration,
                )
                design_responses[state_name] = response
                arrays[f"{resolution_key}_{design_name}_{state_name}_mu"] = response
                accepted_power_by_resolution[resolution_key][design_name][state_name] = float(
                    record["accepted_camera_power"]
                )
                state_records[state_name] = {
                    **record,
                    "response_shape": list(response.shape),
                    "nonzero_pixel_fraction": float(
                        np.count_nonzero(response) / response.size
                    ),
                    "accepted_events_per_pixel": float(
                        record["accepted_event_count"] / response.size
                    ),
                    "total_camera_response_power": float(np.sum(response)),
                }
            jacobian, jacobian_metadata = centered_jacobian(design_responses)
            derivative = derivative_diagnostics(design_responses)
            pairwise = _predefined_pairwise(
                design_responses,
                read_noise_sigma=read_noise_sigma,
            )
            arrays[f"{resolution_key}_{design_name}_center_shallow_jacobian"] = jacobian
            resolution_record["morphologies"][design_name] = {
                "states": state_records,
                "pairwise": pairwise,
                "centered_jacobian_metadata": jacobian_metadata,
                "derivative_diagnostics": derivative,
                "finite_step_diagnostic": finite_step_diagnostic(design_responses),
                "fisher_validity": {
                    "status": "valid_candidate_for_fisher"
                    if derivative["validity"]["pass"]
                    else "blocked_by_derivative_gate",
                    "derivative_gate_pass": derivative["validity"]["pass"],
                },
            }
            pairwise_by_resolution[resolution_key][design_name] = pairwise
            derivative_by_resolution[resolution_key][design_name] = derivative
        resolution_record["pairwise_ordering"] = {
            category: {
                "raw_fixed_covariance": _ordering(
                    pairwise_by_resolution[resolution_key]["candidate49"][category][
                        "noise_normalized_d2"
                    ],
                    pairwise_by_resolution[resolution_key]["nominal"][category][
                        "noise_normalized_d2"
                    ],
                ),
                "gain_profiled_shape": _ordering(
                    pairwise_by_resolution[resolution_key]["candidate49"][category][
                        "gain_profiled_shape_d2"
                    ],
                    pairwise_by_resolution[resolution_key]["nominal"][category][
                        "gain_profiled_shape_d2"
                    ],
                ),
            }
            for category in ("contact_onset", "indentation", "position")
        }
        diagnostics["resolutions"][resolution_key] = resolution_record

    reference_resolution = f"{resolutions[0][0]}x{resolutions[0][1]}"
    power_invariance: dict[str, Any] = {}
    for design_name in ("nominal", "candidate49"):
        state_checks: dict[str, Any] = {}
        for state_name in STATE_DEFINITIONS:
            powers = [
                accepted_power_by_resolution[key][design_name][state_name]
                for key in accepted_power_by_resolution
            ]
            maximum_difference = max(powers) - min(powers)
            state_checks[state_name] = {
                "accepted_camera_power_by_resolution": {
                    key: accepted_power_by_resolution[key][design_name][state_name]
                    for key in accepted_power_by_resolution
                },
                "maximum_absolute_difference": maximum_difference,
                "pass": bool(maximum_difference <= 1.0e-12),
            }
        power_invariance[design_name] = {
            "states": state_checks,
            "pass": all(item["pass"] for item in state_checks.values()),
        }

    raw_orderings = {
        resolution_key: {
            category: diagnostics["resolutions"][resolution_key]["pairwise_ordering"][
                category
            ]["raw_fixed_covariance"]
            for category in ("contact_onset", "indentation", "position")
        }
        for resolution_key in diagnostics["resolutions"]
    }
    profiled_orderings = {
        resolution_key: {
            category: diagnostics["resolutions"][resolution_key]["pairwise_ordering"][
                category
            ]["gain_profiled_shape"]
            for category in ("contact_onset", "indentation", "position")
        }
        for resolution_key in diagnostics["resolutions"]
    }
    raw_stable = len({json.dumps(value, sort_keys=True) for value in raw_orderings.values()}) == 1
    profiled_stable = len({json.dumps(value, sort_keys=True) for value in profiled_orderings.values()}) == 1
    all_derivative_valid = all(
        derivative_by_resolution[key][design_name]["validity"]["pass"]
        for key in derivative_by_resolution
        for design_name in ("nominal", "candidate49")
    )
    if raw_stable and profiled_stable and not all_derivative_valid:
        diagnostic_case = "D2_pairwise_stable_derivatives_unstable"
    elif raw_stable and profiled_stable and all_derivative_valid:
        diagnostic_case = "D4_pairwise_and_derivatives_stable"
    elif raw_stable and not profiled_stable:
        diagnostic_case = "D3_gain_profiled_pairwise_ordering_resolution_sensitive"
    else:
        diagnostic_case = "D3_pairwise_ordering_resolution_sensitive"
    diagnostics["assessment"] = {
        "reference_resolution": reference_resolution,
        "total_accepted_camera_power_invariant": power_invariance,
        "raw_pairwise_ordering_by_resolution": raw_orderings,
        "gain_profiled_pairwise_ordering_by_resolution": profiled_orderings,
        "raw_pairwise_ordering_stable": raw_stable,
        "gain_profiled_pairwise_ordering_stable": profiled_stable,
        "all_derivative_gates_pass": all_derivative_valid,
        "diagnostic_case": diagnostic_case,
        "interpretation": (
            "direct pairwise evidence remains usable; the current hard-binned "
            "derivative/Fisher branch remains unresolved"
            if diagnostic_case == "D2_pairwise_stable_derivatives_unstable"
            else "see resolution-specific evidence before using sensor-facing ordering"
        ),
    }
    return diagnostics, arrays


def _matrix_summary(
    fisher_physical: np.ndarray,
    *,
    state_scales: np.ndarray = STATE_SCALES,
    rank_tolerance: float = FISHER_RANK_TOLERANCE,
) -> dict[str, Any]:
    physical = np.asarray(fisher_physical, dtype=float)
    if physical.shape != (2, 2) or not np.all(np.isfinite(physical)):
        raise ValueError("physical Fisher matrix must be finite and 2x2")
    physical = 0.5 * (physical + physical.T)
    scale_matrix = np.diag(np.asarray(state_scales, dtype=float))
    dimensionless = scale_matrix @ physical @ scale_matrix
    dimensionless = 0.5 * (dimensionless + dimensionless.T)
    eigenvalues = np.linalg.eigvalsh(dimensionless)
    maximum = float(np.max(eigenvalues))
    tolerance = rank_tolerance * max(maximum, 0.0)
    rank = int(np.count_nonzero(eigenvalues > tolerance)) if maximum > 0.0 else 0
    psd_tolerance = max(1.0e-12, 1.0e-10 * max(maximum, 1.0))
    psd_pass = bool(np.min(eigenvalues) >= -psd_tolerance)
    full_rank = rank == 2 and psd_pass
    if maximum > 0.0 and full_rank:
        condition_number: float | None = float(eigenvalues[-1] / eigenvalues[0])
        log_determinant: float | None = float(np.log(np.linalg.det(dimensionless)))
    else:
        condition_number = None
        log_determinant = None
    crlb = np.linalg.pinv(physical, rcond=rank_tolerance)
    crlb = 0.5 * (crlb + crlb.T)
    return {
        "fisher_physical": physical,
        "fisher_dimensionless": dimensionless,
        "eigenvalues_dimensionless": eigenvalues,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "rank": rank,
        "rank_tolerance_relative": rank_tolerance,
        "rank_tolerance_absolute": tolerance,
        "condition_number": condition_number,
        "log_determinant": log_determinant,
        "crlb_physical": crlb,
        "position_crlb_std_mm": float(np.sqrt(max(crlb[0, 0], 0.0))),
        "indentation_crlb_std_mm": float(np.sqrt(max(crlb[1, 1], 0.0))),
        "status": "full_rank" if full_rank else "rank_deficient_or_non_psd",
        "psd_check_pass": psd_pass,
        "symmetry_check_pass": bool(np.allclose(physical, physical.T, atol=1.0e-12)),
    }


def fisher_from_response(
    response: np.ndarray,
    jacobian_physical: np.ndarray,
    *,
    read_noise_sigma: float = READ_NOISE_SIGMA,
    state_scales: np.ndarray = STATE_SCALES,
    rank_tolerance: float = FISHER_RANK_TOLERANCE,
) -> dict[str, Any]:
    """Compute fixed-covariance Gaussian Fisher and gain-marginalized Fisher."""
    mu = np.asarray(response, dtype=float).reshape(-1)
    jacobian = np.asarray(jacobian_physical, dtype=float)
    if jacobian.shape != (len(mu), 2):
        raise ValueError("jacobian must have one row per response pixel")
    variance = float(read_noise_sigma * read_noise_sigma)
    fisher_physical = (jacobian.T @ jacobian) / variance
    base = _matrix_summary(
        fisher_physical,
        state_scales=state_scales,
        rank_tolerance=rank_tolerance,
    )

    joint_jacobian = np.column_stack((jacobian, mu))
    joint_fisher = (joint_jacobian.T @ joint_jacobian) / variance
    joint_fisher = 0.5 * (joint_fisher + joint_fisher.T)
    contact_block = joint_fisher[:2, :2]
    nuisance_block = float(joint_fisher[2, 2])
    nuisance_tolerance = rank_tolerance * max(float(np.max(np.linalg.eigvalsh(joint_fisher))), 1.0)
    if nuisance_block > nuisance_tolerance:
        cross = joint_fisher[:2, 2]
        effective = contact_block - np.outer(cross, cross) / nuisance_block
        effective = 0.5 * (effective + effective.T)
        gain_effective = _matrix_summary(
            effective,
            state_scales=state_scales,
            rank_tolerance=rank_tolerance,
        )
        information_loss = contact_block - effective
        information_loss_eigenvalues = np.linalg.eigvalsh(
            0.5 * (information_loss + information_loss.T)
        )
        gain_consistency_pass = bool(
            np.min(information_loss_eigenvalues)
            >= -max(1.0e-12, 1.0e-10 * max(float(np.max(np.abs(contact_block))), 1.0))
        )
        gain_status = "valid"
    else:
        effective = np.full((2, 2), np.nan, dtype=float)
        gain_effective = {
            "status": "singular_gain_block",
            "fisher_physical": effective,
            "fisher_dimensionless": effective,
        }
        information_loss_eigenvalues = np.full(2, np.nan, dtype=float)
        gain_consistency_pass = False
        gain_status = "singular_gain_block"

    return {
        "likelihood": "fixed-covariance Gaussian mean-response model",
        "noise_variance_per_pixel": variance,
        "read_noise_sigma": read_noise_sigma,
        "fisher": base,
        "joint_fisher_physical_contact_x_delta_gain": joint_fisher,
        "gain_nuisance": {
            "model": "mu(theta,g)=g*mu0(theta), evaluated at g=1",
            "status": gain_status,
            "gain_gain_information": nuisance_block,
            "contact_block_before_marginalization": contact_block,
            "effective_contact_fisher": effective,
            "effective_summary": gain_effective,
            "information_loss_eigenvalues": information_loss_eigenvalues,
            "marginalization_does_not_increase_information": gain_consistency_pass,
        },
    }


def _pair_key(first: str, second: str) -> str:
    return f"{first}__{second}"


def _load_candidate_parameters(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"candidate artifact has no parameters: {path}")
    return dict(parameters)


def _event_from_archive(
    archive: Any,
    prefix: str,
    *,
    suffix: str = "",
) -> dict[str, np.ndarray]:
    def array(name: str) -> np.ndarray:
        key = f"{prefix}_escape_{name}{suffix}"
        if key not in archive:
            raise KeyError(f"missing escape-event array {key}")
        return np.asarray(archive[key])

    return {
        "positions_mm": array("positions"),
        "directions": array("directions"),
        "weights": array("weights"),
    }


def _save_event_archive(
    path: Path,
    state_results: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for state_name, result in state_results.items():
        arrays[f"{state_name}_escape_positions"] = np.asarray(result.escape_positions_mm)
        arrays[f"{state_name}_escape_directions"] = np.asarray(result.escape_directions)
        arrays[f"{state_name}_escape_weights"] = np.asarray(result.escape_weights)
        arrays[f"{state_name}_escape_normals"] = np.asarray(result.escape_surface_normals)
        arrays[f"{state_name}_escape_u"] = np.asarray(result.escape_surface_u)
        arrays[f"{state_name}_escape_z"] = np.asarray(result.escape_surface_z)
        arrays[f"{state_name}_escape_path_lengths_mm"] = np.asarray(result.escape_path_lengths_mm)
        arrays[f"{state_name}_escape_primary_ray_indices"] = np.asarray(result.escape_primary_ray_indices)
        arrays[f"{state_name}_escape_interaction_counts"] = np.asarray(result.escape_interaction_counts)
        arrays[f"{state_name}_event_provenance_json"] = np.asarray(
            json.dumps(_jsonable(provenance), sort_keys=True)
        )
        arrays[f"{state_name}_geometry_metadata_json"] = np.asarray(
            json.dumps(_jsonable(result.geometry_metadata), sort_keys=True)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _state_trace_settings() -> Any:
    from optics.transport3d import Transport3DSettings

    return Transport3DSettings(
        mode="full3d",
        ray_count=RAY_COUNT,
        max_interactions=MAX_INTERACTIONS,
        minimum_ray_weight=MINIMUM_RAY_WEIGHT,
        maximum_segment_count=max(20_000, 24 * RAY_COUNT),
        maximum_periodic_wraps=MAXIMUM_PERIODIC_WRAPS,
        extrusion_depth_mm=EXTRUSION_DEPTH_MM,
        surface_u_bins=SURFACE_U_BINS,
        surface_z_bins=SURFACE_Z_BINS,
        terminate_on_periodic_wrap_limit=True,
        terminate_on_no_event=True,
    )


def _summarize_design_comparison(
    metrics: Mapping[str, Any],
    intrinsic_summary: Mapping[str, Any],
) -> dict[str, Any]:
    nominal = metrics["nominal"]
    candidate = metrics["candidate49"]
    categories = ("contact_onset", "indentation", "position")
    pairwise_ordering = {
        category: _ordering(
            candidate["pairwise"][category]["noise_normalized_d2"],
            nominal["pairwise"][category]["noise_normalized_d2"],
        )
        for category in categories
    }
    gain_profiled_ordering = {
        category: _ordering(
            candidate["pairwise"][category]["gain_profiled_shape_d2"],
            nominal["pairwise"][category]["gain_profiled_shape_d2"],
        )
        for category in categories
    }
    unmarginalized_candidate_better = (
        candidate["fisher"]["fisher"]["minimum_eigenvalue"]
        > nominal["fisher"]["fisher"]["minimum_eigenvalue"] + ORDER_TOLERANCE
    )
    gain_candidate_better = (
        candidate["fisher"]["gain_nuisance"]["effective_summary"]["minimum_eigenvalue"]
        > nominal["fisher"]["gain_nuisance"]["effective_summary"]["minimum_eigenvalue"]
        + ORDER_TOLERANCE
    )
    candidate_pairwise_wins = sum(
        value == "candidate49_gt_nominal" for value in pairwise_ordering.values()
    )
    candidate_pairwise_losses = sum(
        value == "candidate49_lt_nominal" for value in pairwise_ordering.values()
    )
    candidate_profiled_wins = sum(
        value == "candidate49_gt_nominal" for value in gain_profiled_ordering.values()
    )
    candidate_profiled_losses = sum(
        value == "candidate49_lt_nominal" for value in gain_profiled_ordering.values()
    )
    intrinsic_advantage = (
        intrinsic_summary.get("final_ordering", {}).get("j3d_path")
        == "candidate49_gt_nominal"
    )
    derivative_validity = {
        name: metrics[name]["jacobian"]["derivative_diagnostics"]["validity"]
        for name in ("nominal", "candidate49")
    }
    derivative_gate_pass = all(
        bool(value.get("pass")) for value in derivative_validity.values()
    )
    fisher_full_rank = all(
        metrics[name]["fisher"]["fisher"]["status"] == "full_rank"
        and metrics[name]["fisher"]["gain_nuisance"]["effective_summary"].get("status")
        == "full_rank"
        for name in ("nominal", "candidate49")
    )
    if not derivative_gate_pass or not fisher_full_rank:
        outcome = "F_INCONCLUSIVE"
    elif candidate_pairwise_wins == len(categories) and gain_candidate_better:
        outcome = "A_FULL_BASELINE_VERIFICATION"
    elif (
        intrinsic_advantage
        and unmarginalized_candidate_better
        and not gain_candidate_better
        and candidate["camera_power"]["center_shallow"]
        > nominal["camera_power"]["center_shallow"]
    ):
        outcome = "C_GAIN_DOMINATED"
    elif intrinsic_advantage and (
        candidate_pairwise_losses > 0 or not gain_candidate_better
    ):
        outcome = "B_INTRINSIC_TRANSPORT_ONLY"
    elif candidate_pairwise_wins > 0 and candidate_pairwise_losses > 0:
        outcome = "D_STATE_DEPENDENT_TRADEOFF"
    else:
        outcome = "E_NEGATIVE_BASELINE_VERIFICATION"
    return {
        "baseline_outcome": outcome,
        "branch_status": {
            "intrinsic_bridge": "SUPPORTED" if intrinsic_advantage else "CONTRADICTED",
            "camera_pairwise": "CONDITIONALLY_SUPPORTED",
            "gain_profiled_pairwise": "CONDITIONALLY_SUPPORTED",
            "jacobian": "VERIFIED" if derivative_gate_pass else "BLOCKED",
            "fisher": "VERIFIED" if derivative_gate_pass and fisher_full_rank else "BLOCKED",
            "fisher_gain_nuisance": (
                "VERIFIED" if derivative_gate_pass and fisher_full_rank else "BLOCKED"
            ),
        },
        "intrinsic_j3d_path_advantage": intrinsic_advantage,
        "pairwise_ordering": pairwise_ordering,
        "gain_profiled_pairwise_ordering": gain_profiled_ordering,
        "candidate_pairwise_win_count": candidate_pairwise_wins,
        "candidate_pairwise_loss_count": candidate_pairwise_losses,
        "candidate_gain_profiled_win_count": candidate_profiled_wins,
        "candidate_gain_profiled_loss_count": candidate_profiled_losses,
        "unmarginalized_fisher_candidate_better": unmarginalized_candidate_better,
        "gain_marginalized_fisher_candidate_better": gain_candidate_better,
        "derivative_gate_pass": derivative_gate_pass,
        "fisher_full_rank": fisher_full_rank,
        "derivative_validity": derivative_validity,
    }


def _make_figure(output: Path, responses: Mapping[str, Any], metrics: Mapping[str, Any]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path = output / "figures" / "baseline_sensor_metrics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fields = [
        responses["nominal"]["center_shallow"],
        responses["candidate49"]["center_shallow"],
    ]
    vmax = max(float(np.max(field)) for field in fields)
    for axis, field, title in zip(
        axes[0, :2], fields, ("nominal center shallow mu", "candidate49 center shallow mu")
    ):
        axis.imshow(field, origin="lower", vmin=0.0, vmax=vmax, cmap="magma", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("sensor y pixel")
        axis.set_ylabel("sensor z pixel")
    axes[0, 2].axis("off")
    categories = ("contact", "indentation", "position")
    x = np.arange(len(categories))
    width = 0.38
    for offset, name in ((-width / 2.0, "nominal"), (width / 2.0, "candidate49")):
        axes[1, 0].bar(
            x + offset,
            [metrics[name]["pairwise"][key]["noise_normalized_d2"] for key in ("contact_onset", "indentation", "position")],
            width,
            label=name,
        )
    axes[1, 0].set_xticks(x, categories)
    axes[1, 0].set_title("pairwise d2")
    axes[1, 0].legend()
    axes[1, 1].bar(
        [0, 1],
        [
            metrics["nominal"]["fisher"]["gain_nuisance"]["effective_summary"]["minimum_eigenvalue"],
            metrics["candidate49"]["fisher"]["gain_nuisance"]["effective_summary"]["minimum_eigenvalue"],
        ],
        color=("tab:blue", "tab:orange"),
    )
    axes[1, 1].set_xticks([0, 1], ["nominal", "candidate49"])
    axes[1, 1].set_title("gain-marginalized lambda_min")
    axes[1, 1].set_yscale("symlog", linthresh=1.0)
    for offset, name in enumerate(("nominal", "candidate49")):
        axes[1, 2].bar(
            np.asarray([0, 1]) + (offset - 0.5) * width,
            [
                metrics[name]["fisher"]["gain_nuisance"]["effective_summary"]["position_crlb_std_mm"],
                metrics[name]["fisher"]["gain_nuisance"]["effective_summary"]["indentation_crlb_std_mm"],
            ],
            width,
            label=name,
        )
    axes[1, 2].set_xticks([0, 1], ["position CRLB", "indentation CRLB"])
    axes[1, 2].set_title("gain-marginalized CRLB")
    axes[1, 2].legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def run_benchmark(output: Path = OUTPUT) -> dict[str, Any]:
    """Run or resume the bounded baseline sensor-facing benchmark."""
    output.mkdir(parents=True, exist_ok=True)
    intrinsic_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if intrinsic_summary.get("selected_ray_count") != RAY_COUNT:
        raise RuntimeError("the reusable OptiX source artifact has the wrong ray count")
    if not SOURCE_FIELDS.exists():
        raise FileNotFoundError(SOURCE_FIELDS)

    paired_sampling = _paired_sampling_contract(intrinsic_summary)
    event_provenance = _event_provenance()
    stencil_semantics = {
        "evaluation_state": {
            "name": "center_shallow",
            "x_c_mm": 0.0,
            "delta_mm": 0.5,
            "is_true_fem_optix_state": True,
            "response_use": "local mu(theta) and global-gain derivative at the Fisher evaluation point",
        },
        "outer_five_state_stencil": {
            "left_contact": {"x_c_mm": -3.0, "delta_mm": 0.5},
            "center_shallow": {"x_c_mm": 0.0, "delta_mm": 0.5},
            "right_contact": {"x_c_mm": 3.0, "delta_mm": 0.5},
            "unloaded": {"x_c_mm": 0.0, "delta_mm": 0.0},
            "center_deep": {"x_c_mm": 0.0, "delta_mm": 1.0},
            "use": "pairwise contact, position, and indentation diagnostics plus outer derivative check",
        },
        "inner_five_state_stencil": {
            "near_left_contact": {"x_c_mm": -1.5, "delta_mm": 0.5},
            "center_shallow": {"x_c_mm": 0.0, "delta_mm": 0.5},
            "near_right_contact": {"x_c_mm": 1.5, "delta_mm": 0.5},
            "center_low": {"x_c_mm": 0.0, "delta_mm": 0.25},
            "center_high": {"x_c_mm": 0.0, "delta_mm": 0.75},
            "use": "centered Jacobian at the true center state and inner derivative check",
        },
        "center_response_rule": "evaluate center_shallow directly; never average or interpolate neighboring responses",
    }

    choices = {
        "camera": camera_configuration(),
        "contact_state_grid": STATE_DEFINITIONS,
        "finite_difference": {
            "position_step_mm": 1.5,
            "indentation_step_mm": 0.25,
            "representative_state": "center_shallow",
            "validity_check_outer_position_step_mm": 3.0,
            "validity_check_outer_indentation_step_mm": 0.5,
            "choice_change": {
                "previous": "single outer five-state stencil",
                "new": "outer five-state stencil plus inner symmetric four-state check",
                "reason": "outer one-sided derivative directions disagreed; add a bounded step-size validity check without changing baseline pairwise states",
            },
        },
        "derivative_validity_thresholds": {
            "minimum_one_sided_direction_cosine": DERIVATIVE_MIN_ONE_SIDED_DIRECTION_COSINE,
            "minimum_step_change_direction_cosine": DERIVATIVE_MIN_STEP_CHANGE_DIRECTION_COSINE,
            "choice_change": {
                "previous": "no explicit machine-readable derivative acceptance gate",
                "new": "reject Fisher interpretation when one-sided directions disagree or outer/inner step directions have cosine below 0.5",
                "reason": "the completed bounded stencil showed materially inconsistent derivative directions; make the existing scientific validity rule explicit rather than averaging it away",
            },
            "choice_status": "explicit validity gate added to encode a demonstrable derivative defect; not tuned to morphology ordering",
        },
        "ray_count": RAY_COUNT,
        "ray_sampling": "deterministic transport directions; identical ray sequence for every state",
        "noise_model": {
            "likelihood": "fixed-covariance Gaussian",
            "read_noise_sigma": READ_NOISE_SIGMA,
            "variance_per_pixel": READ_NOISE_SIGMA**2,
            "units": "relative launched transport power",
            "calibration": "uncalibrated model-assumed relative comparison",
        },
        "response_scaling": "raw accepted external transported power; no per-image normalization",
        "camera_discretization": {
            "precommitted_resolutions_px": [list(value) for value in DIAGNOSTIC_RESOLUTIONS],
            "method": "re-bin current-provenance escape events; do not retrace",
            "selection_rule": "retain the precommitted 128x64 observation model; do not select a resolution by candidate Fisher score",
        },
        "pairwise_global_gain_profile": {
            "scale_formula": "g_star=(mu_b^T W mu_a)/(mu_b^T W mu_b)",
            "profiled_metric": "(mu_a-g_star*mu_b)^T W (mu_a-g_star*mu_b)",
            "symmetric_metric": "2*(1-whitened_response_cosine)",
            "interpretation_thresholds": {
                "spatial_structural_dominant_min_fraction": 0.8,
                "photometric_dominant_max_fraction": 0.2,
            },
            "claim_limit": "pairwise global-gain-profiled shape separation; not Fisher and not complete nuisance calibration",
        },
        "state_scaling": {
            "x_c_reference_mm": float(STATE_SCALES[0]),
            "delta_reference_mm": float(STATE_SCALES[1]),
            "shared_across_morphologies": True,
        },
        "fisher_rank_tolerance": FISHER_RANK_TOLERANCE,
        "ordering_tolerance": ORDER_TOLERANCE,
        "derivative_check_states": list(STATE_DEFINITIONS),
        "stencil_semantics": stencil_semantics,
        "paired_sampling_contract": paired_sampling,
        "event_provenance": event_provenance,
    }
    manifest = {
        "completed_stage": "precommit",
        "status": "PRECOMMITTED",
        "created_at": _now(),
        "git_revision": _git_revision(),
        "git_status": _git_status(),
        "precommitted_analysis_choices": choices,
        "event_provenance": event_provenance,
        "artifacts_reused": [
            f"{SOURCE_FIELDS} (intrinsic J3D-path and deterministic sampler contract only)",
            str(SOURCE_SUMMARY),
            str(SOURCE_TRANSPORT_OUTPUT / "fea_states"),
            str(output / "fea_states"),
        ],
        "artifacts_produced": [str(output / "precommit_choices.json")],
        "artifacts_invalidated": [
            f"{SOURCE_FIELDS} sensor-facing escape arrays (generated under an older transport revision; intrinsic fields remain valid)",
            str(output / "events" / "nominal_new_states.npz"),
            str(output / "events" / "candidate49_new_states.npz"),
        ],
        "reviewer_status": "not_started",
        "unresolved_blockers": [],
        "next_authorized_stage": "baseline",
    }
    previous_choices_path = output / "precommit_choices.json"
    initial_choices_path = output / "precommit_choices_initial.json"
    if previous_choices_path.exists() and not initial_choices_path.exists():
        initial_choices_path.write_text(
            previous_choices_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    _write_json(output / "precommit_choices.json", choices)
    _write_json(output / "run_manifest.json", manifest)
    if paired_sampling["status"] != "PASS":
        manifest.update(
            {
                "status": "BLOCKED",
                "completed_stage": "precommit",
                "unresolved_blockers": [
                    "reused escape-event artifact failed the paired/common-random-number contract"
                ],
                "next_authorized_stage": "repair_or_derivative_invalidated_rerun",
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        raise RuntimeError("reused escape-event artifact failed paired sampling validation")

    from mesh import mesh_settings_for_level
    from model import Fingertip, FingertipParameters
    from optics.transport3d import trace_3d
    from optics.transport3d.optix_backend import create_runtime
    from optimization.scenarios import ContactScenario
    from validation.optics.transport3d_validation import _solve_contact

    candidate_parameters = _load_candidate_parameters(CANDIDATE_INPUT)
    design_parameters = {
        "nominal": FingertipParameters(),
        "candidate49": FingertipParameters(**candidate_parameters),
    }
    mesh_settings = mesh_settings_for_level("medium")
    designs: dict[str, dict[str, Any]] = {}
    for design_name, parameters in design_parameters.items():
        tip = Fingertip(parameters)
        mesh = tip.mesh(mesh_settings)
        designs[design_name] = {
            "tip": tip,
            "mesh": mesh,
            "states": {"unloaded": mesh},
            "fea": {},
        }

    scenarios = {
        name: ContactScenario(
            float(values["x_c_mm"]),
            float(values["delta_mm"]),
            4.0,
        )
        for name, values in STATE_DEFINITIONS.items()
        if name != "unloaded"
    }
    for design_name, design in designs.items():
        for state_name in STATE_DEFINITIONS:
            if state_name == "unloaded":
                continue
            cache_path = (
                SOURCE_TRANSPORT_OUTPUT / "fea_states" / f"{design_name}_{state_name}.npz"
                if state_name in ("left_contact", "right_contact")
                else output / "fea_states" / f"{design_name}_{state_name}.npz"
            )
            loaded_mesh, fea_record = _solve_contact(
                design["tip"],
                design["mesh"],
                scenarios[state_name],
                cache_path=cache_path,
            )
            design["states"][state_name] = loaded_mesh
            design["fea"][state_name] = fea_record

    event_catalog: dict[str, Any] = {}
    event_data: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in designs}
    for design_name in designs:
        for state_name in STATE_DEFINITIONS:
            state_event_path = output / "events" / f"{design_name}_state_{state_name}.npz"
            archived = None
            archived_path: Path | None = None
            archived = _load_archived_event_state(
                state_event_path,
                state_name,
                event_provenance,
            )
            if archived is None and state_name not in ("left_contact", "right_contact"):
                legacy_event_path = output / "events" / f"{design_name}_new_states.npz"
                archived = _load_archived_event_state(
                    legacy_event_path,
                    state_name,
                    event_provenance,
                )
                if archived is not None:
                    archived_path = legacy_event_path
            elif archived is not None:
                archived_path = state_event_path
            if archived is not None:
                event_data[design_name][state_name] = archived
                geometry_metadata = _archived_event_metadata(archived_path, state_name)
                event_catalog[f"{design_name}:{state_name}"] = {
                    "path": str(archived_path),
                    "prefix": state_name,
                    "reused": True,
                    "artifact_validated": True,
                    "provenance": "current_event_provenance",
                    "interface_normal_orientation_fallback_count": int(
                        geometry_metadata.get(
                            "interface_normal_orientation_fallback_count", 0
                        )
                    ),
                }

    runtime = create_runtime()
    new_event_results: dict[str, dict[str, Any]] = {name: {} for name in designs}
    trace_started = time.perf_counter()
    active_trace: str | None = None
    try:
        settings = _state_trace_settings()
        for design_name, design in designs.items():
            for state_name in STATE_DEFINITIONS:
                if state_name in event_data[design_name]:
                    continue
                active_trace = f"{design_name}:{state_name}"
                result = trace_3d(
                    design["tip"],
                    design["states"][state_name],
                    reference_mesh=design["mesh"],
                    settings=settings,
                    runtime=runtime,
                )
                new_event_results[design_name][state_name] = result
                event_data[design_name][state_name] = {
                    "positions_mm": np.asarray(result.escape_positions_mm),
                    "directions": np.asarray(result.escape_directions),
                    "weights": np.asarray(result.escape_weights),
                }
                checkpoint_path = output / "events" / f"{design_name}_state_{state_name}.npz"
                _save_event_archive(
                    checkpoint_path,
                    {state_name: result},
                    event_provenance,
                )
                event_catalog[f"{design_name}:{state_name}"] = {
                    "path": str(checkpoint_path),
                    "prefix": state_name,
                    "reused": False,
                    "artifact_validated": True,
                    "provenance": "current_event_provenance",
                    "interface_normal_orientation_fallback_count": int(
                        result.geometry_metadata.get(
                            "interface_normal_orientation_fallback_count", 0
                        )
                    ),
                }
                manifest.setdefault("completed_trace_states", []).append(
                    f"{design_name}:{state_name}"
                )
                manifest["status"] = "RUNNING"
                manifest["completed_stage"] = "baseline_trace"
                manifest["next_authorized_stage"] = "baseline_metric_assembly"
                _write_json(output / "run_manifest.json", manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED_INCOMPLETE",
                "completed_stage": "baseline_trace",
                "failed_trace_state": active_trace,
                "unresolved_blockers": [f"OptiX trace failed at {active_trace}: {error}"],
                "next_authorized_stage": "repair_or_resume_baseline_trace",
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        try:
            runtime.cp.cuda.Stream.null.synchronize()
            runtime.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    configuration = camera_configuration()
    responses: dict[str, dict[str, np.ndarray]] = {name: {} for name in designs}
    response_records: dict[str, dict[str, Any]] = {name: {} for name in designs}
    for design_name in designs:
        for state_name in STATE_DEFINITIONS:
            response, record = project_escape_events(
                event_data[design_name][state_name]["positions_mm"],
                event_data[design_name][state_name]["directions"],
                event_data[design_name][state_name]["weights"],
                configuration,
            )
            responses[design_name][state_name] = response
            response_records[design_name][state_name] = {
                **record,
                "total_camera_response_power": float(np.sum(response)),
                "response_shape": list(response.shape),
            }

    metrics: dict[str, Any] = {}
    for design_name in designs:
        design_responses = responses[design_name]
        jacobian, jacobian_metadata = centered_jacobian(design_responses)
        derivative = derivative_diagnostics(design_responses)
        fisher = fisher_from_response(
            design_responses["center_shallow"],
            jacobian,
        )
        pairwise = _predefined_pairwise(design_responses)
        all_pairwise: dict[str, Any] = {}
        state_names = tuple(STATE_DEFINITIONS)
        for first_index, first_name in enumerate(state_names):
            for second_name in state_names[first_index + 1 :]:
                all_pairwise[_pair_key(first_name, second_name)] = pairwise_separation(
                    design_responses[first_name],
                    design_responses[second_name],
                )
        metrics[design_name] = {
            "camera_power": {
                state: response_records[design_name][state]["total_camera_response_power"]
                for state in STATE_DEFINITIONS
            },
            "response_records": response_records[design_name],
            "pairwise": pairwise,
            "all_pairwise": all_pairwise,
            "finite_step_diagnostic": finite_step_diagnostic(design_responses),
            "jacobian": {
                "metadata": jacobian_metadata,
                "derivative_diagnostics": derivative,
                "common_random_numbers": paired_sampling["status"] == "PASS",
                "ray_sampling_variation": "deterministic paired source-ray sequence; no independent Monte Carlo sampling",
                "paired_sampling_contract": paired_sampling,
            },
            "fisher": fisher,
        }

    comparison = _summarize_design_comparison(metrics, intrinsic_summary)
    camera_diagnostic, diagnostic_arrays = camera_discretization_diagnostic(event_data)
    camera_diagnostic_path = output / "camera_discretization.npz"
    np.savez_compressed(camera_diagnostic_path, **diagnostic_arrays)
    diagnostic_assessment = camera_diagnostic["assessment"]
    power_invariant = all(
        value["pass"]
        for value in diagnostic_assessment["total_accepted_camera_power_invariant"].values()
    )
    comparison["branch_status"]["camera_pairwise"] = (
        "VERIFIED"
        if diagnostic_assessment["raw_pairwise_ordering_stable"] and power_invariant
        else "CONDITIONALLY_SUPPORTED"
    )
    comparison["branch_status"]["gain_profiled_pairwise"] = (
        "VERIFIED"
        if diagnostic_assessment["gain_profiled_pairwise_ordering_stable"] and power_invariant
        else "CONDITIONALLY_SUPPORTED"
    )
    comparison["camera_discretization_case"] = diagnostic_assessment["diagnostic_case"]
    responses_path = output / "responses.npz"
    response_arrays: dict[str, np.ndarray] = {}
    for design_name in designs:
        for state_name, response in responses[design_name].items():
            response_arrays[f"{design_name}_{state_name}_mu"] = response
        fisher_physical = np.asarray(
            metrics[design_name]["fisher"]["fisher"]["fisher_physical"]
        )
        response_arrays[f"{design_name}_center_shallow_fisher_physical"] = fisher_physical
        response_arrays[f"{design_name}_center_shallow_fisher_dimensionless"] = np.asarray(
            metrics[design_name]["fisher"]["fisher"]["fisher_dimensionless"]
        )
        response_arrays[f"{design_name}_center_shallow_crlb_physical"] = np.asarray(
            metrics[design_name]["fisher"]["fisher"]["crlb_physical"]
        )
        response_arrays[f"{design_name}_center_shallow_gain_effective_fisher"] = np.asarray(
            metrics[design_name]["fisher"]["gain_nuisance"]["effective_contact_fisher"]
        )
        response_arrays[f"{design_name}_center_shallow_jacobian_physical"] = np.column_stack(
            (
                (
                    responses[design_name]["near_right_contact"]
                    - responses[design_name]["near_left_contact"]
                ).reshape(-1)
                / 3.0,
                (
                    responses[design_name]["center_high"]
                    - responses[design_name]["center_low"]
                ).reshape(-1)
                / 0.5,
            )
        )
    np.savez_compressed(responses_path, **response_arrays)

    figure_path = _make_figure(output, responses, metrics)
    summary = {
        "status": "COMPLETE",
        "created_at": _now(),
        "git_revision": _git_revision(),
        "git_status": _git_status(),
        "output_directory": str(output),
        "precommit_choices": choices,
        "stencil_semantics": stencil_semantics,
        "paired_sampling_contract": paired_sampling,
        "event_provenance": event_provenance,
        "morphologies": {
            "nominal": {"parameters": {key: getattr(design_parameters["nominal"], key) for key in design_parameters["nominal"].__dataclass_fields__}},
            "candidate49": {"parameters": candidate_parameters},
        },
        "contact_state_domain": STATE_DEFINITIONS,
        "camera": configuration,
        "noise_model": choices["noise_model"],
        "state_scaling": choices["state_scaling"],
        "transport_settings": {
            "ray_count": RAY_COUNT,
            "minimum_ray_weight": MINIMUM_RAY_WEIGHT,
            "max_interactions": MAX_INTERACTIONS,
            "maximum_periodic_wraps": MAXIMUM_PERIODIC_WRAPS,
            "extrusion_depth_mm": EXTRUSION_DEPTH_MM,
            "surface_u_bins": SURFACE_U_BINS,
            "surface_z_bins": SURFACE_Z_BINS,
            "deterministic_common_ray_sequence": True,
            "event_provenance": event_provenance,
        },
        "mechanical_model": "deterministic 3D geometric-optics transport on an extruded mechanically deformed cross-section",
        "source_transport_artifact": {
            "fields": str(SOURCE_FIELDS),
            "summary": str(SOURCE_SUMMARY),
            "intrinsic_final_ordering": intrinsic_summary.get("final_ordering"),
            "sensor_escape_events_reused": False,
            "sensor_escape_event_reason": "older transport provenance; all baseline sensor event states traced under current event provenance",
        },
        "event_catalog": event_catalog,
        "interface_normal_orientation_fallback_counts": {
            key: value.get("interface_normal_orientation_fallback_count", 0)
            for key, value in event_catalog.items()
        },
        "new_trace_wall_time_seconds": time.perf_counter() - trace_started,
        "response_artifact": str(responses_path),
        "camera_discretization_artifact": str(camera_diagnostic_path),
        "figure": figure_path,
        "design_metrics": metrics,
        "camera_discretization_diagnostic": camera_diagnostic,
        "baseline_comparison": comparison,
        "reviewer_status": "pending_independent_read_only_review",
        "reviewer_report": None,
        "deferred_stages": [
            "conditional pending independent review and baseline validity",
        ],
    }
    _write_json(output / "summary.json", summary)
    manifest.update(
        {
            "completed_stage": "baseline",
            "status": "COMPLETE",
            "artifacts_produced": [
                str(output / "precommit_choices.json"),
                str(output / "run_manifest.json"),
                str(output / "summary.json"),
                str(responses_path),
                str(camera_diagnostic_path),
                str(output / "events"),
                str(output / "fea_states"),
            ],
            "reviewer_status": "pending_independent_read_only_review",
            "completed_trace_states": [
                key for key, record in event_catalog.items() if not record["reused"]
            ],
            "current_event_states": sorted(event_catalog),
            "current_event_state_provenance": "all states use event_provenance; legacy archives are not reused",
            "unresolved_blockers": (
                [
                    "baseline derivative-validity gate failed for nominal and candidate49; Fisher ordering is not certifiable",
                ]
                if comparison["baseline_outcome"] == "F_INCONCLUSIVE"
                else []
            ),
            "next_authorized_stage": "independent_review",
        }
    )
    _write_json(output / "run_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = run_benchmark(args.output)
    print(json.dumps({"status": summary["status"], "outcome": summary["baseline_comparison"]["baseline_outcome"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
