"""Shared configuration helpers for the production FULL_3D evaluator."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from model import Fingertip, FingertipParameters
from optics.transport3d import Transport3DSettings


LUMO3D_OBSERVATION_LEVEL = "FULL_3D native internal transport redistribution proxy"
CONTACT_STATE_SEPARATION_OBJECTIVE_NAME = "contact_state_separation"
LUMO3D_OPTICAL_X_BOUNDS_MM = (-16.0, 16.0)
LUMO3D_OPTICAL_Y_BOUNDS_MM = (-31.0, 4.5)


def optical_settings() -> Transport3DSettings:
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


def material(tip: Fingertip) -> dict[str, float]:
    return {
        "refractive_index_air": tip.optical.refractive_index_air,
        "refractive_index_silicone": tip.optical.refractive_index_silicone,
        "absorption_per_mm": tip.optical.absorption_per_mm,
        "scattering_per_mm": tip.optical.scattering_per_mm,
    }


def candidate_id(parameters: FingertipParameters) -> str:
    payload = {
        name: float(getattr(parameters, name))
        for name in (
            "flat_pad_height",
            "semielliptical_pad_height",
            "stem_width",
            "stem_height",
            "void_width",
            "void_height",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def energy_record(result: Any) -> dict[str, Any]:
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


__all__ = [
    "CONTACT_STATE_SEPARATION_OBJECTIVE_NAME",
    "LUMO3D_OBSERVATION_LEVEL",
    "LUMO3D_OPTICAL_X_BOUNDS_MM",
    "LUMO3D_OPTICAL_Y_BOUNDS_MM",
    "candidate_id",
    "energy_record",
    "material",
    "optical_settings",
]
