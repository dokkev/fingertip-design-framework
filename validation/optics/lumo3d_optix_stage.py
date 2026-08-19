"""Run the first real FULL_3D OptiX states from the LUMO 3D handoff."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mechanics3d import prepare_fingertip_mechanics_mesh
from mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from model import Fingertip
from optics.transport3d import (
    OptiXTransport,
    Transport3DSettings,
    build_fingertip_volume_state_geometry,
    fingerprint_mapping,
    native_field_separability,
    save_case_artifact,
    transport_configuration,
)
from optics.transport3d.optix_backend import create_runtime
from optics.optix.smoke import run_production_optix_smoke
from validation.mechanics3d.deformed_state_artifact import restore_deformed_optical_state


def _settings(
    *,
    x_bounds_mm: tuple[float, float],
    y_bounds_mm: tuple[float, float],
) -> Transport3DSettings:
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
        x_bounds_mm=x_bounds_mm,
        y_bounds_mm=y_bounds_mm,
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


def _contact_state(case: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    return {
        "contact_state_fingerprint": fingerprint,
        "normalized_location": case["normalized_location"],
        "target_point_mm": case["target_point_mm"],
        "outward_normal": case["outward_normal"],
        "approach_direction": case["approach_direction"],
        "indenter_radius_mm": 5.0,
        "initial_gap_mm": 0.25,
        "first_contact_travel_mm": case["first_contact_travel_mm"],
        "post_contact_travel_mm": case["post_contact_travel_mm"],
        "spawn_clearance_mm": case["spawn_clearance_mm"],
    }


def _result_summary(result, *, artifact: Path, contract: dict[str, Any]) -> dict[str, Any]:
    field = np.asarray(result.field)
    return {
        "artifact": str(artifact),
        "artifact_field": str(artifact.with_suffix(".npz")),
        "contract_fingerprint": fingerprint_mapping(contract),
        "geometry_provenance": contract.get("geometry_provenance"),
        "optical_mode": result.optical_mode,
        "ray_count": result.ray_count,
        "field_shape": list(field.shape),
        "field_axis_order": "x,y,z",
        "field_finite_nonnegative": bool(np.all(np.isfinite(field)) and np.all(field >= 0.0)),
        "field_mass": float(np.sum(field)),
        "launched_weight": result.launched_weight,
        "escaped_weight": result.escaped_weight,
        "absorbed_weight": result.absorbed_weight,
        "terminated_weight": result.terminated_weight,
        "object_interface_incident_weight": result.object_interface_incident_weight,
        "object_absorbed_weight": result.object_absorbed_weight,
        "object_transmitted_weight": result.object_transmitted_weight,
        "object_reflected_weight": result.object_reflected_weight,
        "energy_balance_error": result.energy_balance_error,
        "path_diagnostics": dict(result.path_diagnostics),
    }


def _validated_preflight(preflight: dict[str, Any] | None) -> dict[str, Any]:
    """Require the shared real-runtime smoke contract before tracing.

    A caller may provide preserved smoke evidence, but it must explicitly carry
    a PASS status.  Omitting the evidence runs the same smoke implementation
    in-process; a failed smoke therefore stops this stage before any geometry
    or transport work is started.
    """
    if preflight is None:
        result = run_production_optix_smoke()
        return {"status": "PASS", "evidence": result.to_dict()}
    if preflight.get("status") != "PASS":
        raise RuntimeError(
            "FULL_3D OptiX stage requires a passing production preflight; "
            f"received status={preflight.get('status')!r}"
        )
    return dict(preflight)


def run_lumo3d_optix_stage(
    root: str | Path,
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trace reference and all persisted deformed contact states with OptiX."""

    root = Path(root)
    preflight_evidence = _validated_preflight(preflight)
    stage2_payload = json.loads((root / "stage2_multi_location_contact.json").read_text())
    stage3_payload = json.loads((root / "stage3_deformed_optical_geometry.json").read_text())
    tip = Fingertip()
    volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
    prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
    min_x, min_y, max_x, max_y = tip.geometry.material_geometry.bounds
    fingertip_margin_mm = 1.0
    settings = _settings(
        x_bounds_mm=(float(min_x - fingertip_margin_mm), float(max_x + fingertip_margin_mm)),
        y_bounds_mm=(float(min_y - fingertip_margin_mm), float(max_y + fingertip_margin_mm)),
    )
    material = _material(tip)
    runtime = create_runtime()
    transport = OptiXTransport()
    output = root / "observations" / "stage4_optix"
    output.mkdir(parents=True, exist_ok=True)
    reference_state = FingertipVolumeState.reference(volume_mesh)
    reference_fingerprint = "reference-unloaded"
    reference_contact = {"contact_state_fingerprint": reference_fingerprint, "normalized_location": None}
    reference_geometry = build_fingertip_volume_state_geometry(
        tip,
        reference_state,
        reference_mesh=tip.mesh(),
        metadata={
            "contact_state_fingerprint": reference_fingerprint,
            "optical_state_id": "reference-unloaded",
            "full3d_surface_provenance": "actual_reference_3d_volume_state",
            "mechanics_contract": stage2_payload["search_contract"],
        },
    )
    if reference_geometry.geometry_mode != "full3d_surface":
        raise RuntimeError("reference geometry is not FULL_3D")
    configuration = transport_configuration(
        settings,
        material=material,
        source={"model": "existing Fingertip optical source"},
    )
    reference_contract = {
        "schema": "lumo3d-optix-stage-v1",
        "morphology_id": "nominal",
        "morphology_parameters_fingerprint": volume_mesh.morphology_fingerprint,
        "morphology_fingerprint": volume_mesh.morphology_fingerprint,
        "mechanics_dimension": "3D",
        "mechanics_source": "reference_volume_state",
        "optical_mode": "FULL_3D",
        "ray_count": settings.ray_count,
        "contact_location": "reference",
        "contact_state_fingerprint": reference_fingerprint,
        "geometry_provenance": "actual_reference_3d_volume_state",
        "transport_configuration": configuration,
        "transport_configuration_fingerprint": fingerprint_mapping(configuration),
    }
    reference_result = transport.trace(
        tip,
        reference_geometry,
        settings=settings,
        morphology_id="nominal",
        morphology_fingerprint=volume_mesh.morphology_fingerprint,
        mechanics_source="reference_volume_state",
        mechanics_dimension="3D",
        contact_state=reference_contact,
        transport_configuration=configuration,
        runtime=runtime,
    )
    reference_path = output / "reference.json"
    save_case_artifact(reference_path, reference_result, reference_contract)
    records = {
        "reference": _result_summary(reference_result, artifact=reference_path, contract=reference_contract)
    }
    separability: dict[str, Any] = {}
    stage3_by_location = {
        float(item["normalized_location"]): item for item in stage3_payload["locations"]
    }
    stage2_by_location = {
        float(item["normalized_location"]): item for item in stage2_payload["locations"]
    }
    for location in (0.25, 0.50, 0.75):
        stage2_case = stage2_by_location[location]
        stage3_case = stage3_by_location[location]
        contact_fingerprint = str(stage3_case["contact_state_fingerprint"])
        contact_state = _contact_state(stage2_case, contact_fingerprint)
        restored = restore_deformed_optical_state(
            tip,
            volume_mesh,
            prepared,
            stage3_case["artifact_path"],
            stage3_case["artifact_sha256"],
            metadata={
                "contact_state_fingerprint": contact_fingerprint,
                "contact_location_u": location,
                "mechanics_contract": stage2_payload["search_contract"],
            },
        )
        if restored.geometry.geometry_mode != "full3d_surface":
            raise RuntimeError(f"u={location:g} geometry is not FULL_3D")
        contract = {
            "schema": "lumo3d-optix-stage-v1",
            "morphology_id": "nominal",
            "morphology_parameters_fingerprint": volume_mesh.morphology_fingerprint,
            "morphology_fingerprint": volume_mesh.morphology_fingerprint,
            "mechanics_dimension": "3D",
            "mechanics_source": str(restored.artifact_path),
            "mechanics_artifact_sha256": restored.artifact_sha256,
            "optical_state_id": restored.state_id,
            "optical_mode": "FULL_3D",
            "ray_count": settings.ray_count,
            "contact_location": location,
            "contact_state_fingerprint": contact_fingerprint,
            "geometry_provenance": "actual_deformed_3d_volume_state",
            "contact_state": contact_state,
            "mechanics_contract": stage2_payload["search_contract"],
            "transport_configuration": configuration,
            "transport_configuration_fingerprint": fingerprint_mapping(configuration),
        }
        result = transport.trace(
            tip,
            restored.geometry,
            settings=settings,
            morphology_id="nominal",
            morphology_fingerprint=volume_mesh.morphology_fingerprint,
            mechanics_source=str(restored.artifact_path),
            mechanics_dimension="3D",
            contact_state=contact_state,
            transport_configuration=configuration,
            runtime=runtime,
        )
        path = output / f"location_u_{location:.3f}.json"
        save_case_artifact(path, result, contract)
        records[f"u={location:.3f}"] = _result_summary(result, artifact=path, contract=contract)
        separability[f"u={location:.3f}"] = native_field_separability(reference_result, result)
    summary = {
        "schema": "lumo3d-optix-stage-v1",
        "optical_mode": "FULL_3D",
        "settings": asdict(settings),
        "transport_configuration": configuration,
        "transport_configuration_fingerprint": fingerprint_mapping(configuration),
        "preflight": preflight_evidence,
        "object_interface_optics": "not_present_in_deformation_only_scene",
        "scientific_observation_level": "internal_transport_redistribution_proxy",
        "object_interface_note": (
            "The fixed reference/indenter carrier is geometric only in this "
            "stage; object-interface incident/transmitted/reflected channels "
            "are not an optical contact measurement."
        ),
        "records": records,
        "reference_vs_loaded_separability": separability,
        "generated_artifact_directory": str(output),
    }
    (root / "stage4_full3d_optix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary


__all__ = ["run_lumo3d_optix_stage"]
