"""Bounded orchestration for the unified PLANAR_2D/FULL_3D OptiX study.

This module does not solve FEA.  It consumes separately persisted 2D state
artifacts and true deformed 3D surface artifacts, and refuses to promote
M5 validation summaries into geometry when the latter are absent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from model import Fingertip, FingertipParameters
from model.solid import build_fingertip_solid
from optics.transport3d.result import Transport3DResult
from optics.transport3d.unified import (
    OptiXTransport,
    UnifiedTransportResult,
    load_case_artifact,
    native_field_separability,
    save_case_artifact,
    fingerprint_mapping,
    transport_configuration,
)
from optics.transport3d.artifact import (
    FULL3D_SURFACE_SCHEMA,
    NATIVE_3D_FEA_STATE_SCHEMA,
    load_full3d_surface_artifact,
)
from optics.transport3d.geometry import ExtrudedTransportGeometry
from optics.transport3d.settings import Transport3DSettings
from optics.transport3d.transport import trace_3d
from mesh import mesh_settings_for_level
from optics.transport import TransportResult
from validation.optics.transport3d_validation import CANDIDATE49


OUTPUT = Path("output/optix_unified_transport")
M5_OUTPUT = Path("output/validation/3d_migration/m5_cases")
PLANAR_OUTPUT = Path("output/optix_design_verification/baseline")
M5_CASES = (
    ("nominal", "search", "left"),
    ("nominal", "search", "right"),
    ("candidate49", "search", "left"),
    ("candidate49", "reference", "left"),
)
TRAVEL_MM = 0.5
INITIAL_GAP_MM = 0.2
CONTACT_X_MM = {"left": -3.0, "right": 3.0}
INDENTER_RADIUS_MM = 4.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contact_state_fingerprint(payload: Mapping[str, Any]) -> str:
    """Derive a stable contact-state identity without rewriting M5 history."""
    state = {
        "case_fingerprint": payload.get("case_fingerprint"),
        "morphology_fingerprint": payload.get("morphology_fingerprint"),
        "contact_location": payload.get("contact_location"),
        "contact_location_mm": payload.get("contact_location_mm"),
        "steps": payload.get("steps"),
        "configuration": payload.get("configuration", {}).get("settings", {}),
        "contact_state": payload.get("contact_state", {}),
        "active_contact_diagnostics": payload.get("active_contact_diagnostics", {}),
        "surface_validity": payload.get("surface_validity", {}),
    }
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def morphology_parameters(name: str) -> FingertipParameters:
    if name == "nominal":
        return FingertipParameters()
    if name == "candidate49":
        return FingertipParameters(**CANDIDATE49)
    raise ValueError(f"unsupported morphology: {name!r}")


def m5_case_path(morphology: str, tier: str, location: str) -> Path:
    return M5_OUTPUT / f"{morphology}__{tier}__12__{location}.json"


def load_valid_m5_case(morphology: str, tier: str, location: str) -> dict[str, Any]:
    """Load the existing M5 summary and reject stale/non-production cases."""
    path = m5_case_path(morphology, tier, location)
    if not path.exists():
        raise ValueError(f"required M5 artifact is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = record.get("payload")
    expected_key = f"{morphology}:{tier}:12:{location}"
    if (
        record.get("schema") != "m5-production-case-v1"
        or record.get("key") != expected_key
        or record.get("status") != "PASS"
        or record.get("outcome") != "PASS"
    ):
        raise ValueError(f"M5 artifact is not a successful child: {path}")
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
        raise ValueError(f"M5 artifact has no successful production payload: {path}")
    expected_morphology_fingerprint = build_fingertip_solid(
        Fingertip(morphology_parameters(morphology)).geometry
    ).morphology_fingerprint
    if (
        payload.get("morphology") != morphology
        or payload.get("mesh_tier") != tier
        or payload.get("steps") != 12
        or payload.get("morphology_fingerprint") != expected_morphology_fingerprint
    ):
        raise ValueError(f"M5 artifact morphology/tier contract mismatch: {path}")
    from validation.three_d_migration import _m5_case_fingerprint

    expected_case_fingerprint = _m5_case_fingerprint(
        morphology,
        tier,
        12,
        location,
        expected_morphology_fingerprint,
    )
    if (
        record.get("case_fingerprint") != expected_case_fingerprint
        or payload.get("case_fingerprint") != expected_case_fingerprint
    ):
        raise ValueError(f"M5 artifact case fingerprint mismatch: {path}")
    checks = payload.get("checks", {})
    required_checks = (
        "converged",
        "active_mortar_conditions",
        "deformed_surface_valid",
        "finite_nonzero_reaction",
    )
    if any(checks.get(name) is not True for name in required_checks):
        raise ValueError(f"M5 artifact did not satisfy mechanics contract: {path}")
    settings = payload.get("configuration", {}).get("settings", {})
    if (
        settings.get("indentation_mm") != TRAVEL_MM
        or settings.get("external_contact") is not True
        or settings.get("number_of_steps") != 12
    ):
        raise ValueError(f"M5 artifact has mismatched mechanics settings: {path}")
    if payload.get("contact_location") != location:
        raise ValueError(f"M5 artifact contact location mismatch: {path}")
    if payload.get("contact_location_mm") != CONTACT_X_MM[location]:
        raise ValueError(f"M5 artifact contact x mismatch: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "contact_state_fingerprint": _contact_state_fingerprint(payload),
        "record": record,
        "payload": dict(payload),
    }


def case_contract(
    *,
    morphology: str,
    mechanics_dimension: str,
    mechanics_mode: str,
    fea_artifact: str,
    contact_location: str,
    optical_mode: str,
    ray_count: int,
    optical_configuration: Mapping[str, Any],
    contact_state_fingerprint: str | None = None,
    transport_configuration_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return the complete per-case provenance contract."""
    if mechanics_dimension not in ("2D", "3D"):
        raise ValueError("mechanics_dimension must be 2D or 3D")
    if optical_mode not in ("PLANAR_2D", "FULL_3D"):
        raise ValueError("optical_mode must be PLANAR_2D or FULL_3D")
    parameters = asdict(morphology_parameters(morphology))
    fea_path = Path(fea_artifact)
    if not fea_path.exists():
        raise ValueError(f"FEA artifact does not exist: {fea_path}")
    morphology_fingerprint = build_fingertip_solid(
        Fingertip(morphology_parameters(morphology)).geometry
    ).morphology_fingerprint
    return {
        "schema": "unified-optix-case-contract-v1",
        "morphology_id": morphology,
        "morphology_parameters": parameters,
        "morphology_parameters_fingerprint": morphology_fingerprint,
        "mechanics_dimension": mechanics_dimension,
        "mechanics_mode": mechanics_mode,
        "fea_artifact": fea_artifact,
        "fea_artifact_sha256": _sha256(fea_path),
        "mechanics_source": fea_artifact,
        "contact_location": contact_location,
        "contact_state_fingerprint": contact_state_fingerprint,
        "contact_x_mm": CONTACT_X_MM[contact_location],
        "initial_gap_mm": INITIAL_GAP_MM,
        "total_prescribed_travel_mm": TRAVEL_MM,
        "indenter_radius_mm": INDENTER_RADIUS_MM,
        "optical_mode": optical_mode,
        "ray_count": int(ray_count),
        "optical_configuration": dict(optical_configuration),
        "transport_configuration_fingerprint": transport_configuration_fingerprint,
    }


def find_full3d_surface_manifest(
    m5_artifact: Mapping[str, Any],
    *,
    expected_contact_state_fingerprint: str | None = None,
) -> Path | None:
    """Return an explicitly linked true-surface artifact, if one exists."""
    payload = m5_artifact.get("payload", {})
    for key in (
        "native_3d_artifact",
        "deformed_surface_artifact",
        "full3d_surface_artifact",
        "surface_artifact",
    ):
        value = payload.get(key)
        if value:
            path = Path(str(value))
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("schema") not in {FULL3D_SURFACE_SCHEMA, NATIVE_3D_FEA_STATE_SCHEMA}:
                continue
            if manifest.get("morphology_fingerprint") != payload.get(
                "morphology_fingerprint"
            ):
                continue
            contact_fingerprint = (
                expected_contact_state_fingerprint
                or payload.get("contact_state_fingerprint")
            )
            if not contact_fingerprint or manifest.get("contact_state_fingerprint") != contact_fingerprint:
                continue
            surface_value = manifest.get("native_state_artifact", manifest.get("surface_artifact"))
            if not surface_value:
                continue
            surface_path = Path(str(surface_value))
            if not surface_path.is_absolute():
                surface_path = path.parent / surface_path
            expected_surface_sha = manifest.get("native_state_sha256", manifest.get("surface_sha256"))
            if not surface_path.exists() or expected_surface_sha != _sha256(surface_path):
                continue
            return path
    return None


def find_existing_planar_fea_artifact(morphology: str) -> Path | None:
    """Find a validated persisted 2D state bundle without solving FEA.

    The older sensor-facing run stores the 2D FEA displacement states beside
    its summary rather than as ``fea_record.json``/``states.npz``.  It is
    valid reduced-mechanics input when its morphology parameters and contact
    states match the current contract.  Its historical OptiX fields/events
    are deliberately not consumed here.
    """
    expected_parameters = asdict(morphology_parameters(morphology))

    def parameters_match(observed: Any) -> bool:
        if not isinstance(observed, Mapping) or set(observed) != set(expected_parameters):
            return False
        return all(
            np.isclose(float(observed[name]), float(expected), rtol=0.0, atol=1.0e-12)
            for name, expected in expected_parameters.items()
        )

    candidate_summaries = [PLANAR_OUTPUT / "summary.json"]
    candidate_summaries.extend(
        sorted(Path("output/validation/optics").rglob("summary.json"))
    )
    for summary_path in candidate_summaries:
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") in ("FAILED", "BLOCKED", "FAILED_INCOMPLETE"):
            continue
        morphology_record = summary.get("morphologies", {}).get(morphology, {})
        observed_parameters = morphology_record.get("parameters")
        if morphology == "candidate49" and observed_parameters is None:
            observed_parameters = summary.get("candidate_parameters")
        if not parameters_match(observed_parameters):
            continue
        state_domain = summary.get("contact_state_domain", {})
        state_contracts = {
            state: state_domain.get(state, {})
            for state in ("left_contact", "right_contact")
        }
        if not state_domain:
            state_contracts = {
                state: summary.get("fem", {}).get(morphology, {}).get(state, {}).get(
                    "scenario", {}
                )
                for state in ("left_contact", "right_contact")
            }
        if any(
            state_contracts[state].get("x_c_mm", state_contracts[state].get("location_x_mm"))
            != CONTACT_X_MM[location]
            or state_contracts[state].get("delta_mm", state_contracts[state].get("indentation_mm"))
            != TRAVEL_MM
            or state_contracts[state].get("indenter_radius_mm")
            not in (None, INDENTER_RADIUS_MM)
            for location, state in (("left", "left_contact"), ("right", "right_contact"))
        ):
            continue
        state_paths = {
            location: summary_path.parent / "fea_states" / f"{morphology}_{state}.npz"
            for location, state in (("left", "left_contact"), ("right", "right_contact"))
        }
        valid = True
        for state_path in state_paths.values():
            if not state_path.exists():
                valid = False
                break
            try:
                with np.load(state_path, allow_pickle=False) as archive:
                    displacement = np.asarray(archive["displacement"], dtype=float)
            except (KeyError, OSError, ValueError):
                valid = False
                break
            if (
                displacement.ndim != 2
                or displacement.shape[1] != 2
                or displacement.shape[0] == 0
                or not np.all(np.isfinite(displacement))
            ):
                valid = False
                break
        if valid:
            return summary_path
    fingerprint = build_fingertip_solid(
        Fingertip(morphology_parameters(morphology)).geometry
    ).morphology_fingerprint
    for record_path in sorted(Path("output/validation/optics").rglob("fea_record.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("morphology_fingerprint") != fingerprint:
            continue
        states_path = Path(str(record.get("states_path", "")))
        if not states_path.is_absolute():
            states_path = record_path.parent / states_path.name
        if not states_path.exists():
            continue
        try:
            with np.load(states_path, allow_pickle=False) as archive:
                if {"left_0p5", "right_0p5"}.issubset(archive.files):
                    return states_path
        except (OSError, ValueError):
            continue
    return None


def planar_regression_case(
    legacy: TransportResult,
    planar_raw: Transport3DResult,
    planar: UnifiedTransportResult,
) -> dict[str, Any]:
    """Compare one PLANAR_2D OptiX case with the retained custom tracer.

    The report is an implementation diagnostic only.  It deliberately does
    not turn agreement with the historical tracer into a scientific truth
    claim.
    """
    if planar.optical_mode != "PLANAR_2D" or planar_raw.source_mode != "planar":
        raise ValueError("planar_regression_case requires PLANAR_2D inputs")
    legacy_totals = np.asarray(
        [legacy.escaped_weight, legacy.absorbed_weight, legacy.terminated_weight],
        dtype=float,
    )
    optix_totals = np.asarray(
        [planar_raw.escaped_weight, planar_raw.absorbed_weight, planar_raw.terminated_weight],
        dtype=float,
    )
    total_scale = max(float(np.max(np.abs(legacy_totals))), 1.0e-30)
    total_relative_error = float(np.max(np.abs(legacy_totals - optix_totals)) / total_scale)
    path_count = None
    path_status = "UNCLEAR"
    if planar_raw.retained_segment_lengths_mm is not None:
        path_count = {
            "legacy_segment_count": len(legacy.segments),
            "optix_segment_count": len(planar_raw.retained_segment_lengths_mm),
        }
        path_status = "PASS" if path_count["optix_segment_count"] > 0 else "FAIL"
    spatial = {
        "status": "UNCLEAR",
        "normalized_cosine": None,
        "normalized_l1": None,
    }
    legacy_field_xy = np.asarray(legacy.density, dtype=float).T
    if legacy_field_xy.shape == planar.field.shape:
        first = legacy_field_xy
        second = np.asarray(planar.field, dtype=float)
        first_mass = float(np.sum(first))
        second_mass = float(np.sum(second))
        if first_mass > 0.0 and second_mass > 0.0:
            first_normalized = first / first_mass
            second_normalized = second / second_mass
            cosine_denominator = float(
                np.linalg.norm(first_normalized) * np.linalg.norm(second_normalized)
            )
            spatial = {
                "status": "PASS" if cosine_denominator > 0.0 else "FAIL",
                "normalized_cosine": (
                    float(np.sum(first_normalized * second_normalized) / cosine_denominator)
                    if cosine_denominator > 0.0
                    else None
                ),
                "normalized_l1": float(
                    0.5 * np.sum(np.abs(first_normalized - second_normalized))
                ),
            }
    return {
        "status": (
            "PASS"
            if total_relative_error <= 0.05
            and spatial["status"] == "PASS"
            and path_status in ("PASS", "UNCLEAR")
            else "FAIL"
            if total_relative_error > 0.25
            else "UNCLEAR"
        ),
        "interpretation": "implementation sanity only; not scientific validation",
        "planar_direction_invariant": dict(
            planar_raw.geometry_metadata.get("planar_direction_invariant", {})
        ),
        "total_relative_error": total_relative_error,
        "legacy_totals": legacy_totals.tolist(),
        "optix_totals": optix_totals.tolist(),
        "major_path_diagnostics": path_count,
        "major_path_status": path_status,
        "spatial_structure": spatial,
    }


def reduced_full_smoke_comparison(
    planar_results: Mapping[str, Mapping[str, UnifiedTransportResult]],
    full_results: Mapping[str, Mapping[str, UnifiedTransportResult]],
) -> dict[str, Any]:
    """Compute separate J_planar/J_full3D values for the two anchor morphologies."""
    records: dict[str, Any] = {}
    for morphology in ("nominal", "candidate49"):
        if morphology not in planar_results or morphology not in full_results:
            records[morphology] = {"status": "UNCLEAR", "reason": "missing morphology"}
            continue
        planar_states = planar_results[morphology]
        full_states = full_results[morphology]
        if any(name not in planar_states for name in ("left", "right")) or any(
            name not in full_states for name in ("left", "right")
        ):
            records[morphology] = {"status": "UNCLEAR", "reason": "missing left/right state"}
            continue
        planar_j = native_field_separability(planar_states["left"], planar_states["right"])
        full_j = native_field_separability(full_states["left"], full_states["right"])
        records[morphology] = {
            "status": "PASS",
            "J_planar": planar_j,
            "J_full3D": full_j,
            "comparison_scope": "two-anchor smoke only; no rank claim",
        }
    return {
        "status": "PASS" if all(record.get("status") == "PASS" for record in records.values()) else "UNCLEAR",
        "morphologies": records,
        "total_transport_preserved_separately": True,
        "native_full_field_preserved": True,
    }


def run_or_reuse_optix_case(
    *,
    output: Path,
    tip: Fingertip,
    geometry: ExtrudedTransportGeometry,
    settings: Transport3DSettings,
    contract: Mapping[str, Any],
    morphology_id: str,
    morphology_fingerprint: str,
    mechanics_source: str,
    mechanics_dimension: str,
    contact_state: Mapping[str, Any],
    transport_configuration: Mapping[str, Any],
    runtime: Any | None = None,
) -> tuple[UnifiedTransportResult, dict[str, Any]]:
    """Reuse one exact optical artifact or trace only that invalidated case."""
    expected_contact = contract.get("contact_state_fingerprint")
    if expected_contact is not None and contact_state.get("contact_state_fingerprint") != expected_contact:
        raise ValueError("case contact-state fingerprint does not match its contract")
    expected_configuration = contract.get("transport_configuration_fingerprint")
    actual_configuration = fingerprint_mapping(dict(transport_configuration))
    if expected_configuration is not None and expected_configuration != actual_configuration:
        raise ValueError("case transport configuration does not match its contract")
    output.mkdir(parents=True, exist_ok=True)
    case_name = (
        f"{morphology_id}__{contract['contact_location']}__"
        f"{contract['optical_mode']}__{settings.ray_count}"
    )
    metadata_path = output / f"{case_name}.json"
    try:
        reused = load_case_artifact(metadata_path, expected_contract=contract)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        reused = None
    if reused is not None:
        return reused, {
            "artifact": str(metadata_path),
            "reused": True,
            "fea_rerun": False,
        }
    result = OptiXTransport().trace(
        tip,
        geometry,
        settings=settings,
        morphology_id=morphology_id,
        morphology_fingerprint=morphology_fingerprint,
        mechanics_source=mechanics_source,
        mechanics_dimension=mechanics_dimension,
        contact_state=contact_state,
        transport_configuration=transport_configuration,
        runtime=runtime,
    )
    save_case_artifact(metadata_path, result, contract)
    return result, {
        "artifact": str(metadata_path),
        "reused": False,
        "fea_rerun": False,
    }


def inspect_existing_m5_artifacts(output: Path = OUTPUT) -> dict[str, Any]:
    """Assemble a no-FEA artifact-only readiness manifest."""
    output.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    for morphology, tier, location in M5_CASES:
        key = f"{morphology}:{tier}:12:{location}"
        try:
            m5 = load_valid_m5_case(morphology, tier, location)
            surface = find_full3d_surface_manifest(
                m5,
                expected_contact_state_fingerprint=m5["contact_state_fingerprint"],
            )
            planar_fea = find_existing_planar_fea_artifact(morphology)
            cases[key] = {
                "status": (
                    "READY"
                    if surface is not None and planar_fea is not None
                    else "BLOCKED_MISSING_3D_SURFACE"
                    if surface is None
                    else "BLOCKED_MISSING_2D_STATE"
                ),
                "m5_artifact": m5["path"],
                "m5_artifact_sha256": m5["sha256"],
                "morphology_fingerprint": m5["payload"].get("morphology_fingerprint"),
                "full3d_surface_artifact": None if surface is None else str(surface),
                "planar_fea_artifact": None if planar_fea is None else str(planar_fea),
                "fea_rerun": False,
                "interpretation": (
                    "M5 summary is valid but does not persist the deformed 3D surface coordinates"
                    if surface is None
                    else "the exact-fingerprint 2D deformed state bundle is missing"
                    if planar_fea is None
                    else "an explicitly linked true deformed 3D surface artifact is available"
                ),
            }
        except ValueError as exc:
            cases[key] = {"status": "FAIL", "error": str(exc), "fea_rerun": False}
    summary = {
        "schema": "unified-optix-readiness-v1",
        "created_at": _now(),
        "scope": "artifact-only; no 3D FEA rerun",
        "required_m5_cases": list(M5_CASES),
        "cases": cases,
        "full3d_ready": all(value["status"] == "READY" for value in cases.values()),
        "scientific_gate": (
            "READY_FOR_OPTIX_SMOKE"
            if all(value["status"] == "READY" for value in cases.values())
            else "BLOCKED_UNTIL_TRUE_DEFORMED_3D_SURFACE_ARTIFACTS_EXIST"
        ),
    }
    path = output / "readiness_manifest.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _smoke_settings(mode: str) -> Transport3DSettings:
    return Transport3DSettings(
        mode=mode,  # type: ignore[arg-type]
        ray_count=1024,
        max_interactions=10,
        maximum_segment_count=24000,
        maximum_periodic_wraps=32,
        terminate_on_periodic_wrap_limit=True,
        terminate_on_no_event=True,
        internal_grid_width=48,
        internal_grid_height=48,
        internal_z_bins=16,
        retain_projected_segments=mode == "planar",
        retain_internal_path_field=mode == "full3d",
    )


def _optical_configuration(tip: Fingertip, settings: Transport3DSettings) -> dict[str, Any]:
    return transport_configuration(
        settings,
        material={
            "refractive_index_air": tip.optical.refractive_index_air,
            "refractive_index_silicone": tip.optical.refractive_index_silicone,
            "absorption_per_mm": tip.optical.absorption_per_mm,
            "scattering_per_mm": tip.optical.scattering_per_mm,
        },
    )


def run_nominal_full3d_smoke(output: Path = OUTPUT) -> dict[str, Any]:
    """Run one native FULL_3D smoke and one matched nominal PLANAR_2D smoke."""
    morphology = "nominal"
    m5 = load_valid_m5_case(morphology, "search", "left")
    surface_manifest = find_full3d_surface_manifest(
        m5, expected_contact_state_fingerprint=m5["contact_state_fingerprint"]
    )
    if surface_manifest is None:
        raise RuntimeError("nominal M5 child has no exact-fingerprint native 3D surface manifest")
    parameters = morphology_parameters(morphology)
    expected_morphology = build_fingertip_solid(Fingertip(parameters).geometry).morphology_fingerprint
    artifact = load_full3d_surface_artifact(
        surface_manifest,
        expected_morphology_fingerprint=expected_morphology,
        expected_contact_state_fingerprint=m5["contact_state_fingerprint"],
    )
    tip = Fingertip(parameters)
    contact_state = {
        "contact_state_fingerprint": m5["contact_state_fingerprint"],
        "contact_location": "left",
        "contact_x_mm": CONTACT_X_MM["left"],
        "initial_gap_mm": INITIAL_GAP_MM,
        "total_prescribed_travel_mm": TRAVEL_MM,
        "indenter_radius_mm": INDENTER_RADIUS_MM,
    }
    full_settings = _smoke_settings("full3d")
    full_configuration = _optical_configuration(tip, full_settings)
    full_contract = case_contract(
        morphology=morphology,
        mechanics_dimension="3D",
        mechanics_mode="production",
        fea_artifact=str(surface_manifest),
        contact_location="left",
        optical_mode="FULL_3D",
        ray_count=full_settings.ray_count,
        optical_configuration=full_configuration,
        contact_state_fingerprint=m5["contact_state_fingerprint"],
        transport_configuration_fingerprint=fingerprint_mapping(full_configuration),
    )
    geometry = artifact.geometry(tip)
    transport = OptiXTransport()
    first = transport.trace(
        tip, geometry, settings=full_settings, morphology_id=morphology,
        morphology_fingerprint=expected_morphology, mechanics_source=str(surface_manifest),
        mechanics_dimension="3D", contact_state=contact_state,
        transport_configuration=full_configuration,
    )
    second = transport.trace(
        tip, geometry, settings=full_settings, morphology_id=morphology,
        morphology_fingerprint=expected_morphology, mechanics_source=str(surface_manifest),
        mechanics_dimension="3D", contact_state=contact_state,
        transport_configuration=full_configuration,
    )
    deterministic = bool(
        np.array_equal(first.field, second.field)
        and all(np.array_equal(left, right) for left, right in zip(first.field_axes, second.field_axes))
        and first.total_transport == second.total_transport
        and first.energy_balance_error == second.energy_balance_error
    )
    full_artifact_path = output / "nominal__left__FULL_3D__1024.json"
    save_case_artifact(full_artifact_path, first, full_contract)

    planar_summary = find_existing_planar_fea_artifact(morphology)
    if planar_summary is None:
        raise RuntimeError("nominal reduced 2D state artifact is missing")
    planar_state_path = planar_summary.parent / "fea_states" / "nominal_left_contact.npz"
    with np.load(planar_state_path, allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement"], dtype=float)
    planar_mesh = tip.mesh(mesh_settings_for_level("medium"))
    if displacement.shape != planar_mesh.pad.coordinates.shape:
        raise RuntimeError("nominal reduced state topology does not match the current medium mesh")
    planar_state = planar_mesh.pad.deformed(
        displacement, metadata={"source": str(planar_state_path)}
    )
    planar_settings = _smoke_settings("planar")
    planar_configuration = _optical_configuration(tip, planar_settings)
    planar_raw = trace_3d(
        tip, planar_state, reference_mesh=planar_mesh, settings=planar_settings
    )
    planar_result = UnifiedTransportResult.from_transport_result(
        planar_raw, morphology_id=morphology, morphology_fingerprint=expected_morphology,
        mechanics_source=str(planar_summary), mechanics_dimension="2D",
        contact_state=contact_state,
        transport_configuration_fingerprint=fingerprint_mapping(planar_configuration),
    )
    result = {
        "schema": "native-full3d-smoke-v1",
        "status": "PASS" if deterministic and first.field.size and planar_result.field.size else "FAIL",
        "scope": "nominal one-side true 3D surface plus matched nominal reduced state; no ranking inference",
        "full3d": {
            "status": "PASS" if first.field.size and np.isfinite(first.field).all() and deterministic else "FAIL",
            "artifact_manifest": str(surface_manifest),
            "optix_artifact": str(full_artifact_path),
            "geometry_mode": geometry.geometry_mode,
            "full3d_surface_provenance": geometry.metadata.get("full3d_surface_provenance"),
            "reference_periodic_z_planes_mm": geometry.metadata.get("reference_periodic_z_planes_mm"),
            "deformed_surface_z_extent_mm": geometry.metadata.get("deformed_surface_z_extent_mm"),
            "deformed_surface_exceeds_reference_z_planes": geometry.metadata.get(
                "deformed_surface_exceeds_reference_z_planes"
            ),
            "native_field_shape": list(first.field.shape),
            "native_field_dimension": first.field.ndim,
            "total_transport": first.total_transport,
            "energy_balance_error": first.energy_balance_error,
            "deterministic_repeated_run": deterministic,
            "finite_totals": bool(np.isfinite([first.total_transport, first.energy_balance_error]).all()),
        },
        "matched_nominal_reduced": {
            "status": "PASS" if planar_result.field.size and np.isfinite(planar_result.field).all() else "FAIL",
            "planar_mechanics_source": str(planar_summary),
            "optical_mode": planar_result.optical_mode,
            "native_field_shape": list(planar_result.field.shape),
            "total_transport": planar_result.total_transport,
            "energy_balance_error": planar_result.energy_balance_error,
            "matched_physical_parameters": {
                "contact_x_mm": CONTACT_X_MM["left"],
                "initial_gap_mm": INITIAL_GAP_MM,
                "total_prescribed_travel_mm": TRAVEL_MM,
                "indenter_radius_mm": INDENTER_RADIUS_MM,
            },
            "comparison_scope": "mode/provenance smoke only; no cross-dimensional field or ranking claim",
        },
        "native_contract": {
            "morphology_fingerprint": artifact.morphology_fingerprint,
            "contact_state_fingerprint": artifact.contact_state_fingerprint,
            "mesh_fingerprint": artifact.mesh_fingerprint,
            "mechanics_config_fingerprint": artifact.mechanics_config_fingerprint,
            "node_count": int(len(artifact.node_ids)),
            "surface_triangle_count": int(len(artifact.surface_faces_node_ids)),
            "tetrahedron_count": int(len(artifact.tetrahedra_node_ids)),
            "direct_deformed_coordinates": True,
            "semantic_surface_ids": True,
            "surface_orientation_validated": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "nominal_full3d_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def prepare_void_height_fixed_plan(output: Path = OUTPUT) -> dict[str, Any]:
    """Record the next experiment contract without launching any cases."""
    nominal = asdict(FingertipParameters())
    varied = [
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
    ]
    plan = {
        "schema": "void-height-fixed-experiment-plan-v1",
        "status": "PREPARED_NOT_RUN",
        "execution_performed": False,
        "fixed_parameters": {
            "void_height": nominal["void_height"],
            "source": "FingertipParameters()",
        },
        "first_experiment": {
            "vary_parameters": varied,
            "compare_modes": ["PLANAR_2D", "FULL_3D"],
            "morphology_ranking": "deferred_until_exact_reduced_and_full_artifacts_exist",
        },
        "later_isolated_experiment": {
            "parameter": "void_height",
            "levels": ["low", "nominal", "high"],
            "status": "PREPARED_NOT_RUN",
            "level_values": "to_be_precommitted_before_execution",
        },
        "required_provenance": [
            "morphology_fingerprint",
            "contact_state_fingerprint",
            "mesh_fingerprint",
            "mechanics_config_fingerprint",
            "native_3d_fea_state_checksum",
            "optix_transport_configuration_fingerprint",
        ],
        "no_broad_study_started": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "void_height_fixed_experiment_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-m5", action="store_true")
    parser.add_argument("--run-nominal-full3d-smoke", action="store_true")
    parser.add_argument("--prepare-void-height-plan", action="store_true")
    args = parser.parse_args()
    if args.inspect_m5:
        summary = inspect_existing_m5_artifacts()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["full3d_ready"] else 2
    if args.run_nominal_full3d_smoke:
        summary = run_nominal_full3d_smoke()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 1
    if args.prepare_void_height_plan:
        summary = prepare_void_height_fixed_plan()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    parser.error("the bounded runner supports --inspect-m5 and --run-nominal-full3d-smoke")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
