"""Focused M1--M4 evidence runner for the 3D-native migration.

This module intentionally stops at the first scientifically meaningful blocker.
It records geometry/mesh evidence and the separately executed Kratos contact
preflight result without converting a crashed contact path into a pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from mesh import generate_volume_mesh, volume_mesh_settings_for_tier
from fem.solid3d import SolidFEAError, _support_tie_pairs
from model import Fingertip, FingertipParameters, build_fingertip_solid


OUTPUT = Path("output/validation/3d_migration")
CANDIDATE49 = {
    "flat_pad_height": 3.937175708822906,
    "semielliptical_pad_height": 7.309789158403873,
    "stem_width": 7.289858109783381,
    "stem_height": 5.102298432029784,
    "void_width": 0.6931721470318735,
    "void_height": 1.2690955214202404,
}
REPRESENTATIVE = {
    "flat_pad_height": 4.5,
    "semielliptical_pad_height": 8.0,
    "stem_width": 7.0,
    "stem_height": 5.5,
    "void_width": 0.75,
    "void_height": 0.5,
}


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _designs() -> dict[str, FingertipParameters]:
    nominal = FingertipParameters()
    return {
        "nominal": nominal,
        "candidate49": replace(nominal, **CANDIDATE49),
        "representative": replace(nominal, **REPRESENTATIVE),
    }


def _m1_record(name: str, parameters: FingertipParameters) -> dict[str, Any]:
    solid = build_fingertip_solid(Fingertip(parameters).geometry)
    center_section = solid.cross_section_at(0.0)
    section_match = center_section.symmetric_difference(solid.material_geometry).area <= 1.0e-10
    return {
        "status": "PASS" if solid.watertight and section_match else "FAIL",
        "parameters": asdict(parameters),
        "morphology_fingerprint": solid.morphology_fingerprint,
        "z_bounds_mm": [solid.z_min_mm, solid.z_max_mm],
        "extrusion_depth_mm": solid.extrusion_depth_mm,
        "volume_mm3": solid.volume_mm3,
        "pad_volume_mm3": solid.pad_volume_mm3,
        "rigid_volume_mm3": solid.rigid_volume_mm3,
        "watertight_analytic_check": solid.watertight,
        "self_intersection_check": solid.material_geometry.is_valid,
        "center_section_matches_authoritative_2d": section_match,
        "semantic_surface_names": list(solid.surface_names),
    }


def _m2_record(
    name: str,
    parameters: FingertipParameters,
    tier: str,
) -> dict[str, Any]:
    solid = build_fingertip_solid(Fingertip(parameters).geometry)
    mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier(tier))
    return {
        "status": "PASS" if mesh.validation.passed else "FAIL",
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "tier": tier,
        "quality": asdict(mesh.quality),
        "validation": asdict(mesh.validation),
        "semantic_surface_names": list(mesh.semantic_surface_tags),
    }


def _m3_record(parameters: FingertipParameters) -> dict[str, Any]:
    solid = build_fingertip_solid(Fingertip(parameters).geometry)
    mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier("search"))
    try:
        tie_pairs = _support_tie_pairs(mesh)
        tie_preflight = {
            "status": "PASS",
            "pair_count": len(tie_pairs),
        }
    except SolidFEAError as exception:
        tie_preflight = {
            "status": "FAIL",
            "error": str(exception),
        }
    return {
        "status": "BLOCKED",
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "bonded_interface_preflight": tie_preflight,
        "direct_alm_contact_probe": {
            "status": "FAIL",
            "return_code": "process_abort",
            "stderr_signature": "MortarContactCondition3D3N: normal norm is zero or almost zero; Norm. normal: -nan",
            "configuration": {
                "element": "TotalLagrangianMixedVolumetricStrainElement3D4N",
                "constitutive_law": "HyperElastic3DLaw",
                "condition": "SurfaceCondition3D3N",
                "contact_process": "ALMContactProcess",
                "internal_contact": "none",
            },
        },
    }


def build_manifest() -> dict[str, Any]:
    designs = _designs()
    m1 = {name: _m1_record(name, parameters) for name, parameters in designs.items()}
    # The search tier covers all required M2 morphology examples.  The bounded
    # sensitivity comparison uses the nominal reference tier; it is not an
    # open-ended convergence campaign.
    m2_search = {
        name: _m2_record(name, parameters, "search")
        for name, parameters in designs.items()
    }
    m2_reference = {"nominal": _m2_record("nominal", designs["nominal"], "reference")}
    m1_pass = all(record["status"] == "PASS" for record in m1.values())
    m2_pass = all(record["status"] == "PASS" for record in m2_search.values()) and m2_reference["nominal"]["status"] == "PASS"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "stages": {
            "M1": {
                "status": "PASS" if m1_pass else "FAIL",
                "records": m1,
            },
            "M2": {
                "status": "PASS" if m2_pass else "FAIL",
                "search_records": m2_search,
                "reference_records": m2_reference,
                "precommitted_tiers": {
                    "search": asdict(volume_mesh_settings_for_tier("search")),
                    "reference": asdict(volume_mesh_settings_for_tier("reference")),
                },
            },
            "M3": _m3_record(designs["nominal"]),
            "M4": {
                "status": "UNCLEAR",
                "reason": "M4 cannot be evaluated until the M3 external 3D ALM contact path is stable; no 2D-equivalence claim is made.",
                "precommitted_tolerance_policy": "Do not select force/profile tolerances from a failed 3D contact result.",
            },
            "M5-M9": {
                "status": "UNCLEAR",
                "reason": "Downstream milestones are intentionally not started because M3/M4 are blocked.",
            },
        },
        "historical_evidence_preserved": True,
        "production_cleanup_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "migration_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "M1": manifest["stages"]["M1"]["status"], "M2": manifest["stages"]["M2"]["status"], "M3": manifest["stages"]["M3"]["status"]}))


if __name__ == "__main__":
    main()
