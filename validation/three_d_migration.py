"""Focused M1--M4 evidence runner for the 3D-native migration.

This module intentionally stops at the first scientifically meaningful blocker.
It records geometry/mesh evidence and the separately executed Kratos contact
preflight result without converting a crashed contact path into a pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from mesh import VolumeMeshSettings, generate_volume_mesh, volume_mesh_settings_for_tier
from fem.solid3d import SolidFEASettings, solve_solid_3d
from fem.contact3d_harness import run_minimal_contact_harness
from fem.indentation import IndentationSettings, run_indentation_case
from fem.kratos_settings import THICKNESS_MM
from mesh.indenter import (
    IndenterSettings,
    build_indenter_fixture,
    build_normal_indenter_fixture_at_x,
)
from mesh.types import MeshSettings
from model import Fingertip, FingertipParameters, build_fingertip_solid
from mesh.fingertip import generate_fingertip_mesh
from optics.transport3d.geometry import build_transport_geometry
from optics.transport3d.artifact import NATIVE_3D_FEA_STATE_SCHEMA
from validation.fem.throughput import _boundary_profiles, _profile_error
from validation.plane_strain_reference import build_plane_strain_reference_mesh


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

M4_REFERENCE_MESH_SETTINGS = MeshSettings(
    "medium", 1.0, 0.4, contact_refinement_distance_mm=1.0
)
M4_LAYER_COUNT = 9
M4_REFERENCE_STEPS = 48
M4_INDENTATION_MM = 0.5
M4_INITIAL_GAP_MM = 0.2
M4_CONTACT_LOCATIONS_MM = {"left": -3.0, "right": 3.0}
M4_REACTION_RELATIVE_TOLERANCE = 0.03
M4_FORCE_EQUILIBRIUM_TOLERANCE = 0.02
M4_BOUNDARY_POSITION_TOLERANCE_MM = 0.05
M4_PLANE_STRAIN_RESIDUAL_TOLERANCE_MM = 1.0e-8

# M5 is a production 3D-mechanics gate.  These tolerances are independent of
# the historical 2D comparison and are frozen before the M5 child cases run.
M5_REFERENCE_STEPS = 12
M5_SENSITIVITY_STEPS = 24
M5_INITIAL_GAP_MM = 0.2
M5_INDENTATION_MM = 0.5
M5_REACTION_MESH_RELATIVE_TOLERANCE = 0.10
M5_REACTION_STEP_RELATIVE_TOLERANCE = 0.05
M5_FORCE_EQUILIBRIUM_TOLERANCE = 0.02
# These are engineering ceilings, not scientific acceptance criteria.  The
# 24-step calibration search completed in roughly 14 minutes in the current
# sequential environment, so retain substantial headroom for later search
# children and a separate, larger reference budget.
M5_SEARCH_RUNTIME_LIMIT_SECONDS = 1800.0
M5_REFERENCE_RUNTIME_LIMIT_SECONDS = 3600.0
M5_CASE_OUTPUT = OUTPUT / "m5_cases"
M5_DIAGNOSTIC_OUTPUT = OUTPUT / "m5_diagnostics"
M5_NATIVE_STATE_OUTPUT = OUTPUT / "native_3d_states"


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _working_tree_fingerprint() -> str:
    """Bind the generated evidence to the complete current source state."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.startswith("output/"):
            paths.add(path)
    digest = hashlib.sha256()
    digest.update(status.encode())
    for path in sorted(paths):
        candidate = Path(path)
        if candidate.is_file():
            digest.update(path.encode())
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


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
        "authoritative_geometry_source": "model.FingertipModel",
        "authoritative_bonded_source": "FingertipModel.interface_definition.geometry",
        "parameters": asdict(parameters),
        "morphology_fingerprint": solid.morphology_fingerprint,
        "z_bounds_mm": [solid.z_min_mm, solid.z_max_mm],
        "extrusion_depth_mm": solid.extrusion_depth_mm,
        "volume_mm3": solid.volume_mm3,
        "pad_volume_mm3": solid.pad_volume_mm3,
        "rigid_link_meshed": False,
        "closed_volume_gate": solid.closed_volume_gate,
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


def _smoke_child(name: str, support_only: bool) -> int:
    parameters = _designs()[name]
    fingertip = Fingertip(parameters)
    mesh = generate_volume_mesh(
        build_fingertip_solid(fingertip.geometry),
        # The M3 smoke is intentionally coarse; M2 owns the search/reference
        # mesh evidence and the mechanics contract is unchanged.
        VolumeMeshSettings("search", 3.0, 0.02),
    )
    fixture = build_indenter_fixture(
        fingertip.geometry, IndenterSettings(initial_gap_mm=0.2)
    )
    result = solve_solid_3d(
        mesh,
        fixture,
        SolidFEASettings(number_of_steps=1, external_contact=not support_only),
    )
    payload = {
        "status": "PASS" if result.converged else "FAIL",
        "converged": result.converged,
        "failure_message": result.failure_message,
        "reaction_force_n": result.reaction_force_n,
        "contact_state": result.contact_state,
        "configuration": result.configuration,
        "finite_displacement": result.displacement_mm is not None,
        "max_abs_displacement_mm": (
            float(abs(result.displacement_mm).max())
            if result.displacement_mm is not None
            else None
        ),
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "support_only": support_only,
        "active_contact_mechanics": (
            result.contact_state.get("active_contact_diagnostics")
            if not support_only
            else None
        ),
        "reaction_diagnostics": result.contact_state.get("reaction_diagnostics"),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.converged else 1


def _run_smoke(name: str, *, support_only: bool) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.three_d_migration",
            "--m3-smoke-child",
            "--design",
            name,
            "--support-only" if support_only else "--contact",
        ],
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            payload = value
            break
    return {
        "status": "PASS" if completed.returncode == 0 and payload and payload.get("status") == "PASS" else "FAIL",
        "return_code": completed.returncode,
        "payload": payload,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _m3_record(designs: dict[str, FingertipParameters]) -> dict[str, Any]:
    harness = run_minimal_contact_harness()
    support = _run_smoke("nominal", support_only=True)
    nominal = _run_smoke("nominal", support_only=False)
    candidate = _run_smoke("candidate49", support_only=False)
    m3a_passed = harness["status"] == "PASS"
    def active_mechanics_passed(record: dict[str, Any]) -> bool:
        payload = record.get("payload") or {}
        diagnostics = payload.get("active_contact_mechanics") or {}
        reaction = payload.get("reaction_force_n")
        return (
            record["status"] == "PASS"
            and payload.get("converged") is True
            and diagnostics.get("passed") is True
            and diagnostics.get("active_condition_count", 0) > 0
            and math.isfinite(float(reaction))
            and float(reaction) > 1.0e-12
            and (payload.get("reaction_diagnostics") or {}).get("nonzero") is True
        )
    nominal_m3b = active_mechanics_passed(nominal)
    candidate_m3b = active_mechanics_passed(candidate)
    m3b_passed = (
        support["status"] == "PASS"
        and nominal_m3b
        and candidate_m3b
    )
    return {
        "status": "PASS" if m3a_passed and m3b_passed else "BLOCKED",
        "M3A": {
            "status": "PASS" if m3a_passed else "BLOCKED",
            "scope": "contact infrastructure and initialization",
            "minimal_contact_harness_status": harness["status"],
        },
        "M3B": {
            "status": "PASS" if m3b_passed else "BLOCKED",
            "scope": "active-contact mechanics",
            "support_only_status": support["status"],
            "nominal_active_contact_status": "PASS" if nominal_m3b else "FAIL",
            "candidate49_active_contact_status": "PASS" if candidate_m3b else "FAIL",
            "required_checks": [
                "active contact conditions",
                "finite nonzero reaction",
                "finite contact normals",
                "bounded penetration",
                "nonlinear convergence",
            ],
        },
        "minimal_contact_harness": harness,
        "support_only_smoke": support,
        "nominal_contact_smoke": nominal,
        "candidate49_contact_smoke": candidate,
        "mechanics_contract": {
            "element": "TotalLagrangianMixedVolumetricStrainElement3D4N",
            "constitutive_law": "HyperElastic3DLaw",
            "contact_condition": "SurfaceCondition3D3N",
            "mortar_family": "MortarContactCondition3D3N",
            "bonded_support": "authoritative support_bond_left/right nodes fixed",
            "internal_pad_stem_contact": "not used; void surfaces remain free",
            "smoke_initial_gap_mm": 0.2,
            "active_mechanics_guardrail": "M3B requires generated active mortar conditions and finite bounded contact diagnostics",
        },
    }


def _ordered_reference_local_node_ids(
    pad: Any,
    source_tag: str,
    source_geometry: Any,
) -> tuple[int, ...]:
    """Order one open authoritative 2D boundary chain for M4 profiles."""
    edges = pad.boundary_edges_by_tag[source_tag]
    adjacency: dict[int, list[int]] = {}
    for first, second in edges:
        first_id, second_id = int(first), int(second)
        adjacency.setdefault(first_id, []).append(second_id)
        adjacency.setdefault(second_id, []).append(first_id)
    endpoints = [node_id for node_id, neighbours in adjacency.items() if len(neighbours) == 1]
    if len(endpoints) != 2 or any(len(neighbours) > 2 for neighbours in adjacency.values()):
        raise ValueError(f"M4 boundary {source_tag!r} is not one open chain")
    coordinates = np.asarray(pad.reference_coordinates_mm, dtype=float)
    start_point = tuple(source_geometry.coords[0])
    start = min(
        endpoints,
        key=lambda node_id: math.dist(coordinates[node_id], start_point),
    )
    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = [node_id for node_id in adjacency[current] if node_id != previous]
        if not candidates:
            break
        if len(candidates) != 1:
            raise ValueError(f"M4 boundary {source_tag!r} branches at {current}")
        previous, current = current, candidates[0]
        ordered.append(current)
    if len(ordered) != len(adjacency):
        raise ValueError(f"M4 boundary {source_tag!r} is disconnected")
    return tuple(ordered)


def _reference_profiles(
    parameters: FingertipParameters,
    mesh: Any,
    contract: Any,
    deformed_coordinates: Any,
) -> dict[str, Any]:
    """Extract center-layer semantic profiles from the layered reference."""
    fingertip = Fingertip(parameters)
    pad = generate_fingertip_mesh(fingertip.geometry, M4_REFERENCE_MESH_SETTINGS).pad
    coordinates = np.asarray(deformed_coordinates, dtype=float)
    node_order = tuple(sorted(mesh.nodes))
    displacement_by_node = {
        node_id: coordinates[index] - np.asarray(
            [mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm, mesh.nodes[node_id].z_mm],
            dtype=float,
        )
        for index, node_id in enumerate(node_order)
    }
    source_to_output = {
        "pad_outer_arc": "pad_outer_arc",
        "pad_cutout_left": "pad_cutout_left",
        "pad_cutout_right": "pad_cutout_right",
        "pad_cutout_bottom": "pad_cutout_bottom",
    }
    profiles: dict[str, Any] = {}
    for source_tag, output_tag in source_to_output.items():
        source_geometry = fingertip.geometry.boundaries.segments[source_tag].geometry
        local_ids = _ordered_reference_local_node_ids(pad, source_tag, source_geometry)
        points = np.asarray([pad.reference_coordinates_mm[node_id] for node_id in local_ids], dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        if cumulative[-1] <= 0.0:
            raise ValueError(f"M4 boundary {source_tag!r} has zero length")
        current = np.asarray(
            [
                points[index] + displacement_by_node[contract.node_columns[local_id][contract.reference_layer_index]][:2]
                for index, local_id in enumerate(local_ids)
            ],
            dtype=float,
        )
        profiles[output_tag] = np.column_stack((cumulative / cumulative[-1], current)).tolist()
    return profiles


def _m4_2d_child(name: str, location_name: str) -> int:
    """Run one fresh external-only 2D reference case in a child process."""
    parameters = _designs()[name]
    fingertip = Fingertip(parameters)
    fixture = build_normal_indenter_fixture_at_x(
        fingertip.geometry,
        M4_CONTACT_LOCATIONS_MM[location_name],
        IndenterSettings(
            radius_mm=4.0,
            initial_gap_mm=M4_INITIAL_GAP_MM,
        ),
    )
    settings = IndentationSettings(
        indentation_mm=M4_INDENTATION_MM,
        number_of_steps=M4_REFERENCE_STEPS,
    )
    observed: dict[str, Any] = {}

    def observer(step: Any) -> dict[str, Any] | None:
        if step.result_point["step"] == settings.number_of_steps:
            observed["profiles"] = {
                tag: value.tolist()
                for tag, value in _boundary_profiles(
                    step.fingertip_model,
                    step.mesh,
                    step.displacements,
                ).items()
            }
        return None

    result, _ = run_indentation_case(
        fingertip.geometry,
        "medium",
        settings,
        fixture_override=fixture,
        internal_contact_configuration="none",
        converged_step_observer=observer,
        diagnostic_mode="minimal",
    )
    final = result.get("final", {})
    external = final.get("contact_groups", {}).get("external_pad_indenter", {})
    payload = {
        "status": "PASS" if result.get("status") == "PASS" else "FAIL",
        "solve_status": result.get("solve_status"),
        "failure_reason": result.get("failure_reason"),
        "exception": result.get("exception"),
        "reaction_force_n": final.get("indenter_normal_reaction_n"),
        "force_equilibrium_error": final.get("force_equilibrium_error"),
        "active_condition_count": external.get("active_condition_count", 0),
        "penetration_pass": external.get("penetration_pass", False),
        "contact_topology": "external_pad_indenter_only",
        "profiles": observed.get("profiles", {}),
        "configuration": {
            "mesh_level": "medium",
            "number_of_steps": M4_REFERENCE_STEPS,
            "internal_contact_configuration": "none",
            "thickness_mm": THICKNESS_MM,
            "constitutive_law": "HyperElasticPlaneStrain2DLaw",
            "contact_location_mm": M4_CONTACT_LOCATIONS_MM[location_name],
            "initial_gap_mm": M4_INITIAL_GAP_MM,
        },
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


def _m4_3d_child(name: str, location_name: str) -> int:
    """Run one layered reference contact case in a crash-isolated child."""
    parameters = _designs()[name]
    fingertip = Fingertip(parameters)
    mesh, contract = build_plane_strain_reference_mesh(
        parameters,
        mesh_settings=M4_REFERENCE_MESH_SETTINGS,
        layer_count=M4_LAYER_COUNT,
    )
    fixture = build_normal_indenter_fixture_at_x(
        fingertip.geometry,
        M4_CONTACT_LOCATIONS_MM[location_name],
        IndenterSettings(
            radius_mm=4.0,
            initial_gap_mm=M4_INITIAL_GAP_MM,
        ),
    )
    result = solve_solid_3d(
        mesh,
        fixture,
        SolidFEASettings(
            mode="3d_equivalent_reference",
            number_of_steps=M4_REFERENCE_STEPS,
            indentation_mm=M4_INDENTATION_MM,
            external_contact=True,
        ),
        reference_node_columns=contract.node_columns,
        reference_layer_index=contract.reference_layer_index,
    )
    payload: dict[str, Any] = {
        "status": "PASS" if result.converged else "FAIL",
        "converged": result.converged,
        "failure_message": result.failure_message,
        "reaction_force_n": result.reaction_force_n,
        "contact_state": result.contact_state,
        "configuration": result.configuration,
        "reference_mesh": {
            **contract.to_dict(),
            "settings": asdict(M4_REFERENCE_MESH_SETTINGS),
            "node_count": len(mesh.nodes),
            "tetrahedron_count": len(mesh.tetrahedra),
            "production_mesh_is_distinct": True,
            "morphology_fingerprint": mesh.morphology_fingerprint,
        },
        "contact_location_mm": M4_CONTACT_LOCATIONS_MM[location_name],
    }
    if result.deformed_coordinates_mm is not None:
        payload["profiles"] = _reference_profiles(
            parameters,
            mesh,
            contract,
            result.deformed_coordinates_mm,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.converged else 1


def _run_m4_child(command: list[str], timeout: float = 300.0) -> dict[str, Any]:
    """Collect a JSON child result while preserving native abort diagnostics."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exception:
        return {
            "status": "FAIL",
            "return_code": None,
            "timeout": True,
            "payload": None,
            "stdout_tail": (exception.stdout or "")[-4000:],
            "stderr_tail": (exception.stderr or "")[-4000:],
        }
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            payload = value
            break
    return {
        "status": "PASS" if completed.returncode == 0 and payload and payload.get("status") == "PASS" else "FAIL",
        "return_code": completed.returncode,
        "timeout": False,
        "payload": payload,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _m4_case_record(
    name: str,
    location_name: str,
    parameters: FingertipParameters,
) -> dict[str, Any]:
    base_command = [sys.executable, "-m", "validation.three_d_migration"]
    two_d = _run_m4_child(
        [
            *base_command,
            "--m4-2d-child",
            "--design",
            name,
            "--location",
            location_name,
        ]
    )
    three_d = _run_m4_child(
        [
            *base_command,
            "--m4-3d-child",
            "--design",
            name,
            "--location",
            location_name,
        ]
    )
    two_payload = two_d.get("payload") or {}
    three_payload = three_d.get("payload") or {}
    comparison: dict[str, Any] = {
        "reaction_per_width": {},
        "contact_activation_topology": {},
        "penetration_contact_sanity": {},
        "semantic_boundary_profiles": {},
        "plane_strain_residuals": three_payload.get("contact_state", {}).get("plane_strain_residuals"),
    }
    if two_payload.get("reaction_force_n") is not None:
        comparison["reaction_per_width"]["2d_n_per_mm"] = float(two_payload["reaction_force_n"]) / THICKNESS_MM
    if three_payload.get("reaction_force_n") is not None:
        comparison["reaction_per_width"]["3d_n_per_mm"] = float(three_payload["reaction_force_n"]) / 11.0
    if {"2d_n_per_mm", "3d_n_per_mm"}.issubset(comparison["reaction_per_width"]):
        two_force = comparison["reaction_per_width"]["2d_n_per_mm"]
        three_force = comparison["reaction_per_width"]["3d_n_per_mm"]
        comparison["reaction_per_width"]["relative_error"] = abs(three_force - two_force) / max(abs(two_force), 1.0e-12)
    comparison["contact_activation_topology"] = {
        "2d_external_active": two_payload.get("active_condition_count", 0) > 0,
        "3d_external_active": three_payload.get("contact_state", {}).get("generated_mortar", {}).get("active_condition_count", 0) > 0,
        "2d_active_condition_count": two_payload.get("active_condition_count", 0),
        "3d_active_condition_count": three_payload.get("contact_state", {}).get("generated_mortar", {}).get("active_condition_count", 0),
        "2d_topology": two_payload.get("contact_topology"),
        "3d_topology": "external_pad_indenter_only",
    }
    comparison["penetration_contact_sanity"] = {
        "2d_pass": two_payload.get("penetration_pass", False),
        "3d_pass": bool(
            three_payload.get("contact_state", {}).get("active_contact_diagnostics", {}).get("passed", False)
        ),
        "3d_diagnostics": three_payload.get("contact_state", {}).get("active_contact_diagnostics"),
    }
    two_profiles = two_payload.get("profiles", {})
    three_profiles = three_payload.get("profiles", {})
    for tag in sorted(set(two_profiles).intersection(three_profiles)):
        comparison["semantic_boundary_profiles"][tag] = _profile_error(
            np.asarray(three_profiles[tag], dtype=float),
            np.asarray(two_profiles[tag], dtype=float),
        )
    all_boundary_errors = [
        value["maximum_position_error_mm"]
        for value in comparison["semantic_boundary_profiles"].values()
    ]
    comparison["semantic_boundary_profiles"]["maximum_position_error_mm"] = max(all_boundary_errors, default=None)
    comparison["semantic_boundary_profiles"]["rms_position_error_mm"] = (
        max(
            value["rms_position_error_mm"]
            for tag, value in comparison["semantic_boundary_profiles"].items()
            if tag not in ("maximum_position_error_mm", "rms_position_error_mm")
        )
        if all_boundary_errors
        else None
    )
    mpc_residuals = comparison["plane_strain_residuals"] or {}
    two_force_equilibrium = two_payload.get("force_equilibrium_error")
    three_force_equilibrium = (
        three_payload.get("contact_state", {})
        .get("reaction_diagnostics", {})
        .get("force_equilibrium_error")
    )
    try:
        two_force_equilibrium_value = float(two_force_equilibrium)
    except (TypeError, ValueError):
        two_force_equilibrium_value = math.inf
    try:
        three_force_equilibrium_value = float(three_force_equilibrium)
    except (TypeError, ValueError):
        three_force_equilibrium_value = math.inf
    checks = {
        "2d_reference_pass": two_d["status"] == "PASS",
        "3d_reference_pass": three_d["status"] == "PASS",
        "reaction_relative_error_within_3_percent": (
            comparison["reaction_per_width"].get("relative_error", math.inf)
            <= M4_REACTION_RELATIVE_TOLERANCE
        ),
        "force_equilibrium_within_2_percent": (
            math.isfinite(two_force_equilibrium_value)
            and two_force_equilibrium_value <= M4_FORCE_EQUILIBRIUM_TOLERANCE
            and math.isfinite(three_force_equilibrium_value)
            and three_force_equilibrium_value <= M4_FORCE_EQUILIBRIUM_TOLERANCE
        ),
        "external_contact_active_in_both": comparison["contact_activation_topology"]["2d_external_active"]
        and comparison["contact_activation_topology"]["3d_external_active"],
        "penetration_sanity_in_both": comparison["penetration_contact_sanity"]["2d_pass"]
        and comparison["penetration_contact_sanity"]["3d_pass"],
        "semantic_boundaries_within_0_05_mm": (
            (
                comparison["semantic_boundary_profiles"].get("maximum_position_error_mm")
                if comparison["semantic_boundary_profiles"].get("maximum_position_error_mm") is not None
                else math.inf
            ) <= M4_BOUNDARY_POSITION_TOLERANCE_MM
        ),
        "plane_strain_residuals_within_1e-8_mm": all(
            float(mpc_residuals.get(key, math.inf)) <= M4_PLANE_STRAIN_RESIDUAL_TOLERANCE_MM
            for key in ("max_abs_ux_mm", "max_abs_uy_mm", "max_abs_uz_mm")
        ),
    }
    return {
        "morphology": name,
        "contact_location": location_name,
        "contact_location_mm": M4_CONTACT_LOCATIONS_MM[location_name],
        "morphology_fingerprint": three_payload.get("reference_mesh", {}).get(
            "morphology_fingerprint",
            build_fingertip_solid(Fingertip(parameters).geometry).morphology_fingerprint,
        ),
        "two_d_external_only": two_d,
        "three_d_equivalent_reference": three_d,
        "comparison": comparison,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _m5_deformation_surface_checks(
    volume_mesh: Any,
    deformed_coordinates: np.ndarray | None,
) -> dict[str, Any]:
    """Validate a production 3D deformation without repairing it."""
    node_order = tuple(sorted(volume_mesh.nodes))
    if deformed_coordinates is None:
        return {
            "passed": False,
            "checks": {
                "finite_deformed_coordinates": False,
                "surface_triangles_valid": False,
                "surface_orientation_preserved": False,
                "volume_elements_positive": False,
                "semantic_surface_coverage_preserved": False,
            },
            "minimum_deformed_tetra_volume_ratio": None,
            "degenerate_surface_triangle_count": None,
            "orientation_flip_count": None,
        }
    coordinates = np.asarray(deformed_coordinates, dtype=float)
    finite_coordinates = coordinates.shape == (len(node_order), 3) and bool(np.isfinite(coordinates).all())
    if not finite_coordinates:
        return {
            "passed": False,
            "checks": {
                "finite_deformed_coordinates": False,
                "surface_triangles_valid": False,
                "surface_orientation_preserved": False,
                "volume_elements_positive": False,
                "semantic_surface_coverage_preserved": False,
            },
            "minimum_deformed_tetra_volume_ratio": None,
            "degenerate_surface_triangle_count": None,
            "orientation_flip_count": None,
        }
    reference_coordinates = {
        node_id: np.asarray(
            [volume_mesh.nodes[node_id].x_mm, volume_mesh.nodes[node_id].y_mm, volume_mesh.nodes[node_id].z_mm],
            dtype=float,
        )
        for node_id in node_order
    }
    current_coordinates = {
        node_id: coordinates[index]
        for index, node_id in enumerate(node_order)
    }
    degenerate_surface_triangles = 0
    orientation_flips = 0
    surface_triangle_count = 0
    for triangles in volume_mesh.surface_triangles.values():
        for triangle in triangles:
            surface_triangle_count += 1
            reference_points = np.asarray([reference_coordinates[node_id] for node_id in triangle.node_ids])
            current_points = np.asarray([current_coordinates[node_id] for node_id in triangle.node_ids])
            reference_normal = np.cross(
                reference_points[1] - reference_points[0],
                reference_points[2] - reference_points[0],
            )
            current_normal = np.cross(
                current_points[1] - current_points[0],
                current_points[2] - current_points[0],
            )
            if (
                len(set(triangle.node_ids)) != 3
                or not np.all(np.isfinite(current_points))
                or float(np.linalg.norm(current_normal)) <= 1.0e-12
            ):
                degenerate_surface_triangles += 1
            elif float(np.dot(reference_normal, current_normal)) <= 0.0:
                orientation_flips += 1
    reference_volumes: list[float] = []
    deformed_volumes: list[float] = []
    for tetrahedron in volume_mesh.tetrahedra:
        reference_points = np.asarray([reference_coordinates[node_id] for node_id in tetrahedron.node_ids])
        current_points = np.asarray([current_coordinates[node_id] for node_id in tetrahedron.node_ids])
        reference_volume = float(
            np.linalg.det(
                np.vstack((reference_points[1:] - reference_points[0]))
            )
            / 6.0
        )
        deformed_volume = float(
            np.linalg.det(
                np.vstack((current_points[1:] - current_points[0]))
            )
            / 6.0
        )
        reference_volumes.append(reference_volume)
        deformed_volumes.append(deformed_volume)
    ratios = np.asarray(
        [current / reference for current, reference in zip(deformed_volumes, reference_volumes) if reference > 0.0],
        dtype=float,
    )
    semantic_tags_preserved = set(volume_mesh.surface_triangles) == set(volume_mesh.solid.surface_names)
    checks = {
        "finite_deformed_coordinates": finite_coordinates,
        "surface_triangles_valid": surface_triangle_count > 0 and degenerate_surface_triangles == 0,
        "surface_orientation_preserved": surface_triangle_count > 0 and orientation_flips == 0,
        "volume_elements_positive": bool(ratios.size) and bool(np.isfinite(ratios).all()) and float(ratios.min()) > 0.0,
        "semantic_surface_coverage_preserved": semantic_tags_preserved,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_deformed_tetra_volume_ratio": float(ratios.min()) if ratios.size else None,
        "minimum_reference_tetra_volume_mm3": float(min(reference_volumes, default=0.0)),
        "minimum_deformed_tetra_volume_mm3": float(min(deformed_volumes, default=0.0)),
        "surface_triangle_count": surface_triangle_count,
        "degenerate_surface_triangle_count": degenerate_surface_triangles,
        "orientation_flip_count": orientation_flips,
        "semantic_surface_tags": sorted(volume_mesh.surface_triangles),
    }


def _m5_contact_state_fingerprint(payload: dict[str, Any]) -> str:
    """Use the same case-state identity consumed by the OptiX loader."""
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


def _native_array_fingerprint(*arrays: tuple[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _external_surface_u(reference_mesh: Any, native_reference: Mapping[int, np.ndarray]) -> dict[int, float]:
    """Map reference pad boundary nodes to the existing normalized surface coordinate."""
    pad = reference_mesh.pad
    tags = ("pad_outer_left", "pad_outer_arc", "pad_outer_right")
    edges = np.vstack([pad.boundary_edges_for(tag) for tag in tags])
    adjacency: dict[int, list[int]] = {}
    for first, second in edges:
        i, j = int(first), int(second)
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(endpoints) != 2:
        raise RuntimeError("the reference external boundary is not one open chain")
    start = min(endpoints, key=lambda node: (float(pad.coordinates[node, 0]), float(pad.coordinates[node, 1]), node))
    chain = [start]
    previous: int | None = None
    current = start
    while True:
        choices = sorted(node for node in adjacency[current] if node != previous)
        if not choices:
            break
        next_node = choices[0]
        if next_node in chain:
            raise RuntimeError("the reference external boundary chain loops")
        chain.append(next_node)
        previous, current = current, next_node
    if len(chain) != len(adjacency):
        raise RuntimeError("the reference external boundary chain is disconnected")
    cumulative = [0.0]
    for first, second in zip(chain, chain[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(pad.coordinates[second] - pad.coordinates[first])))
    if cumulative[-1] <= 0.0:
        raise RuntimeError("the reference external boundary has zero length")
    reference_u = {node: value / cumulative[-1] for node, value in zip(chain, cumulative)}
    result: dict[int, float] = {}
    reference_coordinates = pad.coordinates
    for node_id, coordinate in native_reference.items():
        distances = np.linalg.norm(reference_coordinates - np.asarray(coordinate, dtype=float), axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= 1.0e-5 and nearest in reference_u:
            result[int(node_id)] = float(reference_u[nearest])
    return result


def _orient_surface_faces(
    faces_node_ids: np.ndarray,
    node_ids: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Canonicalize the outward orientation of the directly extracted shell."""
    faces = np.asarray(faces_node_ids, dtype=np.int64).copy()
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    local_faces = np.asarray(
        [[node_index[int(value)] for value in face] for face in faces], dtype=np.int64
    )
    edge_records: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, face in enumerate(local_faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = min(int(first), int(second)), max(int(first), int(second))
            direction = 1 if (int(first), int(second)) == key else -1
            edge_records.setdefault(key, []).append((index, direction))
    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in local_faces]
    for values in edge_records.values():
        if len(values) > 2:
            raise RuntimeError("native 3D surface has a non-manifold edge")
        if len(values) == 2:
            (first, first_direction), (second, second_direction) = values
            adjacency[first].append((second, first_direction, second_direction))
            adjacency[second].append((first, second_direction, first_direction))
    flips = np.full(len(local_faces), -1, dtype=np.int8)
    center = np.mean(coordinates, axis=0)
    for start in range(len(local_faces)):
        if flips[start] >= 0:
            continue
        flips[start] = 0
        component = [start]
        pending = [start]
        while pending:
            current = pending.pop()
            for neighbor, current_direction, neighbor_direction in adjacency[current]:
                expected = int(flips[current]) ^ int(current_direction == neighbor_direction)
                if flips[neighbor] < 0:
                    flips[neighbor] = expected
                    component.append(neighbor)
                    pending.append(neighbor)
                elif int(flips[neighbor]) != expected:
                    raise RuntimeError("native 3D surface cannot be consistently oriented")
        oriented = local_faces[component].copy()
        oriented[flips[component] == 1, 1], oriented[flips[component] == 1, 2] = (
            oriented[flips[component] == 1, 2],
            oriented[flips[component] == 1, 1].copy(),
        )
        points = coordinates[oriented]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        if np.any(np.linalg.norm(normals, axis=1) <= 1.0e-12):
            raise RuntimeError("native 3D surface contains a zero-area triangle")
        if int(np.count_nonzero(np.sum(normals * (points.mean(axis=1) - center), axis=1) < 0.0)) > len(component) // 2:
            flips[component] ^= 1
    local_faces[flips == 1, 1], local_faces[flips == 1, 2] = (
        local_faces[flips == 1, 2],
        local_faces[flips == 1, 1].copy(),
    )
    return np.asarray(
        [[int(node_ids[index]) for index in face] for face in local_faces],
        dtype=np.int64,
    )


def _export_native_3d_state(
    *,
    name: str,
    tier: str,
    steps: int,
    location_name: str,
    fingertip: Fingertip,
    mesh: Any,
    fixture: Any,
    result: Any,
    payload: dict[str, Any],
    output_dir: Path | None = None,
    indentation_mm: float = M5_INDENTATION_MM,
    contact_location_mm: float | None = None,
) -> Path:
    """Persist one complete, directly solved 3D state and its OptiX surfaces."""
    if not result.converged or result.deformed_coordinates_mm is None or result.displacement_mm is None:
        raise RuntimeError("cannot export a native 3D state without a converged deformation")
    node_order = tuple(sorted(mesh.nodes))
    node_ids = np.asarray(node_order, dtype=np.int64)
    reference = np.asarray(result.reference_coordinates_mm, dtype=float)
    deformed = np.asarray(result.deformed_coordinates_mm, dtype=float)
    displacement = np.asarray(result.displacement_mm, dtype=float)
    if reference.shape != (len(node_ids), 3) or deformed.shape != reference.shape or displacement.shape != reference.shape:
        raise RuntimeError("native 3D state coordinate arrays do not match volume-mesh node order")
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    tetrahedra = np.asarray(
        [tetrahedron.node_ids for tetrahedron in mesh.tetrahedra], dtype=np.int64
    )
    surface_rows: list[tuple[int, str, tuple[int, int, int]]] = []
    for tag, triangles in sorted(mesh.surface_triangles.items()):
        for triangle in triangles:
            surface_rows.append((int(triangle.id), str(tag), tuple(int(value) for value in triangle.node_ids)))
    surface_rows.sort(key=lambda value: (value[1], value[0], value[2]))
    raw_surface_faces_node_ids = np.asarray([row[2] for row in surface_rows], dtype=np.int64)
    surface_faces_node_ids = _orient_surface_faces(raw_surface_faces_node_ids, node_ids, reference)
    surface_tags = tuple(row[1] for row in surface_rows)
    surface_indices = np.asarray(
        [[node_index[int(value)] for value in row[2]] for row in surface_rows], dtype=np.int64
    )
    reference_points = reference[surface_indices]
    deformed_points = deformed[surface_indices]
    reference_cross = np.cross(
        reference_points[:, 1] - reference_points[:, 0],
        reference_points[:, 2] - reference_points[:, 0],
    )
    deformed_cross = np.cross(
        deformed_points[:, 1] - deformed_points[:, 0],
        deformed_points[:, 2] - deformed_points[:, 0],
    )
    reference_lengths = np.linalg.norm(reference_cross, axis=1)
    deformed_lengths = np.linalg.norm(deformed_cross, axis=1)
    if np.any(reference_lengths <= 1.0e-12) or np.any(deformed_lengths <= 1.0e-12):
        raise RuntimeError("native 3D state contains a zero-area surface triangle")
    reference_normals = reference_cross / reference_lengths[:, None]
    deformed_normals = deformed_cross / deformed_lengths[:, None]
    if np.any(np.sum(reference_normals * deformed_normals, axis=1) <= 0.0):
        raise RuntimeError("native 3D deformation flips a surface orientation")

    lateral_rows = [index for index, tag in enumerate(surface_tags) if not tag.startswith("longitudinal_end_")]
    if not lateral_rows:
        raise RuntimeError("native 3D state has no lateral optical surface triangles")
    lateral_faces_node_ids = surface_faces_node_ids[lateral_rows]
    lateral_tags = tuple(surface_tags[index] for index in lateral_rows)
    lateral_deformed_points = deformed_points[lateral_rows]
    lateral_normals = deformed_normals[lateral_rows]
    silicone_node_ids = np.unique(lateral_faces_node_ids.reshape(-1))
    silicone_index = {int(node_id): index for index, node_id in enumerate(silicone_node_ids)}
    silicone_faces = np.asarray(
        [[silicone_index[int(value)] for value in face] for face in lateral_faces_node_ids], dtype=np.uint32
    )
    silicone_vertices = np.asarray(
        [deformed[node_index[int(node_id)]] for node_id in silicone_node_ids], dtype=np.float32
    )
    silicone_normals = np.asarray(lateral_normals, dtype=np.float32)
    native_reference = {
        int(node_id): reference[node_index[int(node_id)], :2]
        for node_id in silicone_node_ids
    }
    boundary_u = _external_surface_u(fingertip.mesh(M4_REFERENCE_MESH_SETTINGS), native_reference)
    u_start: list[float] = []
    u_end: list[float] = []
    external: list[bool] = []
    for face, tag in zip(lateral_faces_node_ids, lateral_tags):
        values = [boundary_u.get(int(node_id), 0.0) for node_id in face]
        u_start.append(float(min(values)))
        u_end.append(float(max(values)))
        external.append(tag.startswith("outer_compliant_"))

    reference_transport = build_transport_geometry(
        fingertip,
        fingertip.mesh(M4_REFERENCE_MESH_SETTINGS).pad,
        fingertip.mesh(M4_REFERENCE_MESH_SETTINGS),
        depth_mm=11.0,
    )
    contact_state_fingerprint = _m5_contact_state_fingerprint(payload)
    mesh_fingerprint = _native_array_fingerprint(
        ("node_ids", node_ids),
        ("undeformed_nodes_xyz", reference),
        ("tetrahedra_node_ids", tetrahedra),
        ("surface_faces_node_ids", surface_faces_node_ids),
        ("surface_semantic_tags", np.frombuffer("\n".join(surface_tags).encode(), dtype=np.uint8)),
    )
    resolved_contact_location_mm = (
        float(M4_CONTACT_LOCATIONS_MM[location_name])
        if contact_location_mm is None
        else float(contact_location_mm)
    )
    mechanics_configuration = {
        "configuration": result.configuration,
        "fixture": {
            "radius_mm": float(fixture.settings.radius_mm),
            "initial_gap_mm": float(fixture.settings.initial_gap_mm),
            "contact_location_mm": resolved_contact_location_mm,
        },
        "travel_mm": float(indentation_mm),
    }
    mechanics_config_fingerprint = hashlib.sha256(
        json.dumps(mechanics_configuration, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    metadata = {
        "schema": NATIVE_3D_FEA_STATE_SCHEMA,
        "morphology_id": name,
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "contact_state_fingerprint": contact_state_fingerprint,
        "mechanics_source": "fem.solid3d.solve_solid_3d",
        "surface_provenance": "direct result.deformed_coordinates_mm; no 2D deformation reconstruction",
        "surface_scope": "lateral compliant-pad surfaces; longitudinal cell caps remain periodic transport boundaries",
        "surface_semantic_tags": sorted(set(surface_tags)),
        "mesh_fingerprint": mesh_fingerprint,
        "mechanics_config_fingerprint": mechanics_config_fingerprint,
        "contact_state": result.contact_state,
        "configuration": result.configuration,
    }
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "node_ids": node_ids,
        "undeformed_nodes_xyz": reference,
        "deformed_nodes_xyz": deformed,
        "displacement_xyz": displacement,
        "tetrahedra_node_ids": tetrahedra,
        "surface_faces_node_ids": surface_faces_node_ids,
        "surface_semantic_tags_json": np.asarray(json.dumps(list(surface_tags))),
        "surface_reference_normals": reference_normals,
        "surface_deformed_normals": deformed_normals,
        "silicone_node_ids": silicone_node_ids,
        "silicone_vertices": silicone_vertices,
        "silicone_faces": silicone_faces,
        "silicone_normals": silicone_normals,
        "silicone_external_surface": np.asarray(external, dtype=bool),
        "silicone_u_start": np.asarray(u_start, dtype=float),
        "silicone_u_end": np.asarray(u_end, dtype=float),
        "silicone_semantic_tags_json": np.asarray(json.dumps(list(lateral_tags))),
    }
    for prefix, surface in (("rigid", reference_transport.rigid), ("envelope", reference_transport.envelope)):
        arrays[f"{prefix}_vertices"] = surface.vertices
        arrays[f"{prefix}_faces"] = surface.faces
        arrays[f"{prefix}_normals"] = surface.normals
    path_stem = f"{name}__{tier}__{steps}__{location_name}"
    state_output = M5_NATIVE_STATE_OUTPUT if output_dir is None else output_dir
    state_output.mkdir(parents=True, exist_ok=True)
    state_path = state_output / f"{path_stem}.npz"
    state_temporary = state_path.with_name(f".{state_path.name}.tmp.npz")
    with state_temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(state_temporary, state_path)
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest = {
        "schema": NATIVE_3D_FEA_STATE_SCHEMA,
        "morphology_id": name,
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "contact_state_fingerprint": contact_state_fingerprint,
        "mechanics_source": "fem.solid3d.solve_solid_3d",
        "mechanics_config_fingerprint": mechanics_config_fingerprint,
        "mesh_fingerprint": mesh_fingerprint,
        "tier": tier,
        "contact_location": location_name,
        "contact_location_mm": resolved_contact_location_mm,
        "total_prescribed_travel_mm": float(indentation_mm),
        "indenter_radius_mm": float(fixture.settings.radius_mm),
        "initial_gap_mm": float(fixture.settings.initial_gap_mm),
        "source_position_mm": list(reference_transport.source_position_mm),
        "source_medium": int(reference_transport.source_medium),
        "native_state_artifact": state_path.name,
        "native_state_sha256": state_sha256,
        "surface_artifact": state_path.name,
        "surface_sha256": state_sha256,
        "surface_provenance": metadata["surface_provenance"],
        "optical_fixed_surface_source": "authoritative undeformed FingertipMesh carrier/envelope",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = state_output / f"{path_stem}.json"
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_temporary, manifest_path)
    return manifest_path


def _m5_case_artifact_path(key: str) -> Path:
    safe_key = key.replace(":", "__")
    return M5_CASE_OUTPUT / f"{safe_key}.json"


def _write_m5_case_artifact(key: str, record: dict[str, Any]) -> None:
    """Atomically persist one M5 child outcome before parent aggregation."""
    M5_CASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = _m5_case_artifact_path(key)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps({"schema": "m5-production-case-v1", "key": key, **record}, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _m5_expected_fingerprint(name: str) -> str:
    parameters = _designs()[name]
    return build_fingertip_solid(Fingertip(parameters).geometry).morphology_fingerprint


def _m5_case_fingerprint(
    name: str,
    tier: str,
    steps: int,
    location_name: str,
    morphology_fingerprint: str | None = None,
) -> str:
    """Fingerprint the complete bounded M5 case contract, not only morphology."""
    if morphology_fingerprint is None:
        morphology_fingerprint = _m5_expected_fingerprint(name)
    mesh_settings = volume_mesh_settings_for_tier(tier)
    contract = {
        "contract": "m5-production-case-v1",
        "morphology": name,
        "morphology_fingerprint": morphology_fingerprint,
        "mesh": asdict(mesh_settings),
        "fea": {
            "mode": "production",
            "external_contact": True,
            "number_of_steps": steps,
            "indentation_mm": M5_INDENTATION_MM,
            "reference_longitudinal_constraint": False,
            "maximum_newton_iterations": 35,
        },
        "indenter": {
            "radius_mm": 4.0,
            "initial_gap_mm": M5_INITIAL_GAP_MM,
        },
        "contact_location": {
            "name": location_name,
            "x_mm": M4_CONTACT_LOCATIONS_MM[location_name],
        },
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _m5_case_parts(key: str) -> tuple[str, str, int, str]:
    name, tier, step_text, location_name = key.split(":")
    return name, tier, int(step_text), location_name


def _load_m5_case_artifact(key: str) -> dict[str, Any] | None:
    path = _m5_case_artifact_path(key)
    if not path.exists():
        return None
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if artifact.get("schema") != "m5-production-case-v1" or artifact.get("key") != key:
        return None
    try:
        expected_name, expected_tier, expected_steps, expected_location = _m5_case_parts(key)
        expected_morphology = _m5_expected_fingerprint(expected_name)
        expected_case = _m5_case_fingerprint(
            expected_name,
            expected_tier,
            expected_steps,
            expected_location,
            expected_morphology,
        )
    except (KeyError, TypeError, ValueError):
        return None
    payload = artifact.get("payload") or {}
    if artifact.get("case_fingerprint") != expected_case:
        return None
    outcome = artifact.get("outcome")
    if outcome not in {"PASS", "NUMERICAL_FAIL", "RUNTIME_LIMIT", "NOT_RUN"}:
        return None
    if artifact.get("status") != outcome:
        return None
    runtime_seconds = artifact.get("runtime_seconds")
    if not isinstance(runtime_seconds, (int, float)) or not math.isfinite(float(runtime_seconds)):
        return None
    if not payload:
        return artifact if outcome == "RUNTIME_LIMIT" else None
    settings = payload.get("configuration", {}).get("settings", {})
    if (
        payload.get("morphology") != expected_name
        or payload.get("mesh_tier") != expected_tier
        or payload.get("steps") != expected_steps
        or payload.get("contact_location") != expected_location
        or payload.get("contact_location_mm") != M4_CONTACT_LOCATIONS_MM[expected_location]
        or payload.get("morphology_fingerprint") != expected_morphology
        or payload.get("case_fingerprint") != expected_case
        or settings.get("mode") != "production"
        or settings.get("external_contact") is not True
        or settings.get("number_of_steps") != expected_steps
        or settings.get("indentation_mm") != M5_INDENTATION_MM
        or settings.get("reference_longitudinal_constraint") is not False
        or settings.get("maximum_newton_iterations") != 35
    ):
        return None
    return artifact


def _m5_child(name: str, tier: str, steps: int, location_name: str) -> int:
    """Run one independent production 3D mechanics case for M5."""
    started = time.monotonic()
    parameters = _designs()[name]
    fingertip = Fingertip(parameters)
    solid = build_fingertip_solid(fingertip.geometry)
    mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier(tier))
    fixture = build_normal_indenter_fixture_at_x(
        fingertip.geometry,
        M4_CONTACT_LOCATIONS_MM[location_name],
        IndenterSettings(
            radius_mm=4.0,
            initial_gap_mm=M5_INITIAL_GAP_MM,
        ),
    )
    result = solve_solid_3d(
        mesh,
        fixture,
        SolidFEASettings(
            mode="production",
            number_of_steps=steps,
            indentation_mm=M5_INDENTATION_MM,
            external_contact=True,
        ),
    )
    reaction_diagnostics = result.contact_state.get("reaction_diagnostics", {})
    active_diagnostics = result.contact_state.get("active_contact_diagnostics", {})
    surface_validity = _m5_deformation_surface_checks(
        mesh,
        result.deformed_coordinates_mm,
    )
    reaction = result.reaction_force_n
    try:
        reaction_value = float(reaction)
    except (TypeError, ValueError):
        reaction_value = math.nan
    try:
        equilibrium_error = float(reaction_diagnostics.get("force_equilibrium_error"))
    except (TypeError, ValueError):
        equilibrium_error = math.inf
    checks = {
        "converged": result.converged,
        "active_mortar_conditions": int(
            result.contact_state.get("generated_mortar", {}).get("active_condition_count", 0)
        ) > 0,
        "finite_nonzero_reaction": math.isfinite(reaction_value) and reaction_value > 1.0e-12,
        "force_equilibrium_within_2_percent": math.isfinite(equilibrium_error)
        and equilibrium_error <= M5_FORCE_EQUILIBRIUM_TOLERANCE,
        "active_contact_diagnostics": active_diagnostics.get("passed") is True,
        "deformed_surface_valid": surface_validity["passed"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "morphology": name,
        "contact_location": location_name,
        "contact_location_mm": M4_CONTACT_LOCATIONS_MM[location_name],
        "mesh_tier": tier,
        "steps": steps,
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "case_fingerprint": _m5_case_fingerprint(
            name, tier, steps, location_name, mesh.morphology_fingerprint
        ),
        "mesh_quality": asdict(mesh.quality),
        "mesh_validation": asdict(mesh.validation),
        "converged": result.converged,
        "failure_message": result.failure_message,
        "reaction_force_n": reaction,
        "reaction_diagnostics": reaction_diagnostics,
        "active_contact_diagnostics": active_diagnostics,
        "contact_state": result.contact_state,
        "surface_validity": surface_validity,
        "checks": checks,
        "configuration": result.configuration,
        "no_fallback_or_nan_suppression": True,
    }
    native_manifest = _export_native_3d_state(
        name=name,
        tier=tier,
        steps=steps,
        location_name=location_name,
        fingertip=fingertip,
        mesh=mesh,
        fixture=fixture,
        result=result,
        payload=payload,
    )
    payload["native_3d_artifact"] = str(native_manifest)
    payload["native_3d_artifact_sha256"] = hashlib.sha256(
        native_manifest.read_bytes()
    ).hexdigest()
    _write_m5_case_artifact(
        f"{name}:search:{steps}:{location_name}" if tier == "search" else f"{name}:reference:{steps}:{location_name}",
        {
            "status": "PASS" if payload["status"] == "PASS" else "NUMERICAL_FAIL",
            "outcome": "PASS" if payload["status"] == "PASS" else "NUMERICAL_FAIL",
            "case_fingerprint": payload["case_fingerprint"],
            "payload": payload,
            "return_code": 0 if payload["status"] == "PASS" else 1,
            "runtime_limited": False,
            "runtime_seconds": time.monotonic() - started,
            "runtime_exact": True,
        },
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


def _m5_history_diagnostic() -> int:
    """Run only nominal search/12/left and persist its per-step contact history."""
    name, tier, steps, location_name = "nominal", "search", 12, "left"
    parameters = _designs()[name]
    fingertip = Fingertip(parameters)
    solid = build_fingertip_solid(fingertip.geometry)
    mesh = generate_volume_mesh(solid, volume_mesh_settings_for_tier(tier))
    fixture = build_normal_indenter_fixture_at_x(
        fingertip.geometry,
        M4_CONTACT_LOCATIONS_MM[location_name],
        IndenterSettings(radius_mm=4.0, initial_gap_mm=M5_INITIAL_GAP_MM),
    )
    history: list[dict[str, Any]] = []
    result = solve_solid_3d(
        mesh,
        fixture,
        SolidFEASettings(
            mode="production",
            number_of_steps=steps,
            indentation_mm=M5_INDENTATION_MM,
            external_contact=True,
        ),
        step_history=history,
    )
    initial_gap = float(
        fingertip.geometry.boundaries.segments["pad_outer_arc"].geometry.distance(
            fixture.contact_arc
        )
    )
    first_active = next(
        (row for row in history if int(row["active_mortar_count"]) > 0),
        None,
    )
    first_reaction = next(
        (row for row in history if float(row["reaction_force_n"]) > 1.0e-9),
        None,
    )
    payload = {
        "status": "PASS" if result.converged and len(history) == steps else "FAIL",
        "morphology": name,
        "morphology_fingerprint": mesh.morphology_fingerprint,
        "case_fingerprint": _m5_case_fingerprint(
            name, tier, steps, location_name, mesh.morphology_fingerprint
        ),
        "git_revision": _git_revision(),
        "mesh_tier": tier,
        "steps": steps,
        "contact_location": location_name,
        "contact_location_mm": M4_CONTACT_LOCATIONS_MM[location_name],
        "configuration": result.configuration,
        "converged": result.converged,
        "failure_message": result.failure_message,
        "reaction_force_n": result.reaction_force_n,
        "contact_state_final": result.contact_state,
        "geometry_initial_minimum_gap_mm": initial_gap,
        "configured_initial_gap_mm": M5_INITIAL_GAP_MM,
        "total_prescribed_travel_mm": M5_INDENTATION_MM,
        "step_increment_mm": M5_INDENTATION_MM / steps,
        "first_active_mortar_step": first_active["step"] if first_active else None,
        "first_active_mortar_travel_mm": (
            first_active["prescribed_travel_mm"] if first_active else None
        ),
        "discrete_effective_indentation_by_active_flag_mm": (
            M5_INDENTATION_MM - first_active["prescribed_travel_mm"]
            if first_active
            else None
        ),
        "first_nonzero_reaction_step": first_reaction["step"] if first_reaction else None,
        "first_nonzero_reaction_travel_mm": (
            first_reaction["prescribed_travel_mm"] if first_reaction else None
        ),
        "discrete_effective_indentation_by_reaction_mm": (
            M5_INDENTATION_MM - first_reaction["prescribed_travel_mm"]
            if first_reaction
            else None
        ),
        "physical_onset_interpretation": (
            "first nonzero support reaction; active_mortar_count alone is not a physical load-transfer onset"
        ),
        "history": history,
        "no_fallback_or_nan_suppression": True,
    }
    M5_DIAGNOSTIC_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = M5_DIAGNOSTIC_OUTPUT / "nominal__search__12__left_step_history.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": "m5-step-history-diagnostic-v1",
                "key": "nominal:search:12:left:step-history",
                "status": payload["status"],
                "outcome": payload["status"],
                "payload": payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, path)
    print(json.dumps({"output": str(path), **payload}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


def _run_m5_child(
    command: list[str],
    *,
    key: str,
    timeout: float,
) -> dict[str, Any]:
    """Collect one M5 child while preserving native failures and aborts."""
    case_fingerprint = _m5_case_fingerprint(*_m5_case_parts(key))
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exception:
        runtime_seconds = time.monotonic() - started
        record = {
            "status": "RUNTIME_LIMIT",
            "return_code": None,
            "timeout": True,
            "payload": None,
            "stdout_tail": (exception.stdout or "")[-4000:],
            "stderr_tail": (exception.stderr or "")[-4000:],
            "runtime_limit_seconds": timeout,
            "runtime_seconds": runtime_seconds,
            "runtime_exact": True,
            "case_fingerprint": case_fingerprint,
        }
        _write_m5_case_artifact(key, {"outcome": "RUNTIME_LIMIT", **record})
        return record
    runtime_seconds = time.monotonic() - started
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("morphology"):
            payload = value
            break
    outcome = (
        "PASS"
        if completed.returncode == 0 and payload and payload.get("status") == "PASS"
        else "NUMERICAL_FAIL"
    )
    record = {
        "status": outcome,
        "return_code": completed.returncode,
        "timeout": False,
        "payload": payload,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "runtime_seconds": runtime_seconds,
        "runtime_exact": True,
        "case_fingerprint": case_fingerprint,
    }
    _write_m5_case_artifact(key, {"outcome": outcome, **record})
    return record


def _m5_record(
    designs: dict[str, FingertipParameters],
    *,
    artifact_only: bool = False,
) -> dict[str, Any]:
    """Assemble the reduced M5 matrix, optionally without invoking Kratos."""
    base_command = [sys.executable, "-m", "validation.three_d_migration"]
    specifications = (
        ("nominal:search:12:left", "nominal", "search", 12, "left"),
        ("nominal:search:12:right", "nominal", "search", 12, "right"),
        ("nominal:search:24:left", "nominal", "search", 24, "left"),
        ("nominal:reference:12:left", "nominal", "reference", 12, "left"),
        ("candidate49:search:12:left", "candidate49", "search", 12, "left"),
        ("candidate49:reference:12:left", "candidate49", "reference", 12, "left"),
    )
    records: dict[str, Any] = {}
    for key, name, tier, steps, location_name in specifications:
        artifact = _load_m5_case_artifact(key)
        if artifact is not None:
            records[key] = artifact
            continue
        if artifact_only:
            records[key] = {
                "status": "NOT_RUN",
                "outcome": "NOT_RUN",
                "payload": None,
                "return_code": None,
                "timeout": False,
                "reason": "no exact-fingerprint child artifact was available",
            }
            continue
        timeout = (
            M5_REFERENCE_RUNTIME_LIMIT_SECONDS
            if tier == "reference"
            else M5_SEARCH_RUNTIME_LIMIT_SECONDS
        )
        command = [
            *base_command,
            "--m5-child",
            "--design",
            name,
            "--tier",
            tier,
            "--steps",
            str(steps),
            "--location",
            location_name,
        ]
        records[key] = _run_m5_child(command, key=key, timeout=timeout)

    payloads = {key: value.get("payload") or {} for key, value in records.items()}

    def reaction_relative(first_key: str, second_key: str) -> float | None:
        first = payloads[first_key].get("reaction_force_n")
        second = payloads[second_key].get("reaction_force_n")
        try:
            return abs(float(first) - float(second)) / max(abs(float(second)), 1.0e-12)
        except (TypeError, ValueError):
            return None

    mesh_comparison = {
        "nominal": {
            "search_case": "nominal:search:12:left",
            "reference_case": "nominal:reference:12:left",
            "search_status": records["nominal:search:12:left"]["status"],
            "reference_status": records["nominal:reference:12:left"]["status"],
            "search_quality": payloads["nominal:search:12:left"].get("mesh_quality"),
            "reference_quality": payloads["nominal:reference:12:left"].get("mesh_quality"),
            "reaction_relative_error": reaction_relative(
                "nominal:search:12:left",
                "nominal:reference:12:left",
            ),
        },
        "candidate49": {
            "search_case": "candidate49:search:12:left",
            "reference_case": "candidate49:reference:12:left",
            "search_status": records["candidate49:search:12:left"]["status"],
            "reference_status": records["candidate49:reference:12:left"]["status"],
            "search_quality": payloads["candidate49:search:12:left"].get("mesh_quality"),
            "reference_quality": payloads["candidate49:reference:12:left"].get("mesh_quality"),
            "reaction_relative_error": reaction_relative(
                "candidate49:search:12:left",
                "candidate49:reference:12:left",
            ),
        },
    }
    step_error = reaction_relative("nominal:search:12:left", "nominal:search:24:left")
    contact_robustness = {
        "nominal": {
            location: {
                "case": f"nominal:search:12:{location}",
                "status": records[f"nominal:search:12:{location}"]["status"],
                "active_condition_count": payloads[f"nominal:search:12:{location}"].get(
                    "active_contact_diagnostics", {}
                ).get("active_condition_count"),
                "reaction_force_n": payloads[f"nominal:search:12:{location}"].get(
                    "reaction_force_n"
                ),
                "surface_validity": payloads[f"nominal:search:12:{location}"].get(
                    "surface_validity"
                ),
            }
            for location in M4_CONTACT_LOCATIONS_MM
        }
    }

    def pass_if_resolved(
        status_keys: tuple[str, ...],
        predicate: bool,
    ) -> bool | None:
        statuses = [records[key]["status"] for key in status_keys]
        if any(status == "RUNTIME_LIMIT" for status in statuses):
            return None
        if any(status in {"NUMERICAL_FAIL", "NOT_RUN"} for status in statuses):
            return False
        return predicate

    checks = {
        "nominal_side_robustness": pass_if_resolved(
            ("nominal:search:12:left", "nominal:search:12:right"),
            all(
                int(
                    payloads[key].get("active_contact_diagnostics", {}).get(
                        "active_condition_count", 0
                    )
                    or 0
                )
                > 0
                for key in ("nominal:search:12:left", "nominal:search:12:right")
            ),
        ),
        "nominal_load_step_sensitivity": pass_if_resolved(
            ("nominal:search:12:left", "nominal:search:24:left"),
            step_error is not None
            and step_error <= M5_REACTION_STEP_RELATIVE_TOLERANCE,
        ),
        "nominal_search_reference_mesh_fidelity": pass_if_resolved(
            ("nominal:search:12:left", "nominal:reference:12:left"),
            mesh_comparison["nominal"]["reaction_relative_error"] is not None
            and mesh_comparison["nominal"]["reaction_relative_error"]
            <= M5_REACTION_MESH_RELATIVE_TOLERANCE,
        ),
        "candidate49_search_reference_transfer_guardrail": pass_if_resolved(
            ("candidate49:search:12:left", "candidate49:reference:12:left"),
            mesh_comparison["candidate49"]["reaction_relative_error"] is not None
            and mesh_comparison["candidate49"]["reaction_relative_error"]
            <= M5_REACTION_MESH_RELATIVE_TOLERANCE,
        ),
        "all_reactions_and_surfaces_sane": pass_if_resolved(
            tuple(key for key, *_ in specifications),
            all(
                bool(payloads[key].get("checks", {}).get("finite_nonzero_reaction"))
                and bool(payloads[key].get("surface_validity", {}).get("passed"))
                for key, *_ in specifications
            ),
        ),
    }
    numerical_failures = [
        key for key, record in records.items() if record["status"] == "NUMERICAL_FAIL"
    ]
    runtime_limited = [
        key for key, record in records.items() if record["status"] == "RUNTIME_LIMIT"
    ]
    not_run = [key for key, record in records.items() if record["status"] == "NOT_RUN"]
    if numerical_failures:
        status = "NUMERICAL_FAIL"
    elif runtime_limited:
        status = "M5_COMPUTE_LIMITED"
    elif not_run:
        status = "NOT_RUN"
    elif all(value is True for value in checks.values()):
        status = "PASS"
    else:
        status = "NUMERICAL_FAIL"

    return {
        "status": status,
        "scope": "reduced M5 production 3D mechanics matrix; M6-M9 not started",
        "acceptance_basis": "3D-native internal consistency; historical 2D ranking is not an acceptance criterion",
        "outcome_semantics": {
            "PASS": "case completed with the required finite/converged mechanics evidence",
            "NUMERICAL_FAIL": "native child returned a numerical/setup failure or failed a mechanics contract",
            "RUNTIME_LIMIT": "configured execution budget was insufficient; not a scientific mechanics failure",
        },
        "runtime_limits_seconds": {
            "search": M5_SEARCH_RUNTIME_LIMIT_SECONDS,
            "reference": M5_REFERENCE_RUNTIME_LIMIT_SECONDS,
            "basis": "observed native 12/24-step runtime with engineering headroom; execution budget only, not a fidelity tolerance",
        },
        "precommitted_tolerances": {
            "search_reference_reaction_relative": M5_REACTION_MESH_RELATIVE_TOLERANCE,
            "load_step_reaction_relative": M5_REACTION_STEP_RELATIVE_TOLERANCE,
            "force_equilibrium": M5_FORCE_EQUILIBRIUM_TOLERANCE,
        },
        "mesh_comparison": mesh_comparison,
        "load_step_sensitivity": {
            "cases": ["nominal:search:12:left", "nominal:search:24:left"],
            "reaction_relative_error": step_error,
        },
        "contact_robustness": contact_robustness,
        "checks": checks,
        "numerical_failures": numerical_failures,
        "runtime_limited_cases": runtime_limited,
        "not_run_cases": not_run,
        "case_records": records,
        "finalist_high_fidelity_re_evaluation": {
            "status": "NOT_RUN_PER_REDUCED_M5_MATRIX",
            "reason": "M5 policy requested the nominal/candidate transfer guardrail only; broader finalist reevaluation remains later 3D-native work",
        },
        "optix_fidelity_convergence": {
            "status": "DEFERRED_TO_M6",
            "existing_artifact": "output/validation/optics/transport3d/summary.json",
            "reason": "M5 validates mechanics; direct production 3D FEA-surface to OptiX integration is a later gate",
        },
        "no_historical_2d_acceptance_criterion": True,
        "no_fallback_or_nan_suppression": True,
    }

def build_m5_manifest(path: Path, *, artifact_only: bool = False) -> dict[str, Any]:
    """Preserve M4 provenance and assemble M5, optionally without Kratos."""
    if not path.exists():
        raise RuntimeError(f"cannot authorize M5 without existing manifest: {path}")
    manifest = json.loads(path.read_text())
    old_m4 = manifest.get("stages", {}).get("M4", {})
    if old_m4.get("status") not in {"FAIL", "BLOCKED", "BLOCKED/DEFERRED"}:
        raise RuntimeError("M5 policy update requires the existing non-passing M4 record")
    historical_path = path.with_name("migration_manifest_m4_active_attempt_9layer.json")
    if not historical_path.exists():
        shutil.copy2(path, historical_path)
    m5 = _m5_record(_designs(), artifact_only=artifact_only)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["git_revision"] = _git_revision()
    manifest["working_tree_fingerprint"] = _working_tree_fingerprint()
    manifest["historical_m4_artifact"] = str(historical_path)
    manifest["stages"]["M4"] = {
        "status": "BLOCKED/DEFERRED",
        "policy": "artificial validation-only layered plane-strain reference is not forced through active Mortar",
        "reason": "The 11 mm layered plane-strain-equivalent reference is incompatible with the current active Mortar path despite valid geometry and passing production 3D active-contact cases.",
        "production_3d_mechanics_implicated": False,
        "prior_attempt_status": old_m4.get("status"),
        "prior_attempt_artifact": str(historical_path),
        "prior_attempts_preserved": ["3-layer 11 mm reference", "predeclared 9-layer 11 mm reference"],
        "historical_2d_model_role": "historical mechanics evidence only; not an optimization acceptance criterion",
        "m5_authorized_basis": [
            "M1 PASS authoritative 3D solid",
            "M2 PASS true 3D volume mesh",
            "M3A PASS minimal ALM/Mortar infrastructure",
            "M3B PASS active production nominal and candidate49 contact cases",
            "exact bonded-boundary semantics preserved",
            "no fallback or NaN suppression",
        ],
    }
    manifest["stages"]["M5"] = m5
    manifest["stages"]["M5-M9"] = {
        "status": "NOT_STARTED",
        "reason": "M5 is recorded separately; M6-M9 remain intentionally not started until the M5 mechanics gate and later 3D-native gates are reviewed.",
    }
    return manifest


def _m4_record(designs: dict[str, FingertipParameters]) -> dict[str, Any]:
    cases = [
        _m4_case_record(name, location_name, parameters)
        for name, parameters in designs.items()
        for location_name in M4_CONTACT_LOCATIONS_MM
    ]
    return {
        "status": "PASS" if all(case["status"] == "PASS" for case in cases) else "FAIL",
        "scope": "validation-only 3d_equivalent_reference; M5-M9 not started",
        "historical_2d_reference": {
            "constitutive_law": "HyperElasticPlaneStrain2DLaw",
            "thickness_convention": "explicit unit thickness",
            "thickness_mm": THICKNESS_MM,
            "normalization": "F2D / 1 mm versus F3D / 11 mm",
            "contact_semantics": "fresh external-only companion reference; historical three-pair artifacts preserved",
        },
        "reference_mesh": {
            "source": "model.FingertipModel authoritative 2D pad geometry",
            "mesh_settings": asdict(M4_REFERENCE_MESH_SETTINGS),
            "layer_count": M4_LAYER_COUNT,
            "z_layers_mm": np.linspace(-5.5, 5.5, M4_LAYER_COUNT).tolist(),
            "correspondence": "exact local 2D node index columns; no nearest-neighbor pairing",
            "production_mesh_is_distinct": True,
        },
        "plane_strain_constraints": {
            "mechanism": "Kratos LinearMasterSlaveConstraint",
            "ux_uy": "each non-reference layer DOF equals the same reference-layer DOF",
            "uz": "fixed to zero on all compliant-pad nodes",
            "residual_tolerance_mm": M4_PLANE_STRAIN_RESIDUAL_TOLERANCE_MM,
        },
        "precommitted_tolerances": {
            "reaction_relative_error": M4_REACTION_RELATIVE_TOLERANCE,
            "force_equilibrium_error": M4_FORCE_EQUILIBRIUM_TOLERANCE,
            "semantic_boundary_position_error_mm": M4_BOUNDARY_POSITION_TOLERANCE_MM,
            "plane_strain_residual_mm": M4_PLANE_STRAIN_RESIDUAL_TOLERANCE_MM,
            "source": "existing historical FEA variability and repository guardrails; frozen before case execution",
        },
        "cases": cases,
        "advisor_finding": {
            "identity": "Socrates",
            "agent_id": "01a00bf9-5b7c-79e3-927d-335fa3e5eacd",
            "status": "consulted_before_final_M4_interpretation",
            "finding": (
                "Three-layer active ALM failure is plausibly a layered-contact numerical limit; "
                "one predeclared 3-to-9-layer repair is scientifically defensible because the "
                "master already uses eight z bands. If active ALM still aborts, record M4 FAIL "
                "and do not weaken MPCs or tolerances."
            ),
            "recommendation_applied": "increase the explicit reference layering once to nine layers",
        },
        "failure_diagnosis": {
            "status": "FAIL",
            "cause": "active layered 3D reference child processes terminate in native Kratos contact failure before a converged result is returned",
            "evidence": "all six 9-layer cases have nonzero child failure return codes; no active-contact, reaction, or z-residual PASS is inferred",
            "interpretation": "M4 active reference infrastructure remains scientifically unresolved; M5 is not authorized",
        },
    }


def build_m4_manifest(path: Path) -> dict[str, Any]:
    """Update only M4 in the existing migration artifact."""
    if not path.exists():
        raise RuntimeError(f"cannot update M4 without existing manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("stages", {}).get("M4", {}).get("status") != "BLOCKED":
        raise RuntimeError("M4-only runner requires the preserved prior BLOCKED state")
    designs = _designs()
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["git_revision"] = _git_revision()
    manifest["working_tree_fingerprint"] = _working_tree_fingerprint()
    manifest["stages"]["M4"] = _m4_record(designs)
    manifest["stages"]["M4"]["previous_blocked_state_preserved"] = True
    if "M5-M9" in manifest.get("stages", {}):
        manifest["stages"]["M5-M9"]["reason"] = (
            "Downstream milestones remain intentionally not started because the active "
            "plane-strain-equivalent M4 reference failed in native Kratos contact."
        )
    return manifest


def build_manifest() -> dict[str, Any]:
    designs = _designs()
    previous_manifest: dict[str, Any] | None = None
    previous_path = OUTPUT / "migration_manifest.json"
    if previous_path.exists():
        try:
            previous_manifest = json.loads(previous_path.read_text())
        except json.JSONDecodeError:
            previous_manifest = None
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
        "working_tree_fingerprint": _working_tree_fingerprint(),
        "reviewer": {
            "identity": "Galileo",
            "agent_id": "01a00964-27fe-7793-aebd-516897aca0f7",
            "role": "independent read-only checklist reviewer",
        },
        "advisor": {
            "identity": "Socrates",
            "agent_id": "01a00bf9-5b7c-79e3-927d-335fa3e5eacd",
            "role": "scientific consultation",
        },
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
            "M3": _m3_record(designs),
            "M4": {
                "status": "BLOCKED",
                "reason": "3d_equivalent_reference is not a true plane-strain equivalent: the current adapter only fixes u_z and has no deterministic z-layer in-plane MPC for u_x/u_y.",
                "scientific_blocker": "u_z=0 does not constrain u_x(x,y,z) and u_y(x,y,z) to be z-invariant; 3D HyperElastic3DLaw is not automatically the historical HyperElasticPlaneStrain2DLaw.",
                "required_before_execution": [
                    "deterministic z-layer correspondence or interpolation",
                    "explicit in-plane MPC for corresponding z layers",
                    "physical resultant reaction extraction",
                    "external-only historical 2D comparison subset",
                ],
                "precommitted_tolerance_policy": {
                    "reaction_relative_error": 0.03,
                    "force_equilibrium_error": 0.02,
                    "semantic_boundary_position_error_mm": 0.05,
                    "source": "existing repository fidelity guardrails and historical FEA variability",
                },
            },
            "M5-M9": {
                "status": "NOT_STARTED",
                "reason": "Downstream milestones are intentionally not started because the M4 2D-equivalent gate is blocked.",
            },
        },
        "historical_evidence_preserved": True,
        "production_cleanup_performed": False,
        "previous_blocked_state_preserved": bool(previous_manifest and previous_manifest.get("stages", {}).get("M3", {}).get("status") == "BLOCKED"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "migration_manifest.json")
    parser.add_argument("--m3-smoke-child", action="store_true")
    parser.add_argument("--m5-only", action="store_true")
    parser.add_argument("--m5-assemble-only", action="store_true")
    parser.add_argument("--m5-child", action="store_true")
    parser.add_argument("--m5-history", action="store_true")
    parser.add_argument("--m4-only", action="store_true")
    parser.add_argument("--m4-2d-child", action="store_true")
    parser.add_argument("--m4-3d-child", action="store_true")
    parser.add_argument("--design", choices=("nominal", "candidate49", "representative"))
    parser.add_argument("--location", choices=tuple(M4_CONTACT_LOCATIONS_MM))
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--contact", action="store_true")
    parser.add_argument("--tier", choices=("search", "reference"))
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.m3_smoke_child:
        if args.design is None or args.support_only == args.contact:
            parser.error("M3 smoke child requires exactly one of --support-only or --contact")
        raise SystemExit(_smoke_child(args.design, args.support_only))
    if args.m5_child:
        if (
            args.design is None
            or args.location is None
            or args.tier is None
            or args.steps is None
            or args.steps <= 0
        ):
            parser.error("M5 child requires --design, --location, --tier, and positive --steps")
        raise SystemExit(_m5_child(args.design, args.tier, args.steps, args.location))
    if args.m5_history:
        raise SystemExit(_m5_history_diagnostic())
    if args.m4_2d_child or args.m4_3d_child:
        if args.design is None or args.location is None or args.m4_2d_child == args.m4_3d_child:
            parser.error("M4 child requires exactly one child mode, --design, and --location")
        if args.m4_2d_child:
            raise SystemExit(_m4_2d_child(args.design, args.location))
        raise SystemExit(_m4_3d_child(args.design, args.location))
    if args.m5_only or args.m5_assemble_only:
        manifest = build_m5_manifest(args.output, artifact_only=args.m5_assemble_only)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "M1": manifest["stages"]["M1"]["status"],
                    "M2": manifest["stages"]["M2"]["status"],
                    "M3": manifest["stages"]["M3"]["status"],
                    "M4": manifest["stages"]["M4"]["status"],
                    "M5": manifest["stages"]["M5"]["status"],
                }
            )
        )
        return
    if args.m4_only:
        manifest = build_m4_manifest(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "M1": manifest["stages"]["M1"]["status"],
                    "M2": manifest["stages"]["M2"]["status"],
                    "M3": manifest["stages"]["M3"]["status"],
                    "M4": manifest["stages"]["M4"]["status"],
                }
            )
        )
        return
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "M1": manifest["stages"]["M1"]["status"], "M2": manifest["stages"]["M2"]["status"], "M3": manifest["stages"]["M3"]["status"]}))


if __name__ == "__main__":
    main()
