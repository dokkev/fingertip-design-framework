"""Bounded mechanics-only validation for finite ``void_height`` contact.

This module is deliberately an orchestration-level diagnostic.  It does not
change the production design space, the Newton backend, or any optical/FEA
path.  The two carrier modes are explicit:

* ``collision=off`` passes the carrier as ``visual_carrier_mesh`` only;
* ``collision=on`` passes the same mesh as ``rigid_carrier_mesh``.

The default invocation runs one nominal, ``void_height=1 mm``, 3 mm travel
case at ``u=0.5`` in both modes.  ``--matrix`` enables the bounded diagnostic
matrix from the current mechanics contract; it is intentionally never run at
import time.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
from shapely import wkt
from shapely.geometry import Point

from contact import (
    FirstContactSettings,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
    sphere_alignment_at_normalized_location,
)
from physics import (
    IndentationResult,
    IndentationSettings,
    NewtonSettings,
    RigidIndenter3D,
    prepare_fingertip_mesh,
    solve_fingertip_indentation,
)
from mesh import make_distal_phalanx_mesh, make_sphere_mesh
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters


OUTPUT_DIR = Path("output/validation/physics/void_height_carrier_contact")
SEARCH_SPHERE_SUBDIVISIONS = 3
SEARCH_MAX_LOAD_INCREMENT_MM = 0.05
SEARCH_VBD_ITERATIONS = 10
SEARCH_DT_S = 1.0e-3
VALIDATION_MAX_LOAD_INCREMENT_MM = 0.025
VALIDATION_VBD_ITERATIONS = 20
SOFT_CONTACT_MARGIN_MM = 0.02
SOFT_CONTACT_KE = 1.0e3
SOFT_CONTACT_KD = 10.0
RIGID_SDF_TARGET_VOXEL_MM = 0.125
DEFAULT_RADIUS_MM = 5.0
DEFAULT_LOCATION_U = 0.50
DEFAULT_VOID_HEIGHTS_MM = (0.25, 0.5, 1.0, 2.0, 3.0)
DEFAULT_TRAVELS_MM = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

NOMINAL = {
    "flat_pad_height": 5.0,
    "stem_width": 7.6,
    "stem_height": 6.0,
    "void_width": 1.0,
}
CURRENT_BEST = {
    "flat_pad_height": 6.5,
    "stem_width": 6.5,
    "stem_height": 7.5,
    "void_width": 2.0,
}


def load_steps_for_increment(travel_mm: float, increment_mm: float) -> int:
    """Return the frozen load-step count for a post-contact travel."""

    travel = float(travel_mm)
    increment = float(increment_mm)
    if not np.isfinite(travel) or travel <= 0.0:
        raise ValueError("travel_mm must be finite and positive")
    if not np.isfinite(increment) or increment <= 0.0:
        raise ValueError("increment_mm must be finite and positive")
    return max(1, int(math.ceil(travel / increment)))


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)[np.asarray(tetrahedra, dtype=int)]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


def _rms_displacement(displacement: np.ndarray) -> float:
    values = np.asarray(displacement, dtype=float)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _parameters(base: Mapping[str, float], void_height_mm: float) -> FingertipParameters:
    values = dict(base)
    values["void_height"] = float(void_height_mm)
    return FingertipParameters(**values)


def _signed_void_bottom_clearance_mm(
    vertices_mm: np.ndarray,
    prepared: Any,
    carrier_mesh: Any,
) -> float:
    """Return signed XY clearance of the deformed void-bottom surface.

    The carrier cross-section and z extent come from the authoritative rigid
    carrier mesh metadata.  Positive means outside the carrier polygon,
    negative means geometrically inside it.  This is an independent
    diagnostic; it does not participate in Newton contact or alter results.
    """

    try:
        carrier_polygon = wkt.loads(str(carrier_mesh.metadata["cross_section_wkt"]))
        z_min_mm = float(carrier_mesh.metadata["z_min_mm"])
        z_max_mm = float(carrier_mesh.metadata["z_max_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("carrier mesh lacks valid cross-section/z-bound metadata") from exc
    triangles = np.asarray(prepared.surface_triangles["void_bottom"], dtype=np.int64)
    coordinates = np.asarray(vertices_mm, dtype=float)
    distances = []
    # Evaluate a small barycentric grid on every 3D surface triangle rather
    # than only its vertices. This is an independent diagnostic and catches
    # penetration between mesh nodes without changing the Newton solve.
    sample_order = 8
    for triangle in triangles:
        a, b, c = coordinates[triangle]
        for i in range(sample_order + 1):
            for j in range(sample_order + 1 - i):
                k = sample_order - i - j
                point_3d = (i * a + j * b + k * c) / sample_order
                if not z_min_mm - 1.0e-9 <= point_3d[2] <= z_max_mm + 1.0e-9:
                    continue
                point = Point(float(point_3d[0]), float(point_3d[1]))
                distance = float(point.distance(carrier_polygon.boundary))
                distances.append(-distance if carrier_polygon.covers(point) else distance)
    if not distances:
        raise ValueError("void_bottom surface contains no vertices")
    return float(min(distances))


def _case_id(
    morphology: str,
    void_height_mm: float,
    travel_mm: float,
    location_u: float,
    collision: str,
    profile: str,
) -> str:
    raw = (
        f"{morphology}_vh{void_height_mm:g}_travel{travel_mm:g}_"
        f"u{location_u:g}_{collision}_{profile}"
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _settings(profile: str, travel_mm: float) -> dict[str, float | int | str]:
    if profile == "search":
        increment = SEARCH_MAX_LOAD_INCREMENT_MM
        iterations = SEARCH_VBD_ITERATIONS
        dt = SEARCH_DT_S
    elif profile == "validation":
        increment = VALIDATION_MAX_LOAD_INCREMENT_MM
        iterations = VALIDATION_VBD_ITERATIONS
        dt = SEARCH_DT_S
    else:
        raise ValueError("profile must be 'search' or 'validation'")
    return {
        "profile": profile,
        "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
        "max_load_increment_mm": increment,
        "load_steps": load_steps_for_increment(travel_mm, increment),
        "vbd_iterations": iterations,
        "dt_s": dt,
        "soft_contact_margin_mm": SOFT_CONTACT_MARGIN_MM,
        "soft_contact_ke": SOFT_CONTACT_KE,
        "soft_contact_kd": SOFT_CONTACT_KD,
        "rigid_sdf_target_voxel_mm": RIGID_SDF_TARGET_VOXEL_MM,
    }


def _independent_clearance_pair(
    result: IndentationResult,
    prepared: Any,
    carrier_mesh: Any,
) -> tuple[float, float, float]:
    rest = _signed_void_bottom_clearance_mm(
        result.mechanics_result.rest_vertices, prepared, carrier_mesh
    )
    final = _signed_void_bottom_clearance_mm(
        result.mechanics_result.deformed_vertices, prepared, carrier_mesh
    )
    return rest, final, min(rest, final)


def _write_geometry_artifact(
    path: Path,
    result: IndentationResult,
    prepared: Any,
    carrier_mesh: Any,
) -> str:
    mechanics = result.mechanics_result
    np.savez_compressed(
        path,
        rest_vertices_mm=np.asarray(mechanics.rest_vertices, dtype=np.float32),
        deformed_vertices_mm=np.asarray(mechanics.deformed_vertices, dtype=np.float32),
        displacement_mm=np.asarray(mechanics.displacement, dtype=np.float32),
        tetrahedra=np.asarray(mechanics.tetrahedra, dtype=np.int32),
        source_node_ids=np.asarray(prepared.source_node_ids, dtype=np.int64),
        void_bottom_vertex_indices=np.asarray(
            np.unique(prepared.surface_triangles["void_bottom"]), dtype=np.int32
        ),
        carrier_vertices_mm=np.asarray(carrier_mesh.vertices_mm, dtype=np.float32),
        carrier_faces=np.asarray(carrier_mesh.faces, dtype=np.int32),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_case(
    *,
    morphology_name: str,
    parameters: FingertipParameters,
    radius_mm: float,
    location_u: float,
    travel_mm: float,
    collision: str,
    profile: str,
    device: str,
    output_dir: Path,
    volume_mesh: Any,
    prepared: Any,
    carrier_mesh: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    identifier = _case_id(
        morphology_name,
        parameters.void_height,
        travel_mm,
        location_u,
        collision,
        profile,
    )
    contract = _settings(profile, travel_mm)
    record: dict[str, Any] = {
        "case_id": identifier,
        "status": "failed",
        "morphology": morphology_name,
        "parameters": asdict(parameters),
        "void_height_mm": float(parameters.void_height),
        "sphere_radius_mm": float(radius_mm),
        "location_u": float(location_u),
        "travel_mm": float(travel_mm),
        "collision": collision,
        "profile": profile,
        "load_steps": contract["load_steps"],
        "vbd_iterations": contract["vbd_iterations"],
        "dt_s": contract["dt_s"],
        "sphere_subdivisions": contract["sphere_subdivisions"],
        "runtime_s": None,
        "mesh_morphology_fingerprint": volume_mesh.morphology_fingerprint,
    }
    try:
        sphere_mesh = make_sphere_mesh(
            radius_mm,
            subdivisions=int(contract["sphere_subdivisions"]),
        )
        contact_surface = make_outer_compliant_surface(volume_mesh.solid)
        alignment = sphere_alignment_at_normalized_location(
            Fingertip(parameters).geometry,
            sphere_mesh,
            location_u,
            initial_gap_mm=0.25,
        )
        if intersects(contact_surface, sphere_mesh, alignment.nominal_pose):
            raise RuntimeError("nominal sphere pose is not collision-free")
        first_contact = find_first_contact(
            contact_surface,
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
            FirstContactSettings(
                coarse_step_mm=0.25,
                tolerance_mm=1.0e-3,
                spawn_clearance_mm=0.05,
                max_travel_mm=20.0,
            ),
        )
        if intersects(contact_surface, sphere_mesh, first_contact.spawn_pose):
            raise RuntimeError("first-contact spawn pose is not collision-free")
        indenter = RigidIndenter3D(
            sphere_mesh,
            alignment.nominal_pose,
            alignment.approach_direction,
        )
        mechanics = NewtonSettings(
            device=device,
            gravity=0.0,
            dt=float(contract["dt_s"]),
            steps=1,
            iterations=int(contract["vbd_iterations"]),
            fixed_vertex_indices=prepared.support_vertex_indices,
        )
        indentation_settings = IndentationSettings(
            travel_mm=travel_mm,
            load_steps=int(contract["load_steps"]),
            soft_contact_margin_mm=SOFT_CONTACT_MARGIN_MM,
            rigid_sdf_target_voxel_mm=RIGID_SDF_TARGET_VOXEL_MM,
            soft_contact_ke=SOFT_CONTACT_KE,
            soft_contact_kd=SOFT_CONTACT_KD,
        )
        kwargs: dict[str, Any]
        if collision == "off":
            kwargs = {"visual_carrier_mesh": carrier_mesh}
        elif collision == "on":
            kwargs = {"rigid_carrier_mesh": carrier_mesh}
        else:
            raise ValueError("collision must be 'off' or 'on'")
        result = solve_fingertip_indentation(
            prepared,
            indenter,
            mechanics,
            indentation_settings,
            first_contact=first_contact,
            **kwargs,
        )
        mechanics_result = result.mechanics_result
        six_volumes = _six_volumes(
            mechanics_result.deformed_vertices,
            mechanics_result.tetrahedra,
        )
        finite = bool(
            np.all(np.isfinite(mechanics_result.deformed_vertices))
            and np.all(np.isfinite(mechanics_result.displacement))
            and np.all(np.isfinite(six_volumes))
        )
        expected_pose = first_contact.pose_at_post_contact_travel(travel_mm)
        final_pose_error = float(
            np.linalg.norm(
                np.asarray(result.final_indenter_pose.translation_mm)
                - np.asarray(expected_pose.translation_mm)
            )
        )
        rest_gap, final_gap, endpoint_min_gap = _independent_clearance_pair(
            result, prepared, carrier_mesh
        )
        diagnostics = dict(result.diagnostics)
        if collision == "on":
            # Keep Newton's per-step vertex diagnostic and the independent
            # triangle-surface endpoint diagnostic fail-closed.
            min_gap = min(
                float(diagnostics["min_carrier_clearance_mm"]),
                final_gap,
            )
            max_penetration = max(0.0, -min_gap)
            carrier_count = int(diagnostics["max_void_bottom_carrier_contact_count"])
            carrier_count_all = int(diagnostics["max_carrier_soft_contact_count"])
            first_carrier_step = diagnostics["first_carrier_contact_step"]
            carrier_active = bool(diagnostics["carrier_contact_active"])
        else:
            min_gap = endpoint_min_gap
            max_penetration = max(0.0, -min_gap)
            carrier_count = 0
            carrier_count_all = 0
            first_carrier_step = None
            carrier_active = False
        artifact_path = output_dir / "geometry" / f"{identifier}.npz"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_sha256 = _write_geometry_artifact(
            artifact_path, result, prepared, carrier_mesh
        )
        record.update(
            {
                "status": "passed" if finite else "failed",
                "first_sphere_contact_travel_mm": first_contact.travel_to_contact_mm,
                "first_contact_bracket_width_mm": first_contact.bracket_width_mm,
                "initial_void_bottom_carrier_clearance_mm": rest_gap,
                "final_void_bottom_carrier_clearance_mm": final_gap,
                "min_carrier_clearance_mm": min_gap,
                "max_carrier_penetration_mm": max_penetration,
                "max_displacement_mm": float(
                    np.max(np.linalg.norm(mechanics_result.displacement, axis=1))
                ),
                "rms_displacement_mm": _rms_displacement(mechanics_result.displacement),
                "min_six_volume_mm3_x6": float(np.min(six_volumes)),
                "inverted_tet_count": int(np.count_nonzero(six_volumes <= 0.0)),
                "finite_state": finite,
                "sphere_soft_contact_count": int(
                    diagnostics.get("max_sphere_soft_contact_count", 0)
                ),
                "carrier_soft_contact_count": carrier_count,
                "carrier_soft_contact_count_all_surfaces": carrier_count_all,
                "first_carrier_contact_step": first_carrier_step,
                "carrier_contact_active": carrier_active,
                "max_soft_contact_overflow": int(
                    diagnostics.get("max_soft_contact_overflow", 0)
                ),
                "max_rigid_contact_overflow": int(
                    diagnostics.get("max_rigid_contact_overflow", 0)
                ),
                "sphere_carrier_rigid_contact_count": int(
                    diagnostics.get("max_sphere_carrier_rigid_contact_count", 0)
                ),
                "final_sphere_pose_error_mm": final_pose_error,
                "artifact_path": str(artifact_path),
                "artifact_sha256": artifact_sha256,
                "clearance_sampling": (
                    "independent void_bottom triangle barycentric grid (order 8) at "
                    "rest/final states; collision-on additionally reports the "
                    "Newton solver-step minimum"
                ),
                "backend_diagnostics": diagnostics,
            }
        )
        carrier_penetration_tolerance_mm = 0.5 * RIGID_SDF_TARGET_VOXEL_MM
        record["carrier_penetration_tolerance_mm"] = carrier_penetration_tolerance_mm
        record["contact_health_pass"] = bool(
            finite
            and record["inverted_tet_count"] == 0
            and record["max_soft_contact_overflow"] == 0
            and record["max_rigid_contact_overflow"] == 0
            and record["sphere_carrier_rigid_contact_count"] == 0
            and record["final_sphere_pose_error_mm"] <= 1.0e-6
            and record["initial_void_bottom_carrier_clearance_mm"] > 0.0
            and (
                collision != "on"
                or record["max_carrier_penetration_mm"]
                <= carrier_penetration_tolerance_mm
            )
        )
        if not record["contact_health_pass"]:
            record["status"] = "failed"
    except Exception as exc:  # retain a machine-readable failed case
        record["exception"] = f"{type(exc).__name__}: {exc}"
    record["runtime_s"] = time.perf_counter() - started
    return record


def _morphology_bases(selection: str) -> dict[str, dict[str, float]]:
    if selection == "nominal":
        return {"nominal": NOMINAL}
    if selection == "best":
        return {"current_best": CURRENT_BEST}
    if selection == "both":
        return {"nominal": NOMINAL, "current_best": CURRENT_BEST}
    raise ValueError("morphology must be nominal, best, or both")


def _write_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n")


def _write_records(output_dir: Path, records: list[dict[str, Any]]) -> None:
    _write_json(output_dir / "carrier_contact_runs.json", records)
    fields = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if not isinstance(value, (dict, list, tuple))
        }
    )
    with (output_dir / "carrier_contact_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})

    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        pair_key = re.sub(r"_(off|on)_[^_]+$", "", record["case_id"])
        paired.setdefault(pair_key, {})[record["collision"]] = record
    _write_json(output_dir / "collision_off_vs_on.json", paired)


def _write_plots(output_dir: Path, records: list[dict[str, Any]]) -> None:
    successful = [record for record in records if record.get("status") == "passed"]
    if not successful:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"off": "#9aa0a6", "on": "#1769aa"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for collision in ("off", "on"):
        subset = [record for record in successful if record["collision"] == collision]
        if subset:
            ax.scatter(
                [record["travel_mm"] for record in subset],
                [record["min_carrier_clearance_mm"] for record in subset],
                label=f"collision {collision}",
                color=colors[collision],
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="post-contact travel [mm]", ylabel="minimum signed carrier clearance [mm]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "min_gap_vs_travel.png", dpi=160)
    plt.close(fig)

    onset = [
        record
        for record in successful
        if record["collision"] == "on" and record["first_carrier_contact_step"] is not None
    ]
    if onset:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(
            [record["void_height_mm"] for record in onset],
            [record["first_carrier_contact_step"] for record in onset],
            c=[record["travel_mm"] for record in onset],
            cmap="viridis",
        )
        ax.set(xlabel="void height [mm]", ylabel="first carrier-contact step")
        fig.tight_layout()
        fig.savefig(output_dir / "carrier_contact_onset.png", dpi=160)
        plt.close(fig)

    paired = {}
    for record in successful:
        key = (record["morphology"], record["void_height_mm"], record["travel_mm"])
        paired.setdefault(key, {})[record["collision"]] = record
    pairs = [pair for pair in paired.values() if "off" in pair and "on" in pair]
    if pairs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(pairs))
        width = 0.38
        ax.bar(
            x - width / 2,
            [pair["off"]["min_carrier_clearance_mm"] for pair in pairs],
            width,
            label="collision off",
            color=colors["off"],
        )
        ax.bar(
            x + width / 2,
            [pair["on"]["min_carrier_clearance_mm"] for pair in pairs],
            width,
            label="collision on",
            color=colors["on"],
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel="paired morphology / travel case", ylabel="minimum signed clearance [mm]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "collision_off_vs_on.png", dpi=160)
        plt.close(fig)

    if successful:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for collision in ("off", "on"):
            subset = [record for record in successful if record["collision"] == collision]
            if subset:
                ax.plot(
                    [record["travel_mm"] for record in subset],
                    [record["carrier_soft_contact_count"] for record in subset],
                    "o",
                    alpha=0.75,
                    label=f"collision {collision}",
                    color=colors[collision],
                )
        ax.set(xlabel="post-contact travel [mm]", ylabel="maximum void-bottom carrier contacts")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "contact_count_vs_travel.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for collision in ("off", "on"):
            subset = [record for record in successful if record["collision"] == collision]
            if subset:
                ax.plot(
                    [record["travel_mm"] for record in subset],
                    [record["min_six_volume_mm3_x6"] for record in subset],
                    "o",
                    alpha=0.75,
                    label=f"collision {collision}",
                    color=colors[collision],
                )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel="post-contact travel [mm]", ylabel="minimum signed six-volume [mm³]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "tet_health_vs_travel.png", dpi=160)
        plt.close(fig)


def run_validation(
    *,
    matrix: bool = False,
    morphology: str = "nominal",
    profile: str = "search",
    radius_mm: float = DEFAULT_RADIUS_MM,
    location_u: float = DEFAULT_LOCATION_U,
    locations: Iterable[float] | None = None,
    void_height_mm: float = 1.0,
    travel_mm: float = 3.0,
    collisions: Iterable[str] = ("off", "on"),
    device: str = "cuda:0",
    output_dir: str | Path = OUTPUT_DIR,
    make_plots: bool = True,
) -> list[dict[str, Any]]:
    """Run one case or the bounded finite-clearance diagnostic matrix."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_collisions = tuple(collisions)
    if any(value not in ("off", "on") for value in selected_collisions):
        raise ValueError("collisions must contain only 'off' and 'on'")
    if not selected_collisions:
        raise ValueError("at least one collision mode is required")

    if matrix:
        morphology_selection = "both"
        void_heights = DEFAULT_VOID_HEIGHTS_MM
        travels = DEFAULT_TRAVELS_MM
        selected_locations = (DEFAULT_LOCATION_U,)
    else:
        morphology_selection = morphology
        void_heights = (float(void_height_mm),)
        travels = (float(travel_mm),)
        selected_locations = tuple(locations) if locations is not None else (location_u,)

    if any(not 0.0 <= float(value) <= 1.0 for value in selected_locations):
        raise ValueError("locations must lie in [0, 1]")
    if any(float(value) <= 0.0 for value in void_heights + travels):
        raise ValueError("void heights and travels must be positive")

    config = {
        "runner": "validation.physics.void_height_carrier_contact",
        "matrix": matrix,
        "device": device,
        "profile": profile,
        "radius_mm": radius_mm,
        "locations": selected_locations,
        "void_heights_mm": void_heights,
        "travels_mm": travels,
        "collisions": selected_collisions,
        "morphologies": _morphology_bases(morphology_selection),
        "frozen_contract": {
            "search": _settings("search", max(DEFAULT_TRAVELS_MM)),
            "validation": _settings("validation", max(DEFAULT_TRAVELS_MM)),
            "sphere_first_contact": {
                "coarse_step_mm": 0.25,
                "tolerance_mm": 1.0e-3,
                "spawn_clearance_mm": 0.05,
            },
        },
        "optical_model": "not exercised by this mechanics-only runner",
    }
    _write_json(output_path / "config.json", config)

    mesh_cache: dict[FingertipParameters, tuple[Any, Any, Any]] = {}

    def get_meshes(parameters: FingertipParameters) -> tuple[Any, Any, Any]:
        cached = mesh_cache.get(parameters)
        if cached is None:
            fingertip = Fingertip(parameters)
            volume_mesh = generate_volume_mesh(
                fingertip.solid(),
                volume_mesh_settings_for_tier("search"),
            )
            prepared = prepare_fingertip_mesh(volume_mesh)
            carrier = make_distal_phalanx_mesh(volume_mesh.solid)
            cached = (volume_mesh, prepared, carrier)
            mesh_cache[parameters] = cached
        return cached

    records: list[dict[str, Any]] = []
    for morphology_name, base in _morphology_bases(morphology_selection).items():
        for void_height in void_heights:
            parameters = _parameters(base, float(void_height))
            volume_mesh, prepared, carrier = get_meshes(parameters)
            for travel in travels:
                for location in selected_locations:
                    for collision in selected_collisions:
                        records.append(
                            _run_case(
                                morphology_name=morphology_name,
                                parameters=parameters,
                                radius_mm=radius_mm,
                                location_u=float(location),
                                travel_mm=float(travel),
                                collision=collision,
                                profile=profile,
                                device=device,
                                output_dir=output_path,
                                volume_mesh=volume_mesh,
                                prepared=prepared,
                                carrier_mesh=carrier,
                            )
                        )

    _write_records(output_path, records)
    _write_json(
        output_path / "search_vs_validation.json",
        {
            "executed_profile": profile,
            "search_contract": _settings("search", max(DEFAULT_TRAVELS_MM)),
            "validation_contract": _settings("validation", max(DEFAULT_TRAVELS_MM)),
            "note": "Run with --profile validation for the frozen validation spot check.",
        },
    )
    if make_plots:
        _write_plots(output_path, records)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true", help="run the bounded 2x5x6x2 matrix")
    parser.add_argument("--morphology", choices=("nominal", "best", "both"), default="nominal")
    parser.add_argument("--profile", choices=("search", "validation"), default="search")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_MM, dest="radius_mm")
    parser.add_argument("--location", type=float, default=DEFAULT_LOCATION_U, dest="location_u")
    parser.add_argument("--locations", help="comma-separated locations for one-case spot checks")
    parser.add_argument("--void-height", type=float, default=1.0, dest="void_height_mm")
    parser.add_argument("--travel", type=float, default=3.0, dest="travel_mm")
    parser.add_argument("--collision", choices=("off", "on", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, dest="output_dir")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    locations = None
    if args.locations:
        locations = tuple(float(value) for value in args.locations.split(","))
    collisions = ("off", "on") if args.collision == "both" else (args.collision,)
    records = run_validation(
        matrix=args.matrix,
        morphology=args.morphology,
        profile=args.profile,
        radius_mm=args.radius_mm,
        location_u=args.location_u,
        locations=locations,
        void_height_mm=args.void_height_mm,
        travel_mm=args.travel_mm,
        collisions=collisions,
        device=args.device,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )
    passed = sum(record.get("status") == "passed" for record in records)
    failed = len(records) - passed
    print(f"void-height carrier contact: {passed}/{len(records)} cases passed")
    if failed:
        for record in records:
            if record.get("status") != "passed":
                print(f"FAIL {record['case_id']}: {record.get('exception', 'health check failed')}")
        return 1
    for record in records:
        print(
            f"PASS {record['case_id']}: "
            f"carrier_active={record['carrier_contact_active']} "
            f"min_gap_mm={record['min_carrier_clearance_mm']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CURRENT_BEST",
    "DEFAULT_TRAVELS_MM",
    "DEFAULT_VOID_HEIGHTS_MM",
    "NOMINAL",
    "OUTPUT_DIR",
    "RIGID_SDF_TARGET_VOXEL_MM",
    "load_steps_for_increment",
    "main",
    "run_validation",
]
