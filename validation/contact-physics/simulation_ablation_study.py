"""Run the Figure 3 rigid-soft mechanics ablation on one matched scenario."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import newton  # noqa: E402
import newton.viewer  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from lumo.fingertip import (  # noqa: E402
    ACTIVE_Y_BOUNDS_MM,
    DISTAL_END_CAP_LENGTH_MM,
    LED_CENTERS_Y_MM,
    LED_RECESS_DEPTH_MM,
    LED_RECESS_WIDTH_MM,
    TOTAL_Y_BOUNDS_MM,
    BondingInterface,
    Fingertip,
    FingertipGeometry,
    FingertipParameters,
)
from lumo.mesh import FingertipMesh, make_fingertip_mesh  # noqa: E402
from lumo.mesh.carrier_mesh import (  # noqa: E402
    _make_carrier_collision_mesh,
    _make_carrier_mesh,
)
from lumo.mesh.silicone_mesh import (  # noqa: E402
    _build_outer_cross_section,
    _mesh_volume,
)
from lumo.newton import Indenter, build_fingertip_newton_model  # noqa: E402
from lumo.optimization.evaluator import (  # noqa: E402
    _CARRIER_ALBEDO,
    _ENERGY_FIELDS,
    _MAX_BOUNCES,
    _emissions,
    _indenter_trajectory,
    _make_leds,
    _optical_samples,
    _six_tet_volumes,
    _trace_state,
    _zero_contact_travel_m,
)
from lumo.optimization.objective import (  # noqa: E402
    _active_surface_triangles,
    _surface_incidence,
    _triangle_areas,
    compute_contact_objective,
)
from lumo.simulation import LumoSimulation  # noqa: E402
from lumo.ray_tracing import LED, OptixScene, sources_inside_silicone  # noqa: E402
from lumo.visualization import publication_context  # noqa: E402


_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIRECTORY = (
    _ROOT
    / "output"
    / "optimization"
    / "mobo_fingertip_contact_1_2_5_10_05mm"
)
_OUTPUT_DIRECTORY = _ROOT / "output" / "validation" / "hybrid_mechanics_ablation"
_RESULT_PATH = _OUTPUT_DIRECTORY / "ablation_results.npz"
_SUMMARY_PATH = _OUTPUT_DIRECTORY / "summary.json"
_SAMPLE_PATH = _OUTPUT_DIRECTORY / "mechanics_samples.csv"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"
_OPTICAL_SAMPLE_PATH = _OUTPUT_DIRECTORY / "optical_samples.csv"
_GAP_SAMPLE_PATH = _OUTPUT_DIRECTORY / "gap_sensitivity_samples.csv"


def _load_campaign_reference() -> tuple[dict[str, str], dict[str, object]]:
    """Select the current valid mechanics-best campaign observation."""
    with (_CAMPAIGN_DIRECTORY / "trials.csv").open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    valid = tuple(
        row
        for row in rows
        if row["status"] == "COMPLETED"
        and row["analytically_valid"] == "True"
        and not row["failure"]
        and row["J_contact"]
        and np.isfinite(float(row["J_contact"]))
    )
    if not valid:
        raise RuntimeError("reference campaign has no valid completed observations")
    best = max(valid, key=lambda row: float(row["J_contact"]))
    configuration = json.loads(
        (_CAMPAIGN_DIRECTORY / "run_config.json").read_text()
    )
    return best, configuration


_REFERENCE_ROW, _CAMPAIGN_CONFIGURATION = _load_campaign_reference()
_SCIENTIFIC_CONTRACT = _CAMPAIGN_CONFIGURATION["scientific_contract"]
_MECHANICS_CONFIGURATION = _SCIENTIFIC_CONTRACT["mechanics"]
_SCENARIO_CONFIGURATION = _SCIENTIFIC_CONTRACT["scenarios"]
_REFERENCE_TRIAL = int(_REFERENCE_ROW["ax_trial_index"])
_MORPHOLOGY_MM = tuple(
    float(_REFERENCE_ROW[name])
    for name in (
        "geometry.flat_pad_height_mm",
        "geometry.semiellipse_height_mm",
        "geometry.stem_width_mm",
        "geometry.stem_height_mm",
        "geometry.void_width_mm",
    )
)
_CAMPAIGN_J_CONTACT = float(_REFERENCE_ROW["J_contact"])
_FORCE_TARGETS_N = np.asarray(
    _MECHANICS_CONFIGURATION["force_targets_n"],
    dtype=np.float64,
)
if _FORCE_TARGETS_N.shape != (4,):
    raise RuntimeError("J_contact ablation requires the campaign's four force targets")
_LIMITING_SCENARIO_MATCH = re.fullmatch(
    r"sphere_(?P<diameter>[0-9.]+)mm_y(?P<y>[+-]?[0-9.]+)mm",
    _REFERENCE_ROW["limiting_contact_scenario"],
)
if _LIMITING_SCENARIO_MATCH is None:
    raise RuntimeError("non-orientation limiting contact scenario is malformed")
_SPHERE_DIAMETER_MM = float(_LIMITING_SCENARIO_MATCH.group("diameter"))
_CONTACT_Y_MM = float(_LIMITING_SCENARIO_MATCH.group("y"))
_CONTACT_ANGLE_DEG = 0.0
_SIM_FREQUENCY_HZ = float(_MECHANICS_CONFIGURATION["sim_frequency_hz"])
_VBD_ITERATIONS = int(_MECHANICS_CONFIGURATION["vbd_iterations"])
_APPROACH_SPEED_M_S = float(_MECHANICS_CONFIGURATION["approach_speed_m_s"])
_INITIAL_CLEARANCE_M = float(_SCENARIO_CONFIGURATION["initial_clearance_m"])
_MAX_SIM_TIME_S = float(_MECHANICS_CONFIGURATION["max_sim_time_s"])
_ELEMENT_SIZE_MM = float(_MECHANICS_CONFIGURATION["element_size_mm"])
_SOFT_CONTACT_MARGIN_M = float(
    _MECHANICS_CONFIGURATION["soft_contact_margin_m"]
)
_INDENTER_STIFFNESS_N_M = float(
    _MECHANICS_CONFIGURATION["indenter_contact_stiffness_n_m"]
)
_INDENTER_DAMPING_N_S_M = float(
    _MECHANICS_CONFIGURATION["indenter_contact_damping_n_s_m"]
)
_MECHANICS_PRESET = str(_SCIENTIFIC_CONTRACT["mechanics_preset"])
_OPTICAL_PRESET = str(_SCIENTIFIC_CONTRACT["optical_preset"])

_CASE_NAMES = ("soft_only", "bonded_t", "lumo")
_EFFECTIVE_GAPS_MM = (0.01, LED_RECESS_DEPTH_MM, 0.50)
_GAP_LABELS = ("near_zero", "nominal", "large")
_DISPLAY_NAMES = {
    "soft_only": "Soft-only",
    "bonded_t": "Bonded-T",
    "lumo": "LUMO",
}
_STRESS_MAX_KPA = 100.0
_STRESS_BINS = 12


@dataclass(frozen=True)
class AblationCase:
    name: str
    fingertip: Fingertip
    mesh: FingertipMesh
    carrier_interaction: str
    bonded_vertex_indices: np.ndarray
    tied_interface_indices: np.ndarray
    construction: str


def _reference_fingertip() -> Fingertip:
    geometry = FingertipGeometry(
        flat_pad_height_mm=_MORPHOLOGY_MM[0],
        semiellipse_height_mm=_MORPHOLOGY_MM[1],
        stem_width_mm=_MORPHOLOGY_MM[2],
        stem_height_mm=_MORPHOLOGY_MM[3],
        void_width_mm=_MORPHOLOGY_MM[4],
    )
    fingertip = Fingertip(parameters=FingertipParameters(geometry=geometry))
    expected_parameters = _SCIENTIFIC_CONTRACT["fingertip_parameters"]
    for name, expected in expected_parameters["mechanics"].items():
        if not np.isclose(
            float(getattr(fingertip.parameters.mechanics, name)),
            float(expected),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(f"current mechanics parameter {name} changed")
    for name, expected in expected_parameters["optics"].items():
        current = getattr(fingertip.parameters.optics, name)
        if isinstance(expected, str):
            matches = current == expected
        else:
            matches = np.isclose(
                float(current),
                float(expected),
                rtol=0.0,
                atol=0.0,
            )
        if not matches:
            raise RuntimeError(f"current optical parameter {name} changed")
    return fingertip


def _solid_soft_mesh(
    fingertip: Fingertip,
    reference_mesh: FingertipMesh,
) -> FingertipMesh:
    """Fill the complete external envelope with silicone and bond its top."""
    import gmsh

    distal_y_mm = TOTAL_Y_BOUNDS_MM[1]
    total_length_mm = TOTAL_Y_BOUNDS_MM[1] - TOTAL_Y_BOUNDS_MM[0]
    gmsh_z_mm = -distal_y_mm
    top_z_mm = fingertip.parameters.geometry.link_thickness_mm
    half_width_mm = fingertip.silicone.half_width_mm
    mounting = BondingInterface(
        left=((-half_width_mm, top_z_mm), (0.0, top_z_mm)),
        right=((0.0, top_z_mm), (half_width_mm, top_z_mm)),
    )

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lumo_soft_only")
        outer = _build_outer_cross_section(
            gmsh,
            fingertip.silicone,
            z_mm=gmsh_z_mm,
        )
        upper_fill = gmsh.model.occ.addRectangle(
            -half_width_mm,
            0.0,
            gmsh_z_mm,
            2.0 * half_width_mm,
            top_z_mm,
        )
        fused, _ = gmsh.model.occ.fuse([(2, outer)], [(2, upper_fill)])
        surfaces = [tag for dimension, tag in fused if dimension == 2]
        if len(surfaces) != 1:
            raise RuntimeError("soft-only cross-section must form one surface")
        extrusion = gmsh.model.occ.extrude(
            [(2, surfaces[0])],
            0.0,
            0.0,
            total_length_mm,
        )
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        volumes = [tag for dimension, tag in extrusion if dimension == 3]
        if len(volumes) != 1:
            raise RuntimeError("soft-only body must form one volume")
        vertices_mm, tetrahedra, bonded_indices = _mesh_volume(
            gmsh,
            volumes[0],
            mounting,
            element_size_mm=_ELEMENT_SIZE_MM,
            bonded_y_bounds_mm=TOTAL_Y_BOUNDS_MM,
        )
    finally:
        gmsh.finalize()

    silicone = newton.TetMesh(
        vertices=vertices_mm * 1.0e-3,
        tet_indices=tetrahedra.reshape(-1),
    )
    return FingertipMesh(
        fingertip=fingertip,
        silicone=silicone,
        carrier=reference_mesh.carrier,
        carrier_collision=reference_mesh.carrier_collision,
        bonded_vertex_indices=bonded_indices,
    )


def _bonded_t_case(reference: Fingertip) -> AblationCase:
    conformal_geometry = replace(
        reference.parameters.geometry,
        void_width_mm=0.0,
    )
    conformal = Fingertip(
        parameters=replace(reference.parameters, geometry=conformal_geometry)
    )
    mesh = make_fingertip_mesh(conformal, element_size_mm=_ELEMENT_SIZE_MM)
    vertices_mm = np.asarray(mesh.silicone.vertices, dtype=np.float64) * 1.0e3
    half_stem_mm = 0.5 * conformal_geometry.stem_width_mm
    stem_bottom_mm = -conformal_geometry.stem_height_mm
    tolerance_mm = 1.0e-5
    active_y = (
        (vertices_mm[:, 1] >= ACTIVE_Y_BOUNDS_MM[0] - tolerance_mm)
        & (vertices_mm[:, 1] <= ACTIVE_Y_BOUNDS_MM[1] + tolerance_mm)
    )
    in_recess = np.zeros(len(vertices_mm), dtype=bool)
    half_recess_mm = 0.5 * LED_RECESS_WIDTH_MM
    for center_y_mm in LED_CENTERS_Y_MM:
        in_recess |= np.abs(vertices_mm[:, 1] - center_y_mm) < half_recess_mm
    side = (
        np.isclose(np.abs(vertices_mm[:, 0]), half_stem_mm, atol=tolerance_mm)
        & (vertices_mm[:, 2] >= stem_bottom_mm - tolerance_mm)
        & (vertices_mm[:, 2] <= tolerance_mm)
    )
    bottom = (
        np.isclose(vertices_mm[:, 2], stem_bottom_mm, atol=tolerance_mm)
        & (np.abs(vertices_mm[:, 0]) <= half_stem_mm + tolerance_mm)
    )
    tied_interface = np.flatnonzero(active_y & ~in_recess & (side | bottom)).astype(
        np.int32
    )
    if tied_interface.size == 0:
        raise RuntimeError("Bonded-T mesh has no conformal stem-interface vertices")
    all_bonded = np.unique(
        np.concatenate((mesh.bonded_vertex_indices, tied_interface))
    ).astype(np.int32)
    return AblationCase(
        name="bonded_t",
        fingertip=conformal,
        mesh=mesh,
        carrier_interaction="bonded",
        bonded_vertex_indices=all_bonded,
        tied_interface_indices=tied_interface,
        construction=(
            "same outer envelope and carrier as LUMO; zero-clearance conformal "
            "stem cavity with all non-recess stem-side/bottom interface vertices "
            "kinematically tied from the reference state; carrier penalty contact disabled"
        ),
    )


def _cases() -> tuple[AblationCase, ...]:
    reference = _reference_fingertip()
    reference_mesh = make_fingertip_mesh(
        reference,
        element_size_mm=_ELEMENT_SIZE_MM,
    )
    soft_mesh = _solid_soft_mesh(reference, reference_mesh)
    return (
        AblationCase(
            name="soft_only",
            fingertip=reference,
            mesh=soft_mesh,
            carrier_interaction="absent",
            bonded_vertex_indices=soft_mesh.bonded_vertex_indices,
            tied_interface_indices=np.empty(0, dtype=np.int32),
            construction=(
                "carrier and cavity removed; complete reference external envelope "
                "filled by homogeneous silicone; dorsal top face kinematically mounted"
            ),
        ),
        _bonded_t_case(reference),
        AblationCase(
            name="lumo",
            fingertip=reference,
            mesh=reference_mesh,
            carrier_interaction="contact",
            bonded_vertex_indices=reference_mesh.bonded_vertex_indices,
            tied_interface_indices=np.empty(0, dtype=np.int32),
            construction=(
                f"production trial-{_REFERENCE_TRIAL} geometry with "
                f"{_MORPHOLOGY_MM[4]:g} mm lateral clearance and "
                "penalty-based full-surface pad-carrier contact"
            ),
        ),
    )


def _mesh_with_effective_gap(
    case: AblationCase,
    gap_mm: float,
) -> FingertipMesh:
    """Rebuild only the carrier recess for a controlled LUMO gap replay."""
    if case.name != "lumo":
        raise ValueError("effective-gap sensitivity is defined only for LUMO")
    if gap_mm <= 0.0:
        raise ValueError("effective gap must be positive")
    active_length_mm = ACTIVE_Y_BOUNDS_MM[1] - ACTIVE_Y_BOUNDS_MM[0]
    carrier = _make_carrier_mesh(
        case.fingertip.carrier,
        active_length_mm=active_length_mm,
        distal_end_cap_length_mm=DISTAL_END_CAP_LENGTH_MM,
        led_centers_y_mm=LED_CENTERS_Y_MM,
        led_recess_width_mm=LED_RECESS_WIDTH_MM,
        led_recess_depth_mm=gap_mm,
    )
    carrier_collision = _make_carrier_collision_mesh(
        case.fingertip.carrier,
        case.fingertip.silicone,
        active_length_mm=active_length_mm,
        led_centers_y_mm=LED_CENTERS_Y_MM,
        led_recess_width_mm=LED_RECESS_WIDTH_MM,
        led_recess_depth_mm=gap_mm,
    )
    return replace(
        case.mesh,
        carrier=carrier,
        carrier_collision=carrier_collision,
    )


def _leds_for_effective_gap(
    fingertip: Fingertip,
    gap_mm: float,
) -> tuple[LED, ...]:
    """Place each source on the floor of the sensitivity-study recess."""
    source_z_m = 1.0e-3 * (
        min(z_mm for _, z_mm in fingertip.carrier.cross_section) + gap_mm
    )
    normal_W = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    return tuple(
        LED(
            position_W_m=np.asarray((0.0, 1.0e-3 * y_mm, source_z_m)),
            normal_W=normal_W,
            parameters=fingertip.parameters.led,
        )
        for y_mm in LED_CENTERS_Y_MM
    )


def _body_contact_records(
    simulation: LumoSimulation,
    body_index: int,
    vertices_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contacts = simulation.contacts
    emitted = int(contacts.soft_contact_count.numpy()[0])
    stored = min(emitted, int(contacts.soft_contact_max))
    if stored == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    shapes = contacts.soft_contact_shape.numpy()[:stored]
    valid = shapes >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    selected = valid.copy()
    selected[valid] = shape_bodies[shapes[valid]] == body_index
    indices = np.asarray(
        contacts.soft_contact_indices.numpy()[:stored][selected],
        dtype=np.int32,
    )
    weights = np.asarray(
        contacts.soft_contact_barycentric.numpy()[:stored][selected],
        dtype=np.float64,
    )
    normals = np.asarray(
        contacts.soft_contact_normal.numpy()[:stored][selected],
        dtype=np.float64,
    )
    present = indices >= 0
    indices[present] -= simulation.fingertip_model.silicone_particle_start
    if np.any(indices[present] < 0) or np.any(indices[present] >= len(vertices_m)):
        raise RuntimeError("soft contact references a non-silicone particle")
    positions = np.empty((len(indices), 3), dtype=np.float64)
    for row, (primitive, barycentric) in enumerate(
        zip(indices, weights, strict=True)
    ):
        active = primitive >= 0
        positions[row] = np.sum(
            vertices_m[primitive[active]] * barycentric[active, None],
            axis=0,
        )
    return indices, normals, positions


def _patch_area_m2(
    vertices_m: np.ndarray,
    surface_triangles: np.ndarray,
    records: np.ndarray,
    incidence: tuple[dict[int, set[int]], dict[tuple[int, int], set[int]], dict[tuple[int, ...], int]],
) -> float:
    if len(records) == 0:
        return 0.0
    active = _active_surface_triangles(
        records,
        vertex_triangles=incidence[0],
        edge_triangles=incidence[1],
        triangle_ids=incidence[2],
    )
    areas = _triangle_areas(vertices_m, surface_triangles)
    return float(areas[list(active)].sum())


def _fixed_interface_area_m2(case: AblationCase) -> float:
    if case.tied_interface_indices.size == 0:
        return 0.0
    triangles = np.asarray(
        case.mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    mask = np.all(np.isin(triangles, case.tied_interface_indices), axis=1)
    return float(
        _triangle_areas(
            np.asarray(case.mesh.silicone.vertices, dtype=np.float64),
            triangles,
        )[mask].sum()
    )


def _local_linear_stiffness_n_m(
    indentation_m: np.ndarray,
    force_n: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Return a transparent centered local-linear slope with at most five points."""
    stiffness = np.empty(len(force_n), dtype=np.float64)
    radius = window // 2
    for index in range(len(force_n)):
        start = max(0, index - radius)
        stop = min(len(force_n), index + radius + 1)
        x = indentation_m[start:stop]
        y = force_n[start:stop]
        if len(x) < 2 or np.ptp(x) <= 0.0:
            stiffness[index] = np.nan
        else:
            stiffness[index] = np.polyfit(x, y, 1)[0]
    return stiffness


def _run_case(case: AblationCase, *, smoke: bool) -> dict[str, object]:
    initial_center_m, direction = _indenter_trajectory(
        case.fingertip,
        sphere_diameter_mm=_SPHERE_DIAMETER_MM,
        contact_y_mm=_CONTACT_Y_MM,
        fingertip_angle_deg=_CONTACT_ANGLE_DEG,
        initial_clearance_m=_INITIAL_CLEARANCE_M,
    )
    zero_contact_travel_m = _zero_contact_travel_m(
        case.fingertip,
        sphere_diameter_mm=_SPHERE_DIAMETER_MM,
        fingertip_angle_deg=_CONTACT_ANGLE_DEG,
        initial_clearance_m=_INITIAL_CLEARANCE_M,
    )
    initial_tf = wp.transform(wp.vec3(*initial_center_m), wp.quat_identity())
    sphere_resource = files("lumo").joinpath(
        "assets", "objects", "urdf", "sphere_20mm.urdf"
    )
    start_s = perf_counter()
    with as_file(sphere_resource) as urdf_path:
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
        indenter = Indenter.add_urdf(
            builder,
            urdf_path,
            tf=initial_tf,
            contact_stiffness_n_m=_INDENTER_STIFFNESS_N_M,
            contact_damping_n_s_m=_INDENTER_DAMPING_N_S_M,
        )
        fingertip_model = build_fingertip_newton_model(
            case.mesh,
            builder=builder,
            carrier_interaction=case.carrier_interaction,
            bonded_vertex_indices=case.bonded_vertex_indices,
        )
        simulation = LumoSimulation(
            case.fingertip,
            fingertip_model=fingertip_model,
            sim_frequency=_SIM_FREQUENCY_HZ,
            iterations=_VBD_ITERATIONS,
            soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
            soft_contact_stiffness_n_m=_INDENTER_STIFFNESS_N_M,
            soft_contact_damping_n_s_m=_INDENTER_DAMPING_N_S_M,
        )
        simulation.apply_indenter_pose(indenter, initial_tf)
        simulation.collision_pipeline.collide(simulation.state, simulation.contacts)
        if simulation.soft_contact_count(indenter.body_index):
            raise RuntimeError(f"{case.name} starts with external soft contacts")

        reference_vertices = np.asarray(
            case.mesh.silicone.vertices,
            dtype=np.float64,
        )
        tetrahedra = np.asarray(
            case.mesh.silicone.tet_indices,
            dtype=np.int32,
        ).reshape(-1, 4)
        surface_triangles = np.asarray(
            case.mesh.silicone.surface_tri_indices,
            dtype=np.int32,
        ).reshape(-1, 3)
        reference_volumes = _six_tet_volumes(reference_vertices, tetrahedra)
        incidence = _surface_incidence(surface_triangles)
        fixed_interface_area_m2 = _fixed_interface_area_m2(case)

        target_schedule = _FORCE_TARGETS_N[:2] if smoke else _FORCE_TARGETS_N
        force_samples: list[float] = []
        indentation_samples: list[float] = []
        external_area_samples: list[float] = []
        internal_area_samples: list[float] = []
        minimum_det_samples: list[float] = []
        total_contact_samples: list[int] = []
        external_count_samples: list[int] = []
        internal_count_samples: list[int] = []
        overflow_samples: list[int] = []
        step_samples: list[int] = []
        checkpoint_forces: list[float] = []
        checkpoint_indentations: list[float] = []
        checkpoint_vertices: list[np.ndarray] = []
        checkpoint_contact_indices: list[np.ndarray] = []
        checkpoint_contact_normals: list[np.ndarray] = []
        checkpoint_contact_positions: list[np.ndarray] = []
        checkpoint_travel: list[float] = []
        next_target = 0
        travel_m = 0.0
        max_steps = int(_MAX_SIM_TIME_S * _SIM_FREQUENCY_HZ)
        for _ in range(max_steps):
            travel_m += _APPROACH_SPEED_M_S / _SIM_FREQUENCY_HZ
            translation = initial_center_m + travel_m * direction
            simulation.apply_indenter_pose(
                indenter,
                wp.transform(wp.vec3(*translation), wp.quat_identity()),
            )
            simulation.step()
            reaction_force_n = simulation.indenter_reaction_force(
                indenter,
                motion_direction_W=wp.vec3(*direction),
            )
            indentation_m = travel_m - zero_contact_travel_m
            vertices_m = simulation.silicone_vertices().astype(np.float64)
            external = _body_contact_records(
                simulation,
                indenter.body_index,
                vertices_m,
            )
            internal = _body_contact_records(
                simulation,
                fingertip_model.carrier_body,
                vertices_m,
            )
            external_area_m2 = _patch_area_m2(
                vertices_m,
                surface_triangles,
                external[0],
                incidence,
            )
            internal_area_m2 = (
                fixed_interface_area_m2
                if case.name == "bonded_t"
                else _patch_area_m2(
                    vertices_m,
                    surface_triangles,
                    internal[0],
                    incidence,
                )
            )
            det_f = _six_tet_volumes(vertices_m, tetrahedra) / reference_volumes
            overflow = int(
                simulation.solver.body_particle_contact_overflow_max.numpy()[0]
            )
            if indentation_m >= -1.0e-4 or reaction_force_n > 0.0:
                step_samples.append(simulation.step_count)
                force_samples.append(reaction_force_n)
                indentation_samples.append(indentation_m)
                external_area_samples.append(external_area_m2)
                internal_area_samples.append(internal_area_m2)
                minimum_det_samples.append(float(det_f.min()))
                total_contact_samples.append(simulation.soft_contact_count())
                external_count_samples.append(len(external[0]))
                internal_count_samples.append(len(internal[0]))
                overflow_samples.append(overflow)

            if (
                next_target < len(target_schedule)
                and reaction_force_n >= target_schedule[next_target]
            ):
                if len(external[0]) == 0:
                    raise RuntimeError(
                        f"{case.name} crossed force without external contacts"
                    )
                checkpoint_forces.append(reaction_force_n)
                checkpoint_indentations.append(indentation_m)
                checkpoint_vertices.append(vertices_m.astype(np.float32))
                checkpoint_contact_indices.append(external[0].copy())
                checkpoint_contact_normals.append(external[1].copy())
                checkpoint_contact_positions.append(external[2].copy())
                checkpoint_travel.append(travel_m)
                next_target += 1
                if next_target == len(target_schedule):
                    break
        else:
            raise RuntimeError(
                f"{case.name} did not reach {target_schedule[-1]:g} N"
            )

        force = np.asarray(force_samples, dtype=np.float64)
        indentation = np.asarray(indentation_samples, dtype=np.float64)
        stiffness = _local_linear_stiffness_n_m(indentation, force)
        result: dict[str, object] = {
            "case": case,
            "simulation": simulation,
            "indenter": indenter,
            "initial_center_m": initial_center_m,
            "direction": direction,
            "zero_contact_travel_m": zero_contact_travel_m,
            "step": np.asarray(step_samples, dtype=np.int64),
            "force_n": force,
            "indentation_m": indentation,
            "external_area_m2": np.asarray(external_area_samples),
            "internal_area_m2": np.asarray(internal_area_samples),
            "incremental_stiffness_n_m": stiffness,
            "minimum_det_f": np.asarray(minimum_det_samples),
            "total_contact_count": np.asarray(total_contact_samples, dtype=np.int32),
            "external_contact_count": np.asarray(external_count_samples, dtype=np.int32),
            "internal_contact_count": np.asarray(internal_count_samples, dtype=np.int32),
            "contact_buffer_overflow": np.asarray(overflow_samples, dtype=np.int32),
            "checkpoint_forces_n": np.asarray(checkpoint_forces),
            "checkpoint_indentations_m": np.asarray(checkpoint_indentations),
            "checkpoint_vertices_m": np.asarray(checkpoint_vertices),
            "checkpoint_contact_indices": checkpoint_contact_indices,
            "checkpoint_contact_normals": checkpoint_contact_normals,
            "checkpoint_contact_positions_m": checkpoint_contact_positions,
            "checkpoint_travel_m": np.asarray(checkpoint_travel),
            "reference_vertices_m": reference_vertices.astype(np.float32),
            "tetrahedra": tetrahedra,
            "surface_triangles": surface_triangles,
            "fixed_interface_area_m2": fixed_interface_area_m2,
            "runtime_s": perf_counter() - start_s,
        }
        if not smoke:
            offsets = np.empty((1, 4, 2), dtype=np.int64)
            index_chunks: list[np.ndarray] = []
            normal_chunks: list[np.ndarray] = []
            cursor = 0
            for force_index, (indices, normals) in enumerate(
                zip(
                    checkpoint_contact_indices,
                    checkpoint_contact_normals,
                    strict=True,
                )
            ):
                offsets[0, force_index] = (cursor, len(indices))
                cursor += len(indices)
                index_chunks.append(indices)
                normal_chunks.append(normals)
            objective = compute_contact_objective(
                reference_vertices_m=reference_vertices,
                surface_triangles=surface_triangles,
                scenario_names=(
                    f"sphere_{_SPHERE_DIAMETER_MM:g}mm_"
                    f"y{_CONTACT_Y_MM:+g}mm_theta{_CONTACT_ANGLE_DEG:+g}deg",
                ),
                sphere_diameters_mm=np.asarray((_SPHERE_DIAMETER_MM,)),
                contact_angles_deg=np.asarray((_CONTACT_ANGLE_DEG,)),
                contact_y_mm=np.asarray((_CONTACT_Y_MM,)),
                force_targets_n=_FORCE_TARGETS_N,
                actual_forces_n=np.asarray(checkpoint_forces)[None, :],
                indentations_m=np.asarray(checkpoint_indentations)[None, :],
                contact_record_offsets=offsets,
                contact_particle_indices=np.concatenate(index_chunks),
                contact_normals_W=np.concatenate(normal_chunks),
                silicone_vertices_m=np.asarray(checkpoint_vertices)[None, ...],
            )
            result["objective"] = objective
        return result


def _stress_colormap():
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "mechanical_stress",
        (
            (0.00, (0.95, 0.94, 0.88)),
            (0.20, (1.00, 0.91, 0.55)),
            (0.38, (0.99, 0.72, 0.27)),
            (0.55, (0.96, 0.40, 0.12)),
            (0.72, (0.82, 0.10, 0.08)),
            (1.00, (0.48, 0.00, 0.06)),
        ),
    )


def _surface_stress_pa(
    fingertip: Fingertip,
    reference: np.ndarray,
    deformed: np.ndarray,
    tetrahedra: np.ndarray,
    surface_triangles: np.ndarray,
) -> np.ndarray:
    dm = np.stack(
        tuple(reference[tetrahedra[:, i]] - reference[tetrahedra[:, 0]] for i in (1, 2, 3)),
        axis=2,
    )
    ds = np.stack(
        tuple(deformed[tetrahedra[:, i]] - deformed[tetrahedra[:, 0]] for i in (1, 2, 3)),
        axis=2,
    )
    deformation = ds @ np.linalg.inv(dm)
    determinant = np.linalg.det(deformation)
    if np.any(determinant <= 0.0):
        raise RuntimeError("cannot render an inverted ablation state")
    mechanics = fingertip.parameters.mechanics
    mu = mechanics.shear_modulus_pa
    lambda_nh = mechanics.lame_lambda_pa + mu
    alpha = 1.0 + mu / lambda_nh
    cofactor = determinant[:, None, None] * np.swapaxes(
        np.linalg.inv(deformation), 1, 2
    )
    first_piola = mu * deformation + (
        lambda_nh * (determinant - alpha)
    )[:, None, None] * cofactor
    cauchy = (
        first_piola @ np.swapaxes(deformation, 1, 2)
    ) / determinant[:, None, None]
    deviatoric = cauchy - np.trace(cauchy, axis1=1, axis2=2)[:, None, None] * np.eye(3)[None, :, :] / 3.0
    tet_stress = np.sqrt(1.5 * np.sum(deviatoric**2, axis=(1, 2)))
    vertex_stress = np.zeros(len(reference))
    counts = np.zeros(len(reference), dtype=np.int32)
    for local_vertex in range(4):
        np.add.at(vertex_stress, tetrahedra[:, local_vertex], tet_stress)
        np.add.at(counts, tetrahedra[:, local_vertex], 1)
    vertex_stress /= np.maximum(counts, 1)
    return vertex_stress[surface_triangles].mean(axis=1)


def _crop_frame(frame: np.ndarray, padding: int = 24) -> np.ndarray:
    background = 0.5 * (
        frame[:, :1].astype(np.float32) + frame[:, -1:].astype(np.float32)
    )
    foreground = np.max(np.abs(frame.astype(np.float32) - background), axis=2) > 10.0
    rows, columns = np.nonzero(foreground)
    if not len(rows):
        return frame
    return frame[
        max(0, rows.min() - padding) : min(frame.shape[0], rows.max() + padding + 1),
        max(0, columns.min() - padding) : min(frame.shape[1], columns.max() + padding + 1),
    ]


def _render_states(
    result: dict[str, object],
    *,
    force_indices: tuple[int, ...] = (1, 3),
) -> tuple[str, ...]:
    case = result["case"]
    simulation = result["simulation"]
    indenter = result["indenter"]
    assert isinstance(case, AblationCase)
    assert isinstance(simulation, LumoSimulation)
    assert isinstance(indenter, Indenter)
    outputs: list[str] = []
    colormap = _stress_colormap()
    norm = matplotlib.colors.PowerNorm(
        gamma=0.30,
        vmin=0.0,
        vmax=_STRESS_MAX_KPA,
        clip=True,
    )
    model = simulation.fingertip_model.model
    if model.shape_color is not None and model.shape_body is not None:
        colors = model.shape_color.numpy()
        bodies = model.shape_body.numpy()
        colors[bodies == indenter.body_index] = (0.44, 0.46, 0.49)
        colors[bodies == simulation.fingertip_model.carrier_body] = (0.25, 0.27, 0.30)
        model.shape_color.assign(colors)

    for force_index in force_indices:
        vertices = result["checkpoint_vertices_m"][force_index]
        travel = float(result["checkpoint_travel_m"][force_index])
        translation = result["initial_center_m"] + travel * result["direction"]
        loaded = wp.array(vertices, dtype=wp.vec3, device=model.device)
        wp.copy(
            simulation.state.particle_q,
            loaded,
            dest_offset=simulation.fingertip_model.silicone_particle_start,
            count=len(vertices),
        )
        simulation.apply_indenter_pose(
            indenter,
            wp.transform(wp.vec3(*translation), wp.quat_identity()),
        )
        stress = _surface_stress_pa(
            case.fingertip,
            result["reference_vertices_m"],
            vertices,
            result["tetrahedra"],
            result["surface_triangles"],
        )
        normalized = norm(1.0e-3 * stress)
        bins = np.minimum(
            (normalized * _STRESS_BINS).astype(np.int32),
            _STRESS_BINS - 1,
        )
        viewer = newton.viewer.ViewerGL(
            width=1000,
            height=800,
            vsync=False,
            headless=True,
        )
        try:
            viewer.set_model(model)
            viewer.show_triangles = False
            viewer.show_particles = False
            viewer.renderer.draw_fps = False
            viewer.renderer.draw_edges = False
            viewer.renderer.sky_upper = (1.0, 1.0, 1.0)
            viewer.renderer.sky_lower = (1.0, 1.0, 1.0)
            viewer.renderer.ambient_sky = (0.82, 0.82, 0.84)
            viewer.renderer.ambient_ground = (0.45, 0.45, 0.47)
            viewer.renderer._exposure = 1.25
            viewer.set_camera(wp.vec3(0.052, -0.105, -0.012), 0.0, 0.0)
            viewer.camera.look_at(wp.vec3(0.0, -0.004, -0.005))
            viewer.camera.fov = 39.0
            contact_positions = wp.array(
                result["checkpoint_contact_positions_m"][force_index],
                dtype=wp.vec3,
                device=model.device,
            )
            contact_colors = wp.full(
                len(contact_positions),
                wp.vec3(0.95, 0.48, 0.02),
                dtype=wp.vec3,
                device=model.device,
            )
            for _ in range(2):
                viewer.begin_frame(0.0)
                viewer.log_state(simulation.state)
                for bin_index in range(_STRESS_BINS):
                    triangles = result["surface_triangles"][bins == bin_index]
                    if len(triangles) == 0:
                        continue
                    indices = wp.array(
                        triangles.reshape(-1)
                        + simulation.fingertip_model.silicone_particle_start,
                        dtype=wp.int32,
                        device=model.device,
                    )
                    color = tuple(
                        float(channel)
                        for channel in colormap((bin_index + 0.5) / _STRESS_BINS)[:3]
                    )
                    viewer.log_mesh(
                        f"/figure3/stress_{bin_index:02d}",
                        simulation.state.particle_q,
                        indices,
                        color=color,
                        roughness=0.82,
                        metallic=0.0,
                        backface_culling=False,
                    )
                viewer.log_points(
                    "/figure3/contact_patch",
                    contact_positions,
                    radii=8.0e-4,
                    colors=contact_colors,
                )
                viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
        finally:
            viewer.close()
        filename = f"newton_{case.name}_{_FORCE_TARGETS_N[force_index]:g}n.png"
        path = _OUTPUT_DIRECTORY / filename
        with publication_context():
            figure = plt.figure(figsize=(3.0, 2.4))
            axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
            axis.imshow(_crop_frame(frame))
            axis.set_axis_off()
            figure.savefig(
                path,
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.0,
                facecolor="white",
                transparent=False,
            )
            plt.close(figure)
        outputs.append(filename)
    return tuple(outputs)


def _serialize(results: tuple[dict[str, object], ...]) -> None:
    arrays: dict[str, np.ndarray] = {
        "case_names": np.asarray(_CASE_NAMES),
        "display_names": np.asarray(tuple(_DISPLAY_NAMES[name] for name in _CASE_NAMES)),
        "force_targets_n": _FORCE_TARGETS_N,
        "morphology_mm": np.asarray(_MORPHOLOGY_MM),
    }
    sample_rows: list[dict[str, object]] = []
    summary_cases: dict[str, object] = {}
    for result in results:
        case = result["case"]
        objective = result["objective"]
        assert isinstance(case, AblationCase)
        prefix = case.name
        for key in (
            "step",
            "force_n",
            "indentation_m",
            "external_area_m2",
            "internal_area_m2",
            "incremental_stiffness_n_m",
            "minimum_det_f",
            "total_contact_count",
            "external_contact_count",
            "internal_contact_count",
            "contact_buffer_overflow",
            "checkpoint_forces_n",
            "checkpoint_indentations_m",
            "checkpoint_vertices_m",
            "checkpoint_travel_m",
            "reference_vertices_m",
            "tetrahedra",
            "surface_triangles",
        ):
            arrays[f"{prefix}_{key}"] = np.asarray(result[key])
        arrays[f"{prefix}_carrier_vertices_m"] = np.asarray(case.mesh.carrier.vertices)
        arrays[f"{prefix}_carrier_triangles"] = np.asarray(
            case.mesh.carrier.indices,
            dtype=np.int32,
        ).reshape(-1, 3)
        arrays[f"{prefix}_q_form"] = np.asarray(objective.q_form[0])
        arrays[f"{prefix}_q_stable"] = np.asarray(objective.q_stable[0])
        arrays[f"{prefix}_q_stiff"] = np.asarray(objective.q_stiff[0])
        arrays[f"{prefix}_q_contact"] = np.asarray(objective.q_contact[0])
        arrays[f"{prefix}_k_early_n_m"] = np.asarray(objective.k_early_n_m[0])
        arrays[f"{prefix}_k_late_n_m"] = np.asarray(objective.k_late_n_m[0])
        arrays[f"{prefix}_fixed_interface_area_m2"] = np.asarray(
            result["fixed_interface_area_m2"]
        )
        for index in range(len(result["force_n"])):
            sample_rows.append(
                {
                    "case": prefix,
                    "step": int(result["step"][index]),
                    "force_n": float(result["force_n"][index]),
                    "indentation_mm": 1.0e3 * float(result["indentation_m"][index]),
                    "external_contact_area_mm2": 1.0e6 * float(result["external_area_m2"][index]),
                    "internal_engagement_area_mm2": 1.0e6 * float(result["internal_area_m2"][index]),
                    "incremental_stiffness_n_mm": 1.0e-3 * float(result["incremental_stiffness_n_m"][index]),
                    "minimum_det_f": float(result["minimum_det_f"][index]),
                    "total_contact_count": int(result["total_contact_count"][index]),
                    "external_contact_count": int(result["external_contact_count"][index]),
                    "internal_contact_count": int(result["internal_contact_count"][index]),
                    "contact_buffer_overflow": int(result["contact_buffer_overflow"][index]),
                }
            )
        internal_area = np.asarray(result["internal_area_m2"])
        force = np.asarray(result["force_n"])
        engaged = np.flatnonzero(internal_area > 0.0)
        summary_cases[prefix] = {
            "construction": case.construction,
            "runtime_s": float(result["runtime_s"]),
            "checkpoint_actual_forces_n": np.asarray(result["checkpoint_forces_n"]).tolist(),
            "checkpoint_indentations_mm": (1.0e3 * np.asarray(result["checkpoint_indentations_m"])).tolist(),
            "J_contact_matched_scenario": float(objective.J_contact),
            "q_form": float(objective.q_form[0]),
            "q_stable": float(objective.q_stable[0]),
            "q_stiff": float(objective.q_stiff[0]),
            "k_early_n_mm": 1.0e-3 * float(objective.k_early_n_m[0]),
            "k_late_n_mm": 1.0e-3 * float(objective.k_late_n_m[0]),
            "minimum_det_f": float(np.min(result["minimum_det_f"])),
            "inversion_count": int(np.count_nonzero(np.asarray(result["minimum_det_f"]) <= 0.0)),
            "contact_buffer_overflow": int(np.max(result["contact_buffer_overflow"])),
            "carrier_engagement_onset_force_n": (
                float(force[engaged[0]]) if len(engaged) and prefix == "lumo" else None
            ),
            "initial_internal_area_mm2": 1.0e6 * float(internal_area[0]),
            "final_internal_area_mm2": 1.0e6 * float(internal_area[-1]),
            "internal_area_change_mm2": 1.0e6
            * float(internal_area[-1] - internal_area[0]),
            "fixed_tied_interface_area_mm2": 1.0e6 * float(result["fixed_interface_area_m2"]),
            "render_paths": list(result["render_paths"]),
        }

    np.savez_compressed(_RESULT_PATH, **arrays)
    with _SAMPLE_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    summary = {
        "study": "Hybrid morphology mechanics and optical ablation",
        "material": (
            "Dragon Skin 10 NV optical campaign with silicone mechanics preset"
        ),
        "reference_campaign": str(_CAMPAIGN_DIRECTORY.relative_to(_ROOT)),
        "reference_trial": _REFERENCE_TRIAL,
        "reference_campaign_J_contact": _CAMPAIGN_J_CONTACT,
        "morphology_parameter_order": [
            "flat_pad_height_mm",
            "semiellipse_height_mm",
            "stem_width_mm",
            "stem_height_mm",
            "void_width_mm",
        ],
        "morphology_mm": list(_MORPHOLOGY_MM),
        "matched_scenario": {
            "sphere_diameter_mm": _SPHERE_DIAMETER_MM,
            "contact_y_mm": _CONTACT_Y_MM,
            "contact_angle_deg": _CONTACT_ANGLE_DEG,
        },
        "loading_protocol": {
            "type": "constant-speed first threshold crossing",
            "force_targets_n": _FORCE_TARGETS_N.tolist(),
            "approach_speed_m_s": _APPROACH_SPEED_M_S,
            "sim_frequency_hz": _SIM_FREQUENCY_HZ,
            "vbd_iterations": _VBD_ITERATIONS,
            "element_size_mm": _ELEMENT_SIZE_MM,
            "soft_contact_margin_m": _SOFT_CONTACT_MARGIN_M,
            "indenter_contact_stiffness_n_m": _INDENTER_STIFFNESS_N_M,
            "indenter_contact_damping_n_s_m": _INDENTER_DAMPING_N_S_M,
        },
        "incremental_stiffness": "centered local linear slope over at most five consecutive raw samples; raw force-displacement is preserved",
        "cases": summary_cases,
    }
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary)


def _optical_mesh(case: AblationCase) -> FingertipMesh:
    if case.name != "bonded_t":
        return case.mesh
    return replace(
        case.mesh,
        bonded_vertex_indices=case.bonded_vertex_indices,
    )


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.shape != second.shape
        or first.size < 2
        or np.ptp(first) <= 0.0
        or np.ptp(second) <= 0.0
    ):
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def _minimum_state_separation(normalized_response: np.ndarray) -> float:
    minimum = float("inf")
    for first in range(len(normalized_response)):
        for second in range(first + 1, len(normalized_response)):
            minimum = min(
                minimum,
                float(
                    np.linalg.norm(
                        normalized_response[first] - normalized_response[second]
                    )
                ),
            )
    return minimum


def _run_optical_ablation(
    cases: tuple[AblationCase, ...],
    *,
    sample_side_count: int,
    smoke: bool,
) -> tuple[dict[str, object], ...]:
    if not _RESULT_PATH.is_file():
        raise FileNotFoundError(f"missing saved mechanics states: {_RESULT_PATH}")
    os.environ.setdefault(
        "OTK_INCLUDE_DIR",
        str(_ROOT.parent / "optix-toolkit" / "ShaderUtil" / "include"),
    )
    with np.load(_RESULT_PATH, allow_pickle=False) as mechanics:
        reference_case = next(case for case in cases if case.name == "lumo")
        reference_scene = OptixScene(_optical_mesh(reference_case))
        leds = _make_leds(reference_case.fingertip)
        emissions = _emissions(
            reference_scene,
            leds,
            sample_side_count=sample_side_count,
        )
        ray_count = len(emissions[0])
        dielectric_branch_u, carrier_u1, carrier_u2 = _optical_samples(ray_count)
        state_indices = (0, 2, 4) if smoke else tuple(range(5))
        results: list[dict[str, object]] = []

        for case in cases:
            prefix = case.name
            saved_reference = np.asarray(
                mechanics[f"{prefix}_reference_vertices_m"],
                dtype=np.float64,
            )
            saved_tetrahedra = np.asarray(
                mechanics[f"{prefix}_tetrahedra"],
                dtype=np.int32,
            )
            mesh_reference = np.asarray(
                case.mesh.silicone.vertices,
                dtype=np.float64,
            )
            mesh_tetrahedra = np.asarray(
                case.mesh.silicone.tet_indices,
                dtype=np.int32,
            ).reshape(-1, 4)
            if not np.array_equal(saved_tetrahedra, mesh_tetrahedra) or not np.allclose(
                saved_reference,
                mesh_reference,
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError(
                    f"{prefix} rebuilt mesh does not match the saved Newton topology"
                )

            checkpoint_vertices = np.asarray(
                mechanics[f"{prefix}_checkpoint_vertices_m"],
                dtype=np.float64,
            )
            all_vertices = np.concatenate(
                (saved_reference[None, ...], checkpoint_vertices),
                axis=0,
            )
            checkpoint_forces = np.asarray(
                mechanics[f"{prefix}_checkpoint_forces_n"],
                dtype=np.float64,
            )
            all_forces = np.concatenate(((0.0,), checkpoint_forces))
            scene = (
                reference_scene
                if prefix == "lumo"
                else OptixScene(
                    _optical_mesh(case),
                    include_carrier=prefix != "soft_only",
                )
            )
            responses = []
            energies = []
            outside_powers = []
            visible_powers = []
            source_inside_fractions = []
            for state_index in state_indices:
                scene.update_silicone(all_vertices[state_index])
                source_inside_fractions.append(
                    tuple(
                        float(
                            np.mean(
                                sources_inside_silicone(
                                    scene,
                                    led,
                                    emission,
                                )
                            )
                        )
                        for led, emission in zip(leds, emissions, strict=True)
                    )
                )
                response, energy, outside, visible = _trace_state(
                    scene,
                    case.fingertip,
                    leds,
                    emissions,
                    dielectric_branch_u=dielectric_branch_u,
                    carrier_u1=carrier_u1,
                    carrier_u2=carrier_u2,
                )
                responses.append(response)
                energies.append(energy)
                outside_powers.append(outside)
                visible_powers.append(visible)

            response_matrix = np.asarray(responses)
            energy_matrix = np.asarray(energies)
            outside_roi_power = np.asarray(outside_powers)
            visible_side_power = np.asarray(visible_powers)
            combined_response = response_matrix.sum(axis=1)
            emitted_index = _ENERGY_FIELDS.index("emitted_power")
            emitted_power = float(energy_matrix[0, :, emitted_index].sum())
            if not np.isclose(emitted_power, 5.0, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{prefix} emitted power is not five")
            if not np.allclose(
                energy_matrix[:, :, emitted_index].sum(axis=1),
                emitted_power,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(f"{prefix} emitted power changes between states")
            closure_index = _ENERGY_FIELDS.index("closure_error")
            maximum_closure_error = float(
                np.max(np.abs(energy_matrix[:, :, closure_index]))
            )
            if maximum_closure_error > 1.0e-12:
                raise RuntimeError(f"{prefix} optical energy ledger does not close")
            if not np.all(np.isfinite(response_matrix)):
                raise RuntimeError(f"{prefix} optical response is non-finite")

            normalized_response = (
                combined_response - combined_response[0]
            ) / emitted_power
            state_distance = np.linalg.norm(normalized_response, axis=1)
            total_visible_power = visible_side_power.sum(axis=1)
            delta_visible_power = total_visible_power - total_visible_power[0]
            relative_visible_power_change = np.divide(
                delta_visible_power,
                total_visible_power[0],
                out=np.zeros_like(delta_visible_power),
                where=total_visible_power[0] != 0.0,
            )
            outside_total = outside_roi_power.sum(axis=1)
            outside_fraction = np.divide(
                outside_total,
                total_visible_power,
                out=np.zeros_like(outside_total),
                where=total_visible_power > 0.0,
            )
            selected_forces = all_forces[np.asarray(state_indices)]
            results.append(
                {
                    "case": case,
                    "state_indices": np.asarray(state_indices, dtype=np.int32),
                    "force_n": selected_forces,
                    "response_matrix": response_matrix,
                    "energy_matrix": energy_matrix,
                    "outside_roi_power": outside_roi_power,
                    "visible_side_power": visible_side_power,
                    "combined_response": combined_response,
                    "normalized_response": normalized_response,
                    "state_distance": state_distance,
                    "total_visible_power": total_visible_power,
                    "delta_visible_power": delta_visible_power,
                    "relative_visible_power_change": relative_visible_power_change,
                    "outside_roi_fraction": outside_fraction,
                    "source_inside_fraction": np.asarray(source_inside_fractions),
                    "d_onset_diagnostic": float(np.min(state_distance[1:])),
                    "minimum_force_state_separation": (
                        _minimum_state_separation(normalized_response)
                    ),
                    "maximum_energy_closure_error": maximum_closure_error,
                    "ray_count_per_led": ray_count,
                }
            )
    return tuple(results)


def _run_effective_gap_sensitivity(
    lumo_case: AblationCase,
    *,
    sample_side_count: int,
    smoke: bool,
) -> tuple[dict[str, object], ...]:
    """Replay fixed LUMO Newton states through controlled recess depths."""
    if not _RESULT_PATH.is_file():
        raise FileNotFoundError(f"missing saved mechanics states: {_RESULT_PATH}")
    os.environ.setdefault(
        "OTK_INCLUDE_DIR",
        str(_ROOT.parent / "optix-toolkit" / "ShaderUtil" / "include"),
    )
    with np.load(_RESULT_PATH, allow_pickle=False) as mechanics:
        reference_vertices = np.asarray(
            mechanics["lumo_reference_vertices_m"],
            dtype=np.float64,
        )
        checkpoint_vertices = np.asarray(
            mechanics["lumo_checkpoint_vertices_m"],
            dtype=np.float64,
        )
        all_vertices = np.concatenate(
            (reference_vertices[None, ...], checkpoint_vertices),
            axis=0,
        )
        checkpoint_forces = np.asarray(
            mechanics["lumo_checkpoint_forces_n"],
            dtype=np.float64,
        )
        all_forces = np.concatenate(((0.0,), checkpoint_forces))
        state_indices = (0, 2, 4) if smoke else tuple(range(5))
        results: list[dict[str, object]] = []

        for label, gap_mm in zip(
            _GAP_LABELS,
            _EFFECTIVE_GAPS_MM,
            strict=True,
        ):
            mesh = _mesh_with_effective_gap(lumo_case, gap_mm)
            if not np.array_equal(
                np.asarray(mesh.silicone.tet_indices, dtype=np.int32).reshape(-1, 4),
                np.asarray(
                    mechanics["lumo_tetrahedra"],
                    dtype=np.int32,
                ),
            ) or not np.allclose(
                np.asarray(mesh.silicone.vertices, dtype=np.float64),
                reference_vertices,
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError("gap replay changed the saved silicone mesh")

            scene = OptixScene(mesh)
            leds = _leds_for_effective_gap(lumo_case.fingertip, gap_mm)
            led_source_centers_m = np.asarray(
                tuple(led.position_W_m for led in leds),
                dtype=np.float64,
            )
            cavity_bottom_z_m = -1.0e-3 * (
                lumo_case.fingertip.parameters.geometry.stem_height_mm
            )
            realized_gap_mm = 1.0e3 * (
                led_source_centers_m[:, 2] - cavity_bottom_z_m
            )
            if not np.allclose(
                realized_gap_mm,
                gap_mm,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError("rebuilt LED-to-silicone gap is incorrect")
            emissions = _emissions(
                scene,
                leds,
                sample_side_count=sample_side_count,
            )
            ray_count = len(emissions[0])
            dielectric_branch_u, carrier_u1, carrier_u2 = _optical_samples(
                ray_count
            )
            responses = []
            energies = []
            outside_powers = []
            visible_powers = []
            source_inside_fractions = []
            for state_index in state_indices:
                scene.update_silicone(all_vertices[state_index])
                source_inside_fractions.append(
                    tuple(
                        float(
                            np.mean(
                                sources_inside_silicone(
                                    scene,
                                    led,
                                    emission,
                                )
                            )
                        )
                        for led, emission in zip(leds, emissions, strict=True)
                    )
                )
                response, energy, outside, visible = _trace_state(
                    scene,
                    lumo_case.fingertip,
                    leds,
                    emissions,
                    dielectric_branch_u=dielectric_branch_u,
                    carrier_u1=carrier_u1,
                    carrier_u2=carrier_u2,
                )
                responses.append(response)
                energies.append(energy)
                outside_powers.append(outside)
                visible_powers.append(visible)

            response_matrix = np.asarray(responses)
            energy_matrix = np.asarray(energies)
            outside_roi_power = np.asarray(outside_powers)
            visible_side_power = np.asarray(visible_powers)
            combined_response = response_matrix.sum(axis=1)
            emitted_index = _ENERGY_FIELDS.index("emitted_power")
            emitted_power = float(energy_matrix[0, :, emitted_index].sum())
            closure_index = _ENERGY_FIELDS.index("closure_error")
            maximum_closure_error = float(
                np.max(np.abs(energy_matrix[:, :, closure_index]))
            )
            if not np.isclose(emitted_power, 5.0, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{label} gap emitted power is not five")
            if maximum_closure_error > 1.0e-12:
                raise RuntimeError(f"{label} gap energy ledger does not close")
            if not np.all(np.isfinite(response_matrix)):
                raise RuntimeError(f"{label} gap optical response is non-finite")

            normalized_response = (
                combined_response - combined_response[0]
            ) / emitted_power
            state_distance = np.linalg.norm(normalized_response, axis=1)
            total_visible_power = visible_side_power.sum(axis=1)
            delta_visible_power = total_visible_power - total_visible_power[0]
            relative_visible_power_change = np.divide(
                delta_visible_power,
                total_visible_power[0],
                out=np.zeros_like(delta_visible_power),
                where=total_visible_power[0] != 0.0,
            )
            outside_total = outside_roi_power.sum(axis=1)
            outside_fraction = np.divide(
                outside_total,
                total_visible_power,
                out=np.zeros_like(outside_total),
                where=total_visible_power > 0.0,
            )
            results.append(
                {
                    "label": label,
                    "effective_gap_mm": gap_mm,
                    "realized_gap_mm": realized_gap_mm,
                    "led_source_centers_m": led_source_centers_m,
                    "force_n": all_forces[np.asarray(state_indices)],
                    "response_matrix": response_matrix,
                    "energy_matrix": energy_matrix,
                    "outside_roi_power": outside_roi_power,
                    "visible_side_power": visible_side_power,
                    "combined_response": combined_response,
                    "normalized_response": normalized_response,
                    "state_distance": state_distance,
                    "total_visible_power": total_visible_power,
                    "delta_visible_power": delta_visible_power,
                    "relative_visible_power_change": relative_visible_power_change,
                    "outside_roi_fraction": outside_fraction,
                    "source_inside_fraction": np.asarray(
                        source_inside_fractions
                    ),
                    "minimum_force_state_separation": (
                        _minimum_state_separation(normalized_response)
                    ),
                    "maximum_energy_closure_error": maximum_closure_error,
                    "ray_count_per_led": ray_count,
                }
            )
    return tuple(results)


def _extend_results_with_optics(
    optical_results: tuple[dict[str, object], ...],
    gap_results: tuple[dict[str, object], ...],
) -> None:
    with np.load(_RESULT_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}

    optical_rows: list[dict[str, object]] = []
    case_summaries: dict[str, object] = {}
    for result in optical_results:
        case = result["case"]
        assert isinstance(case, AblationCase)
        prefix = case.name
        for name in (
            "force_n",
            "response_matrix",
            "energy_matrix",
            "outside_roi_power",
            "visible_side_power",
            "combined_response",
            "normalized_response",
            "state_distance",
            "total_visible_power",
            "delta_visible_power",
            "relative_visible_power_change",
            "outside_roi_fraction",
            "source_inside_fraction",
        ):
            arrays[f"{prefix}_optical_{name}"] = np.asarray(result[name])
        arrays[f"{prefix}_optical_d_onset_diagnostic"] = np.asarray(
            result["d_onset_diagnostic"]
        )
        arrays[f"{prefix}_optical_minimum_force_state_separation"] = np.asarray(
            result["minimum_force_state_separation"]
        )
        arrays[f"{prefix}_optical_maximum_energy_closure_error"] = np.asarray(
            result["maximum_energy_closure_error"]
        )
        arrays[f"{prefix}_J_obs"] = np.asarray(np.nan)

        force_n = np.asarray(result["force_n"])
        total_visible = np.asarray(result["total_visible_power"])
        state_distance = np.asarray(result["state_distance"])
        delta_visible = np.asarray(result["delta_visible_power"])
        relative_visible = np.asarray(result["relative_visible_power_change"])
        outside_fraction = np.asarray(result["outside_roi_fraction"])
        response = np.asarray(result["combined_response"])
        for state_index in range(len(force_n)):
            row: dict[str, object] = {
                "case": prefix,
                "state_index": state_index,
                "force_n": float(force_n[state_index]),
                "total_visible_power": float(total_visible[state_index]),
                "delta_visible_power": float(delta_visible[state_index]),
                "relative_visible_power_change": float(
                    relative_visible[state_index]
                ),
                "normalized_state_distance": float(state_distance[state_index]),
                "outside_roi_fraction": float(outside_fraction[state_index]),
            }
            for bin_index, value in enumerate(response[state_index]):
                row[f"side_bin_{bin_index + 1:02d}"] = float(value)
            optical_rows.append(row)
        case_summaries[prefix] = {
            "geometry_mode": case.construction,
            "J_obs": None,
            "J_obs_defined": False,
            "J_obs_reason": (
                "production J_obs requires at least two contact-Y locations at "
                "matched sphere, angle, and force; this controlled ablation "
                "contains one saved contact location"
            ),
            "active_production_components": (
                "J_obs plus d_onset diagnostic; no separate intensity or spatial "
                "sub-objectives are active"
            ),
            "d_onset_diagnostic": float(result["d_onset_diagnostic"]),
            "minimum_force_state_separation_diagnostic": float(
                result["minimum_force_state_separation"]
            ),
            "unloaded_visible_power": float(total_visible[0]),
            "ten_n_visible_power": float(total_visible[-1]),
            "ten_n_delta_visible_power": float(delta_visible[-1]),
            "ten_n_relative_visible_power_change": float(relative_visible[-1]),
            "ten_n_state_distance": float(state_distance[-1]),
            "maximum_state_distance": float(np.max(state_distance)),
            "maximum_absolute_visible_power_change": float(
                np.max(np.abs(delta_visible))
            ),
            "state_distance_monotonic_non_decreasing": bool(
                np.all(np.diff(state_distance) >= -1.0e-12)
            ),
            "maximum_outside_roi_fraction": float(np.max(outside_fraction)),
            "maximum_energy_closure_error": float(
                result["maximum_energy_closure_error"]
            ),
            "source_inside_fraction_by_state_and_led": np.asarray(
                result["source_inside_fraction"]
            ).tolist(),
            "representative_ray_render_paths": [],
        }

    arrays["optical_energy_fields"] = np.asarray(_ENERGY_FIELDS)
    arrays["optical_led_source_centers_m"] = np.asarray(
        _reference_fingertip().led_source_centers_m,
        dtype=np.float64,
    )
    arrays["optical_ray_count_per_led"] = np.asarray(
        optical_results[0]["ray_count_per_led"]
    )
    temporary_path = _RESULT_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(_RESULT_PATH)

    with _OPTICAL_SAMPLE_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(optical_rows[0]))
        writer.writeheader()
        writer.writerows(optical_rows)

    gap_rows: list[dict[str, object]] = []
    gap_summaries: dict[str, object] = {}
    arrays["gap_sensitivity_effective_gap_mm"] = np.asarray(
        tuple(result["effective_gap_mm"] for result in gap_results),
        dtype=np.float64,
    )
    arrays["gap_sensitivity_labels"] = np.asarray(
        tuple(result["label"] for result in gap_results)
    )
    for result in gap_results:
        label = str(result["label"])
        prefix = f"gap_{label}"
        for name in (
            "force_n",
            "response_matrix",
            "energy_matrix",
            "outside_roi_power",
            "visible_side_power",
            "combined_response",
            "normalized_response",
            "state_distance",
            "total_visible_power",
            "delta_visible_power",
            "relative_visible_power_change",
            "outside_roi_fraction",
            "source_inside_fraction",
        ):
            arrays[f"{prefix}_{name}"] = np.asarray(result[name])
        arrays[f"{prefix}_maximum_energy_closure_error"] = np.asarray(
            result["maximum_energy_closure_error"]
        )
        arrays[f"{prefix}_realized_gap_mm"] = np.asarray(
            result["realized_gap_mm"]
        )
        arrays[f"{prefix}_led_source_centers_m"] = np.asarray(
            result["led_source_centers_m"]
        )
        arrays[f"{prefix}_J_obs"] = np.asarray(np.nan)

        force_n = np.asarray(result["force_n"], dtype=np.float64)
        state_distance = np.asarray(result["state_distance"], dtype=np.float64)
        total_visible = np.asarray(result["total_visible_power"], dtype=np.float64)
        delta_visible = np.asarray(result["delta_visible_power"], dtype=np.float64)
        relative_visible = np.asarray(
            result["relative_visible_power_change"],
            dtype=np.float64,
        )
        outside_fraction = np.asarray(
            result["outside_roi_fraction"],
            dtype=np.float64,
        )
        response = np.asarray(result["combined_response"], dtype=np.float64)
        for state_index in range(len(force_n)):
            row: dict[str, object] = {
                "gap_label": label,
                "effective_gap_mm": float(result["effective_gap_mm"]),
                "state_index": state_index,
                "force_n": float(force_n[state_index]),
                "total_visible_power": float(total_visible[state_index]),
                "delta_visible_power": float(delta_visible[state_index]),
                "relative_visible_power_change": float(
                    relative_visible[state_index]
                ),
                "normalized_state_distance": float(state_distance[state_index]),
                "outside_roi_fraction": float(outside_fraction[state_index]),
            }
            for bin_index, value in enumerate(response[state_index]):
                row[f"side_bin_{bin_index + 1:02d}"] = float(value)
            gap_rows.append(row)
        gap_summaries[label] = {
            "effective_gap_mm": float(result["effective_gap_mm"]),
            "realized_gap_mm_by_led": np.asarray(
                result["realized_gap_mm"]
            ).tolist(),
            "controlled_state": (
                "nominal LUMO unloaded and 1/2/5/10 N silicone vertices held fixed; "
                "carrier recess floor and LED source plane rebuilt together"
            ),
            "J_obs": None,
            "J_obs_defined": False,
            "J_obs_reason": (
                "production J_obs requires at least two contact-Y locations; "
                "this controlled sensitivity contains one location"
            ),
            "unloaded_visible_power": float(total_visible[0]),
            "one_n_state_distance": float(state_distance[1]),
            "two_n_state_distance": float(state_distance[2]),
            "ten_n_state_distance": float(state_distance[-1]),
            "one_n_delta_visible_power": float(delta_visible[1]),
            "two_n_delta_visible_power": float(delta_visible[2]),
            "ten_n_delta_visible_power": float(delta_visible[-1]),
            "one_n_relative_visible_power_change": float(relative_visible[1]),
            "two_n_relative_visible_power_change": float(relative_visible[2]),
            "ten_n_relative_visible_power_change": float(relative_visible[-1]),
            "minimum_force_state_separation_diagnostic": float(
                result["minimum_force_state_separation"]
            ),
            "maximum_outside_roi_fraction": float(np.max(outside_fraction)),
            "maximum_energy_closure_error": float(
                result["maximum_energy_closure_error"]
            ),
            "source_inside_fraction_by_state_and_led": np.asarray(
                result["source_inside_fraction"]
            ).tolist(),
        }
    with _GAP_SAMPLE_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(gap_rows[0]))
        writer.writeheader()
        writer.writerows(gap_rows)

    summary = json.loads(_SUMMARY_PATH.read_text())
    summary["study"] = "Hybrid morphology mechanics and optical ablation"
    mechanics_parameters = _SCIENTIFIC_CONTRACT["fingertip_parameters"][
        "mechanics"
    ]
    optics_parameters = _SCIENTIFIC_CONTRACT["fingertip_parameters"]["optics"]
    summary["material"] = (
        "Dragon Skin 10 NV optical campaign with silicone mechanics preset"
    )
    summary["material_parameters"] = {
        "mechanics_preset": _MECHANICS_PRESET,
        "mechanics": mechanics_parameters,
        "optical_preset": _OPTICAL_PRESET,
        "optics": optics_parameters,
    }
    summary["optics"] = {
        "pipeline": "production OptiX finite-area five-LED path tracing",
        "ray_count_per_led": int(optical_results[0]["ray_count_per_led"]),
        "max_bounces": _MAX_BOUNCES,
        "carrier_albedo": _CARRIER_ALBEDO,
        "led_source_centers_m": np.asarray(
            _reference_fingertip().led_source_centers_m
        ).tolist(),
        "state_sequence": "unloaded plus exact saved 1/2/5/10 N Newton states",
        "source_control": (
            "one common production emission array generated on the LUMO carrier "
            "recess and replayed unchanged in all cases"
        ),
        "source_mechanics_state_path": str(_RESULT_PATH.relative_to(_ROOT)),
        "soft_only_geometry": (
            "silicone outer body only; carrier IAS instance omitted; virtual LED "
            "sources remain at the common LUMO coordinates"
        ),
        "bonded_t_geometry": (
            "conformal bonded silicone mesh plus the unchanged T-carrier; fully "
            "tied interface triangles excluded from the silicone optical surface"
        ),
        "lumo_geometry": (
            "production silicone and carrier surfaces with load-updated silicone GAS"
        ),
        "J_obs_defined": False,
        "J_obs_reason": (
            "production J_obs is a same-force separation between distinct "
            "contact-Y locations; the saved ablation has only one location"
        ),
        "cases": case_summaries,
    }
    summary["effective_gap_sensitivity"] = {
        "scope": (
            "targeted optical sensitivity only; not a BO variable and not a "
            "mechanics re-evaluation"
        ),
        "production_search_space": (
            "void width is optimized; h_void is absent/fixed zero; the nominal "
            "0.19 mm gap is a fabrication-informed LED recess depth"
        ),
        "controlled_quantities": (
            "same nominal LUMO silicone states, material, detector, finite-area "
            "LED model, power, ray count, and deterministic optical samples"
        ),
        "effective_gap_values_mm": list(_EFFECTIVE_GAPS_MM),
        "cases": gap_summaries,
    }

    lumo_optical = next(
        result
        for result in optical_results
        if isinstance(result["case"], AblationCase)
        and result["case"].name == "lumo"
    )
    dense_force = np.asarray(arrays["lumo_force_n"], dtype=np.float64)
    dense_area = 1.0e6 * np.asarray(
        arrays["lumo_internal_area_m2"],
        dtype=np.float64,
    )
    optical_force = np.asarray(lumo_optical["force_n"], dtype=np.float64)
    engagement_area = np.empty_like(optical_force)
    engagement_area[0] = dense_area[0]
    for index in range(1, len(optical_force)):
        nearest = int(np.argmin(np.abs(dense_force - optical_force[index])))
        engagement_area[index] = dense_area[nearest]
    engagement_change = engagement_area - engagement_area[0]
    lumo_distance = np.asarray(lumo_optical["state_distance"], dtype=np.float64)
    lumo_visible_delta = np.asarray(
        lumo_optical["delta_visible_power"],
        dtype=np.float64,
    )
    arrays["lumo_optical_engagement_area_mm2"] = engagement_area
    arrays["lumo_optical_engagement_area_change_mm2"] = engagement_change
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(_RESULT_PATH)
    summary["mechanics_optics_correlation"] = {
        "scope": "five LUMO states only; descriptive correlation, not causality",
        "force_n": optical_force.tolist(),
        "internal_engagement_area_mm2": engagement_area.tolist(),
        "internal_engagement_area_change_mm2": engagement_change.tolist(),
        "normalized_optical_state_distance": lumo_distance.tolist(),
        "visible_power_change": lumo_visible_delta.tolist(),
        "pearson_engagement_vs_state_distance": _pearson(
            engagement_change,
            lumo_distance,
        ),
        "pearson_engagement_vs_visible_power_change": _pearson(
            engagement_change,
            lumo_visible_delta,
        ),
    }
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary)


def _write_report(summary: dict[str, object]) -> None:
    cases = summary["cases"]
    assert isinstance(cases, dict)
    soft = cases["soft_only"]
    bonded = cases["bonded_t"]
    lumo = cases["lumo"]
    assert isinstance(soft, dict)
    assert isinstance(bonded, dict)
    assert isinstance(lumo, dict)

    rerun_difference_percent = 100.0 * (
        float(lumo["J_contact_matched_scenario"]) - _CAMPAIGN_J_CONTACT
    ) / _CAMPAIGN_J_CONTACT

    def row(name: str, case: dict[str, object]) -> str:
        return (
            f"| {name} | {float(case['J_contact_matched_scenario']):.6f} | "
            f"{float(case['q_form']):.4f} | {float(case['q_stable']):.4f} | "
            f"{float(case['q_stiff']):.4f} | "
            f"{float(case['k_early_n_mm']):.3f} | "
            f"{float(case['k_late_n_mm']):.3f} |\n"
        )

    text = (
        "# Hybrid morphology mechanics and optical ablation\n\n"
        "## Reference design and protocol\n\n"
        f"- Material: {summary['material']}.\n"
        f"- Source campaign: `{summary['reference_campaign']}`.\n"
        f"- Mechanics-best valid design: trial {summary['reference_trial']}.\n"
        "- Morphology order: flat-pad height, semiellipse height, stem width, "
        "stem height, void width.\n"
        f"- Morphology [mm]: `{summary['morphology_mm']}`.\n"
        f"- Campaign worst-case J_contact: {_CAMPAIGN_J_CONTACT:.9f}.\n"
        f"- Matched ablation scenario: {_SPHERE_DIAMETER_MM:g} mm sphere, "
        f"Y={_CONTACT_Y_MM:+g} mm, theta={_CONTACT_ANGLE_DEG:+g} deg.\n"
        "- Loading: production 100 Hz, 10 VBD iterations, 5 mm/s, first "
        "threshold crossings at 1/2/5/10 N.\n\n"
        "The ablation uses the current objective definitions: finite external "
        "patch formation at 2 N, external-patch IoU from 2 to 10 N, and the "
        "ratio of 1-to-2 N and 5-to-10 N secant stiffnesses. Dense tick-level "
        "samples are retained independently of those four checkpoints. The "
        "plotted incremental stiffness is a centered local linear fit over at "
        "most five consecutive raw samples; the raw force-displacement samples "
        "are preserved in the CSV and NPZ.\n\n"
        "## Ablation construction\n\n"
        f"- **Soft-only:** {soft['construction']}.\n"
        f"- **Bonded-T:** {bonded['construction']}. Its reference tied surface "
        f"area is {float(bonded['fixed_tied_interface_area_mm2']):.2f} mm^2.\n"
        f"- **LUMO:** {lumo['construction']}.\n\n"
        "LEDs are absent from the mechanics model in all cases. The external "
        "envelope, material, indenter, loading path, solver settings, and "
        "mounting location are otherwise matched.\n\n"
        "## Objective components in the matched scenario\n\n"
        "| Case | J_contact | q_form | q_stable | q_stiff | k_early [N/mm] | "
        "k_late [N/mm] |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + row("Soft-only", soft)
        + row("Bonded-T", bonded)
        + row("LUMO", lumo)
        + "\n## Interpretation\n\n"
        "Soft-only is the weakest of the three matched cases: it has the "
        "lowest patch stability and progressive-stiffening score. Adding a "
        "load-bearing carrier improves both terms. LUMO therefore demonstrates "
        "the intended broad contribution of the hybrid structure relative to "
        "the homogeneous soft pad.\n\n"
        "The stronger claim that evolving contact is better than a fixed tied "
        "interface is **not supported by this matched scenario**. Bonded-T "
        "achieves the largest J_contact, q_stable, q_stiff, and late stiffness. "
        "Its low-preload q_form is slightly below LUMO, but not enough to reverse "
        "the aggregate ordering.\n\n"
        f"The LUMO internal contact metric is already nonzero at the first "
        f"recorded near-zero-load state "
        f"({float(lumo['initial_internal_area_mm2']):.2f} mm^2) and increases "
        f"by {float(lumo['internal_area_change_mm2']):.2f} mm^2 by the end of "
        "the 10 N trajectory. Consequently, this simulation does not exhibit "
        "a clean finite-load onset of carrier engagement; it exhibits evolving "
        "engaged area from an initially contacting state. The figure and paper "
        "should use that narrower wording.\n\n"
        "## Integrity and caveats\n\n"
        f"- Minimum det(F): Soft-only {float(soft['minimum_det_f']):.6f}, "
        f"Bonded-T {float(bonded['minimum_det_f']):.6f}, "
        f"LUMO {float(lumo['minimum_det_f']):.6f}.\n"
        "- All cases have zero inversions and zero contact-buffer overflow.\n"
        f"- The matched LUMO rerun differs from the campaign value by "
        f"{rerun_difference_percent:+.3f}%.\n"
        "- The campaign score is a worst case over all campaign scenarios; the ablation "
        "scores above are single-scenario values computed with the identical "
        "component equations. They must not be presented as replacement "
        "campaign scores.\n"
        "- Bonded-T requires a conformal zero-clearance cavity and therefore "
        "changes the internal silicone boundary while preserving the outer "
        "envelope and carrier. This is the intended physical tied-interface "
        "baseline, not a high-friction or artificially stiff contact.\n"
    )
    optics = summary.get("optics")
    if isinstance(optics, dict):
        optical_cases = optics["cases"]
        correlation = summary["mechanics_optics_correlation"]
        assert isinstance(optical_cases, dict)
        assert isinstance(correlation, dict)
        strongest_distance = max(
            optical_cases,
            key=lambda name: float(
                optical_cases[name]["maximum_state_distance"]
            ),
        )
        strongest_visible = max(
            optical_cases,
            key=lambda name: float(
                optical_cases[name]["maximum_absolute_visible_power_change"]
            ),
        )
        bonded_inside = np.asarray(
            optical_cases["bonded_t"][
                "source_inside_fraction_by_state_and_led"
            ],
            dtype=np.float64,
        )[-1]
        lumo_inside = np.asarray(
            optical_cases["lumo"][
                "source_inside_fraction_by_state_and_led"
            ],
            dtype=np.float64,
        )[-1]
        bonded_led = int(np.argmax(bonded_inside))
        lumo_led = int(np.argmax(lumo_inside))
        text += (
            "\n## Mechanical vs optical ablation\n\n"
            "The exact saved unloaded and 1/2/5/10 N Newton meshes were replayed "
            "through the production five-LED OptiX transport with common finite-"
            "area emissions and common deterministic branch samples. Bonded-T "
            "remains mechanically strongest in the matched scenario. LUMO still "
            "starts with internal carrier contact and its engaged area changes "
            "with load rather than showing a clean first-contact transition.\n\n"
            "| Case | P_visible(0) | Delta P_visible(10 N) | relative change | "
            "D(10 N) | d_onset diagnostic |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for name in _CASE_NAMES:
            case_optics = optical_cases[name]
            text += (
                f"| {_DISPLAY_NAMES[name]} | "
                f"{float(case_optics['unloaded_visible_power']):.6f} | "
                f"{float(case_optics['ten_n_delta_visible_power']):+.6f} | "
                f"{float(case_optics['ten_n_relative_visible_power_change']):+.2%} | "
                f"{float(case_optics['ten_n_state_distance']):.6f} | "
                f"{float(case_optics['d_onset_diagnostic']):.6f} |\n"
            )
        text += (
            "\nHere D(F) is the existing production-normalized response change "
            "`||(y(F)-y(0))/5||_2`, not a new optimization objective. The current "
            "production J_obs cannot be computed for any ablation because it is "
            "defined as same-force separation between distinct contact-Y "
            "locations, while this controlled study contains one saved contact "
            "location. There are no active intensity and spatial sub-objectives "
            "in the current objective implementation; d_onset remains diagnostic.\n\n"
            f"The largest normalized optical state change occurs for "
            f"**{_DISPLAY_NAMES[strongest_distance]}**, while the largest absolute "
            f"side-visible power change occurs for "
            f"**{_DISPLAY_NAMES[strongest_visible]}**. For the five LUMO states, "
            "the Pearson correlation between internal engagement-area change and "
            f"normalized optical state distance is "
            f"{correlation['pearson_engagement_vs_state_distance']}; the "
            "corresponding visible-power correlation is "
            f"{correlation['pearson_engagement_vs_visible_power_change']}. These "
            "five-point correlations are descriptive and do not establish "
            "causality. Every case has a monotonic non-decreasing D(F) over the "
            "five saved states.\n\n"
            "One source-medium transition is especially important for "
            "interpretation: at the Bonded-T 10 N state, "
            f"{100.0 * float(bonded_inside[bonded_led]):.1f}% of LED "
            f"{bonded_led + 1}'s finite-window samples are classified inside "
            f"silicone. LUMO's maximum is {100.0 * float(lumo_inside[lumo_led]):.1f}% "
            f"at LED {lumo_led + 1}. Soft-only embeds every controlled virtual "
            "source in silicone by construction. Bonded-T's large raw optical "
            "change therefore includes physical recess-gap closure and should "
            "not be interpreted as contact-location observability.\n\n"
            "Because J_obs is undefined in this one-location ablation, the study "
            "does not establish a J_contact-J_obs trade-off and must not use a "
            "three-point objective-space panel. Figure 3 can compare raw "
            "mechanical and optical evolution, but a production observability "
            "ranking requires matched saved states at two or more contact "
            "locations.\n"
        )
        gap_study = summary.get("effective_gap_sensitivity")
        if isinstance(gap_study, dict):
            gap_cases = gap_study["cases"]
            assert isinstance(gap_cases, dict)
            strongest_one_n = max(
                gap_cases,
                key=lambda label: float(
                    gap_cases[label]["one_n_state_distance"]
                ),
            )
            nominal_gap = gap_cases["nominal"]
            near_zero_gap = gap_cases["near_zero"]
            large_gap = gap_cases["large"]
            text += (
                "\n## Fabrication-informed effective-gap sensitivity\n\n"
                "This is a controlled optical replay, not a new optimization "
                "dimension. The BO search optimized void width but did not "
                "optimize `h_void`; the analytic geometry fixes it at zero. The "
                "production 0.19 mm optical cavity instead comes from the "
                "measured LED-top/carrier-stem recess. For this sensitivity "
                "only, the nominal LUMO silicone vertices were held identical "
                "while the carrier recess floor and LED source plane were "
                "rebuilt together at 0.01, 0.19, and 0.50 mm. Thus the comparison "
                "isolates the optical boundary-condition effect and is not a "
                "claim about mechanically re-equilibrated alternate hardware.\n\n"
                "| Effective gap | D(1 N) | D(2 N) | D(10 N) | "
                "Delta P_visible(1 N) | relative Delta P_visible(1 N) |\n"
                "|---:|---:|---:|---:|---:|---:|\n"
            )
            for label in _GAP_LABELS:
                gap_case = gap_cases[label]
                text += (
                    f"| {float(gap_case['effective_gap_mm']):.2f} mm "
                    f"{'(nominal)' if label == 'nominal' else ''} | "
                    f"{float(gap_case['one_n_state_distance']):.6f} | "
                    f"{float(gap_case['two_n_state_distance']):.6f} | "
                    f"{float(gap_case['ten_n_state_distance']):.6f} | "
                    f"{float(gap_case['one_n_delta_visible_power']):+.6f} | "
                    f"{float(gap_case['one_n_relative_visible_power_change']):+.2%} |\n"
                )
            nominal_is_strongest = strongest_one_n == "nominal"
            near_zero_inside = np.asarray(
                near_zero_gap["source_inside_fraction_by_state_and_led"],
                dtype=np.float64,
            )
            text += (
                "\nThe strongest 1 N normalized optical change in this three-"
                f"point sensitivity is the **{float(gap_cases[strongest_one_n]['effective_gap_mm']):.2f} mm** "
                "case. "
                + (
                    "The fabrication-informed 0.19 mm gap therefore amplifies "
                    "the early response relative to both tested alternatives. "
                    if nominal_is_strongest
                    else "The fabrication-informed 0.19 mm gap is not the "
                    "strongest early-response setting among the tested values. "
                )
                + "Its D(1 N) is "
                f"{float(nominal_gap['one_n_state_distance']):.6f}, compared with "
                f"{float(near_zero_gap['one_n_state_distance']):.6f} at 0.01 mm "
                f"and {float(large_gap['one_n_state_distance']):.6f} at 0.50 mm. "
                "These three samples establish sensitivity, not global "
                "optimality or a binary gap-closure event. In the near-zero "
                "case, source-window samples classified inside silicone rise "
                f"from 0% unloaded to {100.0 * float(np.max(near_zero_inside[1])):.1f}% "
                f"at 1 N and {100.0 * float(np.max(near_zero_inside[2])):.1f}% "
                "at 2 N for the most affected LED. The large early response is "
                "therefore associated with a load-dependent source/interface "
                "medium transition. It should not be interpreted as smooth "
                "location observability, and it shows why a finite fabricated "
                "gap cannot be replaced by source/interface coincidence.\n\n"
                "### Safe Figure 3 takeaway\n\n"
                "Carrier-only mechanical shaping is insufficient to motivate "
                "the complete morphology: the fixed Bonded-T interface is the "
                "strongest matched mechanical baseline, while the production "
                "void-mediated LUMO interface follows a distinct evolving-"
                "engagement trajectory. Independently, changing only the "
                "fabrication-informed recess gap changes the low-load optical "
                "response. The combined evidence supports treating carrier "
                "dimensions, lateral void width, and the fixed small recess gap "
                "as distinct design/physical controls. It does not prove that "
                "0.19 mm is globally optimal, and it does not define J_obs for "
                "this one-location study.\n"
            )
    _REPORT_PATH.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", choices=_CASE_NAMES)
    parser.add_argument("--render-case", choices=_CASE_NAMES)
    parser.add_argument("--render-force", type=int, choices=(2, 10))
    parser.add_argument("--optical-smoke", action="store_true")
    parser.add_argument("--optics", action="store_true")
    args = parser.parse_args()
    if (args.render_case is None) != (args.render_force is None):
        parser.error("--render-case and --render-force must be supplied together")
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    if args.optical_smoke or args.optics:
        if args.optical_smoke and args.optics:
            parser.error("choose --optical-smoke or --optics")
        optical_results = _run_optical_ablation(
            cases,
            sample_side_count=16 if args.optical_smoke else 256,
            smoke=args.optical_smoke,
        )
        lumo_case = next(case for case in cases if case.name == "lumo")
        gap_results = _run_effective_gap_sensitivity(
            lumo_case,
            sample_side_count=16 if args.optical_smoke else 256,
            smoke=args.optical_smoke,
        )
        for result in optical_results:
            case = result["case"]
            assert isinstance(case, AblationCase)
            print(
                f"{case.name}: rays/LED={result['ray_count_per_led']} "
                f"D={np.asarray(result['state_distance']).tolist()} "
                f"closure={result['maximum_energy_closure_error']:.3e}"
            )
        for result in gap_results:
            print(
                f"gap={result['effective_gap_mm']:.2f} mm: "
                f"rays/LED={result['ray_count_per_led']} "
                f"D={np.asarray(result['state_distance']).tolist()} "
                f"closure={result['maximum_energy_closure_error']:.3e}"
            )
        if args.optics:
            _extend_results_with_optics(optical_results, gap_results)
            print(_RESULT_PATH)
            print(_OPTICAL_SAMPLE_PATH)
            print(_GAP_SAMPLE_PATH)
            print(_SUMMARY_PATH)
            print(_REPORT_PATH)
        return
    if args.smoke:
        case = next(case for case in cases if case.name == args.smoke)
        result = _run_case(case, smoke=True)
        print(
            f"{case.name}: force={result['checkpoint_forces_n'][-1]:.6f} N "
            f"min_detF={np.min(result['minimum_det_f']):.6f} "
            f"overflow={np.max(result['contact_buffer_overflow'])}"
        )
        return
    if args.render_case is not None:
        case = next(case for case in cases if case.name == args.render_case)
        result = _run_case(case, smoke=False)
        force_index = 1 if args.render_force == 2 else 3
        for path in _render_states(result, force_indices=(force_index,)):
            print(_OUTPUT_DIRECTORY / path)
        return

    results = tuple(_run_case(case, smoke=False) for case in cases)
    for result in results:
        case = result["case"]
        assert isinstance(case, AblationCase)
        result["render_paths"] = tuple(
            f"newton_{case.name}_{force_n}n.png" for force_n in (2, 10)
        )
    _serialize(results)
    for case_name in _CASE_NAMES:
        for force_n in (2, 10):
            subprocess.run(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--render-case",
                    case_name,
                    "--render-force",
                    str(force_n),
                ),
                check=True,
            )
    print(_RESULT_PATH)
    print(_SAMPLE_PATH)
    print(_SUMMARY_PATH)
    print(_REPORT_PATH)


if __name__ == "__main__":
    main()
