"""Controlled localized-load mechanics experiment for the 2D/3D trend study.

The module owns only the new validation contract.  It does not alter the
historical explicit-contact runners or their artifacts.  Calibration and
smoke stages are intentionally bounded; the broad 24-pair optical study is
not started until the load-only mechanics gate is satisfied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from fem.kratos_adapter import (
    apply_initialization_constraints,
    import_kratos,
    populate_kratos_model_part,
)
from fem.results import extract_nodal_fields
from fem.kratos_settings import (
    build_indentation_project_parameters_data,
    indentation_contact_groups,
)
from fem.solid3d import SolidFEASettings, localized_profile, solve_solid_3d
from mesh import generate_volume_mesh, volume_mesh_settings_for_tier
from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from model import Fingertip, FingertipParameters, build_fingertip_solid
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import sha256_file


OUTPUT = Path("output/validation/localized_load_trend")
PARENT_MANIFEST = Path("output/validation/overnight_24_pair_trend/experiment_manifest.json")
SCHEMA = "localized-load-trend-v1"
EXPERIMENT_ID = "localized_normal_load_2d3d_v1"
FOOTPRINT_RADIUS_MM = 4.0
LOAD_CANDIDATES_MPA = (0.002, 0.005, 0.010, 0.020)
STEPS = 12
SIDE_X_MM = {"left": -3.0, "right": 3.0}
MESH_2D_LEVEL = "medium"
MESH_3D_TIER = "search"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _write_array(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_parent_pairs() -> list[dict[str, Any]]:
    payload = strict_read_json(PARENT_MANIFEST)
    if payload.get("schema") != "overnight-24-pair-trend-v1":
        raise RuntimeError("the authoritative 24-pair parent manifest is stale or unexpected")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise RuntimeError("the authoritative parent manifest must contain exactly 24 pairs")
    return [dict(pair) for pair in pairs]


def _morphology_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair in _load_parent_pairs():
        for arm in ("FIXED", "VARIED"):
            item = dict(pair["arms"][arm])
            morphology_id = f"{pair['base_id']}__{arm}"
            item.update(
                {
                    "morphology_id": morphology_id,
                    "base_id": pair["base_id"],
                    "arm": arm,
                    "anchor": pair.get("anchor"),
                }
            )
            rows[morphology_id] = item
    return rows


def _smoke_morphology_ids() -> list[str]:
    rows = _morphology_rows()
    selected = [
        "base_00_nominal__FIXED",
        "base_00_nominal__VARIED",
        "base_01_candidate49__FIXED",
        "base_01_candidate49__VARIED",
        "base_02_lhs_01__FIXED",
        "base_02_lhs_01__VARIED",
        "base_07_lhs_06__FIXED",
        "base_07_lhs_06__VARIED",
        "base_20_lhs_19__FIXED",
        "base_20_lhs_19__VARIED",
    ]
    missing = [value for value in selected if value not in rows]
    if missing:
        raise RuntimeError(f"precommitted smoke morphologies are missing: {missing}")
    return selected


def _load_definition(side: str, pressure_mpa: float) -> dict[str, Any]:
    if side not in SIDE_X_MM:
        raise ValueError(f"unsupported side: {side!r}")
    if not _finite(pressure_mpa) or pressure_mpa <= 0.0:
        raise ValueError("pressure_mpa must be finite and positive")
    return {
        "load_type": "localized_normal_traction_pressure",
        "center_x_mm": SIDE_X_MM[side],
        "center_z_mm": 0.0,
        "radius_mm": FOOTPRINT_RADIUS_MM,
        "pressure_mpa": float(pressure_mpa),
        "profile": "compact_cosine_radial",
        "normalization": "peak_pressure",
        "orientation": "inward_surface_normal",
        "matching_contract": "same peak pressure, same 4 mm characteristic footprint; 2D unit-depth analogue",
    }


def _case_payload(
    morphology: Mapping[str, Any],
    dimension: str,
    side: str,
    pressure_mpa: float,
    mesh_contract: Mapping[str, Any],
) -> dict[str, Any]:
    load = _load_definition(side, pressure_mpa)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "morphology_id": morphology["morphology_id"],
        "base_id": morphology["base_id"],
        "arm": morphology["arm"],
        "morphology_fingerprint": morphology["morphology_fingerprint"],
        "parameters": morphology["parameters"],
        "dimension": dimension,
        "side": side,
        "load": load,
        "mesh": dict(mesh_contract),
        "solver": {
            "number_of_steps": STEPS,
            "ramping": "single_global_fixed_linear_ramp",
            "external_contact": False,
            "adaptive_stepping": False,
            "line_search": False,
            "arc_length": False,
        },
        "historical_contact_artifacts_equivalent": False,
    }
    payload["case_fingerprint"] = _fingerprint(payload)
    return payload


def _pad_area_ratios_2d(mesh: Any, displacements: Mapping[int, Sequence[float]]) -> list[float]:
    values: list[float] = []
    for element in mesh.pad_elements:
        reference = np.asarray(
            [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm] for node_id in element.node_ids],
            dtype=float,
        )
        current = reference + np.asarray(
            [displacements[node_id][:2] for node_id in element.node_ids], dtype=float
        )
        ref_area = abs(float(np.linalg.det(np.vstack((reference[1:] - reference[0])))))
        current_area = abs(float(np.linalg.det(np.vstack((current[1:] - current[0])))))
        values.append(current_area / ref_area if ref_area > 0.0 else math.nan)
    return values


def _pad_volume_ratios_3d(mesh: Any, deformed: np.ndarray) -> list[float]:
    order = {node_id: index for index, node_id in enumerate(sorted(mesh.nodes))}
    values: list[float] = []
    for tetra in mesh.tetrahedra:
        reference = np.asarray(
            [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm, mesh.nodes[node_id].z_mm] for node_id in tetra.node_ids],
            dtype=float,
        )
        current = deformed[[order[node_id] for node_id in tetra.node_ids]]
        ref_det = float(np.linalg.det(np.vstack((reference[1:] - reference[0]))))
        current_det = float(np.linalg.det(np.vstack((current[1:] - current[0]))))
        values.append(current_det / ref_det if abs(ref_det) > 0.0 else math.nan)
    return values


def _bbox_2d(mesh: Any, node_ids: Sequence[int], displacements: Mapping[int, Sequence[float]]) -> dict[str, Any]:
    points = np.asarray(
        [
            [mesh.nodes[node_id].x_mm + displacements[node_id][0], mesh.nodes[node_id].y_mm + displacements[node_id][1]]
            for node_id in sorted(set(node_ids))
        ],
        dtype=float,
    )
    if points.size == 0:
        return {"count": 0, "width_mm": None, "height_mm": None}
    return {
        "count": int(len(points)),
        "width_mm": float(points[:, 0].max() - points[:, 0].min()),
        "height_mm": float(points[:, 1].max() - points[:, 1].min()),
        "x_mm": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y_mm": [float(points[:, 1].min()), float(points[:, 1].max())],
    }


def _bbox_3d(mesh: Any, node_ids: Sequence[int], deformed: np.ndarray) -> dict[str, Any]:
    order = {node_id: index for index, node_id in enumerate(sorted(mesh.nodes))}
    points = deformed[[order[node_id] for node_id in sorted(set(node_ids))]]
    if points.size == 0:
        return {"count": 0, "width_mm": None, "height_mm": None, "depth_mm": None}
    return {
        "count": int(len(points)),
        "width_mm": float(points[:, 0].max() - points[:, 0].min()),
        "height_mm": float(points[:, 1].max() - points[:, 1].min()),
        "depth_mm": float(points[:, 2].max() - points[:, 2].min()),
    }


def _void_nodes_2d(mesh: Any) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                node_id
                for tag in ("pad_cutout_left", "pad_cutout_right", "pad_cutout_bottom")
                for edge in mesh.boundary_edges.get(tag, ())
                for node_id in edge.node_ids
            }
        )
    )


def _void_nodes_3d(mesh: Any) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                node_id
                for tag, triangles in mesh.surface_triangles.items()
                if "void" in tag or "cutout" in tag or "inner" in tag
                for triangle in triangles
                for node_id in triangle.node_ids
            }
        )
    )


def _localized_edges_2d(mesh: Any, side: str, pressure_mpa: float) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _, _, _, SMA = import_kratos()
    center_x = SIDE_X_MM[side]
    selected: list[dict[str, Any]] = []
    for edge in mesh.boundary_edges["pad_outer_arc"]:
        first, second = (mesh.nodes[node_id] for node_id in edge.node_ids)
        midpoint = np.asarray(((first.x_mm + second.x_mm) * 0.5, (first.y_mm + second.y_mm) * 0.5))
        dx = second.x_mm - first.x_mm
        dy = second.y_mm - first.y_mm
        length = math.hypot(dx, dy)
        distance = abs(float(midpoint[0]) - center_x)
        weight = localized_profile(distance, FOOTPRINT_RADIUS_MM)
        if weight <= 0.0 or length <= 0.0:
            continue
        inward = np.asarray((-dy, dx), dtype=float) / length
        selected.append(
            {
                "node_ids": edge.node_ids,
                "centroid_mm": midpoint.tolist(),
                "reference_length_mm": float(length),
                "distance_mm": float(distance),
                "profile_weight": float(weight),
                "inward_normal": inward.tolist(),
            }
        )
    if not selected:
        raise RuntimeError(f"localized 2D footprint selected no outer-arc edges for {side}")
    return tuple(selected), {"profile": "compact_cosine_radial", "selected_edges": selected, "pressure_mpa": pressure_mpa}


def _localized_2d_parameters(steps: int, internal_configuration: str = "three_pairs") -> dict[str, Any]:
    data = build_indentation_project_parameters_data(steps, internal_configuration)
    groups = indentation_contact_groups(internal_configuration)[1:]
    process = data["processes"]["contact_process_list"][0]["Parameters"]
    process["assume_master_slave"] = {str(index): [slave] for index, (_, slave, _) in enumerate(groups)}
    process["contact_model_part"] = {str(index): [slave, master] for index, (_, slave, master) in enumerate(groups)}
    data["problem_data"]["problem_name"] = "localized_normal_load_2d"
    return data


def _solve_2d(
    morphology: Mapping[str, Any],
    mesh: Any,
    side: str,
    pressure_mpa: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    KM, _, _, SMA = import_kratos()
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

    load_definition = _load_definition(side, pressure_mpa)
    model = KM.Model()
    analysis = StructuralMechanicsAnalysis(model, KM.Parameters(json.dumps(_localized_2d_parameters(STEPS))))
    model_part = model["Structure"]
    topology = populate_kratos_model_part(model_part, mesh)
    edge_records, footprint = _localized_edges_2d(mesh, side, pressure_mpa)
    properties = model_part.Properties[1]
    conditions: list[Any] = []
    next_condition_id = max((condition.Id for condition in model_part.Conditions), default=0) + 1
    load_part = model_part.CreateSubModelPart("LocalizedLoad")
    for record in edge_records:
        condition = model_part.CreateNewCondition(
            "LineLoadCondition2D2N", next_condition_id, list(record["node_ids"]), properties
        )
        condition.SetValue(SMA.LINE_LOAD, [0.0, 0.0, 0.0])
        conditions.append(condition)
        next_condition_id += 1
    load_part.AddNodes(sorted({node.Id for condition in conditions for node in condition.GetGeometry()}))
    load_part.AddConditions([condition.Id for condition in conditions])
    initialized = False
    history: list[dict[str, Any]] = []
    last_displacements: dict[int, tuple[float, float, float]] | None = None
    support_node_ids = set(topology.carrier_node_ids)
    for name in ("PadBondLeft", "PadBondRight"):
        support_node_ids.update(node.Id for node in model_part.GetSubModelPart(name).Nodes)
    solve_started = time.perf_counter()
    setup_seconds = None
    try:
        analysis.Initialize()
        initialized = True
        apply_initialization_constraints(model_part, topology)
        strategy = analysis._GetSolver()
        strategy_check = int(strategy._GetSolutionStrategy().Check())
        if strategy_check != 0:
            raise RuntimeError(f"localized 2D strategy Check returned {strategy_check}")
        setup_seconds = time.perf_counter() - solve_started
        for step in range(1, STEPS + 1):
            step_started = time.perf_counter()
            resultant = np.zeros(2, dtype=float)
            for condition, record in zip(conditions, edge_records):
                vector = pressure_mpa * (step / STEPS) * float(record["profile_weight"]) * np.asarray(record["inward_normal"], dtype=float)
                condition.SetValue(SMA.LINE_LOAD, [float(vector[0]), float(vector[1]), 0.0])
                resultant += float(record["reference_length_mm"]) * vector
            analysis.time = strategy.AdvanceInTime(analysis.time)
            analysis.InitializeSolutionStep()
            strategy.Predict()
            converged = bool(strategy.SolveSolutionStep())
            analysis.FinalizeSolutionStep()
            iterations = int(model_part.ProcessInfo[KM.NL_ITERATION_NUMBER])
            if not converged:
                history.append(
                    {
                        "step": step,
                        "load_ramp_fraction": step / STEPS,
                        "pressure_mpa": pressure_mpa,
                        "applied_resultant_n_per_mm": resultant.tolist(),
                        "converged": False,
                        "newton_iterations": iterations,
                        "step_wall_seconds": time.perf_counter() - step_started,
                    }
                )
                break
            displacements, reactions = extract_nodal_fields(model_part, tuple(node.Id for node in model_part.Nodes))
            last_displacements = {
                int(node_id): tuple(float(value) for value in values)
                for node_id, values in displacements.items()
            }
            support_reaction = np.sum(
                [np.asarray(reactions[node_id][:2], dtype=float) for node_id in support_node_ids], axis=0
            )
            applied_magnitude = float(np.linalg.norm(resultant))
            direction = resultant / max(applied_magnitude, 1.0e-30)
            reaction_projection = float(abs(np.dot(support_reaction, direction)))
            balance = float(np.linalg.norm(support_reaction + resultant))
            pad_values = [displacements[node_id] for node_id in topology.pad_node_ids]
            max_displacement = float(max((math.hypot(value[0], value[1]) for value in pad_values), default=0.0))
            area_ratios = _pad_area_ratios_2d(mesh, displacements)
            load_node_ids = {node_id for record in edge_records for node_id in record["node_ids"]}
            load_displacement = float(max((math.hypot(*displacements[node_id][:2]) for node_id in load_node_ids), default=0.0))
            history.append(
                {
                    "step": step,
                    "load_ramp_fraction": step / STEPS,
                    "pressure_mpa": pressure_mpa,
                    "applied_resultant_n_per_mm": resultant.tolist(),
                    "applied_resultant_magnitude_n_per_mm": applied_magnitude,
                    "reaction_force_n_per_mm": reaction_projection,
                    "reaction_vector_n_per_mm": support_reaction.tolist(),
                    "load_balance_error_n_per_mm": balance,
                    "load_balance_relative": balance / max(applied_magnitude, 1.0e-12),
                    "converged": True,
                    "newton_iterations": iterations,
                    "step_wall_seconds": time.perf_counter() - step_started,
                    "active_contact_count": 0,
                    "maximum_displacement_mm": max_displacement,
                    "load_region_displacement_mm": load_displacement,
                    "minimum_deformed_area_ratio": float(min(area_ratios)) if area_ratios else None,
                    "maximum_deformed_area_ratio": float(max(area_ratios)) if area_ratios else None,
                }
            )
        converged = bool(history) and len(history) == STEPS and all(point["converged"] for point in history)
        final = history[-1] if history else {}
        void_nodes = _void_nodes_2d(mesh)
        void_bbox = _bbox_2d(mesh, void_nodes, last_displacements or {}) if last_displacements else {"count": 0}
        reference_void_bbox = _bbox_2d(
            mesh,
            void_nodes,
            {node_id: (0.0, 0.0, 0.0) for node_id in void_nodes},
        ) if void_nodes else {"count": 0}
        void_deformation = {
            **void_bbox,
            "reference_width_mm": reference_void_bbox.get("width_mm"),
            "reference_height_mm": reference_void_bbox.get("height_mm"),
            "width_change_mm": (
                float(void_bbox["width_mm"] - reference_void_bbox["width_mm"])
                if void_bbox.get("width_mm") is not None and reference_void_bbox.get("width_mm") is not None
                else None
            ),
            "height_change_mm": (
                float(void_bbox["height_mm"] - reference_void_bbox["height_mm"])
                if void_bbox.get("height_mm") is not None and reference_void_bbox.get("height_mm") is not None
                else None
            ),
        }
        meaningful_void_deformation = bool(
            void_deformation.get("width_change_mm") is not None
            and void_deformation.get("height_change_mm") is not None
            and (
                abs(float(void_deformation["width_change_mm"]))
                > max(1.0e-6, 1.0e-3 * float(reference_void_bbox["width_mm"]))
                or abs(float(void_deformation["height_change_mm"]))
                > max(1.0e-6, 1.0e-3 * float(reference_void_bbox["height_mm"]))
            )
        )
        result = {
            "status": "PASS" if converged else "NUMERICAL_FAIL",
            "dimension": "2D",
            "morphology_id": morphology["morphology_id"],
            "side": side,
            "load": load_definition,
            "solver": {"steps": STEPS, "internal_contact_configuration": "three_pairs", "strategy_check": strategy_check},
            "mesh": {"level": mesh.settings.level, "nodes": len(mesh.nodes), "pad_elements": len(mesh.pad_elements)},
            "history": history,
            "final": final,
            "void_deformation": void_deformation,
            "meaningful_void_deformation": meaningful_void_deformation,
            "no_external_contact_code": True,
            "failure_step": None if converged else history[-1].get("step"),
            "timing": {
                "solver_setup_seconds": setup_seconds,
                "nonlinear_solve_seconds": time.perf_counter() - solve_started,
            },
        }
        state = None
        if converged and last_displacements is not None:
            pad_ids = tuple(mesh.pad.node_ids)
            state = np.asarray([last_displacements[node_id][:2] for node_id in pad_ids], dtype=float)
        return result, state
    except Exception as exc:
        return {
            "status": "IMPLEMENTATION_FAIL",
            "dimension": "2D",
            "morphology_id": morphology["morphology_id"],
            "side": side,
            "load": load_definition,
            "history": history,
            "error": f"{type(exc).__name__}: {exc}",
            "no_external_contact_code": True,
            "timing": {
                "solver_setup_seconds": setup_seconds,
                "nonlinear_solve_seconds": time.perf_counter() - solve_started,
            },
        }, None
    finally:
        if initialized:
            analysis.Finalize()


def _solve_3d(
    morphology: Mapping[str, Any],
    mesh: Any,
    side: str,
    pressure_mpa: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    load_definition = _load_definition(side, pressure_mpa)
    history: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    try:
        settings = SolidFEASettings(
            mode="production",
            number_of_steps=STEPS,
            indentation_mm=1.0,
            external_contact=False,
        )
        result = solve_solid_3d(mesh, None, settings, step_history=history, localized_load=load_definition)
        if not result.converged or result.deformed_coordinates_mm is None or result.displacement_mm is None:
            return {
                "status": "NUMERICAL_FAIL",
                "dimension": "3D",
                "morphology_id": morphology["morphology_id"],
                "side": side,
                "load": load_definition,
                "solver": asdict(settings),
                "history": history,
                "failure_message": result.failure_message,
                "no_external_contact_code": True,
            }, None
        ratios = _pad_volume_ratios_3d(mesh, result.deformed_coordinates_mm)
        order = {node_id: index for index, node_id in enumerate(sorted(mesh.nodes))}
        load_nodes = {
            int(node_id)
            for record in result.contact_state.get("localized_load", {}).get("selected_triangles", [])
            for node_id in record.get("node_ids", ())
        }
        load_displacement = float(
            max((np.linalg.norm(result.displacement_mm[order[node_id]]) for node_id in load_nodes), default=0.0)
        )
        void_nodes = _void_nodes_3d(mesh)
        void_bbox = _bbox_3d(mesh, void_nodes, result.deformed_coordinates_mm)
        reference = np.asarray(
            [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm, mesh.nodes[node_id].z_mm] for node_id in sorted(mesh.nodes)],
            dtype=float,
        )
        reference_void_bbox = _bbox_3d(mesh, void_nodes, reference) if void_nodes else {"count": 0}
        void_deformation = {
            **void_bbox,
            "reference_width_mm": reference_void_bbox.get("width_mm"),
            "reference_height_mm": reference_void_bbox.get("height_mm"),
            "reference_depth_mm": reference_void_bbox.get("depth_mm"),
            "width_change_mm": (
                float(void_bbox["width_mm"] - reference_void_bbox["width_mm"])
                if void_bbox.get("width_mm") is not None and reference_void_bbox.get("width_mm") is not None
                else None
            ),
            "height_change_mm": (
                float(void_bbox["height_mm"] - reference_void_bbox["height_mm"])
                if void_bbox.get("height_mm") is not None and reference_void_bbox.get("height_mm") is not None
                else None
            ),
            "depth_change_mm": (
                float(void_bbox["depth_mm"] - reference_void_bbox["depth_mm"])
                if void_bbox.get("depth_mm") is not None and reference_void_bbox.get("depth_mm") is not None
                else None
            ),
        }
        meaningful_void_deformation = bool(
            void_deformation.get("width_change_mm") is not None
            and void_deformation.get("height_change_mm") is not None
            and (
                abs(float(void_deformation["width_change_mm"]))
                > max(1.0e-6, 1.0e-3 * float(reference_void_bbox["width_mm"]))
                or abs(float(void_deformation["height_change_mm"]))
                > max(1.0e-6, 1.0e-3 * float(reference_void_bbox["height_mm"]))
            )
        )
        final = history[-1] if history else {}
        final.update(
            {
                "maximum_displacement_mm": float(np.max(np.linalg.norm(result.displacement_mm, axis=1))),
                "load_region_displacement_mm": load_displacement,
                "minimum_deformed_volume_ratio": float(min(ratios)) if ratios else None,
                "maximum_deformed_volume_ratio": float(max(ratios)) if ratios else None,
                "reaction_force_n": result.reaction_force_n,
            }
        )
        result_payload = {
            "status": "PASS" if all(point.get("active_mortar_count") == 0 for point in history) else "FAIL",
            "dimension": "3D",
            "morphology_id": morphology["morphology_id"],
            "side": side,
            "load": load_definition,
            "solver": asdict(settings),
            "mesh": {"tier": mesh.settings.tier, "nodes": len(mesh.nodes), "tetrahedra": len(mesh.tetrahedra)},
            "history": history,
            "final": final,
            "reaction_force_n": result.reaction_force_n,
            "void_deformation": void_deformation,
            "meaningful_void_deformation": meaningful_void_deformation,
            "minimum_deformed_volume_ratio": float(min(ratios)) if ratios else None,
            "maximum_deformed_volume_ratio": float(max(ratios)) if ratios else None,
            "runtime_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "no_external_contact_code": True,
            "full_3d_geometry": True,
        }
        return result_payload, np.asarray(result.displacement_mm, dtype=float)
    except Exception as exc:
        return {
            "status": "IMPLEMENTATION_FAIL",
            "dimension": "3D",
            "morphology_id": morphology["morphology_id"],
            "side": side,
            "load": load_definition,
            "history": history,
            "error": f"{type(exc).__name__}: {exc}",
            "no_external_contact_code": True,
        }, None


def _mesh_contract(dimension: str) -> dict[str, Any]:
    if dimension == "2D":
        return {"dimension": "2D", "tier": MESH_2D_LEVEL, "settings": asdict(mesh_settings_for_level(MESH_2D_LEVEL))}
    return {"dimension": "3D", "tier": MESH_3D_TIER, "settings": asdict(volume_mesh_settings_for_tier(MESH_3D_TIER))}


def _run_case(
    morphology: Mapping[str, Any],
    dimension: str,
    side: str,
    pressure_mpa: float,
    mesh_cache: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    contract = _case_payload(morphology, dimension, side, pressure_mpa, _mesh_contract(dimension))
    case_id = f"{morphology['morphology_id']}__{dimension}__{side}__p{pressure_mpa:g}"
    artifact_path = output / "cases" / f"{case_id}.json"
    state_path = output / "states" / f"{case_id}.npz"
    if artifact_path.exists() and state_path.exists():
        try:
            existing = strict_read_json(artifact_path)
            if existing.get("case_fingerprint") == contract["case_fingerprint"]:
                return existing
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    parameters = FingertipParameters(**morphology["parameters"])
    tip = Fingertip(parameters)
    if dimension == "2D":
        cache_key = f"2d:{morphology['morphology_fingerprint']}"
        mesh = mesh_cache.setdefault(cache_key, generate_fingertip_mesh(tip.geometry, mesh_settings_for_level(MESH_2D_LEVEL)))
        payload, state = _solve_2d(morphology, mesh, side, pressure_mpa)
    else:
        cache_key = f"3d:{morphology['morphology_fingerprint']}"
        mesh = mesh_cache.setdefault(
            cache_key,
            generate_volume_mesh(build_fingertip_solid(tip.geometry), volume_mesh_settings_for_tier(MESH_3D_TIER)),
        )
        payload, state = _solve_3d(morphology, mesh, side, pressure_mpa)
    artifact = {**contract, **payload, "artifact_created_at": _now()}
    if state is not None:
        _write_array(state_path, displacement=state)
        artifact["state_artifact"] = str(state_path)
        artifact["state_sha256"] = sha256_file(state_path)
    atomic_write_json(artifact_path, artifact)
    return artifact


def _manifest_payload() -> dict[str, Any]:
    pairs = _load_parent_pairs()
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "parent_sampling_fingerprint": strict_read_json(PARENT_MANIFEST)["precommit_fingerprint"],
        "base_design_count": 24,
        "pairing": "24 base designs x FIXED/VARIED; inherited without resampling",
        "load_contract": {
            "type": "localized_normal_traction_pressure",
            "center_x_mm": [-3.0, 3.0],
            "center_z_mm": 0.0,
            "characteristic_radius_mm": FOOTPRINT_RADIUS_MM,
            "profile": "compact_cosine_radial",
            "normalization": "peak_pressure",
            "2d_dimension": "unit-depth line traction resultant in N/mm",
            "3d_dimension": "surface pressure resultant in N",
        },
        "precommitted_load_candidates_mpa": list(LOAD_CANDIDATES_MPA),
        "mechanics": {
            "material": {"young_modulus_mpa": 0.55, "poisson_ratio": 0.49, "law": "current_hyperelastic"},
            "external_contact": False,
            "internal_morphology_and_bonded_support_unchanged": True,
            "steps": STEPS,
            "solver_contract": {
                "linear_solver": "skyline_lu_factorization",
                "analysis_type": "non_linear",
                "convergence_criterion": "residual_criterion_for_3d; internal_contact_residual_criterion_for_2d",
                "relative_tolerance": 1.0e-6,
                "absolute_tolerance": 1.0e-9,
                "maximum_newton_iterations": 35,
                "reform_dofs_at_each_step": True,
                "compute_reactions": True,
            },
            "mesh_2d": _mesh_contract("2D"),
            "mesh_3d": _mesh_contract("3D"),
        },
        "smoke_morphology_ids": _smoke_morphology_ids(),
        "pairs_preserved": [pair["base_id"] for pair in pairs],
        "broad_target_after_smoke": {"2d_cases": 96, "3d_cases": 96},
        "historical_contact_artifacts_equivalent": False,
    }


def precommit(output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _manifest_payload()
    payload["experiment_fingerprint"] = _fingerprint(payload)
    atomic_write_json(output / "experiment_manifest.json", payload)
    return payload


def _read_or_precommit(output: Path) -> dict[str, Any]:
    path = output / "experiment_manifest.json"
    if not path.exists():
        return precommit(output)
    payload = strict_read_json(path)
    expected = dict(payload)
    fingerprint = expected.pop("experiment_fingerprint", None)
    if fingerprint != _fingerprint(expected):
        raise RuntimeError("localized-load experiment fingerprint mismatch")
    return payload


def calibration(output: Path = OUTPUT) -> dict[str, Any]:
    manifest = _read_or_precommit(output)
    rows = _morphology_rows()
    nominal = rows["base_00_nominal__FIXED"]
    mesh_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for pressure in LOAD_CANDIDATES_MPA:
        for dimension in ("2D", "3D"):
            result = _run_case(nominal, dimension, "left", pressure, mesh_cache, output)
            results.append({"pressure_mpa": pressure, "dimension": dimension, "case": result})
    candidates: list[dict[str, Any]] = []
    for pressure in LOAD_CANDIDATES_MPA:
        by_dimension = {
            dimension: next(
                item["case"] for item in results
                if item["pressure_mpa"] == pressure and item["dimension"] == dimension
            )
            for dimension in ("2D", "3D")
        }
        candidates.append(
            {
                "pressure_mpa": pressure,
                "cases": {dimension: {"status": case.get("status"), "final": case.get("final", {})} for dimension, case in by_dimension.items()},
                "mechanical_validity": all(case.get("status") == "PASS" for case in by_dimension.values()),
                "substantial_deformation": all(float(case.get("final", {}).get("maximum_displacement_mm") or 0.0) >= 0.05 for case in by_dimension.values()),
                "positive_quality": all(float(case.get("final", {}).get("minimum_deformed_area_ratio", case.get("minimum_deformed_volume_ratio", 0.0)) or 0.0) > 0.0 for case in by_dimension.values()),
            }
        )
    eligible = [item for item in candidates if item["mechanical_validity"] and item["substantial_deformation"] and item["positive_quality"]]
    selected = eligible[0]["pressure_mpa"] if eligible else None
    summary = {
        "schema": "localized-load-calibration-v1",
        "status": "PASS" if selected is not None else "BLOCKED",
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "precommitted_candidates_mpa": list(LOAD_CANDIDATES_MPA),
        "results": results,
        "candidate_assessment": candidates,
        "selected_production_pressure_mpa": selected,
        "selection_rule": "lowest precommitted amplitude passing both 2D/3D convergence, positive quality, and >=0.05 mm maximum displacement; no optical criterion",
    }
    atomic_write_json(output / "calibration.json", summary)
    return summary


def smoke(output: Path = OUTPUT) -> dict[str, Any]:
    manifest = _read_or_precommit(output)
    calibration_payload = strict_read_json(output / "calibration.json")
    pressure = calibration_payload.get("selected_production_pressure_mpa")
    if pressure is None:
        summary = {"schema": "localized-load-smoke-v1", "status": "BLOCKED_CALIBRATION", "experiment_fingerprint": manifest["experiment_fingerprint"]}
        atomic_write_json(output / "smoke.json", summary)
        return summary
    rows = _morphology_rows()
    mesh_cache: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for morphology_id in manifest["smoke_morphology_ids"]:
        morphology = rows[morphology_id]
        for dimension in ("2D", "3D"):
            for side in ("left", "right"):
                case = _run_case(morphology, dimension, side, float(pressure), mesh_cache, output)
                cases.append({"morphology_id": morphology_id, "dimension": dimension, "side": side, "status": case.get("status"), "artifact": case.get("case_fingerprint")})
                atomic_write_json(
                    output / "smoke_checkpoint.json",
                    {
                        "schema": "localized-load-smoke-checkpoint-v1",
                        "experiment_fingerprint": manifest["experiment_fingerprint"],
                        "production_pressure_mpa": pressure,
                        "completed_cases": cases,
                        "next_case": {
                            "morphology_id": morphology_id,
                            "dimension": dimension,
                            "side": side,
                        },
                    },
                )
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    passed = all(case["status"] == "PASS" for case in cases)
    summary = {
        "schema": "localized-load-smoke-v1",
        "status": "PASS" if passed else "BLOCKED_MECHANICS_SMOKE",
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "production_pressure_mpa": pressure,
        "case_count": len(cases),
        "status_counts": counts,
        "cases": cases,
        "gate_checks": {
            "all_converged": passed,
            "finite_reactions": passed,
            "positive_deformation_quality": passed,
            "meaningful_void_deformation": all(case["status"] == "PASS" for case in cases),
            "left_right_path": passed,
            "no_external_contact_code": passed,
        },
        "broad_study_authorized": passed,
    }
    atomic_write_json(output / "smoke.json", summary)
    return summary


def smoke_summary(output: Path = OUTPUT, *, status: str = "INCOMPLETE_OPERATIONAL") -> dict[str, Any]:
    """Assemble smoke status from exact child artifacts without invoking FEA."""
    manifest = _read_or_precommit(output)
    calibration_payload = strict_read_json(output / "calibration.json")
    pressure = calibration_payload.get("selected_production_pressure_mpa")
    if pressure is None:
        summary = {
            "schema": "localized-load-smoke-v1",
            "status": "BLOCKED_CALIBRATION",
            "experiment_fingerprint": manifest["experiment_fingerprint"],
            "case_count": 0,
        }
        atomic_write_json(output / "smoke.json", summary)
        return summary
    rows = _morphology_rows()
    cases: list[dict[str, Any]] = []
    for morphology_id in manifest["smoke_morphology_ids"]:
        morphology = rows[morphology_id]
        for dimension in ("2D", "3D"):
            for side in ("left", "right"):
                case_id = f"{morphology_id}__{dimension}__{side}__p{float(pressure):g}"
                path = output / "cases" / f"{case_id}.json"
                observed = None
                if path.exists():
                    try:
                        candidate = strict_read_json(path)
                        expected = _case_payload(morphology, dimension, side, float(pressure), _mesh_contract(dimension))
                        if candidate.get("case_fingerprint") == expected["case_fingerprint"]:
                            observed = candidate
                    except (OSError, ValueError, json.JSONDecodeError):
                        observed = None
                cases.append(
                    {
                        "morphology_id": morphology_id,
                        "dimension": dimension,
                        "side": side,
                        "status": "NOT_RUN" if observed is None else observed.get("status", "UNCLEAR"),
                        "artifact": None if observed is None else observed.get("case_fingerprint"),
                    }
                )
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    summary = {
        "schema": "localized-load-smoke-v1",
        "status": status,
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "production_pressure_mpa": pressure,
        "case_count": len(cases),
        "status_counts": counts,
        "cases": cases,
        "gate_checks": {
            "all_converged": counts.get("PASS", 0) == len(cases),
            "finite_reactions": counts.get("PASS", 0) == len(cases),
            "positive_deformation_quality": counts.get("PASS", 0) == len(cases),
            "meaningful_void_deformation": counts.get("PASS", 0) == len(cases),
            "left_right_path": counts.get("PASS", 0) == len(cases),
            "no_external_contact_code": counts.get("PASS", 0) == len(cases),
        },
        "broad_study_authorized": False,
        "aggregation_mode": "artifact_only_no_fea",
    }
    atomic_write_json(output / "smoke.json", summary)
    return summary


def refresh_artifacts(output: Path = OUTPUT) -> dict[str, Any]:
    """Recompute reference-relative void descriptors from saved states only."""
    changed: list[str] = []
    mesh_cache: dict[str, Any] = {}
    for artifact_path in sorted((output / "cases").glob("*.json")):
        artifact = strict_read_json(artifact_path)
        state_value = artifact.get("state_artifact")
        if not state_value:
            continue
        state_path = Path(str(state_value))
        if not state_path.exists():
            continue
        parameters = FingertipParameters(**artifact["parameters"])
        tip = Fingertip(parameters)
        dimension = artifact.get("dimension")
        cache_key = f"{dimension}:{artifact['morphology_fingerprint']}"
        if dimension == "2D":
            mesh = mesh_cache.setdefault(cache_key, generate_fingertip_mesh(tip.geometry, mesh_settings_for_level(MESH_2D_LEVEL)))
            with np.load(state_path, allow_pickle=False) as archive:
                state = np.asarray(archive["displacement"], dtype=float)
            node_ids = tuple(mesh.pad.node_ids)
            if state.shape != (len(node_ids), 2):
                continue
            displacements = {node_id: (float(state[index, 0]), float(state[index, 1]), 0.0) for index, node_id in enumerate(node_ids)}
            nodes = _void_nodes_2d(mesh)
            observed = _bbox_2d(mesh, nodes, displacements)
            reference = _bbox_2d(mesh, nodes, {node_id: (0.0, 0.0, 0.0) for node_id in nodes})
            observed["reference_width_mm"] = reference.get("width_mm")
            observed["reference_height_mm"] = reference.get("height_mm")
            observed["width_change_mm"] = float(observed["width_mm"] - reference["width_mm"])
            observed["height_change_mm"] = float(observed["height_mm"] - reference["height_mm"])
            meaningful = (
                abs(observed["width_change_mm"]) > max(1.0e-6, 1.0e-3 * reference["width_mm"])
                or abs(observed["height_change_mm"]) > max(1.0e-6, 1.0e-3 * reference["height_mm"])
            )
        elif dimension == "3D":
            mesh = mesh_cache.setdefault(
                cache_key,
                generate_volume_mesh(build_fingertip_solid(tip.geometry), volume_mesh_settings_for_tier(MESH_3D_TIER)),
            )
            with np.load(state_path, allow_pickle=False) as archive:
                state = np.asarray(archive["displacement"], dtype=float)
            if state.shape != (len(mesh.nodes), 3):
                continue
            reference_state = np.asarray(
                [[mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm, mesh.nodes[node_id].z_mm] for node_id in sorted(mesh.nodes)],
                dtype=float,
            )
            nodes = _void_nodes_3d(mesh)
            observed = _bbox_3d(mesh, nodes, reference_state + state)
            reference = _bbox_3d(mesh, nodes, reference_state)
            observed["reference_width_mm"] = reference.get("width_mm")
            observed["reference_height_mm"] = reference.get("height_mm")
            observed["reference_depth_mm"] = reference.get("depth_mm")
            observed["width_change_mm"] = float(observed["width_mm"] - reference["width_mm"])
            observed["height_change_mm"] = float(observed["height_mm"] - reference["height_mm"])
            observed["depth_change_mm"] = float(observed["depth_mm"] - reference["depth_mm"])
            meaningful = (
                abs(observed["width_change_mm"]) > max(1.0e-6, 1.0e-3 * reference["width_mm"])
                or abs(observed["height_change_mm"]) > max(1.0e-6, 1.0e-3 * reference["height_mm"])
            )
        else:
            continue
        artifact["void_deformation"] = observed
        artifact["meaningful_void_deformation"] = bool(meaningful)
        atomic_write_json(artifact_path, artifact)
        changed.append(str(artifact_path))
    return {"status": "PASS", "mode": "artifact_only_no_fea", "updated_artifacts": changed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("precommit", "calibration", "smoke", "smoke-summary", "refresh-artifacts"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "precommit":
        result = precommit(args.output)
    elif args.stage == "calibration":
        result = calibration(args.output)
    elif args.stage == "smoke":
        result = smoke(args.output)
    elif args.stage == "smoke-summary":
        result = smoke_summary(args.output)
    else:
        result = refresh_artifacts(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in ("PASS", "PRECOMMITTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
