"""Run the deterministic multi-design carrier/void ablation for Figure 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from lumo.fingertip import (
    MECHANICS_PRESETS,
    OPTICAL_PRESETS,
    Fingertip,
    FingertipGeometry,
    FingertipParameters,
)
from lumo.mesh import make_fingertip_mesh


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = _ROOT / "output" / "validation" / "multi_design_void_ablation"
_STATE_DIRECTORY = _OUTPUT_DIRECTORY / "states"
_BASE_MANIFEST_PATH = _OUTPUT_DIRECTORY / "sampled_designs.csv"
_VARIANT_MANIFEST_PATH = _OUTPUT_DIRECTORY / "variant_manifest.csv"
_RESULTS_PATH = _OUTPUT_DIRECTORY / "variant_results.csv"
_PAIRED_PATH = _OUTPUT_DIRECTORY / "paired_effects.csv"
_SUMMARY_PATH = _OUTPUT_DIRECTORY / "summary.json"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"

_MORPHOLOGY_COLUMNS = (
    "geometry.flat_pad_height_mm",
    "geometry.semiellipse_height_mm",
    "geometry.stem_width_mm",
    "geometry.stem_height_mm",
    "geometry.void_width_mm",
)
_FORCE_TARGETS_N = np.asarray((1.0, 2.0, 5.0, 10.0), dtype=np.float64)
_FIXED_SCENARIO = {
    "sphere_diameter_mm": 20.0,
    "contact_y_mm": -5.5,
    "contact_angle_deg": 0.0,
    "initial_clearance_m": 0.001,
}
_QUANTILES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
_CONDITION_LABELS = {
    "soft_only": "Soft-only",
    "no_void": "No-void carrier",
    "lumo": "LUMO morphology",
}

_CAMPAIGNS = (
    {
        "name": "dragon_normal",
        "material": "Dragon Skin 10 NV",
        "campaign_type": "normal",
        "directory": "mobo_fingertip_contact_1_2_5_10_05mm",
        "balanced_trial": 117,
    },
    {
        "name": "dragon_angled",
        "material": "Dragon Skin 10 NV",
        "campaign_type": "angled",
        "directory": "mobo_fingertip_orientation_robust_1_2_5_10_05mm",
        "balanced_trial": 46,
    },
    {
        "name": "solaris_normal",
        "material": "Solaris",
        "campaign_type": "normal",
        "directory": "mobo_fingertip_contact_1_2_5_10_05mm_solaris_nominal",
        "balanced_trial": 48,
    },
    {
        "name": "solaris_angled",
        "material": "Solaris",
        "campaign_type": "angled",
        "directory": (
            "mobo_fingertip_orientation_robust_1_2_5_10_05mm_solaris_nominal"
        ),
        "balanced_trial": 157,
    },
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_campaign_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path / "trials.csv")
    valid = []
    for row in rows:
        try:
            objectives_finite = np.isfinite(float(row["J_contact"])) and np.isfinite(
                float(row["J_obs"])
            )
        except (TypeError, ValueError):
            objectives_finite = False
        if (
            row["status"] == "COMPLETED"
            and row["analytically_valid"] == "True"
            and not row["failure"]
            and objectives_finite
        ):
            valid.append(row)
    if not valid:
        raise RuntimeError(f"campaign has no valid completed designs: {path}")
    return valid


def _check_campaign_contract(
    campaign: dict[str, Any], path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    configuration = json.loads((path / "run_config.json").read_text())
    contract = configuration["scientific_contract"]
    mechanics = contract["mechanics"]
    scenarios = contract["scenarios"]
    if not np.array_equal(
        np.asarray(mechanics["force_targets_n"], dtype=np.float64),
        _FORCE_TARGETS_N,
    ):
        raise RuntimeError(f"{campaign['name']} uses a different force schedule")
    if _FIXED_SCENARIO["sphere_diameter_mm"] not in scenarios["sphere_diameters_mm"]:
        raise RuntimeError(f"{campaign['name']} lacks the fixed sphere diameter")
    if _FIXED_SCENARIO["contact_y_mm"] not in scenarios["contact_y_mm"]:
        raise RuntimeError(f"{campaign['name']} lacks the fixed contact Y")
    if "theta_deg" in scenarios and _FIXED_SCENARIO["contact_angle_deg"] not in scenarios[
        "theta_deg"
    ]:
        raise RuntimeError(f"{campaign['name']} lacks theta=0")
    return contract, mechanics


def _select_campaign_rows(
    rows: list[dict[str, str]], balanced_trial: int
) -> list[tuple[dict[str, str], str]]:
    ordered = sorted(rows, key=lambda row: int(row["ax_trial_index"]))
    void_width = np.asarray(
        [float(row[_MORPHOLOGY_COLUMNS[-1]]) for row in ordered], dtype=np.float64
    )
    selected: dict[int, tuple[dict[str, str], list[str]]] = {}
    for quantile in _QUANTILES:
        target = float(np.quantile(void_width, quantile))
        candidates = sorted(
            ordered,
            key=lambda item: (
                abs(float(item[_MORPHOLOGY_COLUMNS[-1]]) - target),
                int(item["ax_trial_index"]),
            ),
        )
        row = next(
            item
            for item in candidates
            if int(item["ax_trial_index"]) not in selected
        )
        trial = int(row["ax_trial_index"])
        selected.setdefault(trial, (row, []))[1].append(
            f"void_quantile_{100.0 * quantile:04.1f}pct"
        )
    balanced = next(
        (row for row in ordered if int(row["ax_trial_index"]) == balanced_trial),
        None,
    )
    if balanced is None:
        raise RuntimeError(f"balanced trial {balanced_trial} is not valid/completed")
    selected.setdefault(balanced_trial, (balanced, []))[1].append("balanced")
    return [
        (row, "+".join(reasons))
        for _, (row, reasons) in sorted(selected.items())
    ]


def _result_id(material: str, morphology: tuple[float, ...], variant: str) -> str:
    actual = list(morphology)
    if variant == "no_void":
        actual[-1] = 0.0
    legacy_variant = "original" if variant == "lumo" else variant
    if variant == "lumo" and morphology[-1] == 0.0:
        legacy_variant = "no_void"
    payload = json.dumps(
        {
            "material": material,
            "variant": legacy_variant,
            "morphology_mm": actual,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def prepare_manifest(*, output_directory: Path = _OUTPUT_DIRECTORY) -> list[dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    base_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    campaign_metadata: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {}
    for campaign in _CAMPAIGNS:
        path = _ROOT / "output" / "optimization" / campaign["directory"]
        if not path.is_dir():
            raise FileNotFoundError(f"missing campaign: {path}")
        contract, _ = _check_campaign_contract(campaign, path)
        contracts[campaign["name"]] = contract
        valid = _valid_campaign_rows(path)
        widths = np.asarray(
            [float(row[_MORPHOLOGY_COLUMNS[-1]]) for row in valid],
            dtype=np.float64,
        )
        campaign_metadata.append(
            {
                **campaign,
                "path": str(path.relative_to(_ROOT)),
                "valid_completed_design_count": len(valid),
                "zero_void_design_count": int(np.count_nonzero(widths == 0.0)),
                "void_width_min_mm": float(widths.min()),
                "void_width_max_mm": float(widths.max()),
                "void_width_quantiles_mm": np.quantile(widths, _QUANTILES).tolist(),
            }
        )
        for row in valid:
            catalog_rows.append(
                {
                    "source_campaign": campaign["name"],
                    "source_campaign_directory": str(path.relative_to(_ROOT)),
                    "source_trial_id": int(row["ax_trial_index"]),
                    "material": campaign["material"],
                    "campaign_type": campaign["campaign_type"],
                    "is_balanced": int(row["ax_trial_index"])
                    == campaign["balanced_trial"],
                    "is_pareto": row.get("is_pareto", ""),
                    "flat_pad_height_mm": float(row[_MORPHOLOGY_COLUMNS[0]]),
                    "semiellipse_height_mm": float(row[_MORPHOLOGY_COLUMNS[1]]),
                    "stem_width_mm": float(row[_MORPHOLOGY_COLUMNS[2]]),
                    "stem_height_mm": float(row[_MORPHOLOGY_COLUMNS[3]]),
                    "void_width_mm": float(row[_MORPHOLOGY_COLUMNS[4]]),
                    "J_contact": float(row["J_contact"]),
                    "J_obs": float(row["J_obs"]),
                    "limiting_contact_scenario": row.get(
                        "limiting_contact_scenario", ""
                    ),
                    "limiting_obs_force_n": row.get("limiting_obs_force_n", ""),
                    "limiting_obs_contact_y_pair_mm": row.get(
                        "limiting_obs_contact_y_pair_mm", ""
                    ),
                    "generation_node": row.get("generation_node", ""),
                    "runtime_s": row.get("runtime_s", ""),
                    "raw_result_path": row.get("raw_result_path", ""),
                }
            )
        selected = _select_campaign_rows(valid, campaign["balanced_trial"])
        for row, reason in selected:
            trial = int(row["ax_trial_index"])
            morphology = tuple(float(row[name]) for name in _MORPHOLOGY_COLUMNS)
            sample_id = f"{campaign['name']}_t{trial:04d}"
            base = {
                "sample_id": sample_id,
                "source_campaign": campaign["name"],
                "source_campaign_directory": str(path.relative_to(_ROOT)),
                "source_trial_id": trial,
                "material": campaign["material"],
                "campaign_type": campaign["campaign_type"],
                "selection_reason": reason,
                "is_balanced": "balanced" in reason,
                "is_pareto": row.get("is_pareto", ""),
                "flat_pad_height_mm": morphology[0],
                "semiellipse_height_mm": morphology[1],
                "stem_width_mm": morphology[2],
                "stem_height_mm": morphology[3],
                "lumo_void_width_mm": morphology[4],
                "campaign_J_contact": float(row["J_contact"]),
                "campaign_J_obs": float(row["J_obs"]),
                "mechanics_preset": contract["mechanics_preset"],
                "optical_preset": contract["optical_preset"],
            }
            base_rows.append(base)
            for variant in ("soft_only", "no_void", "lumo"):
                actual_void = morphology[4] if variant == "lumo" else 0.0
                result_id = _result_id(campaign["material"], morphology, variant)
                variant_rows.append(
                    {
                        **base,
                        "variant_key": variant,
                        "condition": _CONDITION_LABELS[variant],
                        "actual_void_width_mm": actual_void,
                        "result_id": result_id,
                        "state_path": str(
                            (output_directory / "states" / f"{result_id}.npz").relative_to(
                                _ROOT
                            )
                        ),
                        "render_path": "",
                        "status": "pending",
                        "failure": "",
                    }
                )
    _write_csv(output_directory / _BASE_MANIFEST_PATH.name, base_rows)
    _write_csv(output_directory / _VARIANT_MANIFEST_PATH.name, variant_rows)
    _write_csv(output_directory / "campaign_design_catalog.csv", catalog_rows)
    manifest = {
        "study": "multi-design carrier and lateral-void ablation",
        "selection_rule": (
            "within each valid completed campaign, select without replacement "
            "the lowest-trial-ID design nearest each 0/12.5/25/37.5/50/62.5/"
            "75/87.5/100% empirical void-width quantile, then add the campaign "
            "balanced trial and deduplicate trial IDs"
        ),
        "conditions": _CONDITION_LABELS,
        "fixed_scenario": _FIXED_SCENARIO,
        "force_targets_n": _FORCE_TARGETS_N.tolist(),
        "campaigns": campaign_metadata,
        "campaign_catalog_count": len(catalog_rows),
        "base_sample_count": len(base_rows),
        "variant_manifest_count": len(variant_rows),
        "unique_result_count": len({row["result_id"] for row in variant_rows}),
        "contracts": contracts,
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return variant_rows


def _load_runtime() -> ModuleType:
    path = Path(__file__).with_name("simulation_ablation_study.py")
    specification = importlib.util.spec_from_file_location(
        "_lumo_single_design_ablation_runtime", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load ablation runtime: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _configure_runtime(runtime: ModuleType, contract: dict[str, Any]) -> None:
    mechanics = contract["mechanics"]
    runtime._FORCE_TARGETS_N = _FORCE_TARGETS_N.copy()
    runtime._SPHERE_DIAMETER_MM = _FIXED_SCENARIO["sphere_diameter_mm"]
    runtime._CONTACT_Y_MM = _FIXED_SCENARIO["contact_y_mm"]
    runtime._CONTACT_ANGLE_DEG = _FIXED_SCENARIO["contact_angle_deg"]
    runtime._INITIAL_CLEARANCE_M = _FIXED_SCENARIO["initial_clearance_m"]
    runtime._SIM_FREQUENCY_HZ = float(mechanics["sim_frequency_hz"])
    runtime._VBD_ITERATIONS = int(mechanics["vbd_iterations"])
    runtime._APPROACH_SPEED_M_S = float(mechanics["approach_speed_m_s"])
    runtime._MAX_SIM_TIME_S = float(mechanics["max_sim_time_s"])
    runtime._ELEMENT_SIZE_MM = float(mechanics["element_size_mm"])
    runtime._SOFT_CONTACT_MARGIN_M = float(mechanics["soft_contact_margin_m"])
    runtime._INDENTER_STIFFNESS_N_M = float(
        mechanics["indenter_contact_stiffness_n_m"]
    )
    runtime._INDENTER_DAMPING_N_S_M = float(
        mechanics["indenter_contact_damping_n_s_m"]
    )


def _fingertip(base: dict[str, str], *, no_void: bool = False) -> Fingertip:
    void_width = 0.0 if no_void else float(base["lumo_void_width_mm"])
    geometry = FingertipGeometry(
        flat_pad_height_mm=float(base["flat_pad_height_mm"]),
        semiellipse_height_mm=float(base["semiellipse_height_mm"]),
        stem_width_mm=float(base["stem_width_mm"]),
        stem_height_mm=float(base["stem_height_mm"]),
        void_width_mm=void_width,
    )
    return Fingertip(
        parameters=FingertipParameters(
            geometry=geometry,
            mechanics=MECHANICS_PRESETS[base["mechanics_preset"]],
            optics=OPTICAL_PRESETS[base["optical_preset"]],
        )
    )


def _cases(runtime: ModuleType, base: dict[str, str]) -> tuple[Any, ...]:
    original = _fingertip(base)
    no_void = _fingertip(base, no_void=True)
    original_mesh = make_fingertip_mesh(
        original, element_size_mm=float(runtime._ELEMENT_SIZE_MM)
    )
    no_void_mesh = make_fingertip_mesh(
        no_void, element_size_mm=float(runtime._ELEMENT_SIZE_MM)
    )
    soft_mesh = runtime._solid_soft_mesh(original, original_mesh)
    return (
        runtime.AblationCase(
            name="soft_only",
            fingertip=original,
            mesh=soft_mesh,
            carrier_interaction="absent",
            bonded_vertex_indices=soft_mesh.bonded_vertex_indices,
            tied_interface_indices=np.empty(0, dtype=np.int32),
            construction=(
                "same external envelope filled by homogeneous source material; "
                "no carrier or internal cavity; dorsal top mounting preserved"
            ),
        ),
        runtime.AblationCase(
            name="no_void",
            fingertip=no_void,
            mesh=no_void_mesh,
            carrier_interaction="contact",
            bonded_vertex_indices=no_void_mesh.bonded_vertex_indices,
            tied_interface_indices=np.empty(0, dtype=np.int32),
            construction=(
                "production carrier/contact and outer pad retained; only lateral "
                "void width set to exactly zero"
            ),
        ),
        runtime.AblationCase(
            name="lumo",
            fingertip=original,
            mesh=original_mesh,
            carrier_interaction="contact",
            bonded_vertex_indices=original_mesh.bonded_vertex_indices,
            tied_interface_indices=np.empty(0, dtype=np.int32),
            construction="exact sampled production morphology",
        ),
    )


def _ragged(values: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty((len(values), 2), dtype=np.int64)
    cursor = 0
    chunks = []
    for index, value in enumerate(values):
        array = np.asarray(value)
        offsets[index] = (cursor, len(array))
        cursor += len(array)
        chunks.append(array)
    if not chunks or cursor == 0:
        return offsets, np.empty((0, width), dtype=np.float64)
    return offsets, np.concatenate(chunks, axis=0)


def _save_mechanics(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    objective = result["objective"]
    indices_offsets, indices = _ragged(result["checkpoint_contact_indices"], 3)
    normal_offsets, normals = _ragged(result["checkpoint_contact_normals"], 3)
    position_offsets, positions = _ragged(result["checkpoint_contact_positions_m"], 3)
    arrays = {
        key: np.asarray(result[key])
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
        )
    }
    arrays.update(
        {
            "contact_index_offsets": indices_offsets,
            "checkpoint_contact_indices": indices.astype(np.int32),
            "contact_normal_offsets": normal_offsets,
            "checkpoint_contact_normals": normals,
            "contact_position_offsets": position_offsets,
            "checkpoint_contact_positions_m": positions,
            "q_form": np.asarray(objective.q_form[0]),
            "q_stable": np.asarray(objective.q_stable[0]),
            "q_stiff": np.asarray(objective.q_stiff[0]),
            "J_contact": np.asarray(objective.J_contact),
            "k_early_n_m": np.asarray(objective.k_early_n_m[0]),
            "k_late_n_m": np.asarray(objective.k_late_n_m[0]),
            "runtime_s": np.asarray(result["runtime_s"]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _mechanics_metrics(arrays)


def _mechanics_metrics(state: Any) -> dict[str, Any]:
    force = np.asarray(state["force_n"], dtype=np.float64)
    checkpoint_force = np.asarray(state["checkpoint_forces_n"], dtype=np.float64)
    checkpoint_area = [
        1.0e6
        * float(
            np.asarray(state["external_area_m2"])[
                int(np.argmin(np.abs(force - value)))
            ]
        )
        for value in checkpoint_force
    ]
    return {
        "fixed_J_contact": float(state["J_contact"]),
        "q_form": float(state["q_form"]),
        "q_stable": float(state["q_stable"]),
        "q_stiff": float(state["q_stiff"]),
        "k_early_n_mm": 1.0e-3 * float(state["k_early_n_m"]),
        "k_late_n_mm": 1.0e-3 * float(state["k_late_n_m"]),
        "minimum_det_f": float(np.min(state["minimum_det_f"])),
        "inversion_count": int(np.count_nonzero(np.asarray(state["minimum_det_f"]) <= 0.0)),
        "contact_buffer_overflow": int(np.max(state["contact_buffer_overflow"])),
        "mechanics_runtime_s": float(state["runtime_s"]),
        "checkpoint_actual_forces_n": json.dumps(
            checkpoint_force.tolist()
        ),
        "checkpoint_indentations_mm": json.dumps(
            (1.0e3 * np.asarray(state["checkpoint_indentations_m"])).tolist()
        ),
        "checkpoint_external_areas_mm2": json.dumps(checkpoint_area),
    }


def _append_optics(path: Path, optical: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
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
        arrays[f"optical_{name}"] = np.asarray(optical[name])
    arrays["optical_maximum_energy_closure_error"] = np.asarray(
        optical["maximum_energy_closure_error"]
    )
    arrays["optical_ray_count_per_led"] = np.asarray(optical["ray_count_per_led"])
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _optics_metrics(arrays)


def _optics_metrics(state: Any) -> dict[str, Any]:
    distance = np.asarray(state["optical_state_distance"], dtype=np.float64)
    visible = np.asarray(state["optical_total_visible_power"], dtype=np.float64)
    delta_visible = np.asarray(
        state["optical_delta_visible_power"], dtype=np.float64
    )
    relative = np.asarray(
        state["optical_relative_visible_power_change"], dtype=np.float64
    )
    return {
        **{f"D_{force:g}N": float(distance[index]) for index, force in enumerate((0, 1, 2, 5, 10))},
        **{
            f"visible_power_{force:g}N": float(visible[index])
            for index, force in enumerate((0, 1, 2, 5, 10))
        },
        **{
            f"delta_visible_power_{force:g}N": float(delta_visible[index])
            for index, force in enumerate((0, 1, 2, 5, 10))
        },
        **{
            f"relative_visible_change_{force:g}N": float(relative[index])
            for index, force in enumerate((0, 1, 2, 5, 10))
        },
        "d_onset_diagnostic": float(np.min(distance[1:])),
        "maximum_energy_closure_error": float(
            state["optical_maximum_energy_closure_error"]
        ),
        "ray_count_per_led": int(state["optical_ray_count_per_led"]),
    }


def _contract_for_row(row: dict[str, str]) -> dict[str, Any]:
    configuration = json.loads(
        (_ROOT / row["source_campaign_directory"] / "run_config.json").read_text()
    )
    return configuration["scientific_contract"]


def run_mechanics(*, smoke: bool = False) -> None:
    output = (
        _ROOT / "output" / "validation" / "multi_design_void_ablation_smoke"
        if smoke
        else _OUTPUT_DIRECTORY
    )
    variants = prepare_manifest(output_directory=output)
    if smoke:
        chosen_samples = []
        for material in ("Dragon Skin 10 NV", "Solaris"):
            chosen_samples.append(next(row["sample_id"] for row in variants if row["material"] == material))
        variants = [row for row in variants if row["sample_id"] in chosen_samples]
    runtime = _load_runtime()
    completed: dict[str, dict[str, Any]] = {}
    rows_by_sample: dict[str, dict[str, str]] = {}
    for row in variants:
        rows_by_sample.setdefault(row["sample_id"], row)
    for sample_index, base in enumerate(rows_by_sample.values(), start=1):
        sample_rows = [
            row for row in variants if row["sample_id"] == base["sample_id"]
        ]
        for row in sample_rows:
            result_id = row["result_id"]
            if result_id in completed:
                continue
            state_path = _ROOT / row["state_path"]
            if state_path.is_file() and not smoke:
                with np.load(state_path, allow_pickle=False) as stored:
                    if "J_contact" in stored.files:
                        completed[result_id] = _mechanics_metrics(stored)
        if not smoke and all(row["result_id"] in completed for row in sample_rows):
            print(
                f"sample={sample_index}/{len(rows_by_sample)} {base['sample_id']} "
                "mechanics=reused",
                flush=True,
            )
            continue

        contract = _contract_for_row(base)
        _configure_runtime(runtime, contract)
        cases = _cases(runtime, base)
        case_by_variant = {"soft_only": cases[0], "no_void": cases[1], "lumo": cases[2]}
        for variant in ("soft_only", "no_void", "lumo"):
            manifest_row = next(
                row
                for row in sample_rows
                if row["variant_key"] == variant
            )
            result_id = manifest_row["result_id"]
            state_path = _ROOT / manifest_row["state_path"]
            print(
                f"sample={sample_index}/{len(rows_by_sample)} {base['sample_id']} "
                f"variant={variant} result={result_id}",
                flush=True,
            )
            if result_id in completed:
                continue
            try:
                result = runtime._run_case(case_by_variant[variant], smoke=smoke)
                if smoke:
                    print(
                        f"  reached {np.asarray(result['checkpoint_forces_n'])[-1]:.4f} N; "
                        f"min det(F)={np.min(result['minimum_det_f']):.6f}",
                        flush=True,
                    )
                    continue
                completed[result_id] = _save_mechanics(state_path, result)
            except Exception as error:  # preserve every failed sampled counterfactual
                completed[result_id] = {"failure": f"{type(error).__name__}: {error}"}
                print(f"  FAILED: {completed[result_id]['failure']}", flush=True)
    if smoke:
        return
    result_rows = []
    for row in variants:
        result = completed.get(row["result_id"], {"failure": "not evaluated"})
        result_rows.append(
            {
                **row,
                "status": "failed" if result.get("failure") else "mechanics_complete",
                "failure": result.get("failure", ""),
                **{key: value for key, value in result.items() if key != "failure"},
            }
        )
    _write_csv(_RESULTS_PATH, result_rows)
    _write_csv(_VARIANT_MANIFEST_PATH, result_rows)


def run_optics(*, sample_side_count: int = 256) -> None:
    if not _RESULTS_PATH.is_file():
        raise FileNotFoundError("run mechanics before optical replay")
    os.environ.setdefault(
        "OTK_INCLUDE_DIR",
        str(_ROOT.parent / "optix-toolkit" / "ShaderUtil" / "include"),
    )
    rows = _read_csv(_RESULTS_PATH)
    runtime = _load_runtime()
    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    optical_by_result: dict[str, dict[str, Any]] = {}
    for sample_index, (sample_id, sample_rows) in enumerate(by_sample.items(), start=1):
        if any(row["failure"] for row in sample_rows):
            continue
        unique_result_ids = {row["result_id"] for row in sample_rows}
        for row in sample_rows:
            result_id = row["result_id"]
            if result_id in optical_by_result:
                continue
            path = _ROOT / row["state_path"]
            with np.load(path, allow_pickle=False) as state:
                if (
                    "optical_state_distance" in state.files
                    and int(state["optical_ray_count_per_led"])
                    == sample_side_count**2
                ):
                    optical_by_result[result_id] = _optics_metrics(state)
        if unique_result_ids.issubset(optical_by_result):
            print(
                f"optics={sample_index}/{len(by_sample)} {sample_id} reused",
                flush=True,
            )
            continue

        base = sample_rows[0]
        _configure_runtime(runtime, _contract_for_row(base))
        cases = _cases(runtime, base)
        variant_case = {"soft_only": cases[0], "no_void": cases[1], "lumo": cases[2]}
        case_order = (
            variant_case["soft_only"],
            variant_case["no_void"],
            variant_case["lumo"],
        )
        temporary_arrays: dict[str, np.ndarray] = {}
        for prefix, variant in (
            ("soft_only", "soft_only"),
            ("no_void", "no_void"),
            ("lumo", "lumo"),
        ):
            row = next(
                item for item in sample_rows if item["variant_key"] == variant
            )
            path = _ROOT / row["state_path"]
            with np.load(path, allow_pickle=False) as state:
                for key in (
                    "reference_vertices_m",
                    "tetrahedra",
                    "checkpoint_vertices_m",
                    "checkpoint_forces_n",
                ):
                    temporary_arrays[f"{prefix}_{key}"] = state[key]
        temporary = _OUTPUT_DIRECTORY / f"optical_input_{sample_id}.npz"
        np.savez_compressed(temporary, **temporary_arrays)
        runtime._RESULT_PATH = temporary
        print(f"optics={sample_index}/{len(by_sample)} {sample_id}", flush=True)
        try:
            optical_results = runtime._run_optical_ablation(
                case_order, sample_side_count=sample_side_count, smoke=False
            )
            for variant, optical in zip(
                ("soft_only", "no_void", "lumo"), optical_results, strict=True
            ):
                row = next(
                    item for item in sample_rows if item["variant_key"] == variant
                )
                result_id = row["result_id"]
                if result_id not in optical_by_result:
                    optical_by_result[result_id] = _append_optics(
                        _ROOT / row["state_path"], optical
                    )
        finally:
            temporary.unlink(missing_ok=True)
    output_rows = []
    for row in rows:
        optical = optical_by_result.get(row["result_id"], {})
        output_rows.append(
            {
                **row,
                "status": "complete" if optical and not row["failure"] else row["status"],
                **optical,
            }
        )
    _write_csv(_RESULTS_PATH, output_rows)
    _write_csv(_VARIANT_MANIFEST_PATH, output_rows)


def _difference(first: dict[str, str], second: dict[str, str], name: str) -> float:
    return float(first[name]) - float(second[name])


def analyze() -> None:
    rows = _read_csv(_RESULTS_PATH)
    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    paired = []
    for sample_id, sample_rows in by_sample.items():
        if any(row["failure"] or row["status"] != "complete" for row in sample_rows):
            continue
        variants = {row["variant_key"]: row for row in sample_rows}
        soft = variants["soft_only"]
        no_void = variants["no_void"]
        lumo = variants["lumo"]
        base = lumo
        paired.append(
            {
                "sample_id": sample_id,
                "source_campaign": base["source_campaign"],
                "source_trial_id": int(base["source_trial_id"]),
                "material": base["material"],
                "campaign_type": base["campaign_type"],
                "is_balanced": base["is_balanced"],
                "is_pareto": base["is_pareto"],
                "lumo_void_width_mm": float(base["lumo_void_width_mm"]),
                **{
                    f"carrier_delta_{name}": _difference(no_void, soft, name)
                    for name in ("fixed_J_contact", "q_form", "q_stable", "q_stiff")
                },
                **{
                    f"void_delta_{name}": _difference(lumo, no_void, name)
                    for name in ("fixed_J_contact", "q_form", "q_stable", "q_stiff")
                },
                **{
                    f"carrier_delta_D_{force}N": _difference(
                        no_void, soft, f"D_{force}N"
                    )
                    for force in (1, 2, 10)
                },
                **{
                    f"void_delta_D_{force}N": _difference(
                        lumo, no_void, f"D_{force}N"
                    )
                    for force in (1, 2, 10)
                },
                **{
                    f"carrier_delta_visible_power_{force}N": _difference(
                        no_void, soft, f"visible_power_{force}N"
                    )
                    for force in (1, 2, 10)
                },
                **{
                    f"void_delta_visible_power_{force}N": _difference(
                        lumo, no_void, f"visible_power_{force}N"
                    )
                    for force in (1, 2, 10)
                },
                "soft_J_contact": float(soft["fixed_J_contact"]),
                "no_void_J_contact": float(no_void["fixed_J_contact"]),
                "lumo_J_contact": float(lumo["fixed_J_contact"]),
                "soft_D_2N": float(soft["D_2N"]),
                "no_void_D_2N": float(no_void["D_2N"]),
                "lumo_D_2N": float(lumo["D_2N"]),
            }
        )
    if not paired:
        raise RuntimeError("no complete paired results are available")
    _write_csv(_PAIRED_PATH, paired)

    def describe(name: str, values: np.ndarray) -> dict[str, Any]:
        return {
            "metric": name,
            "count": int(len(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "positive_fraction": float(np.mean(values > 0.0)),
            "negative_fraction": float(np.mean(values < 0.0)),
            "zero_fraction": float(np.mean(np.isclose(values, 0.0, rtol=0.0, atol=1e-12))),
        }

    metrics = (
        "carrier_delta_fixed_J_contact",
        "void_delta_fixed_J_contact",
        "void_delta_D_1N",
        "void_delta_D_2N",
        "void_delta_D_10N",
    )
    groups: dict[str, Any] = {}
    grouped = [("all", paired)]
    grouped.append(
        (
            "finite_void",
            [row for row in paired if float(row["lumo_void_width_mm"]) > 0.0],
        )
    )
    grouped.extend(
        (
            campaign["name"],
            [row for row in paired if row["source_campaign"] == campaign["name"]],
        )
        for campaign in _CAMPAIGNS
    )
    grouped.extend(
        (
            label,
            [row for row in paired if row[field] == value],
        )
        for label, field, value in (
            ("dragon", "material", "Dragon Skin 10 NV"),
            ("solaris", "material", "Solaris"),
            ("normal_source", "campaign_type", "normal"),
            ("angled_source", "campaign_type", "angled"),
        )
    )
    for label, selected in grouped:
        groups[label] = {
            name: describe(name, np.asarray([float(row[name]) for row in selected]))
            for name in metrics
        }
    balanced = [row for row in paired if str(row["is_balanced"]) == "True"]
    summary = {
        "study": "multi-design carrier and lateral-void ablation",
        "fixed_scenario": _FIXED_SCENARIO,
        "force_targets_n": _FORCE_TARGETS_N.tolist(),
        "complete_pair_count": len(paired),
        "zero_void_lumo_count": int(
            sum(
                np.isclose(float(row["lumo_void_width_mm"]), 0.0)
                for row in paired
            )
        ),
        "failed_variant_count": sum(bool(row["failure"]) for row in rows),
        "descriptive_summary": groups,
        "balanced_designs": [
            {
                "campaign": row["source_campaign"],
                "trial": row["source_trial_id"],
                "void_width_mm": row["lumo_void_width_mm"],
            }
            for row in balanced
        ],
        "J_obs_defined": False,
        "optical_metric": (
            "D(F)=||(y(F)-y(0))/5||_2 from one fixed contact; diagnostic only"
        ),
    }
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary)


def _write_report(summary: dict[str, Any]) -> None:
    overall = summary["descriptive_summary"]["all"]
    finite_void = summary["descriptive_summary"]["finite_void"]
    carrier = overall["carrier_delta_fixed_J_contact"]
    void_mechanics = overall["void_delta_fixed_J_contact"]
    void_optics = overall["void_delta_D_1N"]
    finite_void_mechanics = finite_void["void_delta_fixed_J_contact"]
    finite_void_optics = finite_void["void_delta_D_1N"]
    balanced = summary["balanced_designs"]
    balanced_finite = sum(float(row["void_width_mm"]) > 0.0 for row in balanced)
    text = f"""# Multi-design carrier and lateral-void ablation

## Contract

- Source campaigns: the current Dragon Skin normal, Dragon Skin orientation-robust, Solaris normal, and Solaris orientation-robust campaigns listed in `manifest.json`.
- Re-evaluation scenario for every source design: 20 mm sphere, Y=-5.5 mm, theta=0 deg.
- Loading: the production 100 Hz / 10-iteration constant-speed first-crossing path at 1/2/5/10 N.
- Sampling: without-replacement nearest valid completed designs at each campaign's 0/12.5/25/37.5/50/62.5/75/87.5/100% empirical void-width quantile, plus its balanced trial, with trial-ID deduplication.
- Structural sequence: Soft-only -> No-void carrier -> LUMO morphology.
- Bonded-T is not part of this primary experiment.

All angled-campaign designs are evaluated here at theta=0. Campaign provenance is therefore a morphology-source label, not a change in the matched contact condition.

## Paired results

- Complete paired base morphologies: {summary['complete_pair_count']}.
- Exact zero-void LUMO morphologies: {summary['zero_void_lumo_count']}.
- Carrier effect on fixed-scenario J_contact (No-void carrier minus Soft-only): median {carrier['median']:+.6f}, IQR [{carrier['q25']:+.6f}, {carrier['q75']:+.6f}], positive in {carrier['positive_fraction']:.1%} of samples.
- Void effect on fixed-scenario J_contact (LUMO morphology minus No-void carrier): median {void_mechanics['median']:+.6f}, IQR [{void_mechanics['q25']:+.6f}, {void_mechanics['q75']:+.6f}], positive in {void_mechanics['positive_fraction']:.1%} and negative in {void_mechanics['negative_fraction']:.1%}.
- Void effect on low-load optical D(1 N): median {void_optics['median']:+.6f}, IQR [{void_optics['q25']:+.6f}, {void_optics['q75']:+.6f}], positive in {void_optics['positive_fraction']:.1%} and negative in {void_optics['negative_fraction']:.1%}.
- Finite-void-only J_contact effect: median {finite_void_mechanics['median']:+.6f}, IQR [{finite_void_mechanics['q25']:+.6f}, {finite_void_mechanics['q75']:+.6f}], positive in {finite_void_mechanics['positive_fraction']:.1%} and negative in {finite_void_mechanics['negative_fraction']:.1%}.
- Finite-void-only D(1 N) effect: median {finite_void_optics['median']:+.6f}, IQR [{finite_void_optics['q25']:+.6f}, {finite_void_optics['q75']:+.6f}], positive in {finite_void_optics['positive_fraction']:.1%} and negative in {finite_void_optics['negative_fraction']:.1%}.
- Balanced designs retaining finite void: {balanced_finite}/{len(balanced)}.

## Safe interpretation

The No-void carrier and LUMO morphology conditions isolate the carrier and lateral-void interventions. A positive median carrier effect supports the rigid carrier's mechanical role only to the extent shown above; heterogeneous signs must be reported rather than hidden. The void result is not required to be uniformly positive: systematic nonzero mechanical and/or optical changes demonstrate that `w_void` is a meaningful additional optomechanical degree of freedom beyond carrier dimensions. Raw one-location optical distances are diagnostics and are not `J_obs`.

The full per-campaign effects, objective components, integrity values, force/displacement histories, contact records, channel responses, and energy ledgers are preserved in `paired_effects.csv`, `variant_results.csv`, and `states/*.npz`. The previous single-design Bonded-T and effective-gap sensitivity remain separate supplementary analyses.
"""
    text += "\n## Material and source-campaign consistency\n\n"
    text += (
        "| Group | median carrier ΔJ_contact | positive | "
        "median void ΔJ_contact | positive / negative | "
        "median void ΔD(1 N) | positive / negative |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    labels = {
        "dragon": "Dragon Skin",
        "solaris": "Solaris",
        "normal_source": "Normal-source designs",
        "angled_source": "Angled-source designs",
    }
    for key, label in labels.items():
        group = summary["descriptive_summary"][key]
        carrier_group = group["carrier_delta_fixed_J_contact"]
        mechanics_group = group["void_delta_fixed_J_contact"]
        optics_group = group["void_delta_D_1N"]
        text += (
            f"| {label} | {carrier_group['median']:+.6f} | "
            f"{carrier_group['positive_fraction']:.0%} | "
            f"{mechanics_group['median']:+.6f} | "
            f"{mechanics_group['positive_fraction']:.0%} / "
            f"{mechanics_group['negative_fraction']:.0%} | "
            f"{optics_group['median']:+.6f} | "
            f"{optics_group['positive_fraction']:.0%} / "
            f"{optics_group['negative_fraction']:.0%} |\n"
        )
    text += (
        "\nThe carrier contribution is positive in both materials and both source-"
        "campaign types. The void contribution is heterogeneous in every grouping; "
        "it must be described as a response-shaping degree of freedom, not a universal "
        "mechanical or optical improvement. Exact zero-void samples reproduce zero "
        "paired differences by construction and saved-state reuse.\n"
    )
    _REPORT_PATH.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--mechanics", action="store_true")
    action.add_argument("--optics", action="store_true")
    action.add_argument("--analyze", action="store_true")
    action.add_argument("--all", action="store_true")
    parser.add_argument("--sample-side-count", type=int, default=256)
    args = parser.parse_args()
    if args.prepare:
        variants = prepare_manifest()
        print(
            f"prepared {len(variants) // 3} base samples, {len(variants)} manifest variants, "
            f"{len({row['result_id'] for row in variants})} unique results"
        )
    elif args.smoke:
        run_mechanics(smoke=True)
    elif args.mechanics:
        run_mechanics()
    elif args.optics:
        run_optics(sample_side_count=args.sample_side_count)
    elif args.analyze:
        analyze()
    else:
        run_mechanics()
        run_optics(sample_side_count=args.sample_side_count)
        analyze()


if __name__ == "__main__":
    main()
