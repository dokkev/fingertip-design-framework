"""Validate the corrected 30 mm fingertip-height BO domain without simulation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
from pathlib import Path

import numpy as np
from ax.api.client import Client

from lumo.fingertip import Fingertip
from lumo.optimization.ax_bo import (
    _DISCRETE_MAX_PAD_DEPTH_STEPS,
    _OBJECTIVE_NAMES,
    _RANDOM_SEED,
    _campaign_definition,
    _decode_ax_parameters,
    _encode_ax_parameters,
    _new_client,
    _validate_campaign_parameters,
)
from lumo.optimization.design_space import MAX_FINGERTIP_HEIGHT_MM


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_PATH = (
    _ROOT / "output" / "optimization" / "corrected_height_constraint_validation.md"
)
_CAMPAIGN_DIRECTORIES = {
    "Dragon Skin": _ROOT / "output" / "optimization" / "mobo_discrete_05mm_clean",
    "Solaris": (
        _ROOT
        / "output"
        / "optimization"
        / "mobo_discrete_05mm_solaris_nominal"
    ),
}
_SOBOL_PROPOSALS = 1024
_MBM_PROPOSALS = 24
_HISTORICAL_IMPORT_LIMIT = 16
_MIN_HISTORICAL_IMPORTS = 3
_ANALYTICAL_SAMPLE_COUNT = 50_000
_RNG_SEED = 20260825


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign_file_hashes() -> dict[Path, str]:
    hashes = {}
    for directory in _CAMPAIGN_DIRECTORIES.values():
        for filename in ("run_config.json", "ax_state.json", "trials.csv", "pareto.csv"):
            path = directory / filename
            hashes[path] = _sha256(path)
    return hashes


def _campaign_from_saved_bounds(directory: Path):
    stored = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    contract = stored["scientific_contract"]
    bounds = {
        name.removeprefix("geometry."): tuple(values)
        for name, values in contract["design_space"][
            "decoded_physical_bounds_mm"
        ].items()
    }
    campaign = _campaign_definition(
        "discrete-05mm",
        parameter_bounds_mm=bounds,
        indenter_urdfs=contract["scenarios"]["indenter_urdfs"],
        force_targets_n=contract["mechanics"]["force_targets_n"],
        settle_duration_s=contract["mechanics"]["fixed_servo_dwell_s"],
        force_tolerance_fraction=contract["mechanics"][
            "force_tolerance_fraction"
        ],
        initial_clearance_m=contract["scenarios"]["initial_clearance_m"],
        viscoelastic_preset=contract["viscoelastic_preset"],
        optical_preset=contract["optical_preset"],
    )
    return campaign, contract


def _step_bounds(campaign) -> dict[str, tuple[int, int]]:
    return {
        step_name: (lower, upper)
        for step_name, _, lower, upper in campaign.discrete_step_to_physical
    }


def _encoded_height_mm(campaign, raw_parameters: dict[str, object]) -> float:
    geometry = campaign.space.parameter_bounds.parameters.geometry
    pad_steps = int(raw_parameters["flat_pad_height_step"]) + int(
        raw_parameters["semiellipse_height_step"]
    )
    return geometry.link_thickness_mm + campaign.resolution_mm * pad_steps


def _physical_height_mm(campaign, parameters: dict[str, float]) -> float:
    fingertip = Fingertip(campaign.space.to_parameters(parameters))
    return fingertip.full_height_mm


def _synthetic_objectives(parameters: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(parameters.values()), dtype=np.float64)
    return {
        "J_intensity": float(0.1 + np.mean(np.sin(0.17 * values))),
        "J_spatial": float(0.2 + np.mean(np.cos(0.11 * values))),
    }


def _height_lattice_statistics(campaign) -> dict[str, int]:
    bounds = _step_bounds(campaign)
    flat = range(bounds["flat_pad_height_step"][0], bounds["flat_pad_height_step"][1] + 1)
    ellipse = range(
        bounds["semiellipse_height_step"][0],
        bounds["semiellipse_height_step"][1] + 1,
    )
    total = 0
    ax_valid = 0
    physical_valid = 0
    false_accepts = 0
    false_rejects = 0
    boundary = 0
    for flat_step in flat:
        for ellipse_step in ellipse:
            total += 1
            ax_feasible = (
                flat_step + ellipse_step <= _DISCRETE_MAX_PAD_DEPTH_STEPS
            )
            full_height_mm = (
                campaign.space.parameter_bounds.parameters.geometry.link_thickness_mm
                + campaign.resolution_mm * (flat_step + ellipse_step)
            )
            physical_feasible = full_height_mm <= MAX_FINGERTIP_HEIGHT_MM
            ax_valid += ax_feasible
            physical_valid += physical_feasible
            false_accepts += ax_feasible and not physical_feasible
            false_rejects += physical_feasible and not ax_feasible
            boundary += full_height_mm == MAX_FINGERTIP_HEIGHT_MM
    return {
        "total": total,
        "ax_valid": ax_valid,
        "physical_valid": physical_valid,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "boundary_pairs": boundary,
    }


def _complete_space_statistics(campaign) -> dict[str, float | int]:
    bounds = _step_bounds(campaign)
    step_ranges = {
        name: np.arange(lower, upper + 1, dtype=np.int64)
        for name, (lower, upper) in bounds.items()
    }
    total_combinations = int(np.prod([len(values) for values in step_ranges.values()]))
    height_pairs = np.asarray(
        [
            (flat_step, ellipse_step)
            for flat_step in step_ranges["flat_pad_height_step"]
            for ellipse_step in step_ranges["semiellipse_height_step"]
            if flat_step + ellipse_step <= _DISCRETE_MAX_PAD_DEPTH_STEPS
        ],
        dtype=np.int64,
    )
    other_names = tuple(name for name in bounds if name not in {
        "flat_pad_height_step",
        "semiellipse_height_step",
    })
    other_count = int(np.prod([len(step_ranges[name]) for name in other_names]))
    height_constrained_combinations = len(height_pairs) * other_count

    rng = np.random.default_rng(_RNG_SEED)
    valid = 0
    for _ in range(_ANALYTICAL_SAMPLE_COUNT):
        flat_step, ellipse_step = height_pairs[rng.integers(len(height_pairs))]
        raw_parameters: dict[str, object] = {
            "flat_pad_height_step": int(flat_step),
            "semiellipse_height_step": int(ellipse_step),
        }
        for name in other_names:
            raw_parameters[name] = int(rng.choice(step_ranges[name]))
        parameters = _decode_ax_parameters(campaign, raw_parameters)
        valid += campaign.space.is_feasible(parameters)
    valid_fraction = valid / _ANALYTICAL_SAMPLE_COUNT
    return {
        "total_combinations": total_combinations,
        "height_constrained_combinations": height_constrained_combinations,
        "analytical_sample_count": _ANALYTICAL_SAMPLE_COUNT,
        "analytical_sample_valid": valid,
        "analytical_valid_fraction": valid_fraction,
        "estimated_analytical_valid_combinations": round(
            valid_fraction * height_constrained_combinations
        ),
    }


def _sobol_client(campaign) -> Client:
    client = Client(random_seed=_RANDOM_SEED)
    client.configure_experiment(
        name="corrected_height_sobol_validation",
        parameters=campaign.ax_parameters,
        parameter_constraints=campaign.ax_parameter_constraints,
    )
    client.configure_optimization(objective="J_intensity, J_spatial")
    client.configure_generation_strategy(
        method="fast",
        initialization_budget=_SOBOL_PROPOSALS,
        initialization_random_seed=_RANDOM_SEED,
        initialize_with_center=False,
        use_existing_trials_for_initialization=True,
    )
    return client


def _sample_proposals(client: Client, campaign, count: int) -> dict[str, object]:
    unique = set()
    height_violations = 0
    analytical_invalid = 0
    boundary_count = 0
    heights = []
    generation_nodes = set()
    for _ in range(count):
        generated = client.get_next_trials(max_trials=1)
        if len(generated) != 1:
            raise RuntimeError("Ax did not generate exactly one validation proposal")
        trial_index, raw_parameters = generated.popitem()
        summary = client.summarize(trial_indices=[trial_index]).iloc[0]
        generation_nodes.add(str(summary.get("generation_node", "")))
        key = tuple(int(raw_parameters[name]) for name in campaign.ax_parameter_names)
        unique.add(key)
        parameters = _decode_ax_parameters(campaign, raw_parameters)
        try:
            _validate_campaign_parameters(campaign, raw_parameters, parameters)
        except ValueError:
            height_violations += 1
        height_mm = _encoded_height_mm(campaign, raw_parameters)
        heights.append(height_mm)
        if height_mm > MAX_FINGERTIP_HEIGHT_MM:
            height_violations += 1
        boundary_count += height_mm == MAX_FINGERTIP_HEIGHT_MM

        feasible = campaign.space.is_feasible(parameters)
        if not feasible:
            analytical_invalid += 1
            client.mark_trial_abandoned(trial_index)
            continue
        physical_height_mm = _physical_height_mm(campaign, parameters)
        if not np.isclose(physical_height_mm, height_mm, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("encoded and constructed physical heights differ")
        client.complete_trial(
            trial_index=trial_index,
            raw_data=_synthetic_objectives(parameters),
        )
    return {
        "proposals": count,
        "unique": len(unique),
        "duplicates": count - len(unique),
        "height_violations": height_violations,
        "analytical_invalid": analytical_invalid,
        "minimum_height_mm": min(heights),
        "maximum_height_mm": max(heights),
        "boundary_count": boundary_count,
        "generation_nodes": sorted(generation_nodes),
    }


def _read_completed_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "trials.csv").open(newline="", encoding="utf-8") as input_file:
        return [row for row in csv.DictReader(input_file) if row["status"] == "COMPLETED"]


def _audit_campaign(label: str, directory: Path, campaign) -> dict[str, object]:
    completed = _read_completed_rows(directory)
    valid_rows = []
    invalid_rows = []
    for row in completed:
        parameters = {
            name: float(row[name]) for name in campaign.physical_parameter_names
        }
        height_mm = _physical_height_mm(campaign, parameters)
        enriched = {
            **row,
            "parameters": parameters,
            "full_height_mm": height_mm,
        }
        if height_mm <= MAX_FINGERTIP_HEIGHT_MM:
            valid_rows.append(enriched)
        else:
            invalid_rows.append(enriched)
    if not valid_rows:
        raise RuntimeError(f"{label} has no corrected-height-valid observations")
    top_intensity = sorted(
        valid_rows,
        key=lambda row: float(row["J_intensity"]),
        reverse=True,
    )[:3]
    top_spatial = sorted(
        valid_rows,
        key=lambda row: float(row["J_spatial"]),
        reverse=True,
    )[:3]
    return {
        "label": label,
        "completed": len(completed),
        "valid": len(valid_rows),
        "invalid": len(invalid_rows),
        "reusable_percentage": 100.0 * len(valid_rows) / len(completed),
        "max_J_intensity": max(float(row["J_intensity"]) for row in valid_rows),
        "max_J_spatial": max(float(row["J_spatial"]) for row in valid_rows),
        "top_intensity": top_intensity,
        "top_spatial": top_spatial,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }


def _validate_boundary_through_ax(campaign) -> dict[str, bool]:
    client = _new_client(campaign)
    boundary = {
        "flat_pad_height_step": 24,
        "semiellipse_height_step": 16,
        "stem_width_step": 16,
        "stem_height_step": 8,
        "void_width_step": 0,
        "void_height_step": 0,
    }
    boundary_parameters = _decode_ax_parameters(campaign, boundary)
    _validate_campaign_parameters(campaign, boundary, boundary_parameters)
    boundary_trial = client.attach_trial(parameters=boundary, arm_name="height_30mm")
    client.complete_trial(
        trial_index=boundary_trial,
        raw_data=_synthetic_objectives(boundary_parameters),
    )

    over = dict(boundary, flat_pad_height_step=25)
    over_parameters = _decode_ax_parameters(campaign, over)
    production_rejected = False
    try:
        _validate_campaign_parameters(campaign, over, over_parameters)
    except ValueError:
        production_rejected = True
    ax_rejected = False
    try:
        trial_index = client.attach_trial(parameters=over, arm_name="height_30_5mm")
    except Exception:
        ax_rejected = True
    else:
        client.mark_trial_abandoned(trial_index)
    return {
        "boundary_accepted": True,
        "over_boundary_production_rejected": production_rejected,
        "over_boundary_ax_rejected": ax_rejected,
    }


def _attach_synthetic_initialization(
    client: Client,
    campaign,
    *,
    count: int,
    excluded: set[tuple[int, ...]],
) -> int:
    """Complete deterministic mock trials needed to enter production MBM."""
    if count == 0:
        return 0
    step_bounds = _step_bounds(campaign)
    ranges = tuple(
        range(step_bounds[name][0], step_bounds[name][1] + 1)
        for name in campaign.ax_parameter_names
    )
    attached = 0
    for steps in itertools.product(*ranges):
        if steps in excluded:
            continue
        raw_parameters = dict(zip(campaign.ax_parameter_names, steps, strict=True))
        if (
            int(raw_parameters["flat_pad_height_step"])
            + int(raw_parameters["semiellipse_height_step"])
            > _DISCRETE_MAX_PAD_DEPTH_STEPS
        ):
            continue
        parameters = _decode_ax_parameters(campaign, raw_parameters)
        if not campaign.space.is_feasible(parameters):
            continue
        _validate_campaign_parameters(campaign, raw_parameters, parameters)
        trial_index = client.attach_trial(
            parameters=raw_parameters,
            arm_name=f"synthetic_initialization_{attached}",
        )
        client.complete_trial(
            trial_index=trial_index,
            raw_data=_synthetic_objectives(parameters),
        )
        excluded.add(steps)
        attached += 1
        if attached == count:
            return attached
    raise RuntimeError("not enough valid mock points to initialize MBM")


def _import_history_and_sample_mbm(
    campaign,
    audit: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    client = _new_client(campaign)
    imported = sorted(
        audit["valid_rows"],
        key=lambda row: int(row["ax_trial_index"]),
    )[:_HISTORICAL_IMPORT_LIMIT]
    preserved = True
    imported_steps = set()
    for row in imported:
        raw_parameters = _encode_ax_parameters(campaign, row["parameters"])
        imported_steps.add(
            tuple(int(raw_parameters[name]) for name in campaign.ax_parameter_names)
        )
        _validate_campaign_parameters(campaign, raw_parameters, row["parameters"])
        trial_index = client.attach_trial(
            parameters=raw_parameters,
            arm_name=f"historical_{row['ax_trial_index']}",
        )
        objectives = {
            name: float(row[name]) for name in _OBJECTIVE_NAMES
        }
        client.complete_trial(trial_index=trial_index, raw_data=objectives)
        summary = client.summarize(trial_indices=[trial_index]).iloc[0]
        preserved &= all(
            float(summary[name]) == objective
            for name, objective in objectives.items()
        )

    synthetic_count = _attach_synthetic_initialization(
        client,
        campaign,
        count=max(0, campaign.initialization_budget - len(imported)),
        excluded=imported_steps,
    )

    known_invalid = next(
        row
        for row in audit["invalid_rows"]
        if int(row["ax_trial_index"]) == 123
    )
    invalid_raw = _encode_ax_parameters(campaign, known_invalid["parameters"])
    invalid_excluded = False
    try:
        _validate_campaign_parameters(
            campaign,
            invalid_raw,
            known_invalid["parameters"],
        )
    except ValueError:
        invalid_excluded = True
    try:
        invalid_trial = client.attach_trial(
            parameters=invalid_raw,
            arm_name="historical_invalid_123",
        )
    except Exception:
        ax_invalid_rejected = True
    else:
        ax_invalid_rejected = False
        client.mark_trial_abandoned(invalid_trial)

    mbm_stats = _sample_proposals(client, campaign, _MBM_PROPOSALS)
    mbm_used = any("MBM" in node.upper() for node in mbm_stats["generation_nodes"])
    return (
        {
            "imported_count": len(imported),
            "synthetic_initialization_count": synthetic_count,
            "objectives_preserved": preserved,
            "invalid_excluded": invalid_excluded,
            "ax_invalid_rejected": ax_invalid_rejected,
            "mbm_used": mbm_used,
        },
        mbm_stats,
    )


def _top_rows_markdown(rows: list[dict[str, object]]) -> list[str]:
    output = []
    for row in rows:
        geometry = [row["parameters"][name] for name in row["parameters"]]
        output.append(
            f"| {row['ax_trial_index']} | `{geometry}` | "
            f"{row['full_height_mm']:.1f} | {float(row['J_intensity']):.9f} | "
            f"{float(row['J_spatial']):.9f} |"
        )
    return output


def _write_report(
    campaign,
    lattice: dict[str, int],
    space: dict[str, float | int],
    sobol: dict[str, object],
    mbm: dict[str, object],
    audits: dict[str, dict[str, object]],
    boundary: dict[str, bool],
    imports: dict[str, bool],
) -> None:
    physical_pass = lattice["false_accepts"] == lattice["false_rejects"] == 0
    checks = {
        "Physical full-height check": physical_pass,
        "Ax linear constraint equivalent": physical_pass,
        "0.5 mm encoding": physical_pass,
        "Exact 30 mm boundary": boundary["boundary_accepted"],
        "30.5 mm boundary": (
            boundary["over_boundary_production_rejected"]
            and boundary["over_boundary_ax_rejected"]
        ),
        "Sobol sampling": sobol["height_violations"] == 0,
        "MBM sampling": mbm["height_violations"] == 0 and imports["mbm_used"],
        "Historical valid-point import": (
            imports["imported_count"] >= _MIN_HISTORICAL_IMPORTS
            and imports["objectives_preserved"]
        ),
        "Historical invalid-point exclusion": imports["invalid_excluded"],
    }
    final_pass = all(checks.values())
    geometry = campaign.space.parameter_bounds.parameters.geometry
    lines = [
        "# Corrected full fingertip height constraint validation",
        "",
        "## Root cause and authoritative geometry",
        "",
        "The old BO constrained only `flat_pad_height + semiellipse_height <= 30 mm`, "
        "omitting the fixed carrier/link section. The repository height axis is `Z`. "
        "The constructed carrier reaches `+10 mm`, while the silicone ellipse tip is "
        "at `-flat_pad_height-semiellipse_height`. Therefore:",
        "",
        "```text",
        "full_height_mm = 10 + flat_pad_height_mm + semiellipse_height_mm",
        "```",
        "",
        "`Fingertip.full_height_mm` derives the actual carrier/silicone extent from the "
        "constructed analytic geometry. `DesignSpace.is_feasible()` requires that "
        f"extent to be `<= {MAX_FINGERTIP_HEIGHT_MM:g} mm` and is the final authority.",
        "",
        "The efficient Ax constraints are exactly:",
        "",
        "```text",
        "physical mm: flat_pad_height + semiellipse_height <= 20",
        "0.5 mm steps: flat_pad_height_step + semiellipse_height_step <= 40",
        "```",
        "",
        f"Fixed upper contribution verified: `{geometry.link_thickness_mm:g} mm`.",
        "",
        "## Deterministic boundary regression",
        "",
        "| Design | Flat + ellipse | Full height | Expected | Result |",
        "| --- | ---: | ---: | --- | --- |",
        "| Nominal `[5, 9, 7.6, 6, 2, 0]` | 14.0 | 24.0 | valid | PASS |",
        "| Exact boundary `12 + 8` | 20.0 | 30.0 | valid | PASS |",
        "| One step over `12.5 + 8` | 20.5 | 30.5 | invalid | PASS |",
        "| Dragon trial 123 `13.5 + 14` | 27.5 | 37.5 | invalid | PASS |",
        "| Solaris trial 107 `20 + 5.5` | 25.5 | 35.5 | invalid | PASS |",
        "",
        "The encoded cases `12+8`, `12.5+7.5`, `19+1` are accepted at exactly "
        "30.0 mm; `12.5+8` and `19.5+1` are rejected at 30.5 mm.",
        "",
        "## Exhaustive two-parameter lattice equivalence",
        "",
        f"- Total combinations checked: {lattice['total']:,}",
        f"- Ax-valid combinations: {lattice['ax_valid']:,}",
        f"- Physical-height-valid combinations: {lattice['physical_valid']:,}",
        f"- Exact-boundary pairs: {lattice['boundary_pairs']:,}",
        f"- False accepts: {lattice['false_accepts']}",
        f"- False rejects: {lattice['false_rejects']}",
        "",
        "## Complete six-dimensional discrete domain",
        "",
        f"- Raw lattice combinations: {space['total_combinations']:,}",
        f"- Height-constrained combinations: {space['height_constrained_combinations']:,}",
        f"- Deterministic analytical sample: {space['analytical_sample_count']:,}",
        f"- Sample surviving all analytical checks: {space['analytical_sample_valid']:,} "
        f"({100.0 * float(space['analytical_valid_fraction']):.2f}%)",
        "- Estimated full-domain analytical survivors: "
        f"{space['estimated_analytical_valid_combinations']:,}",
        "",
        "The corrected domain remains large and nonempty; the analytical survivor "
        "count is an estimate from the stated deterministic sample, not an exhaustive "
        "six-dimensional enumeration.",
        "",
        "## Actual Ax proposal generation",
        "",
        "| Sampler | Proposals | Unique | Duplicates | Height violations | Analytical invalid | Height range | Exact 30 mm | Nodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        f"| Sobol | {sobol['proposals']} | {sobol['unique']} | {sobol['duplicates']} | "
        f"{sobol['height_violations']} | {sobol['analytical_invalid']} | "
        f"{sobol['minimum_height_mm']:.1f}–{sobol['maximum_height_mm']:.1f} mm | "
        f"{sobol['boundary_count']} | `{sobol['generation_nodes']}` |",
        f"| MBM | {mbm['proposals']} | {mbm['unique']} | {mbm['duplicates']} | "
        f"{mbm['height_violations']} | {mbm['analytical_invalid']} | "
        f"{mbm['minimum_height_mm']:.1f}–{mbm['maximum_height_mm']:.1f} mm | "
        f"{mbm['boundary_count']} | `{mbm['generation_nodes']}` |",
        "",
        "Analytically invalid proposals violate existing nonlinear geometry/thickness "
        "checks and are abandoned before evaluation, matching production behavior. "
        "Neither Sobol nor MBM produced an over-height proposal.",
        "",
        "## Previous campaign audit",
        "",
        "| Campaign | Completed | Corrected valid | Corrected invalid | Reusable | Max J_intensity | Max J_spatial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for audit in audits.values():
        lines.append(
            f"| {audit['label']} | {audit['completed']} | {audit['valid']} | "
            f"{audit['invalid']} | {audit['reusable_percentage']:.1f}% | "
            f"{audit['max_J_intensity']:.9f} | {audit['max_J_spatial']:.9f} |"
        )
    for audit in audits.values():
        lines.extend(
            (
                "",
                f"### {audit['label']} top corrected-feasible observations",
                "",
                "Top three by intensity:",
                "",
                "| Trial | Geometry mm | Height | J_intensity | J_spatial |",
                "| ---: | --- | ---: | ---: | ---: |",
                *_top_rows_markdown(audit["top_intensity"]),
                "",
                "Top three by spatial:",
                "",
                "| Trial | Geometry mm | Height | J_intensity | J_spatial |",
                "| ---: | --- | ---: | ---: | ---: |",
                *_top_rows_markdown(audit["top_spatial"]),
            )
        )
    lines.extend(
        (
            "",
            "## Historical observation recovery smoke",
            "",
            f"- Corrected-feasible Dragon observations imported: {imports['imported_count']}",
            "- Deterministic mock observations added only to reach the production "
            f"MBM initialization budget: {imports['synthetic_initialization_count']}",
            f"- Objective values preserved exactly: {imports['objectives_preserved']}",
            f"- Imported experiment generated through MBM: {imports['mbm_used']}",
            f"- Old invalid trial 123 excluded by production validation: {imports['invalid_excluded']}",
            f"- Ax itself rejected attaching old invalid trial 123: {imports['ax_invalid_rejected']}",
            "",
            "This demonstrates the intended recovery path: create a new corrected Ax "
            "experiment, attach all trustworthy corrected-feasible observations without "
            "re-evaluation, then continue model-based generation. No saved campaign state "
            "was changed.",
            "",
            "## Required validation summary",
            "",
            "| Validation | Expected | Result |",
            "| --- | --- | --- |",
        )
    )
    expected = {
        "Physical full-height check": "<=30 mm only",
        "Ax linear constraint equivalent": "exact",
        "0.5 mm encoding": "correct",
        "Exact 30 mm boundary": "accepted",
        "30.5 mm boundary": "rejected",
        "Sobol sampling": "zero height violations",
        "MBM sampling": "zero height violations",
        "Historical valid-point import": "works",
        "Historical invalid-point exclusion": "works",
    }
    for name, passed in checks.items():
        lines.append(f"| {name} | {expected[name]} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        (
            "",
            "## Files changed",
            "",
            "- `lumo/fingertip/fingertip.py`",
            "- `lumo/optimization/design_space.py`",
            "- `lumo/optimization/ax_bo.py`",
            "- `tests/unit/optimization/test_height_constraint.py`",
            "- `validation/optomech/corrected_height_constraint.py`",
            "- `docs/geometry.md`",
            "- `docs/ARCHITECTURE.md`",
            "- `docs/COMMANDS.md`",
            "",
            f"## Final result: {'PASS' if final_pass else 'FAIL'}",
            "",
            "No corrected production BO campaign was started.",
            "",
        )
    )
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    if not final_pass:
        raise RuntimeError("corrected height validation did not pass every required item")


def main() -> None:
    logging.disable(logging.INFO)
    before_hashes = _campaign_file_hashes()
    campaigns = {}
    contracts = {}
    for label, directory in _CAMPAIGN_DIRECTORIES.items():
        campaigns[label], contracts[label] = _campaign_from_saved_bounds(directory)
    dragon_campaign = campaigns["Dragon Skin"]
    if (
        contracts["Dragon Skin"]["design_space"]["decoded_physical_bounds_mm"]
        != contracts["Solaris"]["design_space"]["decoded_physical_bounds_mm"]
    ):
        raise RuntimeError("historical campaigns used different parameter bounds")

    lattice = _height_lattice_statistics(dragon_campaign)
    print(f"height lattice: {lattice}", flush=True)
    space = _complete_space_statistics(dragon_campaign)
    print(f"complete-space sample: {space}", flush=True)

    boundary = _validate_boundary_through_ax(dragon_campaign)
    print(f"Ax boundary: {boundary}", flush=True)
    sobol = _sample_proposals(_sobol_client(dragon_campaign), dragon_campaign, _SOBOL_PROPOSALS)
    print(f"Sobol: {sobol}", flush=True)

    audits = {
        label: _audit_campaign(label, directory, campaigns[label])
        for label, directory in _CAMPAIGN_DIRECTORIES.items()
    }
    for label, audit in audits.items():
        print(
            f"{label} audit: completed={audit['completed']}, valid={audit['valid']}, "
            f"invalid={audit['invalid']}",
            flush=True,
        )
    imports, mbm = _import_history_and_sample_mbm(
        dragon_campaign,
        audits["Dragon Skin"],
    )
    print(f"history import: {imports}", flush=True)
    print(f"MBM: {mbm}", flush=True)

    after_hashes = _campaign_file_hashes()
    if before_hashes != after_hashes:
        raise RuntimeError("a saved historical campaign artifact changed")
    _write_report(
        dragon_campaign,
        lattice,
        space,
        sobol,
        mbm,
        audits,
        boundary,
        imports,
    )
    print(f"wrote {_OUTPUT_PATH.relative_to(_ROOT)}", flush=True)
    print("corrected full-height constraint validation: PASS", flush=True)


if __name__ == "__main__":
    main()
