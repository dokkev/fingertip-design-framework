"""Validate one LED longitudinal pitch with the extruded Mitsuba side field."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from mesh import mesh_settings_for_level
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from validation.common.io import atomic_write_json, strict_read_json
from validation.optics.pre_bo_mitsuba_light_field import (
    CANDIDATE_INPUT,
    CONTACTS,
    REDUCED_ORDER_TV,
    SMOKE_SEED,
    _camera_for_union,
    _compute_metrics,
    _git_revision,
    _load_candidate_parameters,
    _make_figures,
    _run_design,
    _source_positions_mm,
)


OUTPUT_PATH = Path("output/validation/optics/pre_bo_mitsuba_single_cell")
RESEARCH_LOG_PATH = Path("docs/validation/pre-bo-mitsuba-single-cell.md")
EXTRUSION_LENGTH_MM = 11.0
LED_Z_POSITIONS_MM = (0.0,)
SMOKE_SPP = 256
FINAL_SPP = 2048
FINAL_SEEDS = {
    "nominal": {
        "unloaded": 20260824,
        "left_contact": 20260825,
        "right_contact": 20260826,
    },
    "candidate49": {
        "unloaded": 20260827,
        "left_contact": 20260828,
        "right_contact": 20260829,
    },
}
NOISE_REPEAT_SEED = 20260830


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_research_log(
    path: Path,
    summary: Mapping[str, Any],
    initial_summary: Mapping[str, Any] | None,
) -> None:
    metrics = summary.get("metrics", {})
    nominal = metrics.get("nominal", {})
    candidate = metrics.get("candidate49", {})
    designs = summary.get("designs", {})
    initial_metrics = (initial_summary or {}).get("metrics", {})
    initial_nominal = initial_metrics.get("nominal", {})
    initial_candidate = initial_metrics.get("candidate49", {})
    initial_note = (
        "No escalation was required; this is the initial 2048 spp run."
        if initial_summary is None
        else (
            "The initial 2048 spp summary is retained at "
            f"{summary.get('initial_summary_path')}. It measured nominal TV "
            f"{initial_nominal.get('mitsuba_side_field_tv')} and candidate 49 TV "
            f"{initial_candidate.get('mitsuba_side_field_tv')}, with nominal "
            f"noise {initial_summary.get('noise_floor_tv')} and candidate 49 "
            f"noise {initial_summary.get('noise_floor_tv_by_design', {}).get('candidate49')}."
        )
    )
    source_coordinates = {
        name: design.get("source_positions_mm")
        for name, design in designs.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Pre-BO Mitsuba single-LED cell validation

## Purpose

This experiment compares nominal morphology with exact pre-BO sweep candidate
49 using one representative longitudinal LED cell. It tests whether the 2D
morphology ranking survives 3D optical propagation through an 11 mm uniform
extrusion. The full 64.8 mm/five-LED configuration is intentionally excluded.

This validation uses an ideal orthographic side-field sampler rather than a
physical camera model. The deformation is two-dimensional and uniformly
extruded along the longitudinal direction; it is not full 3D contact mechanics.

## Protocol

- Git revision: {summary.get('git_revision')}
- Exact candidate source: {summary.get('candidate_input')}
- Nominal parameters: {json.dumps(designs.get('nominal', {}).get('parameters', {}), sort_keys=True)}
- Candidate 49 parameters: {json.dumps(designs.get('candidate49', {}).get('parameters', {}), sort_keys=True)}
- Reduced-order TVs: nominal {REDUCED_ORDER_TV['nominal']:.10f}, candidate 49 {REDUCED_ORDER_TV['candidate49']:.10f}
- Extrusion: {summary.get('extrusion_length_mm')} mm, z bounds {summary.get('extrusion_z_bounds_mm')}; point LED at z=0.
- Exact source coordinates: {json.dumps(source_coordinates, sort_keys=True)}
- Source x=0 and y is the geometric center of each morphology's physical LED package.
- The cell represents the 9 mm package plus 2 mm gap, bounded at pitch midpoints.
- States: unloaded, left shallow (-3.0, 0.5, 4.0), right shallow (3.0, 0.5, 4.0).
- FEM: medium mesh, 48 steps, internal_contact=three_pairs.
- Optical material, LED RGB, and LED power are identical between morphologies.
- Fixed sampler: {json.dumps(summary.get('camera', {}), sort_keys=True)}

## Results

| Morphology | Reduced 2D TV | Single-cell Mitsuba TV | Left energy | Right energy | Relative energy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | {nominal.get('reduced_order_tv', float('nan')):.10f} | {nominal.get('mitsuba_side_field_tv', float('nan')):.10f} | {nominal.get('left_total_energy', float('nan')):.8g} | {nominal.get('right_total_energy', float('nan')):.8g} | {nominal.get('relative_total_energy_difference', float('nan')):.8g} |
| Candidate 49 | {candidate.get('reduced_order_tv', float('nan')):.10f} | {candidate.get('mitsuba_side_field_tv', float('nan')):.10f} | {candidate.get('left_total_energy', float('nan')):.8g} | {candidate.get('right_total_energy', float('nan')):.8g} | {candidate.get('relative_total_energy_difference', float('nan')):.8g} |

- 256 spp smoke status: {summary.get('smoke_status')}.
- Final spp: {summary.get('render', {}).get('final_spp')}.
- Same-state left-contact noise TV: nominal {summary.get('noise_floor_tv_by_design', {}).get('nominal')}, candidate 49 {summary.get('noise_floor_tv_by_design', {}).get('candidate49')}, maximum {summary.get('noise_floor_tv')}.
- Absolute morphology TV gap: {summary.get('morphology_tv_gap')}.
- Ranking preserved: {summary.get('ranking_preserved')}.
- SPP decision: {initial_note}

## Artifacts

- Main comparison: {summary.get('figures', {}).get('main_comparison')}
- Quantitative comparison: {summary.get('figures', {}).get('quantitative_comparison')}
- Raw fields: {summary.get('output_directory')}/fields/
- Difference fields: {summary.get('output_directory')}/differences/

## Interpretation and limitations

The primary claim is ranking preservation between the reduced model and this
single-cell side-field measurement, not equality of absolute TV values. This
test excludes five-LED interaction, PCB end effects, longitudinal illumination
nonuniformity, camera orientation, and full fingertip length. It is not a
camera-performance result or a full-3D-mechanics result. Bayesian optimization
remains on hold until this single-cell result is interpreted.
""",
        encoding="utf-8",
    )


def run_validation(
    output: Path = OUTPUT_PATH,
    *,
    final_spp: int = FINAL_SPP,
    initial_summary_path: Path | None = None,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    initial_summary = None
    if initial_summary_path is not None:
        initial_summary = strict_read_json(initial_summary_path)
    try:
        candidate_parameters = _load_candidate_parameters(CANDIDATE_INPUT)
        nominal_parameters = FingertipParameters()
        led = LED()
        optical = OpticalMaterial()
        mesh_settings = mesh_settings_for_level("medium")
        nominal_tip = Fingertip(nominal_parameters, led=led, optical=optical)
        candidate_tip = Fingertip(candidate_parameters, led=led, optical=optical)
        nominal_mesh = nominal_tip.mesh(mesh_settings)
        candidate_mesh = candidate_tip.mesh(mesh_settings)
        camera, camera_configuration = _camera_for_union(
            ((nominal_tip, nominal_mesh), (candidate_tip, candidate_mesh)),
            extrusion_length_mm=EXTRUSION_LENGTH_MM,
        )
        summary: dict[str, Any] = {
            "status": "RUNNING",
            "created_at": _now(),
            "output_directory": str(output),
            "git_revision": _git_revision(),
            "candidate_input": str(CANDIDATE_INPUT),
            "designs": {
                "nominal": {"parameters": asdict(nominal_parameters)},
                "candidate49": {"parameters": asdict(candidate_parameters)},
            },
            "extrusion_length_mm": EXTRUSION_LENGTH_MM,
            "extrusion_z_bounds_mm": [
                -EXTRUSION_LENGTH_MM / 2.0,
                EXTRUSION_LENGTH_MM / 2.0,
            ],
            "led_z_positions_mm": list(LED_Z_POSITIONS_MM),
            "led": asdict(led),
            "optical": asdict(optical),
            "camera": camera_configuration,
            "mesh_settings": asdict(mesh_settings),
            "fem": {
                "steps": 48,
                "internal_contact": "three_pairs",
                "contacts": {name: asdict(value) for name, value in CONTACTS.items()},
            },
            "render": {
                "variant": "scalar_rgb",
                "smoke_spp": SMOKE_SPP,
                "final_spp": final_spp,
                "smoke_seed": SMOKE_SEED,
                "final_seeds": FINAL_SEEDS,
                "noise_repeat_seed": NOISE_REPEAT_SEED,
                "point_emitter_scale": 30.0,
                "source_epsilon_mm": 0.0,
            },
            "reduced_order_tv": REDUCED_ORDER_TV,
        }
        if initial_summary_path is not None:
            summary["initial_summary_path"] = str(initial_summary_path)
        atomic_write_json(output / "summary.json", summary)

        for design_name, parameters, tip, mesh in (
            ("nominal", nominal_parameters, nominal_tip, nominal_mesh),
            ("candidate49", candidate_parameters, candidate_tip, candidate_mesh),
        ):
            design_result = _run_design(
                design_name=design_name,
                parameters=parameters,
                camera=camera,
                camera_configuration=camera_configuration,
                output=output,
                led=led,
                optical=optical,
                extrusion_length_mm=EXTRUSION_LENGTH_MM,
                led_z_positions_mm=LED_Z_POSITIONS_MM,
                smoke_spp=SMOKE_SPP,
                final_spp=final_spp,
                final_seeds=FINAL_SEEDS[design_name],
                noise_repeat_seed=NOISE_REPEAT_SEED,
            )
            design_result["source_positions_mm"] = [
                list(position)
                for position in _source_positions_mm(tip, LED_Z_POSITIONS_MM)
            ]
            summary["designs"][design_name] = design_result
            atomic_write_json(output / "summary.json", summary)
            if design_result["status"] != "success":
                summary["status"] = "FAILED"
                summary["failure_message"] = (
                    f"{design_name}: {design_result.get('failure_message')}"
                )
                atomic_write_json(output / "summary.json", summary)
                return 2

        summary["smoke_status"] = "success"
        summary["final_status"] = "success"
        summary["metrics"] = _compute_metrics(output, summary["designs"])
        for design_name in summary["metrics"]:
            summary["designs"][design_name]["metrics_difference_path"] = (
                output / "differences" / f"{design_name}_differences.npz"
            ).as_posix()
        noise_by_design = {
            name: summary["designs"][name]["noise_floor_tv"]
            for name in ("nominal", "candidate49")
        }
        summary["noise_floor_tv_by_design"] = noise_by_design
        summary["noise_floor_tv"] = max(noise_by_design.values())
        summary["morphology_tv_gap"] = abs(
            summary["metrics"]["candidate49"]["mitsuba_side_field_tv"]
            - summary["metrics"]["nominal"]["mitsuba_side_field_tv"]
        )
        summary["figures"] = _make_figures(
            output, summary["designs"], summary["metrics"]
        )
        summary["ranking_preserved"] = (
            summary["metrics"]["candidate49"]["mitsuba_side_field_tv"]
            > summary["metrics"]["nominal"]["mitsuba_side_field_tv"]
        )
        summary["status"] = "COMPLETE"
        summary["completed_at"] = _now()
        atomic_write_json(output / "summary.json", summary)
        _write_research_log(RESEARCH_LOG_PATH, summary, initial_summary)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "completed_at": _now(),
            "failure_message": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(output / "summary.json", failure)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--final-spp", type=int, default=FINAL_SPP)
    parser.add_argument("--initial-summary", type=Path)
    arguments = parser.parse_args()
    return run_validation(
        arguments.output_directory.expanduser().resolve(),
        final_spp=arguments.final_spp,
        initial_summary_path=(
            arguments.initial_summary.expanduser().resolve()
            if arguments.initial_summary is not None
            else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
