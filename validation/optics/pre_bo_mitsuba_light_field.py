"""Validate pre-BO morphology ranking with an extruded Mitsuba side field."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from fem import solve
from mesh import mesh_settings_for_level
from mesh.indenter import IndenterSettings
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from optics.mitsuba import Camera, MitsubaRenderer, RenderSettings
from optimization.scenarios import ContactScenario
from validation.common.io import atomic_write_json, strict_read_json


OUTPUT_PATH = Path("output/validation/optics/pre_bo_mitsuba_light_field")
CANDIDATE_INPUT = Path(
    "output/validation/optimization/pre_bo_nominal_sweep/inputs/candidate_0049.json"
)
EXTRUSION_LENGTH_MM = 64.8
LED_Z_POSITIONS_MM = (-19.9, -8.9, 2.1, 13.1, 24.1)
CONTACTS = {
    "left_contact": ContactScenario(-3.0, 0.5, 4.0),
    "right_contact": ContactScenario(3.0, 0.5, 4.0),
}
SMOKE_SPP = 256
FINAL_SPP = 8192
SMOKE_SEED = 20260815
FINAL_SEEDS = {
    "nominal": {
        "unloaded": 20260816,
        "left_contact": 20260817,
        "right_contact": 20260818,
    },
    "candidate49": {
        "unloaded": 20260819,
        "left_contact": 20260820,
        "right_contact": 20260821,
    },
}
NOISE_REPEAT_SEED = 20260822
DEFAULT_RESOLUTION = (384, 1024)
DEFAULT_MARGIN_MM = 2.0
REDUCED_ORDER_TV = {
    "nominal": 0.07510662148936212,
    "candidate49": 0.12737679674855767,
}


def scalar_light_field(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert raw nonnegative linear RGB into one scalar light field."""
    image = np.asarray(linear_rgb, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("linear_rgb must have shape (height, width, 3)")
    if not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError("linear_rgb must be finite and nonnegative")
    return np.sum(image, axis=2)


def normalized_total_variation(
    left_field: np.ndarray,
    right_field: np.ndarray,
) -> float:
    """Return normalized spatial total variation between two scalar fields."""
    left = np.asarray(left_field, dtype=float)
    right = np.asarray(right_field, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left_field and right_field must be matching 2D fields")
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        raise ValueError("fields must be finite and nonnegative")
    left_energy = float(np.sum(left))
    right_energy = float(np.sum(right))
    if left_energy <= 0.0 or right_energy <= 0.0:
        raise ValueError("fields must have positive spatial energy")
    return float(
        0.5
        * np.sum(
            np.abs(left / left_energy - right / right_energy),
            dtype=float,
        )
    )


def absolute_difference(left_field: np.ndarray, right_field: np.ndarray) -> np.ndarray:
    """Return an absolute scalar-field difference without energy normalization."""
    left = np.asarray(left_field, dtype=float)
    right = np.asarray(right_field, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("fields must be matching 2D fields")
    difference = np.abs(left - right)
    if not np.all(np.isfinite(difference)):
        raise ValueError("field difference must be finite")
    return difference


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_candidate_parameters(path: Path) -> FingertipParameters:
    payload = strict_read_json(path)
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"candidate artifact has no parameters: {path}")
    return FingertipParameters(**parameters)


def _source_y_mm(tip: Fingertip) -> float:
    """Return the geometric center y-coordinate of the LED package."""
    return float(tip.led_package_geometry.centroid.y)


def _source_positions_mm(
    tip: Fingertip,
    led_z_positions_mm: Sequence[float] = LED_Z_POSITIONS_MM,
) -> tuple[tuple[float, float, float], ...]:
    source_y = _source_y_mm(tip)
    return tuple((0.0, source_y, float(z_mm)) for z_mm in led_z_positions_mm)


def _camera_for_union(
    tips_and_meshes: Sequence[tuple[Fingertip, Any]],
    extrusion_length_mm: float = EXTRUSION_LENGTH_MM,
) -> tuple[Camera, dict[str, Any]]:
    xy_points: list[np.ndarray] = []
    for tip, mesh in tips_and_meshes:
        xy_points.append(np.asarray(mesh.pad.coordinates, dtype=float))
        min_x, min_y, max_x, max_y = tip.geometry.link_geometry.bounds
        xy_points.append(
            np.asarray(
                [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]],
                dtype=float,
            )
        )
    coordinates = np.vstack(xy_points)
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)
    center_y = 0.5 * (float(min_y) + float(max_y))
    y_span = float(max_y - min_y)
    frame_span = max(y_span, extrusion_length_mm) + 2.0 * DEFAULT_MARGIN_MM
    camera = Camera(
        position_mm=(float(max_x) + DEFAULT_MARGIN_MM, center_y, 0.0),
        target_mm=(0.0, center_y, 0.0),
        up=(0.0, 0.0, 1.0),
        resolution_px=DEFAULT_RESOLUTION,
        projection="orthographic",
        orthographic_scale_mm=frame_span,
    )
    return camera, {
        "position_mm": list(camera.position_mm),
        "target_mm": list(camera.target_mm),
        "up": list(camera.up),
        "resolution_px": list(camera.resolution_px),
        "projection": camera.projection,
        "orthographic_scale_mm": camera.orthographic_scale_mm,
        "frame_margin_mm": DEFAULT_MARGIN_MM,
        "union_y_bounds_mm": [float(min_y), float(max_y)],
        "union_z_bounds_mm": [
            -extrusion_length_mm / 2.0,
            extrusion_length_mm / 2.0,
        ],
    }


def _mesh_configuration() -> dict[str, Any]:
    settings = mesh_settings_for_level("medium")
    return asdict(settings)


def _save_render(
    output: Path,
    design_name: str,
    state_name: str,
    result: Any,
) -> dict[str, Any]:
    linear_rgb = np.asarray(result.linear_rgb, dtype=np.float32)
    scalar = scalar_light_field(linear_rgb).astype(np.float32)
    path = output / "fields" / f"{design_name}_{state_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, linear_rgb=linear_rgb, scalar_field=scalar)
    return {
        "path": str(path),
        "shape": list(linear_rgb.shape),
        "spp": result.spp,
        "relative_led_power": result.relative_led_power,
        "total_energy": float(np.sum(scalar, dtype=float)),
    }


def _run_design(
    *,
    design_name: str,
    parameters: FingertipParameters,
    camera: Camera,
    camera_configuration: Mapping[str, Any],
    output: Path,
    led: LED,
    optical: OpticalMaterial,
    extrusion_length_mm: float = EXTRUSION_LENGTH_MM,
    led_z_positions_mm: Sequence[float] = LED_Z_POSITIONS_MM,
    smoke_spp: int = SMOKE_SPP,
    final_spp: int = FINAL_SPP,
    final_seeds: Mapping[str, int] | None = None,
    noise_repeat_seed: int = NOISE_REPEAT_SEED,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "running",
        "parameters": asdict(parameters),
        "source_positions_mm": None,
        "states": {},
        "fem": {},
        "failure_message": None,
    }
    try:
        tip = Fingertip(parameters, led=led, optical=optical)
        mesh = tip.mesh(mesh_settings_for_level("medium"))
        source_positions = _source_positions_mm(tip, led_z_positions_mm)
        result["source_positions_mm"] = [list(position) for position in source_positions]
        result["source_center_y_mm"] = _source_y_mm(tip)
        renderer = MitsubaRenderer(
            tip,
            mesh,
            depth_mm=extrusion_length_mm,
            camera=camera,
            settings=RenderSettings(
                variant="scalar_rgb",
                spp=smoke_spp,
                max_depth=12,
                optical_depth_mm=extrusion_length_mm,
                point_emitter_scale=30.0,
                source_epsilon_mm=0.0,
            ),
            source_positions_mm=source_positions,
        )
        state_meshes: dict[str, Any] = {"unloaded": None}
        for state_index, state_name in enumerate(
            ("unloaded", "left_contact", "right_contact")
        ):
            if state_name != "unloaded":
                scenario = CONTACTS[state_name]
                fem_started = time.perf_counter()
                fea = solve(
                    tip,
                    mesh,
                    indentation=scenario.indentation_mm,
                    surface_x_mm=scenario.location_x_mm,
                    steps=48,
                    indenter=IndenterSettings(radius_mm=scenario.indenter_radius_mm),
                    internal_contact="three_pairs",
                )
                fem_record = {
                    "status": "success" if fea.converged else "failure",
                    "converged": fea.converged,
                    "wall_time_seconds": time.perf_counter() - fem_started,
                    "scenario": asdict(scenario),
                    "reaction_force_n": fea.reaction_force,
                }
                result["fem"][state_name] = fem_record
                if not fea.converged:
                    raise RuntimeError(
                        f"FEM did not converge for {state_name}: "
                        f"{fea.details.get('failure_reason', 'unknown reason')}"
                    )
                state_meshes[state_name] = fea.deformed_mesh

            smoke_started = time.perf_counter()
            smoke_seed = SMOKE_SEED + state_index
            smoke = renderer.render(
                mesh=state_meshes[state_name],
                spp=smoke_spp,
                seed=smoke_seed,
            )
            smoke_record = {
                "status": "success",
                "spp": smoke_spp,
                "seed": smoke_seed,
                "wall_time_seconds": time.perf_counter() - smoke_started,
                "shape": list(smoke.linear_rgb.shape),
            }
            final_started = time.perf_counter()
            selected_final_seeds = (
                FINAL_SEEDS[design_name] if final_seeds is None else final_seeds
            )
            final_seed = selected_final_seeds[state_name]
            final = renderer.render(
                mesh=state_meshes[state_name],
                spp=final_spp,
                seed=final_seed,
            )
            final_record = _save_render(output, design_name, state_name, final)
            final_record.update(
                {
                    "status": "success",
                    "spp": final_spp,
                    "seed": final_seed,
                    "wall_time_seconds": time.perf_counter() - final_started,
                }
            )
            result["states"][state_name] = {
                "smoke": smoke_record,
                "final": final_record,
            }
            if state_name == "left_contact":
                repeat = renderer.render(
                    mesh=state_meshes[state_name],
                    spp=final_spp,
                    seed=noise_repeat_seed,
                )
                repeat_field = scalar_light_field(repeat.linear_rgb)
                final_field = scalar_light_field(final.linear_rgb)
                result["noise_floor_tv"] = normalized_total_variation(
                    final_field,
                    repeat_field,
                )
                result["noise_repeat"] = {
                    "spp": final_spp,
                    "seed": noise_repeat_seed,
                    "wall_time_seconds": time.perf_counter() - final_started,
                }
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "failure"
        result["failure_message"] = f"{type(exc).__name__}: {exc}"
    result["wall_time_seconds"] = time.perf_counter() - started
    return result


def _field_from_record(record: Mapping[str, Any]) -> np.ndarray:
    with np.load(record["path"], allow_pickle=False) as data:
        return np.asarray(data["scalar_field"], dtype=float)


def _save_difference(
    output: Path,
    design_name: str,
    left: np.ndarray,
    right: np.ndarray,
    unloaded: np.ndarray,
) -> dict[str, Any]:
    left_right = absolute_difference(left, right).astype(np.float32)
    unloaded_left = absolute_difference(unloaded, left).astype(np.float32)
    unloaded_right = absolute_difference(unloaded, right).astype(np.float32)
    path = output / "differences" / f"{design_name}_differences.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        left_right=left_right,
        unloaded_left=unloaded_left,
        unloaded_right=unloaded_right,
    )
    return {
        "path": str(path),
        "left_right_max": float(np.max(left_right)),
        "unloaded_left_max": float(np.max(unloaded_left)),
        "unloaded_right_max": float(np.max(unloaded_right)),
    }


def _compute_metrics(
    output: Path,
    designs: Mapping[str, Mapping[str, Any]],
    reduced_order_tv: Mapping[str, float] = REDUCED_ORDER_TV,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for design_name, design in designs.items():
        states = design["states"]
        unloaded = _field_from_record(states["unloaded"]["final"])
        left = _field_from_record(states["left_contact"]["final"])
        right = _field_from_record(states["right_contact"]["final"])
        left_energy = float(np.sum(left, dtype=float))
        right_energy = float(np.sum(right, dtype=float))
        differences = _save_difference(output, design_name, left, right, unloaded)
        metrics[design_name] = {
            "reduced_order_tv": reduced_order_tv[design_name],
            "mitsuba_side_field_tv": normalized_total_variation(left, right),
            "left_total_energy": left_energy,
            "right_total_energy": right_energy,
            "relative_total_energy_difference": (
                (right_energy - left_energy) / left_energy
                if left_energy > 0.0
                else None
            ),
            "difference_fields": differences,
        }
    return metrics


def _make_figures(
    output: Path,
    designs: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fields: dict[str, dict[str, np.ndarray]] = {}
    for design_name, design in designs.items():
        fields[design_name] = {
            state_name: _field_from_record(design["states"][state_name]["final"])
            for state_name in ("unloaded", "left_contact", "right_contact")
        }
    difference_fields: dict[str, np.ndarray] = {}
    for design_name, design in designs.items():
        with np.load(
            design["metrics_difference_path"], allow_pickle=False
        ) as data:
            difference_fields[design_name] = np.asarray(data["left_right"], dtype=float)

    raw_max = max(
        float(np.max(field))
        for state_fields in fields.values()
        for field in state_fields.values()
    )
    difference_max = max(float(np.max(field)) for field in difference_fields.values())
    figure_path = output / "figures" / "nominal_vs_candidate49_light_fields.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(13, 16), constrained_layout=True)
    panel_names = ("unloaded", "left_contact", "right_contact")
    for row, design_name in enumerate(("nominal", "candidate49")):
        for column, state_name in enumerate(panel_names):
            axes[row, column].imshow(
                fields[design_name][state_name],
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=raw_max,
                aspect="auto",
            )
            axes[row, column].set_title(f"{design_name} {state_name}")
            axes[row, column].set_xlabel("projected y pixel")
            axes[row, column].set_ylabel("projected z pixel")
        axes[row, 3].imshow(
            difference_fields[design_name],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=difference_max,
            aspect="auto",
        )
        axes[row, 3].set_title(f"{design_name} left-right difference")
        axes[row, 3].set_xlabel("projected y pixel")
        axes[row, 3].set_ylabel("projected z pixel")
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    quantitative_path = output / "figures" / "reduced_vs_mitsuba_tv.png"
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    positions = np.arange(2)
    width = 0.36
    reduced = [metrics[name]["reduced_order_tv"] for name in ("nominal", "candidate49")]
    mitsuba = [metrics[name]["mitsuba_side_field_tv"] for name in ("nominal", "candidate49")]
    axis.bar(positions - width / 2.0, reduced, width, label="reduced 2D TV")
    axis.bar(positions + width / 2.0, mitsuba, width, label="Mitsuba 3D side-field TV")
    axis.set_xticks(positions, ["nominal", "candidate 49"])
    axis.set_ylabel("normalized spatial TV")
    axis.legend()
    figure.savefig(quantitative_path, dpi=150)
    plt.close(figure)
    return {
        "main_comparison": str(figure_path),
        "quantitative_comparison": str(quantitative_path),
    }


def _write_research_log(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    metrics = summary.get("metrics", {})
    nominal = metrics.get("nominal", {})
    candidate = metrics.get("candidate49", {})
    ranking = summary.get("ranking_preserved")
    text = f"""# Pre-BO Mitsuba 3D light-field validation

## Purpose

This validation compares the current nominal morphology with exact pre-BO
sweep candidate 49 using an extruded 2D deformation and a 3D volumetric
Mitsuba model with the actual five-LED longitudinal layout. Candidate 49 was
selected because it was the best successful point in the pre-BO sweep.

This validation uses an ideal orthographic side-field sampler rather than a
physical camera model. The deformation is two-dimensional and uniformly
extruded along the longitudinal direction.

## Provenance and designs

- Git revision: `{summary.get('git_revision')}`
- Nominal parameters: `{json.dumps(summary['designs']['nominal']['parameters'], sort_keys=True)}`
- Candidate 49 parameters: `{json.dumps(summary['designs']['candidate49']['parameters'], sort_keys=True)}`
- Reduced-order nominal TV: `{REDUCED_ORDER_TV['nominal']:.10f}`
- Reduced-order candidate 49 TV: `{REDUCED_ORDER_TV['candidate49']:.10f}`

## Optical and mechanical protocol

- Extrusion length: `{EXTRUSION_LENGTH_MM} mm`, centered at `z=0`.
- LED z positions: `{summary['led_z_positions_mm']}`.
- LED source positions are package centers with `x=0`; the exact per-design
  coordinates are recorded in `summary.json`.
- All five LEDs are on simultaneously with identical default LED RGB and
  relative power. They are separate point sources, not one five-times source.
- Contact states: unloaded, left shallow `(-3.0, 0.5, 4.0)`, and right
  shallow `(3.0, 0.5, 4.0)` in `(location_x_mm, indentation_mm,
  indenter_radius_mm)`.
- FEM: medium mesh, `fem_steps=48`, `internal_contact=three_pairs`.
- Optical material and LED properties are identical between designs.
- Sampler configuration is fixed across both morphologies and all states:
  `{json.dumps(summary['camera'], sort_keys=True)}`

## Results

| Morphology | Reduced 2D TV | Mitsuba side-field TV | Left energy | Right energy | Relative energy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | {nominal.get('reduced_order_tv', float('nan')):.10f} | {nominal.get('mitsuba_side_field_tv', float('nan')):.10f} | {nominal.get('left_total_energy', float('nan')):.8g} | {nominal.get('right_total_energy', float('nan')):.8g} | {nominal.get('relative_total_energy_difference', float('nan')):.8g} |
| Candidate 49 | {candidate.get('reduced_order_tv', float('nan')):.10f} | {candidate.get('mitsuba_side_field_tv', float('nan')):.10f} | {candidate.get('left_total_energy', float('nan')):.8g} | {candidate.get('right_total_energy', float('nan')):.8g} | {candidate.get('relative_total_energy_difference', float('nan')):.8g} |

- 256 spp smoke renders: `{summary.get('smoke_status')}`.
- The initial 2048 spp comparison produced nominal TV `0.0475354843`,
  candidate-49 TV `0.0369228490`, and same-state noise TV `0.0470995254`.
  Because the morphology gap was below that noise floor, the final comparison
  was rerun at the higher spp below before interpretation; retained raw fields
  are from the higher-spp run.
- {summary.get('render', {}).get('final_spp')} spp final renders: `{summary.get('final_status')}`.
- Same-state repeated loaded render TV noise floor: `{summary.get('noise_floor_tv')}`.
- Absolute nominal-versus-candidate Mitsuba TV gap: `{abs(nominal.get('mitsuba_side_field_tv', float('nan')) - candidate.get('mitsuba_side_field_tv', float('nan'))):.10f}`.
- Final render wall times and FEM wall times are recorded in `summary.json`.
- Ranking preserved: **{ranking}**.

## Figures and artifacts

- Main comparison: `{summary.get('figures', {}).get('main_comparison')}`.
- Quantitative comparison: `{summary.get('figures', {}).get('quantitative_comparison')}`.
- Raw final linear RGB and scalar fields: `output/validation/optics/pre_bo_mitsuba_light_field/fields/`.
- Absolute difference fields: `output/validation/optics/pre_bo_mitsuba_light_field/differences/`.

## Interpretation and limitations

The primary claim is ranking preservation between the two morphologies, not
agreement of absolute reduced-order and 3D TV values. The side-field sampler
is an ideal numerical orthographic field sampler, not a calibrated camera or
lens model. The mechanical state is an extruded 2D FEM deformation, not full
3D contact mechanics. No optical calibration or parameter tuning was applied
to candidate 49, and no BO configuration was changed. In this run the
candidate-49 Mitsuba TV is below nominal, and the absolute morphology gap is
smaller than the repeated loaded-state noise floor; the reduced-order ranking
is therefore not supported by this 3D side-field measurement.
"""
    path.write_text(text, encoding="utf-8")


def run_validation(output: Path = OUTPUT_PATH) -> int:
    output.mkdir(parents=True, exist_ok=True)
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
            ((nominal_tip, nominal_mesh), (candidate_tip, candidate_mesh))
        )
        summary: dict[str, Any] = {
            "status": "RUNNING",
            "created_at": _now(),
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
                "final_spp": FINAL_SPP,
                "smoke_seed": SMOKE_SEED,
                "final_seeds": FINAL_SEEDS,
                "noise_repeat_seed": NOISE_REPEAT_SEED,
                "point_emitter_scale": 30.0,
                "source_epsilon_mm": 0.0,
            },
            "reduced_order_tv": REDUCED_ORDER_TV,
        }
        atomic_write_json(output / "summary.json", summary)

        for design_name, parameters in (
            ("nominal", nominal_parameters),
            ("candidate49", candidate_parameters),
        ):
            tip = nominal_tip if design_name == "nominal" else candidate_tip
            mesh = nominal_mesh if design_name == "nominal" else candidate_mesh
            design_result = _run_design(
                design_name=design_name,
                parameters=parameters,
                camera=camera,
                camera_configuration=camera_configuration,
                output=output,
                led=led,
                optical=optical,
            )
            design_result["source_positions_mm"] = [
                list(position) for position in _source_positions_mm(tip)
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
        for design_name, design_metrics in summary["metrics"].items():
            summary["designs"][design_name]["metrics_difference_path"] = (
                output / "differences" / f"{design_name}_differences.npz"
            ).as_posix()
        summary["noise_floor_tv"] = summary["designs"]["nominal"]["noise_floor_tv"]
        summary["figures"] = _make_figures(output, summary["designs"], summary["metrics"])
        summary["ranking_preserved"] = (
            summary["metrics"]["candidate49"]["mitsuba_side_field_tv"]
            > summary["metrics"]["nominal"]["mitsuba_side_field_tv"]
        )
        summary["status"] = "COMPLETE"
        summary["completed_at"] = _now()
        atomic_write_json(output / "summary.json", summary)
        _write_research_log(
            Path("docs/validation/pre-bo-mitsuba-light-field.md"),
            summary,
        )
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
    arguments = parser.parse_args()
    return run_validation(arguments.output_directory.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
