"""Run one nominal and one diagnostic optomechanical design evaluation."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from mesh import mesh_settings_for_level
from model import FingertipParameters, LED, OpticalMaterial
from optics import TraceSettings
from optimization.evaluator import DesignEvaluation, DesignEvaluator
from optimization.scenarios import ContactScenario, ScenarioGrid, ScenarioPair


OUTPUT_PATH = Path("output/validation/optimization/dry_run/result.json")


def _scenario_to_dict(scenario: ContactScenario) -> dict[str, float]:
    return {
        "location_x_mm": scenario.location_x_mm,
        "indentation_mm": scenario.indentation_mm,
        "indenter_radius_mm": scenario.indenter_radius_mm,
    }


def _pair_to_dict(pair: ScenarioPair) -> dict[str, Any]:
    return {
        "first": _scenario_to_dict(pair.first),
        "second": _scenario_to_dict(pair.second),
        "axis": pair.axis,
    }


def _scenario_evaluation_to_dict(result: Any) -> dict[str, Any]:
    return {
        **_scenario_to_dict(result.scenario),
        "reaction_force_n": result.reaction_force_n,
        "reference_field_difference": result.reference_field_difference,
        "centroid_shift_mm": result.centroid_shift_mm,
        "escaped_fraction_change": result.escaped_fraction_change,
        "absorbed_fraction_change": result.absorbed_fraction_change,
    }


def _pair_evaluation_to_dict(result: Any) -> dict[str, Any]:
    return {
        **_pair_to_dict(result.pair),
        "separability": result.separability,
    }


def _attempted_scenario_count(
    evaluation: DesignEvaluation,
    total_scenarios: int,
) -> int:
    completed = len(evaluation.scenarios)
    if completed == total_scenarios:
        return completed
    if evaluation.failure_message and "scenario " in evaluation.failure_message:
        return min(completed + 1, total_scenarios)
    return completed


def _evaluation_to_dict(
    evaluation: DesignEvaluation,
    *,
    wall_time_seconds: float,
    total_scenarios: int,
) -> dict[str, Any]:
    scenarios_attempted = _attempted_scenario_count(evaluation, total_scenarios)
    return {
        "status": evaluation.status,
        "score": evaluation.score,
        "minimum_separability": evaluation.minimum_separability,
        "mean_separability": evaluation.mean_separability,
        "median_separability": evaluation.median_separability,
        "minimum_location_separability": evaluation.minimum_location_separability,
        "minimum_indentation_separability": evaluation.minimum_indentation_separability,
        "minimum_radius_separability": evaluation.minimum_radius_separability,
        "minimum_reference_field_difference": evaluation.minimum_reference_field_difference,
        "limiting_pair": (
            None
            if evaluation.limiting_pair is None
            else _pair_to_dict(evaluation.limiting_pair)
        ),
        "failure_message": evaluation.failure_message,
        "wall_time_seconds": wall_time_seconds,
        "scenarios_attempted": scenarios_attempted,
        "fem_solves_attempted": scenarios_attempted,
        "scenarios": [
            _scenario_evaluation_to_dict(result)
            for result in evaluation.scenarios
        ],
        "pairs": [
            _pair_evaluation_to_dict(result) for result in evaluation.pairs
        ],
    }


def _run_design(
    name: str,
    parameters: FingertipParameters,
    evaluator: DesignEvaluator,
) -> dict[str, Any]:
    start = time.perf_counter()
    evaluation = evaluator.evaluate(parameters)
    wall_time_seconds = time.perf_counter() - start
    result = {
        "name": name,
        "parameters": asdict(parameters),
        "attempted": True,
        "evaluation": _evaluation_to_dict(
            evaluation,
            wall_time_seconds=wall_time_seconds,
            total_scenarios=len(evaluator.scenario_grid.scenarios),
        ),
    }
    _print_design_summary(result)
    return result


def _not_run_design(parameters: FingertipParameters) -> dict[str, Any]:
    return {
        "name": "void_width_1.5_diagnostic",
        "parameters": asdict(parameters),
        "attempted": False,
        "evaluation": {
            "status": "not_run",
            "failure_message": (
                "Nominal evaluation failed; diagnostic perturbation was not run."
            ),
            "wall_time_seconds": None,
            "scenarios_attempted": 0,
            "fem_solves_attempted": 0,
            "scenarios": [],
            "pairs": [],
        },
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _pair_label(pair: dict[str, Any] | None) -> str:
    if pair is None:
        return "n/a"
    first = pair["first"]
    second = pair["second"]
    return (
        f"{pair['axis']} ("
        f"{first['location_x_mm']:.3g}, {first['indentation_mm']:.3g}, "
        f"{first['indenter_radius_mm']:.3g}) -> ("
        f"{second['location_x_mm']:.3g}, {second['indentation_mm']:.3g}, "
        f"{second['indenter_radius_mm']:.3g})"
    )


def _print_design_summary(result: dict[str, Any]) -> None:
    evaluation = result["evaluation"]
    print(f"design: {result['name']}")
    print(f"  status: {evaluation['status']}")
    print(f"  wall time [s]: {_format_metric(evaluation['wall_time_seconds'])}")
    print(f"  scenarios attempted: {evaluation['scenarios_attempted']}")
    print(
        "  minimum separability: "
        f"{_format_metric(evaluation.get('minimum_separability'))}"
    )
    print(
        "  mean separability: "
        f"{_format_metric(evaluation.get('mean_separability'))}"
    )
    print(
        "  location minimum: "
        f"{_format_metric(evaluation.get('minimum_location_separability'))}"
    )
    print(
        "  indentation minimum: "
        f"{_format_metric(evaluation.get('minimum_indentation_separability'))}"
    )
    print(
        "  radius minimum: "
        f"{_format_metric(evaluation.get('minimum_radius_separability'))}"
    )
    print(
        "  minimum reference-field difference: "
        f"{_format_metric(evaluation.get('minimum_reference_field_difference'))}"
    )
    print(f"  limiting pair: {_pair_label(evaluation.get('limiting_pair'))}")
    if evaluation.get("failure_message"):
        print(f"  failure: {evaluation['failure_message']}")


def _configuration(
    scenario_grid: ScenarioGrid,
    mesh_settings: Any,
    trace_settings: TraceSettings,
    led: LED,
    optical: OpticalMaterial,
) -> dict[str, Any]:
    return {
        "scenario_grid": {
            "locations_x_mm": list(scenario_grid.locations_x_mm),
            "indentations_mm": list(scenario_grid.indentations_mm),
            "indenter_radii_mm": list(scenario_grid.indenter_radii_mm),
            "scenario_count": len(scenario_grid.scenarios),
            "pair_count": len(scenario_grid.adjacent_pairs),
        },
        "mesh_settings": asdict(mesh_settings),
        "trace_settings": asdict(trace_settings),
        "led": asdict(led),
        "optical": asdict(optical),
        "fem_steps": 48,
        "internal_contact": "three_pairs",
        "protocol": "temporary optomechanical dry-run protocol",
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Evaluate the nominal design and one void-width diagnostic sample."""
    nominal_parameters = FingertipParameters()
    perturbed_parameters = replace(nominal_parameters, void_width=1.5)
    scenario_grid = ScenarioGrid(
        locations_x_mm=(-3.0, 3.0),
        indentations_mm=(0.5, 1.0),
        indenter_radii_mm=(4.0,),
    )
    mesh_settings = mesh_settings_for_level("medium")
    trace_settings = TraceSettings()
    led = LED()
    optical = OpticalMaterial()
    evaluator = DesignEvaluator(
        scenario_grid,
        mesh_settings=mesh_settings,
        trace_settings=trace_settings,
        led=led,
        optical=optical,
        fem_steps=48,
        internal_contact="three_pairs",
    )

    nominal_result = _run_design("nominal", nominal_parameters, evaluator)
    if nominal_result["evaluation"]["status"] == "success":
        perturbed_result = _run_design(
            "void_width_1.5_diagnostic",
            perturbed_parameters,
            evaluator,
        )
    else:
        perturbed_result = _not_run_design(perturbed_parameters)

    nominal_evaluation = nominal_result["evaluation"]
    perturbed_evaluation = perturbed_result["evaluation"]
    delta = None
    if (
        nominal_evaluation["status"] == "success"
        and perturbed_evaluation["status"] == "success"
    ):
        delta = (
            perturbed_evaluation["minimum_separability"]
            - nominal_evaluation["minimum_separability"]
        )
        print(f"delta minimum separability: {delta:.6g}")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nominal_parameters": asdict(nominal_parameters),
        "perturbed_parameters": asdict(perturbed_parameters),
        "configuration": _configuration(
            scenario_grid,
            mesh_settings,
            trace_settings,
            led,
            optical,
        ),
        "designs": {
            "nominal": nominal_result,
            "void_width_1.5_diagnostic": perturbed_result,
        },
        "minimum_separability_delta_perturbed_minus_nominal": delta,
        "total_real_fem_solves": sum(
            design["evaluation"]["fem_solves_attempted"]
            for design in (nominal_result, perturbed_result)
        ),
    }
    _write_result(OUTPUT_PATH, result)
    print(f"result: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
