"""Run one explicitly requested nominal morphology evaluation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from mesh import mesh_settings_for_level
from model import FingertipParameters, LED, OpticalMaterial
from optics.transport3d import Transport3DSettings
from optimization.evaluator import DesignEvaluation, DesignEvaluator
from optimization.scenarios import ScenarioGrid


OUTPUT_PATH = Path("output/validation/optimization/dry_run/result.json")


def _state_to_dict(result: Any) -> dict[str, Any]:
    state = result.state
    contact_diagnostics = dict(result.contact_diagnostics)
    contact_diagnostics.pop("exact_indenter_pose", None)
    return {
        "location_x_mm": state.location_x_mm,
        "indentation_mm": state.indentation_mm,
        "indenter_radius_mm": state.indenter_radius_mm,
        "reaction_force_n": result.reaction_force_n,
        "contact_metric": result.contact_metric,
        "contact_diagnostics": contact_diagnostics,
        "optical_diagnostics": result.optical_diagnostics,
    }


def _evaluation_to_dict(
    evaluation: DesignEvaluation,
    *,
    wall_time_seconds: float,
    total_states: int,
) -> dict[str, Any]:
    return {
        "status": evaluation.status,
        "score": evaluation.score,
        "minimum_auc": evaluation.minimum_auc,
        "mean_auc": evaluation.mean_auc,
        "median_auc": evaluation.median_auc,
        "minimum_raw_contact_metric": evaluation.minimum_raw_contact_metric,
        "mean_raw_contact_metric": evaluation.mean_raw_contact_metric,
        "limiting_trajectory": None
        if evaluation.limiting_trajectory is None
        else {
            "location_x_mm": evaluation.limiting_trajectory.location_x_mm,
            "diameter_mm": evaluation.limiting_trajectory.diameter_mm,
        },
        "limiting_diameter_mm": evaluation.limiting_diameter_mm,
        "limiting_location_x_mm": evaluation.limiting_location_x_mm,
        "limiting_depth_mm": evaluation.limiting_depth_mm,
        "failure_message": evaluation.failure_message,
        "wall_time_seconds": wall_time_seconds,
        "captured_states_attempted": len(evaluation.states),
        "fem_trajectories_attempted": len(evaluation.trajectories),
        "total_states": total_states,
        "states": [_state_to_dict(result) for result in evaluation.states],
        "trajectories": [
            {
                "location_x_mm": result.trajectory.location_x_mm,
                "diameter_mm": result.trajectory.diameter_mm,
                "auc": result.auc,
            }
            for result in evaluation.trajectories
        ],
        "diagnostics": evaluation.diagnostics,
    }


def _run_design(
    name: str,
    parameters: FingertipParameters,
    evaluator: DesignEvaluator,
) -> dict[str, Any]:
    start = time.perf_counter()
    evaluation = evaluator.evaluate(parameters)
    record = {
        "name": name,
        "parameters": asdict(parameters),
        "evaluation": _evaluation_to_dict(
            evaluation,
            wall_time_seconds=time.perf_counter() - start,
            total_states=evaluator.scenario_grid.captured_state_count,
        ),
    }
    print(
        f"{name}: status={evaluation.status}, "
        f"minimum_auc={evaluation.minimum_auc}"
    )
    return record


def _configuration(
    evaluator: DesignEvaluator,
) -> dict[str, Any]:
    grid = evaluator.scenario_grid
    return {
        "scenario_grid": {
            "locations_x_mm": list(grid.locations_x_mm),
            "indenter_radii_mm": list(grid.indenter_radii_mm),
            "captured_depths_mm": list(grid.captured_depths_mm),
            "trajectory_count": grid.trajectory_count,
            "captured_state_count": grid.captured_state_count,
        },
        "mesh_settings": asdict(evaluator.mesh_settings),
        "trace_settings": asdict(evaluator.trace_settings),
        "fem_steps": evaluator.fem_steps,
        "internal_contact": evaluator.internal_contact,
        "basal_interface": evaluator.basal_interface,
        "protocol": "12 monotonic trajectories, 48 exact captured PLANAR_2D states",
    }


def main() -> int:
    parameters = FingertipParameters()
    evaluator = DesignEvaluator(
        ScenarioGrid(),
        mesh_settings=mesh_settings_for_level("medium"),
        trace_settings=Transport3DSettings(mode="planar"),
        led=LED(),
        optical=OpticalMaterial(),
        fem_steps=48,
        internal_contact="sides_separate",
        basal_interface="bonded",
    )
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": asdict(parameters),
        "configuration": _configuration(evaluator),
        "evaluation": _run_design("nominal", parameters, evaluator),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"result: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
