"""Run a bounded production LUMO Ax campaign.

The file deliberately keeps the user-owned scientific configuration visible at
the top.  ``--preflight`` performs environment and backend checks only.  A
campaign must be requested explicitly with ``--trials``; this prevents an
accidental long GPU run while still exercising the real production evaluator
when requested. ``--trials`` counts Ax-generated proposals; the nominal
baseline is evaluated separately.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from lumo import DEFAULT_MECHANICS_CONTRACT
from lumo.simulation import lumo_optical_settings
from lumo.finger import (
    Fingertip,
    FingertipParameters,
    LED,
    OpticalParameters,
    ViscoelasticParameters,
    fingertip_parameters_fingerprint,
)
from lumo.ray_tracing.optical_mechanics.settings import Transport3DSettings
from lumo.optimization.adapters.ax import (
    AxSettings,
    CampaignInfrastructureError,
    run_ax_optimization,
)
from lumo.optimization.evaluation_registry import EvaluationRegistry
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    ParameterSpec,
    PRODUCTION_LINEAR_CONSTRAINTS,
)
from lumo.optimization.evaluator import (
    TRAJECTORY_EVALUATION_SCHEMA,
    Lumo3DTrajectoryEvaluator,
)
from lumo.optimization.objectives import (
    TRAJECTORY_SEPARATION_OBJECTIVE,
    TrajectoryObjectiveConfig,
)
from lumo.optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)


# ------------------------------ USER CONFIG ------------------------------
DEVICE = "cuda:0"
SEED = 20260820
INITIALIZATION_TRIALS = 1
MAX_CONSECUTIVE_KNOWN_PROPOSALS = 20
DEFAULT_OUTPUT = Path("output/optimization/bo")

USER_VISCOELASTIC = ViscoelasticParameters(
    density_kg_m3=1.0e3,
    k_mu_pa=1.0e5,
    k_lambda_pa=1.0e5,
    k_damp=10.0,
)
USER_OPTICAL = OpticalParameters(
    refractive_index_air=1.0,
    refractive_index_silicone=1.41,
    absorption_per_mm=0.02,
)
USER_PARAMETERS = FingertipParameters(
    flat_pad_height=5.0,
    semielliptical_pad_height=9.0,
    stem_width=7.6,
    stem_height=6.0,
    void_width=1.0,
    void_height=0.25,
    viscoelastic=USER_VISCOELASTIC,
    optical=USER_OPTICAL,
)
USER_LED = LED(
    relative_radiant_power=1.0,
    emission_half_angle_deg=80.0,
)
USER_PROTOCOL = replace(
    DEFAULT_TRAJECTORY_PROTOCOL,
    contact_locations_u=(0.25, 0.50, 0.75),
    indenter_radii_mm=(4.0, 5.0),
    checkpoint_depths_mm=(0.5, 1.0, 1.5),
    initial_gap_mm=0.25,
)
SMOKE_PROTOCOL = TrajectoryEvaluationProtocol(
    contact_locations_u=(0.25, 0.75),
    indenter_radii_mm=(5.0,),
    checkpoint_depths_mm=(1.0,),
    initial_gap_mm=0.25,
)
USER_OPTICAL_SETTINGS: Transport3DSettings = replace(
    lumo_optical_settings(),
    ray_count=256,
)
USER_MECHANICS_CONTRACT = replace(
    DEFAULT_MECHANICS_CONTRACT,
    vbd_iterations=10,
    max_load_increment_mm=0.05,
    first_contact=replace(
        DEFAULT_MECHANICS_CONTRACT.first_contact,
        coarse_step_mm=0.25,
        tolerance_mm=1.0e-3,
        spawn_clearance_mm=0.05,
        max_travel_mm=20.0,
    ),
)
USER_OBJECTIVE = TrajectoryObjectiveConfig(
    radius_penalty_weight=1.0,
)
USER_SEARCH_BOUNDS = (
    ParameterSpec("flat_pad_height", 0.5, 29.5),
    ParameterSpec("semielliptical_pad_height", 0.5, 29.5),
    ParameterSpec("stem_width", 1.0, 20.0),
    ParameterSpec("stem_height", 1.0, 25.0),
    ParameterSpec("void_width", 0.0, 10.0),
    ParameterSpec("void_height", 0.0, 25.0),
)


def _search_bounds_payload(
    search_bounds: tuple[ParameterSpec, ...],
) -> list[dict[str, object]]:
    return [bound.to_dict() for bound in search_bounds]


def _design_space(
    nominal_parameters: FingertipParameters,
    search_bounds: tuple[ParameterSpec, ...],
) -> DesignSpace:
    return DesignSpace(
        nominal_parameters,
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in search_bounds
        ),
        linear_constraints=PRODUCTION_LINEAR_CONSTRAINTS,
    )


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_optix_smoke() -> Any:
    """Run the tooling-owned environment smoke at the preflight boundary."""
    from scripts.tools.optix_smoke import run

    return run()


def _run_newton_smoke() -> dict[str, Any]:
    """Advance a tiny neutral tetrahedral block through the real Newton backend."""
    from lumo.physics import NewtonSettings, TetMeshData
    from lumo.physics.newton.solve import solve

    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float32,
    )
    tetrahedra = np.asarray(
        (
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 3, 4, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
        ),
        dtype=np.int32,
    )
    result = solve(
        TetMeshData(vertices, tetrahedra),
        settings=NewtonSettings(
            device=DEVICE,
            gravity=0.0,
            steps=1,
            iterations=1,
            fixed_vertex_indices=(0, 1, 2, 3),
        ),
    )
    if result.deformed_vertices.shape != vertices.shape:
        raise RuntimeError("Newton smoke returned an unexpected vertex shape")
    if not np.all(np.isfinite(result.deformed_vertices)):
        raise RuntimeError("Newton smoke returned non-finite deformed vertices")
    return {
        "device": DEVICE,
        "vertex_count": int(result.deformed_vertices.shape[0]),
        "tetrahedron_count": int(result.tetrahedra.shape[0]),
        "finite_result": True,
    }


def _user_config_payload(
    *,
    trials: int | None = None,
    protocol: TrajectoryEvaluationProtocol = USER_PROTOCOL,
    campaign_mode: str = "production",
    objective_config: TrajectoryObjectiveConfig = USER_OBJECTIVE,
    search_bounds: tuple[ParameterSpec, ...] = USER_SEARCH_BOUNDS,
) -> dict[str, Any]:
    tip = Fingertip(
        USER_PARAMETERS,
        led=USER_LED,
    )
    payload = {
        "schema": "lumo-production-bo-user-config-v1",
        "device": DEVICE,
        "seed": SEED,
        "campaign_mode": campaign_mode,
        "nominal_parameters": asdict(USER_PARAMETERS),
        "search_bounds": _search_bounds_payload(search_bounds),
        "led": asdict(USER_LED),
        "optical_parameters": asdict(USER_PARAMETERS.optical),
        "trajectory_protocol": protocol.to_dict(),
        "mechanics_contract": USER_MECHANICS_CONTRACT.to_dict(),
        "transport_3d_settings": asdict(USER_OPTICAL_SETTINGS),
        "objective": {
            "name": TRAJECTORY_SEPARATION_OBJECTIVE.serialized_name,
            "direction": "maximize",
            "config": asdict(objective_config),
        },
        "ax": {
            "initialization_trials": INITIALIZATION_TRIALS,
            "search_trials": max(
                0,
                (trials or INITIALIZATION_TRIALS) - INITIALIZATION_TRIALS,
            ),
            "max_proposals": trials,
            "max_consecutive_known_proposals": MAX_CONSECUTIVE_KNOWN_PROPOSALS,
            "trials_semantics": (
                "number of Ax-generated proposals; the nominal baseline is "
                "evaluated separately"
            ),
            "linear_constraints": [
                constraint.expression
                for constraint in PRODUCTION_LINEAR_CONSTRAINTS
            ],
        },
        "tip_validation": {
            "led_source_mm": list(tip.led_source),
            "morphology_fingerprint": fingertip_parameters_fingerprint(tip.parameters),
        },
    }
    return payload


def _preflight_payload(
    output: str | Path | None = None,
    *,
    protocol: TrajectoryEvaluationProtocol = USER_PROTOCOL,
    objective_config: TrajectoryObjectiveConfig = USER_OBJECTIVE,
    search_bounds: tuple[ParameterSpec, ...] = USER_SEARCH_BOUNDS,
) -> dict[str, Any]:
    """Check dependencies, domain construction, and tiny real backend launches."""
    preflight_root = DEFAULT_OUTPUT if output is None else Path(output)
    checks: dict[str, Any] = {}
    for module_name in (
        "gmsh",
        "warp",
        "newton",
        "cupy",
        "optix",
        "cuda.bindings.nvrtc",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            checks[module_name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            checks[module_name] = {"status": "PASS"}

    try:
        Fingertip(USER_PARAMETERS, led=USER_LED)
        design_space = _design_space(USER_PARAMETERS, search_bounds)
        evaluator = Lumo3DTrajectoryEvaluator(
            preflight_root / "preflight-artifacts",
            protocol=protocol,
            objective_config=objective_config,
            mechanics_contract=USER_MECHANICS_CONTRACT,
            device=DEVICE,
            optical_settings=USER_OPTICAL_SETTINGS,
            led=USER_LED,
            fixed_parameters=USER_PARAMETERS,
        )
        checks["production_configuration"] = {
            "status": "PASS",
            "active_variables": [
                variable.name.value for variable in design_space.active_variables
            ],
            "protocol_fingerprint": protocol.fingerprint,
            "mechanics_contract_fingerprint": USER_MECHANICS_CONTRACT.fingerprint,
            "search_bounds": _search_bounds_payload(search_bounds),
        }
    except Exception as exc:
        checks["production_configuration"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        mechanics_smoke = _run_newton_smoke()
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["newton_smoke"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        checks["newton_smoke"] = {"status": "PASS", "evidence": mechanics_smoke}

    try:
        smoke = _run_optix_smoke()
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["optix_smoke"] = {
            "status": "FAIL",
            "stage": getattr(exc, "stage", "optix_runtime_initialization"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        checks["optix_smoke"] = {"status": "PASS", "evidence": smoke.to_dict()}

    failed = [name for name, value in checks.items() if value["status"] != "PASS"]
    return {
        "schema": "lumo-production-bo-preflight-v1",
        "status": "PASS" if not failed else "FAIL_EXTERNAL_PREREQUISITE",
        "failed_checks": failed,
        "checks": checks,
    }


def _record_payload(record: Any) -> dict[str, Any]:
    evaluation = record.evaluation
    return {
        "trial_index": record.trial_index,
        "phase": record.phase,
        "parameters": dict(record.parameters),
        "status": record.status,
        "objective": (
            None
            if evaluation is None
            else getattr(evaluation, "objective_value", None)
        ),
        "failure_message": record.failure_message,
        "result_artifact_path": (
            None
            if evaluation is None
            else getattr(evaluation, "result_artifact_path", None)
        ),
        "wall_time_seconds": record.wall_time_seconds,
        "registry_key": record.registry_key,
    }


def run_campaign(
    output: str | Path,
    *,
    trials: int,
    smoke: bool = False,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("trials must be a positive integer")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)

    protocol = SMOKE_PROTOCOL if smoke else USER_PROTOCOL
    preflight = _preflight_payload(
        root,
        protocol=protocol,
        objective_config=USER_OBJECTIVE,
        search_bounds=USER_SEARCH_BOUNDS,
    )
    _json_write(root / "preflight.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError(
            "external preflight failed; no candidate was submitted: "
            + ", ".join(preflight["failed_checks"])
        )

    design_space = _design_space(USER_PARAMETERS, USER_SEARCH_BOUNDS)
    evaluator = Lumo3DTrajectoryEvaluator(
        root / "artifacts",
        protocol=protocol,
        objective_config=USER_OBJECTIVE,
        mechanics_contract=USER_MECHANICS_CONTRACT,
        device=DEVICE,
        optical_settings=USER_OPTICAL_SETTINGS,
        led=USER_LED,
        fixed_parameters=USER_PARAMETERS,
    )
    selected_registry_path = (
        root / "registry.json" if registry_path is None else Path(registry_path)
    )
    config = _user_config_payload(
        trials=trials,
        protocol=protocol,
        campaign_mode="smoke" if smoke else "production",
        objective_config=USER_OBJECTIVE,
        search_bounds=USER_SEARCH_BOUNDS,
    )
    config.update(
        {
            "contract_id": evaluator.evaluation_contract_id,
            "evaluation_schema": TRAJECTORY_EVALUATION_SCHEMA,
            "output": str(root),
            "registry": str(selected_registry_path),
        }
    )
    _json_write(root / "config.json", config)
    registry = EvaluationRegistry(selected_registry_path)
    records: list[dict[str, Any]] = []

    def persist(_client: Any, observed: tuple[Any, ...]) -> None:
        records[:] = [_record_payload(record) for record in observed]
        _json_write(root / "trials.json", records)

    settings = AxSettings(
        initialization_trials=INITIALIZATION_TRIALS,
        search_trials=max(0, trials - INITIALIZATION_TRIALS),
        seed=SEED,
        objective=TRAJECTORY_SEPARATION_OBJECTIVE,
        max_consecutive_known_proposals=MAX_CONSECUTIVE_KNOWN_PROPOSALS,
    )
    try:
        result = run_ax_optimization(
            design_space,
            evaluator,
            settings,
            on_record=persist,
            evaluation_registry=registry,
            evaluation_contract_id=evaluator.evaluation_contract_id,
            campaign_id=root.name,
            max_proposals=trials,
        )
    except CampaignInfrastructureError as exc:
        summary = {
            "status": "INFRASTRUCTURE_FAILED",
            "signature": exc.signature,
            "error": f"{type(exc).__name__}: {exc}",
            "records": records,
        }
        _json_write(root / "summary.json", summary)
        raise

    summary = {
        "status": result.status,
        "objective_name": result.objective_name,
        "ax_proposal_count": result.ax_proposal_count,
        "new_evaluation_count": result.new_evaluation_count,
        "unique_success_count": result.unique_success_count,
        "unique_failure_count": result.unique_failure_count,
        "best_trial": (
            None
            if result.best_record is None
            else _record_payload(result.best_record)
        ),
        "records": records,
    }
    _json_write(root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--trials",
        type=int,
        default=0,
        help=(
            "number of Ax-generated proposals; the nominal baseline is "
            "evaluated separately"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the explicit reduced two-state protocol",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="reuse an existing exact-result registry outside the output directory",
    )
    args = parser.parse_args(argv)

    if args.preflight:
        preflight = _preflight_payload(
            args.output,
            protocol=SMOKE_PROTOCOL if args.smoke else USER_PROTOCOL,
        )
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        if args.trials == 0:
            return 0 if preflight["status"] == "PASS" else 2

    if args.trials < 1:
        parser.error("provide --trials N for a campaign, or use --preflight")
    try:
        summary = run_campaign(
            args.output,
            trials=args.trials,
            smoke=args.smoke,
            registry_path=args.registry,
        )
    except CampaignInfrastructureError as exc:
        print(f"INFRASTRUCTURE_FAILED [{exc.signature}]: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"CAMPAIGN_ABORTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
