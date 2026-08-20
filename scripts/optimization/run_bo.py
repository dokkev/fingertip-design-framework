"""Run a bounded production LUMO Ax campaign.

The file deliberately keeps the user-owned scientific configuration visible at
the top.  ``--preflight`` performs environment checks only.  A campaign must
be requested explicitly with ``--trials``; this prevents an accidental long
GPU run while still exercising the real production evaluator when requested.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from typing import Any

# Support the documented ``python scripts/optimization/run_bo.py`` invocation
# as well as ``python -m scripts.optimization.run_bo``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lumo import DEFAULT_MECHANICS_CONTRACT
from lumo.simulation import lumo_optical_settings
from model import (
    Fingertip,
    FingertipParameters,
    LED,
    OpticalMaterial,
    fingertip_parameters_fingerprint,
)
from optics.transport3d.settings import Transport3DSettings
from optimization.adapters.ax import (
    AxSettings,
    CampaignInfrastructureError,
    PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS,
    run_ax_optimization,
)
from optimization.evaluation_registry import EvaluationRegistry
from optimization.evaluator import (
    TRAJECTORY_EVALUATION_SCHEMA,
    create_lumo3d_trajectory_study,
)
from optimization.objectives import OBJECTIVE_NAME
from optimization.protocol import TrajectoryEvaluationProtocol
from validation.optics.optix_smoke import OptixSmokeError, run as run_optix_smoke


# ------------------------------ USER CONFIG ------------------------------
DEVICE = "cuda:0"
SEED = 20260820
DEFAULT_OUTPUT = Path("output/optimization/bo")

USER_PARAMETERS = FingertipParameters(
    flat_pad_height=5.0,
    semielliptical_pad_height=9.0,
    stem_width=7.6,
    stem_height=6.0,
    void_width=1.0,
    void_height=0.25,
)
USER_LED = LED(
    relative_radiant_power=1.0,
    emission_half_angle_deg=80.0,
)
USER_OPTICAL_MATERIAL = OpticalMaterial(
    refractive_index_air=1.0,
    refractive_index_silicone=1.41,
    absorption_per_mm=0.02,
)
USER_PROTOCOL = TrajectoryEvaluationProtocol(
    contact_locations_u=(0.25, 0.75),
    indenter_radii_mm=(5.0,),
    checkpoint_depths_mm=(1.0,),
    initial_gap_mm=0.25,
)
USER_OPTICAL_SETTINGS: Transport3DSettings = lumo_optical_settings()
USER_MECHANICS_CONTRACT = DEFAULT_MECHANICS_CONTRACT


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _user_config_payload(*, trials: int | None = None) -> dict[str, Any]:
    tip = Fingertip(
        USER_PARAMETERS,
        led=USER_LED,
        optical=USER_OPTICAL_MATERIAL,
    )
    payload = {
        "schema": "lumo-production-bo-user-config-v1",
        "device": DEVICE,
        "seed": SEED,
        "nominal_parameters": asdict(USER_PARAMETERS),
        "led": asdict(USER_LED),
        "optical_material": asdict(USER_OPTICAL_MATERIAL),
        "trajectory_protocol": USER_PROTOCOL.to_dict(),
        "mechanics_contract": USER_MECHANICS_CONTRACT.to_dict(),
        "transport_3d_settings": asdict(USER_OPTICAL_SETTINGS),
        "objective": {
            "name": OBJECTIVE_NAME,
            "direction": "maximize",
        },
        "ax": {
            "initialization_trials": 1,
            "search_trials": max(0, (trials or 1) - 1),
            "max_proposals": trials,
            "linear_constraints": list(PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS),
        },
        "tip_validation": {
            "led_source_mm": list(tip.led_source),
            "morphology_fingerprint": fingertip_parameters_fingerprint(tip.parameters),
        },
    }
    return payload


def _preflight_payload() -> dict[str, Any]:
    """Check Python dependencies, domain construction, and one real OptiX launch."""
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
        Fingertip(USER_PARAMETERS, led=USER_LED, optical=USER_OPTICAL_MATERIAL)
        study = create_lumo3d_trajectory_study(
            DEFAULT_OUTPUT / "preflight-artifacts",
            protocol=USER_PROTOCOL,
            mechanics_contract=USER_MECHANICS_CONTRACT,
            device=DEVICE,
            optical_settings=USER_OPTICAL_SETTINGS,
            led=USER_LED,
            optical_material=USER_OPTICAL_MATERIAL,
            nominal_parameters=USER_PARAMETERS,
        )
        checks["production_configuration"] = {
            "status": "PASS",
            "active_variables": [
                variable.name.value for variable in study.design_space.active_variables
            ],
            "protocol_fingerprint": USER_PROTOCOL.fingerprint,
            "mechanics_contract_fingerprint": USER_MECHANICS_CONTRACT.fingerprint,
        }
    except Exception as exc:
        checks["production_configuration"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        smoke = run_optix_smoke()
    except OptixSmokeError as exc:  # pragma: no cover - environment dependent
        checks["optix_smoke"] = {
            "status": "FAIL",
            "stage": exc.stage,
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["optix_smoke"] = {
            "status": "FAIL",
            "stage": "optix_runtime_initialization",
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
        "wall_time_seconds": record.wall_time_seconds,
        "registry_key": record.registry_key,
    }


def run_campaign(output: str | Path, *, trials: int) -> dict[str, Any]:
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("trials must be a positive integer")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)

    preflight = _preflight_payload()
    _json_write(root / "preflight.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError(
            "external preflight failed; no candidate was submitted: "
            + ", ".join(preflight["failed_checks"])
        )

    study = create_lumo3d_trajectory_study(
        root / "artifacts",
        protocol=USER_PROTOCOL,
        mechanics_contract=USER_MECHANICS_CONTRACT,
        device=DEVICE,
        optical_settings=USER_OPTICAL_SETTINGS,
        led=USER_LED,
        optical_material=USER_OPTICAL_MATERIAL,
        nominal_parameters=USER_PARAMETERS,
    )
    config = _user_config_payload(trials=trials)
    config.update(
        {
            "contract_id": study.evaluation_contract_id,
            "evaluation_schema": TRAJECTORY_EVALUATION_SCHEMA,
            "output": str(root),
        }
    )
    _json_write(root / "config.json", config)
    registry = EvaluationRegistry(root / "registry.json")
    records: list[dict[str, Any]] = []

    def persist(_client: Any, observed: tuple[Any, ...]) -> None:
        records[:] = [_record_payload(record) for record in observed]
        _json_write(root / "trials.json", records)

    settings = AxSettings(
        initialization_trials=1,
        search_trials=max(0, trials - 1),
        seed=SEED,
        objective_name=OBJECTIVE_NAME,
    )
    try:
        result = run_ax_optimization(
            study,
            settings,
            on_record=persist,
            evaluation_registry=registry,
            evaluation_contract_id=study.evaluation_contract_id,
            campaign_id=root.name,
            result_artifact_path=str((root / "trials.json").resolve()),
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
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.preflight:
        preflight = _preflight_payload()
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        if args.trials == 0:
            return 0 if preflight["status"] == "PASS" else 2

    if args.trials < 1:
        parser.error("provide --trials N for a campaign, or use --preflight")
    try:
        summary = run_campaign(args.output, trials=args.trials)
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
