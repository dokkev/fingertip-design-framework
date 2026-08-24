"""Run a bounded production LUMO Ax campaign.

The file keeps the user-owned scientific definition visible at the top while
loading expert mesh, Newton, and transport settings from one typed YAML
boundary. Production campaigns require explicit success targets and independent
evaluation/proposal caps; the nominal baseline is evaluated separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from lumo.config import LumoExecutionConfig, load_lumo_execution_config
from lumo.finger import (
    Fingertip,
    FingertipParameters,
    LED,
    OpticalParameters,
    ViscoelasticParameters,
    fingertip_parameters_fingerprint,
)
from lumo.optimization.adapters.ax import (
    AxCheckpointEvent,
    AxPendingTrial,
    AxResumeState,
    AxSettings,
    AxTerminationReason,
    CampaignInfrastructureError,
    ax_trial_record_from_payload,
    run_ax_optimization,
)
from lumo.optimization.checkpoint import (
    CampaignCheckpointStore,
    CheckpointError,
)
from lumo.optimization.evaluation_registry import (
    EvaluationRegistry,
    evaluation_registry_lock_path,
    evaluation_registry_writer_lock,
)
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
from lumo.optimization.optical_contract import (
    summarize_optical_failure_diagnostics,
)
from lumo.optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)
from lumo.optimization.runtime_identity import runtime_identity_for_device


# ------------------------------ USER CONFIG ------------------------------
SEED = 20260820
# The one-proposal integration smoke uses a stable interior Sobol fixture;
# production keeps its independent campaign seed and failure slack.
SMOKE_SEED = 20260842
SMOKE_INITIALIZATION_SUCCESS_TARGET = 1
PRODUCTION_INITIALIZATION_SUCCESS_TARGET = 6
MAX_CONSECUTIVE_KNOWN_PROPOSALS = 20
MAX_FEASIBILITY_RESAMPLES = 100
DEFAULT_OUTPUT = Path("output/optimization/bo")
DEFAULT_EXECUTION_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "lumo_execution.yaml"
)

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
USER_OBJECTIVE = TrajectoryObjectiveConfig(
    radius_penalty_weight=1.0,
)


@dataclass(frozen=True)
class CampaignBudget:
    """Independent scientific success targets and hard resource caps."""

    initialization_success_target: int
    search_success_target: int
    maximum_evaluations: int
    maximum_proposals: int

    def __post_init__(self) -> None:
        for name in (
            "initialization_success_target",
            "search_success_target",
            "maximum_evaluations",
            "maximum_proposals",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.initialization_success_target < 1:
            raise ValueError("initialization_success_target must be at least 1")
        if self.search_success_target < 0:
            raise ValueError("search_success_target must be nonnegative")
        minimum_evaluations = (
            1 + self.initialization_success_target + self.search_success_target
        )
        if self.maximum_evaluations < minimum_evaluations:
            raise ValueError(
                "maximum_evaluations must cover nominal plus all success targets "
                f"(at least {minimum_evaluations})"
            )
        minimum_proposals = (
            self.initialization_success_target + self.search_success_target
        )
        if self.maximum_proposals < minimum_proposals:
            raise ValueError(
                "maximum_proposals must cover all generated success targets "
                f"(at least {minimum_proposals})"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
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
    *,
    fixed_led: LED = USER_LED,
) -> DesignSpace:
    return DesignSpace(
        nominal_parameters,
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in search_bounds
        ),
        linear_constraints=PRODUCTION_LINEAR_CONSTRAINTS,
        fixed_led=fixed_led,
    )


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_json_payload(payload: Any) -> Any:
    """Compare persisted contracts after the JSON tuple/list normalization."""
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def _source_provenance(
    repository_root: Path | None = None,
    *,
    excluded_paths: tuple[str | Path, ...] = (),
) -> dict[str, object]:
    """Record source bytes while excluding declared mutable run artifacts."""
    repository_root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            cwd=repository_root,
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.STDOUT,
            cwd=repository_root,
        )
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            stderr=subprocess.STDOUT,
            cwd=repository_root,
        )
        untracked_output = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.STDOUT,
            cwd=repository_root,
        )
        excluded = tuple(
            (
                Path(path).resolve()
                if Path(path).is_absolute()
                else (repository_root / path).resolve()
            )
            for path in excluded_paths
        )
        untracked_paths = tuple(
            sorted(
                relative
                for part in untracked_output.split(b"\0")
                if part
                for relative in (
                    part.decode("utf-8", errors="surrogateescape"),
                )
                if not any(
                    (repository_root / relative).resolve() == target
                    or target in (repository_root / relative).resolve().parents
                    for target in excluded
                )
            )
        )
        untracked_hasher = hashlib.sha256()
        for relative in untracked_paths:
            encoded_path = relative.encode("utf-8", errors="surrogateescape")
            source_path = repository_root / relative
            content = source_path.read_bytes()
            untracked_hasher.update(len(encoded_path).to_bytes(8, "big"))
            untracked_hasher.update(encoded_path)
            untracked_hasher.update(len(content).to_bytes(8, "big"))
            untracked_hasher.update(content)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "git_commit": None,
            "git_dirty": None,
            "tracked_diff_sha256": None,
            "untracked_content_sha256": None,
            "source_id": None,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
    tracked_dirty = bool(tracked_status.strip())
    untracked_dirty = bool(untracked_paths)
    tracked_diff_sha256 = (
        hashlib.sha256(tracked_diff).hexdigest() if tracked_dirty else None
    )
    untracked_content_sha256 = (
        untracked_hasher.hexdigest() if untracked_dirty else None
    )
    identity = {
        "git_commit": commit,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_content_sha256": untracked_content_sha256,
    }
    return {
        "git_commit": commit,
        "git_dirty": tracked_dirty or untracked_dirty,
        "tracked_dirty": tracked_dirty,
        "untracked_dirty": untracked_dirty,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_content_sha256": untracked_content_sha256,
        "untracked_paths": list(untracked_paths),
        "source_id": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "status": "available",
    }


def _enforce_source_policy(
    source: dict[str, object],
    *,
    smoke: bool,
    allow_dirty: bool,
) -> None:
    """Reject unauditable or dirty production sources before expensive work."""

    if source.get("status") != "available" or not source.get("source_id"):
        raise RuntimeError("Git source provenance is unavailable")
    if not smoke and source.get("git_dirty") and not allow_dirty:
        raise RuntimeError(
            "production campaign requires a clean Git worktree; use "
            "--allow-dirty to persist and explicitly accept this source snapshot"
        )


def _reject_git_tracked_mutable_path(
    path: str | Path,
    repository_root: Path | None = None,
) -> None:
    """Reject mutable run state that would invalidate its own source identity."""

    repository_root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError:
        return
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        raise ValueError(
            "mutable campaign state must not be a Git-tracked file because its "
            "updates would change source provenance and break exact resume: "
            f"{resolved}"
        )
    if result.returncode != 1:
        raise RuntimeError(
            "cannot determine whether evaluation registry is Git-tracked: "
            f"{result.stderr.strip()}"
        )


def _ax_package_version() -> str:
    import ax

    version = getattr(ax, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("Ax package does not expose a version")
    return version


def _campaign_resume_contract(
    *,
    root: Path,
    design_space: DesignSpace,
    evaluator: Lumo3DTrajectoryEvaluator,
    protocol: TrajectoryEvaluationProtocol,
    objective_config: TrajectoryObjectiveConfig,
    registry_path: Path,
    execution: LumoExecutionConfig,
    budget: CampaignBudget,
    max_feasibility_resamples: int,
    source: dict[str, object],
    allow_cross_revision_cache: bool,
    allow_dirty: bool,
    seed: int,
) -> dict[str, object]:
    """Build the exact fixed-input contract required before resuming."""
    return {
        "campaign_id": root.name,
        "evaluation_contract_id": evaluator.evaluation_contract_id,
        "objective_identifier": {
            "name": evaluator.objective_identifier.name,
            "version": evaluator.objective_identifier.version,
        },
        "design_space": design_space.to_dict(),
        "parameterization_version": design_space.parameterization_version,
        "protocol": protocol.to_dict(),
        "execution_config": execution.to_dict(),
        "mechanics_contract": execution.mechanics.to_dict(),
        "transport_3d_settings": asdict(execution.transport),
        "volume_mesh_settings": asdict(execution.volume_mesh),
        "device": execution.device,
        "runtime_identity": evaluator.runtime_identity,
        "fixed_parameters": asdict(USER_PARAMETERS),
        "led": asdict(USER_LED),
        "objective_config": asdict(objective_config),
        "registry_path": str(registry_path.resolve()),
        "seed": seed,
        "budget": budget.to_dict()
        | {"max_feasibility_resamples": max_feasibility_resamples},
        "ax_package_version": _ax_package_version(),
        "source": source,
        "registry_cache_policy": {
            "allow_cross_revision_cache": allow_cross_revision_cache,
        },
        "source_policy": {
            "allow_dirty": allow_dirty,
            "dirty_accepted": bool(source.get("git_dirty") and allow_dirty),
        },
    }


def _resume_root(path: Path) -> Path:
    """Accept a campaign root or its current atomic checkpoint pointer."""
    candidate = path.expanduser()
    if candidate.is_file():
        if candidate.name != "checkpoint.json":
            raise CheckpointError(
                "--resume file must be the campaign checkpoint.json pointer"
            )
        return candidate.parent
    if candidate.is_dir() and (candidate / "state.json").is_file():
        raise CheckpointError(
            "--resume does not accept an immutable checkpoint directory; use "
            "the campaign root or its current checkpoint.json pointer"
        )
    return candidate


def _resume_state(
    store: CampaignCheckpointStore,
    *,
    expected_contract: dict[str, object],
) -> AxResumeState:
    checkpoint = store.load_latest()
    state = checkpoint.state
    if _canonical_json_payload(state.get("resume_contract")) != _canonical_json_payload(
        expected_contract
    ):
        raise CheckpointError(
            "resume contract mismatch; evaluator, fixed inputs, budget, source, "
            "or Ax serialization contract changed"
        )
    pending_index = state.get("pending_trial_index")
    pending: AxPendingTrial | None = None
    if pending_index is not None:
        pending = AxPendingTrial(
            trial_index=int(pending_index),
            phase=state["pending_phase"],
            latent_parameters=state["pending_latent_parameters"],
            physical_parameters=state["pending_physical_parameters"],
            registry_key=state.get("registry_key"),
        )
    counts = state.get("counts")
    if not isinstance(counts, dict):
        raise CheckpointError("checkpoint counts must be an object")
    return AxResumeState(
        client=store.load_ax_client(checkpoint),
        records=tuple(ax_trial_record_from_payload(item) for item in checkpoint.trials),
        pending_trial=pending,
        historical_success_count=int(counts.get("historical_success_count", 0)),
        historical_failure_count=int(counts.get("historical_failure_count", 0)),
    )


def _run_optix_smoke(device: str) -> Any:
    """Run the tooling-owned environment smoke at the preflight boundary."""
    from scripts.tools.optix_smoke import run

    if not device.startswith("cuda:"):
        raise ValueError("OptiX preflight device must use cuda:<index>")
    return run(int(device.removeprefix("cuda:")))


def _run_newton_smoke(device: str) -> dict[str, Any]:
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
            device=device,
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
        "device": device,
        "vertex_count": int(result.deformed_vertices.shape[0]),
        "tetrahedron_count": int(result.tetrahedra.shape[0]),
        "finite_result": True,
    }


def _user_config_payload(
    *,
    execution: LumoExecutionConfig,
    budget: CampaignBudget,
    protocol: TrajectoryEvaluationProtocol = USER_PROTOCOL,
    campaign_mode: str = "production",
    objective_config: TrajectoryObjectiveConfig = USER_OBJECTIVE,
    search_bounds: tuple[ParameterSpec, ...] = USER_SEARCH_BOUNDS,
    seed: int = SEED,
) -> dict[str, Any]:
    tip = Fingertip(
        USER_PARAMETERS,
        led=USER_LED,
    )
    payload = {
        "schema": "lumo-production-bo-user-config-v2",
        "device": execution.device,
        "seed": seed,
        "campaign_mode": campaign_mode,
        "nominal_parameters": asdict(USER_PARAMETERS),
        "search_bounds": _search_bounds_payload(search_bounds),
        "parameterization_version": (
            _design_space(USER_PARAMETERS, search_bounds).parameterization_version
        ),
        "led": asdict(USER_LED),
        "optical_parameters": asdict(USER_PARAMETERS.optical),
        "trajectory_protocol": protocol.to_dict(),
        "execution_config": execution.to_dict(),
        "volume_mesh_settings": asdict(execution.volume_mesh),
        "mechanics_contract": execution.mechanics.to_dict(),
        "transport_3d_settings": asdict(execution.transport),
        "objective": {
            "name": TRAJECTORY_SEPARATION_OBJECTIVE.serialized_name,
            "direction": "maximize",
            "config": asdict(objective_config),
        },
        "ax": {
            **budget.to_dict(),
            "max_consecutive_known_proposals": MAX_CONSECUTIVE_KNOWN_PROPOSALS,
            "max_feasibility_resamples": MAX_FEASIBILITY_RESAMPLES,
            "budget_semantics": (
                "success targets are independent from actual-evaluation and "
                "all-generated-proposal hard caps; nominal counts only toward "
                "maximum_evaluations"
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
    execution: LumoExecutionConfig,
    source: Mapping[str, object] | None = None,
    protocol: TrajectoryEvaluationProtocol = USER_PROTOCOL,
    objective_config: TrajectoryObjectiveConfig = USER_OBJECTIVE,
    search_bounds: tuple[ParameterSpec, ...] = USER_SEARCH_BOUNDS,
) -> dict[str, Any]:
    """Check dependencies, domain construction, and tiny real backend launches."""
    preflight_root = DEFAULT_OUTPUT if output is None else Path(output)
    resolved_source = (
        _source_provenance(excluded_paths=(preflight_root,))
        if source is None
        else dict(source)
    )
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
        runtime_identity = runtime_identity_for_device(execution.device)
        if runtime_identity.get("status") != "available":
            raise RuntimeError(
                "GPU/runtime identity is unavailable: "
                f"{runtime_identity.get('error', 'unknown error')}"
            )
        checks["runtime_identity"] = {
            "status": "PASS",
            "evidence": runtime_identity,
        }
        Fingertip(USER_PARAMETERS, led=USER_LED)
        design_space = _design_space(USER_PARAMETERS, search_bounds)
        evaluator = Lumo3DTrajectoryEvaluator(
            preflight_root / "preflight-artifacts",
            protocol=protocol,
            objective_config=objective_config,
            mechanics_contract=execution.mechanics,
            device=execution.device,
            optical_settings=execution.transport,
            led=USER_LED,
            fixed_parameters=USER_PARAMETERS,
            volume_mesh_settings=execution.volume_mesh,
            runtime_identity=runtime_identity,
        )
        checks["production_configuration"] = {
            "status": "PASS",
            "active_variables": [
                variable.name.value for variable in design_space.active_variables
            ],
            "protocol_fingerprint": protocol.fingerprint,
            "mechanics_contract_fingerprint": execution.mechanics.fingerprint,
            "execution_config": execution.to_dict(),
            "search_bounds": _search_bounds_payload(search_bounds),
        }
    except Exception as exc:
        checks.setdefault(
            "runtime_identity",
            {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
        )
        checks["production_configuration"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        mechanics_smoke = _run_newton_smoke(execution.device)
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["newton_smoke"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        checks["newton_smoke"] = {"status": "PASS", "evidence": mechanics_smoke}

    try:
        smoke = _run_optix_smoke(execution.device)
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
        "source": resolved_source,
        "checks": checks,
    }


def _record_payload(record: Any, design_space: DesignSpace) -> dict[str, Any]:
    evaluation = record.evaluation
    physical_parameters: dict[str, float] | None = None
    if not record.feasibility_rejection:
        physical_parameters = design_space.physical_values(
            design_space.decode(record.parameters)
        )
    report = (
        getattr(evaluation, "report", {}) if evaluation is not None else {}
    )
    failure_diagnostics: dict[str, Any] = {}
    if isinstance(report, Mapping):
        candidate = report.get("failure_diagnostics", report)
        if isinstance(candidate, Mapping):
            failure_diagnostics = dict(candidate)
    return {
        "trial_index": record.trial_index,
        "phase": record.phase,
        "parameters": dict(record.parameters),
        "latent_parameters": dict(record.parameters),
        "physical_parameters": physical_parameters,
        "status": record.status,
        "feasibility_rejection": record.feasibility_rejection,
        "feasibility_constraint": record.feasibility_constraint,
        "objective": (
            None
            if evaluation is None
            else getattr(evaluation, "objective_value", None)
        ),
        "failure_message": record.failure_message,
        "failure_scenario": (
            None
            if evaluation is None
            else getattr(evaluation, "failure_scenario", None)
        ),
        "failure_diagnostics": failure_diagnostics,
        "result_artifact_path": (
            None
            if evaluation is None
            else getattr(evaluation, "result_artifact_path", None)
        ),
        "wall_time_seconds": record.wall_time_seconds,
        "registry_key": record.registry_key,
        "duplicate_of_trial_index": record.duplicate_of_trial_index,
        "duplicate_of_campaign_id": record.duplicate_of_campaign_id,
        "duplicate_of_artifact_path": record.duplicate_of_artifact_path,
        "reused_evaluation": record.reused_evaluation,
        "reused_evaluation_status": record.reused_evaluation_status,
    }


def _campaign_acceptance(
    result: Any,
    *,
    budget: CampaignBudget,
) -> tuple[str, list[str]]:
    """Apply the production/smoke acceptance contract to one Ax result."""
    failures: list[str] = []
    if not result.nominal_successful:
        failures.append("nominal_failed")
    if result.successful_initialization_count < budget.initialization_success_target:
        failures.append("initialization_success_target_not_reached")
    if result.successful_search_count < budget.search_success_target:
        failures.append("search_success_target_not_reached")
    if result.termination_reason != AxTerminationReason.REQUESTED_BUDGET_REACHED:
        failures.append(f"termination:{result.termination_reason.value}")
    return ("PASS" if not failures else "FAILED"), failures


def _actual_evaluation_count(records: Any) -> int:
    """Count evaluator invocations without registry replays or rejected proposals."""

    return sum(
        record.status not in ("duplicate_skipped", "feasibility_rejected")
        and not record.reused_evaluation
        for record in records
    )


def run_campaign(
    output: str | Path,
    *,
    budget: CampaignBudget,
    smoke: bool = False,
    registry_path: str | Path | None = None,
    resume: str | Path | None = None,
    execution_config: str | Path | LumoExecutionConfig = DEFAULT_EXECUTION_CONFIG,
    allow_dirty: bool = False,
    allow_cross_revision_cache: bool = False,
) -> dict[str, Any]:
    """Run a new campaign or explicitly resume one existing campaign."""
    if not isinstance(budget, CampaignBudget):
        raise TypeError("budget must be a CampaignBudget")
    if (
        not smoke
        and budget.initialization_success_target
        < PRODUCTION_INITIALIZATION_SUCCESS_TARGET
    ):
        raise ValueError(
            "production campaign requires at least "
            f"{PRODUCTION_INITIALIZATION_SUCCESS_TARGET} successful Sobol "
            "observations"
        )
    if not smoke and budget.search_success_target < 1:
        raise ValueError(
            "production campaign requires at least one successful MBM/search "
            "observation"
        )
    if not smoke and registry_path is not None:
        raise ValueError(
            "production external registry reuse is disabled until same-contract "
            "evaluator reproducibility is established"
        )
    output_path = Path(output).expanduser()
    if resume is None:
        root = output_path
    else:
        root = _resume_root(Path(resume))
        if output_path != DEFAULT_OUTPUT and output_path.resolve() != root.resolve():
            raise ValueError("--output and --resume must identify the same campaign")
    selected_registry_path = (
        root / "registry.json" if registry_path is None else Path(registry_path)
    )
    _reject_git_tracked_mutable_path(selected_registry_path)
    store = CampaignCheckpointStore(root)
    _reject_git_tracked_mutable_path(
        evaluation_registry_lock_path(selected_registry_path)
    )
    _reject_git_tracked_mutable_path(store.lock_path)
    with store.writer_lock(), evaluation_registry_writer_lock(
        selected_registry_path
    ):
        return _run_campaign_locked(
            root,
            budget=budget,
            smoke=smoke,
            registry_path=selected_registry_path,
            resume=resume is not None,
            checkpoint_store=store,
            execution_config=execution_config,
            allow_dirty=allow_dirty,
            allow_cross_revision_cache=allow_cross_revision_cache,
        )


def _run_campaign_locked(
    output: str | Path,
    *,
    budget: CampaignBudget,
    smoke: bool = False,
    registry_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_store: CampaignCheckpointStore,
    execution_config: str | Path | LumoExecutionConfig = DEFAULT_EXECUTION_CONFIG,
    allow_dirty: bool = False,
    allow_cross_revision_cache: bool = False,
) -> dict[str, Any]:
    if not isinstance(budget, CampaignBudget):
        raise TypeError("budget must be a CampaignBudget")
    root = Path(output)
    selected_registry_path = (
        root / "registry.json" if registry_path is None else Path(registry_path)
    )
    registry_lock = evaluation_registry_lock_path(selected_registry_path)
    existing_entries = (
        tuple(
            path
            for path in root.iterdir()
            if path.name != ".checkpoint.lock"
            and path.resolve() != registry_lock.resolve()
        )
        if root.exists()
        else ()
    )
    if not resume and existing_entries:
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    if resume and not root.is_dir():
        raise CheckpointError(f"resume campaign directory does not exist: {root}")
    execution = (
        execution_config
        if isinstance(execution_config, LumoExecutionConfig)
        else load_lumo_execution_config(execution_config)
    )
    source = _source_provenance(
        excluded_paths=(root, selected_registry_path, registry_lock),
    )
    _enforce_source_policy(source, smoke=smoke, allow_dirty=allow_dirty)
    root.mkdir(parents=True, exist_ok=True)

    protocol = SMOKE_PROTOCOL if smoke else USER_PROTOCOL
    campaign_seed = SMOKE_SEED if smoke else SEED
    design_space = _design_space(USER_PARAMETERS, USER_SEARCH_BOUNDS)
    evaluator = Lumo3DTrajectoryEvaluator(
        root / "artifacts",
        protocol=protocol,
        objective_config=USER_OBJECTIVE,
        mechanics_contract=execution.mechanics,
        device=execution.device,
        optical_settings=execution.transport,
        led=USER_LED,
        fixed_parameters=USER_PARAMETERS,
        volume_mesh_settings=execution.volume_mesh,
        runtime_identity=runtime_identity_for_device(execution.device),
    )
    expected_resume_contract = _campaign_resume_contract(
        root=root,
        design_space=design_space,
        evaluator=evaluator,
        protocol=protocol,
        objective_config=USER_OBJECTIVE,
        registry_path=selected_registry_path,
        execution=execution,
        budget=budget,
        max_feasibility_resamples=MAX_FEASIBILITY_RESAMPLES,
        source=source,
        allow_cross_revision_cache=allow_cross_revision_cache,
        allow_dirty=allow_dirty,
        seed=campaign_seed,
    )
    config = _user_config_payload(
        execution=execution,
        budget=budget,
        protocol=protocol,
        campaign_mode="smoke" if smoke else "production",
        objective_config=USER_OBJECTIVE,
        search_bounds=USER_SEARCH_BOUNDS,
        seed=campaign_seed,
    )
    config.update(
        {
            "contract_id": evaluator.evaluation_contract_id,
            "evaluation_schema": TRAJECTORY_EVALUATION_SCHEMA,
            "parameterization_version": design_space.parameterization_version,
            "design_space": design_space.to_dict(),
            "output": str(root),
            "registry": str(selected_registry_path),
            "runtime_identity": evaluator.runtime_identity,
            "source": source,
            "resume_contract": expected_resume_contract,
            "registry_cache_policy": {
                "allow_cross_revision_cache": allow_cross_revision_cache,
            },
            "source_policy": {
                "allow_dirty": allow_dirty,
                "dirty_accepted": bool(source.get("git_dirty") and allow_dirty),
            },
        }
    )
    if resume:
        try:
            existing_config = json.loads(
                (root / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read campaign config for resume: {exc}") from exc
        if _canonical_json_payload(
            existing_config.get("resume_contract")
        ) != _canonical_json_payload(expected_resume_contract):
            raise CheckpointError(
                "campaign config does not match current evaluator, fixed inputs, "
                "budget, source, or Ax serialization contract"
            )
    else:
        _json_write(root / "config.json", config)
    registry = EvaluationRegistry(selected_registry_path)
    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise RuntimeError("source provenance has no source_id")
    registry_source_audit = registry.source_audit(
        evaluator.evaluation_contract_id,
        source_id,
    )
    if not allow_cross_revision_cache and (
        registry_source_audit["different_source"]
        or registry_source_audit["unknown_source"]
    ):
        raise RuntimeError(
            "registry contains same-contract evaluations from a different or "
            "unknown source; use a new registry or explicitly pass "
            "--allow-cross-revision-cache"
        )

    preflight = _preflight_payload(
        root,
        execution=execution,
        source=source,
        protocol=protocol,
        objective_config=USER_OBJECTIVE,
        search_bounds=USER_SEARCH_BOUNDS,
    )
    _json_write(
        root / ("preflight_resume.json" if resume else "preflight.json"),
        preflight,
    )
    if preflight["status"] != "PASS":
        raise RuntimeError(
            "external preflight failed; no candidate was submitted: "
            + ", ".join(preflight["failed_checks"])
        )

    records: list[dict[str, Any]] = []
    resume_state = (
        _resume_state(
            checkpoint_store,
            expected_contract=expected_resume_contract,
        )
        if resume
        else None
    )

    def persist(_client: Any, observed: tuple[Any, ...]) -> None:
        attempt = 0
        persisted: list[dict[str, Any]] = []
        for record in observed:
            payload = _record_payload(record, design_space)
            payload["generation_attempt_index"] = (
                None if record.phase == "nominal" else attempt
            )
            if record.phase != "nominal":
                attempt += 1
            persisted.append(payload)
        records[:] = persisted
        _json_write(root / "trials.json", records)

    def checkpoint(event: AxCheckpointEvent) -> None:
        observed: list[dict[str, Any]] = []
        attempt = 0
        for record in event.records:
            payload = _record_payload(record, design_space)
            payload["generation_attempt_index"] = (
                None if record.phase == "nominal" else attempt
            )
            if record.phase != "nominal":
                attempt += 1
            observed.append(payload)
        records[:] = observed
        _json_write(root / "trials.json", records)
        pending = event.pending_trial
        state = {
            "phase": event.phase,
            "campaign_id": root.name,
            "evaluation_contract_id": evaluator.evaluation_contract_id,
            "objective_identifier": {
                "name": evaluator.objective_identifier.name,
                "version": evaluator.objective_identifier.version,
            },
            "design_space": design_space.to_dict(),
            "parameterization_version": design_space.parameterization_version,
            "ax_package_version": _ax_package_version(),
            "seed": campaign_seed,
            "budget": expected_resume_contract["budget"],
            "counts": {
                "historical_success_count": event.historical_success_count,
                "historical_failure_count": event.historical_failure_count,
                "proposal_count": sum(
                    record.phase != "nominal" for record in event.records
                ),
                "evaluation_count": _actual_evaluation_count(event.records),
                "successful_initialization_count": sum(
                    record.phase == "initialization" and record.status == "success"
                    for record in event.records
                ),
                "successful_search_count": sum(
                    record.phase == "search" and record.status == "success"
                    for record in event.records
                ),
            },
            "termination_reason": event.termination_reason,
            "pending_trial_index": (
                None if pending is None else pending.trial_index
            ),
            "pending_phase": None if pending is None else pending.phase,
            "pending_latent_parameters": (
                None if pending is None else dict(pending.latent_parameters)
            ),
            "pending_physical_parameters": (
                None
                if pending is None or pending.physical_parameters is None
                else dict(pending.physical_parameters)
            ),
            "registry_key": None if pending is None else pending.registry_key,
            "source": source,
            "source_policy": config["source_policy"],
            "resume_contract": expected_resume_contract,
        }
        checkpoint_store.write(
            ax_client=event.client,
            trials=observed,
            state=state,
        )

    settings = AxSettings(
        initialization_trials=budget.initialization_success_target,
        search_trials=budget.search_success_target,
        seed=campaign_seed,
        objective=evaluator.objective_identifier,
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
            max_proposals=budget.maximum_proposals,
            max_evaluations=budget.maximum_evaluations,
            max_feasibility_resamples=MAX_FEASIBILITY_RESAMPLES,
            on_checkpoint=checkpoint,
            resume_state=resume_state,
            producer_source=source,
        )
    except CampaignInfrastructureError as exc:
        summary = {
            "status": "INFRASTRUCTURE_FAILED",
            "signature": exc.signature,
            "error": f"{type(exc).__name__}: {exc}",
            "source": source,
            "source_policy": config["source_policy"],
            "records": records,
        }
        _json_write(root / "summary.json", summary)
        raise

    summary = {
        "status": result.status,
        "campaign_id": root.name,
        "resumed": resume,
        "source": source,
        "source_policy": config["source_policy"],
        "checkpoint": str(checkpoint_store.pointer_path),
        "ax_status": result.status,
        "ax_termination_reason": result.termination_reason.value,
        "objective_name": result.objective_name,
        "ax_proposal_count": result.ax_proposal_count,
        "proposal_count": result.proposal_count,
        "new_evaluation_count": result.new_evaluation_count,
        "maximum_evaluations": budget.maximum_evaluations,
        "maximum_proposals": budget.maximum_proposals,
        "initialization_success_target": budget.initialization_success_target,
        "search_success_target": budget.search_success_target,
        "unique_success_count": result.unique_success_count,
        "unique_failure_count": result.unique_failure_count,
        "nominal_successful": result.nominal_successful,
        "successful_initialization_count": result.successful_initialization_count,
        "successful_search_count": result.successful_search_count,
        "successful_generated_count": result.successful_generated_count,
        "failure_count_by_status": dict(result.failure_count_by_status),
        "reused_evaluation_count": result.reused_evaluation_count,
        "pending_trial": result.pending_trial,
        "best_trial": (
            None
            if result.best_record is None
            else _record_payload(result.best_record, design_space)
        ),
        "records": records,
        "feasible_proposal_count": result.feasible_proposal_count,
        "generation_attempt_count": result.generation_attempt_count,
        "feasibility_rejection_count": result.feasibility_rejection_count,
        "feasibility_rejection_counts": dict(result.feasibility_rejection_counts),
        "last_feasibility_rejection": result.last_feasibility_rejection,
        "optical_failure_summary": summarize_optical_failure_diagnostics(records),
        "registry_source_policy": {
            "allow_cross_revision_cache": allow_cross_revision_cache,
            **registry_source_audit,
        },
    }
    campaign_status, acceptance_failures = _campaign_acceptance(
        result,
        budget=budget,
    )
    summary["campaign_acceptance"] = campaign_status
    summary["acceptance_failures"] = acceptance_failures
    summary["status"] = campaign_status
    _json_write(root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--initialization-successes",
        type=int,
        help="required successful Sobol observations (production default: 6)",
    )
    parser.add_argument(
        "--search-successes",
        type=int,
        help="required successful MBM observations",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help="hard cap on actual evaluator calls, including nominal",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        help="hard cap on every Ax-generated proposal, including rejects/duplicates",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execution-config",
        type=Path,
        help="typed mesh/Newton/transport YAML (default: config/lumo_execution.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=None,
        help="use the explicit reduced two-state protocol",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "reuse an existing exact-result registry for smoke/validation only; "
            "production external reuse is currently rejected"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "explicitly resume a campaign output or its current "
            "checkpoint.json pointer"
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="explicitly allow a dirty production source snapshot",
    )
    parser.add_argument(
        "--allow-cross-revision-cache",
        action="store_true",
        help="explicitly reuse same-contract registry results from other sources",
    )
    args = parser.parse_args(argv)

    resume_config: dict[str, Any] | None = None
    if args.resume is not None:
        try:
            resume_root = _resume_root(args.resume)
            resume_config = json.loads(
                (resume_root / "config.json").read_text(encoding="utf-8")
            )
        except (CheckpointError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"CHECKPOINT_ABORTED: cannot read resume config: {exc}", file=sys.stderr)
            return 2
        if args.smoke is None:
            args.smoke = resume_config.get("campaign_mode") == "smoke"
    if args.smoke is None:
        args.smoke = False

    if args.execution_config is None:
        if resume_config is not None:
            try:
                args.execution_config = Path(
                    resume_config["execution_config"]["source"]["path"]
                )
            except (KeyError, TypeError) as exc:
                print(
                    f"CHECKPOINT_ABORTED: resume config has no execution YAML: {exc}",
                    file=sys.stderr,
                )
                return 2
        else:
            args.execution_config = DEFAULT_EXECUTION_CONFIG

    try:
        execution = load_lumo_execution_config(args.execution_config)
    except (OSError, TypeError, ValueError) as exc:
        print(f"CONFIGURATION_ABORTED: {exc}", file=sys.stderr)
        return 2

    if args.preflight:
        source = _source_provenance(excluded_paths=(args.output,))
        try:
            _enforce_source_policy(source, smoke=True, allow_dirty=True)
        except RuntimeError as exc:
            print(f"PREFLIGHT_ABORTED: {exc}", file=sys.stderr)
            return 2
        preflight = _preflight_payload(
            args.output,
            execution=execution,
            source=source,
            protocol=SMOKE_PROTOCOL if args.smoke else USER_PROTOCOL,
        )
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return 0 if preflight["status"] == "PASS" else 2

    if resume_config is not None:
        try:
            persisted_budget = resume_config["ax"]
            initialization_successes = (
                args.initialization_successes
                if args.initialization_successes is not None
                else int(persisted_budget["initialization_success_target"])
            )
            search_successes = (
                args.search_successes
                if args.search_successes is not None
                else int(persisted_budget["search_success_target"])
            )
            maximum_evaluations = (
                args.max_evaluations
                if args.max_evaluations is not None
                else int(persisted_budget["maximum_evaluations"])
            )
            maximum_proposals = (
                args.max_proposals
                if args.max_proposals is not None
                else int(persisted_budget["maximum_proposals"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"CHECKPOINT_ABORTED: invalid persisted budget: {exc}", file=sys.stderr)
            return 2
    elif args.smoke:
        initialization_successes = (
            SMOKE_INITIALIZATION_SUCCESS_TARGET
            if args.initialization_successes is None
            else args.initialization_successes
        )
        search_successes = 0 if args.search_successes is None else args.search_successes
        maximum_evaluations = (
            1 + initialization_successes + search_successes
            if args.max_evaluations is None
            else args.max_evaluations
        )
        maximum_proposals = (
            initialization_successes + search_successes
            if args.max_proposals is None
            else args.max_proposals
        )
    else:
        initialization_successes = (
            PRODUCTION_INITIALIZATION_SUCCESS_TARGET
            if args.initialization_successes is None
            else args.initialization_successes
        )
        if (
            args.search_successes is None
            or args.max_evaluations is None
            or args.max_proposals is None
        ):
            parser.error(
                "production requires --search-successes, --max-evaluations, "
                "and --max-proposals"
            )
        search_successes = args.search_successes
        maximum_evaluations = args.max_evaluations
        maximum_proposals = args.max_proposals
        if initialization_successes < PRODUCTION_INITIALIZATION_SUCCESS_TARGET:
            parser.error(
                "production requires --initialization-successes >= "
                f"{PRODUCTION_INITIALIZATION_SUCCESS_TARGET}"
            )
        if search_successes < 1:
            parser.error("production requires --search-successes >= 1")
    try:
        budget = CampaignBudget(
            initialization_success_target=initialization_successes,
            search_success_target=search_successes,
            maximum_evaluations=maximum_evaluations,
            maximum_proposals=maximum_proposals,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        summary = run_campaign(
            args.output,
            budget=budget,
            smoke=args.smoke,
            registry_path=args.registry,
            resume=args.resume,
            execution_config=execution,
            allow_dirty=args.allow_dirty,
            allow_cross_revision_cache=args.allow_cross_revision_cache,
        )
    except CampaignInfrastructureError as exc:
        print(f"INFRASTRUCTURE_FAILED [{exc.signature}]: {exc}", file=sys.stderr)
        return 2
    except (CheckpointError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"CAMPAIGN_ABORTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary.get("status") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
