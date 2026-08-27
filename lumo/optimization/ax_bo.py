"""Sequential Ax multi-objective optimization for the current fingertip."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib.resources import as_file, files
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.observation import ObservationFeatures
from torch.quasirandom import SobolEngine

from lumo.fingertip import (
    MECHANICS_PRESETS,
    OPTICAL_PRESETS,
    Fingertip,
    FingertipParameters,
)

from .campaign_io import (
    AX_STATE_FILENAME,
    RUN_CONFIG_FILENAME,
    TRIALS_FILENAME,
    apply_result_to_row,
    ax_statuses,
    completed_trial_count,
    finalize_outputs,
    initialize_campaign,
    persist_campaign,
    resume_campaign,
    running_row,
    save_ax,
    save_trial_result,
    trial_result_path,
    write_tables,
)
from .design_space import (
    MAX_FINGERTIP_HEIGHT_MM,
    DesignSpace,
    ParameterBound,
)


_STEP_TO_PHYSICAL_NAMES = (
    ("flat_pad_height_step", "geometry.flat_pad_height_mm"),
    ("semiellipse_height_step", "geometry.semiellipse_height_mm"),
    ("stem_width_step", "geometry.stem_width_mm"),
    ("stem_height_step", "geometry.stem_height_mm"),
    ("void_width_step", "geometry.void_width_mm"),
)
_RESOLUTION_MM = 0.5
_FIXED_FLAT_PAD_WIDTH_MM = 30.0
_FIXED_LINK_THICKNESS_MM = FingertipParameters().geometry.link_thickness_mm
_MAX_PAD_DEPTH_STEPS = round(
    (MAX_FINGERTIP_HEIGHT_MM - _FIXED_LINK_THICKNESS_MM) / _RESOLUTION_MM
)
_MAX_CUTOUT_WIDTH_STEPS = (
    ceil(
        (
            _FIXED_FLAT_PAD_WIDTH_MM
            - 2.0 * FingertipParameters().geometry.bond_extension_width_mm
        )
        / _RESOLUTION_MM
    )
    - 1
)
_DEFAULT_PARAMETER_BOUNDS_MM = {
    "geometry.flat_pad_height_mm": (2.0, 29.0),
    "geometry.semiellipse_height_mm": (1.0, 20.0),
    "geometry.stem_width_mm": (6.0, 10.0),
    "geometry.stem_height_mm": (4.0, 10.0),
    "geometry.void_width_mm": (0.0, 4.0),
}
_DEFAULT_INDENTER_URDFS = (
    "sphere_5mm.urdf",
    "sphere_10mm.urdf",
    "sphere_20mm.urdf",
)
_DEFAULT_SPHERE_DIAMETERS_MM = (5.0, 10.0, 20.0)
_DEFAULT_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_DEFAULT_CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)
_DEFAULT_INITIAL_CLEARANCE_M = 1.0e-3
_DEFAULT_MECHANICS_PRESET = "silicone"
_DEFAULT_OPTICAL_PRESET = "dragon_skin_10_nv_nominal"
_RANDOM_SEED = 20260823
_INITIALIZATION_BUDGET = 13
_ACQUISITION_POOL_SIZE = 256
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_MAX_PROPOSALS_PER_COMPLETED_TRIAL = 50


@dataclass(frozen=True)
class CampaignDefinition:
    """Frozen search, scenario, and material contract for one Ax campaign."""

    space: DesignSpace
    ax_parameters: tuple[RangeParameterConfig, ...]
    ax_parameter_constraints: tuple[str, ...]
    step_to_physical: tuple[tuple[str, str, int, int], ...]
    indenters: tuple[tuple[str, str], ...]
    sphere_diameters_mm: tuple[float, ...]
    force_targets_n: tuple[float, ...]
    initial_clearance_m: float
    contact_y_mm: tuple[float, ...]
    mechanics_preset: str
    optical_preset: str
    resolution_mm: float = _RESOLUTION_MM
    initialization_budget: int = _INITIALIZATION_BUDGET
    random_seed: int = _RANDOM_SEED
    acquisition_pool_size: int = _ACQUISITION_POOL_SIZE
    approach_speed_m_s: float = _APPROACH_SPEED_M_S
    max_sim_time_s: float = _MAX_SIM_TIME_S

    @property
    def ax_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.ax_parameters)

    @property
    def physical_parameter_names(self) -> tuple[str, ...]:
        return self.space.variable_names

    @property
    def parameter_columns(self) -> tuple[str, ...]:
        return self.ax_parameter_names + self.physical_parameter_names

    @property
    def indenter_names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self.indenters)

    def decode(self, raw_parameters: Mapping[str, object]) -> dict[str, float]:
        """Decode Ax integer steps into physical millimetres."""
        if set(raw_parameters) != set(self.ax_parameter_names):
            raise ValueError("Ax candidate does not match the campaign parameters")
        parameters = {}
        for step_name, physical_name, lower, upper in self.step_to_physical:
            raw_step = raw_parameters[step_name]
            step = int(raw_step)
            if isinstance(raw_step, bool) or float(raw_step) != step:
                raise ValueError(f"{step_name} must be an exact integer")
            if not lower <= step <= upper:
                raise ValueError(f"{step_name} lies outside its integer bounds")
            parameters[physical_name] = self.resolution_mm * step
        return parameters

    def encode(self, parameters: Mapping[str, float]) -> dict[str, int]:
        """Encode physical millimetres as exact Ax half-millimetre steps."""
        if set(parameters) != set(self.physical_parameter_names):
            raise ValueError("candidate does not match the physical parameters")
        encoded = {}
        for step_name, physical_name, lower, upper in self.step_to_physical:
            scaled = float(parameters[physical_name]) / self.resolution_mm
            step = round(scaled)
            if not np.isclose(scaled, step, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"{physical_name} is not on the 0.5 mm grid")
            if not lower <= step <= upper:
                raise ValueError(f"{physical_name} lies outside its bounds")
            encoded[step_name] = step
        return encoded

    def validate(
        self,
        raw_parameters: Mapping[str, object],
        parameters: Mapping[str, float],
    ) -> None:
        """Require exact encoding and both mirrored Ax linear constraints."""
        if self.encode(parameters) != {
            name: raw_parameters[name] for name in self.ax_parameter_names
        }:
            raise ValueError("Ax candidate does not round-trip through physical units")
        if (
            int(raw_parameters["flat_pad_height_step"])
            + int(raw_parameters["semiellipse_height_step"])
            > _MAX_PAD_DEPTH_STEPS
        ):
            raise ValueError("Ax candidate violates the full-height constraint")
        if (
            int(raw_parameters["stem_width_step"])
            + 2 * int(raw_parameters["void_width_step"])
            > _MAX_CUTOUT_WIDTH_STEPS
        ):
            raise ValueError("Ax candidate violates the bonded-side-width constraint")
        if not self.space.is_feasible(parameters):
            raise ValueError("Ax candidate is outside the physical design space")


def _parameter_bounds(
    parameter_bounds_mm: Mapping[str, tuple[float, float]] | None,
) -> tuple[dict[str, tuple[float, float]], tuple[tuple[str, str, int, int], ...]]:
    if parameter_bounds_mm is None:
        qualified_bounds = dict(_DEFAULT_PARAMETER_BOUNDS_MM)
    else:
        expected = {
            physical_name.removeprefix("geometry.")
            for _, physical_name in _STEP_TO_PHYSICAL_NAMES
        }
        if set(parameter_bounds_mm) != expected:
            raise ValueError(
                f"parameter_bounds_mm must define exactly {sorted(expected)!r}"
            )
        qualified_bounds = {
            f"geometry.{name}": tuple(bounds)
            for name, bounds in parameter_bounds_mm.items()
        }

    step_to_physical = []
    for step_name, physical_name in _STEP_TO_PHYSICAL_NAMES:
        bounds = qualified_bounds[physical_name]
        if len(bounds) != 2:
            raise ValueError(f"{physical_name} bounds must contain two values")
        lower, upper = (float(value) for value in bounds)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError(
                f"{physical_name} bounds must be finite with lower < upper"
            )
        lower_step = round(lower / _RESOLUTION_MM)
        upper_step = round(upper / _RESOLUTION_MM)
        if not np.isclose(
            lower, _RESOLUTION_MM * lower_step, atol=1.0e-12
        ) or not np.isclose(upper, _RESOLUTION_MM * upper_step, atol=1.0e-12):
            raise ValueError(f"{physical_name} bounds must lie on the 0.5 mm grid")
        qualified_bounds[physical_name] = (lower, upper)
        step_to_physical.append((step_name, physical_name, lower_step, upper_step))
    return qualified_bounds, tuple(step_to_physical)


def _indenter_definitions(
    indenter_urdfs: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    definitions = []
    for raw_name in indenter_urdfs:
        if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
            raise ValueError("indenter URDFs must be plain filenames")
        if Path(raw_name).suffix.lower() != ".urdf":
            raise ValueError("indenter filenames must end in .urdf")
        resource = files("lumo").joinpath("assets", "objects", "urdf", raw_name)
        if not resource.is_file():
            raise FileNotFoundError(resource)
        definitions.append((raw_name, Path(raw_name).stem))
    if not definitions:
        raise ValueError("indenter_urdfs must contain at least one URDF")
    if len({name for _, name in definitions}) != len(definitions):
        raise ValueError("indenter filenames must have unique stems")
    return tuple(definitions)


def _force_targets(force_targets_n: Iterable[float]) -> tuple[float, ...]:
    targets = tuple(float(target) for target in force_targets_n)
    if len(targets) < 3:
        raise ValueError("force_targets_n must contain at least three targets")
    if any(not isfinite(target) or target <= 0.0 for target in targets):
        raise ValueError("force targets must be finite and positive")
    if any(current <= previous for previous, current in zip(targets, targets[1:])):
        raise ValueError("force_targets_n must be strictly increasing")
    return targets


def build_campaign(
    *,
    parameter_bounds_mm: Mapping[str, tuple[float, float]] | None = None,
    indenter_urdfs: Iterable[str] = _DEFAULT_INDENTER_URDFS,
    sphere_diameters_mm: Iterable[float] = _DEFAULT_SPHERE_DIAMETERS_MM,
    force_targets_n: Iterable[float] = _DEFAULT_FORCE_TARGETS_N,
    initial_clearance_m: float = _DEFAULT_INITIAL_CLEARANCE_M,
    contact_y_mm: Iterable[float] = _DEFAULT_CONTACT_Y_MM,
    mechanics_preset: str = _DEFAULT_MECHANICS_PRESET,
    optical_preset: str = _DEFAULT_OPTICAL_PRESET,
) -> CampaignDefinition:
    """Build the one current half-millimetre production campaign."""
    indenters = _indenter_definitions(indenter_urdfs)
    diameters = tuple(float(value) for value in sphere_diameters_mm)
    if len(diameters) != len(indenters) or any(
        not isfinite(value) or value <= 0.0 for value in diameters
    ):
        raise ValueError(
            "sphere_diameters_mm must contain one positive value per indenter"
        )
    if len(set(diameters)) != len(diameters):
        raise ValueError("sphere_diameters_mm must be unique")

    locations = tuple(float(value) for value in contact_y_mm)
    if not locations or any(
        not isfinite(value) or not -27.5 <= value <= 27.5 for value in locations
    ):
        raise ValueError(
            "contact_y_mm must contain finite locations inside [-27.5, 27.5] mm"
        )
    if len(set(locations)) != len(locations):
        raise ValueError("contact_y_mm must be unique")

    clearance = float(initial_clearance_m)
    if not isfinite(clearance) or clearance < 0.0:
        raise ValueError("initial_clearance_m must be finite and nonnegative")
    if mechanics_preset not in MECHANICS_PRESETS:
        raise ValueError(
            f"mechanics_preset must be one of {sorted(MECHANICS_PRESETS)!r}"
        )
    if optical_preset not in OPTICAL_PRESETS:
        raise ValueError(f"optical_preset must be one of {sorted(OPTICAL_PRESETS)!r}")

    bounds, step_to_physical = _parameter_bounds(parameter_bounds_mm)
    base_parameters = FingertipParameters(
        geometry=replace(
            FingertipParameters().geometry,
            flat_pad_width_mm=_FIXED_FLAT_PAD_WIDTH_MM,
        ),
        mechanics=MECHANICS_PRESETS[mechanics_preset],
        optics=OPTICAL_PRESETS[optical_preset],
    )
    space = DesignSpace(
        base_parameters=base_parameters,
        geometry_bounds={
            name.removeprefix("geometry."): ParameterBound(*limits)
            for name, limits in bounds.items()
        },
    )
    return CampaignDefinition(
        space=space,
        ax_parameters=tuple(
            RangeParameterConfig(
                name=step_name,
                bounds=(lower, upper),
                parameter_type="int",
            )
            for step_name, _, lower, upper in step_to_physical
        ),
        ax_parameter_constraints=(
            f"flat_pad_height_step + semiellipse_height_step <= {_MAX_PAD_DEPTH_STEPS}",
            f"stem_width_step + 2 * void_width_step <= {_MAX_CUTOUT_WIDTH_STEPS}",
        ),
        step_to_physical=step_to_physical,
        indenters=indenters,
        sphere_diameters_mm=diameters,
        force_targets_n=_force_targets(force_targets_n),
        initial_clearance_m=clearance,
        contact_y_mm=locations,
        mechanics_preset=mechanics_preset,
        optical_preset=optical_preset,
    )


def new_client(campaign: CampaignDefinition) -> Client:
    """Create the Ax experiment for the current two-objective campaign."""
    client = Client(random_seed=campaign.random_seed)
    client.configure_experiment(
        name="lumo_fingertip_objectives_discrete_05mm",
        description=(
            "LUMO fingertip contact and observation optimization over "
            f"indenters {campaign.indenter_names}, contact Y locations "
            f"{campaign.contact_y_mm} mm, and force states "
            f"{campaign.force_targets_n} N"
        ),
        parameters=campaign.ax_parameters,
        parameter_constraints=campaign.ax_parameter_constraints,
    )
    client.configure_optimization(objective="J_contact, J_obs")
    client.configure_generation_strategy(
        method="fast",
        initialization_budget=campaign.initialization_budget,
        initialization_random_seed=campaign.random_seed,
        initialize_with_center=False,
        use_existing_trials_for_initialization=True,
    )
    return client


def _used_step_tuples(
    client: Client,
    campaign: CampaignDefinition,
) -> set[tuple[int, ...]]:
    return {
        tuple(int(arm.parameters[name]) for name in campaign.ax_parameter_names)
        for trial in client._experiment.trials.values()
        for arm in trial.arms
    }


def feasible_candidate_pool(
    campaign: CampaignDefinition,
    *,
    count: int,
    seed: int,
    excluded: set[tuple[int, ...]],
) -> list[dict[str, int]]:
    """Return deterministic lattice candidates accepted by ``DesignSpace``."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    step_bounds = np.asarray(
        [(lower, upper) for _, _, lower, upper in campaign.step_to_physical],
        dtype=np.int64,
    )
    lower = step_bounds[:, 0]
    cardinality = step_bounds[:, 1] - lower + 1
    engine = SobolEngine(
        dimension=len(campaign.ax_parameter_names),
        scramble=True,
        seed=seed,
    )
    max_draws = max(4096, 64 * count)
    seen = set(excluded)
    candidates = []
    draws = 0
    while len(candidates) < count and draws < max_draws:
        batch_size = min(256, max_draws - draws)
        batch = lower + np.floor(
            engine.draw(batch_size).cpu().numpy() * cardinality
        ).astype(np.int64)
        draws += batch_size
        for raw_steps in batch:
            steps = tuple(int(step) for step in raw_steps)
            if steps in seen:
                continue
            seen.add(steps)
            raw_parameters = dict(zip(campaign.ax_parameter_names, steps, strict=True))
            try:
                parameters = campaign.decode(raw_parameters)
                campaign.validate(raw_parameters, parameters)
            except ValueError:
                continue
            candidates.append(raw_parameters)
            if len(candidates) == count:
                break
    if len(candidates) != count:
        raise RuntimeError(
            "could not construct the requested exact-feasible lattice pool: "
            f"found {len(candidates)} of {count} after {draws} Sobol points"
        )
    return candidates


def propose_feasible_trial(
    client: Client,
    campaign: CampaignDefinition,
) -> tuple[int, dict[str, int], dict[str, float], str]:
    """Attach one exact-feasible Sobol or model-based Ax trial."""
    strategy = client._generation_strategy
    adapter = strategy.fit(client._experiment)
    predictive = adapter is not None and adapter.can_predict
    pool = feasible_candidate_pool(
        campaign,
        count=campaign.acquisition_pool_size if predictive else 1,
        seed=campaign.random_seed + len(client._experiment.trials),
        excluded=_used_step_tuples(client, campaign),
    )
    if predictive:
        features = [ObservationFeatures(parameters=parameters) for parameters in pool]
        acquisition = np.asarray(
            adapter.evaluate_acquisition_function(features), dtype=np.float64
        )
        finite = np.isfinite(acquisition)
        if not np.any(finite):
            raise RuntimeError("Ax acquisition is nonfinite for every feasible point")
        acquisition[~finite] = -np.inf
        raw_parameters = pool[int(np.argmax(acquisition))]
    else:
        raw_parameters = pool[0]

    parameters = campaign.decode(raw_parameters)
    campaign.validate(raw_parameters, parameters)
    trial_index = client.attach_trial(parameters=raw_parameters)
    return (
        trial_index,
        raw_parameters,
        parameters,
        f"FEASIBLE_{strategy.current_node_name}",
    )


def _evaluate_candidate(
    campaign: CampaignDefinition,
    parameters: dict[str, float],
):
    from .evaluator import evaluate_fingertip

    fingertip = Fingertip(campaign.space.to_parameters(parameters))
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    with ExitStack() as resources:
        indenter_paths = tuple(
            resources.enter_context(as_file(resource_root.joinpath(filename)))
            for filename, _ in campaign.indenters
        )
        return evaluate_fingertip(
            fingertip,
            indenter_paths,
            campaign.sphere_diameters_mm,
            campaign.contact_y_mm,
            force_targets_n=campaign.force_targets_n,
            initial_clearance_m=campaign.initial_clearance_m,
            approach_speed_m_s=campaign.approach_speed_m_s,
            max_sim_time_s=campaign.max_sim_time_s,
        )


def objective_details(evaluation: object) -> dict[str, object]:
    from .objective import compute_objectives_from_raw

    contact, observation = compute_objectives_from_raw(vars(evaluation))
    return {
        "J_contact": contact.J_contact,
        "J_obs": observation.J_obs,
        "contact": contact,
        "observation": observation,
        "max_outside_roi_power_fraction": float(
            max(
                evaluation.no_contact_outside_roi_power_fraction,
                np.max(evaluation.outside_roi_power_fraction),
            )
        ),
    }


def _validate_optix_environment() -> None:
    required = {
        "OPTIX_INCLUDE_DIR": Path("optix.h"),
        "OTK_INCLUDE_DIR": Path("OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h"),
    }
    failures = []
    for variable, relative_path in required.items():
        raw_directory = os.environ.get(variable)
        if not raw_directory:
            failures.append(f"{variable} is not set")
        elif not (Path(raw_directory) / relative_path).is_file():
            failures.append(f"{variable} does not contain {relative_path}")
    if failures:
        raise RuntimeError("OptiX environment preflight failed: " + "; ".join(failures))


def _is_infrastructure_failure(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in message
        for marker in (
            "optix_include_dir",
            "otk_include_dir",
            "cuda driver",
            "cuda error",
            "nvrtc",
            "failed to load optix",
            "no cuda",
            "out of memory",
        )
    )


def run(
    *,
    output_directory: Path,
    target_bo_trials: int,
    parameter_bounds_mm: Mapping[str, tuple[float, float]] | None = None,
    indenter_urdfs: Iterable[str] = _DEFAULT_INDENTER_URDFS,
    sphere_diameters_mm: Iterable[float] = _DEFAULT_SPHERE_DIAMETERS_MM,
    force_targets_n: Iterable[float] = _DEFAULT_FORCE_TARGETS_N,
    initial_clearance_m: float = _DEFAULT_INITIAL_CLEARANCE_M,
    contact_y_mm: Iterable[float] = _DEFAULT_CONTACT_Y_MM,
    mechanics_preset: str = _DEFAULT_MECHANICS_PRESET,
    optical_preset: str = _DEFAULT_OPTICAL_PRESET,
) -> list[dict[str, object]]:
    """Create or resume the current sequential Ax campaign."""
    if target_bo_trials < 0:
        raise ValueError("target_bo_trials must be nonnegative")
    command_start_s = perf_counter()
    output_directory = output_directory.resolve()
    campaign = build_campaign(
        parameter_bounds_mm=parameter_bounds_mm,
        indenter_urdfs=indenter_urdfs,
        sphere_diameters_mm=sphere_diameters_mm,
        force_targets_n=force_targets_n,
        initial_clearance_m=initial_clearance_m,
        contact_y_mm=contact_y_mm,
        mechanics_preset=mechanics_preset,
        optical_preset=optical_preset,
    )
    campaign_exists = any(
        (output_directory / filename).exists()
        for filename in (RUN_CONFIG_FILENAME, AX_STATE_FILENAME, TRIALS_FILENAME)
    )
    if campaign_exists:
        client, rows, config = resume_campaign(output_directory, campaign)
    else:
        client = new_client(campaign)
        rows, config = initialize_campaign(output_directory, campaign, client)

    initial_completed = completed_trial_count(rows)
    if initial_completed < target_bo_trials:
        _validate_optix_environment()
    remaining = target_bo_trials - initial_completed
    proposal_limit = max(1, remaining) * _MAX_PROPOSALS_PER_COMPLETED_TRIAL
    proposal_count = 0

    while completed_trial_count(rows) < target_bo_trials:
        if proposal_count >= proposal_limit:
            raise RuntimeError(
                "Ax did not produce enough successful candidates "
                f"within {proposal_limit} proposals"
            )
        trial_index, raw_parameters, parameters, generation_node = (
            propose_feasible_trial(client, campaign)
        )
        row = running_row(
            trial_index,
            raw_parameters,
            parameters,
            generation_node,
        )
        rows.append(row)
        proposal_count += 1
        persist_campaign(client, rows, output_directory, campaign)
        current_trial = completed_trial_count(rows) + 1
        print(
            f"current_trial={current_trial}/{target_bo_trials} | "
            f"proposed Ax trial {trial_index} ({generation_node}): "
            f"Ax={raw_parameters}, physical_mm={parameters}",
            flush=True,
        )

        evaluation_start_s = perf_counter()
        try:
            evaluation = _evaluate_candidate(campaign, parameters)
            runtime_s = perf_counter() - evaluation_start_s
            details = objective_details(evaluation)
            raw_path = trial_result_path(output_directory, trial_index)
            save_trial_result(
                raw_path,
                campaign=campaign,
                evaluation=evaluation,
                details=details,
                parameters=parameters,
                runtime_s=runtime_s,
            )
            apply_result_to_row(
                row,
                details,
                runtime_s=runtime_s,
                raw_result_path=raw_path.relative_to(output_directory).as_posix(),
            )
            row["status"] = "EVALUATED"
            write_tables(output_directory, rows, campaign)
            client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "J_contact": float(row["J_contact"]),
                    "J_obs": float(row["J_obs"]),
                },
            )
            save_ax(client, output_directory / AX_STATE_FILENAME)
            row["status"] = "COMPLETED"
            write_tables(output_directory, rows, campaign)
            del evaluation
        except Exception as error:
            row["runtime_s"] = perf_counter() - evaluation_start_s
            row["status"] = "FAILED"
            row["failure"] = f"{type(error).__name__}: {error}"
            if ax_statuses(client)[trial_index] == "RUNNING":
                client.mark_trial_abandoned(trial_index)
            persist_campaign(client, rows, output_directory, campaign)
            print(f"trial {trial_index} failed: {row['failure']}", flush=True)
            if _is_infrastructure_failure(error):
                raise RuntimeError(
                    f"trial {trial_index} evaluation failed; campaign was saved"
                ) from error
            continue

        print(
            f"completed trial {trial_index}: "
            f"J_contact={float(row['J_contact']):.9e}, "
            f"J_obs={float(row['J_obs']):.9e}, "
            f"runtime={float(row['runtime_s']):.3f} s",
            flush=True,
        )

    summary = finalize_outputs(
        output_directory,
        rows,
        config,
        campaign,
        command_wall_runtime_s=perf_counter() - command_start_s,
    )
    print(
        f"target reached: {completed_trial_count(rows)} completed BO trials; "
        f"Pareto={summary['counts']['pareto']}",
        flush=True,
    )
    print(f"outputs: {output_directory}", flush=True)
    return rows


__all__ = [
    "CampaignDefinition",
    "build_campaign",
    "objective_details",
    "run",
]
