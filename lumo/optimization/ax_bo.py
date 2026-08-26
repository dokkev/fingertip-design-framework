"""Ax campaign definitions for LUMO morphology optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from lumo.fingertip import (
    OPTICAL_PRESETS,
    VISCOELASTIC_PRESETS,
    FingertipParameters,
)

from .design_param_bound import DesignParameterBounds, ParameterBound
from .design_space import (
    MAX_FINGERTIP_HEIGHT_MM,
    DesignSpace,
    LinearConstraint,
)


_CONTINUOUS_PARAMETER_NAMES = (
    "geometry.flat_pad_width_mm",
    "geometry.flat_pad_height_mm",
    "geometry.semiellipse_height_mm",
    "geometry.stem_width_mm",
    "geometry.void_width_mm",
    "geometry.void_height_mm",
)
_DISCRETE_STEP_TO_PHYSICAL_NAMES = (
    ("flat_pad_height_step", "geometry.flat_pad_height_mm"),
    ("semiellipse_height_step", "geometry.semiellipse_height_mm"),
    ("stem_width_step", "geometry.stem_width_mm"),
    ("stem_height_step", "geometry.stem_height_mm"),
    ("void_width_step", "geometry.void_width_mm"),
)
_DISCRETE_RESOLUTION_MM = 0.5
_DISCRETE_FIXED_FLAT_PAD_WIDTH_MM = 30.0
_FIXED_UPPER_HEIGHT_MM = FingertipParameters().geometry.link_thickness_mm
_MAX_PAD_DEPTH_MM = MAX_FINGERTIP_HEIGHT_MM - _FIXED_UPPER_HEIGHT_MM
_DISCRETE_MAX_PAD_DEPTH_STEPS = round(
    _MAX_PAD_DEPTH_MM / _DISCRETE_RESOLUTION_MM
)
_DEFAULT_DISCRETE_PARAMETER_BOUNDS_MM = {
    "geometry.flat_pad_height_mm": (2.0, 29.0),
    "geometry.semiellipse_height_mm": (1.0, 20.0),
    "geometry.stem_width_mm": (6.0, 10.0),
    "geometry.stem_height_mm": (4.0, 10.0),
    "geometry.void_width_mm": (0.0, 4.0),
}
_DISCRETE_SEED_STEPS = (
    ("continuous_pareto_0116", (13, 40, 16, 12, 0)),
    ("continuous_pareto_0164", (16, 24, 14, 12, 0)),
    ("continuous_pareto_0226", (16, 40, 16, 12, 0)),
    ("continuous_pareto_0251", (16, 40, 14, 12, 0)),
    ("continuous_pareto_0271", (16, 36, 14, 12, 0)),
    ("continuous_pareto_0282", (16, 40, 14, 12, 1)),
)
_OBJECTIVE_NAMES = ("J_contact", "J_obs")
_DEFAULT_INDENTER_URDFS = (
    "sphere_5mm.urdf",
    "sphere_10mm.urdf",
    "sphere_20mm.urdf",
)
_DEFAULT_SPHERE_DIAMETERS_MM = (5.0, 10.0, 20.0)
_DEFAULT_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_DEFAULT_INITIAL_CLEARANCE_M = 1.0e-3
_DEFAULT_CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)
_DEFAULT_VISCOELASTIC_PRESET = "silicone"
_DEFAULT_OPTICAL_PRESET = "dragon_skin_10_nv_nominal"
_CONTINUOUS_WARM_START_RESULT_FIELDS = (
    ("sphere_5mm", "J_intensity_5mm", "J_spatial_5mm"),
    ("sphere_10mm", "J_intensity_10mm", "J_spatial_10mm"),
    ("sphere_20mm", "J_intensity_20mm", "J_spatial_20mm"),
)
_CONTINUOUS_WARM_START_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "validation"
    / "sensing_objective_tradeoff.csv"
)
_CONTINUOUS_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "output" / "optimization" / "mobo"
)
_DISCRETE_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "optimization"
    / "mobo_full_finger_instantaneous_05mm"
)
_RUN_CONFIG_FILENAME = "run_config.json"
_AX_STATE_FILENAME = "ax_state.json"
_TRIALS_FILENAME = "trials.csv"
_PARETO_FILENAME = "pareto.csv"
_SUMMARY_FILENAME = "run_summary.json"
_TRIAL_RESULT_DIRECTORY = "trials"
_RANDOM_SEED = 20260823
_CONTINUOUS_WARM_START_COUNT = 13
_APPROACH_SPEED_M_S = 5.0e-3
_MAX_SIM_TIME_S = 60.0
_MAX_PROPOSALS_PER_COMPLETED_TRIAL = 50
_OBJECTIVE_DEFINITION = (
    "full-finger-contact-and-threshold-conditioned-observation-v2"
)
_RUN_CONFIG_SCHEMA = 7


@dataclass(frozen=True)
class CampaignDefinition:
    """One concrete Ax search-space contract."""

    name: str
    experiment_name: str
    default_output_directory: Path
    space: DesignSpace
    physical_parameter_names: tuple[str, ...]
    ax_parameters: tuple[RangeParameterConfig, ...]
    ax_parameter_constraints: tuple[str, ...]
    initialization_budget: int
    warm_start_path: Path | None = None
    warm_start_count: int = 0
    resolution_mm: float | None = None
    fixed_geometry: tuple[tuple[str, float], ...] = ()
    discrete_step_to_physical: tuple[tuple[str, str, int, int], ...] = ()
    seed_steps: tuple[tuple[str, tuple[int, ...]], ...] = ()
    indenters: tuple[tuple[str, str], ...] = ()
    sphere_diameters_mm: tuple[float, ...] = _DEFAULT_SPHERE_DIAMETERS_MM
    force_targets_n: tuple[float, ...] = _DEFAULT_FORCE_TARGETS_N
    initial_clearance_m: float = _DEFAULT_INITIAL_CLEARANCE_M
    contact_y_mm: tuple[float, ...] = _DEFAULT_CONTACT_Y_MM
    viscoelastic_preset: str = _DEFAULT_VISCOELASTIC_PRESET
    optical_preset: str = _DEFAULT_OPTICAL_PRESET

    @property
    def ax_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.ax_parameters)

    @property
    def parameter_columns(self) -> tuple[str, ...]:
        if self.ax_parameter_names == self.physical_parameter_names:
            return self.physical_parameter_names
        return self.ax_parameter_names + self.physical_parameter_names

    @property
    def is_discrete(self) -> bool:
        return self.resolution_mm is not None

    @property
    def indenter_names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self.indenters)

    @property
    def per_indenter_fields(self) -> tuple[str, ...]:
        return ()


def _continuous_design_space() -> DesignSpace:
    bounds = DesignParameterBounds(
        parameters=FingertipParameters(),
        geometry={
            "flat_pad_width_mm": ParameterBound(25.0, 35.0),
            "flat_pad_height_mm": ParameterBound(3.0, 8.0),
            "semiellipse_height_mm": ParameterBound(6.0, 20.0),
            "stem_width_mm": ParameterBound(7.0, 10.0),
            "void_width_mm": ParameterBound(0.0, 3.0),
            "void_height_mm": ParameterBound(0.0, 3.0),
        },
    )
    return DesignSpace(
        parameter_bounds=bounds,
        linear_constraints=(
            LinearConstraint(
                coefficients={
                    "geometry.flat_pad_height_mm": 1.0,
                    "geometry.semiellipse_height_mm": 1.0,
                },
                upper=_MAX_PAD_DEPTH_MM,
            ),
        ),
        minimum_silicone_thickness_mm=5.0,
    )


def _discrete_parameter_bounds(
    parameter_bounds_mm: Mapping[str, tuple[float, float]] | None,
) -> tuple[dict[str, tuple[float, float]], tuple[tuple[str, str, int, int], ...]]:
    if parameter_bounds_mm is None:
        qualified_bounds = dict(_DEFAULT_DISCRETE_PARAMETER_BOUNDS_MM)
    else:
        expected_names = {
            physical_name.removeprefix("geometry.")
            for _, physical_name in _DISCRETE_STEP_TO_PHYSICAL_NAMES
        }
        if set(parameter_bounds_mm) != expected_names:
            raise ValueError(
                f"parameter_bounds_mm must define exactly {sorted(expected_names)!r}"
            )
        qualified_bounds = {
            f"geometry.{name}": tuple(bounds)
            for name, bounds in parameter_bounds_mm.items()
        }

    step_to_physical = []
    for step_name, physical_name in _DISCRETE_STEP_TO_PHYSICAL_NAMES:
        bounds = qualified_bounds[physical_name]
        if len(bounds) != 2:
            raise ValueError(f"{physical_name} bounds must contain two values")
        lower, upper = (float(value) for value in bounds)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError(
                f"{physical_name} bounds must be finite with lower < upper"
            )
        lower_step = round(lower / _DISCRETE_RESOLUTION_MM)
        upper_step = round(upper / _DISCRETE_RESOLUTION_MM)
        if not np.isclose(
            lower,
            _DISCRETE_RESOLUTION_MM * lower_step,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.isclose(
            upper,
            _DISCRETE_RESOLUTION_MM * upper_step,
            rtol=0.0,
            atol=1.0e-12,
        ):
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
        resource = files("lumo").joinpath(
            "assets",
            "objects",
            "urdf",
            raw_name,
        )
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
    if len(targets) < 2:
        raise ValueError("force_targets_n must contain at least two targets")
    if any(not isfinite(target) or target <= 0.0 for target in targets):
        raise ValueError("force targets must be finite and positive")
    if any(current <= previous for previous, current in zip(targets, targets[1:])):
        raise ValueError("force_targets_n must be strictly increasing")
    return targets


def _discrete_design_space(
    parameter_bounds: Mapping[str, tuple[float, float]],
    parameters: FingertipParameters,
) -> DesignSpace:
    parameters = replace(
        parameters,
        geometry=replace(
            parameters.geometry,
            flat_pad_width_mm=_DISCRETE_FIXED_FLAT_PAD_WIDTH_MM,
            void_height_mm=0.0,
        ),
    )
    bounds = DesignParameterBounds(
        parameters=parameters,
        geometry={
            name.removeprefix("geometry."): ParameterBound(*bounds)
            for name, bounds in parameter_bounds.items()
        },
    )
    return DesignSpace(
        parameter_bounds=bounds,
        linear_constraints=(
            LinearConstraint(
                coefficients={
                    "geometry.flat_pad_height_mm": 1.0,
                    "geometry.semiellipse_height_mm": 1.0,
                },
                upper=_MAX_PAD_DEPTH_MM,
            ),
        ),
        minimum_silicone_thickness_mm=5.0,
    )


def _campaign_definition(
    name: str,
    *,
    parameter_bounds_mm: Mapping[str, tuple[float, float]] | None = None,
    indenter_urdfs: Iterable[str] = _DEFAULT_INDENTER_URDFS,
    sphere_diameters_mm: Iterable[float] = _DEFAULT_SPHERE_DIAMETERS_MM,
    force_targets_n: Iterable[float] = _DEFAULT_FORCE_TARGETS_N,
    initial_clearance_m: float = _DEFAULT_INITIAL_CLEARANCE_M,
    contact_y_mm: Iterable[float] = _DEFAULT_CONTACT_Y_MM,
    viscoelastic_preset: str = _DEFAULT_VISCOELASTIC_PRESET,
    optical_preset: str = _DEFAULT_OPTICAL_PRESET,
) -> CampaignDefinition:
    indenters = _indenter_definitions(indenter_urdfs)
    sphere_diameters = tuple(float(value) for value in sphere_diameters_mm)
    if len(sphere_diameters) != len(indenters) or any(
        not isfinite(value) or value <= 0.0 for value in sphere_diameters
    ):
        raise ValueError(
            "sphere_diameters_mm must contain one positive value per indenter"
        )
    if len(set(sphere_diameters)) != len(sphere_diameters):
        raise ValueError("sphere_diameters_mm must be unique")
    targets = _force_targets(force_targets_n)
    initial_clearance_m = float(initial_clearance_m)
    contact_locations_y_mm = tuple(float(value) for value in contact_y_mm)
    if not isfinite(initial_clearance_m) or initial_clearance_m < 0.0:
        raise ValueError("initial_clearance_m must be finite and nonnegative")
    if not contact_locations_y_mm or any(
        not isfinite(value) or not -27.5 <= value <= 27.5
        for value in contact_locations_y_mm
    ):
        raise ValueError(
            "contact_y_mm must contain finite locations inside [-27.5, 27.5] mm"
        )
    if len(set(contact_locations_y_mm)) != len(contact_locations_y_mm):
        raise ValueError("contact_y_mm must be unique")
    if (
        not isinstance(viscoelastic_preset, str)
        or viscoelastic_preset not in VISCOELASTIC_PRESETS
    ):
        raise ValueError(
            f"viscoelastic_preset must be one of {sorted(VISCOELASTIC_PRESETS)!r}"
        )
    if not isinstance(optical_preset, str) or optical_preset not in OPTICAL_PRESETS:
        raise ValueError(f"optical_preset must be one of {sorted(OPTICAL_PRESETS)!r}")
    fingertip_parameters = FingertipParameters(
        viscoelastic=VISCOELASTIC_PRESETS[viscoelastic_preset],
        optical=OPTICAL_PRESETS[optical_preset],
    )
    if name == "continuous":
        raise ValueError(
            "the continuous J_intensity/J_spatial campaign is historical; "
            "use discrete-05mm for the full-finger J_contact/J_obs campaign"
        )
    if name == "discrete-05mm":
        parameter_bounds, step_to_physical = _discrete_parameter_bounds(
            parameter_bounds_mm
        )
        space = _discrete_design_space(parameter_bounds, fingertip_parameters)
        seed_steps = tuple(
            (source_design, steps)
            for source_design, steps in _DISCRETE_SEED_STEPS
            if all(
                lower <= step <= upper
                for step, (_, _, lower, upper) in zip(
                    steps,
                    step_to_physical,
                    strict=True,
                )
            )
            and steps[0] + steps[1] <= _DISCRETE_MAX_PAD_DEPTH_STEPS
            and space.is_feasible(
                {
                    physical_name: _DISCRETE_RESOLUTION_MM * step
                    for step, (_, physical_name, _, _) in zip(
                        steps,
                        step_to_physical,
                        strict=True,
                    )
                }
            )
        )
        return CampaignDefinition(
            name=name,
            experiment_name="lumo_full_finger_objectives_discrete_05mm",
            default_output_directory=_DISCRETE_OUTPUT_DIRECTORY,
            space=space,
            physical_parameter_names=space.variable_names,
            ax_parameters=tuple(
                RangeParameterConfig(
                    name=step_name,
                    bounds=(lower, upper),
                    parameter_type="int",
                )
                for step_name, _, lower, upper in step_to_physical
            ),
            ax_parameter_constraints=(
                "flat_pad_height_step + semiellipse_height_step "
                f"<= {_DISCRETE_MAX_PAD_DEPTH_STEPS}",
            ),
            initialization_budget=13,
            resolution_mm=_DISCRETE_RESOLUTION_MM,
            fixed_geometry=(
                (
                    "geometry.flat_pad_width_mm",
                    _DISCRETE_FIXED_FLAT_PAD_WIDTH_MM,
                ),
                ("geometry.void_height_mm", 0.0),
            ),
            discrete_step_to_physical=step_to_physical,
            seed_steps=seed_steps,
            indenters=indenters,
            sphere_diameters_mm=sphere_diameters,
            force_targets_n=targets,
            initial_clearance_m=initial_clearance_m,
            contact_y_mm=contact_locations_y_mm,
            viscoelastic_preset=viscoelastic_preset,
            optical_preset=optical_preset,
        )
    raise ValueError(f"unknown campaign {name!r}")


def _read_warm_start(
    campaign: CampaignDefinition,
) -> list[dict[str, object]]:
    path = campaign.warm_start_path
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as input_file:
        for raw_row in csv.DictReader(input_file):
            if raw_row["status"] != "PASS":
                continue
            parameters = {
                name: float(raw_row[name]) for name in campaign.physical_parameter_names
            }
            objectives = {name: float(raw_row[name]) for name in _OBJECTIVE_NAMES}
            if not campaign.space.is_feasible(parameters):
                raise ValueError(
                    f"warm-start design {raw_row['design']!r} is analytically invalid"
                )
            if not all(isfinite(value) for value in objectives.values()):
                raise ValueError(
                    f"warm-start design {raw_row['design']!r} has invalid objectives"
                )
            rows.append(
                {
                    "design": raw_row["design"],
                    "parameters": parameters,
                    "objectives": objectives,
                    "per_indenter": {
                        **{
                            f"J_intensity_{indenter_name}": float(
                                raw_row[intensity_field]
                            )
                            for indenter_name, intensity_field, _ in (
                                _CONTINUOUS_WARM_START_RESULT_FIELDS
                            )
                        },
                        **{
                            f"J_spatial_{indenter_name}": float(raw_row[spatial_field])
                            for indenter_name, _, spatial_field in (
                                _CONTINUOUS_WARM_START_RESULT_FIELDS
                            )
                        },
                    },
                    "worst_intensity_indenter": (
                        f"sphere_{float(raw_row['worst_intensity_diameter_mm']):g}mm"
                    ),
                    "worst_intensity_force_pair_n": raw_row[
                        "worst_intensity_force_pair_n"
                    ],
                    "worst_spatial_indenter": (
                        f"sphere_{float(raw_row['worst_spatial_diameter_mm']):g}mm"
                    ),
                    "worst_spatial_force_pair_n": raw_row["worst_spatial_force_pair_n"],
                    "runtime_s": float(raw_row["runtime_s"]),
                }
            )

    if len(rows) != campaign.warm_start_count:
        raise ValueError(
            f"expected {campaign.warm_start_count} completed warm-start designs, "
            f"found {len(rows)}"
        )
    if len({str(row["design"]) for row in rows}) != len(rows):
        raise ValueError("warm-start design names must be unique")
    return rows


def _new_client(campaign: CampaignDefinition) -> Client:
    client = Client(random_seed=_RANDOM_SEED)
    client.configure_experiment(
        name=campaign.experiment_name,
        description=(
            "Full-finger LUMO contact and observation optimization over "
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
        initialization_random_seed=_RANDOM_SEED,
        initialize_with_center=False,
        use_existing_trials_for_initialization=True,
    )
    return client


def _decode_ax_parameters(
    campaign: CampaignDefinition,
    raw_parameters: dict[str, object],
) -> dict[str, float]:
    if set(raw_parameters) != set(campaign.ax_parameter_names):
        raise ValueError("Ax candidate does not match the campaign parameters")
    if not campaign.is_discrete:
        return {
            name: float(raw_parameters[name])
            for name in campaign.physical_parameter_names
        }

    parameters = {}
    for step_name, physical_name, lower, upper in campaign.discrete_step_to_physical:
        raw_step = raw_parameters[step_name]
        step = int(raw_step)
        if isinstance(raw_step, bool) or float(raw_step) != step:
            raise ValueError(f"{step_name} must be an exact integer")
        if not lower <= step <= upper:
            raise ValueError(f"{step_name} lies outside its integer bounds")
        parameters[physical_name] = campaign.resolution_mm * step
    return parameters


def _encode_ax_parameters(
    campaign: CampaignDefinition,
    parameters: dict[str, float],
) -> dict[str, float | int]:
    if not campaign.is_discrete:
        return dict(parameters)
    encoded = {}
    for step_name, physical_name, lower, upper in campaign.discrete_step_to_physical:
        scaled = parameters[physical_name] / campaign.resolution_mm
        step = round(scaled)
        if not np.isclose(scaled, step, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{physical_name} is not on the 0.5 mm grid")
        if not lower <= step <= upper:
            raise ValueError(f"{physical_name} lies outside its bounds")
        encoded[step_name] = step
    return encoded


def _validate_campaign_parameters(
    campaign: CampaignDefinition,
    raw_parameters: dict[str, object],
    parameters: dict[str, float],
) -> None:
    if set(parameters) != set(campaign.physical_parameter_names):
        raise ValueError("decoded candidate does not match physical parameters")
    if _encode_ax_parameters(campaign, parameters) != {
        name: raw_parameters[name] for name in campaign.ax_parameter_names
    }:
        raise ValueError("Ax candidate does not round-trip through physical units")
    if campaign.is_discrete:
        fixed_width = (
            campaign.space.parameter_bounds.parameters.geometry.flat_pad_width_mm
        )
        if fixed_width != _DISCRETE_FIXED_FLAT_PAD_WIDTH_MM:
            raise ValueError("discrete flat_pad_width_mm must remain fixed at 30 mm")
        if (
            int(raw_parameters["flat_pad_height_step"])
            + int(raw_parameters["semiellipse_height_step"])
            > _DISCRETE_MAX_PAD_DEPTH_STEPS
        ):
            raise ValueError("Ax candidate violates the full-height constraint")


def _verify_discrete_search_space(campaign: CampaignDefinition) -> None:
    if not campaign.is_discrete:
        return
    client = _new_client(campaign)
    print(f"Ax search space:\n{client._experiment.search_space}", flush=True)
    stem_height_steps = set()
    for probe_index in range(12):
        generated = client.get_next_trials(max_trials=1)
        if len(generated) != 1:
            raise RuntimeError("Ax search-space probe did not return one candidate")
        trial_index, raw_parameters = generated.popitem()
        parameters = _decode_ax_parameters(campaign, raw_parameters)
        _validate_campaign_parameters(campaign, raw_parameters, parameters)
        if not campaign.space.is_feasible(parameters):
            client.mark_trial_abandoned(trial_index)
            continue
        stem_height_steps.add(int(raw_parameters["stem_height_step"]))
        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "J_contact": float(probe_index),
                "J_obs": float(-probe_index),
            },
        )
    if len(stem_height_steps) < 2:
        raise RuntimeError("Ax probes did not vary stem_height_step")
    with TemporaryDirectory(prefix="lumo_ax_preflight_") as temporary_directory:
        state_path = Path(temporary_directory) / _AX_STATE_FILENAME
        _atomic_save_ax(client, state_path)
        restored = Client.load_from_json_file(filepath=str(state_path))
        if _ax_statuses(restored) != _ax_statuses(client):
            raise RuntimeError("Ax save/load preflight changed trial statuses")
    print(
        "discrete search-space PASS: 0.5 mm grid, fixed width=30 mm, "
        "full-height pad steps<=40, Ax save/load, stem_height_steps="
        f"{sorted(stem_height_steps)}",
        flush=True,
    )


def _fieldnames(campaign: CampaignDefinition) -> list[str]:
    return [
        "ax_trial_index",
        "source",
        "design",
        "generation_node",
        "status",
        "analytically_valid",
        *campaign.parameter_columns,
        *campaign.per_indenter_fields,
        *_OBJECTIVE_NAMES,
        "limiting_contact_scenario",
        "limiting_obs_sphere_diameter_mm",
        "limiting_obs_force_n",
        "limiting_obs_contact_y_pair_mm",
        "d_onset_diagnostic",
        "max_outside_roi_power_fraction",
        "runtime_s",
        "raw_result_path",
        "failure",
        "is_pareto",
    ]


def _empty_result_fields(campaign: CampaignDefinition) -> dict[str, object]:
    return {
        **{name: "" for name in campaign.per_indenter_fields},
        **{name: "" for name in _OBJECTIVE_NAMES},
        "limiting_contact_scenario": "",
        "limiting_obs_sphere_diameter_mm": "",
        "limiting_obs_force_n": "",
        "limiting_obs_contact_y_pair_mm": "",
        "d_onset_diagnostic": "",
        "max_outside_roi_power_fraction": "",
        "runtime_s": "",
        "raw_result_path": "",
        "failure": "",
        "is_pareto": False,
    }


def _read_trials(
    path: Path,
    campaign: CampaignDefinition,
) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows: list[dict[str, object]] = []
        for raw_row in csv.DictReader(input_file):
            row: dict[str, object] = dict(raw_row)
            row["ax_trial_index"] = int(raw_row["ax_trial_index"])
            row["analytically_valid"] = raw_row["analytically_valid"] == "True"
            row["is_pareto"] = raw_row["is_pareto"] == "True"
            if campaign.is_discrete:
                for name in campaign.ax_parameter_names:
                    value = raw_row.get(name, "")
                    row[name] = int(float(value)) if value else ""
            numeric_fields = (
                *campaign.physical_parameter_names,
                *campaign.per_indenter_fields,
                *_OBJECTIVE_NAMES,
                "limiting_obs_sphere_diameter_mm",
                "limiting_obs_force_n",
                "d_onset_diagnostic",
                "max_outside_roi_power_fraction",
                "runtime_s",
            )
            for name in numeric_fields:
                value = raw_row.get(name, "")
                row[name] = float(value) if value else ""
            rows.append(row)
    return rows


def _update_pareto_status(rows: list[dict[str, object]]) -> None:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    for row in rows:
        row["is_pareto"] = False
    for candidate in completed:
        candidate_objectives = np.array(
            [candidate[name] for name in _OBJECTIVE_NAMES],
            dtype=np.float64,
        )
        candidate["is_pareto"] = not any(
            np.all(
                np.array([other[name] for name in _OBJECTIVE_NAMES])
                >= candidate_objectives
            )
            and np.any(
                np.array([other[name] for name in _OBJECTIVE_NAMES])
                > candidate_objectives
            )
            for other in completed
            if other is not candidate
        )


def _flush_file(output_file: object) -> None:
    output_file.flush()
    os.fsync(output_file.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_save_ax(client: Client, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    client.save_to_json_file(filepath=str(temporary))
    with temporary.open("rb") as input_file:
        os.fsync(input_file.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _write_tables(
    output_directory: Path,
    rows: list[dict[str, object]],
    campaign: CampaignDefinition,
) -> None:
    _update_pareto_status(rows)
    _atomic_write_csv(
        output_directory / _TRIALS_FILENAME,
        rows,
        _fieldnames(campaign),
    )
    pareto_rows = [
        row for row in rows if row["status"] == "COMPLETED" and row["is_pareto"]
    ]
    _atomic_write_csv(
        output_directory / _PARETO_FILENAME,
        pareto_rows,
        _fieldnames(campaign),
    )


def _persist_ax_and_tables(
    client: Client,
    rows: list[dict[str, object]],
    output_directory: Path,
    campaign: CampaignDefinition,
) -> None:
    _atomic_save_ax(client, output_directory / _AX_STATE_FILENAME)
    _write_tables(output_directory, rows, campaign)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    source_root = repository_root / "lumo"
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".cu", ".urdf"}:
            continue
        relative_path = path.relative_to(repository_root).as_posix()
        if relative_path == "lumo/optimization/ax_bo.py":
            continue
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _git_output(repository_root: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_config(campaign: CampaignDefinition) -> dict[str, object]:
    from . import evaluator
    from lumo.mesh import (
        LED_RECESS_DEPTH_MM,
        LED_RECESS_WIDTH_MM,
        MAIN_Y_BOUNDS_MM,
        led_centers_y_mm,
    )
    from lumo.ray_tracing import LONGITUDINAL_SIDE_BIN_COUNT

    repository_root = Path(__file__).resolve().parents[2]
    parameter_bounds = {
        f"geometry.{name}": [bound.lower, bound.upper]
        for name, bound in campaign.space.parameter_bounds.geometry.items()
    }
    linear_constraints = [
        {
            "coefficients": dict(constraint.coefficients),
            "lower": constraint.lower,
            "upper": constraint.upper,
        }
        for constraint in campaign.space.linear_constraints
    ]
    warm_start_sha256 = (
        _sha256_file(campaign.warm_start_path)
        if campaign.warm_start_path is not None and campaign.warm_start_path.is_file()
        else None
    )
    design_space_contract: dict[str, object] = {
        "bounds": parameter_bounds,
        "linear_constraints": linear_constraints,
        "minimum_silicone_thickness_mm": (campaign.space.minimum_silicone_thickness_mm),
        "full_fingertip_height_max_mm": MAX_FINGERTIP_HEIGHT_MM,
        "full_height_relation_mm": (
            "link_thickness + flat_pad_height + semiellipse_height"
        ),
        "fixed_link_thickness_mm": _FIXED_UPPER_HEIGHT_MM,
    }
    if campaign.is_discrete:
        design_space_contract = {
            "representation": "integer_half_millimeter_steps",
            "resolution_mm": campaign.resolution_mm,
            "fixed": dict(campaign.fixed_geometry),
            "integer_step_bounds": {
                name: [int(parameter.bounds[0]), int(parameter.bounds[1])]
                for name, parameter in zip(
                    campaign.ax_parameter_names,
                    campaign.ax_parameters,
                    strict=True,
                )
            },
            "decoded_physical_bounds_mm": parameter_bounds,
            "ax_linear_constraints": list(campaign.ax_parameter_constraints),
            "physical_linear_constraints": linear_constraints,
            "minimum_silicone_thickness_mm": (
                campaign.space.minimum_silicone_thickness_mm
            ),
            "full_fingertip_height_max_mm": MAX_FINGERTIP_HEIGHT_MM,
            "full_height_relation_mm": (
                "link_thickness + flat_pad_height + semiellipse_height"
            ),
            "fixed_link_thickness_mm": _FIXED_UPPER_HEIGHT_MM,
            "new_observation_seed_steps": [
                {
                    "source_design": source_design,
                    **dict(
                        zip(
                            campaign.ax_parameter_names,
                            steps,
                            strict=True,
                        )
                    ),
                }
                for source_design, steps in campaign.seed_steps
            ],
        }
    return {
        "schema_version": _RUN_CONFIG_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_revision": _git_output(repository_root, ["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_output(repository_root, ["status", "--porcelain"])),
            "scientific_source_sha256": _scientific_source_sha256(repository_root),
            "optimizer_source_sha256": _sha256_file(Path(__file__).resolve()),
            "warm_start_sha256": warm_start_sha256,
            "versions": {
                "ax-platform": _package_version("ax-platform"),
                "botorch": _package_version("botorch"),
                "newton": _package_version("newton"),
                "warp-lang": _package_version("warp-lang"),
                "numpy": _package_version("numpy"),
            },
        },
        "scientific_contract": {
            "viscoelastic_preset": campaign.viscoelastic_preset,
            "optical_preset": campaign.optical_preset,
            "fingertip_parameters": asdict(campaign.space.parameter_bounds.parameters),
            "mechanics": {
                "loading_mode": "first_crossing",
                "capture_rule": "first reaction-force sample >= threshold",
                "backend": (
                    f"gpu_first_crossing_graph_parallel_"
                    f"{evaluator._PRODUCTION_PARALLEL_WORLD_COUNT}"
                ),
                "parallel_world_count": (
                    evaluator._PRODUCTION_PARALLEL_WORLD_COUNT
                ),
                "direct_reference_available": True,
                "sim_frequency_hz": evaluator._SIM_FREQUENCY_HZ,
                "vbd_iterations": evaluator._VBD_ITERATIONS,
                "approach_speed_m_s": _APPROACH_SPEED_M_S,
                "displacement_m_tick": (
                    _APPROACH_SPEED_M_S / evaluator._SIM_FREQUENCY_HZ
                ),
                "force_targets_n": list(campaign.force_targets_n),
                "snapshot_dwell_s": 0.0,
                "force_feedback": False,
                "max_sim_time_s": _MAX_SIM_TIME_S,
                "element_size_mm": evaluator._ELEMENT_SIZE_MM,
                "soft_contact_margin_m": evaluator._SOFT_CONTACT_MARGIN_M,
                "carrier_contact_stiffness_n_m": (
                    evaluator._CARRIER_CONTACT_STIFFNESS_N_M
                ),
                "indenter_contact_stiffness_n_m": (evaluator._CONTACT_STIFFNESS_N_M),
                "indenter_contact_damping_n_s_m": (evaluator._CONTACT_DAMPING_N_S_M),
            },
            "scenarios": {
                "indenter_urdfs": [filename for filename, _ in campaign.indenters],
                "indenter_names": list(campaign.indenter_names),
                "sphere_diameters_mm": list(campaign.sphere_diameters_mm),
                "initial_clearance_m": campaign.initial_clearance_m,
                "contact_x_mm": 0.0,
                "contact_y_mm": list(campaign.contact_y_mm),
            },
            "optics": {
                "sample_side_count": evaluator._SAMPLE_SIDE_COUNT,
                "ray_count": evaluator._SAMPLE_SIDE_COUNT**2,
                "max_bounces": evaluator._MAX_BOUNCES,
                "deterministic_seed": evaluator._RNG_SEED,
                "source_seed": evaluator._SOURCE_RNG_SEED,
                "source_model": "uniform_finite_package_window",
                "source_window_mm": [
                    campaign.space.parameter_bounds.parameters.led.emitting_window_x_mm,
                    campaign.space.parameter_bounds.parameters.led.emitting_window_y_mm,
                ],
                "carrier_albedo": evaluator._CARRIER_ALBEDO,
                "source_medium": "resolved per geometry from LED air-gap boundary",
                "led_centers_y_mm": list(led_centers_y_mm()),
                "led_count": len(led_centers_y_mm()),
                "led_recess_width_mm": LED_RECESS_WIDTH_MM,
                "led_recess_depth_mm": LED_RECESS_DEPTH_MM,
                "observation_view_direction": "+X",
                "longitudinal_coordinate": "Y",
                "spatial_roi_y_mm": list(MAIN_Y_BOUNDS_MM),
                "spatial_bin_count": LONGITUDINAL_SIDE_BIN_COUNT,
                "spatial_bin_width_mm": (
                    (MAIN_Y_BOUNDS_MM[1] - MAIN_Y_BOUNDS_MM[0])
                    / LONGITUDINAL_SIDE_BIN_COUNT
                ),
                "simultaneous_emitted_power": float(len(led_centers_y_mm())),
            },
            "design_space": {
                **design_space_contract,
            },
            "objectives": {
                "names": list(_OBJECTIVE_NAMES),
                "directions": ["maximize", "maximize"],
                "definition": _OBJECTIVE_DEFINITION,
                "J_contact": (
                    "min over diameter/location scenarios of "
                    "cuberoot(q_form*q_stable*q_stiff)"
                ),
                "J_obs": (
                    "min over diameter, force threshold, and distinct contact-Y "
                    "pairs of "
                    "L2((y-y0)/P_emit); d_onset is diagnostic only"
                ),
            },
            "ax_random_seed": _RANDOM_SEED,
        },
    }


def _validate_run_config(
    stored: dict[str, object],
    current: dict[str, object],
) -> None:
    if stored.get("schema_version") != _RUN_CONFIG_SCHEMA:
        raise RuntimeError("stored run_config.json has an unsupported schema")
    if stored.get("scientific_contract") != current.get("scientific_contract"):
        raise RuntimeError(
            "current scientific settings differ from run_config.json; "
            "refusing to mix incompatible evaluations"
        )
    stored_provenance = stored.get("provenance")
    current_provenance = current.get("provenance")
    if not isinstance(stored_provenance, dict) or not isinstance(
        current_provenance, dict
    ):
        raise RuntimeError("run_config.json has invalid provenance")
    if stored_provenance.get("scientific_source_sha256") != current_provenance.get(
        "scientific_source_sha256"
    ):
        raise RuntimeError(
            "LUMO scientific source differs from the saved run contract; "
            "resume with the original source snapshot"
        )
    if stored_provenance.get("optimizer_source_sha256") != current_provenance.get(
        "optimizer_source_sha256"
    ):
        raise RuntimeError(
            "Ax campaign code differs from the saved run contract; "
            "resume with the original optimizer source snapshot"
        )
    if stored_provenance.get("versions") != current_provenance.get("versions"):
        raise RuntimeError(
            "dependency versions differ from the saved run contract; "
            "resume with the original environment"
        )


def _verify_warm_start_in_ax(
    client: Client,
    warm_start: list[dict[str, object]],
    campaign: CampaignDefinition,
) -> None:
    summary = client.summarize()
    warm_rows = summary.iloc[: campaign.warm_start_count]
    if (
        len(summary) < campaign.warm_start_count
        or len(warm_rows) != campaign.warm_start_count
    ):
        raise RuntimeError("Ax does not contain all warm-start observations")
    for (_, ax_row), expected in zip(warm_rows.iterrows(), warm_start, strict=True):
        if str(ax_row["trial_status"]).upper().split(".")[-1] != "COMPLETED":
            raise RuntimeError("an Ax warm-start trial is not completed")
        expected_parameters = expected["parameters"]
        expected_objectives = expected["objectives"]
        expected_ax_parameters = _encode_ax_parameters(
            campaign,
            expected_parameters,
        )
        for name in campaign.ax_parameter_names:
            if not np.isclose(
                float(ax_row[name]),
                float(expected_ax_parameters[name]),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(f"Ax warm-start parameter mismatch for {name}")
        for name in _OBJECTIVE_NAMES:
            if not np.isclose(
                float(ax_row[name]),
                float(expected_objectives[name]),
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError(f"Ax warm-start objective mismatch for {name}")


def _evaluate_candidate(
    campaign: CampaignDefinition,
    parameters: dict[str, float],
):
    from lumo.fingertip import Fingertip

    from .evaluator import (
        _PRODUCTION_PARALLEL_WORLD_COUNT,
        evaluate_full_finger,
    )
    from lumo.simulation import FIRST_CROSSING_LOADING

    fingertip = Fingertip(campaign.space.to_parameters(parameters))
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    with ExitStack() as resources:
        indenter_paths = tuple(
            resources.enter_context(as_file(resource_root.joinpath(filename)))
            for filename, _ in campaign.indenters
        )
        return evaluate_full_finger(
            fingertip,
            indenter_paths,
            campaign.sphere_diameters_mm,
            campaign.contact_y_mm,
            force_targets_n=campaign.force_targets_n,
            settle_duration_s=0.0,
            initial_clearance_m=campaign.initial_clearance_m,
            approach_speed_m_s=_APPROACH_SPEED_M_S,
            max_sim_time_s=_MAX_SIM_TIME_S,
            loading_mode=FIRST_CROSSING_LOADING,
            use_cuda_graph=True,
            parallel_world_count=_PRODUCTION_PARALLEL_WORLD_COUNT,
        )


def _objective_details(evaluation: object) -> dict[str, object]:
    from .objective import compute_objectives_from_raw

    contact, observation = compute_objectives_from_raw(vars(evaluation))
    return {
        "J_contact": contact.J_contact,
        "J_obs": observation.J_obs,
        "contact": contact,
        "observation": observation,
        "emitted_power": float(
            evaluation.no_contact_energy[
                :,
                tuple(evaluation.energy_fields).index("emitted_power"),
            ].sum()
        ),
        "max_outside_roi_power_fraction": float(
            max(
                evaluation.no_contact_outside_roi_power_fraction,
                np.max(evaluation.outside_roi_power_fraction),
            )
        ),
    }


def _trial_result_path(output_directory: Path, trial_index: int) -> Path:
    return output_directory / _TRIAL_RESULT_DIRECTORY / f"trial_{trial_index:04d}.npz"


def _save_trial_result(
    path: Path,
    *,
    campaign: CampaignDefinition,
    evaluation: object,
    details: dict[str, object],
    parameters: dict[str, float],
    runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            reference_vertices_m=np.asarray(evaluation.reference_vertices_m),
            tet_indices=np.asarray(evaluation.tet_indices),
            surface_triangles=np.asarray(evaluation.surface_triangles),
            bonded_vertex_indices=np.asarray(evaluation.bonded_vertex_indices),
            led_centers_m=np.asarray(evaluation.led_centers_m),
            no_contact_response=np.asarray(evaluation.no_contact_response),
            no_contact_energy=np.asarray(evaluation.no_contact_energy),
            no_contact_inside_roi_power=np.asarray(
                evaluation.no_contact_inside_roi_power
            ),
            no_contact_outside_roi_power=np.asarray(
                evaluation.no_contact_outside_roi_power
            ),
            no_contact_visible_side_power=np.asarray(
                evaluation.no_contact_visible_side_power
            ),
            no_contact_outside_roi_power_fraction=np.asarray(
                evaluation.no_contact_outside_roi_power_fraction
            ),
            response_matrix=np.asarray(evaluation.response_matrix),
            energy_matrix=np.asarray(evaluation.energy_matrix),
            energy_fields=np.asarray(evaluation.energy_fields),
            inside_roi_power=np.asarray(evaluation.inside_roi_power),
            outside_roi_power=np.asarray(evaluation.outside_roi_power),
            visible_side_power=np.asarray(evaluation.visible_side_power),
            outside_roi_power_fraction=np.asarray(
                evaluation.outside_roi_power_fraction
            ),
            actual_forces_n=np.asarray(evaluation.actual_forces_n),
            indentations_m=np.asarray(evaluation.indentations_m),
            checkpoint_steps=np.asarray(evaluation.checkpoint_steps),
            checkpoint_times_s=np.asarray(evaluation.checkpoint_times_s),
            maximum_particle_speeds_m_s=np.asarray(
                evaluation.maximum_particle_speeds_m_s
            ),
            mean_particle_speeds_m_s=np.asarray(
                evaluation.mean_particle_speeds_m_s
            ),
            rms_particle_speeds_m_s=np.asarray(
                evaluation.rms_particle_speeds_m_s
            ),
            particle_speed_p95_m_s=np.asarray(
                evaluation.particle_speed_p95_m_s
            ),
            kinetic_energy_j=np.asarray(evaluation.kinetic_energy_j),
            force_references_n=np.asarray(evaluation.force_references_n),
            reaction_force_rates_n_s=np.asarray(
                evaluation.reaction_force_rates_n_s
            ),
            indentation_rates_m_s=np.asarray(evaluation.indentation_rates_m_s),
            servo_errors_n=np.asarray(evaluation.servo_errors_n),
            settle_window_force_drifts_n=np.asarray(
                evaluation.settle_window_force_drifts_n
            ),
            settle_window_indentation_drifts_m=np.asarray(
                evaluation.settle_window_indentation_drifts_m
            ),
            indenter_contact_counts=np.asarray(evaluation.indenter_contact_counts),
            total_contact_counts=np.asarray(evaluation.total_contact_counts),
            contact_buffer_overflow=np.asarray(evaluation.contact_buffer_overflow),
            minimum_det_f=np.asarray(evaluation.minimum_det_f),
            inverted_tet_counts=np.asarray(evaluation.inverted_tet_counts),
            contact_centroids_W_m=np.asarray(evaluation.contact_centroids_W_m),
            contact_record_offsets=np.asarray(evaluation.contact_record_offsets),
            contact_particle_indices=np.asarray(
                evaluation.contact_particle_indices
            ),
            contact_barycentric=np.asarray(evaluation.contact_barycentric),
            contact_positions_W_m=np.asarray(evaluation.contact_positions_W_m),
            contact_normals_W=np.asarray(evaluation.contact_normals_W),
            contact_body_positions=np.asarray(evaluation.contact_body_positions),
            silicone_vertices_m=np.asarray(evaluation.silicone_vertices_m),
            scenario_runtime_s=np.asarray(evaluation.scenario_runtime_s),
            checkpoint_optics_runtime_s=np.asarray(
                evaluation.checkpoint_optics_runtime_s
            ),
            no_contact_optics_runtime_s=np.asarray(
                evaluation.no_contact_optics_runtime_s
            ),
            mechanics_backend=np.asarray(evaluation.mechanics_backend),
            graph_replay_counts=np.asarray(evaluation.graph_replay_counts),
            force_servo_host_intervention_counts=np.asarray(
                evaluation.force_servo_host_intervention_counts
            ),
            force_servo_host_sync_counts=np.asarray(
                evaluation.force_servo_host_sync_counts
            ),
            force_servo_average_ticks_per_host_intervention=np.asarray(
                evaluation.force_servo_average_ticks_per_host_intervention
            ),
            loading_mode=np.asarray(evaluation.loading_mode),
            force_ramp_rate_n_s=np.asarray(
                np.nan
                if evaluation.force_ramp_rate_n_s is None
                else evaluation.force_ramp_rate_n_s
            ),
            scenario_names=np.asarray(evaluation.scenario_names),
            sphere_diameters_mm=np.asarray(evaluation.sphere_diameters_mm),
            contact_y_mm=np.asarray(evaluation.contact_y_mm),
            force_targets_n=np.asarray(evaluation.force_targets_n),
            J_contact=np.asarray(details["J_contact"]),
            limiting_contact_scenario=np.asarray(
                details["contact"].limiting_scenario
            ),
            q_form=np.asarray(details["contact"].q_form),
            q_stable=np.asarray(details["contact"].q_stable),
            q_stiff=np.asarray(details["contact"].q_stiff),
            q_contact=np.asarray(details["contact"].q_contact),
            q_normal_diagnostic=np.asarray(details["contact"].q_normal),
            patch_area_5_m2=np.asarray(details["contact"].patch_area_5_m2),
            k_early_n_m=np.asarray(details["contact"].k_early_n_m),
            k_late_n_m=np.asarray(details["contact"].k_late_n_m),
            J_obs=np.asarray(details["J_obs"]),
            limiting_obs_sphere_diameter_mm=np.asarray(
                details["observation"].limiting_sphere_diameter_mm
            ),
            limiting_obs_force_n=np.asarray(
                details["observation"].limiting_force_n
            ),
            limiting_obs_contact_y_pair_mm=np.asarray(
                details["observation"].limiting_contact_y_pair_mm
            ),
            d_onset_diagnostic=np.asarray(details["observation"].d_onset),
            normalized_observation=np.asarray(
                details["observation"].normalized_response
            ),
            observation_sphere_diameters_mm=np.asarray(
                details["observation"].sphere_diameters_mm
            ),
            observation_contact_y_mm=np.asarray(
                details["observation"].contact_y_mm
            ),
            same_force_location_separations=np.asarray(
                details["observation"].location_separations
            ),
            evaluation_runtime_s=np.asarray(runtime_s),
            parameter_names=np.asarray(campaign.physical_parameter_names),
            parameter_values=np.asarray(
                [parameters[name] for name in campaign.physical_parameter_names]
            ),
        )
        _flush_file(output_file)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _apply_result_to_row(
    row: dict[str, object],
    details: dict[str, object],
    *,
    campaign: CampaignDefinition,
    runtime_s: float,
    raw_result_path: str,
) -> None:
    contact = details["contact"]
    observation = details["observation"]
    row.update(
        J_contact=float(details["J_contact"]),
        J_obs=float(details["J_obs"]),
        limiting_contact_scenario=contact.limiting_scenario,
        limiting_obs_sphere_diameter_mm=(
            observation.limiting_sphere_diameter_mm
        ),
        limiting_obs_force_n=observation.limiting_force_n,
        limiting_obs_contact_y_pair_mm=(
            f"{observation.limiting_contact_y_pair_mm[0]:g},"
            f"{observation.limiting_contact_y_pair_mm[1]:g}"
        ),
        d_onset_diagnostic=observation.d_onset,
        max_outside_roi_power_fraction=details[
            "max_outside_roi_power_fraction"
        ],
        runtime_s=runtime_s,
        raw_result_path=raw_result_path,
        failure="",
    )


def _ax_statuses(client: Client) -> dict[int, str]:
    summary = client.summarize()
    return {
        int(row["trial_index"]): str(row["trial_status"]).upper().split(".")[-1]
        for _, row in summary.iterrows()
    }


def _ax_parameters_from_summary_row(
    row: object,
    campaign: CampaignDefinition,
) -> dict[str, object]:
    return {name: row[name] for name in campaign.ax_parameter_names}


def _running_row(
    trial_index: int,
    ax_parameters: dict[str, object],
    parameters: dict[str, float],
    generation_node: str,
    campaign: CampaignDefinition,
) -> dict[str, object]:
    return {
        "ax_trial_index": trial_index,
        "source": "bo",
        "design": f"bo_{trial_index:04d}",
        "generation_node": generation_node,
        "status": "RUNNING",
        "analytically_valid": False,
        **ax_parameters,
        **parameters,
        **_empty_result_fields(campaign),
    }


def _reconcile_resume(
    client: Client,
    rows: list[dict[str, object]],
    output_directory: Path,
    campaign: CampaignDefinition,
) -> None:
    summary = client.summarize()
    ax_statuses = _ax_statuses(client)
    row_indices = {int(row["ax_trial_index"]) for row in rows}
    ax_changed = False

    for _, summary_row in summary.iterrows():
        trial_index = int(summary_row["trial_index"])
        if trial_index in row_indices:
            continue
        status = ax_statuses[trial_index]
        if status != "RUNNING":
            raise RuntimeError(
                f"Ax trial {trial_index} is absent from trials.csv with status {status}"
            )
        ax_parameters = _ax_parameters_from_summary_row(summary_row, campaign)
        parameters = _decode_ax_parameters(campaign, ax_parameters)
        _validate_campaign_parameters(campaign, ax_parameters, parameters)
        row = _running_row(
            trial_index,
            ax_parameters,
            parameters,
            str(summary_row.get("generation_node", "")),
            campaign,
        )
        row["runtime_s"] = 0.0
        row["failure"] = "interrupted before proposal CSV persistence"
        client.mark_trial_abandoned(trial_index)
        ax_changed = True
        row["status"] = "ABANDONED"
        rows.append(row)

    ax_indices = set(ax_statuses)
    unexpected_rows = {int(row["ax_trial_index"]) for row in rows} - ax_indices
    if unexpected_rows:
        raise RuntimeError(
            f"trials.csv contains trials absent from Ax: {sorted(unexpected_rows)}"
        )

    ax_statuses = _ax_statuses(client)
    for row in rows:
        trial_index = int(row["ax_trial_index"])
        csv_status = str(row["status"])
        ax_status = ax_statuses[trial_index]
        if csv_status == "EVALUATED":
            raw_path = output_directory / str(row["raw_result_path"])
            if not raw_path.is_file():
                raise RuntimeError(
                    f"evaluated trial {trial_index} has no raw-result NPZ"
                )
            if ax_status == "RUNNING":
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={name: float(row[name]) for name in _OBJECTIVE_NAMES},
                )
                ax_changed = True
            elif ax_status != "COMPLETED":
                raise RuntimeError(
                    f"evaluated trial {trial_index} has Ax status {ax_status}"
                )
            row["status"] = "COMPLETED"
        elif csv_status == "RUNNING":
            if ax_status != "RUNNING":
                raise RuntimeError(
                    f"running CSV trial {trial_index} has Ax status {ax_status}"
                )
            client.mark_trial_abandoned(trial_index)
            ax_changed = True
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = "interrupted during morphology evaluation"
        elif csv_status == "FAILED" and ax_status == "ABANDONED":
            continue
        elif csv_status != ax_status:
            raise RuntimeError(
                f"trial {trial_index} status mismatch: CSV={csv_status}, Ax={ax_status}"
            )

    if ax_changed:
        _atomic_save_ax(client, output_directory / _AX_STATE_FILENAME)
    _write_tables(output_directory, rows, campaign)


def _completed_bo_count(rows: list[dict[str, object]]) -> int:
    return sum(row["source"] == "bo" and row["status"] == "COMPLETED" for row in rows)


def _duplicate_terminal(
    rows: list[dict[str, object]],
    parameters: dict[str, float],
    campaign: CampaignDefinition,
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in rows
            if row["status"] in {"COMPLETED", "FAILED"}
            and all(
                np.isclose(
                    parameters[name],
                    float(row[name]),
                    rtol=0.0,
                    atol=1.0e-8,
                )
                for name in campaign.physical_parameter_names
            )
        ),
        None,
    )


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
            continue
        expected = Path(raw_directory) / relative_path
        if not expected.is_file():
            failures.append(f"{variable} does not contain {relative_path}")
    if failures:
        raise RuntimeError("OptiX environment preflight failed: " + "; ".join(failures))


def _is_infrastructure_failure(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    markers = (
        "optix_include_dir",
        "otk_include_dir",
        "cuda driver",
        "cuda error",
        "nvrtc",
        "failed to load optix",
        "no cuda",
        "out of memory",
    )
    return any(marker in message for marker in markers)


def _make_warm_row(
    trial_index: int,
    warm: dict[str, object],
    campaign: CampaignDefinition,
) -> dict[str, object]:
    return {
        "ax_trial_index": trial_index,
        "source": "warm_start",
        "design": warm["design"],
        "generation_node": "",
        "status": "COMPLETED",
        "analytically_valid": True,
        **_encode_ax_parameters(campaign, warm["parameters"]),
        **warm["parameters"],
        **warm["per_indenter"],
        **warm["objectives"],
        "worst_intensity_indenter": warm["worst_intensity_indenter"],
        "worst_intensity_force_pair_n": warm["worst_intensity_force_pair_n"],
        "worst_spatial_indenter": warm["worst_spatial_indenter"],
        "worst_spatial_force_pair_n": warm["worst_spatial_force_pair_n"],
        "runtime_s": warm["runtime_s"],
        "raw_result_path": "",
        "failure": "",
        "is_pareto": False,
    }


def _initialize_campaign(
    output_directory: Path,
    campaign: CampaignDefinition,
) -> tuple[Client, list[dict[str, object]], dict[str, object]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    contents = list(output_directory.iterdir())
    if contents:
        raise FileExistsError(f"fresh campaign output is not empty: {output_directory}")
    config = _run_config(campaign)
    _atomic_write_json(output_directory / _RUN_CONFIG_FILENAME, config)
    warm_start = _read_warm_start(campaign)
    client = _new_client(campaign)
    rows = []
    for warm in warm_start:
        trial_index = client.attach_trial(
            parameters=_encode_ax_parameters(campaign, warm["parameters"]),
            arm_name=str(warm["design"]),
        )
        client.complete_trial(
            trial_index=trial_index,
            raw_data=dict(warm["objectives"]),
        )
        rows.append(_make_warm_row(trial_index, warm, campaign))
        _persist_ax_and_tables(client, rows, output_directory, campaign)
    if not warm_start:
        _persist_ax_and_tables(client, rows, output_directory, campaign)
        print("created campaign with no reused objective observations", flush=True)
    else:
        _verify_warm_start_in_ax(client, warm_start, campaign)
        print(
            f"loaded and verified {len(warm_start)} completed warm-start trials",
            flush=True,
        )
    return client, rows, config


def _resume_campaign(
    output_directory: Path,
    campaign: CampaignDefinition,
) -> tuple[Client, list[dict[str, object]], dict[str, object]]:
    required = (
        output_directory / _RUN_CONFIG_FILENAME,
        output_directory / _AX_STATE_FILENAME,
        output_directory / _TRIALS_FILENAME,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "campaign directory is incomplete; missing " + ", ".join(missing)
        )
    with required[0].open(encoding="utf-8") as input_file:
        stored_config = json.load(input_file)
    _validate_run_config(stored_config, _run_config(campaign))
    client = Client.load_from_json_file(filepath=str(required[1]))
    rows = _read_trials(required[2], campaign)
    if sum(row["source"] == "warm_start" for row in rows) != campaign.warm_start_count:
        raise RuntimeError("persisted campaign has the wrong warm-start count")
    _reconcile_resume(client, rows, output_directory, campaign)
    if campaign.warm_start_path is not None:
        _verify_warm_start_in_ax(
            client,
            _read_warm_start(campaign),
            campaign,
        )
    print(
        f"resumed campaign with {_completed_bo_count(rows)} completed BO trials",
        flush=True,
    )
    return client, rows, stored_config


def _write_plots(output_directory: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    contact = np.asarray(
        [float(row["J_contact"]) for row in completed],
        dtype=np.float64,
    )
    observation = np.asarray(
        [float(row["J_obs"]) for row in completed],
        dtype=np.float64,
    )
    is_pareto = np.asarray(
        [bool(row["is_pareto"]) for row in completed],
        dtype=bool,
    )
    is_warm = np.asarray(
        [row["source"] == "warm_start" for row in completed],
        dtype=bool,
    )

    figure, axes = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axes.scatter(
        contact[is_warm],
        observation[is_warm],
        color="tab:gray",
        label="warm start",
        alpha=0.8,
    )
    axes.scatter(
        contact[~is_warm],
        observation[~is_warm],
        color="tab:blue",
        label="BO",
        alpha=0.8,
    )
    axes.set_xlabel("J_contact")
    axes.set_ylabel("J_obs")
    axes.set_title("Observed LUMO full-finger objectives")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(output_directory / "objective_scatter.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axes.scatter(contact, observation, color="0.7", label="dominated")
    axes.scatter(
        contact[is_pareto],
        observation[is_pareto],
        color="tab:red",
        label="observed Pareto",
        zorder=3,
    )
    if np.count_nonzero(is_pareto) > 1:
        order = np.argsort(contact[is_pareto])
        axes.plot(
            contact[is_pareto][order],
            observation[is_pareto][order],
            color="tab:red",
            alpha=0.6,
        )
    axes.set_xlabel("J_contact")
    axes.set_ylabel("J_obs")
    axes.set_title("Observed nondominated front")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(output_directory / "pareto_front.png", dpi=180)
    plt.close(figure)


def _finalize_outputs(
    output_directory: Path,
    rows: list[dict[str, object]],
    config: dict[str, object],
    campaign: CampaignDefinition,
    *,
    command_wall_runtime_s: float,
) -> dict[str, object]:
    _write_tables(output_directory, rows, campaign)
    _write_plots(output_directory, rows)
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    best_contact = max(completed, key=lambda row: float(row["J_contact"]))
    best_observation = max(completed, key=lambda row: float(row["J_obs"]))
    previous_active_runtime_s = 0.0
    summary_path = output_directory / _SUMMARY_FILENAME
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as input_file:
            previous_summary = json.load(input_file)
        previous_active_runtime_s = float(
            previous_summary.get("active_wall_runtime_s", 0.0)
        )
    created_utc = datetime.fromisoformat(str(config["created_utc"]))
    now_utc = datetime.now(UTC)
    summary = {
        "updated_utc": now_utc.isoformat(),
        "campaign_elapsed_wall_runtime_s": (now_utc - created_utc).total_seconds(),
        "active_wall_runtime_s": (previous_active_runtime_s + command_wall_runtime_s),
        "total_evaluation_runtime_s": sum(
            float(row["runtime_s"])
            for row in rows
            if row["source"] == "bo"
            and row["status"] in {"COMPLETED", "FAILED"}
            and row["runtime_s"] != ""
        ),
        "counts": {
            "warm_start_completed": sum(
                row["source"] == "warm_start" and row["status"] == "COMPLETED"
                for row in rows
            ),
            "bo_completed": _completed_bo_count(rows),
            "bo_failed": sum(
                row["source"] == "bo" and row["status"] == "FAILED" for row in rows
            ),
            "bo_abandoned": sum(
                row["source"] == "bo" and row["status"] == "ABANDONED" for row in rows
            ),
            "pareto": sum(
                row["status"] == "COMPLETED" and row["is_pareto"] for row in rows
            ),
        },
        "best_J_contact": {
            "design": best_contact["design"],
            "ax_trial_index": best_contact["ax_trial_index"],
            "value": best_contact["J_contact"],
        },
        "best_J_obs": {
            "design": best_observation["design"],
            "ax_trial_index": best_observation["ax_trial_index"],
            "value": best_observation["J_obs"],
        },
    }
    _atomic_write_json(summary_path, summary)
    return summary


def run(
    *,
    output_directory: Path,
    target_bo_trials: int,
    campaign_name: str = "discrete-05mm",
    parameter_bounds_mm: Mapping[str, tuple[float, float]] | None = None,
    indenter_urdfs: Iterable[str] = _DEFAULT_INDENTER_URDFS,
    sphere_diameters_mm: Iterable[float] = _DEFAULT_SPHERE_DIAMETERS_MM,
    force_targets_n: Iterable[float] = _DEFAULT_FORCE_TARGETS_N,
    initial_clearance_m: float = _DEFAULT_INITIAL_CLEARANCE_M,
    contact_y_mm: Iterable[float] = _DEFAULT_CONTACT_Y_MM,
    viscoelastic_preset: str = _DEFAULT_VISCOELASTIC_PRESET,
    optical_preset: str = _DEFAULT_OPTICAL_PRESET,
) -> list[dict[str, object]]:
    """Create or resume the full-finger sequential Ax campaign."""
    if target_bo_trials < 0:
        raise ValueError("target_bo_trials must be nonnegative")
    command_start_s = perf_counter()
    output_directory = output_directory.resolve()
    campaign = _campaign_definition(
        campaign_name,
        parameter_bounds_mm=parameter_bounds_mm,
        indenter_urdfs=indenter_urdfs,
        sphere_diameters_mm=sphere_diameters_mm,
        force_targets_n=force_targets_n,
        initial_clearance_m=initial_clearance_m,
        contact_y_mm=contact_y_mm,
        viscoelastic_preset=viscoelastic_preset,
        optical_preset=optical_preset,
    )
    if output_directory == _CONTINUOUS_OUTPUT_DIRECTORY.resolve():
        raise ValueError(
            "the full-finger campaign cannot use the historical continuous output"
        )
    campaign_files_exist = any(
        (output_directory / filename).exists()
        for filename in (
            _RUN_CONFIG_FILENAME,
            _AX_STATE_FILENAME,
            _TRIALS_FILENAME,
        )
    )
    if campaign_files_exist:
        client, rows, config = _resume_campaign(output_directory, campaign)
    else:
        _verify_discrete_search_space(campaign)
        client, rows, config = _initialize_campaign(output_directory, campaign)

    initial_completed_bo = _completed_bo_count(rows)
    if initial_completed_bo < target_bo_trials:
        _validate_optix_environment()
    remaining = target_bo_trials - initial_completed_bo
    proposal_limit = max(1, remaining) * _MAX_PROPOSALS_PER_COMPLETED_TRIAL
    proposal_count = 0
    smoke_pending = initial_completed_bo == 0 and target_bo_trials > 0

    while _completed_bo_count(rows) < target_bo_trials:
        if proposal_count >= proposal_limit:
            raise RuntimeError(
                "Ax did not produce enough successful feasible candidates "
                f"within {proposal_limit} proposals"
            )
        used_seed_count = sum(row["generation_node"] == "PARETO_SEED" for row in rows)
        if used_seed_count < len(campaign.seed_steps):
            seed_name, seed_steps = campaign.seed_steps[used_seed_count]
            raw_parameters = dict(
                zip(campaign.ax_parameter_names, seed_steps, strict=True)
            )
            parameters = _decode_ax_parameters(campaign, raw_parameters)
            _validate_campaign_parameters(campaign, raw_parameters, parameters)
            if not campaign.space.is_feasible(parameters):
                raise RuntimeError(f"snapped Pareto seed {seed_name} is invalid")
            trial_index = client.attach_trial(
                parameters=raw_parameters,
                arm_name=seed_name,
            )
            generation_node = "PARETO_SEED"
        else:
            generated = client.get_next_trials(max_trials=1)
            if len(generated) != 1:
                raise RuntimeError("Ax did not return exactly one sequential trial")
            trial_index, raw_parameters = generated.popitem()
            parameters = _decode_ax_parameters(campaign, raw_parameters)
            _validate_campaign_parameters(campaign, raw_parameters, parameters)
            summary_row = client.summarize(trial_indices=[trial_index]).iloc[0]
            generation_node = str(summary_row.get("generation_node", ""))
        row = _running_row(
            trial_index,
            raw_parameters,
            parameters,
            generation_node,
            campaign,
        )
        rows.append(row)
        proposal_count += 1
        _persist_ax_and_tables(client, rows, output_directory, campaign)
        current_trial = _completed_bo_count(rows) + 1
        print(
            f"current_trial={current_trial}/{target_bo_trials} | "
            f"proposed Ax trial {trial_index} ({row['generation_node']}): "
            f"Ax={raw_parameters}, physical_mm={parameters}",
            flush=True,
        )

        duplicate = _duplicate_terminal(rows[:-1], parameters, campaign)
        if duplicate is not None:
            client.mark_trial_abandoned(trial_index)
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = f"duplicates terminal design {duplicate['design']}"
            _persist_ax_and_tables(client, rows, output_directory, campaign)
            print(f"abandoned duplicate trial {trial_index}", flush=True)
            continue

        if not campaign.space.is_feasible(parameters):
            client.mark_trial_abandoned(trial_index)
            row["status"] = "ABANDONED"
            row["runtime_s"] = 0.0
            row["failure"] = "analytically invalid morphology"
            _persist_ax_and_tables(client, rows, output_directory, campaign)
            print(
                f"abandoned analytically invalid trial {trial_index}",
                flush=True,
            )
            continue

        row["analytically_valid"] = True
        evaluation_start_s = perf_counter()
        try:
            evaluation = _evaluate_candidate(campaign, parameters)
            runtime_s = perf_counter() - evaluation_start_s
            details = _objective_details(evaluation)
            raw_result_path = _trial_result_path(output_directory, trial_index)
            _save_trial_result(
                raw_result_path,
                campaign=campaign,
                evaluation=evaluation,
                details=details,
                parameters=parameters,
                runtime_s=runtime_s,
            )
            _apply_result_to_row(
                row,
                details,
                campaign=campaign,
                runtime_s=runtime_s,
                raw_result_path=raw_result_path.relative_to(
                    output_directory
                ).as_posix(),
            )
            row["status"] = "EVALUATED"
            _write_tables(output_directory, rows, campaign)
            objectives = {name: float(row[name]) for name in _OBJECTIVE_NAMES}
            client.complete_trial(trial_index=trial_index, raw_data=objectives)
            _atomic_save_ax(client, output_directory / _AX_STATE_FILENAME)
            row["status"] = "COMPLETED"
            _write_tables(output_directory, rows, campaign)
            del evaluation
        except Exception as error:
            runtime_s = perf_counter() - evaluation_start_s
            row["runtime_s"] = runtime_s
            row["status"] = "FAILED"
            row["failure"] = f"{type(error).__name__}: {error}"
            if _ax_statuses(client)[trial_index] == "RUNNING":
                client.mark_trial_abandoned(trial_index)
            _persist_ax_and_tables(client, rows, output_directory, campaign)
            print(f"trial {trial_index} failed: {row['failure']}", flush=True)
            if _is_infrastructure_failure(error):
                raise RuntimeError(
                    f"trial {trial_index} evaluation failed; campaign was saved"
                ) from error
            # A mechanics failure is a valid observation about the design
            # space, including before the first successful morphology.  Keep
            # the resume smoke pending until a complete raw result exists.
            continue

        print(
            f"completed trial {trial_index}: "
            f"J_contact={float(row['J_contact']):.9e}, "
            f"J_obs={float(row['J_obs']):.9e}, "
            f"runtime={float(row['runtime_s']):.3f} s",
            flush=True,
        )
        if smoke_pending:
            client = Client.load_from_json_file(
                filepath=str(output_directory / _AX_STATE_FILENAME)
            )
            rows = _read_trials(output_directory / _TRIALS_FILENAME, campaign)
            _reconcile_resume(client, rows, output_directory, campaign)
            if _completed_bo_count(rows) != 1:
                raise RuntimeError("smoke resume check lost or repeated its BO trial")
            smoke_pending = False
            print(
                "smoke PASS: raw NPZ, CSV, atomic Ax state, and resume verified",
                flush=True,
            )

    summary = _finalize_outputs(
        output_directory,
        rows,
        config,
        campaign,
        command_wall_runtime_s=perf_counter() - command_start_s,
    )
    print(
        f"target reached: {_completed_bo_count(rows)} completed BO trials; "
        f"Pareto={summary['counts']['pareto']}",
        flush=True,
    )
    print(f"outputs: {output_directory}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full-finger sequential Ax multi-objective campaign."
    )
    parser.add_argument(
        "--campaign",
        choices=("discrete-05mm",),
        default="discrete-05mm",
        help="five-dimensional half-millimeter production search space",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="campaign output directory; an existing campaign resumes automatically",
    )
    parser.add_argument(
        "--target-bo-trials",
        type=int,
        required=True,
        help="cumulative successful evaluation target",
    )
    arguments = parser.parse_args()
    campaign = _campaign_definition(arguments.campaign)
    output_directory = (
        arguments.output
        if arguments.output is not None
        else campaign.default_output_directory
    )
    run(
        output_directory=output_directory,
        target_bo_trials=arguments.target_bo_trials,
        campaign_name=arguments.campaign,
    )


if __name__ == "__main__":
    main()
