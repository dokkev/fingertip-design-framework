"""Validate one complete orientation-aware production morphology evaluation."""

from __future__ import annotations

import json
import os
import sys
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lumo.fingertip import Fingertip  # noqa: E402
from lumo.optimization.ax_bo import build_campaign, objective_details  # noqa: E402
from lumo.optimization.campaign_io import (  # noqa: E402
    build_run_config,
    save_trial_result,
)
from lumo.optimization.evaluator import evaluate_fingertip  # noqa: E402
from lumo.optimization.objective import compute_objectives_from_raw  # noqa: E402
from scripts import run_mobo as production  # noqa: E402


_OUTPUT_DIRECTORY = Path(
    "output/validation/orientation_aware_mobo_smoke"
)
_RAW_PATH = _OUTPUT_DIRECTORY / "dragon_skin_trial_117.npz"
_RUN_CONFIG_PATH = _OUTPUT_DIRECTORY / "run_config.json"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"
_TRIAL_117_PARAMETERS = {
    "geometry.flat_pad_height_mm": 14.5,
    "geometry.semiellipse_height_mm": 4.0,
    "geometry.stem_width_mm": 5.0,
    "geometry.stem_height_mm": 12.5,
    "geometry.void_width_mm": 5.0,
}


def _raw_arrays() -> dict[str, np.ndarray]:
    with np.load(_RAW_PATH) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def _verify_raw(
    data: dict[str, np.ndarray],
) -> tuple[object, object, dict[str, float | int | bool]]:
    diameters = production.SPHERE_DIAMETERS_MM
    angles = production.INDENTATION_ANGLES_DEG
    locations = production.CONTACT_Y_MM
    forces = production.FORCE_TARGETS_N
    expected_scenarios = tuple(
        (diameter, angle, location)
        for diameter in diameters
        for angle in angles
        for location in locations
    )
    expected_names = tuple(
        f"sphere_{diameter:g}mm_y{location:+g}mm_theta{angle:+g}deg"
        for diameter, angle, location in expected_scenarios
    )
    expected_diameters = np.asarray(
        [scenario[0] for scenario in expected_scenarios]
    )
    expected_angles = np.asarray([scenario[1] for scenario in expected_scenarios])
    expected_locations = np.asarray(
        [scenario[2] for scenario in expected_scenarios]
    )
    scenario_count = len(expected_scenarios)
    state_shape = (scenario_count, len(forces))

    if scenario_count != 75 or len(set(expected_scenarios)) != 75:
        raise RuntimeError("the production scenario specification is not 75 unique cases")
    if tuple(str(name) for name in data["scenario_names"]) != expected_names:
        raise RuntimeError("saved scenario names do not preserve production ordering")
    if not np.array_equal(data["sphere_diameters_mm"], expected_diameters):
        raise RuntimeError("saved sphere diameters do not preserve production ordering")
    if not np.array_equal(data["contact_angles_deg"], expected_angles):
        raise RuntimeError("saved raw NPZ did not preserve contact_angles_deg")
    if not np.array_equal(data["contact_y_mm"], expected_locations):
        raise RuntimeError("saved contact locations do not preserve production ordering")
    if data["actual_forces_n"].shape != state_shape:
        raise RuntimeError("the smoke did not produce all 300 force checkpoints")
    if data["response_matrix"].shape != (*state_shape, 5, 11):
        raise RuntimeError("the smoke did not produce all 300 five-LED optical states")

    targets = np.asarray(forces, dtype=np.float64)
    if np.any(data["actual_forces_n"] < targets[None, :]):
        raise RuntimeError("a saved first-crossing force is below its threshold")
    if np.any(np.diff(data["checkpoint_steps"], axis=1) <= 0):
        raise RuntimeError("force checkpoints are not strictly ordered")
    if np.any(np.diff(data["indentations_m"], axis=1) <= 0.0):
        raise RuntimeError("checkpoint indentation is not strictly increasing")
    if np.any(data["contact_buffer_overflow"] != 0):
        raise RuntimeError("a scenario overflowed the body-particle contact buffer")
    if np.any(data["inverted_tet_counts"] != 0):
        raise RuntimeError("a scenario contains an inverted tetrahedron")
    if np.any(data["indenter_contact_counts"] <= 0):
        raise RuntimeError("a force checkpoint has no indenter contact")

    finite_fields = (
        "actual_forces_n",
        "indentations_m",
        "maximum_particle_speeds_m_s",
        "minimum_det_f",
        "contact_centroids_W_m",
        "silicone_vertices_m",
        "response_matrix",
        "energy_matrix",
    )
    if any(not np.all(np.isfinite(data[field])) for field in finite_fields):
        raise RuntimeError("saved mechanics or optical diagnostics are non-finite")
    if np.any(data["minimum_det_f"] <= 0.0):
        raise RuntimeError("minimum det(F) is non-positive")

    theta_zero = data["contact_angles_deg"] == 0.0
    theta_zero_travel = data["zero_contact_travel_m"][theta_zero]
    if not np.allclose(
        theta_zero_travel,
        production.INITIAL_CLEARANCE_M,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError("theta=0 tangency is not the ordinary pad-normal geometry")

    energy_fields = tuple(str(field) for field in data["energy_fields"])
    closure_index = energy_fields.index("closure_error")
    maximum_closure_error = max(
        float(np.max(np.abs(data["energy_matrix"][..., closure_index]))),
        float(np.max(np.abs(data["no_contact_energy"][:, closure_index]))),
    )
    if maximum_closure_error > 1.0e-12:
        raise RuntimeError("OptiX energy accounting does not close")

    contact, observation = compute_objectives_from_raw(data)
    if not np.isclose(contact.J_contact, float(data["J_contact"]), atol=1.0e-12):
        raise RuntimeError("J_contact does not reproduce from the raw NPZ")
    if not np.isclose(observation.J_obs, float(data["J_obs"]), atol=1.0e-12):
        raise RuntimeError("J_obs does not reproduce from the raw NPZ")
    if contact.limiting_contact_angle_deg not in angles:
        raise RuntimeError("J_contact limiting angle is outside the scenario contract")
    if observation.limiting_contact_angle_deg not in angles:
        raise RuntimeError("J_obs limiting angle is outside the scenario contract")

    limiting_index = contact.limiting_scenario_index
    metrics: dict[str, float | int | bool] = {
        "scenario_count": scenario_count,
        "checkpoint_count": int(data["actual_forces_n"].size),
        "minimum_det_f": float(np.min(data["minimum_det_f"])),
        "maximum_energy_closure_error": maximum_closure_error,
        "maximum_contact_buffer_overflow": int(
            np.max(data["contact_buffer_overflow"])
        ),
        "inverted_tet_count": int(np.max(data["inverted_tet_counts"])),
        "theta_zero_tangency_error_m": float(
            np.max(np.abs(theta_zero_travel - production.INITIAL_CLEARANCE_M))
        ),
        "limiting_q_form": float(contact.q_form[limiting_index]),
        "limiting_q_stable": float(contact.q_stable[limiting_index]),
        "limiting_q_stiff": float(contact.q_stiff[limiting_index]),
        "raw_recomputation_pass": True,
        "raw_contact_angles_round_trip_pass": True,
        "initial_contact_gate_pass": True,
    }
    return contact, observation, metrics


def _write_report(
    contact: object,
    observation: object,
    metrics: dict[str, float | int | bool],
    runtime_s: float,
) -> None:
    campaign_hours = runtime_s * production.TARGET_MORPHOLOGIES / 3600.0
    lines = [
        "# Orientation-aware production MOBO smoke",
        "",
        "Result: PASS",
        "",
        "## Contract",
        "",
        "- morphology: Dragon Skin trial 117 `[14.5, 4.0, 5.0, 12.5, 5.0]` mm",
        f"- angles [deg]: `{list(production.INDENTATION_ANGLES_DEG)}`",
        f"- sphere diameters [mm]: `{list(production.SPHERE_DIAMETERS_MM)}`",
        f"- contact Y [mm]: `{list(production.CONTACT_Y_MM)}`",
        f"- force thresholds [N]: `{list(production.FORCE_TARGETS_N)}`",
        "- ordering: sphere diameter major, contact angle middle, contact Y minor",
        "- physical +theta fingertip rotation is represented by inverse `Ry(-theta)` of both sphere center and motion direction",
        "- pivot: longitudinal world Y axis through X=0, Z=0",
        "",
        "## Integrity",
        "",
        f"- completed scenarios: {int(metrics['scenario_count'])}/75",
        f"- completed checkpoints and optical states: {int(metrics['checkpoint_count'])}/300",
        "- initial-contact runtime gate: PASS",
        f"- theta=0 tangency error: {float(metrics['theta_zero_tangency_error_m']):.3e} m",
        f"- minimum det(F): {float(metrics['minimum_det_f']):.9f}",
        f"- inverted tetrahedra: {int(metrics['inverted_tet_count'])}",
        f"- maximum contact-buffer overflow: {int(metrics['maximum_contact_buffer_overflow'])}",
        f"- maximum optical energy closure error: {float(metrics['maximum_energy_closure_error']):.3e}",
        "- raw NPZ contact-angle round trip: PASS",
        "- objective recomputation from raw NPZ: PASS",
        "",
        "## Objectives",
        "",
        f"- J_contact: {contact.J_contact:.9f}",
        (
            "- limiting contact: "
            f"theta={contact.limiting_contact_angle_deg:+g} deg, "
            f"sphere={contact.limiting_sphere_diameter_mm:g} mm, "
            f"Y={contact.limiting_contact_y_mm:+g} mm"
        ),
        (
            "- limiting contact components: "
            f"q_form={float(metrics['limiting_q_form']):.9f}, "
            f"q_stable={float(metrics['limiting_q_stable']):.9f}, "
            f"q_stiff={float(metrics['limiting_q_stiff']):.9f}"
        ),
        f"- J_obs: {observation.J_obs:.9f}",
        (
            "- limiting observation: "
            f"theta={observation.limiting_contact_angle_deg:+g} deg, "
            f"sphere={observation.limiting_sphere_diameter_mm:g} mm, "
            f"force={observation.limiting_force_n:g} N, "
            f"Y={observation.limiting_contact_y_pair_mm} mm"
        ),
        f"- d_onset diagnostic: {observation.d_onset:.9f}",
        "",
        "## Runtime",
        "",
        f"- one 75-scenario morphology: {runtime_s:.3f} s ({runtime_s / 60.0:.2f} min)",
        f"- estimated serial 120-morphology campaign: {campaign_hours:.2f} h",
        "",
        "No production BO campaign was started.",
    ]
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.environ.setdefault("OTK_INCLUDE_DIR", str(production.OTK_INCLUDE_DIR))
    campaign = build_campaign(
        parameter_bounds_mm=production.PARAMETER_BOUNDS_MM,
        indenter_urdfs=production.INDENTER_URDFS,
        sphere_diameters_mm=production.SPHERE_DIAMETERS_MM,
        indentation_angles_deg=production.INDENTATION_ANGLES_DEG,
        force_targets_n=production.FORCE_TARGETS_N,
        initial_clearance_m=production.INITIAL_CLEARANCE_M,
        contact_y_mm=production.CONTACT_Y_MM,
        mechanics_preset=production.MECHANICS_PRESET,
        optical_preset=production.OPTICAL_PRESET,
        initial_morphologies_mm=production.INITIAL_MORPHOLOGIES_MM,
    )
    if campaign.mechanics_preset != "silicone" or (
        campaign.optical_preset != "dragon_skin_10_nv_nominal"
    ):
        raise RuntimeError("the production campaign is not the requested Dragon setup")
    if not campaign.space.is_feasible(_TRIAL_117_PARAMETERS):
        raise RuntimeError("Dragon Skin trial 117 is outside the production space")

    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _RUN_CONFIG_PATH.write_text(
        json.dumps(build_run_config(campaign), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fingertip = Fingertip(campaign.space.to_parameters(_TRIAL_117_PARAMETERS))
    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    start_s = perf_counter()
    with ExitStack() as resources:
        sphere_paths = tuple(
            resources.enter_context(as_file(resource_root.joinpath(filename)))
            for filename in production.INDENTER_URDFS
        )
        evaluation = evaluate_fingertip(
            fingertip,
            sphere_paths,
            production.SPHERE_DIAMETERS_MM,
            production.CONTACT_Y_MM,
            force_targets_n=production.FORCE_TARGETS_N,
            indentation_angles_deg=production.INDENTATION_ANGLES_DEG,
            initial_clearance_m=production.INITIAL_CLEARANCE_M,
            approach_speed_m_s=campaign.approach_speed_m_s,
            max_sim_time_s=campaign.max_sim_time_s,
        )
    runtime_s = perf_counter() - start_s
    details = objective_details(evaluation)
    save_trial_result(
        _RAW_PATH,
        campaign=campaign,
        evaluation=evaluation,
        details=details,
        parameters=_TRIAL_117_PARAMETERS,
        runtime_s=runtime_s,
    )
    del evaluation

    contact, observation, metrics = _verify_raw(_raw_arrays())
    _write_report(contact, observation, metrics, runtime_s)
    print("Orientation-aware production MOBO smoke: PASS", flush=True)
    print(f"runtime={runtime_s:.3f} s", flush=True)
    print(
        f"J_contact={contact.J_contact:.9f} "
        f"(theta={contact.limiting_contact_angle_deg:+g}, "
        f"sphere={contact.limiting_sphere_diameter_mm:g}, "
        f"Y={contact.limiting_contact_y_mm:+g})",
        flush=True,
    )
    print(
        f"J_obs={observation.J_obs:.9f} "
        f"(theta={observation.limiting_contact_angle_deg:+g}, "
        f"sphere={observation.limiting_sphere_diameter_mm:g}, "
        f"force={observation.limiting_force_n:g}, "
        f"Y={observation.limiting_contact_y_pair_mm})",
        flush=True,
    )
    print(f"estimated_120h={runtime_s * 120.0 / 3600.0:.3f}", flush=True)
    print(f"report={_REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
