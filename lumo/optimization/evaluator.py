"""One concrete Newton-to-OptiX sensing evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton import Indenter
from lumo.ray_tracing import (
    LED,
    OptixScene,
    safe_secondary_origins,
    side_view_observation,
    trace_bounded_paths,
)
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK

_SAMPLE_SIDE_COUNT = 256
_MAX_BOUNCES = 24
_RNG_SEED = 20260823
_CARRIER_ALBEDO = 0.7

_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_SERVO_SETTLE_DURATION_S = 5.0
_SERVO_SETTLE_DISPLACEMENT_TOLERANCE_M = None
_FORCE_GAIN_M_S_N = 2.5e-4
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547
_ELEMENT_SIZE_MM = 1.0
_SOFT_CONTACT_MARGIN_M = 1.0e-4
_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6
_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_FORCE_TOLERANCE_FRACTION = 0.1
_ENERGY_FIELDS = (
    "emitted_power",
    "escaped_power",
    "carrier_absorbed_power",
    "bulk_loss_power",
    "unresolved_internal_miss_power",
    "remaining_power",
    "accounted_power",
    "closure_error",
)


@dataclass(frozen=True)
class ContactSensingEvaluation:
    """Compact no-contact and scenario-by-force sensing results."""

    no_contact_response: np.ndarray
    no_contact_energy: np.ndarray
    scenario_names: tuple[str, ...]
    force_targets_n: np.ndarray
    actual_forces_n: np.ndarray
    indentations_m: np.ndarray
    response_matrix: np.ndarray
    energy_fields: tuple[str, ...]
    energy_matrix: np.ndarray
    checkpoint_times_s: np.ndarray
    scenario_runtime_s: np.ndarray


def _make_led(fingertip: Fingertip, fingertip_mesh: FingertipMesh) -> LED:
    vertices = np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float64)
    extrusion_center_y_m = 0.5 * float(vertices[:, 1].min() + vertices[:, 1].max())
    stem_bottom_z_m = -1.0e-3 * fingertip.parameters.geometry.stem_height_mm
    return LED(
        position_W_m=np.array((0.0, extrusion_center_y_m, stem_bottom_z_m)),
        normal_W=np.array((0.0, 0.0, -1.0)),
        parameters=fingertip.parameters.led,
    )


def _emit_from_stem_boundary(
    scene: OptixScene,
    led: LED,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    probe_distance_m = 0.5e-3 * led.parameters.height_mm
    probe_origin = (led.position_W_m - probe_distance_m * led.normal_W)[None, :]
    direction = led.normal_W[None, :]
    carrier_hit = scene.trace_closest(
        probe_origin,
        direction,
        mask=_CARRIER_MASK,
    )
    if not carrier_hit["hit"][0]:
        raise RuntimeError("carrier probe did not find the LED stem boundary")
    hit_position = probe_origin[0] + carrier_hit["t"][0] * led.normal_W
    if not np.allclose(
        hit_position,
        led.position_W_m,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("carrier probe found the wrong LED stem boundary")

    emission = led.emit(u1, u2)
    emission["origin_W_m"] = safe_secondary_origins(
        carrier_hit,
        direction,
    )[0]
    return emission


def _source_inside_silicone(
    scene: OptixScene,
    led: LED,
    emission: np.ndarray,
) -> bool:
    silicone_hit = scene.trace_closest(
        emission["origin_W_m"][:1],
        led.normal_W[None, :],
        mask=_SILICONE_MASK,
    )[0]
    if not silicone_hit["hit"]:
        raise RuntimeError("the LED normal does not reach silicone")
    normal_projection = float(np.dot(silicone_hit["normal_W"], led.normal_W))
    if abs(normal_projection) <= 1.0e-6:
        raise RuntimeError("the LED source interface is geometrically ambiguous")
    return normal_projection > 0.0


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    *,
    inside_silicone: bool,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    optics = fingertip.parameters.optical
    paths = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=inside_silicone,
        n_air=1.0,
        n_silicone=optics.refractive_index,
        extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_MAX_BOUNCES,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        mask=_ALL_MASK,
    )
    response = side_view_observation(
        paths.escaped_rays,
        fingertip=fingertip,
    )
    energy = np.array(
        (
            paths.emitted_power,
            paths.escaped_power,
            paths.absorbed_power,
            paths.bulk_loss_power,
            paths.unresolved_internal_miss_power,
            paths.remaining_power,
            paths.accounted_power,
            paths.closure_error,
        ),
        dtype=np.float64,
    )
    del paths
    return response, energy


def evaluate_contact_sensing(
    fingertip: Fingertip,
    trials: Iterable[DesignTrial],
    *,
    settle_duration_s: float = _SERVO_SETTLE_DURATION_S,
    settle_displacement_tolerance_m: float | None = (
        _SERVO_SETTLE_DISPLACEMENT_TOLERANCE_M
    ),
) -> ContactSensingEvaluation:
    """Evaluate one no-contact state and force-servo contact scenarios.

    The immutable fingertip mesh and OptiX scene are built once. The contact
    trials reuse that mesh in independent Newton runtimes. Each accepted force
    checkpoint updates the existing silicone GAS and IAS before tracing with
    the same deterministic optical samples. Defaults implement the validated
    five-second force-band dwell; scalar overrides support focused convergence
    validation without changing the mechanics or optical pipeline.
    """
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")
    if isinstance(trials, DesignTrial):
        raise TypeError("trials must be an iterable of DesignTrial objects")
    trial_tuple = tuple(trials)
    if not trial_tuple:
        raise ValueError("trials must contain at least one DesignTrial")
    if any(not isinstance(trial, DesignTrial) for trial in trial_tuple):
        raise TypeError("trials must contain only DesignTrial objects")

    fingertip_mesh = make_fingertip_mesh(fingertip)
    scene = OptixScene(
        fingertip_mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    led = _make_led(fingertip, fingertip_mesh)

    sample_i, sample_j = np.meshgrid(
        np.arange(_SAMPLE_SIDE_COUNT),
        np.arange(_SAMPLE_SIDE_COUNT),
        indexing="ij",
    )
    emission_u1 = (sample_i.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission_u2 = (sample_j.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission = _emit_from_stem_boundary(
        scene,
        led,
        emission_u1,
        emission_u2,
    )

    rng = np.random.default_rng(_RNG_SEED)
    sample_shape = (_MAX_BOUNCES, len(emission))
    dielectric_branch_u = rng.random(sample_shape)
    carrier_u1 = rng.random(sample_shape)
    carrier_u2 = rng.random(sample_shape)

    no_contact_response, no_contact_energy = _trace_state(
        scene,
        fingertip,
        emission,
        inside_silicone=_source_inside_silicone(scene, led, emission),
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
    )

    scenario_count = len(trial_tuple)
    force_count = len(_FORCE_TARGETS_N)
    response_matrix = np.empty((scenario_count, force_count, 4), dtype=np.float64)
    energy_matrix = np.empty(
        (scenario_count, force_count, len(_ENERGY_FIELDS)),
        dtype=np.float64,
    )
    actual_forces_n = np.empty((scenario_count, force_count), dtype=np.float64)
    indentations_m = np.empty((scenario_count, force_count), dtype=np.float64)
    checkpoint_times_s = np.empty(
        (scenario_count, force_count),
        dtype=np.float64,
    )
    scenario_runtime_s = np.empty(scenario_count, dtype=np.float64)
    scenario_indices = {id(trial): index for index, trial in enumerate(trial_tuple)}
    next_force_indices = np.zeros(scenario_count, dtype=np.int64)
    scenario_start_s = perf_counter()

    def trace_contact(
        completed_trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal scenario_start_s
        if simulation.fingertip_mesh is not fingertip_mesh:
            raise RuntimeError("Newton did not reuse the evaluator fingertip mesh")
        if simulation.soft_contact_count(indenter.body_index) == 0:
            raise RuntimeError(f"{completed_trial.name} has no indenter contact")
        silicone_vertices = simulation.silicone_vertices()
        if not np.all(np.isfinite(silicone_vertices)):
            raise RuntimeError(
                f"{completed_trial.name} produced non-finite silicone vertices"
            )

        scenario_index = scenario_indices[id(completed_trial)]
        force_index = int(next_force_indices[scenario_index])
        if force_index >= force_count:
            raise RuntimeError(f"{completed_trial.name} produced an extra checkpoint")
        if (
            completed_trial.reaction_force_n is None
            or completed_trial.travel_m is None
            or completed_trial.simulation_time_s is None
        ):
            raise RuntimeError(
                f"{completed_trial.name} checkpoint has no mechanics result"
            )

        scene.update_silicone(silicone_vertices)
        response, energy = _trace_state(
            scene,
            fingertip,
            emission,
            inside_silicone=_source_inside_silicone(scene, led, emission),
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
        )
        response_matrix[scenario_index, force_index] = response
        energy_matrix[scenario_index, force_index] = energy
        actual_forces_n[scenario_index, force_index] = completed_trial.reaction_force_n
        indentations_m[scenario_index, force_index] = (
            completed_trial.travel_m - completed_trial.initial_clearance_m
        )
        checkpoint_times_s[scenario_index, force_index] = (
            completed_trial.simulation_time_s
        )
        next_force_indices[scenario_index] += 1
        if next_force_indices[scenario_index] == force_count:
            now_s = perf_counter()
            scenario_runtime_s[scenario_index] = now_s - scenario_start_s
            scenario_start_s = now_s

    DesignStudy(
        fingertip,
        trial_tuple,
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        settle_duration_s=settle_duration_s,
        settle_displacement_tolerance_m=settle_displacement_tolerance_m,
        force_tolerance_fraction=_FORCE_TOLERANCE_FRACTION,
        force_targets_n=_FORCE_TARGETS_N,
        force_gain_m_s_n=_FORCE_GAIN_M_S_N,
        element_size_mm=_ELEMENT_SIZE_MM,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        carrier_contact_stiffness_n_m=_CARRIER_CONTACT_STIFFNESS_N_M,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    ).run(inspect_trial=trace_contact)

    if np.any(next_force_indices != force_count):
        raise RuntimeError("not every force checkpoint was evaluated")
    return ContactSensingEvaluation(
        no_contact_response=no_contact_response,
        no_contact_energy=no_contact_energy,
        scenario_names=tuple(trial.name for trial in trial_tuple),
        force_targets_n=np.asarray(_FORCE_TARGETS_N, dtype=np.float64),
        actual_forces_n=actual_forces_n,
        indentations_m=indentations_m,
        response_matrix=response_matrix,
        energy_fields=_ENERGY_FIELDS,
        energy_matrix=energy_matrix,
        checkpoint_times_s=checkpoint_times_s,
        scenario_runtime_s=scenario_runtime_s,
    )


__all__ = ["ContactSensingEvaluation", "evaluate_contact_sensing"]
