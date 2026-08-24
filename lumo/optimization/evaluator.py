"""One concrete Newton-to-OptiX sensing evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumo.fingertip import Fingertip
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton import Indenter
from lumo.ray_tracing import (
    LED,
    OptixScene,
    PathTraceResult,
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

_SIM_FREQUENCY_HZ = 500.0
_VBD_ITERATIONS = 10
_FORCE_TOLERANCE_N = 1.0
_SERVO_SETTLE_DURATION_S = 5.0
_FORCE_GAIN_M_S_N = 1.25e-3
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547


@dataclass(frozen=True)
class ContactSensingEvaluation:
    """Side-view responses and path-energy ledgers before and after contact."""

    no_contact_response: np.ndarray
    no_contact_paths: PathTraceResult
    contact_response: np.ndarray
    contact_paths: PathTraceResult


def _make_led(fingertip: Fingertip, fingertip_mesh: FingertipMesh) -> LED:
    vertices = np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float64)
    extrusion_center_y_m = 0.5 * float(
        vertices[:, 1].min() + vertices[:, 1].max()
    )
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
    probe_origin = (
        led.position_W_m - probe_distance_m * led.normal_W
    )[None, :]
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
    initial_hits = scene.trace_closest(
        emission["origin_W_m"],
        emission["direction_W"],
        mask=_ALL_MASK,
    )
    if not np.all(initial_hits["hit"]) or np.any(
        initial_hits["instance_id"] != _SILICONE_INSTANCE_ID
    ):
        raise RuntimeError("an emitted primary ray is obstructed before silicone")

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
) -> tuple[np.ndarray, PathTraceResult]:
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
    return response, paths


def evaluate_contact_sensing(
    fingertip: Fingertip,
    trial: DesignTrial,
) -> ContactSensingEvaluation:
    """Evaluate one no-contact state and one force-servo contact state.

    The immutable fingertip mesh and OptiX scene are built once. The contact
    trial reuses that mesh in Newton, and its final silicone vertices update the
    existing silicone GAS and IAS in place before the second optical trace.
    """
    if not isinstance(fingertip, Fingertip):
        raise TypeError("fingertip must be a Fingertip")
    if not isinstance(trial, DesignTrial):
        raise TypeError("trial must be a DesignTrial")

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

    no_contact_response, no_contact_paths = _trace_state(
        scene,
        fingertip,
        emission,
        inside_silicone=_source_inside_silicone(scene, led, emission),
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
    )

    contact_result: tuple[np.ndarray, PathTraceResult] | None = None

    def trace_contact(
        completed_trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal contact_result
        if simulation.fingertip_mesh is not fingertip_mesh:
            raise RuntimeError("Newton did not reuse the evaluator fingertip mesh")
        if simulation.soft_contact_count(indenter.body_index) == 0:
            raise RuntimeError(f"{completed_trial.name} has no indenter contact")
        silicone_vertices = simulation.silicone_vertices()
        if not np.all(np.isfinite(silicone_vertices)):
            raise RuntimeError(
                f"{completed_trial.name} produced non-finite silicone vertices"
            )

        scene.update_silicone(silicone_vertices)
        contact_result = _trace_state(
            scene,
            fingertip,
            emission,
            inside_silicone=_source_inside_silicone(scene, led, emission),
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
        )

    DesignStudy(
        fingertip,
        (trial,),
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        force_tolerance_n=_FORCE_TOLERANCE_N,
        settle_duration_s=_SERVO_SETTLE_DURATION_S,
        force_gain_m_s_n=_FORCE_GAIN_M_S_N,
        iterations=_VBD_ITERATIONS,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
    ).run(inspect_trial=trace_contact)

    if contact_result is None:
        raise RuntimeError("contact optical state was not evaluated")
    contact_response, contact_paths = contact_result
    return ContactSensingEvaluation(
        no_contact_response=no_contact_response,
        no_contact_paths=no_contact_paths,
        contact_response=contact_response,
        contact_paths=contact_paths,
    )


__all__ = ["ContactSensingEvaluation", "evaluate_contact_sensing"]
