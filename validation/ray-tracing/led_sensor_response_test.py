"""Compare bounded LED/receiver response before and after indentation."""

from __future__ import annotations

from importlib.resources import as_file, files
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.newton import Indenter
from lumo.ray_tracing import LED, OptixScene, trace_bounded_paths
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


SILICONE_INSTANCE_ID = 1
CARRIER_INSTANCE_ID = 2
SILICONE_MASK = 0x01
CARRIER_MASK = 0x02
ALL_MASK = SILICONE_MASK | CARRIER_MASK

_N_AIR = 1.0
_CARRIER_ALBEDO = 0.7
_SAMPLE_SIDE_COUNT = 64
# Four segment queries are the first cap that can observe escape after the
# source -> silicone -> carrier -> silicone surface interactions.
_BOUNCE_CAPS = (4, 8, 16, 24)
_RNG_SEED = 20260823

# Hardware optical defaults come from LED. Placement remains an uncalibrated
# validation-local point-source approximation fixed in the carrier frame.
_SOURCE_POSITION_W_M = np.array((-5.0e-3, 0.0, -20.0e-3))
_SOURCE_NORMAL_W = np.array((0.0, 0.0, 1.0))
_RECEIVER_CENTER_W_M = np.array((-10.0e-3, 0.0, -20.0e-3))
_RECEIVER_SIZE_XY_M = np.array((6.0e-3, 4.0e-3))

_SIM_FREQUENCY_HZ = 1.0e3
_SPHERE_RADIUS_M = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_TRANSLATION_STEP_M = 2.5e-5
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 5.0
_FORCE_DURATION_S = 5.0e-3
_MAX_SIM_TIME_S = 30.0
_MAX_SEARCH_ITERATIONS = 256


def _ideal_receiver_response(escaped: np.ndarray) -> tuple[float, int, float]:
    """Return received power/count and escaped power outside the aperture."""
    if not len(escaped):
        return 0.0, 0, 0.0

    origins = escaped["origin_W_m"]
    directions = escaped["direction_W"]
    power = escaped["power"]
    direction_z = directions[:, 2]
    intersects_plane = np.abs(direction_z) > np.finfo(np.float64).tiny
    distance_to_plane = np.full(len(escaped), np.nan)
    distance_to_plane[intersects_plane] = (
        _RECEIVER_CENTER_W_M[2] - origins[intersects_plane, 2]
    ) / direction_z[intersects_plane]
    intersection_W_m = origins + distance_to_plane[:, None] * directions
    receiver_hit = (
        intersects_plane
        & (distance_to_plane > 0.0)
        & (
            np.abs(intersection_W_m[:, 0] - _RECEIVER_CENTER_W_M[0])
            <= 0.5 * _RECEIVER_SIZE_XY_M[0]
        )
        & (
            np.abs(intersection_W_m[:, 1] - _RECEIVER_CENTER_W_M[1])
            <= 0.5 * _RECEIVER_SIZE_XY_M[1]
        )
    )
    received_power = float(power[receiver_hit].sum())
    return (
        received_power,
        int(np.count_nonzero(receiver_hit)),
        float(power[~receiver_hit].sum()),
    )


def _run_convergence(
    scene: OptixScene,
    emission: np.ndarray,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
    *,
    n_silicone: float,
) -> dict[int, dict[str, float | int]]:
    responses: dict[int, dict[str, float | int]] = {}
    for max_bounces in _BOUNCE_CAPS:
        escaped, statistics = trace_bounded_paths(
            scene,
            emission["origin_W_m"],
            emission["direction_W"],
            emission["power"],
            inside_silicone=False,
            n_air=_N_AIR,
            n_silicone=n_silicone,
            carrier_albedo=_CARRIER_ALBEDO,
            max_bounces=max_bounces,
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
            silicone_instance_id=SILICONE_INSTANCE_ID,
            carrier_instance_id=CARRIER_INSTANCE_ID,
            mask=ALL_MASK,
        )
        received_power, received_count, escaped_not_received = (
            _ideal_receiver_response(escaped)
        )
        response = dict(statistics)
        response.update(
            {
                "received_power": received_power,
                "received_ray_count": received_count,
                "escaped_not_received_power": escaped_not_received,
            }
        )
        if not np.isclose(
            received_power + escaped_not_received,
            float(statistics["escaped_power"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError("ideal receiver did not classify escaped power")
        responses[max_bounces] = response
    return responses


def main() -> None:
    wall_start = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)
    scene = OptixScene(
        fingertip_mesh,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        silicone_visibility_mask=SILICONE_MASK,
        carrier_visibility_mask=CARRIER_MASK,
    )

    grid_i, grid_j = np.meshgrid(
        np.arange(_SAMPLE_SIDE_COUNT),
        np.arange(_SAMPLE_SIDE_COUNT),
        indexing="ij",
    )
    emission_u1 = (grid_i.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission_u2 = (grid_j.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    led = LED(
        position_W_m=_SOURCE_POSITION_W_M,
        normal_W=_SOURCE_NORMAL_W,
    )
    emission = led.emit(emission_u1, emission_u2)
    if not np.allclose(
        np.linalg.norm(emission["direction_W"], axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("LED emission directions are not normalized")
    if np.any(emission["direction_W"] @ led.normal_W <= 0.0):
        raise AssertionError("LED emission left the source hemisphere")
    if not np.isclose(
        emission["power"].sum(),
        led.power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("LED ray powers do not sum to source power")

    ray_count = len(emission)
    max_bounces = max(_BOUNCE_CAPS)
    rng = np.random.default_rng(_RNG_SEED)
    dielectric_branch_u = rng.random((max_bounces, ray_count))
    carrier_u1 = rng.random((max_bounces, ray_count))
    carrier_u2 = rng.random((max_bounces, ray_count))
    n_silicone = fingertip.parameters.optical.refractive_index

    undeformed = _run_convergence(
        scene,
        emission,
        dielectric_branch_u,
        carrier_u1,
        carrier_u2,
        n_silicone=n_silicone,
    )
    deformed: dict[int, dict[str, float | int]] | None = None

    def inspect_deformed_trial(
        trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        nonlocal deformed
        if simulation.soft_contact_count(indenter.body_index) == 0:
            raise AssertionError("force-stable trial has no sphere contact")
        final_vertices = simulation.silicone_vertices()
        if not np.all(np.isfinite(final_vertices)):
            raise AssertionError("deformed silicone vertices are not finite")
        if final_vertices.shape != np.asarray(
            fingertip_mesh.silicone.vertices
        ).shape:
            raise AssertionError("deformed silicone vertex count changed")
        if not np.array_equal(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            fingertip_mesh.silicone.surface_tri_indices,
        ):
            raise AssertionError("deformed silicone topology changed")

        scene.update_silicone(final_vertices)
        deformed = _run_convergence(
            scene,
            emission,
            dielectric_branch_u,
            carrier_u1,
            carrier_u2,
            n_silicone=n_silicone,
        )

    initial_sphere_z_m = (
        fingertip.tip_z_m - _INITIAL_CLEARANCE_M - _SPHERE_RADIUS_M
    )
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_10mm.urdf",
    )
    with as_file(sphere_resource) as sphere_urdf_path:
        trial = DesignTrial(
            name="sphere_10mm_center_20N",
            urdf_path=sphere_urdf_path,
            initial_tf=wp.transform(
                wp.vec3(0.0, 0.0, initial_sphere_z_m),
                wp.quat_identity(),
            ),
            translation_step_W_m=wp.vec3(0.0, 0.0, _TRANSLATION_STEP_M),
            target_force_n=_TARGET_FORCE_N,
            max_sim_time_s=_MAX_SIM_TIME_S,
        )
        DesignStudy(
            fingertip,
            (trial,),
            sim_frequency=_SIM_FREQUENCY_HZ,
            force_tolerance_n=_FORCE_TOLERANCE_N,
            force_duration_s=_FORCE_DURATION_S,
            max_search_iterations=_MAX_SEARCH_ITERATIONS,
        ).run(inspect_trial=inspect_deformed_trial)

    if deformed is None or trial.reaction_force_n is None:
        raise AssertionError("deformed optical response was not evaluated")

    print("Green Sequin LED bounded-path validation")
    print(
        f"hardware: Adafruit Product {LED.ADAFRUIT_PRODUCT_ID} | "
        f"{LED.LED_PART_NUMBER} | {LED.PACKAGE}"
    )
    print(
        f"spectral metadata: dominant={led.dominant_wavelength_nm:g} nm | "
        f"peak={led.peak_wavelength_nm:g} nm | "
        f"half-width={led.spectral_half_width_nm:g} nm | "
        f"viewing half-angle={led.viewing_half_angle_deg:g} deg"
    )
    print(
        "source model: ideal point Lambertian at (-5, 0, -20) mm | "
        f"rays={ray_count} | normalized P={led.power:g} | uncalibrated"
    )
    print(
        "receiver: ideal planar 6 x 4 mm aperture centered at "
        "(-10, 0, -20) mm | validation-local"
    )
    print(
        f"transport: n_air={_N_AIR:g} | n_silicone={n_silicone:g} | "
        f"carrier_albedo={_CARRIER_ALBEDO:g} placeholder | "
        "bulk attenuation=not modeled"
    )
    print()

    previous_response_delta: float | None = None
    for bounce_cap in _BOUNCE_CAPS:
        undeformed_response = undeformed[bounce_cap]
        deformed_response = deformed[bounce_cap]
        received_undeformed = float(undeformed_response["received_power"])
        received_deformed = float(deformed_response["received_power"])
        response_delta = received_deformed - received_undeformed
        relative_change = (
            response_delta / received_undeformed
            if received_undeformed > 0.0
            else float("nan")
        )
        convergence_delta = (
            response_delta - previous_response_delta
            if previous_response_delta is not None
            else float("nan")
        )

        for label, response in (
            ("U", undeformed_response),
            ("D", deformed_response),
        ):
            if not np.all(
                np.isfinite(
                    [
                        float(response["received_power"]),
                        float(response["absorbed_power"]),
                        float(response["remaining_power"]),
                        float(response["unresolved_internal_miss_power"]),
                        float(response["accounted_power"]),
                    ]
                )
            ):
                raise AssertionError("bounded response contains invalid power")
            if not np.isclose(
                float(response["accounted_power"]),
                led.power,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise AssertionError("bounded response energy closure failed")
            print(
                f"b={bounce_cap:2d} {label}: "
                f"P_recv={float(response['received_power']):.9f} | "
                f"rays={int(response['received_ray_count']):4d} | "
                f"P_abs={float(response['absorbed_power']):.9f} | "
                f"P_escape_other="
                f"{float(response['escaped_not_received_power']):.9f} | "
                f"P_internal_miss="
                f"{float(response['unresolved_internal_miss_power']):.9f} | "
                f"P_remaining={float(response['remaining_power']):.9f} | "
                f"closure={float(response['accounted_power']):.9f}"
            )
        print(
            f"         delta={response_delta:+.9f} | "
            f"relative={relative_change:+.3%} | "
            f"convergence_delta={convergence_delta:+.9f}"
        )
        previous_response_delta = response_delta

    largest_cap = _BOUNCE_CAPS[-1]
    largest_undeformed = float(undeformed[largest_cap]["received_power"])
    largest_deformed = float(deformed[largest_cap]["received_power"])
    if largest_undeformed <= 0.0 or largest_deformed <= 0.0:
        raise AssertionError("ideal receiver captured no largest-depth rays")
    print()
    print(
        f"mechanics: F={trial.reaction_force_n:.4f} N | "
        f"travel={1.0e3 * float(trial.travel_m):.4f} mm | "
        f"t={float(trial.simulation_time_s):.3f} s"
    )
    print(f"wall runtime: {perf_counter() - wall_start:.3f} s")
    print()
    print("Undeformed/deformed bounded LED-sensor convergence: PASS")


if __name__ == "__main__":
    main()
